"""
Treina os modelos da Fase B (B1, B2a, B2b, B3) e salva checkpoints via RunManager.

Uso:
  python scripts/train_phase_b.py                   # treina todos
  python scripts/train_phase_b.py --model B3        # treina apenas B3
  python scripts/train_phase_b.py --epochs 50 --lr 5e-4
  python scripts/train_phase_b.py --run-id 20260523_001  # continua run existente
"""

import argparse
import pickle
from pathlib import Path

from src.timeformer.dataset import load_corpus, MLMDataset, B3Dataset, make_continuation_split
from src.timeformer.models import build_model, DEFAULT_HPARAMS
from src.timeformer.memory import PrototypeMemory
from src.timeformer.train import MLMTrainer
from src.timeformer.run import RunManager

CORPUS_PATH = Path("data/corpus.tsv")
ALL_MODELS  = ["B1", "B2a", "B2b", "B3"]

TRAIN_DEFAULTS = {
    "n_epochs":   30,
    "batch_size": 64,
    "lr":         1e-3,
    "seed":       42,
}


def train_model(name: str, args: argparse.Namespace, run: RunManager) -> None:
    print(f"\n{'='*50}")
    print(f"Treinando {name}")
    print(f"{'='*50}")

    rows = load_corpus(CORPUS_PATH)
    # t8/t9 são held-out para continuation; excluí-los do treino evita leakage
    train_rows, _ = make_continuation_split(rows)
    val_rows      = [r for r in rows if r["split"] == "test"]

    model    = build_model(name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros: {n_params:,}")

    train_ds = B3Dataset(train_rows, seed=args.seed) if name == "B3" else MLMDataset(train_rows, seed=args.seed)
    val_ds   = MLMDataset(val_rows,  seed=args.seed)
    memory   = PrototypeMemory(d_model=DEFAULT_HPARAMS["d_model"]) if name == "B3" else None

    out_dir = run.model_dir(name)
    trainer = MLMTrainer(model, output_dir=out_dir, device=args.device)
    history = trainer.train(
        train_ds, val_ds,
        memory=memory,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    final_loss = history[-1]["train_loss"]
    best_val   = min(
        (r["val_loss"] for r in history if r["val_loss"] is not None),
        default=float("nan"),
    )
    print(f"\n{name} concluído: train_loss={final_loss:.4f}  best_val={best_val:.4f}")
    print(f"Checkpoints em {out_dir}/")

    if memory is not None:
        run.save_memory(name, memory)
        print(f"Memória B3 salva em {out_dir}/memory.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina modelos da Fase B do Timeformer")
    parser.add_argument("--model",      type=str, default=None, choices=ALL_MODELS,
                        help="Modelo a treinar (default: todos)")
    parser.add_argument("--epochs",     type=int,   default=TRAIN_DEFAULTS["n_epochs"])
    parser.add_argument("--batch-size", type=int,   default=TRAIN_DEFAULTS["batch_size"])
    parser.add_argument("--lr",         type=float, default=TRAIN_DEFAULTS["lr"])
    parser.add_argument("--seed",       type=int,   default=TRAIN_DEFAULTS["seed"])
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--run-id",     type=str,   default=None,
                        help="ID de uma run existente para continuar (default: nova run)")
    args = parser.parse_args()

    config = {
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "seed":       args.seed,
        "device":     args.device,
        **DEFAULT_HPARAMS,
    }

    if args.run_id:
        run = RunManager.load(args.run_id)
    else:
        run = RunManager()
        run.setup(config)

    models_to_train = [args.model] if args.model else ALL_MODELS

    print(f"Fase B — Treino  [{run.run_id}]")
    print(f"  Modelos: {models_to_train}")
    print(f"  Épocas:  {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}")
    print(f"  Device:  {args.device}")

    for name in models_to_train:
        train_model(name, args, run)

    print("\nTreino concluído.")
    print(f"Run: {run.run_id}  ({run.run_dir}/)")


if __name__ == "__main__":
    main()
