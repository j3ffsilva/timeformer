"""
Orquestrador da Fase B do Timeformer.

Executa em sequência dentro do mesmo processo:
  1. Gera conjunto contrastivo (data/contrastive_set.tsv)
  2. Treina B1, B2a, B2b, B3  (outputs/runs/{run_id}/{model}/)
  3. Avalia todos os modelos  (outputs/runs/{run_id}/results/)

Uso:
  python run_phase_b.py
  python run_phase_b.py --epochs 50 --device cpu
  python run_phase_b.py --skip-contrastive    # pula geração do conjunto contrastivo
  python run_phase_b.py --skip-train          # pula treino — requer --run-id
  python run_phase_b.py --run-id 20260523_001 --skip-train  # avalia run existente
"""

import argparse
import pickle
from pathlib import Path

import torch

from src.timeformer.dataset import load_corpus, MLMDataset, B3Dataset, make_continuation_split
from src.timeformer.models import build_model, DEFAULT_HPARAMS
from src.timeformer.memory import PrototypeMemory
from src.timeformer.train import MLMTrainer
from src.timeformer.eval import Evaluator, save_results
from src.timeformer.run import RunManager

CORPUS_PATH      = Path("data/corpus.tsv")
AMBIGUOUS_PATH   = Path("data/corpus_ambiguous.tsv")
CONTRASTIVE_PATH = Path("data/contrastive_set.tsv")
ALL_MODELS       = ["B1", "B2a", "B2b", "B3"]


# ── Geração do conjunto contrastivo ───────────────────────────────────────────

def generate_contrastive() -> None:
    import importlib, sys
    print("  Importando scripts/generate_contrastive.py ...")
    spec = importlib.util.spec_from_file_location(
        "generate_contrastive", "scripts/generate_contrastive.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


# ── Treino ────────────────────────────────────────────────────────────────────

def train_all(args: argparse.Namespace, run: RunManager) -> None:
    rows       = load_corpus(CORPUS_PATH)
    # t8/t9 held-out para continuation; excluídos do treino para evitar leakage
    train_rows, _ = make_continuation_split(rows)
    val_rows      = [r for r in rows if r["split"] == "test"]

    for name in ALL_MODELS:
        print(f"\n{'='*50}")
        print(f"Treinando {name}")
        print(f"{'='*50}")

        model    = build_model(name)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parâmetros: {n_params:,}")

        train_ds = B3Dataset(train_rows, seed=args.seed) if name == "B3" else MLMDataset(train_rows, seed=args.seed)
        val_ds   = MLMDataset(val_rows, seed=args.seed)
        memory   = PrototypeMemory(d_model=DEFAULT_HPARAMS["d_model"]) if name == "B3" else None

        trainer = MLMTrainer(model, output_dir=run.model_dir(name), device=args.device)
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
        print(f"{name}: train_loss={final_loss:.4f}  best_val={best_val:.4f}")

        if memory is not None:
            run.save_memory(name, memory)


# ── Avaliação ─────────────────────────────────────────────────────────────────

def eval_all(args: argparse.Namespace, run: RunManager) -> dict:
    from src.timeformer.train import load_checkpoint

    evaluator = Evaluator(
        corpus_path=CORPUS_PATH,
        ambiguous_path=AMBIGUOUS_PATH,
        contrastive_path=CONTRASTIVE_PATH,
        device=args.device,
    )

    available = [m for m in ALL_MODELS if run.checkpoint_path(m, "best").exists()]
    missing   = [m for m in ALL_MODELS if m not in available]
    if missing:
        print(f"Modelos sem checkpoint (ignorados): {missing}")

    results = {}
    for name in available:
        print(f"\n--- {name} ---")
        model = build_model(name)
        load_checkpoint(model, run.checkpoint_path(name, "best"))
        model.to(torch.device(args.device))

        memory = None
        if name == "B3":
            memory = run.load_memory(name)
            if memory is not None:
                memory.to(args.device)
            else:
                print("  Aviso: memory.pkl não encontrado para B3")

        res = evaluator.evaluate(model, memory=memory)
        results[name] = res

        for sp in ("test", "ambiguous_test", "continuation", "contrastive"):
            sp_res = res.get(sp, {})
            if sp_res.get("skipped"):
                continue
            if sp == "contrastive":
                sfr = sp_res.get("sign_flip_rate", float("nan"))
                print(f"  {sp}: sign_flip={sfr:.3f}")
            else:
                acc = sp_res.get("probe_subj", {}).get("accuracy", float("nan"))
                f1  = sp_res.get("probe_subj", {}).get("f1", float("nan"))
                print(f"  {sp}: acc={acc:.3f}  f1={f1:.3f}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestrador da Fase B do Timeformer")
    parser.add_argument("--epochs",           type=int,   default=30)
    parser.add_argument("--batch-size",       type=int,   default=64)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--device",           type=str,   default="cpu")
    parser.add_argument("--skip-contrastive", action="store_true",
                        help="Pula geração do conjunto contrastivo")
    parser.add_argument("--skip-train",       action="store_true",
                        help="Pula treino — requer --run-id com checkpoints existentes")
    parser.add_argument("--run-id",           type=str, default=None,
                        help="ID de run existente (para --skip-train ou continuar)")
    parser.add_argument("--notes",            type=str, default="",
                        help="Notas para o runs_index.csv")
    args = parser.parse_args()

    # ── 1. Conjunto contrastivo ────────────────────────────────────────────
    if not args.skip_contrastive:
        if CONTRASTIVE_PATH.exists():
            print(f"Conjunto contrastivo já existe em {CONTRASTIVE_PATH} — pulando.")
        else:
            print("\n=== Gerando conjunto contrastivo ===")
            generate_contrastive()
    else:
        print("Geração do conjunto contrastivo pulada (--skip-contrastive).")

    # ── 2. Treino ──────────────────────────────────────────────────────────
    config = {
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "seed":       args.seed,
        "device":     args.device,
        **DEFAULT_HPARAMS,
    }

    if args.skip_train:
        if not args.run_id:
            parser.error("--skip-train requer --run-id")
        run = RunManager.load(args.run_id)
        print(f"\nTreino pulado. Usando run existente: {run.run_id}")
    else:
        if args.run_id:
            run = RunManager.load(args.run_id)
        else:
            run = RunManager()
            run.setup(config)
        print("\n=== Treinando modelos ===")
        train_all(args, run)

    # ── 3. Avaliação ───────────────────────────────────────────────────────
    print("\n=== Avaliando modelos ===")
    results = eval_all(args, run)

    save_results(results, run.results_dir())
    run.update_index(results, notes=args.notes)

    print(f"\nFase B concluída.")
    print(f"  Run:       {run.run_id}")
    print(f"  Resultados: {run.results_dir()}/")
    print(f"  Índice:     outputs/runs/runs_index.csv")


if __name__ == "__main__":
    main()
