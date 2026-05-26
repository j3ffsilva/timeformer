"""
Avalia os modelos treinados da Fase B e produz tabelas de resultados.

Requer checkpoints salvos por scripts/train_phase_b.py.

Uso:
  python scripts/eval_phase_b.py --run-id 20260523_001   # avalia run específica
  python scripts/eval_phase_b.py                         # avalia run mais recente
  python scripts/eval_phase_b.py --model Timeformer       # avalia apenas Timeformer
"""

import argparse
from pathlib import Path

import torch

from src.timeformer.models import build_model
from src.timeformer.nomenclature import model_label
from src.timeformer.train import load_checkpoint
from src.timeformer.eval import Evaluator, save_results
from src.timeformer.run import RunManager

CORPUS_PATH      = Path("data/corpus.tsv")
AMBIGUOUS_PATH   = Path("data/corpus_ambiguous.tsv")
CONTRASTIVE_PATH = Path("data/contrastive_set.tsv")
ALL_MODELS       = ["Static", "Additive", "Joint", "Timeformer"]


def load_model_and_memory(name: str, run: RunManager, device: str):
    ckpt_path = run.checkpoint_path(name, "best")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {ckpt_path}\n"
            f"Execute primeiro: python scripts/train_phase_b.py --run-id {run.run_id} --model {name}"
        )

    model = build_model(name)
    load_checkpoint(model, ckpt_path)
    model.to(torch.device(device))

    memory = None
    if name == "Timeformer":
        memory = run.load_memory(name)
        if memory is None:
            print(f"  Aviso: memory.pkl não encontrado para Timeformer — avaliando sem memória")
        else:
            memory.to(device)

    return model, memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia modelos da Fase B do Timeformer")
    parser.add_argument("--run-id", type=str, default=None,
                        help="ID da run a avaliar (default: mais recente)")
    parser.add_argument("--model",  type=str, default=None, choices=ALL_MODELS,
                        help="Modelo a avaliar (default: todos) — use Static, Additive, Joint, ou Timeformer")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.run_id:
        run = RunManager.load(args.run_id)
    else:
        run = RunManager.load_latest()

    print(f"Fase B — Avaliação  [{run.run_id}]")

    models_to_eval = [args.model] if args.model else ALL_MODELS
    available = [m for m in models_to_eval if run.checkpoint_path(m, "best").exists()]
    missing   = [m for m in models_to_eval if m not in available]
    if missing:
        print(f"Modelos sem checkpoint (ignorados): {missing}")
    if not available:
        print("Nenhum modelo disponível para avaliação.")
        return

    print(f"  Modelos: {available}")

    evaluator = Evaluator(
        corpus_path=CORPUS_PATH,
        ambiguous_path=AMBIGUOUS_PATH,
        contrastive_path=CONTRASTIVE_PATH,
        device=args.device,
    )

    results = {}
    for name in available:
        print(f"\n--- {name}: {model_label(name)} ---")
        model, memory = load_model_and_memory(name, run, args.device)
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

    results_dir = run.results_dir()
    save_results(results, results_dir)
    run.update_index(results)
    print("\nAvaliação concluída.")


if __name__ == "__main__":
    main()
