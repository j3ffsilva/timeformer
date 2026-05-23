"""
Fase A.4 — Experimento comparativo: Static vs Independent vs TimeGlobal vs TimeInteractive.

Sweep: p_canon ∈ {0.60, 0.70, 0.75, 0.85} × seeds {42, 43, 44}
Modelos: A=Static, B=Independent, C=TimeGlobal, D=TimeInteractive
Métrica principal: corr(P_real, P_pred) por sujeito ao longo de t0..t5
"""

import json
from pathlib import Path

import numpy as np
import torch

from src.corpus_generator import (
    SUBJECTS,
    generate_corpus_probabilistic,
)
from src.models import (
    StaticSkipGram,
    TimeGlobalSkipGram,
    TimeInteractiveSkipGram,
    compute_p_pred_trajectory,
    train_joint_model,
)
from src.train_embeddings import (
    CONTEXT_A_INDICES,
    EPOCHS_ORDER,
    TOKEN_TO_IDX,
    SkipGram,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    LEARNING_RATE,
    TRAIN_STEPS,
    _build_skipgram_pairs,
)
from src.trajectory_eval import (
    aggregate_over_seeds,
    evaluate_all,
    get_p_real,
    print_sweep_summary,
    summarize_results,
)

import torch.nn as nn
import torch.optim as optim

P_CANON_VALUES = [0.60, 0.70, 0.75, 0.85]
SEEDS = [42, 43, 44]
OUTPUT_DIR = Path("outputs_a4")


# ─── Modelo B: treino independente por época ──────────────────────────────────

def _train_independent_epoch(sentences: list[str], seed: int) -> SkipGram:
    torch.manual_seed(seed)
    centers, contexts = _build_skipgram_pairs(sentences)
    model = SkipGram(VOCAB_SIZE, EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad()
        loss = criterion(model(centers), contexts)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def compute_p_pred_independent(corpus_rows: list[dict], seed: int) -> dict[str, list[float]]:
    """Treina um SkipGram por época e retorna P(ctx=A) por sujeito."""
    by_epoch: dict[str, list[str]] = {}
    for row in corpus_rows:
        if row.get("split", "train") == "train":
            by_epoch.setdefault(row["epoch"], []).append(row["sentence"])

    epoch_models: dict[str, SkipGram] = {}
    for ep in EPOCHS_ORDER:
        epoch_models[ep] = _train_independent_epoch(by_epoch[ep], seed)

    result: dict[str, list[float]] = {s: [] for s in SUBJECTS}
    with torch.no_grad():
        for ep in EPOCHS_ORDER:
            model = epoch_models[ep]
            for subject in SUBJECTS:
                logits = model(torch.tensor([TOKEN_TO_IDX[subject]]))
                probs = torch.softmax(logits, dim=-1).squeeze()
                p_a = float(probs[CONTEXT_A_INDICES].sum())
                result[subject].append(p_a)
    return result


# ─── Runner principal ─────────────────────────────────────────────────────────

def run_one(p_canon: float, seed: int) -> dict[str, dict[str, dict[str, float]]]:
    """
    Para um (p_canon, seed), gera corpus, treina 4 modelos, avalia.
    Retorna results_by_model: dict[model_name → dict[subject → metrics]].
    """
    corpus_path = OUTPUT_DIR / f"corpus_p{p_canon:.2f}_s{seed}.tsv"
    corpus_rows = generate_corpus_probabilistic(corpus_path, p_canon=p_canon, seed=seed)

    p_real = get_p_real()

    # Modelo A — Static
    model_a = StaticSkipGram()
    train_joint_model(model_a, corpus_rows, seed=seed)
    p_pred_a = compute_p_pred_trajectory(model_a, SUBJECTS)

    # Modelo B — Independent
    p_pred_b = compute_p_pred_independent(corpus_rows, seed=seed)

    # Modelo C — TimeGlobal
    model_c = TimeGlobalSkipGram()
    train_joint_model(model_c, corpus_rows, seed=seed)
    p_pred_c = compute_p_pred_trajectory(model_c, SUBJECTS)

    # Modelo D — TimeInteractive
    model_d = TimeInteractiveSkipGram()
    train_joint_model(model_d, corpus_rows, seed=seed)
    p_pred_d = compute_p_pred_trajectory(model_d, SUBJECTS)

    return {
        "Static":      evaluate_all(p_pred_a, p_real),
        "Independent": evaluate_all(p_pred_b, p_real),
        "TimeGlobal":  evaluate_all(p_pred_c, p_real),
        "TimeInteractive": evaluate_all(p_pred_d, p_real),
    }


def save_results_tsv(
    sweep_results: dict[float, list[dict]],
    output_path: Path,
) -> None:
    """Salva resultados em TSV: p_canon, seed, model, subject, corr, mae, brier."""
    rows = []
    for p_canon, seed_list in sweep_results.items():
        for seed_idx, results_by_model in enumerate(seed_list):
            seed = SEEDS[seed_idx]
            for model_name, subj_metrics in results_by_model.items():
                for subject, metrics in subj_metrics.items():
                    rows.append({
                        "p_canon": p_canon,
                        "seed": seed,
                        "model": model_name,
                        "subject": subject,
                        "corr": metrics["corr"],
                        "mae": metrics["mae"],
                        "brier": metrics["brier"],
                    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("p_canon\tseed\tmodel\tsubject\tcorr\tmae\tbrier\n")
        for r in rows:
            corr_str = f"{r['corr']:.6f}" if not (isinstance(r['corr'], float) and r['corr'] != r['corr']) else "nan"
            f.write(f"{r['p_canon']:.2f}\t{r['seed']}\t{r['model']}\t{r['subject']}\t{corr_str}\t{r['mae']:.6f}\t{r['brier']:.6f}\n")

    print(f"\nResultados salvos em {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # sweep_results: dict[p_canon → list[results_by_model per seed]]
    sweep_results: dict[float, list[dict]] = {}

    for p_canon in P_CANON_VALUES:
        sweep_results[p_canon] = []
        print(f"\n{'='*60}")
        print(f"p_canon = {p_canon:.2f}")
        print(f"{'='*60}")

        for seed in SEEDS:
            print(f"\n--- seed={seed} ---")
            result = run_one(p_canon, seed)
            sweep_results[p_canon].append(result)

            # Detalhe por seed
            summarize_results(result)

    # Tabela de sweep
    print_sweep_summary(sweep_results)

    # Resultado principal: p_canon=0.75 (ponto de partida da proposta)
    print("\n" + "=" * 60)
    print("Agregado (mean ± std) para p_canon = 0.75")
    print("=" * 60)
    agg = aggregate_over_seeds(sweep_results[0.75])
    for model_name in ["Static", "Independent", "TimeGlobal", "TimeInteractive"]:
        print(f"\n  {model_name}:")
        for subject in SUBJECTS:
            c = agg[model_name][subject]["corr"]
            m = agg[model_name][subject]["mae"]
            print(f"    {subject}: corr={c['mean']:+.3f}±{c['std']:.3f}  mae={m['mean']:.3f}±{m['std']:.3f}")

    save_results_tsv(sweep_results, OUTPUT_DIR / "results.tsv")

    # Salva JSON completo para análise posterior
    with open(OUTPUT_DIR / "sweep_results.json", "w") as f:
        json.dump(
            {str(k): v for k, v in sweep_results.items()},
            f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
        )
    print(f"JSON completo salvo em {OUTPUT_DIR}/sweep_results.json")


if __name__ == "__main__":
    main()
