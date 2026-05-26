"""
Métricas de avaliação para o experimento comparativo da Fase A.4.

Métrica principal:
  corr(P_real, P_pred) — correlação de Pearson entre a trajetória real plantada
  e a trajetória predita pelo modelo, por sujeito ao longo de t0..t5.

  Uma correlação alta significa que o modelo capturou a FORMA da trajetória
  semântica (deriva, bifurcação, estabilidade), não apenas o nível médio.
  Um modelo que prediz P=0.75 em todas as épocas teria corr ≈ 0 para S2/S3.

Métricas secundárias:
  MAE   — erro absoluto médio |P_real - P_pred|
  Brier — erro quadrático médio (P_real - P_pred)²

Referência para P_real: N1 fractions em corpus_generator.py.
"""

import numpy as np
from scipy.stats import pearsonr

from src.corpus_generator import SUBJECTS, SUBJECT_CLASSES
from src.train_embeddings import EPOCHS_ORDER

PHENOMENON_LABELS = {"stable": "Estável", "drift": "Deriva", "bifurcation": "Bifurcação"}


def get_p_real(
    fractions: dict[str, list[float]],
    subjects: list[str] = SUBJECTS,
) -> dict[str, list[float]]:
    """Retorna P_real para os sujeitos dado o dict de frações plantadas."""
    return {s: list(fractions[s]) for s in subjects}


def trajectory_metrics(
    p_real: list[float],
    p_pred: list[float],
) -> dict[str, float]:
    """Computa métricas entre uma trajetória real e predita."""
    r = np.array(p_real)
    p = np.array(p_pred)
    mae   = float(np.mean(np.abs(r - p)))
    brier = float(np.mean((r - p) ** 2))
    if np.std(r) < 1e-6 or np.std(p) < 1e-6:
        corr = float("nan")
    else:
        corr = float(pearsonr(r, p)[0])
    return {"corr": corr, "mae": mae, "brier": brier}


def evaluate_all(
    p_pred_dict: dict[str, list[float]],
    p_real_dict: dict[str, list[float]],
    subjects: list[str] = SUBJECTS,
) -> dict[str, dict[str, float]]:
    """
    Para cada sujeito, computa corr/MAE/Brier entre P_real e P_pred.

    Retorna dict[subject] → {"corr": ..., "mae": ..., "brier": ...}.
    """
    return {
        s: trajectory_metrics(p_real_dict[s], p_pred_dict[s])
        for s in subjects
    }


def summarize_results(
    results_by_model: dict[str, dict[str, dict[str, float]]],
    subjects: list[str] = SUBJECTS,
) -> None:
    """Imprime tabela comparativa de correlação por modelo e sujeito."""
    models = list(results_by_model.keys())
    print(f"\n{'Correlação P_real vs P_pred por sujeito e modelo':}")
    print(f"  {'sujeito':<8} {'classe':<13}" + "".join(f"  {m:>10}" for m in models))
    print("  " + "-" * (8 + 13 + 12 * len(models)))

    for subj in subjects:
        cls = PHENOMENON_LABELS.get(SUBJECT_CLASSES.get(subj, ""), subj)
        row = f"  {subj:<8} {cls:<13}"
        for m in models:
            v = results_by_model[m][subj]["corr"]
            row += f"  {v:>10.3f}" if not np.isnan(v) else f"  {'n/a':>10}"
        print(row)

    print(f"\n  {'média':<21}" + "".join(
        f"  {np.nanmean([results_by_model[m][s]['corr'] for s in subjects]):>10.3f}"
        for m in models
    ))

    print(f"\n{'MAE médio por modelo':}")
    for m in models:
        mae_vals = [results_by_model[m][s]["mae"] for s in subjects]
        print(f"  {m:<12}: {np.mean(mae_vals):.4f}")


def aggregate_over_seeds(
    seed_results: list[dict[str, dict[str, dict[str, float]]]],
    subjects: list[str] = SUBJECTS,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Agrega resultados de múltiplos seeds: retorna média ± std por (modelo, sujeito, métrica).

    Entrada: lista de dicts results_by_model (um por seed).
    Saída: mesmo formato, mas com valores {"mean": ..., "std": ...} em vez de scalar.
    """
    models = list(seed_results[0].keys())
    agg: dict[str, dict[str, dict[str, dict]]] = {}

    for m in models:
        agg[m] = {}
        for s in subjects:
            agg[m][s] = {}
            for metric in ["corr", "mae", "brier"]:
                vals = [r[m][s][metric] for r in seed_results if not np.isnan(r[m][s][metric])]
                agg[m][s][metric] = {
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std":  float(np.std(vals))  if vals else float("nan"),
                }
    return agg


def print_sweep_summary(
    sweep_results: dict[float, list[dict]],
    subjects: list[str] = SUBJECTS,
) -> None:
    """
    Imprime tabela de correlação média por (p_canon, modelo).

    sweep_results: dict[p_canon → lista de results_by_model (um por seed)]
    """
    all_models = list(next(iter(sweep_results.values()))[0].keys())

    print("\n" + "=" * 70)
    print("Fase A.4 — Sweep de p_canon: Correlação média (±std) por modelo")
    print("=" * 70)
    print(f"  {'p_canon':<9}" + "".join(f"  {m:>14}" for m in all_models))
    print("  " + "-" * (9 + 16 * len(all_models)))

    for p_canon in sorted(sweep_results.keys()):
        agg = aggregate_over_seeds(sweep_results[p_canon], subjects)
        row = f"  {p_canon:<9.2f}"
        for m in all_models:
            mean_corr = np.nanmean([agg[m][s]["corr"]["mean"] for s in subjects])
            std_corr  = np.nanmean([agg[m][s]["corr"]["std"]  for s in subjects])
            row += f"  {mean_corr:>6.3f}±{std_corr:.3f}"
        print(row)

    print()
