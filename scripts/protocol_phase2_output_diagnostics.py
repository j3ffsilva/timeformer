#!/usr/bin/env python3
"""
Phase 2 of the restructured Timeformer evaluation protocol.

Computes output-side diagnostics that target temporal traceability more directly
than the aggregate kNN drift score:

  2A. Class contrast:
      total_path_length(Drift) - total_path_length(Stable)

  2B. Directed semantic drift:
      projection of subject trajectory onto the N1->N2 semantic axis, again
      contrasted as Drift - Stable.

  2C. Continuous trajectory Spearman:
      Spearman between planted P(N1|s,t) and a continuous observed score
      sim(h_s, centroid_N1) - sim(h_s, centroid_N2).

Outputs:
  outputs/protocol/phase2_output_diagnostics.csv
  outputs/protocol/phase2_output_diagnostics_summary.csv
  outputs/protocol/phase2_output_diagnostics.json
  tmp/protocol_phase2_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from src.timeformer.dataset import MLMDataset, load_corpus

from scripts.protocol_phase1_layerwise_cka import (
    DISPLAY,
    MODELS,
    extract_layerwise_subject_reps,
    load_memory,
    load_run_ids,
    load_trained_model,
)


RAW_DEFAULT = Path("outputs/multiseed/multiseed_raw.json")
CORPUS_DEFAULT = Path("data/corpus.tsv")
PARAMS_DEFAULT = Path("data/corpus.params.json")
OUT_DIR = Path("outputs/protocol")
SUMMARY_PATH = Path("tmp/protocol_phase2_summary.md")

STABLE_IDX = list(range(0, 10))
DRIFT_IDX = list(range(10, 20))
BIFURC_IDX = list(range(20, 30))
CLASS_OF = {i: "stable" for i in STABLE_IDX}
CLASS_OF.update({i: "drift" for i in DRIFT_IDX})
CLASS_OF.update({i: "bifurc" for i in BIFURC_IDX})
CLASSES = ("stable", "drift", "bifurc")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / denom)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def subject_epoch_centroids(h: np.ndarray, subject_idx: np.ndarray,
                            epoch_idx: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    centroids = {}
    for s in range(30):
        for t in range(10):
            mask = (subject_idx == s) & (epoch_idx == t)
            if mask.any():
                centroids[(s, t)] = h[mask].mean(axis=0)
    return centroids


def total_path_length(h_by_t: list[np.ndarray]) -> float:
    return float(sum(cosine_distance(h_by_t[i], h_by_t[i + 1])
                     for i in range(len(h_by_t) - 1)))


def semantic_axis(h: np.ndarray, context: np.ndarray) -> np.ndarray:
    mu_n1 = h[context == 0].mean(axis=0)
    mu_n2 = h[context == 1].mean(axis=0)
    axis = mu_n2 - mu_n1
    norm = np.linalg.norm(axis)
    if norm == 0:
        return axis
    return axis / norm


def directed_path(h_by_t: list[np.ndarray], axis: np.ndarray) -> float:
    return float(sum(np.dot(h_by_t[i + 1] - h_by_t[i], axis)
                     for i in range(len(h_by_t) - 1)))


def load_planted(params_path: Path) -> dict[int, list[float]]:
    with params_path.open(encoding="utf-8") as f:
        params = json.load(f)
    planted = {}
    for s_name, values in params["context_a_fractions"].items():
        subject_num = int(s_name[1:])
        planted[subject_num - 1] = [float(v) for v in values]
    return planted


def trajectory_spearman(
    centroids: dict[tuple[int, int], np.ndarray],
    h: np.ndarray,
    context: np.ndarray,
    planted: dict[int, list[float]],
    subjects: list[int],
) -> float:
    cen_n1 = h[context == 0].mean(axis=0)
    cen_n2 = h[context == 1].mean(axis=0)
    rhos = []
    for s in subjects:
        observed, target = [], []
        for t in range(10):
            c = centroids.get((s, t))
            if c is None:
                continue
            observed.append(cosine_similarity(c, cen_n1) - cosine_similarity(c, cen_n2))
            target.append(planted[s][t])
        if len(observed) >= 3 and len(set(np.round(target, 8))) > 1:
            rho, _ = stats.spearmanr(observed, target)
            if not math.isnan(float(rho)):
                rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else float("nan")


def evaluate_model_reps(
    run_id: str,
    model_name: str,
    reps: dict[str, np.ndarray],
    dataset: MLMDataset,
    planted: dict[int, list[float]],
) -> list[dict]:
    h = reps["h2"]
    subject_idx = np.array([int(item["subject_idx"]) for item in dataset._items])
    epoch_idx = np.array([int(item["epoch_idx"]) for item in dataset._items])
    context = np.array([int(item["true_context"]) for item in dataset._items])
    centroids = subject_epoch_centroids(h, subject_idx, epoch_idx)
    axis = semantic_axis(h, context)

    rows: list[dict] = []
    movement_by_class: dict[str, list[float]] = {c: [] for c in CLASSES}
    directed_by_class: dict[str, list[float]] = {c: [] for c in CLASSES}

    for s in range(30):
        h_by_t = [centroids.get((s, t)) for t in range(10)]
        if any(v is None for v in h_by_t):
            continue
        cls = CLASS_OF[s]
        movement_by_class[cls].append(total_path_length(h_by_t))
        directed_by_class[cls].append(directed_path(h_by_t, axis))

    for cls in CLASSES:
        rows.append({
            "run_id": run_id,
            "model": model_name,
            "model_label": DISPLAY[model_name],
            "metric": "path_length",
            "class": cls,
            "value": float(np.mean(movement_by_class[cls])),
        })
        rows.append({
            "run_id": run_id,
            "model": model_name,
            "model_label": DISPLAY[model_name],
            "metric": "directed_drift",
            "class": cls,
            "value": float(np.mean(directed_by_class[cls])),
        })

    rows.extend([
        {
            "run_id": run_id,
            "model": model_name,
            "model_label": DISPLAY[model_name],
            "metric": "path_contrast_drift_minus_stable",
            "class": "drift-stable",
            "value": float(np.mean(movement_by_class["drift"]) -
                           np.mean(movement_by_class["stable"])),
        },
        {
            "run_id": run_id,
            "model": model_name,
            "model_label": DISPLAY[model_name],
            "metric": "directed_contrast_drift_minus_stable",
            "class": "drift-stable",
            "value": float(np.mean(directed_by_class["drift"]) -
                           np.mean(directed_by_class["stable"])),
        },
        {
            "run_id": run_id,
            "model": model_name,
            "model_label": DISPLAY[model_name],
            "metric": "spearman_drift",
            "class": "drift",
            "value": trajectory_spearman(centroids, h, context, planted, DRIFT_IDX),
        },
        {
            "run_id": run_id,
            "model": model_name,
            "model_label": DISPLAY[model_name],
            "metric": "spearman_bifurc",
            "class": "bifurc",
            "value": trajectory_spearman(centroids, h, context, planted, BIFURC_IDX),
        },
    ])
    return rows


def mean_ci(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    se = sd / math.sqrt(len(arr)) if len(arr) else float("nan")
    return {
        "n": int(len(arr)),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def paired_delta(rows: list[dict], metric: str,
                 left: str = "Joint", right: str = "Additive") -> dict:
    by_run_model = {
        (r["run_id"], r["model"]): r["value"]
        for r in rows
        if r["metric"] == metric
    }
    diffs = []
    for run_id in sorted({r["run_id"] for r in rows}):
        a = by_run_model.get((run_id, left))
        b = by_run_model.get((run_id, right))
        if a is not None and b is not None:
            diffs.append(a - b)
    summary = mean_ci(diffs)
    if len(diffs) > 1:
        t_stat, p_val = stats.ttest_1samp(diffs, popmean=0.0)
    else:
        t_stat, p_val = float("nan"), float("nan")
    summary.update({
        "comparison": f"{DISPLAY[left]} - {DISPLAY[right]}",
        "metric": metric,
        "t": float(t_stat),
        "p_two_sided": float(p_val),
    })
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "model", "model_label", "metric", "class", "value"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(summary_rows: list[dict], path: Path) -> None:
    fields = ["model", "model_label", "metric", "class", "n", "mean", "sd", "ci95_low", "ci95_high"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_markdown(summary_rows: list[dict], deltas: list[dict], path: Path) -> None:
    wanted_metrics = [
        "path_contrast_drift_minus_stable",
        "directed_contrast_drift_minus_stable",
        "spearman_drift",
        "spearman_bifurc",
    ]
    lines = [
        "# Fase 2 — Diagnósticos direcionados de saída",
        "",
        "Médias em 31 seeds no split de teste.",
        "",
    ]
    for metric in wanted_metrics:
        lines.append(f"## {metric}")
        for model in MODELS:
            matches = [r for r in summary_rows if r["metric"] == metric and r["model"] == model]
            if not matches:
                continue
            r = matches[0]
            lines.append(
                f"- {DISPLAY[model]}: {r['mean']:+.4f} "
                f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
            )
        delta = [d for d in deltas if d["metric"] == metric]
        if delta:
            d = delta[0]
            lines.append(
                f"- Token-Time − Additive: {d['mean']:+.4f} "
                f"[{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}], "
                f"p={d['p_two_sided']:.4g}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--params", type=Path, default=PARAMS_DEFAULT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    device = torch.device(args.device)
    run_ids = load_run_ids(args.raw)
    if args.max_runs is not None:
        run_ids = run_ids[:args.max_runs]

    rows = load_corpus(args.corpus)
    split_rows = [r for r in rows if r["split"] == args.split]
    dataset = MLMDataset(split_rows, seed=42)
    planted = load_planted(args.params)

    all_rows: list[dict] = []
    for idx, run_id in enumerate(run_ids, start=1):
        print(f"[{idx}/{len(run_ids)}] {run_id}")
        memory = load_memory(run_id)
        if memory is not None:
            memory.to(device)

        for model_name in MODELS:
            model = load_trained_model(run_id, model_name, device)
            reps = extract_layerwise_subject_reps(
                model,
                dataset,
                memory if model_name == "Timeformer" else None,
                args.batch_size,
                device,
            )
            all_rows.extend(evaluate_model_reps(run_id, model_name, reps, dataset, planted))
            del model

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["model"], row["metric"], row["class"])].append(row["value"])

    summary_rows = []
    for (model, metric, cls), values in sorted(grouped.items()):
        s = mean_ci(values)
        summary_rows.append({
            "model": model,
            "model_label": DISPLAY[model],
            "metric": metric,
            "class": cls,
            **s,
        })

    delta_metrics = [
        "path_contrast_drift_minus_stable",
        "directed_contrast_drift_minus_stable",
        "spearman_drift",
        "spearman_bifurc",
    ]
    deltas = [paired_delta(all_rows, metric) for metric in delta_metrics]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "phase2_output_diagnostics.csv"
    summary_path = OUT_DIR / "phase2_output_diagnostics_summary.csv"
    json_path = OUT_DIR / "phase2_output_diagnostics.json"
    write_csv(all_rows, rows_path)
    write_summary_csv(summary_rows, summary_path)
    json_path.write_text(json.dumps({
        "raw_path": str(args.raw),
        "corpus": str(args.corpus),
        "params": str(args.params),
        "split": args.split,
        "n_examples": len(dataset),
        "n_runs": len(run_ids),
        "summary": summary_rows,
        "token_time_minus_additive": deltas,
    }, indent=2), encoding="utf-8")
    write_markdown(summary_rows, deltas, SUMMARY_PATH)

    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
