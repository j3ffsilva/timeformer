#!/usr/bin/env python3
"""
Phase 4B of the restructured Timeformer evaluation protocol.

Bifurcation as mode preservation.

For bifurcating subjects in late periods, evaluate whether per-occurrence
subject representations preserve two context modes (N1 vs N2) or collapse to a
single centroid. Metrics:

  - silhouette_cosine: silhouette score of h_s occurrences with true_context labels
  - centroid_cosine_distance: 1 - cosine(centroid_N1, centroid_N2)
  - semantic_axis_auroc: AUROC of projection onto global N1->N2 semantic axis

Each metric is computed in two ways:
  - pooled: all late bifurcating occurrences together
  - subject_mean: compute per subject, then average over subjects with both labels

Outputs:
  outputs/protocol/phase4_bifurcation_modes.csv
  outputs/protocol/phase4_bifurcation_modes_summary.csv
  outputs/protocol/phase4_bifurcation_modes_deltas.csv
  outputs/protocol/phase4_bifurcation_modes.json
  tmp/protocol_phase4b_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, silhouette_score

from src.timeformer.dataset import MLMDataset, load_corpus
from src.timeformer.probe import extract_reps
from src.timeformer.train import load_checkpoint
from src.timeformer.models import build_model

from scripts.protocol_phase1_layerwise_cka import DISPLAY, load_run_ids


RAW_DEFAULT = Path("outputs/multiseed/multiseed_raw.json")
CORPUS_DEFAULT = Path("data/corpus.tsv")
OUT_DIR = Path("outputs/protocol")
SUMMARY_MD = Path("tmp/protocol_phase4b_summary.md")

MODELS = ("Static", "Additive", "Joint", "Timeformer")
BIFURC_SUBJECTS = set(range(20, 30))
METRICS = ("silhouette_cosine", "centroid_cosine_distance", "semantic_axis_auroc")


def parse_splits(value: str) -> set[str]:
    return {v.strip() for v in value.split(",") if v.strip()}


def load_memory(run_id: str):
    model_dir = Path("outputs/runs") / run_id / "Timeformer"
    for name in ("memory_best.pkl", "memory.pkl"):
        path = model_dir / name
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
    return None


def load_model(run_id: str, model_name: str, device: str):
    model = build_model(model_name)
    ckpt = Path("outputs/runs") / run_id / model_name / "best.pt"
    load_checkpoint(model, ckpt)
    model.to(torch.device(device))
    model.eval()
    return model


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / denom)


def semantic_axis(h: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu0 = h[y == 0].mean(axis=0)
    mu1 = h[y == 1].mean(axis=0)
    u = mu1 - mu0
    norm = np.linalg.norm(u)
    return u / norm if norm > 0 else u


def auroc_on_axis(h: np.ndarray, y: np.ndarray, u: np.ndarray) -> float:
    scores = h @ u
    auc = float(roc_auc_score(y, scores))
    return max(auc, 1.0 - auc)


def valid_two_mode(y: np.ndarray) -> bool:
    return len(y) >= 4 and len(np.unique(y)) == 2 and min(np.bincount(y.astype(int))) >= 2


def mode_metrics(h: np.ndarray, y: np.ndarray, u: np.ndarray) -> dict[str, float]:
    if not valid_two_mode(y):
        return {m: float("nan") for m in METRICS}
    return {
        "silhouette_cosine": float(silhouette_score(h, y, metric="cosine")),
        "centroid_cosine_distance": cosine_distance(h[y == 0].mean(axis=0), h[y == 1].mean(axis=0)),
        "semantic_axis_auroc": auroc_on_axis(h, y, u),
    }


def evaluate_reps(run_id: str, model_name: str, reps: dict, late_start: int) -> list[dict]:
    h = reps["h_subj"]
    y = reps["true_context"].astype(int)
    subj = reps["subject_idx"].astype(int)
    epoch = reps["epoch_idx"].astype(int)
    u = semantic_axis(h, y)

    late_bifurc = np.array([
        int(s) in BIFURC_SUBJECTS and int(t) >= late_start
        for s, t in zip(subj, epoch)
    ])

    rows: list[dict] = []
    pooled = mode_metrics(h[late_bifurc], y[late_bifurc], u)
    for metric, value in pooled.items():
        rows.append({
            "run_id": run_id,
            "model": model_name,
            "model_label": DISPLAY[model_name],
            "scope": "pooled",
            "metric": metric,
            "value": value,
            "n_occurrences": int(late_bifurc.sum()),
            "n_subjects": 10,
        })

    per_subject_values: dict[str, list[float]] = {m: [] for m in METRICS}
    valid_subjects = 0
    for s in sorted(BIFURC_SUBJECTS):
        mask = late_bifurc & (subj == s)
        if not valid_two_mode(y[mask]):
            continue
        valid_subjects += 1
        sm = mode_metrics(h[mask], y[mask], u)
        for metric, value in sm.items():
            if not math.isnan(value):
                per_subject_values[metric].append(value)

    for metric, values in per_subject_values.items():
        rows.append({
            "run_id": run_id,
            "model": model_name,
            "model_label": DISPLAY[model_name],
            "scope": "subject_mean",
            "metric": metric,
            "value": float(np.mean(values)) if values else float("nan"),
            "n_occurrences": int(late_bifurc.sum()),
            "n_subjects": valid_subjects,
        })

    return rows


def mean_ci(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    mean = float(arr.mean()) if len(arr) else float("nan")
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    se = sd / math.sqrt(len(arr)) if len(arr) else float("nan")
    return {
        "n": int(len(arr)),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["scope"], row["metric"])].append(row["value"])
    summary = []
    for (model, scope, metric), values in sorted(grouped.items()):
        summary.append({
            "model": model,
            "model_label": DISPLAY[model],
            "scope": scope,
            "metric": metric,
            **mean_ci(values),
        })
    return summary


def paired_deltas(rows: list[dict]) -> list[dict]:
    pairs = (
        ("Joint", "Additive"),
        ("Timeformer", "Joint"),
        ("Timeformer", "Additive"),
    )
    by_key = {
        (r["run_id"], r["model"], r["scope"], r["metric"]): r["value"]
        for r in rows
    }
    deltas = []
    run_ids = sorted({r["run_id"] for r in rows})
    for left, right in pairs:
        for scope in ("pooled", "subject_mean"):
            for metric in METRICS:
                values = []
                for run_id in run_ids:
                    a = by_key.get((run_id, left, scope, metric))
                    b = by_key.get((run_id, right, scope, metric))
                    if a is not None and b is not None and not (math.isnan(a) or math.isnan(b)):
                        values.append(a - b)
                s = mean_ci(values)
                if len(values) > 1:
                    t_stat, p_val = stats.ttest_1samp(values, popmean=0.0)
                else:
                    t_stat = p_val = float("nan")
                deltas.append({
                    "comparison": f"{DISPLAY[left]} - {DISPLAY[right]}",
                    "left": left,
                    "right": right,
                    "scope": scope,
                    "metric": metric,
                    **s,
                    "t": float(t_stat),
                    "p_two_sided": float(p_val),
                })
    return deltas


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(summary: list[dict], deltas: list[dict], path: Path) -> None:
    lines = [
        "# Fase 4B — Bifurcação como dois modos",
        "",
        "Métricas em ocorrências tardias de sujeitos bifurcantes. Valores maiores indicam maior separação N1/N2.",
        "",
    ]
    for scope in ("pooled", "subject_mean"):
        lines.append(f"## {scope}")
        for metric in METRICS:
            lines.append(f"### {metric}")
            for model in MODELS:
                match = next(
                    (r for r in summary if r["model"] == model and r["scope"] == scope and r["metric"] == metric),
                    None,
                )
                if match:
                    lines.append(
                        f"- {DISPLAY[model]}: {match['mean']:+.4f} "
                        f"[{match['ci95_low']:+.4f}, {match['ci95_high']:+.4f}]"
                    )
            for comparison in ("Token-Time - Additive", "Memory-Augmented - Token-Time"):
                d = next(
                    (r for r in deltas if r["comparison"] == comparison and r["scope"] == scope and r["metric"] == metric),
                    None,
                )
                if d:
                    lines.append(
                        f"- {comparison}: {d['mean']:+.4f} "
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
    parser.add_argument("--splits", default="test,hard_verb,hard_both")
    parser.add_argument("--late-start", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_ids = load_run_ids(args.raw)
    if args.max_runs is not None:
        run_ids = run_ids[:args.max_runs]

    split_set = parse_splits(args.splits)
    corpus_rows = load_corpus(args.corpus)
    eval_rows = [r for r in corpus_rows if r["split"] in split_set]
    dataset = MLMDataset(eval_rows, seed=42)

    rows: list[dict] = []
    for idx, run_id in enumerate(run_ids, start=1):
        print(f"[{idx}/{len(run_ids)}] {run_id}")
        memory = load_memory(run_id)
        if memory is not None:
            memory.to(device)
        for model_name in MODELS:
            model = load_model(run_id, model_name, args.device)
            reps = extract_reps(
                model,
                dataset,
                memory=memory if model_name == "Timeformer" else None,
                batch_size=args.batch_size,
                device=device,
            )
            rows.extend(evaluate_reps(run_id, model_name, reps, args.late_start))
            del model

    summary = summarize(rows)
    deltas = paired_deltas(rows)

    rows_path = OUT_DIR / "phase4_bifurcation_modes.csv"
    summary_path = OUT_DIR / "phase4_bifurcation_modes_summary.csv"
    deltas_path = OUT_DIR / "phase4_bifurcation_modes_deltas.csv"
    json_path = OUT_DIR / "phase4_bifurcation_modes.json"

    write_csv(
        rows,
        rows_path,
        ["run_id", "model", "model_label", "scope", "metric", "value", "n_occurrences", "n_subjects"],
    )
    write_csv(
        summary,
        summary_path,
        ["model", "model_label", "scope", "metric", "n", "mean", "sd", "ci95_low", "ci95_high"],
    )
    write_csv(
        deltas,
        deltas_path,
        ["comparison", "left", "right", "scope", "metric", "n", "mean", "sd",
         "ci95_low", "ci95_high", "t", "p_two_sided"],
    )
    json_path.write_text(json.dumps({
        "raw_path": str(args.raw),
        "corpus": str(args.corpus),
        "splits": sorted(split_set),
        "late_start": args.late_start,
        "n_examples": len(dataset),
        "n_runs": len(run_ids),
        "rows": rows,
        "summary": summary,
        "deltas": deltas,
    }, indent=2), encoding="utf-8")
    write_markdown(summary, deltas, SUMMARY_MD)

    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {deltas_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
