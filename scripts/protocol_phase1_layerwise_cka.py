#!/usr/bin/env python3
"""
Phase 1 of the restructured Timeformer evaluation protocol.

Extracts subject representations at three forward-pass points:
  z  = input subject embedding after temporal conditioning, before encoder
  h1 = subject hidden state after encoder layer 1
  h2 = subject hidden state after encoder layer 2

Then computes matched linear CKA and normalized orthogonal Procrustes disparity
for informative architecture pairs across paired seeds.

Outputs:
  outputs/protocol/phase1_layerwise_cka.csv
  outputs/protocol/phase1_layerwise_cka_summary.csv
  outputs/protocol/phase1_layerwise_cka.json
  tmp/protocol_phase1_summary.md
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
from scipy.spatial import procrustes
from torch.utils.data import DataLoader

from src.timeformer.dataset import MLMDataset, POS_SUBJECT, load_corpus
from src.timeformer.memory import PrototypeMemory
from src.timeformer.models import build_model
from src.timeformer.train import load_checkpoint


RAW_DEFAULT = Path("outputs/multiseed/multiseed_raw.json")
CORPUS_DEFAULT = Path("data/corpus.tsv")
OUT_DIR = Path("outputs/protocol")
SUMMARY_PATH = Path("tmp/protocol_phase1_summary.md")

MODELS = ("Static", "Additive", "Joint", "Timeformer")
POINTS = ("z", "h1", "h2")
PAIRS = (
    ("Additive", "Joint"),
    ("Static", "Additive"),
    ("Static", "Joint"),
    ("Joint", "Timeformer"),
)
DISPLAY = {
    "Static": "Standard",
    "Additive": "Additive",
    "Joint": "Token-Time",
    "Timeformer": "Memory-Augmented",
}


def load_run_ids(raw_path: Path) -> list[str]:
    with raw_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return [r["run_id"] for r in raw]


def load_memory(run_id: str, prefer_best: bool = True) -> PrototypeMemory | None:
    model_dir = Path("outputs/runs") / run_id / "Timeformer"
    names = ("memory_best.pkl", "memory.pkl") if prefer_best else ("memory.pkl",)
    for name in names:
        path = model_dir / name
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
    return None


def load_trained_model(run_id: str, model_name: str, device: torch.device) -> torch.nn.Module:
    model = build_model(model_name)
    ckpt = Path("outputs/runs") / run_id / model_name / "best.pt"
    load_checkpoint(model, ckpt)
    model.to(device)
    model.eval()
    return model


def memory_batch(
    memory: PrototypeMemory,
    epoch_idx: torch.Tensor,
    subject_idx: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mem_list, mask_list = [], []
    for i in range(epoch_idx.size(0)):
        k = int(epoch_idx[i].item())
        s = subject_idx[i:i + 1]
        m_i, mk_i = memory.get(s, epoch_k=k)
        mem_list.append(m_i)
        mask_list.append(mk_i)

    max_hist = max(m.size(1) for m in mem_list)
    d = memory.d_model
    mem_b = torch.zeros(len(mem_list), max_hist, d, device=device)
    mask_b = torch.zeros(len(mem_list), max_hist, dtype=torch.bool, device=device)
    for i, (m_i, mk_i) in enumerate(zip(mem_list, mask_list)):
        h = m_i.size(1)
        if h > 0:
            mem_b[i, :h, :] = m_i.to(device)
            mask_b[i, :h] = mk_i.to(device)
    return mem_b, mask_b


@torch.no_grad()
def extract_layerwise_subject_reps(
    model: torch.nn.Module,
    dataset: MLMDataset,
    memory: PrototypeMemory | None,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Return z, h1, h2 arrays of shape (N, d), matched to dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model_name = type(model).__name__
    reps: dict[str, list[np.ndarray]] = {p: [] for p in POINTS}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        epoch_idx = batch["epoch_idx"].to(device)
        subject_idx = batch["subject_idx"].to(device)

        if model_name == "Static":
            x = model.embed(input_ids)
        elif model_name in ("Additive", "Joint"):
            x = model.embed(input_ids, epoch_idx)
        else:
            if memory is None:
                x = model.embed(input_ids, epoch_idx, memory=None)
            else:
                mem_b, mask_b = memory_batch(memory, epoch_idx, subject_idx, device)
                x = model.embed(input_ids, epoch_idx, memory=mem_b, memory_mask=mask_b)

        reps["z"].append(x[:, POS_SUBJECT, :].detach().cpu().numpy())
        for layer_idx, layer in enumerate(model.encoder.encoder.layers, start=1):
            x = layer(x)
            if layer_idx <= 2:
                reps[f"h{layer_idx}"].append(x[:, POS_SUBJECT, :].detach().cpu().numpy())

    return {p: np.concatenate(chunks, axis=0) for p, chunks in reps.items()}


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x_t = torch.as_tensor(x, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    x_t = x_t - x_t.mean(dim=0, keepdim=True)
    y_t = y_t - y_t.mean(dim=0, keepdim=True)
    xty = x_t.T @ y_t
    xtx = x_t.T @ x_t
    yty = y_t.T @ y_t
    hsic = (xty ** 2).sum()
    denom = torch.sqrt((xtx ** 2).sum()) * torch.sqrt((yty ** 2).sum())
    if denom.item() == 0:
        return float("nan")
    return float((hsic / denom).item())


def procrustes_disparity(x: np.ndarray, y: np.ndarray) -> float:
    # scipy.spatial.procrustes centers and normalizes both matrices internally.
    _, _, disparity = procrustes(x, y)
    return float(disparity)


def mean_ci(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    se = sd / math.sqrt(len(arr)) if len(arr) > 0 else float("nan")
    return {
        "n": int(len(arr)),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def write_rows_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "pair", "left", "right", "point", "cka", "procrustes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(summary_rows: list[dict], path: Path) -> None:
    fields = [
        "pair", "point", "n",
        "cka_mean", "cka_ci95_low", "cka_ci95_high",
        "procrustes_mean", "procrustes_ci95_low", "procrustes_ci95_high",
    ]
    with path.open("w", encoding="utf-8", newline="",) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_markdown(summary_rows: list[dict], path: Path, n_runs: int, n_examples: int) -> None:
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        by_pair[row["pair"]].append(row)

    lines = [
        "# Fase 1 — CKA camada a camada",
        "",
        f"Médias em {n_runs} seed(s), usando {n_examples} exemplos pareados na mesma ordem.",
        "",
    ]
    for pair in sorted(by_pair):
        lines.append(f"## {pair}")
        for row in sorted(by_pair[pair], key=lambda r: POINTS.index(r["point"])):
            lines.append(
                f"- {row['point']}: CKA={row['cka_mean']:.3f} "
                f"[{row['cka_ci95_low']:.3f}, {row['cka_ci95_high']:.3f}], "
                f"Procrustes={row['procrustes_mean']:.3f}"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--split", default="test",
                        help="Corpus split used as matched evaluation batch.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_ids = load_run_ids(args.raw)
    if args.max_runs is not None:
        run_ids = run_ids[:args.max_runs]

    rows = load_corpus(args.corpus)
    split_rows = [r for r in rows if r["split"] == args.split]
    if not split_rows:
        raise SystemExit(f"No rows found for split={args.split!r}")
    dataset = MLMDataset(split_rows, seed=42)

    result_rows: list[dict] = []
    for idx, run_id in enumerate(run_ids, start=1):
        print(f"[{idx}/{len(run_ids)}] {run_id}")
        reps_by_model: dict[str, dict[str, np.ndarray]] = {}
        memory = load_memory(run_id)
        if memory is not None:
            memory.to(device)

        for model_name in MODELS:
            model = load_trained_model(run_id, model_name, device)
            model_memory = memory if model_name == "Timeformer" else None
            reps_by_model[model_name] = extract_layerwise_subject_reps(
                model, dataset, model_memory, args.batch_size, device
            )
            del model

        for left, right in PAIRS:
            pair_name = f"{DISPLAY[left]} vs {DISPLAY[right]}"
            for point in POINTS:
                x = reps_by_model[left][point]
                y = reps_by_model[right][point]
                result_rows.append({
                    "run_id": run_id,
                    "pair": pair_name,
                    "left": left,
                    "right": right,
                    "point": point,
                    "cka": linear_cka(x, y),
                    "procrustes": procrustes_disparity(x, y),
                })

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"cka": [], "procrustes": []}
    )
    for row in result_rows:
        grouped[(row["pair"], row["point"])]["cka"].append(row["cka"])
        grouped[(row["pair"], row["point"])]["procrustes"].append(row["procrustes"])

    summary_rows: list[dict] = []
    for (pair, point), metrics in sorted(grouped.items()):
        cka_stats = mean_ci(metrics["cka"])
        proc_stats = mean_ci(metrics["procrustes"])
        summary_rows.append({
            "pair": pair,
            "point": point,
            "n": cka_stats["n"],
            "cka_mean": cka_stats["mean"],
            "cka_ci95_low": cka_stats["ci95_low"],
            "cka_ci95_high": cka_stats["ci95_high"],
            "procrustes_mean": proc_stats["mean"],
            "procrustes_ci95_low": proc_stats["ci95_low"],
            "procrustes_ci95_high": proc_stats["ci95_high"],
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "phase1_layerwise_cka.csv"
    summary_csv = OUT_DIR / "phase1_layerwise_cka_summary.csv"
    json_path = OUT_DIR / "phase1_layerwise_cka.json"
    write_rows_csv(result_rows, rows_path)
    write_summary_csv(summary_rows, summary_csv)
    json_path.write_text(json.dumps({
        "raw_path": str(args.raw),
        "corpus": str(args.corpus),
        "split": args.split,
        "n_examples": len(dataset),
        "n_runs": len(run_ids),
        "rows": result_rows,
        "summary": summary_rows,
    }, indent=2), encoding="utf-8")
    write_markdown(summary_rows, SUMMARY_PATH, n_runs=len(run_ids), n_examples=len(dataset))

    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {json_path}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
