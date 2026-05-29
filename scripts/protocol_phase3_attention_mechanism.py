#!/usr/bin/env python3
"""
Phase 3 of the restructured Timeformer evaluation protocol.

Mechanistic attention diagnostics:

  3A. Descriptive attention:
      replay each Transformer layer while requesting per-head attention weights.
      Report subject-query attention mass to verb/object, entropy, and variance
      across periods.

  3B. Causal head ablation:
      for Token-Time (Joint), zero one self-attention head before the output
      projection and recompute CKA(Additive, Token-Time_ablated) at h2.

Outputs:
  outputs/protocol/phase3_attention_by_epoch.csv
  outputs/protocol/phase3_attention_summary.csv
  outputs/protocol/phase3_head_ablation.csv
  outputs/protocol/phase3_head_ablation_summary.csv
  outputs/protocol/phase3_attention_mechanism.json
  tmp/protocol_phase3_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader

from src.timeformer.dataset import (
    MLMDataset,
    POS_OBJECT,
    POS_SUBJECT,
    POS_VERB,
    load_corpus,
)

from scripts.protocol_phase1_layerwise_cka import (
    DISPLAY,
    linear_cka,
    load_run_ids,
    load_trained_model,
)


RAW_DEFAULT = Path("outputs/multiseed/multiseed_raw.json")
CORPUS_DEFAULT = Path("data/corpus.tsv")
OUT_DIR = Path("outputs/protocol")
SUMMARY_PATH = Path("tmp/protocol_phase3_summary.md")

MODELS = ("Additive", "Joint")
N_LAYERS = 2
N_HEADS = 4


def entropy(prob: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = prob.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=dim)


def self_attention_with_weights(
    module: torch.nn.MultiheadAttention,
    x: torch.Tensor,
    ablate_head: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-attention for batch_first=True MHA, with optional pre-out_proj head ablation."""
    if not module.batch_first:
        raise ValueError("This script expects batch_first=True MultiheadAttention.")

    batch, seq_len, embed_dim = x.shape
    n_heads = module.num_heads
    head_dim = embed_dim // n_heads
    if module.in_proj_weight is None:
        raise ValueError("Separate q/k/v projection weights are not supported here.")

    qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    q = q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)
    head_out = torch.matmul(attn, v)

    if ablate_head is not None:
        head_out = head_out.clone()
        head_out[:, ablate_head, :, :] = 0.0

    concat = head_out.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
    out = module.out_proj(concat)
    return out, attn


def replay_layer(
    layer: torch.nn.TransformerEncoderLayer,
    x: torch.Tensor,
    ablate_head: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay one pre-LN TransformerEncoderLayer and return output plus attention weights."""
    if not getattr(layer, "norm_first", False):
        attn_out, attn = self_attention_with_weights(layer.self_attn, x, ablate_head)
        x = layer.norm1(x + layer.dropout1(attn_out))
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm2(x + layer.dropout2(ff))
        return x, attn

    attn_out, attn = self_attention_with_weights(layer.self_attn, layer.norm1(x), ablate_head)
    x = x + layer.dropout1(attn_out)
    ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x)))))
    x = x + layer.dropout2(ff)
    return x, attn


@torch.no_grad()
def initial_embeddings(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    epoch_idx: torch.Tensor,
) -> torch.Tensor:
    model_name = type(model).__name__
    if model_name == "Static":
        return model.embed(input_ids)
    return model.embed(input_ids, epoch_idx)


@torch.no_grad()
def extract_attention_and_hidden(
    model: torch.nn.Module,
    dataset: MLMDataset,
    batch_size: int,
    device: torch.device,
    ablate: tuple[int, int] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Return h2 reps and per-example/layer/head attention metrics."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    h2_chunks: list[np.ndarray] = []
    attention_rows: list[dict] = []

    example_offset = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        epoch_idx = batch["epoch_idx"].to(device)
        x = initial_embeddings(model, input_ids, epoch_idx)

        for layer_idx, layer in enumerate(model.encoder.encoder.layers):
            ablate_head = None
            if ablate is not None and ablate[0] == layer_idx:
                ablate_head = ablate[1]
            x, attn = replay_layer(layer, x, ablate_head=ablate_head)

            # attn: (B, heads, query_len, key_len)
            subj_attn = attn[:, :, POS_SUBJECT, :].detach().cpu()
            epoch_np = batch["epoch_idx"].numpy()
            for b in range(subj_attn.size(0)):
                for head in range(subj_attn.size(1)):
                    weights = subj_attn[b, head]
                    attention_rows.append({
                        "example_idx": example_offset + b,
                        "epoch": int(epoch_np[b]),
                        "layer": layer_idx,
                        "head": head,
                        "verb_mass": float(weights[POS_VERB].item()),
                        "object_mass": float(weights[POS_OBJECT].item()),
                        "context_mass": float((weights[POS_VERB] + weights[POS_OBJECT]).item()),
                        "entropy": float(entropy(weights).item()),
                    })

        h2_chunks.append(x[:, POS_SUBJECT, :].detach().cpu().numpy())
        example_offset += input_ids.size(0)

    return {"h2": np.concatenate(h2_chunks, axis=0)}, attention_rows


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


def summarize_attention(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Aggregate per run/model/layer/head/epoch and then across seeds."""
    by_epoch_values: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["run_id"], row["model"], row["layer"], row["head"], row["epoch"])
        for metric in ("verb_mass", "object_mass", "context_mass", "entropy"):
            by_epoch_values[key][metric].append(row[metric])

    epoch_rows = []
    for (run_id, model, layer, head, epoch), metrics in sorted(by_epoch_values.items()):
        out = {
            "run_id": run_id,
            "model": model,
            "model_label": DISPLAY[model],
            "layer": layer,
            "head": head,
            "epoch": epoch,
        }
        for metric, values in metrics.items():
            out[metric] = float(np.mean(values))
        epoch_rows.append(out)

    # Per-seed period variance for each head, then summarize across seeds.
    by_seed_head: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in epoch_rows:
        key = (row["run_id"], row["model"], row["layer"], row["head"])
        for metric in ("verb_mass", "object_mass", "context_mass", "entropy"):
            by_seed_head[key][metric].append(row[metric])

    variance_rows = []
    for (run_id, model, layer, head), metrics in sorted(by_seed_head.items()):
        for metric, values in metrics.items():
            variance_rows.append({
                "run_id": run_id,
                "model": model,
                "model_label": DISPLAY[model],
                "layer": layer,
                "head": head,
                "metric": f"{metric}_period_variance",
                "value": float(np.var(values, ddof=1)),
            })

    by_summary: dict[tuple, list[float]] = defaultdict(list)
    for row in variance_rows:
        by_summary[(row["model"], row["layer"], row["head"], row["metric"])].append(row["value"])

    summary_rows = []
    for (model, layer, head, metric), values in sorted(by_summary.items()):
        summary_rows.append({
            "model": model,
            "model_label": DISPLAY[model],
            "layer": layer,
            "head": head,
            "metric": metric,
            **mean_ci(values),
        })
    return epoch_rows, summary_rows


def summarize_ablation(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        by_key[(row["layer"], row["head"], row["metric"])].append(row["value"])

    summary = []
    for (layer, head, metric), values in sorted(by_key.items()):
        s = mean_ci(values)
        if metric == "cka_delta_vs_baseline":
            t_stat, p_val = stats.ttest_1samp(values, popmean=0.0)
            s["t"] = float(t_stat)
            s["p_two_sided"] = float(p_val)
        summary.append({
            "layer": layer,
            "head": head,
            "metric": metric,
            **s,
        })
    return summary


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(attn_summary: list[dict], ablation_summary: list[dict], path: Path) -> None:
    context_var = [
        r for r in attn_summary
        if r["metric"] == "context_mass_period_variance"
    ]
    context_var = sorted(context_var, key=lambda r: r["mean"], reverse=True)

    delta_rows = [
        r for r in ablation_summary
        if r["metric"] == "cka_delta_vs_baseline"
    ]
    delta_rows = sorted(delta_rows, key=lambda r: r["mean"])

    lines = [
        "# Fase 3 — Atenção e ablação de heads",
        "",
        "## Atenção sujeito→contexto",
        "",
        "Heads com maior variância temporal da massa de atenção em verbo+objeto:",
    ]
    for r in context_var[:8]:
        lines.append(
            f"- {r['model_label']} L{r['layer']}H{r['head']}: "
            f"{r['mean']:.6f} [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}]"
        )

    lines.extend([
        "",
        "## Ablation Token-Time",
        "",
        "Delta = CKA(Additive, Token-Time com head ablado) − CKA baseline. "
        "Valores negativos indicam que a ablação fez Token-Time divergir de Additive.",
    ])
    for r in delta_rows:
        lines.append(
            f"- L{r['layer']}H{r['head']}: {r['mean']:+.4f} "
            f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}], "
            f"p={r.get('p_two_sided', float('nan')):.4g}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_ids = load_run_ids(args.raw)
    if args.max_runs is not None:
        run_ids = run_ids[:args.max_runs]

    corpus_rows = load_corpus(args.corpus)
    split_rows = [r for r in corpus_rows if r["split"] == args.split]
    dataset = MLMDataset(split_rows, seed=42)

    attention_rows: list[dict] = []
    ablation_rows: list[dict] = []

    for idx, run_id in enumerate(run_ids, start=1):
        print(f"[{idx}/{len(run_ids)}] {run_id}")
        additive = load_trained_model(run_id, "Additive", device)
        joint = load_trained_model(run_id, "Joint", device)

        reps_a, attn_a = extract_attention_and_hidden(
            additive, dataset, args.batch_size, device
        )
        reps_t, attn_t = extract_attention_and_hidden(
            joint, dataset, args.batch_size, device
        )
        baseline_cka = linear_cka(reps_a["h2"], reps_t["h2"])

        for model_name, rows in (("Additive", attn_a), ("Joint", attn_t)):
            for row in rows:
                row["run_id"] = run_id
                row["model"] = model_name
                row["model_label"] = DISPLAY[model_name]
                attention_rows.append(row)

        ablation_rows.append({
            "run_id": run_id,
            "layer": -1,
            "head": -1,
            "metric": "baseline_cka",
            "value": baseline_cka,
        })
        for layer in range(N_LAYERS):
            for head in range(N_HEADS):
                reps_t_ab, _ = extract_attention_and_hidden(
                    joint, dataset, args.batch_size, device, ablate=(layer, head)
                )
                cka_ab = linear_cka(reps_a["h2"], reps_t_ab["h2"])
                ablation_rows.extend([
                    {
                        "run_id": run_id,
                        "layer": layer,
                        "head": head,
                        "metric": "cka_ablated",
                        "value": cka_ab,
                    },
                    {
                        "run_id": run_id,
                        "layer": layer,
                        "head": head,
                        "metric": "cka_delta_vs_baseline",
                        "value": cka_ab - baseline_cka,
                    },
                ])

    epoch_rows, attn_summary = summarize_attention(attention_rows)
    ablation_summary = summarize_ablation(ablation_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    attention_epoch_path = OUT_DIR / "phase3_attention_by_epoch.csv"
    attention_summary_path = OUT_DIR / "phase3_attention_summary.csv"
    ablation_path = OUT_DIR / "phase3_head_ablation.csv"
    ablation_summary_path = OUT_DIR / "phase3_head_ablation_summary.csv"
    json_path = OUT_DIR / "phase3_attention_mechanism.json"

    write_csv(
        epoch_rows,
        attention_epoch_path,
        ["run_id", "model", "model_label", "layer", "head", "epoch",
         "verb_mass", "object_mass", "context_mass", "entropy"],
    )
    write_csv(
        attn_summary,
        attention_summary_path,
        ["model", "model_label", "layer", "head", "metric", "n", "mean",
         "sd", "ci95_low", "ci95_high"],
    )
    write_csv(
        ablation_rows,
        ablation_path,
        ["run_id", "layer", "head", "metric", "value"],
    )
    write_csv(
        ablation_summary,
        ablation_summary_path,
        ["layer", "head", "metric", "n", "mean", "sd", "ci95_low", "ci95_high",
         "t", "p_two_sided"],
    )
    json_path.write_text(json.dumps({
        "raw_path": str(args.raw),
        "corpus": str(args.corpus),
        "split": args.split,
        "n_examples": len(dataset),
        "n_runs": len(run_ids),
        "attention_summary": attn_summary,
        "ablation_summary": ablation_summary,
    }, indent=2), encoding="utf-8")
    write_markdown(attn_summary, ablation_summary, SUMMARY_PATH)

    print(f"Wrote {attention_epoch_path}")
    print(f"Wrote {attention_summary_path}")
    print(f"Wrote {ablation_path}")
    print(f"Wrote {ablation_summary_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
