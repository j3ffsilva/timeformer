#!/usr/bin/env python3
"""
Phase 0 of the restructured Timeformer evaluation protocol.

Runs an equivalence audit for the context drift score, focusing on the
Additive-vs-Token-Time comparison over the 31 paired seeds.

Outputs:
  outputs/protocol/phase0_equivalence.json
  outputs/protocol/phase0_drift_values.csv
  tmp/protocol_phase0_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import optimize, stats


RAW_DEFAULT = Path("outputs/multiseed/multiseed_raw.json")
OUT_DIR = Path("outputs/protocol")
SUMMARY_PATH = Path("tmp/protocol_phase0_summary.md")

MODEL_LABELS = {
    "Static": "Standard",
    "Additive": "Additive",
    "Joint": "Token-Time",
    "Timeformer": "Memory-Augmented",
}


@dataclass
class TOSTResult:
    delta: float
    paired: bool
    n: int
    diff_mean: float
    se: float
    df: float
    ci90_low: float
    ci90_high: float
    p_lower: float
    p_upper: float
    p_tost: float
    equivalent: bool


@dataclass
class MDEResult:
    paired: bool
    n: int
    alpha: float
    power: float
    cohen_d: float
    raw_units: float
    sd_scale: float


def _valid(x: float | None) -> bool:
    return x is not None and not math.isnan(float(x))


def load_paired_values(raw_path: Path, cls: str) -> list[dict]:
    with raw_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    rows: list[dict] = []
    for item in raw:
        run_id = item["run_id"]
        seed = item.get("seed")
        drift = item.get("drift", {})
        record = {"run_id": run_id, "seed": seed}
        ok = True
        for model in ("Static", "Additive", "Joint", "Timeformer"):
            value = drift.get(model, {}).get(cls, {}).get("delta")
            if not _valid(value):
                ok = False
                break
            record[model] = float(value)
        if ok:
            record["Additive_minus_Joint"] = record["Additive"] - record["Joint"]
            record["Joint_minus_Additive"] = record["Joint"] - record["Additive"]
            rows.append(record)
    return rows


def tost_equivalence(x: np.ndarray, y: np.ndarray, delta: float, paired: bool) -> TOSTResult:
    if paired:
        d = x - y
        diff = float(d.mean())
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        df = len(d) - 1
        n = len(d)
    else:
        diff = float(x.mean() - y.mean())
        nx, ny = len(x), len(y)
        sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
        se = float(math.sqrt(sp2 * (1 / nx + 1 / ny)))
        df = nx + ny - 2
        n = min(nx, ny)

    t_lower = (diff + delta) / se
    t_upper = (diff - delta) / se
    p_lower = float(stats.t.sf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)

    tcrit = float(stats.t.ppf(0.95, df))
    ci90_low = diff - tcrit * se
    ci90_high = diff + tcrit * se

    return TOSTResult(
        delta=delta,
        paired=paired,
        n=n,
        diff_mean=diff,
        se=se,
        df=float(df),
        ci90_low=float(ci90_low),
        ci90_high=float(ci90_high),
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        equivalent=bool(ci90_low > -delta and ci90_high < delta),
    )


def _power_two_sided_t(effect_size: float, n: int, alpha: float, paired: bool) -> float:
    if paired:
        df = n - 1
        ncp = effect_size * math.sqrt(n)
    else:
        df = 2 * n - 2
        ncp = effect_size * math.sqrt(n / 2)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))


def mde(n: int, sd_scale: float, alpha: float = 0.05, power: float = 0.80,
        paired: bool = True) -> MDEResult:
    def objective(d: float) -> float:
        return _power_two_sided_t(d, n=n, alpha=alpha, paired=paired) - power

    cohen_d = float(optimize.brentq(objective, 1e-6, 10.0))
    return MDEResult(
        paired=paired,
        n=n,
        alpha=alpha,
        power=power,
        cohen_d=cohen_d,
        raw_units=float(cohen_d * sd_scale),
        sd_scale=float(sd_scale),
    )


def mean_ci95(values: np.ndarray) -> dict:
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "mean": mean,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "sd": float(values.std(ddof=1)),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id", "seed", "Static", "Additive", "Joint", "Timeformer",
        "Additive_minus_Joint", "Joint_minus_Additive",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t05 = result["tost"]["delta_0.05"]
    t034 = result["tost"]["delta_0.034"]
    mde_p = result["mde"]["paired"]
    summary = result["summary"]

    lines = [
        "# Fase 0 — Auditoria de equivalência do drift score",
        "",
        f"- Fonte: `{result['raw_path']}`",
        f"- Classe avaliada: `{result['class']}`",
        f"- Seeds pareados válidos: {result['n']}",
        f"- Comparação: Additive − Token-Time (`Joint`) em Δ(t9−t0)",
        "",
        "## Médias de Δ por arquitetura",
        "",
    ]
    for model in ("Static", "Additive", "Joint", "Timeformer"):
        s = summary[model]
        lines.append(
            f"- {MODEL_LABELS[model]}: {s['mean']:+.4f} "
            f"[{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}]"
        )

    lines.extend([
        "",
        "## TOST pareado",
        "",
        (
            f"- δ = 0.050: diff={t05['diff_mean']:+.4f}, "
            f"IC90%=[{t05['ci90_low']:+.4f}, {t05['ci90_high']:+.4f}], "
            f"p_TOST={t05['p_tost']:.4g}, "
            f"{'equivalente' if t05['equivalent'] else 'não equivalente'}"
        ),
        (
            f"- δ = 0.034: diff={t034['diff_mean']:+.4f}, "
            f"IC90%=[{t034['ci90_low']:+.4f}, {t034['ci90_high']:+.4f}], "
            f"p_TOST={t034['p_tost']:.4g}, "
            f"{'equivalente' if t034['equivalent'] else 'não equivalente'}"
        ),
        "",
        "## Poder",
        "",
        (
            f"- MDE pareado, 80% poder, α=0.05: d={mde_p['cohen_d']:.3f}, "
            f"{mde_p['raw_units']:.4f} unidades de drift "
            f"(escala = DP das diferenças pareadas)."
        ),
        "",
        "## Leitura curta",
        "",
        result["interpretation"],
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--class", dest="cls", default="drift",
                        choices=("stable", "drift", "bifurc"))
    parser.add_argument("--independent", action="store_true",
                        help="Use independent-samples TOST instead of paired-by-seed.")
    args = parser.parse_args()

    rows = load_paired_values(args.raw, args.cls)
    if not rows:
        raise SystemExit(f"No valid rows found in {args.raw}")

    x = np.array([r["Additive"] for r in rows], dtype=float)
    y = np.array([r["Joint"] for r in rows], dtype=float)
    paired = not args.independent

    tost_results = {
        "delta_0.05": asdict(tost_equivalence(x, y, delta=0.05, paired=paired)),
        "delta_0.034": asdict(tost_equivalence(x, y, delta=0.034, paired=paired)),
    }

    paired_diff_sd = float((x - y).std(ddof=1))
    pooled_sd = float(math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2))
    mde_results = {
        "paired": asdict(mde(len(rows), paired_diff_sd, paired=True)),
        "independent": asdict(mde(len(rows), pooled_sd, paired=False)),
    }

    summary = {model: mean_ci95(np.array([r[model] for r in rows], dtype=float))
               for model in ("Static", "Additive", "Joint", "Timeformer")}

    diff = x.mean() - y.mean()
    if tost_results["delta_0.05"]["equivalent"]:
        interpretation = (
            "Com pareamento por seed, o drift agregado de Additive e Token-Time "
            "é equivalente dentro de δ=0.05. Isto sustenta a leitura de que a "
            "forma de condicionamento não muda materialmente o D2 agregado, "
            "mesmo que os mecanismos de entrada sejam distintos."
        )
    else:
        interpretation = (
            "O teste não permite afirmar equivalência dentro de δ=0.05. "
            "Neste caso, o empate visual/por IC deve ser tratado como "
            "evidência inconclusiva, e as fases camada-a-camada ficam ainda "
            "mais importantes."
        )

    result = {
        "raw_path": str(args.raw),
        "class": args.cls,
        "n": len(rows),
        "paired": paired,
        "comparison": "Additive - Joint(Token-Time)",
        "mean_difference": float(diff),
        "summary": summary,
        "tost": tost_results,
        "mde": mde_results,
        "interpretation": interpretation,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "phase0_equivalence.json"
    csv_path = OUT_DIR / "phase0_drift_values.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(rows, csv_path)
    write_summary(result, SUMMARY_PATH)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Additive - Token-Time diff: {diff:+.4f}")
    for name, res in tost_results.items():
        print(
            f"{name}: IC90%=[{res['ci90_low']:+.4f}, {res['ci90_high']:+.4f}], "
            f"p_TOST={res['p_tost']:.4g}, equivalent={res['equivalent']}"
        )
    print(
        "MDE paired: "
        f"d={mde_results['paired']['cohen_d']:.3f}, "
        f"raw={mde_results['paired']['raw_units']:.4f}"
    )


if __name__ == "__main__":
    main()
