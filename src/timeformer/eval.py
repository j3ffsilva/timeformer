"""
Pipeline de avaliação da Fase B refatorado em classe Evaluator.

Evaluator       — avalia um modelo em todos os splits, treina probe e retorna métricas
compare_models  — calcula deltas da cadeia de ablação
save_results    — persiste JSON + CSV + ablation_table
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .dataset import (
    load_corpus, MLMDataset, ContrastiveDataset,
    make_continuation_split, make_interpolation_split,
)
from .memory import PrototypeMemory, make_shuffled, make_nohistory
from .nomenclature import (
    ABLATION_DISPLAY,
    ABLATION_LABELS,
    LEGACY_ABLATION_ALIASES,
    model_label,
)
from .probe import (
    LinearProbe, extract_reps,
    evaluate_contrastive, precision_at_k, clustering_metrics,
)

SPLIT_NAMES = ("test", "hard_verb", "hard_both", "ambiguous_test",
               "continuation", "interpolation")


class Evaluator:
    """
    Avalia um modelo treinado em todos os splits.

    Mantém estado (corpus, caminhos, device) para reaproveitar entre chamadas.

    Uso:
        ev = Evaluator(corpus_path, ambiguous_path, contrastive_path, device="cpu")
        results = ev.evaluate(model, memory=memory)
    """

    def __init__(
        self,
        corpus_path: str | Path,
        ambiguous_path: str | Path,
        contrastive_path: str | Path,
        device: str | torch.device = "cpu",
        batch_size: int = 128,
    ) -> None:
        self.corpus_path      = Path(corpus_path)
        self.ambiguous_path   = Path(ambiguous_path)
        self.contrastive_path = Path(contrastive_path)
        self.device           = torch.device(device)
        self.batch_size       = batch_size

        rows = load_corpus(self.corpus_path)
        self._splits: dict[str, list[dict]] = self._build_splits(rows)

    # ── Construção dos splits ──────────────────────────────────────────────

    def _build_splits(self, rows: list[dict]) -> dict[str, list[dict]]:
        splits: dict[str, list[dict]] = {}
        for sp in ("test", "hard_verb", "hard_both"):
            splits[sp] = [r for r in rows if r["split"] == sp]
        splits["ambiguous_test"] = load_corpus(self.ambiguous_path)
        _, splits["continuation"]  = make_continuation_split(rows)
        _, splits["interpolation"] = make_interpolation_split(rows)
        return splits

    # ── Avaliação principal ────────────────────────────────────────────────

    def evaluate(
        self,
        model: nn.Module,
        memory: PrototypeMemory | None = None,
    ) -> dict:
        """
        Avalia modelo em todos os splits.

        Probe treinada em h(sujeito) do split 'test', avaliada em cada split.
        Para Timeformer: requer PrototypeMemory já atualizada após treino.

        Retorna dict aninhado: {split: {métrica: valor}}
        """
        # Extrai representações do split test para treinar a probe
        test_ds   = MLMDataset(self._splits["test"])
        test_reps = extract_reps(model, test_ds, memory, self.batch_size, self.device)

        probe_subj = LinearProbe().fit(test_reps["h_subj"], test_reps["true_context"])
        probe_sent = LinearProbe().fit(test_reps["h_cls"],  test_reps["true_context"])

        results: dict[str, dict] = {}

        for sp_name, sp_rows in self._splits.items():
            results[sp_name] = self._eval_split(
                model, memory, sp_rows, sp_name, probe_subj, probe_sent
            )

        # Avaliação contrastiva
        results["contrastive"] = self._eval_contrastive(model, memory)

        # Controles Timeformer (apenas se modelo é Timeformer e memória disponível)
        if type(model).__name__ == "Timeformer" and memory is not None:
            results.update(self._eval_b3_controls(model, memory, test_ds))

        return results

    def _eval_split(
        self,
        model: nn.Module,
        memory: PrototypeMemory | None,
        sp_rows: list[dict],
        sp_name: str,
        probe_subj: LinearProbe,
        probe_sent: LinearProbe,
    ) -> dict:
        if not sp_rows:
            return {"skipped": True, "reason": "split vazio"}

        sp_ds   = MLMDataset(sp_rows)
        sp_reps = extract_reps(model, sp_ds, memory, self.batch_size, self.device)

        if len(np.unique(sp_reps["true_context"])) < 2:
            return {"skipped": True, "reason": "apenas uma classe no split"}

        rep_metrics = {
            "precision_at_5": precision_at_k(sp_reps["h_subj"], sp_reps["true_context"], k=5),
            **{f"cluster_{k}": v for k, v in
               clustering_metrics(sp_reps["h_subj"], sp_reps["true_context"]).items()},
        }

        return {
            "n":          len(sp_rows),
            "probe_subj": probe_subj.evaluate(sp_reps["h_subj"], sp_reps["true_context"]),
            "probe_sent": probe_sent.evaluate(sp_reps["h_cls"],  sp_reps["true_context"]),
            **rep_metrics,
        }

    def _eval_contrastive(
        self,
        model: nn.Module,
        memory: PrototypeMemory | None,
    ) -> dict:
        if not self.contrastive_path.exists():
            return {"skipped": True, "reason": "contrastive_set.tsv não encontrado"}
        cont_ds = ContrastiveDataset(self.contrastive_path)
        return evaluate_contrastive(model, cont_ds, memory, self.batch_size, self.device)

    def _eval_b3_controls(
        self,
        model: nn.Module,
        memory: PrototypeMemory,
        test_ds: MLMDataset,
    ) -> dict:
        controls = {}
        cont_rows = self._splits.get("continuation") or self._splits["test"]
        cont_ds   = MLMDataset(cont_rows)

        for ctrl_name, ctrl_mem in [
            ("timeformer_shuffled_subject", make_shuffled(memory, mode="subject")),
            ("timeformer_nohistory",        make_nohistory(
                memory.n_subjects, memory.n_epochs, memory.d_model, self.device
            )),
        ]:
            ctrl_test_reps = extract_reps(
                model, test_ds, ctrl_mem, self.batch_size, self.device
            )
            probe_ctrl = LinearProbe().fit(
                ctrl_test_reps["h_subj"], ctrl_test_reps["true_context"]
            )
            ctrl_reps = extract_reps(
                model, cont_ds, ctrl_mem, self.batch_size, self.device
            )
            controls[ctrl_name] = probe_ctrl.evaluate(
                ctrl_reps["h_subj"], ctrl_reps["true_context"]
            )

        return controls


# ── Análise comparativa ────────────────────────────────────────────────────────

def compare_models(
    results_by_model: dict[str, dict],
    primary_split: str = "ambiguous_test",
    b3_split: str = "continuation",
) -> dict:
    """
    Calcula deltas da cadeia de ablação:
      delta_time_conditioning = Additive − Static  (ambiguous_test)
      delta_token_time_interaction = Joint − Additive (ambiguous_test)
      delta_memory = Timeformer − Joint (continuation)
      delta_spurious_memory = Timeformer-shuffled − Joint (continuation)
    """
    def _acc(res: dict, split: str) -> float:
        try:
            return res[split]["probe_subj"]["accuracy"]
        except (KeyError, TypeError):
            return float("nan")

    deltas: dict[str, dict] = {}

    if "Timeformer" in results_by_model:
        b3_res      = results_by_model["Timeformer"]
        b3_shuffled = b3_res.get("timeformer_shuffled_subject", {})
        try:
            shuffled_acc = b3_shuffled.get("accuracy", float("nan"))
        except AttributeError:
            shuffled_acc = float("nan")
        delta_key = "delta_spurious_memory"
        deltas[delta_key] = {
            "split": b3_split,
            "label": ABLATION_LABELS[delta_key],
            "display": ABLATION_DISPLAY[delta_key],
            "legacy_label": LEGACY_ABLATION_ALIASES[delta_key],
            "base_model": "Joint",
            "base_model_label": model_label("Joint"),
            "new_model": "Timeformer-shuffled",
            "new_model_label": "Shuffled-memory Timeformer",
            "Timeformer_shuffled_accuracy": shuffled_acc,
        }

    pairs = [
        ("delta_time_conditioning", "Static",  "Additive", primary_split),
        ("delta_token_time_interaction",  "Additive", "Joint", primary_split),
        ("delta_memory",        "Joint", "Timeformer",  b3_split),
    ]
    for label, m_base, m_new, split in pairs:
        if m_base in results_by_model and m_new in results_by_model:
            base_acc = _acc(results_by_model[m_base], split)
            new_acc  = _acc(results_by_model[m_new],  split)
            deltas[label] = {
                "split":              split,
                "label":              ABLATION_LABELS[label],
                "display":            ABLATION_DISPLAY[label],
                "legacy_label":       LEGACY_ABLATION_ALIASES[label],
                "base_model":         m_base,
                "base_model_label":   model_label(m_base),
                "new_model":          m_new,
                "new_model_label":    model_label(m_new),
                f"{m_base}_accuracy": base_acc,
                f"{m_new}_accuracy":  new_acc,
                "delta": (
                    round(new_acc - base_acc, 4)
                    if not (np.isnan(base_acc) or np.isnan(new_acc))
                    else float("nan")
                ),
            }

    return {
        "ablation_deltas": deltas,
        "models": list(results_by_model.keys()),
        "model_labels": {
            model_id: model_label(model_id) for model_id in results_by_model
        },
    }


# ── Persistência de resultados ─────────────────────────────────────────────────

def save_results(
    results_by_model: dict[str, dict],
    output_dir: str | Path,
) -> None:
    """Salva resultados em JSON e tabela CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "results_full.json").write_text(
        json.dumps(results_by_model, indent=2, default=str)
    )

    rows_csv = []
    for model_name, model_res in results_by_model.items():
        for split_name in SPLIT_NAMES + ("contrastive",):
            split_res = model_res.get(split_name, {})
            if split_res.get("skipped"):
                continue
            row = {
                "model": model_name,
                "model_label": model_label(model_name),
                "split": split_name,
            }
            for probe_key in ("probe_subj", "probe_sent"):
                for metric in ("accuracy", "f1", "auroc"):
                    val = split_res.get(probe_key, {}).get(metric, "")
                    row[f"{probe_key}_{metric}"] = (
                        round(val, 4) if isinstance(val, float) else val
                    )
            for metric in ("precision_at_5", "cluster_ari", "cluster_nmi"):
                val = split_res.get(metric, "")
                row[metric] = round(val, 4) if isinstance(val, float) else val
            if split_name == "contrastive":
                row["sign_flip_rate"] = round(
                    split_res.get("sign_flip_rate", float("nan")), 4
                )
            rows_csv.append(row)

    if rows_csv:
        fieldnames = list(dict.fromkeys(k for row in rows_csv for k in row))
        with open(output_dir / "results_table.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore",
                                    restval="")
            writer.writeheader()
            writer.writerows(rows_csv)

    ablation = compare_models(results_by_model)
    (output_dir / "ablation_table.json").write_text(
        json.dumps(ablation, indent=2, default=str)
    )

    print(f"Resultados salvos em {output_dir}/")
    print(f"  results_full.json, results_table.csv, ablation_table.json")
