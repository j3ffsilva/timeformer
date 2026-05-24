"""
Pipeline de avaliação completa da Fase B.

evaluate_model  — roda todos os splits para um modelo treinado
compare_models  — gera tabela de ablação com nomes públicos e aliases legados
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .dataset import (
    load_corpus, MLMDataset, ContrastiveDataset,
    make_continuation_split, make_interpolation_split,
)
from .memory import PrototypeMemory, make_shuffled, make_nohistory
from src.timeformer.nomenclature import (
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


def evaluate_model(
    model: nn.Module,
    corpus_path: str | Path,
    ambiguous_path: str | Path,
    contrastive_path: str | Path,
    memory: PrototypeMemory | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 128,
) -> dict:
    """
    Avalia um modelo em todos os splits.

    Probe treinada em 'test' (representações extraídas do split de avaliação padrão),
    avaliada em cada split separadamente.

    Para B3: requer PrototypeMemory já atualizada (treino concluído).
    Para B3-shuffled/nohistory: passar a PrototypeMemory correspondente.

    Retorna dict aninhado: {split: {metric: valor}}
    """
    device = torch.device(device)
    rows = load_corpus(corpus_path)

    # ── Datasets por split ──
    splits_map: dict[str, list[dict]] = {}
    for sp in ("test", "hard_verb", "hard_both"):
        splits_map[sp] = [r for r in rows if r["split"] == sp]
    splits_map["ambiguous_test"] = load_corpus(ambiguous_path)

    train_rows = [r for r in rows if r["split"] == "train"]
    _, cont_rows   = make_continuation_split(rows)
    _, interp_rows = make_interpolation_split(rows)
    splits_map["continuation"]  = cont_rows
    splits_map["interpolation"] = interp_rows

    # ── Extrai representações do split 'test' para treinar a probe ──
    test_ds   = MLMDataset(splits_map["test"])
    test_reps = extract_reps(model, test_ds, memory, batch_size, device)

    # Treina probe em h(sujeito) do split 'test'
    probe_subj = LinearProbe().fit(test_reps["h_subj"], test_reps["true_context"])
    probe_sent = LinearProbe().fit(test_reps["h_cls"],  test_reps["true_context"])

    results: dict[str, dict] = {}

    # ── Avalia em cada split ──
    for sp_name, sp_rows in splits_map.items():
        if not sp_rows:
            results[sp_name] = {"skipped": True, "reason": "split vazio"}
            continue

        sp_ds   = MLMDataset(sp_rows)
        sp_reps = extract_reps(model, sp_ds, memory, batch_size, device)

        if len(np.unique(sp_reps["true_context"])) < 2:
            results[sp_name] = {"skipped": True, "reason": "apenas uma classe no split"}
            continue

        probe_subj_metrics = probe_subj.evaluate(sp_reps["h_subj"], sp_reps["true_context"])
        probe_sent_metrics = probe_sent.evaluate(sp_reps["h_cls"],  sp_reps["true_context"])

        rep_metrics = {
            "precision_at_5":  precision_at_k(sp_reps["h_subj"], sp_reps["true_context"], k=5),
            **{f"cluster_{k}": v for k, v in
               clustering_metrics(sp_reps["h_subj"], sp_reps["true_context"]).items()},
        }

        results[sp_name] = {
            "n":          len(sp_rows),
            "probe_subj": probe_subj_metrics,
            "probe_sent": probe_sent_metrics,
            **rep_metrics,
        }

    # ── Avaliação contrastiva ──
    if Path(contrastive_path).exists():
        cont_ds = ContrastiveDataset(contrastive_path)
        results["contrastive"] = evaluate_contrastive(
            model, cont_ds, memory, batch_size, device
        )
    else:
        results["contrastive"] = {"skipped": True, "reason": "contrastive_set.tsv não encontrado"}

    # ── Controles B3-shuffled e B3-nohistory (apenas se modelo é B3) ──
    if type(model).__name__ == "B3" and memory is not None:
        for ctrl_name, ctrl_mem in [
            ("b3_shuffled_subject", make_shuffled(memory, mode="subject")),
            ("b3_nohistory",        make_nohistory(memory.n_subjects,
                                                   memory.n_epochs,
                                                   memory.d_model,
                                                   device)),
        ]:
            test_reps_ctrl = extract_reps(model, test_ds, ctrl_mem, batch_size, device)
            probe_ctrl = LinearProbe().fit(
                test_reps_ctrl["h_subj"], test_reps_ctrl["true_context"]
            )
            sp_reps_ctrl = extract_reps(
                model, MLMDataset(splits_map.get("continuation", splits_map["test"])),
                ctrl_mem, batch_size, device,
            )
            results[ctrl_name] = probe_ctrl.evaluate(
                sp_reps_ctrl["h_subj"], sp_reps_ctrl["true_context"]
            )

    return results


def compare_models(
    results_by_model: dict[str, dict],
    primary_split: str = "ambiguous_test",
    b3_split: str = "continuation",
) -> dict:
    """
    Calcula deltas da cadeia de ablação:
      delta_time_conditioning = B2a − B1  (ambiguous_test)
      delta_token_time_interaction = B2b − B2a (ambiguous_test)
      delta_memory = B3 − B2b (continuation)
      delta_spurious_memory = B3-shuffled − B2b (continuation)

    Retorna dict com os deltas por métrica.
    """
    def _acc(res: dict, split: str) -> float:
        try:
            return res[split]["probe_subj"]["accuracy"]
        except (KeyError, TypeError):
            return float("nan")

    models = list(results_by_model.keys())
    deltas: dict[str, dict] = {}

    pairs = [
        ("delta_time_conditioning", "B1",  "B2a", primary_split),
        ("delta_token_time_interaction",  "B2a", "B2b", primary_split),
        ("delta_memory",        "B2b", "B3",  b3_split),
    ]
    if "B3" in results_by_model:
        b3_res = results_by_model["B3"]
        b3_shuffled = b3_res.get("b3_shuffled_subject", {})
        b3_nohist   = b3_res.get("b3_nohistory", {})
        b3_acc  = _acc(results_by_model, b3_split) if "B3" in results_by_model else float("nan")
        # Controle de memória espúria: accuracy de B3-shuffled em continuation.
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
            "base_model": "B2b",
            "base_model_label": model_label("B2b"),
            "new_model": "B3-shuffled",
            "new_model_label": "Shuffled-memory Timeformer",
            "B3_shuffled_accuracy": shuffled_acc,
        }

    for label, m_base, m_new, split in pairs:
        if m_base in results_by_model and m_new in results_by_model:
            base_acc = _acc(results_by_model[m_base], split)
            new_acc  = _acc(results_by_model[m_new],  split)
            deltas[label] = {
                "split":    split,
                "label":    ABLATION_LABELS[label],
                "display":  ABLATION_DISPLAY[label],
                "legacy_label": LEGACY_ABLATION_ALIASES[label],
                "base_model": m_base,
                "base_model_label": model_label(m_base),
                "new_model": m_new,
                "new_model_label": model_label(m_new),
                f"{m_base}_accuracy": base_acc,
                f"{m_new}_accuracy":  new_acc,
                "delta":    round(new_acc - base_acc, 4) if not (
                    np.isnan(base_acc) or np.isnan(new_acc)) else float("nan"),
            }

    return {
        "ablation_deltas": deltas,
        "models": models,
        "model_labels": {model_id: model_label(model_id) for model_id in models},
    }


def save_results(
    results_by_model: dict[str, dict],
    output_dir: str | Path,
) -> None:
    """Salva resultados em JSON e tabela CSV para o paper."""
    import csv
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON completo
    (output_dir / "results_full.json").write_text(
        json.dumps(results_by_model, indent=2, default=str)
    )

    # CSV: uma linha por (modelo, split), colunas = métricas principais
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
        # Coleta todos os campos presentes em qualquer linha
        fieldnames = list(dict.fromkeys(k for row in rows_csv for k in row))
        with open(output_dir / "results_table.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore",
                                    restval="")
            writer.writeheader()
            writer.writerows(rows_csv)

    # Tabela de ablação
    ablation = compare_models(results_by_model)
    (output_dir / "ablation_table.json").write_text(
        json.dumps(ablation, indent=2, default=str)
    )

    print(f"Resultados salvos em {output_dir}/")
    print(f"  results_full.json, results_table.csv, ablation_table.json")
