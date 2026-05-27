"""
Datasets para a Fase B do Timeformer.

MLMDataset        — uma sentença por item; para Static, Additive, Joint.
TimeformerDataset — item = (sentença atual @t_k, índices de épocas históricas t_0..t_{k-1});
               a memória de protótipos é injetada externamente pelo trainer.
Funções auxiliares:
  make_continuation_split  — t8-t9 held-out (mesmos sujeitos)
  make_interpolation_split — t5 held-out (mesmos sujeitos)
  build_vocab              — vocabulário completo de 46 tokens + especiais
"""

from __future__ import annotations

import random
from pathlib import Path
from collections import defaultdict

import torch
from torch import Tensor
from torch.utils.data import Dataset

# ─── Vocabulário ──────────────────────────────────────────────────────────────

SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
SUBJECTS  = [f"S{i}" for i in range(1, 31)]   # S1-S30
VERBS     = [f"V{i}" for i in range(1, 9)]    # V1-V8
OBJECTS   = [f"O{i}" for i in range(1, 9)]    # O1-O8
VOCAB_TOKENS = SPECIAL_TOKENS + SUBJECTS + VERBS + OBJECTS  # 5 + 30 + 8 + 8 = 51

PAD_ID  = 0
CLS_ID  = 1
SEP_ID  = 2
MASK_ID = 3
UNK_ID  = 4

TOKEN2ID: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB_TOKENS)}
ID2TOKEN: dict[int, str] = {i: tok for tok, i in TOKEN2ID.items()}

VOCAB_SIZE = len(VOCAB_TOKENS)

# Posições na sequência [CLS, S, V, O, SEP]
POS_CLS     = 0
POS_SUBJECT = 1
POS_VERB    = 2
POS_OBJECT  = 3
POS_SEP     = 4
SEQ_LEN     = 5

# Índices de época
EPOCH2IDX: dict[str, int] = {f"t{i}": i for i in range(10)}
IDX2EPOCH: dict[int, str] = {i: f"t{i}" for i in range(10)}
N_EPOCHS = 10


# ─── Carregamento do corpus ───────────────────────────────────────────────────

def context_to_id(label: str) -> int:
    """Map legacy A/B and current N1/N2 context labels to integer ids."""
    if label in {"N1", "A"}:
        return 0
    if label in {"N2", "B"}:
        return 1
    raise ValueError(f"Unknown true_context label: {label!r}")

def load_corpus(path: str | Path) -> list[dict]:
    """
    Lê corpus.tsv e retorna lista de dicts com keys:
      epoch, sentence, true_context, split, subject, verb, obj
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = __import__("csv").DictReader(f, delimiter="\t")
        for row in reader:
            tokens = row["sentence"].split()
            rows.append({
                "epoch":        row["epoch"],
                "epoch_idx":    EPOCH2IDX[row["epoch"]],
                "sentence":     row["sentence"],
                "true_context": row["true_context"],
                "split":        row["split"],
                "subject":      tokens[0],
                "verb":         tokens[1],
                "obj":          tokens[2],
                "subject_id":   TOKEN2ID[tokens[0]] - len(SPECIAL_TOKENS),  # 0-indexed
            })
    return rows


def make_continuation_split(
    rows: list[dict],
    held_out_epochs: tuple[int, ...] = (8, 9),
) -> tuple[list[dict], list[dict]]:
    """
    Separa train (t0-t7) e continuation_test (t8-t9).

    Restrição: continuation_test usa apenas sujeitos conhecidos (vistos em t0-t7).
    Filtra apenas linhas com split='train' para evitar contaminação com os
    splits de avaliação originais.
    """
    train_rows = [r for r in rows if r["split"] == "train"
                  and r["epoch_idx"] not in held_out_epochs]
    cont_rows  = [r for r in rows if r["split"] == "train"
                  and r["epoch_idx"] in held_out_epochs]
    return train_rows, cont_rows


def make_interpolation_split(
    rows: list[dict],
    held_out_epoch: int = 5,
) -> tuple[list[dict], list[dict]]:
    """
    Separa train (t0-t4, t6-t9) e interpolation_test (t5).
    Filtra apenas split='train' por segurança.
    """
    train_rows = [r for r in rows if r["split"] == "train"
                  and r["epoch_idx"] != held_out_epoch]
    interp_rows = [r for r in rows if r["split"] == "train"
                   and r["epoch_idx"] == held_out_epoch]
    return train_rows, interp_rows


# ─── Mascaramento ─────────────────────────────────────────────────────────────

def _mask_sentence(
    subject_id: int,
    verb_id: int,
    obj_id: int,
    rng: random.Random,
    mask_subject: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Monta [CLS, S, V, O, SEP] e mascara exatamente 1 token (verbo ou objeto).
    Sujeito sempre visível por default.

    Retorna:
      input_ids : (SEQ_LEN,) — sequência com [MASK] na posição mascarada
      labels    : (SEQ_LEN,) — token original na posição mascarada, -100 nas demais
                               (-100 é ignorado pelo CrossEntropyLoss)
    """
    ids = [CLS_ID, subject_id, verb_id, obj_id, SEP_ID]
    labels = [-100] * SEQ_LEN

    # Posições elegíveis para mascaramento: verbo (2) e objeto (3)
    # Sujeito (1) permanece visível — é o token cuja trajetória queremos avaliar
    eligible = [POS_VERB, POS_OBJECT]
    if mask_subject:
        eligible.append(POS_SUBJECT)

    mask_pos = rng.choice(eligible)
    labels[mask_pos] = ids[mask_pos]
    ids[mask_pos] = MASK_ID

    return (
        torch.tensor(ids,    dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# ─── MLMDataset ───────────────────────────────────────────────────────────────

class MLMDataset(Dataset):
    """
    Dataset MLM padrão para Static, Additive, Joint.

    Cada item é uma sentença SVO com 1 token mascarado (verbo ou objeto).
    O sujeito permanece sempre visível.

    Args:
        rows:     lista de dicts do corpus (output de load_corpus)
        splits:   subconjunto de splits a incluir, ex. ["train"]
        seed:     semente para mascaramento determinístico
    """

    def __init__(
        self,
        rows: list[dict],
        splits: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        if splits is not None:
            rows = [r for r in rows if r["split"] in splits]
        self.rows = rows
        self._rng = random.Random(seed)
        # Pré-computa os itens para garantir mascaramento determinístico
        self._items = [self._make_item(r) for r in self.rows]

    def _make_item(self, row: dict) -> dict:
        subject_id = TOKEN2ID[row["subject"]]
        verb_id    = TOKEN2ID[row["verb"]]
        obj_id     = TOKEN2ID[row["obj"]]
        input_ids, labels = _mask_sentence(subject_id, verb_id, obj_id, self._rng)
        return {
            "input_ids":    input_ids,
            "labels":       labels,
            "epoch_idx":    torch.tensor(row["epoch_idx"], dtype=torch.long),
            "subject_idx":  torch.tensor(TOKEN2ID[row["subject"]] - len(SPECIAL_TOKENS),
                                         dtype=torch.long),
            "true_context": context_to_id(row["true_context"]),
        }

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


# ─── TimeformerDataset ────────────────────────────────────────────────────────

class TimeformerDataset(Dataset):
    """
    Dataset para Timeformer.

    Cada item é a sentença atual @t_k mais os índices das épocas históricas
    disponíveis [0..k-1]. A memória de protótipos em si (vetores h(S,t)) é
    injetada externamente pelo trainer via PrototypeMemory — o dataset não
    acessa representações do modelo.

    Organização: agrupado por sujeito × sequência de épocas para que o
    trainer possa atualizar protótipos de forma coerente.

    Args:
        rows:         lista de dicts do corpus (split='train' recomendado)
        seed:         semente para mascaramento
        max_history:  número máximo de épocas históricas a incluir (None = todas)
    """

    def __init__(
        self,
        rows: list[dict],
        seed: int = 42,
        max_history: int | None = None,
    ) -> None:
        self.rows = rows
        self.max_history = max_history
        self._rng = random.Random(seed)
        self._items = [self._make_item(r) for r in self.rows]

    def _make_item(self, row: dict) -> dict:
        subject_id = TOKEN2ID[row["subject"]]
        verb_id    = TOKEN2ID[row["verb"]]
        obj_id     = TOKEN2ID[row["obj"]]
        input_ids, labels = _mask_sentence(subject_id, verb_id, obj_id, self._rng)

        k = row["epoch_idx"]
        history_epochs = list(range(k))  # épocas disponíveis: t_0..t_{k-1}
        if self.max_history is not None:
            # Janela das épocas mais recentes se max_history for especificado
            history_epochs = history_epochs[-self.max_history:]

        return {
            "input_ids":       input_ids,
            "labels":          labels,
            "epoch_idx":       torch.tensor(k, dtype=torch.long),
            "subject_idx":     torch.tensor(TOKEN2ID[row["subject"]] - len(SPECIAL_TOKENS),
                                            dtype=torch.long),
            "history_epochs":  history_epochs,   # lista de ints; collate_fn lida com padding
            "true_context":    context_to_id(row["true_context"]),
        }

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


def timeformer_collate_fn(batch: list[dict]) -> dict:
    """
    Collate para TimeformerDataset.

    history_epochs tem comprimento variável por item (t0 tem histórico vazio,
    t9 tem 9 épocas). Padeia com -1 e retorna máscara de validade.
    """
    max_hist = max(len(item["history_epochs"]) for item in batch)

    input_ids   = torch.stack([item["input_ids"]   for item in batch])
    labels      = torch.stack([item["labels"]      for item in batch])
    epoch_idx   = torch.stack([item["epoch_idx"]   for item in batch])
    subject_idx = torch.stack([item["subject_idx"] for item in batch])
    true_context = torch.tensor([item["true_context"] for item in batch], dtype=torch.long)

    # history_epochs: (batch, max_hist), preenchido com -1
    hist_tensor = torch.full((len(batch), max_hist), fill_value=-1, dtype=torch.long)
    hist_mask   = torch.zeros(len(batch), max_hist, dtype=torch.bool)
    for i, item in enumerate(batch):
        h = item["history_epochs"]
        if h:
            hist_tensor[i, :len(h)] = torch.tensor(h, dtype=torch.long)
            hist_mask[i, :len(h)]   = True

    return {
        "input_ids":      input_ids,
        "labels":         labels,
        "epoch_idx":      epoch_idx,
        "subject_idx":    subject_idx,
        "history_epochs": hist_tensor,   # (batch, max_hist)
        "history_mask":   hist_mask,     # (batch, max_hist) — True onde válido
        "true_context":   true_context,
    }


# ─── Dataset contrastivo (avaliação) ─────────────────────────────────────────

class ContrastiveDataset(Dataset):
    """
    Dataset para avaliação contrastiva.

    Carrega data/contrastive_set.tsv (gerado por scripts/generate_contrastive.py).
    Cada item é uma sentença com verbo mascarado; pair_id agrupa os dois itens
    de cada par (mesmo sujeito/objeto, épocas diferentes).
    """

    def __init__(self, path: str | Path) -> None:
        import csv as _csv
        self._items: list[dict] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f, delimiter="\t")
            for row in reader:
                subject_id = TOKEN2ID[row["subject"]]
                obj_id     = TOKEN2ID[row["obj"]]
                epoch_idx  = int(row["epoch_idx"])
                # Sentença sempre com verbo mascarado
                input_ids = torch.tensor(
                    [CLS_ID, subject_id, MASK_ID, obj_id, SEP_ID], dtype=torch.long
                )
                self._items.append({
                    "input_ids":    input_ids,
                    "epoch_idx":    torch.tensor(epoch_idx, dtype=torch.long),
                    "subject_idx":  torch.tensor(subject_id - len(SPECIAL_TOKENS),
                                                 dtype=torch.long),
                    "true_context": context_to_id(row["true_context"]),
                    "pair_id":      int(row["pair_id"]),
                })

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]
