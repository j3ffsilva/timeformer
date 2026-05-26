"""
Pipeline de avaliação de representações para a Fase B.

LinearProbe        — classifica true_context ∈ {A, B} a partir de h(sujeito) ou h([CLS])
ContrastiveEval    — avalia inversão de distribuição em pares (mesma superfície, épocas distintas)
precision_at_k     — qualidade de vizinhança semântica
clustering_metrics — ARI e NMI dos clusters por true_context
extract_reps       — extrai representações de todos os exemplos de um split
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .dataset import POS_SUBJECT, POS_CLS, VOCAB_SIZE
from .memory import PrototypeMemory


# ─── Extração de representações ───────────────────────────────────────────────

@torch.no_grad()
def extract_reps(
    model: nn.Module,
    dataset: Dataset,
    memory: PrototypeMemory | None = None,
    batch_size: int = 128,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """
    Extrai h(sujeito), h([CLS]), logits e labels de true_context para todos
    os exemplos do dataset.

    Para Timeformer, injeta memória causal via PrototypeMemory.get().
    A memória deve ter sido atualizada antes desta chamada.

    Retorna dict com:
      h_subj:      (N, d_model)
      h_cls:       (N, d_model)
      true_context:(N,)  int  0=A, 1=B
      epoch_idx:   (N,)  int
      subject_idx: (N,)  int
      logits:      (N, seq, vocab_size)
    """
    device = torch.device(device)
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model_name = type(model).__name__

    all_h_subj, all_h_cls, all_logits = [], [], []
    all_ctx, all_epoch, all_subj = [], [], []

    for batch in loader:
        input_ids   = batch["input_ids"].to(device)
        epoch_idx   = batch["epoch_idx"].to(device)
        subject_idx = batch["subject_idx"].to(device)

        if model_name == "Static":
            out = model(input_ids)
        elif model_name in ("Additive", "Joint"):
            out = model(input_ids, epoch_idx)
        else:  # Timeformer
            if memory is not None:
                mem_list, mask_list = [], []
                for i in range(input_ids.size(0)):
                    k = epoch_idx[i].item()
                    s = subject_idx[i:i+1]
                    m_i, mk_i = memory.get(s, epoch_k=k)
                    mem_list.append(m_i)
                    mask_list.append(mk_i)
                max_hist = max(m.size(1) for m in mem_list)
                d = memory.d_model
                mem_b  = torch.zeros(len(mem_list), max_hist, d, device=device)
                mask_b = torch.zeros(len(mem_list), max_hist, dtype=torch.bool, device=device)
                for i, (m, mk) in enumerate(zip(mem_list, mask_list)):
                    h = m.size(1)
                    if h > 0:
                        mem_b[i, :h, :]  = m.to(device)
                        mask_b[i, :h]    = mk.to(device)
                out = model(input_ids, epoch_idx, memory=mem_b, memory_mask=mask_b)
            else:
                out = model(input_ids, epoch_idx, memory=None)

        all_h_subj.append(out["h_subj"].cpu().numpy())
        all_h_cls.append(out["h_cls"].cpu().numpy())
        all_logits.append(out["logits"].cpu().numpy())
        all_ctx.append(batch["true_context"].numpy())
        all_epoch.append(epoch_idx.cpu().numpy())
        all_subj.append(subject_idx.cpu().numpy())

    return {
        "h_subj":       np.concatenate(all_h_subj,  axis=0),
        "h_cls":        np.concatenate(all_h_cls,   axis=0),
        "logits":       np.concatenate(all_logits,  axis=0),
        "true_context": np.concatenate(all_ctx,     axis=0),
        "epoch_idx":    np.concatenate(all_epoch,   axis=0),
        "subject_idx":  np.concatenate(all_subj,    axis=0),
    }


# ─── Linear Probe ─────────────────────────────────────────────────────────────

class LinearProbe:
    """
    Probe linear (regressão logística) sobre representações de sujeito ou [CLS].

    Treina num split e avalia noutro — não usa true_context durante treino do modelo.
    """

    def __init__(self, max_iter: int = 1000, C: float = 1.0) -> None:
        self._clf = LogisticRegression(
            max_iter=max_iter, C=C, solver="lbfgs", random_state=42,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        assert self._fitted, "Chame fit() antes de evaluate()"
        if len(np.unique(y)) < 2:
            return {"accuracy": float("nan"), "f1": float("nan"), "auroc": float("nan")}
        pred  = self._clf.predict(X)
        prob  = self._clf.predict_proba(X)[:, 1]
        return {
            "accuracy": float((pred == y).mean()),
            "f1":       float(f1_score(y, pred, zero_division=0)),
            "auroc":    float(roc_auc_score(y, prob)),
        }


# ─── Avaliação contrastiva ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_contrastive(
    model: nn.Module,
    contrastive_dataset: Dataset,
    memory: PrototypeMemory | None = None,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """
    Avalia pares do conjunto contrastivo.

    Para cada par (S, O, t_early, t_late):
      sign_flip: P(verbo_A | S, O, t_early) > P(verbo_B | S, O, t_early)
                 E P(verbo_B | S, O, t_late) > P(verbo_A | S, O, t_late)

    Métrica principal: fração de pares com sign_flip correto.
    """
    n1_verb_ids, n2_verb_ids = _get_verb_ids()

    reps = extract_reps(model, contrastive_dataset, memory, batch_size, device)
    logits     = reps["logits"]         # (N, seq, vocab_size)
    ctx_labels = reps["true_context"]   # (N,) 0=A, 1=B
    pair_ids   = np.array([
        contrastive_dataset[i]["pair_id"]
        for i in range(len(contrastive_dataset))
    ])

    # Probabilidade de verbos de contexto A vs B na posição mascarada (pos=2 = verbo)
    verb_pos = 2
    verb_logits = logits[:, verb_pos, :]   # (N, vocab_size)
    p_a = _softmax_sum(verb_logits, n1_verb_ids)  # (N,)
    p_b = _softmax_sum(verb_logits, n2_verb_ids)  # (N,)

    # Agrupa por pair_id
    unique_pairs = np.unique(pair_ids)
    n_correct = 0
    n_pairs   = 0
    for pid in unique_pairs:
        mask = pair_ids == pid
        if mask.sum() != 2:
            continue
        idxs = np.where(mask)[0]
        ctx0, ctx1 = ctx_labels[idxs[0]], ctx_labels[idxs[1]]
        # Identifica qual item é o early (ctx=A=0) e qual é o late (ctx=B=1)
        if ctx0 == 0 and ctx1 == 1:
            i_early, i_late = idxs[0], idxs[1]
        elif ctx0 == 1 and ctx1 == 0:
            i_early, i_late = idxs[1], idxs[0]
        else:
            continue  # par inconsistente

        correct_early = p_a[i_early] > p_b[i_early]   # early deve favorecer A
        correct_late  = p_b[i_late]  > p_a[i_late]    # late deve favorecer B
        n_correct += int(correct_early and correct_late)
        n_pairs   += 1

    sign_flip_rate = n_correct / max(n_pairs, 1)
    return {
        "sign_flip_rate": sign_flip_rate,
        "n_pairs":        n_pairs,
        "n_correct":      n_correct,
    }


def _softmax_sum(logits: np.ndarray, token_ids: list[int]) -> np.ndarray:
    """Soma das probabilidades softmax para um conjunto de token_ids."""
    exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
    return probs[:, token_ids].sum(axis=-1)


# ─── Métricas representacionais ───────────────────────────────────────────────

def precision_at_k(
    X: np.ndarray, y: np.ndarray, k: int = 5
) -> float:
    """
    Para cada exemplo, encontra os k vizinhos mais próximos (excluindo ele mesmo).
    Retorna a fração de vizinhos que pertencem ao mesmo true_context.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    distances, indices = nn.kneighbors(X)
    # indices[:, 0] é o próprio exemplo
    neighbor_labels = y[indices[:, 1:]]         # (N, k)
    correct = (neighbor_labels == y[:, None])   # (N, k)
    return float(correct.mean())


def clustering_metrics(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """
    Agrupa X com K-Means (k=2) e mede ARI e NMI contra true_context.
    Alternativa: usa as labels previstas pela probe para não depender de K-Means.
    Aqui usa K-Means para medir separação geométrica independente da probe.
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    pred = km.fit_predict(X)
    return {
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
    }


# ─── Constantes de vocabulário para avaliação contrastiva ────────────────────
# Importadas pelos módulos de avaliação

def _build_context_verb_ids() -> tuple[list[int], list[int]]:
    from .dataset import TOKEN2ID
    a_ids = [TOKEN2ID[f"V{i}"] for i in range(1, 5)]
    b_ids = [TOKEN2ID[f"V{i}"] for i in range(5, 9)]
    return a_ids, b_ids


# Inicialização lazy para evitar import circular
_NEIGH_1_VERB_IDS: list[int] | None = None
_NEIGH_2_VERB_IDS: list[int] | None = None

# Backward-compatible alias names
_CONTEXT_A_VERB_IDS = _NEIGH_1_VERB_IDS
_CONTEXT_B_VERB_IDS = _NEIGH_2_VERB_IDS


def _get_verb_ids() -> tuple[list[int], list[int]]:
    global _NEIGH_1_VERB_IDS, _NEIGH_2_VERB_IDS
    if _NEIGH_1_VERB_IDS is None:
        _NEIGH_1_VERB_IDS, _NEIGH_2_VERB_IDS = _build_context_verb_ids()
    return _NEIGH_1_VERB_IDS, _NEIGH_2_VERB_IDS


# Patch para o módulo conseguir importar as constantes
import sys as _sys
_mod = _sys.modules[__name__]


def __getattr__(name: str):
    if name in ("NEIGH_1_VERB_IDS", "CONTEXT_A_VERB_IDS"):
        a, b = _get_verb_ids()
        _mod.NEIGH_1_VERB_IDS = a
        _mod.NEIGH_2_VERB_IDS = b
        return a
    if name in ("NEIGH_2_VERB_IDS", "CONTEXT_B_VERB_IDS"):
        a, b = _get_verb_ids()
        _mod.NEIGH_1_VERB_IDS = a
        _mod.NEIGH_2_VERB_IDS = b
        return b
    raise AttributeError(name)
