"""
Modelos para o experimento comparativo da Fase A.4 do Timeformer.

Quatro modelos treinados no mesmo corpus para comparar a capacidade de
recuperar trajetórias semânticas marginais P(ctx=A | token, época):

  Modelo A — Static:        E[token]                         (sem tempo)
  Modelo B — Independent:   E_t[token]                       (baseline atual, 1 modelo/época)
  Modelo C — TimeGlobal:    E[token] + T[epoch]               (TimeEncoding aditivo global)
  Modelo D — TimeInterative: MLP(concat(E[token], T[epoch]))   (TimeEncoding com interação)

Os Modelos A, C, D são treinados em TODAS as épocas conjuntamente.
O Modelo B (já existente em train_embeddings.py) é o baseline de referência.

Tarefa do skip-gram: dado token central, prever token de contexto.
Com marcadores probabilísticos (p_canon < 1.0), verbos/objetos são evidência
ruidosa do contexto — a acurácia máxima teórica de e(verb) é ≈ p_canon.

Hipóteses:
  A: P_pred constante por token → corr ≈ 0 para tokens com deriva/bifurcação
  B: P_pred correto por época mas ruidoso (corpus pequeno, sem continuidade)
  C: T[epoch] global conflita entre trajetórias díspares → corr intermediário
  D: MLP captura interação token×época → corr superior para todos os sujeitos
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.train_embeddings import (
    VOCAB_SIZE,
    EMBEDDING_DIM,
    TOKEN_TO_IDX,
    CONTEXT_A_INDICES,
    CONTEXT_B_INDICES,
    EPOCHS_ORDER,
)

NUM_EPOCHS_CORPUS = len(EPOCHS_ORDER)  # 6
EPOCH_TO_IDX: dict[str, int] = {ep: i for i, ep in enumerate(EPOCHS_ORDER)}

LEARNING_RATE = 0.01
TRAIN_STEPS   = 2000
SEED          = 42


# ─── Modelo A — Static ────────────────────────────────────────────────────────

class StaticSkipGram(nn.Module):
    """Um único embedding por token, sem dimensão temporal."""

    def __init__(self):
        super().__init__()
        self.embeddings   = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.output_proj  = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE, bias=False)

    def forward(self, center_idx: torch.Tensor, epoch_idx: torch.Tensor | None = None) -> torch.Tensor:
        return self.output_proj(self.embeddings(center_idx))

    def subject_repr(self, token_idx: int, epoch_idx: int) -> np.ndarray:
        with torch.no_grad():
            return self.embeddings(torch.tensor([token_idx])).squeeze().numpy()


# ─── Modelo C — Time Global ───────────────────────────────────────────────────

class TimeGlobalSkipGram(nn.Module):
    """
    E[token] + T[epoch]: TimeEncoding aditivo com T[epoch] compartilhado.

    T[epoch] é o mesmo vetor para TODOS os tokens naquela época. Isso cria
    conflito de gradientes quando sujeitos têm trajetórias distintas.
    Incluído como ablação para isolar o efeito da adição temporal simples.
    """

    def __init__(self):
        super().__init__()
        self.embeddings      = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.time_embeddings = nn.Embedding(NUM_EPOCHS_CORPUS, EMBEDDING_DIM)
        self.output_proj     = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE, bias=False)

    def forward(self, center_idx: torch.Tensor, epoch_idx: torch.Tensor) -> torch.Tensor:
        h = self.embeddings(center_idx) + self.time_embeddings(epoch_idx)
        return self.output_proj(h)

    def subject_repr(self, token_idx: int, epoch_idx: int) -> np.ndarray:
        with torch.no_grad():
            e = self.embeddings(torch.tensor([token_idx]))
            t = self.time_embeddings(torch.tensor([epoch_idx]))
            return (e + t).squeeze().numpy()


# ─── Modelo D — Time Interactive ─────────────────────────────────────────────

class TimeInteractiveSkipGram(nn.Module):
    """
    MLP(concat(E[token], T[epoch])): TimeEncoding com interação token×época.

    A não-linearidade permite que o modelo aprenda trajetórias específicas por
    token. S1 e S3 podem responder de forma diferente ao mesmo T[t3] porque
    seus E[token] são distintos e o MLP aprende a combinação.

    Arquitetura: 2d → ReLU → d, com d = EMBEDDING_DIM.
    """

    def __init__(self):
        super().__init__()
        d = EMBEDDING_DIM
        self.embeddings      = nn.Embedding(VOCAB_SIZE, d)
        self.time_embeddings = nn.Embedding(NUM_EPOCHS_CORPUS, d)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d),
        )
        self.output_proj = nn.Linear(d, VOCAB_SIZE, bias=False)

    def forward(self, center_idx: torch.Tensor, epoch_idx: torch.Tensor) -> torch.Tensor:
        e = self.embeddings(center_idx)
        t = self.time_embeddings(epoch_idx)
        h = self.mlp(torch.cat([e, t], dim=-1))
        return self.output_proj(h)

    def subject_repr(self, token_idx: int, epoch_idx: int) -> np.ndarray:
        with torch.no_grad():
            e = self.embeddings(torch.tensor([token_idx]))
            t = self.time_embeddings(torch.tensor([epoch_idx]))
            h = self.mlp(torch.cat([e, t], dim=-1))
            return h.squeeze().numpy()


# ─── Dados de treino para modelos conjuntos (A, C, D) ────────────────────────

def build_skipgram_pairs(
    corpus_rows: list[dict],
    split: str = "train",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constrói pares (center, epoch, context) para treino conjunto.

    Para Modelo A: epoch_idx é ignorado no forward.
    Para Modelos C, D: epoch_idx é usado para condicionar a representação.
    """
    centers, epochs_t, contexts = [], [], []
    for row in corpus_rows:
        if row.get("split", "train") != split:
            continue
        epoch_idx = EPOCH_TO_IDX[row["epoch"]]
        indices   = [TOKEN_TO_IDX[t] for t in row["sentence"].split()]
        for i, c in enumerate(indices):
            for j, ctx in enumerate(indices):
                if i != j:
                    centers.append(c)
                    epochs_t.append(epoch_idx)
                    contexts.append(ctx)
    return (
        torch.tensor(centers,  dtype=torch.long),
        torch.tensor(epochs_t, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
    )


# ─── Treino ───────────────────────────────────────────────────────────────────

def train_joint_model(
    model: nn.Module,
    corpus_rows: list[dict],
    steps: int = TRAIN_STEPS,
    lr: float = LEARNING_RATE,
    seed: int = SEED,
) -> nn.Module:
    """
    Treina Modelo A, C ou D em todas as épocas conjuntamente.
    O Modelo A ignora epoch_idx; os modelos C/D o utilizam.
    """
    torch.manual_seed(seed)
    centers, epochs_t, contexts = build_skipgram_pairs(corpus_rows, split="train")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    uses_time = isinstance(model, (TimeGlobalSkipGram, TimeInteractiveSkipGram))

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logits = model(centers, epochs_t) if uses_time else model(centers)
        loss   = criterion(logits, contexts)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ─── Predição de P(ctx=A | token, época) ─────────────────────────────────────

def compute_p_pred_trajectory(
    model: nn.Module,
    subjects: list[str],
) -> dict[str, list[float]]:
    """
    Para cada sujeito, retorna a lista de P(ctx=A) predita pelo modelo em t0..t5.

    P(ctx=A | S, t) = soma das probabilidades softmax sobre todos os tokens
    de contexto A (verbos V1-V4 + objetos O1-O4) dado S como token central.

    Para Modelo A: a mesma representação é usada em todas as épocas.
    Para Modelos C/D: a representação muda por época.
    """
    model.eval()
    result: dict[str, list[float]] = {}

    with torch.no_grad():
        for subject in subjects:
            token_idx = TOKEN_TO_IDX[subject]
            p_preds   = []

            for epoch in EPOCHS_ORDER:
                epoch_idx = EPOCH_TO_IDX[epoch]
                uses_time = isinstance(model, (TimeGlobalSkipGram, TimeInteractiveSkipGram))

                if uses_time:
                    logits = model(
                        torch.tensor([token_idx]),
                        torch.tensor([epoch_idx]),
                    )
                else:
                    logits = model(torch.tensor([token_idx]))

                probs = torch.softmax(logits, dim=-1).squeeze()
                p_a   = float(probs[CONTEXT_A_INDICES].sum())
                p_preds.append(p_a)

            result[subject] = p_preds

    return result
