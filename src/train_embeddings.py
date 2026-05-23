"""
Treina embeddings skip-gram separadamente por época.

Decisão de arquitetura:
- Skip-gram com softmax completo: viável porque o vocabulário tem apenas 22 tokens.
  Negative sampling seria desnecessário aqui e tornaria o código mais complexo.
- Cada época é treinada com o mesmo seed inicial para garantir que diferenças
  entre embeddings reflitam os dados, não variação de inicialização.
- Treinamento em batch completo (todas as pares de uma vez): com ~3000 pares
  por época, isso converge rápido e é determinístico.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
EMBEDDING_DIM = 16  # 32D é superparametrizado para 22 tokens; 16D melhora a qualidade do Procrustes
LEARNING_RATE = 0.01
TRAIN_STEPS = 2000  # passos de gradiente por época (não confundir com épocas do corpus)

# Vocabulário fixo — a ordem define os índices usados em todos os módulos
# Contexto A: V1-V4, O1-O4  |  Contexto B: V5-V8, O5-O8
VOCAB = [
    # 30 sujeitos (v2)
    "S1",  "S2",  "S3",  "S4",  "S5",  "S6",  "S7",  "S8",  "S9",  "S10",
    "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20",
    "S21", "S22", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30",
    # verbos e objetos (contexto A: V1-V4, O1-O4 | contexto B: V5-V8, O5-O8)
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8",
]
VOCAB_SIZE = len(VOCAB)
TOKEN_TO_IDX: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
IDX_TO_TOKEN: dict[int, str] = {i: tok for tok, i in TOKEN_TO_IDX.items()}

EPOCHS_ORDER = [f"t{i}" for i in range(10)]


class SkipGram(nn.Module):
    """
    Embeddings treináveis com predição de contexto via skip-gram.
    A matriz de output não compartilha pesos com a de input — isso
    estabiliza o treinamento com vocabulários pequenos.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, center_indices: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.embeddings(center_indices))


def _build_skipgram_pairs(sentences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Para cada frase SVO, gera todos os pares (centro, contexto) dentro da janela 2.
    Com frases de comprimento 3, isso cobre a frase inteira: cada token
    prediz os outros dois.
    """
    centers, contexts = [], []
    for sentence in sentences:
        indices = [TOKEN_TO_IDX[t] for t in sentence.split()]
        for i, c in enumerate(indices):
            for j, ctx in enumerate(indices):
                if i != j:
                    centers.append(c)
                    contexts.append(ctx)
    return (
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
    )


def _train_one_epoch(
    sentences: list[str], epoch_label: str, model_save_path: Path | None = None
) -> tuple[np.ndarray, "SkipGram"]:
    """Treina e retorna (matriz de embeddings, modelo treinado)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    centers, contexts = _build_skipgram_pairs(sentences)

    model = SkipGram(VOCAB_SIZE, EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        optimizer.zero_grad()
        loss = criterion(model(centers), contexts)
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"  [{epoch_label}] passo {step:>4}/{TRAIN_STEPS}  loss={loss.item():.4f}")

    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)

    return model.embeddings.weight.detach().cpu().numpy(), model


# Tokens de cada contexto — usados para computar probabilidades de predição
CONTEXT_A_TOKENS = ["V1", "V2", "V3", "V4", "O1", "O2", "O3", "O4"]
CONTEXT_B_TOKENS = ["V5", "V6", "V7", "V8", "O5", "O6", "O7", "O8"]
CONTEXT_A_INDICES = [TOKEN_TO_IDX[t] for t in CONTEXT_A_TOKENS]
CONTEXT_B_INDICES = [TOKEN_TO_IDX[t] for t in CONTEXT_B_TOKENS]


def compute_context_probs(model: "SkipGram") -> dict[str, tuple[float, float]]:
    """
    Para cada sujeito, calcula P(ContextoA) e P(ContextoB) segundo o modelo.

    Usa as probabilidades de predição (softmax sobre o vocabulário) em vez
    de similaridade cosseno nos embeddings, porque o skip-gram aprende
    e(sujeito) · w_out(token) — a afinidade correta está no produto
    com a projeção de saída, não entre embeddings de entrada.
    """
    model.eval()
    result: dict[str, tuple[float, float]] = {}
    with torch.no_grad():
        for subject in ["S1", "S2", "S3", "S4", "S5", "S6"]:
            logits = model(torch.tensor([TOKEN_TO_IDX[subject]]))
            probs = torch.softmax(logits, dim=-1).squeeze()
            prob_a = float(probs[CONTEXT_A_INDICES].sum())
            prob_b = float(probs[CONTEXT_B_INDICES].sum())
            result[subject] = (prob_a, prob_b)
    return result


def train_all_epochs(
    corpus_rows: list[dict],
    output_dir: Path,
) -> dict[str, np.ndarray]:
    """
    Treina uma matriz de embeddings para cada época do corpus.
    Salva cada matriz como .npy e o mapeamento vocab→idx como JSON.
    Retorna dict[epoch_label → ndarray(VOCAB_SIZE, EMBEDDING_DIM)].
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Agrupa frases de TREINO por época — o split 'test' é reservado para o probe
    by_epoch: dict[str, list[str]] = {}
    for row in corpus_rows:
        if row.get("split", "train") == "train":
            by_epoch.setdefault(row["epoch"], []).append(row["sentence"])

    all_embeddings: dict[str, np.ndarray] = {}
    all_models: dict[str, "SkipGram"] = {}
    for epoch_label in EPOCHS_ORDER:
        sentences = by_epoch[epoch_label]
        print(f"\nTreinando embeddings para {epoch_label} ({len(sentences)} frases) …")
        model_path = output_dir / f"model_{epoch_label}.pt"
        emb, model = _train_one_epoch(sentences, epoch_label, model_save_path=model_path)
        all_embeddings[epoch_label] = emb
        all_models[epoch_label] = model
        np.save(output_dir / f"embeddings_{epoch_label}.npy", emb)

    # Persiste o mapeamento vocab para permitir carregar os .npy isoladamente
    with open(output_dir / "vocab.json", "w") as f:
        json.dump(TOKEN_TO_IDX, f, indent=2)

    print(f"\nEmbeddings e modelos salvos em {output_dir}/")
    return all_embeddings, all_models


def load_embeddings(embeddings_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, "SkipGram"]]:
    """Carrega embeddings (.npy) e modelos (.pt) salvos anteriormente."""
    embeddings = {
        f"t{i}": np.load(embeddings_dir / f"embeddings_t{i}.npy")
        for i in range(6)
    }
    models: dict[str, SkipGram] = {}
    for i in range(6):
        ep = f"t{i}"
        m = SkipGram(VOCAB_SIZE, EMBEDDING_DIM)
        m.load_state_dict(torch.load(embeddings_dir / f"model_{ep}.pt"))
        m.eval()
        models[ep] = m
    return embeddings, models
