"""
Modelos da Fase B do Timeformer.

Cadeia de ablação:
  B1   — Transformer textual sem época
  B2a  — B1 + TimeEncoding aditivo global
  B2b  — B1 + interação token×época (TokenTimeInteraction)
  B3   — B2b + atenção temporal fatorada sobre memória histórica (Timeformer)

Todos os modelos:
  - Entrada: [CLS] S V O [SEP]  (SEQ_LEN=5)
  - Saída pré-treino: logits sobre vocab (MLM head)
  - Saída de representação: hidden states da última camada encoder
    → h(sujeito) = hidden[:, POS_SUBJECT, :]
    → h([CLS])   = hidden[:, POS_CLS, :]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dataset import VOCAB_SIZE, SEQ_LEN, POS_SUBJECT, POS_CLS
from .encoding import TimeEncoding, TokenTimeInteraction

# ─── Transformer encoder base ─────────────────────────────────────────────────

class _TransformerEncoderBase(nn.Module):
    """
    Transformer encoder padrão (BERT-like) compartilhado por todos os modelos.
    Não inclui embeddings de entrada nem MLM head — cada modelo define os seus.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: mais estável em corpus pequeno
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq, d_model) → (batch, seq, d_model)"""
        return self.encoder(x)


class MLMHead(nn.Module):
    """Projeção linear de d_model → vocab_size. Compartilhada por todos os modelos."""

    def __init__(self, d_model: int, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.proj  = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: Tensor) -> Tensor:
        """hidden: (batch, seq, d_model) → logits: (batch, seq, vocab_size)"""
        return self.proj(F.gelu(self.norm(hidden)))


# ─── B1: Transformer sem tempo ────────────────────────────────────────────────

class B1(nn.Module):
    """
    Transformer contextual padrão sem informação de época.

    embedding = TokenEmbedding(token) + PositionalEncoding(pos)

    Adversário base que usa apenas co-ocorrência aprendida durante treino.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1,
                 vocab_size: int = VOCAB_SIZE, seq_len: int = SEQ_LEN) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(seq_len, d_model)
        self.drop      = nn.Dropout(dropout)
        self.encoder   = _TransformerEncoderBase(d_model, n_heads, n_layers, d_ff, dropout)
        self.mlm_head  = MLMHead(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)

    def embed(self, input_ids: Tensor) -> Tensor:
        """input_ids: (batch, seq) → embeddings: (batch, seq, d_model)"""
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        return self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

    def encode(self, input_ids: Tensor, **_kwargs) -> Tensor:
        """Retorna hidden states: (batch, seq, d_model)."""
        return self.encoder(self.embed(input_ids))

    def forward(self, input_ids: Tensor, **kwargs) -> dict:
        """
        Retorna dict com:
          logits:   (batch, seq, vocab_size)  — para MLM loss
          hidden:   (batch, seq, d_model)     — para probe
          h_subj:   (batch, d_model)          — h(sujeito)
          h_cls:    (batch, d_model)          — h([CLS])
        """
        hidden = self.encode(input_ids, **kwargs)
        return {
            "logits": self.mlm_head(hidden),
            "hidden": hidden,
            "h_subj": hidden[:, POS_SUBJECT, :],
            "h_cls":  hidden[:, POS_CLS, :],
        }


# ─── B2a: Transformer + TimeEncoding aditivo global ───────────────────────────

class B2a(B1):
    """
    B1 + TimeEncoding aditivo global.

    embedding = TokenEmbedding(token) + PositionalEncoding(pos) + TimeEncoding(t)

    O mesmo vetor TimeEncoding(t) é somado a todos os tokens da sentença.
    Ablação auxiliar: isola o efeito mínimo de injetar época globalmente.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1,
                 vocab_size: int = VOCAB_SIZE, seq_len: int = SEQ_LEN,
                 d_sin: int = 32, n_epochs: int = 10) -> None:
        super().__init__(d_model, n_heads, n_layers, d_ff, dropout, vocab_size, seq_len)
        self.time_enc = TimeEncoding(d_model, d_sin, n_epochs)

    def embed(self, input_ids: Tensor, epoch_idx: Tensor | None = None) -> Tensor:
        base = super().embed(input_ids)  # (batch, seq, d_model)
        if epoch_idx is not None:
            t = self.time_enc(epoch_idx)           # (batch, d_model)
            base = base + t.unsqueeze(1)           # broadcast sobre todos os tokens
        return base

    def encode(self, input_ids: Tensor, epoch_idx: Tensor | None = None, **_) -> Tensor:
        return self.encoder(self.embed(input_ids, epoch_idx))

    def forward(self, input_ids: Tensor, epoch_idx: Tensor | None = None, **_) -> dict:
        hidden = self.encode(input_ids, epoch_idx)
        return {
            "logits": self.mlm_head(hidden),
            "hidden": hidden,
            "h_subj": hidden[:, POS_SUBJECT, :],
            "h_cls":  hidden[:, POS_CLS, :],
        }


# ─── B2b: Transformer + interação token×época ─────────────────────────────────

class B2b(nn.Module):
    """
    Transformer com interação token×época.

    embedding = TokenTimeInteraction(TokenEmbedding(token), TimeEncoding(t))
                + PositionalEncoding(pos)

    A época interage com cada token individualmente — diferente de B2a
    que soma o mesmo vetor a todos os tokens.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1,
                 vocab_size: int = VOCAB_SIZE, seq_len: int = SEQ_LEN,
                 d_sin: int = 32, n_epochs: int = 10) -> None:
        super().__init__()
        self.d_model   = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(seq_len, d_model)
        self.time_enc  = TimeEncoding(d_model, d_sin, n_epochs)
        self.tti       = TokenTimeInteraction(d_model)
        self.drop      = nn.Dropout(dropout)
        self.encoder   = _TransformerEncoderBase(d_model, n_heads, n_layers, d_ff, dropout)
        self.mlm_head  = MLMHead(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)

    def embed(self, input_ids: Tensor, epoch_idx: Tensor | None = None) -> Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        tok = self.token_emb(input_ids)               # (batch, seq, d_model)
        pos = self.pos_emb(positions)                 # (seq, d_model) → broadcast
        if epoch_idx is not None:
            t   = self.time_enc(epoch_idx)            # (batch, d_model)
            tok = self.tti(tok, t)                    # interação token×época
        return self.drop(tok + pos)

    def encode(self, input_ids: Tensor, epoch_idx: Tensor | None = None, **_) -> Tensor:
        return self.encoder(self.embed(input_ids, epoch_idx))

    def forward(self, input_ids: Tensor, epoch_idx: Tensor | None = None, **_) -> dict:
        hidden = self.encode(input_ids, epoch_idx)
        return {
            "logits": self.mlm_head(hidden),
            "hidden": hidden,
            "h_subj": hidden[:, POS_SUBJECT, :],
            "h_cls":  hidden[:, POS_CLS, :],
        }


# ─── B3: Timeformer ────────────────────────────────────────────────────────────

class _TemporalCrossAttention(nn.Module):
    """
    Cross-attention temporal: h(sujeito) consulta memória histórica {m(S, t<k)}.

    query:  representação atual do sujeito h(S, t_k)  — (batch, d_model)
    key/value: protótipos históricos m(S, t<k)         — (batch, hist_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        # Gate inicializado em 0: tanh(0)=0, logo B3 começa idêntico a B2b.
        # Abre apenas se o gradiente MLM recompensar o uso da memória.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h_subj: Tensor, memory: Tensor,
                memory_mask: Tensor | None = None) -> Tensor:
        """
        h_subj:      (batch, d_model)
        memory:      (batch, hist_len, d_model)  — zeros se sem histórico
        memory_mask: (batch, hist_len) bool — True onde válido (False = padding)
        retorna:     (batch, d_model)  — h_subj atualizado

        Linhas sem nenhum protótipo válido mantêm h_subj inalterado.
        PyTorch MHA retorna NaN quando todos os keys são mascarados — a correção
        é forçar pelo menos um key válido por linha (no mínimo o zero-token),
        depois restaurar h_subj para as linhas sem histórico real.
        """
        has_valid: Tensor | None = None
        key_padding_mask: Tensor | None = None

        if memory_mask is not None:
            has_valid = memory_mask.any(dim=-1)   # (batch,) — True se tem >= 1 proto válido
            if not has_valid.any():
                return h_subj                     # nenhuma linha tem histórico — early exit

            kpm = ~memory_mask                    # True onde IGNORAR (convenção PyTorch)
            needs_fix = ~has_valid
            if needs_fix.any():
                kpm = kpm.clone()
                kpm[needs_fix, 0] = False         # força 1 key "válido" para evitar NaN
            key_padding_mask = kpm

        query = h_subj.unsqueeze(1)               # (batch, 1, d_model)
        attn_out, _ = self.attn(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask,
        )
        attn_out = attn_out.squeeze(1)            # (batch, d_model)

        # Gated residual: delta = norm(h + attn) − h; gate=0 → updated = h_subj
        delta   = self.norm(h_subj + self.drop(attn_out)) - h_subj
        updated = h_subj + torch.tanh(self.gate) * delta

        if has_valid is not None and not has_valid.all():
            updated = torch.where(has_valid.unsqueeze(-1), updated, h_subj)

        return updated


class B3(nn.Module):
    """
    Timeformer: atenção textual + atenção temporal fatorada.

    Arquitetura (v2 — temporal update ANTES do encoder):
      1. embedding: TokenTimeInteraction (igual ao B2b)
      2. atenção temporal: cross-attention de emb(sujeito) sobre memória histórica
         aplicada NO EMBEDDING, antes do encoder textual
      3. encoder textual: Transformer sobre tokens — h_verb/h_object veem h_subj
         já contextualizado historicamente, criando caminho de gradiente MLM
      4. gated residual com gate=0 inicial: B3 começa idêntico a B2b

    A memória histórica {m(S, t_0..t_{k-1})} é injetada externamente pelo
    trainer via PrototypeMemory — o modelo não a computa internamente.

    Causalidade garantida externamente: memory deve conter apenas t < t_k.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1,
                 vocab_size: int = VOCAB_SIZE, seq_len: int = SEQ_LEN,
                 d_sin: int = 32, n_epochs: int = 10) -> None:
        super().__init__()
        self.d_model   = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(seq_len, d_model)
        self.time_enc  = TimeEncoding(d_model, d_sin, n_epochs)
        self.tti       = TokenTimeInteraction(d_model)
        self.drop      = nn.Dropout(dropout)
        self.encoder   = _TransformerEncoderBase(d_model, n_heads, n_layers, d_ff, dropout)
        self.temp_attn = _TemporalCrossAttention(d_model, n_heads, dropout)
        self.mlm_head  = MLMHead(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)

    def embed(self, input_ids: Tensor, epoch_idx: Tensor,
              memory: Tensor | None = None,
              memory_mask: Tensor | None = None) -> Tensor:
        """
        Computa embeddings com update temporal aplicado ANTES do encoder.

        O sujeito chega ao encoder já contextualizado historicamente, de modo
        que h_verb/h_object (computados pelo encoder) recebem influência de
        m(S, t<k) via self-attention textual — criando caminho de gradiente
        do MLM loss até os pesos de temp_attn.
        """
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        tok = self.token_emb(input_ids)
        t   = self.time_enc(epoch_idx)
        tok = self.tti(tok, t)
        emb = self.drop(tok + self.pos_emb(positions))   # (batch, seq, d_model)

        if memory is not None and memory.size(1) > 0:
            h_subj          = emb[:, POS_SUBJECT, :]
            h_subj_updated  = self.temp_attn(h_subj, memory, memory_mask)
            emb             = emb.clone()
            emb[:, POS_SUBJECT, :] = h_subj_updated

        return emb

    def encode(self, input_ids: Tensor, epoch_idx: Tensor,
               memory: Tensor | None = None,
               memory_mask: Tensor | None = None) -> Tensor:
        return self.encoder(self.embed(input_ids, epoch_idx, memory, memory_mask))

    def forward(self, input_ids: Tensor, epoch_idx: Tensor,
                memory: Tensor | None = None,
                memory_mask: Tensor | None = None, **_) -> dict:
        hidden = self.encode(input_ids, epoch_idx, memory, memory_mask)
        return {
            "logits": self.mlm_head(hidden),
            "hidden": hidden,
            "h_subj": hidden[:, POS_SUBJECT, :],
            "h_cls":  hidden[:, POS_CLS, :],
        }


# ─── Fábrica de modelos ────────────────────────────────────────────────────────

_MODEL_CLASSES = {"B1": B1, "B2a": B2a, "B2b": B2b, "B3": B3}

DEFAULT_HPARAMS = {
    "d_model":  64,
    "n_heads":  4,
    "n_layers": 2,
    "d_ff":     128,
    "dropout":  0.1,
    "d_sin":    32,
    "n_epochs": 10,
}


def build_model(name: str, **hparams) -> nn.Module:
    """
    Instancia um modelo pelo nome com hiperparâmetros opcionais.
    Valores não fornecidos usam DEFAULT_HPARAMS.

    Ex: build_model("B3", d_model=128)
    """
    if name not in _MODEL_CLASSES:
        raise ValueError(f"Modelo desconhecido: {name!r}. Opções: {list(_MODEL_CLASSES)}")
    params = {**DEFAULT_HPARAMS, **hparams}
    cls = _MODEL_CLASSES[name]
    # B1 não aceita d_sin/n_epochs — filtra kwargs irrelevantes
    import inspect
    valid = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in params.items() if k in valid}
    return cls(**filtered)
