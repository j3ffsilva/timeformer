"""
TimeEncoding contínuo para a Fase B do Timeformer.

TimeEncoding(t) = MLP(sinusoidal_features(t / T))

  sinusoidal_features: vetor fixo (não aprendido) com sin/cos de frequências
                       exponencialmente espaçadas — fórmula do Transformer original.
                       Fixo garante comportamento regular para épocas não vistas.
  MLP: Linear(d_sin → d_hidden) + GELU + Linear(d_hidden → d_model).
       d_hidden <= d_sin para evitar memorização de t0-t7 e extrapolação arbitrária.

TokenTimeInteraction: f(token_emb, time_emb) = Linear(concat(token_emb, time_emb))
  Usado por Joint para interação token×época.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class TimeEncoding(nn.Module):
    """
    Encoding temporal contínuo: features sinusoidais fixas + MLP pequena.

    Args:
        d_model:   dimensão de saída (igual ao d_model do Transformer)
        d_sin:     dimensão das features sinusoidais (fixas, não aprendidas)
        n_epochs:  número total de épocas (usado para normalizar t para [0, 1])
    """

    def __init__(self, d_model: int, d_sin: int = 32, n_epochs: int = 10) -> None:
        super().__init__()
        assert d_sin % 2 == 0, "d_sin deve ser par (sin + cos)"
        self.d_sin = d_sin
        self.T = max(n_epochs - 1, 1)  # normaliza t ∈ {0..n_epochs-1} → [0, 1]

        # Frequências fixas: 1/10000^(2i/d_sin) — análogo ao pos encoding do Transformer
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, d_sin, 2).float() / d_sin
        )
        self.register_buffer("freqs", freqs)  # (d_sin/2,)

        # MLP pequena: d_sin → d_sin → d_model (d_hidden = d_sin ≤ d_sin)
        self.mlp = nn.Sequential(
            nn.Linear(d_sin, d_sin),
            nn.GELU(),
            nn.Linear(d_sin, d_model),
        )

    def _sinusoidal(self, t_norm: Tensor) -> Tensor:
        """
        t_norm: (batch,) floats em [0, 1]
        retorna: (batch, d_sin)
        """
        angles = t_norm.unsqueeze(-1) * self.freqs.unsqueeze(0)  # (batch, d_sin/2)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (batch, d_sin)

    def forward(self, epoch_idx: Tensor) -> Tensor:
        """
        epoch_idx: (batch,) int  — índice da época (0-indexado)
        retorna:   (batch, d_model)
        """
        t_norm = epoch_idx.float() / self.T
        sin_feats = self._sinusoidal(t_norm)
        return self.mlp(sin_feats)


class TokenTimeInteraction(nn.Module):
    """
    Interação token×época para Joint: f(token_emb, time_emb) = Linear(concat(...)).

    Projeta a concatenação [token_emb; time_emb] de volta para d_model.
    A projeção permite que a rede aprenda quais dimensões temporais são
    relevantes para cada token específico.

    Args:
        d_model: dimensão dos embeddings de token e da saída
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, token_emb: Tensor, time_emb: Tensor) -> Tensor:
        """
        token_emb: (batch, seq, d_model)
        time_emb:  (batch, d_model)  — mesmo vetor de época para todos os tokens
        retorna:   (batch, seq, d_model)
        """
        # Expande time_emb para todos os tokens da sequência
        time_expanded = time_emb.unsqueeze(1).expand_as(token_emb)  # (batch, seq, d_model)
        combined = torch.cat([token_emb, time_expanded], dim=-1)     # (batch, seq, 2*d_model)
        return self.proj(combined)
