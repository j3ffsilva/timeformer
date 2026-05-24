"""
Prototype Memory Manager para o B3 (Timeformer).

m(S, t) = mean_pool(h(S) nas frases de treino de S em época t)

Propriedades:
  - Stop-gradient: protótipos computados com torch.no_grad()
  - Atualização: uma vez por época de treino (não por batch)
  - Causal: get() retorna apenas t < epoch_k
  - Fonte: apenas frases de split='train' (sem vazamento de test/continuation)
  - t0 sem histórico: get() retorna tensor vazio (batch, 0, d_model)

Para B3-shuffled: make_shuffled() permuta a associação sujeito→protótipos.
Para B3-nohistory: make_nohistory() retorna zeros para todos os sujeitos/épocas.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import SUBJECTS, N_EPOCHS


class PrototypeMemory:
    """
    Armazena e serve protótipos m(S, t) para o B3.

    Args:
        n_subjects: número de sujeitos (30)
        n_epochs:   número de épocas (10)
        d_model:    dimensão dos embeddings
        device:     dispositivo onde os protótipos ficam armazenados
    """

    def __init__(
        self,
        n_subjects: int = len(SUBJECTS),
        n_epochs: int = N_EPOCHS,
        d_model: int = 64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.n_subjects = n_subjects
        self.n_epochs   = n_epochs
        self.d_model    = d_model
        self.device     = torch.device(device)

        # Protótipos: (n_subjects, n_epochs, d_model) — inicializa com zeros
        self._protos = torch.zeros(n_subjects, n_epochs, d_model, device=self.device)
        # Máscara: True onde o protótipo foi computado ao menos uma vez
        self._valid  = torch.zeros(n_subjects, n_epochs, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def update(self, model: nn.Module, dataloader: DataLoader, epoch: int) -> None:
        """
        Recalcula m(S, epoch) para todos os sujeitos usando frases do dataloader.

        Deve ser chamado ao fim de cada época de treino, antes de usar get().
        O dataloader deve conter apenas frases de split='train'.

        O modelo é chamado em modo eval com no_grad — protótipos são stop-gradient.
        """
        model.eval()

        # Acumuladores: soma de h(S) e contagem, por sujeito
        accum  = torch.zeros(self.n_subjects, self.d_model, device=self.device)
        counts = torch.zeros(self.n_subjects, device=self.device)

        for batch in dataloader:
            input_ids   = batch["input_ids"].to(self.device)
            epoch_idx_b = batch["epoch_idx"].to(self.device)
            subject_idx = batch["subject_idx"].to(self.device)

            # Filtra apenas frases da época atual
            mask = epoch_idx_b == epoch
            if not mask.any():
                continue

            input_ids_e   = input_ids[mask]
            epoch_idx_e   = epoch_idx_b[mask]
            subject_idx_e = subject_idx[mask]

            # Forward pass sem gradiente
            if hasattr(model, "encode"):
                if isinstance(model, type) or "B3" in type(model).__name__:
                    # B3 precisa de memory=None para esta passagem
                    hidden = model.encode(input_ids_e, epoch_idx_e, memory=None)
                else:
                    try:
                        hidden = model.encode(input_ids_e, epoch_idx_e)
                    except TypeError:
                        hidden = model.encode(input_ids_e)
            else:
                out = model(input_ids_e, epoch_idx=epoch_idx_e)
                hidden = out["hidden"]

            # h(sujeito) = hidden[:, POS_SUBJECT, :]
            from .dataset import POS_SUBJECT
            h_subj = hidden[:, POS_SUBJECT, :].to(self.device)  # (batch_e, d_model)

            # Acumula por sujeito
            for i in range(h_subj.size(0)):
                s = subject_idx_e[i].item()
                accum[s]  += h_subj[i]
                counts[s] += 1

        # Calcula mean-pool e atualiza protótipos
        valid_mask = counts > 0
        if valid_mask.any():
            protos = torch.zeros_like(accum)
            protos[valid_mask] = accum[valid_mask] / counts[valid_mask].unsqueeze(-1)
            self._protos[:, epoch, :] = protos
            self._valid[:, epoch] = valid_mask

    def get(
        self,
        subject_idx: Tensor,
        epoch_k: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Retorna protótipos históricos para os sujeitos do batch.

        Causal: retorna apenas épocas t < epoch_k.

        Args:
            subject_idx: (batch,) — índices dos sujeitos no batch
            epoch_k:     época atual (escalar)

        Retorna:
            memory:      (batch, hist_len, d_model)  — protótipos t_0..t_{k-1}
            memory_mask: (batch, hist_len) bool       — True onde protótipo é válido
        """
        hist_epochs = list(range(epoch_k))  # épocas disponíveis: 0..k-1
        hist_len    = len(hist_epochs)

        if hist_len == 0:
            # t0: sem histórico
            empty_mem  = torch.zeros(len(subject_idx), 0, self.d_model, device=self.device)
            empty_mask = torch.zeros(len(subject_idx), 0, dtype=torch.bool, device=self.device)
            return empty_mem, empty_mask

        # Coleta protótipos: (batch, hist_len, d_model)
        subj = subject_idx.to(self.device)
        memory = self._protos[subj][:, hist_epochs, :]         # (batch, hist_len, d_model)
        mask   = self._valid[subj][:, hist_epochs]             # (batch, hist_len)

        return memory.detach(), mask.detach()

    def to(self, device: torch.device | str) -> "PrototypeMemory":
        self.device = torch.device(device)
        self._protos = self._protos.to(self.device)
        self._valid  = self._valid.to(self.device)
        return self


def make_shuffled(
    memory: PrototypeMemory,
    mode: str = "subject",
    seed: int = 0,
) -> PrototypeMemory:
    """
    Cria uma PrototypeMemory com protótipos embaralhados para controle B3-shuffled.

    mode="subject": permuta a associação sujeito→protótipos
                    (sujeito S recebe o histórico de outro sujeito S')
    mode="time":    embaralha a ordem das épocas para cada sujeito
                    (ordem temporal incorreta, sujeito correto)

    Se B3 real não vencer B3-shuffled com margem, o ganho não é pela trajetória.
    """
    import random
    rng = random.Random(seed)

    shuffled = PrototypeMemory(
        memory.n_subjects, memory.n_epochs, memory.d_model, memory.device
    )

    if mode == "subject":
        perm = list(range(memory.n_subjects))
        rng.shuffle(perm)
        shuffled._protos = memory._protos[perm].clone()
        shuffled._valid  = memory._valid[perm].clone()

    elif mode == "time":
        protos = memory._protos.clone()  # (n_subjects, n_epochs, d_model)
        valid  = memory._valid.clone()
        for s in range(memory.n_subjects):
            epoch_perm = list(range(memory.n_epochs))
            rng.shuffle(epoch_perm)
            shuffled._protos[s] = protos[s, epoch_perm, :]
            shuffled._valid[s]  = valid[s, epoch_perm]

    else:
        raise ValueError(f"mode deve ser 'subject' ou 'time', não {mode!r}")

    return shuffled


def make_nohistory(
    n_subjects: int = len(SUBJECTS),
    n_epochs: int = N_EPOCHS,
    d_model: int = 64,
    device: torch.device | str = "cpu",
) -> PrototypeMemory:
    """
    PrototypeMemory zerada — mesma arquitetura de B3, sem informação histórica.

    Controle B3-nohistory:
      B3 > B3-nohistory → arquitetura temporal tem efeito mesmo sem histórico real
      B3 > B3-shuffled  → o histórico correto especificamente importa
    """
    return PrototypeMemory(n_subjects, n_epochs, d_model, device)
    # Retorna zeros (inicialização padrão) — _valid permanece False
