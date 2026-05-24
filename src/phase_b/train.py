"""
Loop de treino MLM genérico para a Fase B.

Funciona para B1, B2a, B2b, B3.
Para B3, aceita PrototypeMemory e atualiza os protótipos ao fim de cada época
(stop-gradient — não no loop de batch).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .dataset import b3_collate_fn
from .memory import PrototypeMemory


def _mlm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Cross-entropy apenas nas posições mascaradas (labels != -100).
    logits: (batch, seq, vocab_size)
    labels: (batch, seq)  — -100 nas posições não mascaradas
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


def _forward(
    model: nn.Module,
    batch: dict,
    memory: PrototypeMemory | None,
    device: torch.device,
) -> dict:
    """
    Forward pass unificado para todos os modelos.
    Injeta memória no B3 quando disponível.
    """
    input_ids   = batch["input_ids"].to(device)
    epoch_idx   = batch["epoch_idx"].to(device)
    labels      = batch["labels"].to(device)
    subject_idx = batch["subject_idx"].to(device)

    model_name = type(model).__name__

    if model_name == "B1":
        out = model(input_ids)
    elif model_name in ("B2a", "B2b"):
        out = model(input_ids, epoch_idx)
    else:  # B3
        if memory is not None and "history_epochs" in batch:
            # Injeta memória histórica por sujeito
            mem_list, mask_list = [], []
            for i in range(input_ids.size(0)):
                k = epoch_idx[i].item()
                s = subject_idx[i:i+1]
                mem_i, mask_i = memory.get(s, epoch_k=k)  # (1, k, d_model)
                mem_list.append(mem_i)
                mask_list.append(mask_i)
            # Padding para tamanho uniforme no batch
            max_hist = max(m.size(1) for m in mem_list)
            d = memory.d_model
            mem_batch  = torch.zeros(len(mem_list), max_hist, d, device=device)
            mask_batch = torch.zeros(len(mem_list), max_hist, dtype=torch.bool, device=device)
            for i, (m, mk) in enumerate(zip(mem_list, mask_list)):
                h = m.size(1)
                if h > 0:
                    mem_batch[i, :h, :] = m.to(device)
                    mask_batch[i, :h]   = mk.to(device)
            out = model(input_ids, epoch_idx, memory=mem_batch, memory_mask=mask_batch)
        else:
            out = model(input_ids, epoch_idx, memory=None)

    out["loss"] = _mlm_loss(out["logits"], labels)
    return out


class MLMTrainer:
    """
    Treina um modelo com objetivo MLM.

    Args:
        model:      B1, B2a, B2b ou B3
        output_dir: diretório para salvar checkpoints e log
        device:     dispositivo de treino
    """

    def __init__(
        self,
        model: nn.Module,
        output_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model      = model
        self.output_dir = Path(output_dir)
        self.device     = torch.device(device)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        memory: PrototypeMemory | None = None,
        n_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> list[dict]:
        """
        Treina o modelo por n_epochs e retorna histórico de loss por época.

        Para B3: memory deve ser uma PrototypeMemory inicializada.
        A memória é atualizada (stop-gradient) ao fim de cada época de treino.
        """
        torch.manual_seed(seed)

        use_b3_collate = type(self.model).__name__ == "B3"

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=b3_collate_fn if use_b3_collate else None,
        )
        # Val loader usa collate padrão — memória injetada via subject_idx/epoch_idx
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        ) if val_dataset is not None else None

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        history = []
        best_val_loss = float("inf")

        for epoch in range(n_epochs):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, optimizer, memory)
            val_loss   = self._eval_epoch(val_loader, memory) if val_loader else None
            scheduler.step()

            record = {
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "elapsed_s":  round(time.time() - t0, 2),
            }
            history.append(record)
            self._log(record)

            # Salva melhor checkpoint por val_loss (ou train_loss se sem val)
            monitor = val_loss if val_loss is not None else train_loss
            if monitor < best_val_loss:
                best_val_loss = monitor
                self._save_checkpoint("best.pt")

            # Atualiza protótipos de B3 ao fim da época (stop-gradient)
            if memory is not None:
                for ep_idx in range(10):
                    memory.update(self.model, train_loader, epoch=ep_idx)

        self._save_checkpoint("final.pt")
        self._save_history(history)
        return history

    def _train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer,
        memory: PrototypeMemory | None,
    ) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            out = _forward(self.model, batch, memory, self.device)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += out["loss"].item()
            n_batches  += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(
        self, loader: DataLoader | None, memory: PrototypeMemory | None,
    ) -> float | None:
        if loader is None:
            return None
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            out = _forward(self.model, batch, memory, self.device)
            total_loss += out["loss"].item()
            n_batches  += 1
        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, name: str) -> None:
        torch.save(self.model.state_dict(), self.output_dir / name)

    def _save_history(self, history: list[dict]) -> None:
        import json
        (self.output_dir / "train_history.json").write_text(
            json.dumps(history, indent=2)
        )

    def _log(self, record: dict) -> None:
        ep  = record["epoch"]
        tl  = record["train_loss"]
        vl  = record["val_loss"]
        el  = record["elapsed_s"]
        vl_str = f"val={vl:.4f}" if vl is not None else "val=—"
        print(f"  epoch {ep:3d}  train={tl:.4f}  {vl_str}  ({el}s)")


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> nn.Module:
    """Carrega pesos de um checkpoint salvo pelo MLMTrainer."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model
