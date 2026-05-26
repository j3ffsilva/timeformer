"""
Oracle Memory Diagnostic para o Timeformer.

Substitui os protótipos aprendidos por vetores que codificam explicitamente
P(N1 | S, t) calculado do split de treino do corpus (ground truth).

Hipótese testada:
  Se Timeformer-oracle > Joint  →  arquitetura funciona; problema são os protótipos aprendidos
  Se Timeformer-oracle ≈ Joint  →  MLM/SVO não beneficia de memória independente da qualidade

Protocolo:
  1. Constrói OracleMemory a partir do corpus (ground truth P(N1|S,t))
  2. Treina Timeformer com oracle memory fixada (não atualizada durante treino)
  3. Avalia Timeformer-oracle e compara com Joint e Timeformer-learned da run de referência

Uso:
  python scripts/oracle_diagnostic.py                        # run mais recente como referência
  python scripts/oracle_diagnostic.py --run-id 20260523_004
  python scripts/oracle_diagnostic.py --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.timeformer.dataset import load_corpus, MLMDataset, TimeformerDataset, SUBJECTS, N_EPOCHS, make_continuation_split
from src.timeformer.models import build_model, DEFAULT_HPARAMS
from src.timeformer.memory import PrototypeMemory
from src.timeformer.train import MLMTrainer, load_checkpoint
from src.timeformer.eval import Evaluator, save_results
from src.timeformer.run import RunManager

CORPUS_PATH      = Path("data/corpus.tsv")
AMBIGUOUS_PATH   = Path("data/corpus_ambiguous.tsv")
CONTRASTIVE_PATH = Path("data/contrastive_set.tsv")


# ── Construção da Oracle Memory ───────────────────────────────────────────────

def build_oracle_memory(
    corpus_path: Path,
    d_model: int,
    proto_norm: float = 11.0,
    device: str | torch.device = "cpu",
) -> PrototypeMemory:
    """
    Cria uma PrototypeMemory onde m(S, t) codifica P(ctx=A | S, t) do corpus.

    Encoding: m(S, t) = p_a * v_A + (1 - p_a) * v_B
      v_A, v_B: vetores aleatórios fixos (seed=0), ortonormais, escalados para proto_norm
      p_a: fração de frases de treino com true_context=A para (S, t)

    Propriedade: cos(m(S,t), v_A) é monotônico em p_a — a cross-attention pode ler
    a trajetória de S projetando a query sobre a direção v_A.
    """
    rows = load_corpus(corpus_path)
    train_rows, _ = make_continuation_split(rows)  # exclui t8/t9

    n_subjects = len(SUBJECTS)
    subj2idx = {s: i for i, s in enumerate(SUBJECTS)}

    # Conta frases por (sujeito, época, contexto)
    counts_a = np.zeros((n_subjects, N_EPOCHS))
    counts_total = np.zeros((n_subjects, N_EPOCHS))
    for r in train_rows:
        s = subj2idx[r["subject"]]
        t = int(r["epoch"][1:])   # "t3" → 3
        counts_total[s, t] += 1
        if r["true_context"] == "N1":
            counts_a[s, t] += 1

    p_a = np.where(counts_total > 0, counts_a / counts_total, 0.5)  # (n_subjects, N_EPOCHS)

    # Dois vetores ortonormais fixos como base da codificação
    rng = np.random.default_rng(0)
    raw_A = rng.standard_normal(d_model).astype(np.float32)
    raw_B = rng.standard_normal(d_model).astype(np.float32)
    raw_B -= raw_B.dot(raw_A) / (raw_A.dot(raw_A)) * raw_A   # ortogonalizar
    v_A = torch.tensor(raw_A / np.linalg.norm(raw_A) * proto_norm)
    v_B = torch.tensor(raw_B / np.linalg.norm(raw_B) * proto_norm)

    mem = PrototypeMemory(n_subjects, N_EPOCHS, d_model, device)
    for s in range(n_subjects):
        for t in range(N_EPOCHS):
            if counts_total[s, t] > 0:
                pa = float(p_a[s, t])
                mem._protos[s, t, :] = pa * v_A + (1 - pa) * v_B
                mem._valid[s, t] = True

    print(f"  Oracle memory: {mem._valid.sum().item()} / {n_subjects * N_EPOCHS} protótipos válidos")
    print(f"  p_a média por classe:")
    print(f"    estável    (S1-S10):  {p_a[:10].mean():.3f}")
    print(f"    deriva     (S11-S20): {p_a[10:20].mean():.3f}")
    print(f"    bifurcação (S21-S30): {p_a[20:].mean():.3f}")
    return mem


# ── Treino de Timeformer com oracle memory ───────────────────────────────────

class _OracleMLMTrainer(MLMTrainer):
    """
    MLMTrainer que usa uma OracleMemory fixada (não atualizada durante treino).
    Herda todo o loop de treino — sobrescreve apenas a atualização de memória.
    """

    def train(self, train_dataset, val_dataset=None, memory=None, **kwargs):
        # Chama o train padrão, mas suprime o update da memória (oracle é fixo)
        return super().train(train_dataset, val_dataset, memory=memory, **kwargs)

    def _update_memory(self, memory, loader, n_epochs):
        # Não atualiza — oracle memory é imutável
        pass


def _patch_trainer_no_memory_update(trainer: MLMTrainer) -> None:
    """Monkey-patch: impede que MLMTrainer sobrescreva a oracle memory."""
    import types

    def _patched_train(self, train_dataset, val_dataset=None, memory=None,
                       n_epochs=30, batch_size=64, lr=1e-3, seed=42):
        import time, torch, torch.nn as nn
        from torch.utils.data import DataLoader
        from src.timeformer.dataset import timeformer_collate_fn

        torch.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=timeformer_collate_fn,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) \
            if val_dataset is not None else None

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        history = []
        best_val_loss = float("inf")

        for epoch in range(n_epochs):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, optimizer, memory)
            val_loss   = self._eval_epoch(val_loader, memory) if val_loader else None
            scheduler.step()

            record = {"epoch": epoch, "train_loss": train_loss,
                      "val_loss": val_loss, "elapsed_s": round(time.time() - t0, 2)}
            history.append(record)
            self._log(record)

            monitor = val_loss if val_loss is not None else train_loss
            if monitor < best_val_loss:
                best_val_loss = monitor
                self._save_checkpoint("best.pt")

            # ← NÃO atualiza memória: oracle é imutável

        self._save_checkpoint("final.pt")
        self._save_history(history)
        return history

    trainer.train = types.MethodType(_patched_train, trainer)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle Memory Diagnostic para Timeformer")
    parser.add_argument("--run-id",     type=str, default=None)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    if args.run_id:
        ref_run = RunManager.load(args.run_id)
    else:
        ref_run = RunManager.load_latest()

    print(f"Run de referência: {ref_run.run_id}")
    print(f"\n=== Construindo Oracle Memory ===")
    oracle_mem = build_oracle_memory(
        CORPUS_PATH, d_model=DEFAULT_HPARAMS["d_model"], device=args.device
    )

    # Cria nova run para o Timeformer-oracle
    config = {
        "experiment": "oracle_diagnostic",
        "ref_run": ref_run.run_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        **DEFAULT_HPARAMS,
    }
    oracle_run = RunManager()
    oracle_run.setup(config)
    print(f"Oracle run: {oracle_run.run_id}")

    print(f"\n=== Treinando Timeformer-oracle ({args.epochs} épocas) ===")
    rows       = load_corpus(CORPUS_PATH)
    train_rows, _ = make_continuation_split(rows)
    val_rows      = [r for r in rows if r["split"] == "test"]

    train_ds = TimeformerDataset(train_rows, seed=args.seed)
    val_ds   = MLMDataset(val_rows, seed=args.seed)

    model_oracle = build_model("Timeformer")
    out_dir = oracle_run.model_dir("Timeformer_oracle")
    trainer = MLMTrainer(model_oracle, output_dir=out_dir, device=args.device)
    _patch_trainer_no_memory_update(trainer)

    history = trainer.train(
        train_ds, val_ds,
        memory=oracle_mem,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    best_val = min(r["val_loss"] for r in history if r["val_loss"] is not None)
    print(f"\nTimeformer-oracle: best_val={best_val:.4f}")

    # Gate aprendido
    gate_val = model_oracle.temp_attn.gate.item()
    print(f"Gate aprendido: {gate_val:.6f}  tanh={torch.tanh(torch.tensor(gate_val)).item():.6f}")

    print(f"\n=== Avaliando ===")
    evaluator = Evaluator(
        corpus_path=CORPUS_PATH,
        ambiguous_path=AMBIGUOUS_PATH,
        contrastive_path=CONTRASTIVE_PATH,
        device=args.device,
    )

    results = {}

    # Timeformer-oracle (checkpoint do treino com oracle memory)
    load_checkpoint(model_oracle, out_dir / "best.pt")
    results["Timeformer_oracle"] = evaluator.evaluate(model_oracle, memory=oracle_mem)

    # Joint de referência (para comparação direta)
    joint_ckpt = ref_run.checkpoint_path("Joint", "best")
    if joint_ckpt.exists():
        model_joint = build_model("Joint")
        load_checkpoint(model_joint, joint_ckpt)
        results["Joint_ref"] = evaluator.evaluate(model_joint, memory=None)

    # Timeformer-learned de referência (checkpoint treinado com protótipos aprendidos)
    tf_ckpt = ref_run.checkpoint_path("Timeformer", "best")
    if tf_ckpt.exists():
        model_tf = build_model("Timeformer")
        load_checkpoint(model_tf, tf_ckpt)
        learned_mem = ref_run.load_memory("Timeformer")
        if learned_mem:
            learned_mem.to(args.device)
        results["Timeformer_learned"] = evaluator.evaluate(model_tf, memory=learned_mem)

    # Sumário
    print(f"\n{'model':<20} {'test':<8} {'ambig':<8} {'cont':<8} {'sign_flip':<10}")
    print("-" * 55)
    for name, res in results.items():
        t   = res.get("test", {}).get("probe_subj", {}).get("accuracy", float("nan"))
        a   = res.get("ambiguous_test", {}).get("probe_subj", {}).get("accuracy", float("nan"))
        c   = res.get("continuation", {}).get("probe_subj", {}).get("accuracy", float("nan"))
        sfr = res.get("contrastive", {}).get("sign_flip_rate", float("nan"))
        print(f"{name:<20} {t:<8.3f} {a:<8.3f} {c:<8.3f} {sfr:<10.3f}")

    print(f"\nInterpretação:")
    joint_c  = results.get("Joint_ref", {}).get("continuation", {}).get("probe_subj", {}).get("accuracy", float("nan"))
    bor_c  = results.get("Timeformer_oracle", {}).get("continuation", {}).get("probe_subj", {}).get("accuracy", float("nan"))
    bln_c  = results.get("Timeformer_learned", {}).get("continuation", {}).get("probe_subj", {}).get("accuracy", float("nan"))
    delta_oracle  = bor_c - joint_c
    delta_learned = bln_c - joint_c

    print(f"  Δ oracle  (Timeformer-oracle  − Joint): {delta_oracle:+.4f}")
    print(f"  Δ learned (Timeformer-learned − Joint): {delta_learned:+.4f}")

    if delta_oracle > 0.01:
        print("  → Arquitetura FUNCIONA com oracle; problema são os protótipos aprendidos")
    elif delta_oracle > -0.01:
        print("  → Resultado ambíguo — diferença dentro da margem de ruído")
    else:
        print("  → MLM/SVO não beneficia de memória histórica independente da qualidade")

    save_results(results, oracle_run.results_dir())
    print(f"\nResultados salvos em {oracle_run.results_dir()}/")


if __name__ == "__main__":
    main()
