"""
Diagnóstico da PrototypeMemory do B3.

Métricas calculadas:
  1. Separabilidade inter vs intra-sujeito por época
  2. Separabilidade inter vs intra-sujeito global (todos os protótipos)
  3. Clustering de sujeitos: silhouette, ARI, NMI (K-Means com k=30)
  4. Probe linear: acurácia de classificação de sujeito a partir dos protótipos
  5. Probe linear: acurácia de classificação de classe (estável/deriva/bifurcação)
  6. Norma média dos protótipos por época (deteta se ficam zerados)
  7. Overlap entre protótipos do sujeito correto vs shuffled

Uso:
  python scripts/diagnose_prototypes.py                   # run mais recente
  python scripts/diagnose_prototypes.py --run-id 20260523_003
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score,
    normalized_mutual_info_score, silhouette_score,
)
from sklearn.cluster import KMeans

from src.timeformer.run import RunManager
from src.timeformer.memory import PrototypeMemory

# Classes de sujeito (índices 0-indexados no corpus, S1=idx0 ... S30=idx29)
# Estável:    S1–S10  → idx 0–9
# Deriva:     S11–S20 → idx 10–19
# Bifurcação: S21–S30 → idx 20–29
STABLE_IDX     = list(range(0,  10))
DRIFT_IDX      = list(range(10, 20))
BIFURC_IDX     = list(range(20, 30))
CLASS_LABELS   = (["estável"] * 10 + ["deriva"] * 10 + ["bifurcação"] * 10)
CLASS_INT      = [0] * 10 + [1] * 10 + [2] * 10   # para sklearn


def load_memory(run: RunManager) -> PrototypeMemory:
    mem = run.load_memory("B3")
    if mem is None:
        raise FileNotFoundError(f"memory.pkl não encontrado em {run.run_dir}/B3/")
    return mem


def proto_matrix(mem: PrototypeMemory) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna (protos, valid) como arrays numpy.
    protos: (n_subjects, n_epochs, d_model)
    valid:  (n_subjects, n_epochs) bool
    """
    return mem._protos.cpu().numpy(), mem._valid.cpu().numpy()


# ── 1. Separabilidade por época ───────────────────────────────────────────────

def separability_per_epoch(protos: np.ndarray, valid: np.ndarray) -> None:
    """
    Para cada época t, calcula:
      - distância intra-sujeito: norma do protótipo (só há 1 proto por sujeito/época)
      - distância inter-sujeito: distância L2 média entre pares de sujeitos
      - razão inter/intra (quanto mais > 1, mais separados estão os sujeitos)
    """
    n_subjects, n_epochs, d = protos.shape
    print("\n─── 1. Separabilidade inter-sujeito por época ───────────────────────────────")
    print(f"{'época':<8} {'valid':<8} {'norma_média':<14} {'dist_inter':<14} {'frac_zero':<12}")
    print("-" * 56)

    for t in range(n_epochs):
        v = valid[:, t]              # (n_subjects,) bool
        ps = protos[v, t, :]         # (n_valid, d)
        n_valid = ps.shape[0]
        if n_valid < 2:
            print(f"  t{t:<5} {n_valid:<8} {'—':<14} {'—':<14} {'—':<12}")
            continue

        norms     = np.linalg.norm(ps, axis=-1)
        norma_med = norms.mean()
        frac_zero = (norms < 1e-6).mean()

        # Distância L2 média entre todos os pares
        diffs = ps[:, None, :] - ps[None, :, :]       # (n, n, d)
        dists = np.linalg.norm(diffs, axis=-1)         # (n, n)
        np.fill_diagonal(dists, np.nan)
        dist_inter = np.nanmean(dists)

        print(f"  t{t:<5} {n_valid:<8} {norma_med:<14.4f} {dist_inter:<14.4f} {frac_zero:<12.3f}")


# ── 2. Separabilidade global ──────────────────────────────────────────────────

def separability_global(protos: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Achata todos os protótipos válidos em uma matriz (N, d).
    Calcula dist intra-sujeito (variância ao longo das épocas) vs inter-sujeito.
    Retorna a matriz para uso nos testes seguintes.
    """
    n_subjects, n_epochs, d = protos.shape
    rows, labels_subj, labels_class = [], [], []

    for s in range(n_subjects):
        for t in range(n_epochs):
            if valid[s, t]:
                rows.append(protos[s, t, :])
                labels_subj.append(s)
                labels_class.append(CLASS_INT[s])

    X = np.array(rows)                 # (N, d)
    y_subj  = np.array(labels_subj)   # (N,) sujeito
    y_class = np.array(labels_class)  # (N,) classe

    print("\n─── 2. Separabilidade global (todos os protótipos válidos) ──────────────────")
    print(f"  Protótipos válidos: {len(X)}")

    # Intra-sujeito: variância média de h(S) ao longo das épocas
    intra_vars = []
    for s in range(n_subjects):
        ps = X[y_subj == s]
        if len(ps) >= 2:
            intra_vars.append(np.var(ps, axis=0).mean())
    intra = np.mean(intra_vars) if intra_vars else float("nan")

    # Inter-sujeito: variância entre os centróides de cada sujeito
    centroids = []
    for s in range(n_subjects):
        ps = X[y_subj == s]
        if len(ps) > 0:
            centroids.append(ps.mean(axis=0))
    centroids = np.array(centroids)
    inter = np.var(centroids, axis=0).mean() if len(centroids) > 1 else float("nan")

    ratio = inter / intra if intra > 0 else float("nan")
    print(f"  Variância intra-sujeito (média):    {intra:.6f}")
    print(f"  Variância inter-sujeito (centróides): {inter:.6f}")
    print(f"  Razão inter/intra:                  {ratio:.3f}  (>1 = sujeitos separados)")

    return X, y_subj, y_class


# ── 3. Clustering ─────────────────────────────────────────────────────────────

def clustering_diagnosis(X: np.ndarray, y_subj: np.ndarray, y_class: np.ndarray) -> None:
    print("\n─── 3. Clustering K-Means ────────────────────────────────────────────────────")

    # K = 30 sujeitos
    km30 = KMeans(n_clusters=30, random_state=42, n_init=10).fit(X)
    ari30  = adjusted_rand_score(y_subj, km30.labels_)
    nmi30  = normalized_mutual_info_score(y_subj, km30.labels_)
    sil30  = silhouette_score(X, y_subj, metric="euclidean") if len(np.unique(y_subj)) > 1 else float("nan")
    print(f"  k=30 (por sujeito):   ARI={ari30:.3f}  NMI={nmi30:.3f}  silhouette={sil30:.3f}")

    # K = 3 classes
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    ari3  = adjusted_rand_score(y_class, km3.labels_)
    nmi3  = normalized_mutual_info_score(y_class, km3.labels_)
    sil3  = silhouette_score(X, y_class, metric="euclidean") if len(np.unique(y_class)) > 1 else float("nan")
    print(f"  k=3  (por classe):    ARI={ari3:.3f}  NMI={nmi3:.3f}  silhouette={sil3:.3f}")
    print("  (ARI/NMI=1 = clustering perfeito; 0 = aleatório)")


# ── 4. Probe linear — sujeito ─────────────────────────────────────────────────

def probe_subject(X: np.ndarray, y_subj: np.ndarray) -> None:
    print("\n─── 4. Probe linear: classificação de sujeito (30 classes) ─────────────────")
    # Usa metade das épocas para treinar, outra metade para avaliar
    # (os protótipos têm até 10 épocas por sujeito)
    # Aqui usamos leave-one-epoch-out simplificado: treina em t0-t7, avalia em t8-t9
    n_subjects = 30
    n_epochs   = X.shape[0] // n_subjects if X.shape[0] >= n_subjects else 1

    # Recria split por época a partir de y_subj
    # X está organizado por (sujeito, época) — percorre na mesma ordem de protos válidos
    # Simples: treina LogReg com todos os dados (checar se separa sujeitos linearmente)
    if len(np.unique(y_subj)) < 2:
        print("  Apenas um sujeito — skip.")
        return

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(X, y_subj)
    pred = clf.predict(X)
    acc_train = accuracy_score(y_subj, pred)
    print(f"  Acc treino (fit em tudo): {acc_train:.3f}  (1.0 = protótipos são linearmente separáveis por sujeito)")

    # Cross-val simples: separa última época disponível para teste
    subj_rows: dict[int, list[int]] = {}
    for i, s in enumerate(y_subj):
        subj_rows.setdefault(s, []).append(i)

    X_train, y_train, X_test, y_test = [], [], [], []
    for s, idxs in subj_rows.items():
        if len(idxs) >= 2:
            X_train.extend(X[idxs[:-1]])
            y_train.extend([s] * (len(idxs) - 1))
            X_test.append(X[idxs[-1]])
            y_test.append(s)
        else:
            X_train.extend(X[idxs])
            y_train.extend([s] * len(idxs))

    if X_test:
        clf2 = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
        clf2.fit(np.array(X_train), np.array(y_train))
        pred2 = clf2.predict(np.array(X_test))
        acc_test = accuracy_score(np.array(y_test), pred2)
        print(f"  Acc leave-last-epoch-out: {acc_test:.3f}  (chance = 1/30 ≈ 0.033)")


# ── 5. Probe linear — classe ─────────────────────────────────────────────────

def probe_class(X: np.ndarray, y_class: np.ndarray) -> None:
    print("\n─── 5. Probe linear: classificação de classe (estável/deriva/bifurcação) ────")
    if len(np.unique(y_class)) < 2:
        print("  Apenas uma classe — skip.")
        return

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(X, y_class)
    pred = clf.predict(X)
    acc  = accuracy_score(y_class, pred)
    print(f"  Acc treino: {acc:.3f}  (chance = 0.333)")

    for cls_int, cls_name in enumerate(["estável", "deriva", "bifurcação"]):
        mask = y_class == cls_int
        if mask.sum() > 0:
            acc_cls = accuracy_score(y_class[mask], pred[mask])
            print(f"    {cls_name:<14}: {acc_cls:.3f}  (n={mask.sum()})")


# ── 6. Overlap shuffled vs real ───────────────────────────────────────────────

def shuffled_overlap(mem: PrototypeMemory) -> None:
    print("\n─── 6. Similaridade coseno: protótipo correto vs shuffled ──────────────────")
    protos = mem._protos.cpu().numpy()   # (n_subjects, n_epochs, d)
    valid  = mem._valid.cpu().numpy()

    sims_correct   = []   # sim(m(S,t), m(S,t)) — sempre 1, referência
    sims_shuffled  = []   # sim(m(S,t), m(S',t)) — sujeito errado, mesma época
    sims_random    = []   # sim(m(S,t), m(S',t')) — sujeito e época aleatórios

    rng = np.random.default_rng(42)
    n_subjects, n_epochs, d = protos.shape

    for s in range(n_subjects):
        for t in range(n_epochs):
            if not valid[s, t]:
                continue
            p = protos[s, t, :]
            norm_p = np.linalg.norm(p)
            if norm_p < 1e-8:
                continue

            # Shuffled: sujeito diferente, mesma época
            candidates = [s2 for s2 in range(n_subjects) if s2 != s and valid[s2, t]]
            if candidates:
                s2 = rng.choice(candidates)
                q  = protos[s2, t, :]
                norm_q = np.linalg.norm(q)
                if norm_q > 1e-8:
                    sims_shuffled.append(np.dot(p, q) / (norm_p * norm_q))

            # Random: sujeito e época aleatórios
            valid_pairs = [(s2, t2) for s2 in range(n_subjects) for t2 in range(n_epochs)
                           if (s2 != s or t2 != t) and valid[s2, t2]]
            if valid_pairs:
                s2, t2 = valid_pairs[rng.integers(len(valid_pairs))]
                q = protos[s2, t2, :]
                norm_q = np.linalg.norm(q)
                if norm_q > 1e-8:
                    sims_random.append(np.dot(p, q) / (norm_p * norm_q))

    def _stats(arr: list[float], label: str) -> None:
        if arr:
            a = np.array(arr)
            print(f"  {label:<35} média={a.mean():.4f}  std={a.std():.4f}  min={a.min():.4f}  max={a.max():.4f}")
        else:
            print(f"  {label:<35} (sem dados)")

    _stats(sims_shuffled, "sim(S, S' errado, mesma época t)")
    _stats(sims_random,   "sim(S, S' aleat, época aleat t')")
    print("  Interpretação: se as duas linhas são ≈iguais, os protótipos não")
    print("  distinguem sujeitos. Se shuffled >> random, há estrutura temporal")
    print("  mas não por identidade de sujeito.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico da PrototypeMemory do B3")
    parser.add_argument("--run-id", type=str, default=None,
                        help="ID da run (default: mais recente)")
    args = parser.parse_args()

    if args.run_id:
        run = RunManager.load(args.run_id)
    else:
        run = RunManager.load_latest()

    print(f"Run: {run.run_id}  |  {run.run_dir}/B3/memory.pkl")

    mem    = load_memory(run)
    protos, valid = proto_matrix(mem)

    print(f"\nProtótipos: {protos.shape}  (n_subjects × n_epochs × d_model)")
    print(f"Válidos:    {valid.sum()} / {valid.size}  ({100*valid.mean():.1f}%)")
    print(f"Por época:  {valid.sum(axis=0).tolist()}")

    separability_per_epoch(protos, valid)
    X, y_subj, y_class = separability_global(protos, valid)
    clustering_diagnosis(X, y_subj, y_class)
    probe_subject(X, y_subj)
    probe_class(X, y_class)
    shuffled_overlap(mem)

    print("\nDiagnóstico concluído.")


if __name__ == "__main__":
    main()
