"""
Probe de trajetória temporal — experimento central da Fase A do Timeformer.

Motivação arquitetural:
  Um snapshot estático (embedding em uma única época) não consegue distinguir
  um token ESTÁVEL de um token BIFURCADO quando ambos terminam numa posição
  "próxima à origem" no espaço alinhado. A bifurcação produz um embedding médio
  estabilizado — visualmente similar a estabilidade.

  A trajetória temporal (sequência de embeddings ao longo das épocas) tem uma
  forma característica para cada fenômeno:
    Estável:     distâncias de t0 consistentemente baixas (padrão plano)
    Deriva:      distâncias crescentes de forma monótona (padrão ascendente)
    Bifurcação:  distâncias sobem e depois descem — o embedding avança em direção
                 ao ponto médio e se estabiliza ali (padrão "corcunda")

  Com 2 sujeitos por classe (S1/S4=estável, S2/S5=deriva, S3/S6=bifurcação),
  demonstramos que:
    1. Snapshot único em t5: estável e bifurcação são indistinguíveis
    2. Trajetória completa (t1-t5): todas as classes são separáveis por KNN-1

  Este resultado justifica o TimeEncoding: o modelo precisa aprender o padrão
  temporal da trajetória, não apenas a posição estática do embedding.
"""

import numpy as np

from src.train_embeddings import TOKEN_TO_IDX, EPOCHS_ORDER
from src.corpus_generator import SUBJECTS, SUBJECT_CLASSES

EPOCHS = EPOCHS_ORDER  # ["t0", "t1", ..., "t5"]

PHENOMENON_LABELS = {"stable": "Estável", "drift": "Deriva", "bifurcation": "Bifurcação"}


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_trajectory_signatures(
    aligned_embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Para cada sujeito, computa a assinatura de trajetória:
    vetor de 5 distâncias cosseno do embedding em t1..t5 em relação a t0.

    Retorna dict[subject] -> np.array(5,)
    """
    signatures: dict[str, np.ndarray] = {}
    for subject in SUBJECTS:
        idx = TOKEN_TO_IDX[subject]
        emb_t0 = aligned_embeddings["t0"][idx]
        dists = np.array([
            _cosine_distance(emb_t0, aligned_embeddings[ep][idx])
            for ep in EPOCHS[1:]
        ])
        signatures[subject] = dists
    return signatures


def _hump_features(signatures: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Extrai features de forma de trajetória invariantes à escala.

    Para cada sujeito, computa:
      feat[0] = distância em t5 (endpoint)
      feat[1] = distância máxima na trajetória (peak)
      feat[2] = razão peak/endpoint  (>1.5 = corcunda → bifurcação)
      feat[3] = época do pico (normalizada: 0=t1, 1=t5)
      feat[4] = slope final: (endpoint - valor em t3) / endpoint  (derivada tardia)

    Com esta representação:
      Estável:     endpoint baixo, peak ≈ endpoint, slope ≈ 0
      Deriva:      endpoint alto, peak ≈ endpoint (monotônico), slope positivo
      Bifurcação:  peak >> endpoint (corcunda clara), slope negativo
    """
    feats: dict[str, np.ndarray] = {}
    for subj, sig in signatures.items():
        endpoint = sig[-1]
        peak = float(np.max(sig))
        peak_epoch = float(np.argmax(sig)) / (len(sig) - 1)
        hump_ratio = peak / (endpoint + 1e-10)
        slope_final = (endpoint - sig[-3]) / (endpoint + 1e-10)
        feats[subj] = np.array([endpoint, peak, hump_ratio, peak_epoch, slope_final])
    return feats


def _knn1_loo(
    signatures: dict[str, np.ndarray],
    mode: str = "raw",
) -> dict[str, str]:
    """
    Classificador KNN-1 leave-one-out.

    mode="raw"       → 5D distâncias absolutas (t1-t5)
    mode="snapshot"  → 1D distância em t5 apenas
    mode="hump"      → 5D features de forma (endpoint, peak, razão, etc.)
    """
    subjects = list(signatures.keys())
    if mode == "hump":
        features = _hump_features(signatures)
    elif mode == "snapshot":
        features = {s: sig[-1:] for s, sig in signatures.items()}
    else:
        features = signatures

    predictions: dict[str, str] = {}
    for test_subj in subjects:
        best_dist = float("inf")
        best_class = ""
        for train_subj in subjects:
            if train_subj == test_subj:
                continue
            d = float(np.linalg.norm(features[test_subj] - features[train_subj]))
            if d < best_dist:
                best_dist = d
                best_class = SUBJECT_CLASSES[train_subj]
        predictions[test_subj] = best_class
    return predictions


def run_trajectory_probe(
    aligned_embeddings: dict[str, np.ndarray],
) -> dict:
    """
    Executa o probe de trajetória e retorna resultados estruturados.

    Compara três classificadores KNN-1 leave-one-out:
      - Snapshot (1D): usa apenas a distância cosseno em t5
      - Trajetória bruta (5D): usa as distâncias em t1-t5
      - Hump features (5D): endpoint, peak, razão peak/endpoint, época do pico, slope

    Resultado esperado:
      Snapshot: confunde estável com bifurcação (ambos terminam perto de t0)
      Trajetória bruta: melhora sobre snapshot (padrão temporal ajuda)
      Hump features: melhor resultado (forma normalizada separa os três padrões)
    """
    signatures = compute_trajectory_signatures(aligned_embeddings)
    true_classes = {s: SUBJECT_CLASSES[s] for s in SUBJECTS}

    preds_snapshot   = _knn1_loo(signatures, mode="snapshot")
    preds_trajectory = _knn1_loo(signatures, mode="raw")
    preds_hump       = _knn1_loo(signatures, mode="hump")

    def acc(preds):
        return sum(preds[s] == true_classes[s] for s in SUBJECTS) / len(SUBJECTS)

    hump_feats = _hump_features(signatures)

    return {
        "signatures": signatures,
        "hump_features": hump_feats,
        "true_classes": true_classes,
        "preds_snapshot":   preds_snapshot,
        "preds_trajectory": preds_trajectory,
        "preds_hump":       preds_hump,
        "acc_snapshot":     acc(preds_snapshot),
        "acc_trajectory":   acc(preds_trajectory),
        "acc_hump":         acc(preds_hump),
    }


def print_trajectory_results(results: dict) -> None:
    signatures = results["signatures"]
    hump_feats = results["hump_features"]
    true_classes = results["true_classes"]
    preds_snapshot   = results["preds_snapshot"]
    preds_trajectory = results["preds_trajectory"]
    preds_hump       = results["preds_hump"]

    print("\nAssinaturas de trajetória (distância cosseno de t0 em cada época):")
    header = f"  {'sujeito':<8} {'classe':<13}" + "".join(f"  {ep:>6}" for ep in EPOCHS[1:])
    print(header)
    print("  " + "-" * (8 + 13 + 8 * 5))
    for subj in SUBJECTS:
        cls = PHENOMENON_LABELS[true_classes[subj]]
        sig = signatures[subj]
        row = f"  {subj:<8} {cls:<13}" + "".join(f"  {d:>6.4f}" for d in sig)
        print(row)

    print("\nFeatures de forma (hump): endpoint | peak | razão peak/endpoint | época_pico | slope_final")
    print(f"  {'sujeito':<8} {'classe':<13} {'endpoint':>9} {'peak':>7} {'razão':>7} {'t_pico':>7} {'slope':>7}")
    print("  " + "-" * 58)
    for subj in SUBJECTS:
        cls = PHENOMENON_LABELS[true_classes[subj]]
        f = hump_feats[subj]
        print(f"  {subj:<8} {cls:<13} {f[0]:>9.4f} {f[1]:>7.4f} {f[2]:>7.2f}x {f[3]:>6.2f} {f[4]:>+7.2f}")

    print(f"\nClassificação KNN-1 leave-one-out:")
    print(f"  {'sujeito':<8} {'real':<13} {'snapshot':>10}  {'traj. bruta':>12}  {'hump feats':>12}")
    print("  " + "-" * 62)
    for subj in SUBJECTS:
        real = PHENOMENON_LABELS[true_classes[subj]]
        snap = PHENOMENON_LABELS.get(preds_snapshot[subj],   preds_snapshot[subj])
        traj = PHENOMENON_LABELS.get(preds_trajectory[subj], preds_trajectory[subj])
        hump = PHENOMENON_LABELS.get(preds_hump[subj],       preds_hump[subj])
        sm = "✓" if preds_snapshot[subj]   == true_classes[subj] else "✗"
        tm = "✓" if preds_trajectory[subj] == true_classes[subj] else "✗"
        hm = "✓" if preds_hump[subj]       == true_classes[subj] else "✗"
        print(f"  {subj:<8} {real:<13} {sm} {snap:<10} {tm} {traj:<10} {hm} {hump}")

    print(f"\n  Acurácia snapshot (t5 apenas):       {results['acc_snapshot']:.0%}")
    print(f"  Acurácia trajetória bruta (t1-t5):   {results['acc_trajectory']:.0%}")
    print(f"  Acurácia hump features (forma):      {results['acc_hump']:.0%}")
