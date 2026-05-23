"""
Alinhamento de Procrustes ortogonal para embeddings cross-epoch.

Problema: embeddings treinados independentemente por época podem estar
rotacionados arbitrariamente entre si (a solução do skip-gram é única
módulo rotação ortogonal). Isso significa que parte da distância cosseno
medida entre épocas é ruído de rotação, não deriva semântica real.

Solução: encontrar a rotação Q que melhor alinha E_tN ao espaço de E_t0:
  min_Q  ||E_t0 - E_tN @ Q||_F   sujeito a Q^T Q = I

Solução analítica via SVD: E_tN^T @ E_t0 = U S V^T  →  Q = V U^T

Usa o vocabulário completo como conjunto de âncoras. Âncoragem parcial
(só verbos/objetos estáveis) é matematicamente subdeterminada em 16D com
16 âncoras — os 32 graus de liberdade de SO(16) não ficam bem constrainados.
Com 19 tokens no espaço de 16D, a estimativa é robusta.

Referência: Procrustes Analysis em Word Vectors — Hamilton et al. (2016).
"""

import numpy as np
from pathlib import Path
from scipy.linalg import orthogonal_procrustes

REFERENCE_EPOCH = "t0"


def align_to_reference(
    reference: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Retorna `target` rotacionado para o sistema de referência de `reference`.

    `orthogonal_procrustes(A, B)` retorna R que minimiza ||A @ R - B||_F.
    Queremos R tal que `target @ R ≈ reference`, então chamamos com (target, reference).
    """
    R, _ = orthogonal_procrustes(target, reference)
    return target @ R


def align_all_epochs(
    all_embeddings: dict[str, np.ndarray],
    reference_epoch: str = REFERENCE_EPOCH,
) -> dict[str, np.ndarray]:
    """
    Alinha todos os espaços de embedding ao espaço de `reference_epoch`.
    A época de referência é retornada sem modificação.
    """
    reference = all_embeddings[reference_epoch]
    aligned: dict[str, np.ndarray] = {}
    for epoch, emb in all_embeddings.items():
        aligned[epoch] = emb.copy() if epoch == reference_epoch else align_to_reference(reference, emb)
    return aligned


def alignment_quality(
    all_embeddings: dict[str, np.ndarray],
    aligned_embeddings: dict[str, np.ndarray],
    reference_epoch: str = REFERENCE_EPOCH,
) -> None:
    """Imprime o erro Frobenius antes e depois do alinhamento por época."""
    ref = all_embeddings[reference_epoch]
    print(f"\nQualidade do alinhamento (||E_t0 - E_tN||_F):")
    print(f"  {'época':<6}  {'bruto':>8}  {'alinhado':>10}  {'redução':>8}")
    for ep, emb in sorted(all_embeddings.items()):
        if ep == reference_epoch:
            continue
        err_before = float(np.linalg.norm(ref - emb))
        err_after  = float(np.linalg.norm(ref - aligned_embeddings[ep]))
        reduction  = (err_before - err_after) / err_before * 100
        print(f"  {ep:<6}  {err_before:>8.4f}  {err_after:>10.4f}  {reduction:>7.1f}%")
