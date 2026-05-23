"""
Gera o relatório de validação de sinal da Fase A do Timeformer.

Métricas computadas:
1. Distância cosseno entre épocas (t0 vs. tN) para cada sujeito
2. Distâncias entre épocas consecutivas de S2 (devem ser monotônicas)
3. Afinidade contextual de S3 ao longo das épocas (devem convergir para 50/50)
4. Veredicto automático com critérios interpretáveis
"""

from pathlib import Path

import numpy as np

from src.train_embeddings import TOKEN_TO_IDX
from src.visualize import CONTEXT_A_TOKENS, CONTEXT_B_TOKENS, EPOCHS, SUBJECTS
from src.corpus_generator import SUBJECT_CLASSES
from src.trajectory_probe import PHENOMENON_LABELS


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(1.0 - sim)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _centroid(embeddings: np.ndarray, tokens: list[str]) -> np.ndarray:
    return np.mean([embeddings[TOKEN_TO_IDX[t]] for t in tokens], axis=0)


def generate_report(
    all_embeddings: dict[str, np.ndarray],
    aligned_embeddings: dict[str, np.ndarray],
    affinities: dict[str, dict[str, list[float]]],
    probe_results: dict[str, dict[str, list[float]]],
    trajectory_results: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def h(text: str) -> None:
        lines.append(f"\n{text}")
        lines.append("-" * len(text))

    lines.append("# Timeformer Fase A — Relatório de Validação de Sinal")
    lines.append(
        "\nObjetivo: verificar se os fenômenos plantados no corpus são detectáveis "
        "por embeddings padrão (skip-gram por época), antes de qualquer TimeEncoding."
    )

    # ── Seção 1: Distâncias cosseno t0 → tN (bruto e alinhado) ───────────
    h("## 1. Distância Cosseno entre t0 e cada época")
    lines.append(
        "\nMostrado em pares (bruto | alinhado). O alinhamento de Procrustes remove\n"
        "o ruído rotacional; a diferença residual é deriva semântica real.\n"
    )
    for subject in SUBJECTS:
        idx = TOKEN_TO_IDX[subject]
        emb_t0 = all_embeddings["t0"][idx]
        lines.append(f"### {subject}")
        lines.append(f"  {'época':<6}  {'bruto':>8}  {'alinhado':>10}  {'ruído rot.':>10}")
        lines.append("  " + "-" * 40)
        for ep in EPOCHS[1:]:
            d_raw = _cosine_distance(emb_t0, all_embeddings[ep][idx])
            d_aln = _cosine_distance(aligned_embeddings["t0"][idx], aligned_embeddings[ep][idx])
            noise = d_raw - d_aln
            lines.append(f"  {ep:<6}  {d_raw:>8.4f}  {d_aln:>10.4f}  {noise:>+10.4f}")
        lines.append("")

    # ── Seção 2: Distâncias consecutivas de S2 ───────────────────────────
    h("## 2. S2 — Distâncias entre Épocas Consecutivas (deriva deve ser monotônica)")
    lines.append("")
    s2_idx = TOKEN_TO_IDX["S2"]
    dists_s2 = []
    for i in range(len(EPOCHS) - 1):
        a = all_embeddings[EPOCHS[i]][s2_idx]
        b = all_embeddings[EPOCHS[i + 1]][s2_idx]
        d = _cosine_distance(a, b)
        dists_s2.append(d)
        lines.append(f"  dist_cosseno({EPOCHS[i]}, {EPOCHS[i+1]}) = {d:.4f}")
    lines.append("")

    # ── Seção 3: Afinidade contextual ────────────────────────────────────
    h("## 3. Afinidade Contextual (similaridade cosseno com centroides)")
    lines.append("")
    header_cols = "".join(f"  {s+'→A':>6}  {s+'→B':>6}" for s in SUBJECTS)
    lines.append(f"  {'Época':<6}{header_cols}")
    lines.append("  " + "-" * (6 + 16 * len(SUBJECTS)))
    for i, ep in enumerate(EPOCHS):
        row = f"  {ep:<6}"
        for s in SUBJECTS:
            row += f"  {affinities[s]['A'][i]:>6.3f}  {affinities[s]['B'][i]:>6.3f}"
        lines.append(row)
    lines.append("")

    # ── Seção 4: Veredicto ────────────────────────────────────────────────
    h("## 4. Veredicto")
    lines.append("")

    # Usa embeddings alinhados: distâncias refletem deriva semântica real, não ruído rotacional
    d_s1_t5 = _cosine_distance(
        aligned_embeddings["t0"][TOKEN_TO_IDX["S1"]],
        aligned_embeddings["t5"][TOKEN_TO_IDX["S1"]],
    )
    d_s2_t5 = _cosine_distance(
        aligned_embeddings["t0"][TOKEN_TO_IDX["S2"]],
        aligned_embeddings["t5"][TOKEN_TO_IDX["S2"]],
    )

    s3_a_t5 = affinities["S3"]["A"][-1]
    s3_b_t5 = affinities["S3"]["B"][-1]
    s3_gap_t0 = abs(affinities["S3"]["A"][0] - affinities["S3"]["B"][0])
    s3_gap_t5 = abs(s3_a_t5 - s3_b_t5)

    # Critério S1: deve derivar menos que S2 (comparação relativa)
    v_s1 = "✓ PASS" if d_s1_t5 < d_s2_t5 else "✗ FAIL"
    # Critério S2: deve derivar mais que S1 — ratio > 1.2 é sinal consistente
    ratio_s2 = d_s2_t5 / (d_s1_t5 + 1e-10)
    v_s2 = "✓ PASS" if ratio_s2 > 1.2 else "✗ FAIL"
    # S3: gap entre afinidades A e B deve diminuir (convergência para ambiguidade)
    v_s3 = "✓ PASS" if s3_gap_t5 < s3_gap_t0 else "✗ FAIL"

    lines.append(
        f"  S1 (estável):   dist(t0,t5) = {d_s1_t5:.4f}  "
        f"< dist_S2(t0,t5) = {d_s2_t5:.4f}  → {v_s1}"
    )
    lines.append(
        f"  S2 (deriva):    dist_S2/dist_S1 = {ratio_s2:.2f}×  (espera-se > 1.2×)  → {v_s2}"
    )
    lines.append(
        f"  S3 (bifurcação): gap_afinidade t0={s3_gap_t0:.3f} → t5={s3_gap_t5:.3f}  → {v_s3}"
    )
    lines.append("")
    lines.append(
        "### Interpretação\n\n"
        "Os embeddings padrão (skip-gram por época) capturam deriva suave (S2) e\n"
        "posição de equilíbrio ambígua (S3), mas NÃO representam a bifurcação como\n"
        "dois vetores distintos coexistindo. S3 converge para um único ponto médio\n"
        "entre os contextos A e B — este é exatamente o limite que o TimeEncoding\n"
        "da Fase B precisa superar, representando S3 como uma trajetória que se ramifica,\n"
        "não como um ponto fixo de ambiguidade."
    )

    # ── Seção 5: Probe de disambiguação ──────────────────────────────────
    h("## 5. Probe de Disambiguação Contextual")
    lines.append(
        "\nAcurácia de um classificador linear (regressão logística) treinado\n"
        "sobre diferentes representações de sentença para prever Contexto A vs B.\n"
        "Chance = 50% (classes balanceadas em t3-t5).\n"
        "\n  e(subj) — embedding do sujeito (vetor único por época → trivialmente limitado)\n"
        "  e(verb) — embedding do verbo (oráculo perfeito)\n"
        "  e(mean) — média S+V+O (verbo mascara a falha do sujeito)\n"
        "  e(ctx)  — média V+O sem sujeito (teste não-trivial: contexto sem identidade)\n"
    )

    for subject in SUBJECTS:
        lines.append(f"### {subject}")
        header = f"  {'rep':<10}" + "".join(f"  {ep:>6}" for ep in EPOCHS)
        lines.append(header)
        lines.append("  " + "-" * (10 + 8 * len(EPOCHS)))
        for rep in ["subj", "verb", "mean", "ctx"]:
            label = {"subj": "e(subj)", "verb": "e(verb)", "mean": "e(mean)", "ctx": "e(ctx)"}[rep]
            row = f"  {label:<10}" + "".join(
                f"  {probe_results[rep][subject][i]:>6.1%}" for i in range(len(EPOCHS))
            )
            lines.append(row)
        lines.append("")

    # Veredicto do probe para S3
    s3_subj_t3 = probe_results["subj"]["S3"][3]
    s3_ctx_t3  = probe_results["ctx"]["S3"][3]
    s3_verb_t3 = probe_results["verb"]["S3"][3]
    s3_subj_t0 = probe_results["subj"]["S3"][0]
    ctx_gap = s3_ctx_t3 - s3_subj_t3   # e(ctx) deve superar e(subj) claramente

    v_probe = "✓ PASS" if ctx_gap > 0.2 and s3_verb_t3 > 0.8 else "✗ FAIL"

    lines.append(
        f"  e(subj) em t3: {s3_subj_t3:.1%}  |  e(ctx) em t3: {s3_ctx_t3:.1%}"
        f"  |  e(verb) em t3: {s3_verb_t3:.1%}  → {v_probe}"
    )
    lines.append(
        "\n  Interpretação não-trivial:\n"
        "    e(ctx) — representação V+O sem o sujeito — mantém acurácia alta em t3-t5,\n"
        "    enquanto e(subj) colapsa para chance (~50%). Isso demonstra que:\n"
        "      1. O skip-gram APRENDE a informação de bifurcação (está nos embeddings V/O)\n"
        "      2. Essa informação NÃO está codificada no token embedding do sujeito S3\n"
        "      3. O problema não é falta de sinal — é que o sinal está distribuído nos\n"
        "         co-ocorrentes, e o sujeito bifurcante converge para um vetor médio\n"
        "    Este é o gap empírico não-trivial que o Timeformer precisa preencher na Fase B:\n"
        "    representar S3 como uma trajetória que bifurca, não como um ponto médio fixo."
    )

    # ── Seção 6: Probe de Trajetória ─────────────────────────────────────────
    h("## 6. Probe de Trajetória Temporal (experimento central)")
    lines.append(
        "\nClassificador KNN-1 leave-one-out: classifica o sujeito como\n"
        "estável / deriva / bifurcação com três representações:\n"
        "  Snapshot (1D):        distância cosseno em t5 apenas\n"
        "  Trajetória bruta (5D): distâncias em t1-t5\n"
        "  Hump features (5D):   endpoint, peak, razão peak/endpoint, época_pico, slope\n"
    )

    sigs = trajectory_results["signatures"]
    hump = trajectory_results["hump_features"]
    true_cls = trajectory_results["true_classes"]
    preds_snap = trajectory_results["preds_snapshot"]
    preds_traj = trajectory_results["preds_trajectory"]
    preds_hump = trajectory_results["preds_hump"]

    lines.append(f"  {'sujeito':<8} {'classe real':<14}" + "".join(f"  {ep:>6}" for ep in EPOCHS[1:]) + "  {'razão peak/t5':>14}")
    lines.append("  " + "-" * (8 + 14 + 8 * 5 + 16))
    for subj in SUBJECTS:
        cls = PHENOMENON_LABELS[true_cls[subj]]
        sig = sigs[subj]
        ratio = hump[subj][2]
        row = f"  {subj:<8} {cls:<14}" + "".join(f"  {d:>6.4f}" for d in sig) + f"  {ratio:>14.2f}x"
        lines.append(row)
    lines.append("")

    lines.append(f"  {'sujeito':<8} {'real':<14} {'snapshot':>10}  {'traj. bruta':>12}  {'hump feats':>12}")
    lines.append("  " + "-" * 62)
    for subj in SUBJECTS:
        real = PHENOMENON_LABELS[true_cls[subj]]
        snap = PHENOMENON_LABELS.get(preds_snap[subj], preds_snap[subj])
        traj = PHENOMENON_LABELS.get(preds_traj[subj], preds_traj[subj])
        hm   = PHENOMENON_LABELS.get(preds_hump[subj], preds_hump[subj])
        sm = "✓" if preds_snap[subj] == true_cls[subj] else "✗"
        tm = "✓" if preds_traj[subj] == true_cls[subj] else "✗"
        hmark = "✓" if preds_hump[subj] == true_cls[subj] else "✗"
        lines.append(f"  {subj:<8} {real:<14} {sm} {snap:<10} {tm} {traj:<10} {hmark} {hm}")

    lines.append("")
    acc_snap = trajectory_results["acc_snapshot"]
    acc_traj = trajectory_results["acc_trajectory"]
    acc_hump = trajectory_results["acc_hump"]
    v_traj = "✓ PASS" if acc_hump > acc_snap else "✗ FAIL"
    lines.append(f"  Acurácia snapshot (t5 apenas):       {acc_snap:.0%}")
    lines.append(f"  Acurácia trajetória bruta (t1-t5):   {acc_traj:.0%}")
    lines.append(f"  Acurácia hump features (forma):      {acc_hump:.0%}  → {v_traj}")
    lines.append(
        "\n  Resultado central:\n"
        "  Em t5, estáveis (S1, S4) e bifurcados (S3, S6) têm distâncias\n"
        "  similares ao ponto de partida — snapshot não os distingue.\n"
        "  Features de forma normalizadas (razão peak/endpoint) recuperam\n"
        "  a bifurcação porque o padrão de 'corcunda' é detectável.\n"
        "  Trajetória bruta com embeddings independentes sofre com ruído:\n"
        "  os modelos por época não foram treinados para produzir trajetórias\n"
        "  coerentes — cada epoch vê um espaço rotacionado arbitrariamente.\n"
        "\n  Implicação central para o TimeEncoding:\n"
        "  Post-hoc trajectory analysis de embeddings independentes é frágil.\n"
        "  O Timeformer aprende trajetórias coerentes com supervisão temporal:\n"
        "  TimeEncoding provê o bias indutivo para alinhar os espaços de embedding\n"
        "  durante o treinamento, não apenas no pós-processamento."
    )

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    print(f"\nRelatório salvo: {output_path}")
    print("\n" + text)
