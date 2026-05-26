"""
Visualização dos embeddings da Fase A do Timeformer.

Dois tipos de gráfico:
1. Trajetórias PCA 2D — projeta todos os embeddings de S1/S2/S3 num espaço
   comum e plota o caminho percorrido ao longo das épocas.
   Caveat: embeddings de épocas diferentes podem ter rotações distintas;
   a projeção PCA conjunta minimiza esse problema mas não elimina.

2. Afinidade contextual — métrica invariante a rotação. Para cada sujeito
   e época, mede a similaridade cosseno com os centroides dos Contextos A e B.
   Este gráfico valida diretamente os fenômenos plantados.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import PCA

from src.train_embeddings import TOKEN_TO_IDX, VOCAB

EPOCHS = [f"t{i}" for i in range(6)]
SUBJECTS = ["S1", "S2", "S3", "S4", "S5", "S6"]

# Tokens que definem cada vizinhança semântica
NEIGH_1_TOKENS = ["V1", "V2", "V3", "V4", "O1", "O2", "O3", "O4"]
NEIGH_2_TOKENS = ["V5", "V6", "V7", "V8", "O5", "O6", "O7", "O8"]

# Backward-compatible aliases
CONTEXT_A_TOKENS = NEIGH_1_TOKENS
CONTEXT_B_TOKENS = NEIGH_2_TOKENS

SUBJECT_PALETTES = {
    "S1": "Blues", "S4": "Blues",
    "S2": "Greens", "S5": "Greens",
    "S3": "Reds", "S6": "Reds",
}
SUBJECT_COLORS = {
    "S1": "#1565C0", "S4": "#42A5F5",
    "S2": "#2E7D32", "S5": "#66BB6A",
    "S3": "#C62828", "S6": "#EF9A9A",
}
SUBJECT_LABELS = {
    "S1": "S1 — Estável",
    "S4": "S4 — Estável (variante)",
    "S2": "S2 — Deriva (rápida)",
    "S5": "S5 — Deriva (lenta)",
    "S3": "S3 — Bifurcação (início t1)",
    "S6": "S6 — Bifurcação (início t3)",
}

PHENOMENON_COLORS = {"stable": "#1565C0", "drift": "#2E7D32", "bifurcation": "#C62828"}


def _get_subject_trajectories(
    all_embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Retorna dict[sujeito] → array (num_epochs, embedding_dim)."""
    return {
        s: np.stack([all_embeddings[ep][TOKEN_TO_IDX[s]] for ep in EPOCHS])
        for s in SUBJECTS
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _context_centroid(embeddings: np.ndarray, token_list: list[str]) -> np.ndarray:
    return np.mean([embeddings[TOKEN_TO_IDX[t]] for t in token_list], axis=0)


# ─── Plot 1: Trajetórias PCA ───────────────────────────────────────────────

def plot_pca_trajectories(
    all_embeddings: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trajs = _get_subject_trajectories(all_embeddings)

    # PCA ajustado sobre todos os sujeitos em todas as épocas → espaço comum
    all_vecs = np.vstack([trajs[s] for s in SUBJECTS])
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(all_vecs)
    var_exp = pca.explained_variance_ratio_

    # ── Subplots individuais por sujeito ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Timeformer Fase A — Trajetórias de Embeddings (PCA 2D)", fontsize=13)

    for ax, subject in zip(axes, SUBJECTS):
        proj = pca.transform(trajs[subject])  # (6, 2)
        cmap = cm.get_cmap(SUBJECT_PALETTES[subject])
        colors = [cmap(0.35 + 0.65 * i / 5) for i in range(6)]

        # Setas entre épocas consecutivas
        for i in range(len(EPOCHS) - 1):
            ax.annotate(
                "",
                xy=proj[i + 1],
                xytext=proj[i],
                arrowprops=dict(arrowstyle="->", color=colors[i + 1], lw=1.8),
            )

        # Pontos e labels de época
        for i, (x, y) in enumerate(proj):
            ax.scatter(x, y, color=colors[i], s=110, zorder=5, edgecolors="white", linewidths=0.5)
            ax.annotate(
                EPOCHS[i],
                (x, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                color=colors[i],
            )

        ax.set_title(SUBJECT_LABELS[subject], fontsize=10)
        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = output_dir / "trajectories_individual.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {path}")

    # ── Gráfico combinado ──
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("S1 / S2 / S3 — Trajetórias Combinadas (PCA 2D)", fontsize=12)

    for subject in SUBJECTS:
        proj = pca.transform(trajs[subject])
        color = SUBJECT_COLORS[subject]
        ax.plot(proj[:, 0], proj[:, 1], "-o", color=color,
                label=SUBJECT_LABELS[subject], linewidth=2, markersize=9)
        for i, (x, y) in enumerate(proj):
            ax.annotate(
                f"{EPOCHS[i]}",
                (x, y),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
                color=color,
            )

    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = output_dir / "trajectories_combined.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {path}")


# ─── Plot 2: Probabilidades de Predição por Contexto ──────────────────────

def plot_context_affinity(
    all_embeddings: dict[str, np.ndarray],
    output_dir: Path,
    all_models: dict | None = None,
) -> dict[str, dict[str, list[float]]]:
    """
    Para cada sujeito e época, calcula P(Contexto A) e P(Contexto B).

    Usa probabilidades de predição do modelo treinado (softmax sobre o
    vocabulário) quando `all_models` é fornecido. Essa métrica é mais
    fiel ao que o skip-gram aprendeu do que similaridade cosseno entre
    embeddings de entrada, porque o skip-gram aprende
    e(sujeito) · w_out(token), não e(sujeito) · e(token).

    Retorna dict[subject][context] → lista de probabilidades por época.
    """
    from src.train_embeddings import compute_context_probs

    output_dir.mkdir(parents=True, exist_ok=True)

    affinities: dict[str, dict[str, list[float]]] = {
        s: {"N1": [], "N2": []} for s in SUBJECTS
    }

    for ep in EPOCHS:
        if all_models and ep in all_models:
            probs = compute_context_probs(all_models[ep])
            for subject in SUBJECTS:
                affinities[subject]["N1"].append(probs[subject][0])
                affinities[subject]["N2"].append(probs[subject][1])
        else:
            # Fallback: similaridade cosseno com centroides (menos preciso)
            emb = all_embeddings[ep]
            centroid_a = _context_centroid(emb, NEIGH_1_TOKENS)
            centroid_b = _context_centroid(emb, NEIGH_2_TOKENS)
            for subject in SUBJECTS:
                vec = emb[TOKEN_TO_IDX[subject]]
                affinities[subject]["N1"].append(_cosine_similarity(vec, centroid_a))
                affinities[subject]["N2"].append(_cosine_similarity(vec, centroid_b))

    using_probs = all_models is not None
    ylabel = "P(contexto)" if using_probs else "Similaridade cosseno"
    y_lim = (0.0, 0.55) if using_probs else (-0.1, 1.05)
    subtitle = (
        "probabilidade de predição P(Contexto A/B) pelo modelo treinado"
        if using_probs
        else "similaridade cosseno com centroides dos contextos"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        f"Timeformer Fase A — Afinidade Contextual por Época\n({subtitle})",
        fontsize=12,
    )
    x = list(range(len(EPOCHS)))

    for ax, subject in zip(axes, SUBJECTS):
        aff_a = affinities[subject]["N1"]
        aff_b = affinities[subject]["N2"]
        ax.plot(x, aff_a, "o-", color="#1565C0", label="N1", linewidth=2, markersize=8)
        ax.plot(x, aff_b, "s--", color="#C62828", label="N2", linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(EPOCHS)
        ax.set_title(SUBJECT_LABELS[subject], fontsize=10)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel if subject == "S1" else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(*y_lim)

    plt.tight_layout()
    path = output_dir / "context_affinity.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {path}")

    return affinities


# ─── Plot 3: Probe de Disambiguação ───────────────────────────────────────

def plot_probe_results(
    probe_results: dict[str, dict[str, list[float]]],
    output_dir: Path,
) -> None:
    """
    Plota acurácia do probe de disambiguação por época.

    Layout: 3 subplots (S1, S2, S3), cada um com 3 linhas (e(subj), e(verb), e(mean)).
    A linha e(subj) de S3 é o elemento central: deve colapsar para ~50% em t3-t5.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    REP_STYLES = {
        "subj": {"color": "#C62828", "ls": "-",  "marker": "o", "lw": 2.5,
                 "label": "e(subj) — sujeito isolado"},
        "verb": {"color": "#1565C0", "ls": "--", "marker": "s", "lw": 1.8,
                 "label": "e(verb) — oráculo (verbo)"},
        "mean": {"color": "#558B2F", "ls": ":",  "marker": "^", "lw": 1.8,
                 "label": "e(mean) — média S+V+O"},
        "ctx":  {"color": "#E65100", "ls": "-.", "marker": "D", "lw": 2.0,
                 "label": "e(ctx) — verbo+objeto (sem sujeito)"},
    }

    x = list(range(len(EPOCHS)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        "Timeformer Fase A — Probe de Disambiguação Contextual\n"
        "Pode a representação do sujeito classificar Contexto A vs B?",
        fontsize=12,
    )

    for ax, subject in zip(axes, SUBJECTS):
        for rep, style in REP_STYLES.items():
            accs = probe_results[rep][subject]
            ax.plot(x, accs, color=style["color"], ls=style["ls"],
                    marker=style["marker"], linewidth=style["lw"],
                    markersize=7, label=style["label"])

        # Linha de chance em 50%
        ax.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6, label="Chance (50%)")

        ax.set_xticks(x)
        ax.set_xticklabels(EPOCHS)
        ax.set_title(SUBJECT_LABELS[subject], fontsize=10)
        ax.set_xlabel("Época")
        ax.set_ylabel("Acurácia" if subject == "S1" else "")
        ax.set_ylim(-0.05, 1.08)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = output_dir / "probe_disambiguation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {path}")

    # ── Plot secundário: foco em S3 com anotações ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        "S3 (Bifurcação) — Probe de Disambiguação\n"
        "e(subj) isolado vs. e(ctx) contextual vs. oráculo de verbo",
        fontsize=12,
    )
    for rep in ["subj", "ctx", "verb"]:
        style = REP_STYLES[rep]
        ax.plot(x, probe_results[rep]["S3"], color=style["color"], ls=style["ls"],
                marker=style["marker"], linewidth=style["lw"], markersize=10,
                label=style["label"])
        for i, acc in enumerate(probe_results[rep]["S3"]):
            ax.annotate(f"{acc:.0%}", (i, acc), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=style["color"])

    ax.axhline(0.5, color="gray", ls="--", lw=1.2, alpha=0.7, label="Chance (50%)")
    # Anotação da bifurcação
    ax.axvspan(2.5, 5.5, alpha=0.07, color="#C62828", label="Bifurcação estabiliza (t3-t5)")
    ax.set_xticks(x)
    ax.set_xticklabels(EPOCHS)
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia do probe")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = output_dir / "probe_s3_focus.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {path}")


def plot_trajectory_signatures(
    trajectory_results: dict,
    output_dir: Path,
) -> None:
    """
    Plota as assinaturas de trajetória (distância cosseno de t0 ao longo de t1-t5)
    para todos os sujeitos, agrupados por fenômeno.

    Demonstração visual central: em t5, estável e bifurcação têm distâncias
    semelhantes (ambas retornam ao ponto de partida). Apenas a trajetória
    completa distingue os dois padrões.
    """
    from src.corpus_generator import SUBJECT_CLASSES

    output_dir.mkdir(parents=True, exist_ok=True)

    signatures = trajectory_results["signatures"]
    true_classes = trajectory_results["true_classes"]
    x = list(range(1, len(EPOCHS)))  # t1..t5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Timeformer Fase A — Assinaturas de Trajetória Temporal\n"
        "Distância cosseno (alinhada) de cada sujeito em relação ao seu embedding em t0",
        fontsize=12,
    )

    # ── Painel esquerdo: trajetórias por sujeito ──
    ax = axes[0]
    ax.set_title("Trajetórias individuais por sujeito", fontsize=10)
    for subj in SUBJECTS:
        cls = true_classes[subj]
        color = PHENOMENON_COLORS[cls]
        ls = "-" if subj in ("S1", "S2", "S3") else "--"
        ax.plot(x, signatures[subj], color=color, ls=ls, marker="o",
                linewidth=2, markersize=7, label=SUBJECT_LABELS[subj])
    ax.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"t{i}" for i in range(1, 6)])
    ax.set_xlabel("Época")
    ax.set_ylabel("Distância cosseno de t0 (pós-alinhamento)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Painel direito: por que snapshot falha ──
    ax = axes[1]
    ax.set_title(
        "Por que snapshot em t5 falha: estável ≈ bifurcação\n"
        "(trajetória distingue; snapshot não)",
        fontsize=10,
    )
    for subj in SUBJECTS:
        cls = true_classes[subj]
        color = PHENOMENON_COLORS[cls]
        ls = "-" if subj in ("S1", "S2", "S3") else "--"
        # Destaca apenas o ponto t5
        ax.scatter([5], [signatures[subj][-1]], color=color, s=120,
                   zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(subj, (5, signatures[subj][-1]),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=9, color=color)

    # Linhas verticais para mostrar a zona de ambiguidade
    snap_vals = {s: signatures[s][-1] for s in SUBJECTS}
    stable_vals = [snap_vals[s] for s in ["S1", "S4"]]
    bifur_vals  = [snap_vals[s] for s in ["S3", "S6"]]
    overlap_min = min(min(stable_vals), min(bifur_vals)) - 0.01
    overlap_max = max(max(stable_vals), max(bifur_vals)) + 0.01
    ax.axhspan(overlap_min, overlap_max, alpha=0.08, color="#C62828",
               label=f"Zona de sobreposição estável/bifurcação\n(snapshot t5 insuficiente)")

    ax.set_xlim(4.0, 6.2)
    ax.set_xticks([5])
    ax.set_xticklabels(["t5"])
    ax.set_xlabel("Época")
    ax.set_ylabel("Distância cosseno de t0 em t5")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = output_dir / "trajectory_signatures.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {path}")


SEED = 42
