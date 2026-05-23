"""
Timeformer Fase A — Experimento Controlado com Linguagem Artificial

Orquestrador dos quatro passos:
  1. Geração do corpus (linguagem artificial, ground truth absoluto)
  2. Treinamento de embeddings skip-gram por época (sem TimeEncoding)
  3. Visualização das trajetórias (PCA 2D + afinidade contextual)
  4. Relatório de validação de sinal

Execute:
  python run_phase_a.py [--skip-training]

  --skip-training  carrega embeddings já treinados de outputs/embeddings/
                   (útil para re-gerar gráficos sem re-treinar)
"""

import argparse
import sys
from pathlib import Path

DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORT_PATH = OUTPUTS_DIR / "report.md"


def main(skip_training: bool = False) -> None:
    # Garante que src/ é importável quando executado da raiz do projeto
    sys.path.insert(0, str(Path(__file__).parent))

    from src.corpus_generator import generate_corpus
    from src.train_embeddings import train_all_epochs, load_embeddings
    from src.align import align_all_epochs, alignment_quality
    from src.visualize import plot_pca_trajectories, plot_context_affinity, plot_probe_results, plot_trajectory_signatures
    from src.probe import run_probe, print_probe_table
    from src.trajectory_probe import run_trajectory_probe, print_trajectory_results
    from src.report import generate_report

    sep = "=" * 60

    # ── Passo 1: Corpus ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Passo 1 — Geração do corpus")
    print(sep)
    corpus_rows = generate_corpus(DATA_DIR / "corpus.tsv")

    # ── Passo 2: Embeddings ────────────────────────────────────────────
    print(f"\n{sep}")
    print("Passo 2 — Embeddings por época (skip-gram, sem TimeEncoding)")
    print(sep)

    if skip_training and EMBEDDINGS_DIR.exists():
        print("Carregando embeddings e modelos existentes …")
        all_embeddings, all_models = load_embeddings(EMBEDDINGS_DIR)
    else:
        all_embeddings, all_models = train_all_epochs(corpus_rows, EMBEDDINGS_DIR)

    # Alinha espaços de embedding ao sistema de coordenadas de t0
    # (usado para PCA e distâncias cross-epoch; probe e afinidade usam métricas intra-época)
    aligned_embeddings = align_all_epochs(all_embeddings)
    alignment_quality(all_embeddings, aligned_embeddings)

    # ── Passo 3: Visualização ──────────────────────────────────────────
    print(f"\n{sep}")
    print("Passo 3 — Visualização das trajetórias")
    print(sep)
    plot_pca_trajectories(aligned_embeddings, PLOTS_DIR)
    affinities = plot_context_affinity(all_embeddings, PLOTS_DIR, all_models=all_models)

    print("\nRodando probe de disambiguação …")
    probe_results = run_probe(corpus_rows, all_embeddings)
    print_probe_table(probe_results)
    plot_probe_results(probe_results, PLOTS_DIR)

    print("\nRodando probe de trajetória …")
    trajectory_results = run_trajectory_probe(aligned_embeddings)
    print_trajectory_results(trajectory_results)
    plot_trajectory_signatures(trajectory_results, PLOTS_DIR)

    # ── Passo 4: Relatório ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("Passo 4 — Relatório de validação")
    print(sep)
    generate_report(all_embeddings, aligned_embeddings, affinities, probe_results, trajectory_results, REPORT_PATH)

    print(f"\n{sep}")
    print(f"Fase A concluída. Saídas em: {OUTPUTS_DIR}/")
    print(f"  Corpus:     {DATA_DIR}/corpus.tsv")
    print(f"  Embeddings: {EMBEDDINGS_DIR}/")
    print(f"  Gráficos:   {PLOTS_DIR}/")
    print(f"  Relatório:  {REPORT_PATH}")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timeformer Fase A")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Pula o treinamento e carrega embeddings já salvos",
    )
    args = parser.parse_args()
    main(skip_training=args.skip_training)
