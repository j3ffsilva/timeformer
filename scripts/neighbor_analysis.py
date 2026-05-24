"""
Análise de vizinhança temporal — demonstração de rastreabilidade semântica.

Extrai h(sujeito) do split de teste para B1, B2b e B3 e computa:

  1. context_drift_score: p̂(ctx=A | epoch) entre os k-NN por classe de sujeito
     → mostra se o modelo "move" a vizinhança em sujeitos com drift, mantendo
       sujeitos estáveis como controle

  2. trajectory_distance: distância cosseno de h(S,t) ao centróide h(S,t0)
     → por classe (estável / deriva / bifurcação), epoch t0..t9

  3. context_coherence: fração de k-NN com mesmo true_context que a query
     → tabela por (modelo × classe × época)

  4. qualitative_table: para o sujeito em deriva com drift mais pronunciado,
     top-k vizinhos em t0, t5, t9 (centroides de sujeito×época)

Uso:
  python scripts/neighbor_analysis.py                     # run mais recente
  python scripts/neighbor_analysis.py --run-id 20260523_006
  python scripts/neighbor_analysis.py --k 10 --models B1 B2b B3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.timeformer.dataset import load_corpus, MLMDataset
from src.timeformer.models import build_model
from src.timeformer.probe import extract_reps
from src.timeformer.train import load_checkpoint
from src.timeformer.run import RunManager

CORPUS_PATH = Path("data/corpus.tsv")

# Índices de sujeito por classe (0-indexados)
STABLE_IDX = list(range(0,  10))
DRIFT_IDX  = list(range(10, 20))
BIFURC_IDX = list(range(20, 30))

CLASS_OF = {i: "stable" for i in STABLE_IDX}
CLASS_OF.update({i: "drift"  for i in DRIFT_IDX})
CLASS_OF.update({i: "bifurc" for i in BIFURC_IDX})


# ── Carregamento ──────────────────────────────────────────────────────────────

def load_model_reps(
    name: str,
    run: RunManager,
    test_ds: MLMDataset,
    device: str,
    batch_size: int = 128,
) -> dict:
    """Carrega modelo + memória e extrai representações do test_ds."""
    ckpt = run.checkpoint_path(name, "best")
    if not ckpt.exists():
        print(f"  {name}: checkpoint não encontrado — skip")
        return {}

    model = build_model(name)
    load_checkpoint(model, ckpt)
    model.to(torch.device(device))

    memory = None
    if name == "B3":
        memory = run.load_memory(name)   # prefere memory_best.pkl
        if memory is None:
            print(f"  B3: memory não encontrada — avaliando sem memória")
        else:
            memory.to(device)

    reps = extract_reps(model, test_ds, memory, batch_size, device)
    return reps


# ── 1. Context drift score ────────────────────────────────────────────────────

def context_drift_score(
    reps: dict,
    k: int = 10,
    subject_class: str = "drift",
) -> dict[int, float]:
    """
    Para cada época, calcula p̂(ctx=A) entre os k-NN de uma classe de sujeitos.

    Retorna dict: epoch_idx (0..9) → média de p(ctx=A | kNN) sobre queries
    da classe pedida.
    """
    H    = reps["h_subj"]          # (N, d)
    ctx  = reps["true_context"]    # (N,) 0=A 1=B
    ep   = reps["epoch_idx"]       # (N,) 0..9
    subj = reps["subject_idx"]     # (N,) 0..29

    class_to_subjects = {
        "stable": set(STABLE_IDX),
        "drift":  set(DRIFT_IDX),
        "bifurc": set(BIFURC_IDX),
    }
    if subject_class not in class_to_subjects:
        raise ValueError(f"Classe desconhecida: {subject_class}")

    query_subjects = class_to_subjects[subject_class]
    query_mask = np.array([int(s) in query_subjects for s in subj])

    # Similaridade cosseno entre todas as pares
    sim = cosine_similarity(H)     # (N, N)
    np.fill_diagonal(sim, -np.inf)

    result: dict[int, list[float]] = {t: [] for t in range(10)}

    for i in np.where(query_mask)[0]:
        t = int(ep[i])
        # k-NN por similaridade cosseno (excluindo a própria sentença)
        nn_idx = np.argsort(sim[i])[::-1][:k]
        # Fração ctx=A entre vizinhos
        frac_a = float((ctx[nn_idx] == 0).mean())
        result[t].append(frac_a)

    return {t: float(np.mean(v)) if v else float("nan")
            for t, v in result.items()}


def context_drift_score_by_class(reps: dict, k: int = 10) -> dict[str, dict[int, float]]:
    """Context drift score separado por classe de sujeito."""
    return {
        cls: context_drift_score(reps, k=k, subject_class=cls)
        for cls in ("stable", "drift", "bifurc")
    }


# ── 2. Trajectory distance ────────────────────────────────────────────────────

def trajectory_distance(reps: dict) -> dict[str, dict[int, float]]:
    """
    Para cada classe, calcula a distância cosseno média de h(S,t) ao h(S,t0).

    Retorna: {class_name: {epoch_idx: mean_cos_dist}}
    """
    H    = reps["h_subj"]
    ep   = reps["epoch_idx"]
    subj = reps["subject_idx"]

    # Centróides por (sujeito, época)
    centroids: dict[tuple[int,int], np.ndarray] = {}
    for s in range(30):
        for t in range(10):
            mask = (subj == s) & (ep == t)
            if mask.sum() > 0:
                centroids[(s, t)] = H[mask].mean(axis=0)

    result: dict[str, dict[int, list[float]]] = {
        "stable": {t: [] for t in range(10)},
        "drift":  {t: [] for t in range(10)},
        "bifurc": {t: [] for t in range(10)},
    }

    for (s, t), c_t in centroids.items():
        c_0 = centroids.get((s, 0))
        if c_0 is None:
            continue
        cls = CLASS_OF[s]
        # distância cosseno = 1 − cosine_similarity
        cos_sim = float(np.dot(c_t, c_0) / (np.linalg.norm(c_t) * np.linalg.norm(c_0) + 1e-8))
        dist = 1.0 - cos_sim
        result[cls][t].append(dist)

    return {
        cls: {t: float(np.mean(v)) if v else float("nan")
              for t, v in epoch_dict.items()}
        for cls, epoch_dict in result.items()
    }


# ── 3. Context coherence ──────────────────────────────────────────────────────

def context_coherence(reps: dict, k: int = 10) -> dict[str, dict[int, float]]:
    """
    Fração de k-NN que compartilham true_context com a query.
    Retorna: {class_name: {epoch_idx: mean_coherence}}
    """
    H    = reps["h_subj"]
    ctx  = reps["true_context"]
    ep   = reps["epoch_idx"]
    subj = reps["subject_idx"]

    sim = cosine_similarity(H)
    np.fill_diagonal(sim, -np.inf)

    result: dict[str, dict[int, list[float]]] = {
        "stable": {t: [] for t in range(10)},
        "drift":  {t: [] for t in range(10)},
        "bifurc": {t: [] for t in range(10)},
    }

    for i in range(len(H)):
        cls = CLASS_OF[int(subj[i])]
        t   = int(ep[i])
        nn_idx = np.argsort(sim[i])[::-1][:k]
        coherence = float((ctx[nn_idx] == ctx[i]).mean())
        result[cls][t].append(coherence)

    return {
        cls: {t: float(np.mean(v)) if v else float("nan")
              for t, v in epoch_dict.items()}
        for cls, epoch_dict in result.items()
    }


# ── 4. Tabela qualitativa ─────────────────────────────────────────────────────

def qualitative_table(
    reps_by_model: dict[str, dict],
    k: int = 5,
    target_subject: int | None = None,
) -> None:
    """
    Para cada modelo, mostra os k vizinhos mais próximos do centróide de
    target_subject nas épocas t0, t5, t9.
    """
    # Escolhe sujeito em deriva com maior variação de p_A entre t0 e t9
    if target_subject is None:
        # Usa as reps do primeiro modelo disponível para escolher
        first_reps = next(iter(reps_by_model.values()))
        H    = first_reps["h_subj"]
        ctx  = first_reps["true_context"]
        ep   = first_reps["epoch_idx"]
        subj = first_reps["subject_idx"]
        best_s, best_diff = DRIFT_IDX[0], -1.0
        for s in DRIFT_IDX:
            t0_mask = (subj == s) & (ep == 0)
            t9_mask = (subj == s) & (ep == 9)
            if t0_mask.sum() == 0 or t9_mask.sum() == 0:
                continue
            pA_t0 = float((ctx[t0_mask] == 0).mean())
            pA_t9 = float((ctx[t9_mask] == 0).mean())
            if abs(pA_t0 - pA_t9) > best_diff:
                best_diff = abs(pA_t0 - pA_t9)
                best_s = s
        target_subject = best_s
        print(f"\nSubjeito alvo: S{target_subject+1} (deriva, Δp_A(t0→t9)={best_diff:.2f})")

    probe_epochs = [0, 5, 9]
    print(f"\n{'─'*70}")
    print(f"Tabela de vizinhos — S{target_subject+1} em t0 / t5 / t9  (k={k})")
    print(f"{'─'*70}")

    for model_name, reps in reps_by_model.items():
        H    = reps["h_subj"]
        ctx  = reps["true_context"]
        ep   = reps["epoch_idx"]
        subj = reps["subject_idx"]

        # Centróides de todos os (sujeito, época)
        centroids: dict[tuple[int,int], np.ndarray] = {}
        centroid_meta: dict[tuple[int,int], dict] = {}
        for s in range(30):
            for t in range(10):
                mask = (subj == s) & (ep == t)
                if mask.sum() > 0:
                    centroids[(s, t)] = H[mask].mean(axis=0)
                    pA = float((ctx[mask] == 0).mean())
                    centroid_meta[(s, t)] = {
                        "subject": s, "epoch": t,
                        "class": CLASS_OF[s], "pA": pA, "n": int(mask.sum()),
                    }

        print(f"\n  Modelo: {model_name}")
        for t_probe in probe_epochs:
            query_key = (target_subject, t_probe)
            if query_key not in centroids:
                print(f"    t{t_probe}: sem dados")
                continue

            query = centroids[query_key]
            query_meta = centroid_meta[query_key]
            print(f"\n    Query  S{target_subject+1}@t{t_probe}  "
                  f"ctx_A={query_meta['pA']:.2f}  n={query_meta['n']}")

            # Ordena todos os centroides por similaridade cosseno com a query
            keys = [k for k in centroids if k != query_key]
            sims = [
                float(np.dot(query, centroids[kk]) /
                      (np.linalg.norm(query) * np.linalg.norm(centroids[kk]) + 1e-8))
                for kk in keys
            ]
            top_k = sorted(zip(sims, keys), reverse=True)[:k]

            for rank, (sim_val, (s, t)) in enumerate(top_k, 1):
                m = centroid_meta[(s, t)]
                print(f"      {rank}. S{s+1}@t{t}  "
                      f"class={m['class']:<7} ctx_A={m['pA']:.2f}  "
                      f"cos={sim_val:.3f}  n={m['n']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Análise de vizinhança temporal")
    parser.add_argument("--run-id",  type=str, default=None)
    parser.add_argument("--k",       type=int, default=10)
    parser.add_argument("--device",  type=str, default="cpu")
    parser.add_argument("--models",  nargs="+", default=["B1", "B2b", "B3"])
    parser.add_argument("--subject", type=int, default=None,
                        help="Índice 0-based do sujeito para tabela qualitativa")
    args = parser.parse_args()

    run = RunManager.load(args.run_id) if args.run_id else RunManager.load_latest()
    print(f"Run: {run.run_id}")

    rows     = load_corpus(CORPUS_PATH)
    test_rows = [r for r in rows if r["split"] == "test"]
    test_ds   = MLMDataset(test_rows)
    print(f"Test split: {len(test_rows)} frases\n")

    reps_by_model: dict[str, dict] = {}
    for name in args.models:
        print(f"Extraindo representações: {name}")
        r = load_model_reps(name, run, test_ds, args.device)
        if r:
            reps_by_model[name] = r

    if not reps_by_model:
        print("Nenhum modelo carregado.")
        return

    # ── 1. Context drift score ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"1. Context drift score — p̂(ctx=A) entre k={args.k} vizinhos por classe")
    print(f"{'─'*70}")

    drift_scores_by_class: dict[str, dict] = {}
    for name, reps in reps_by_model.items():
        drift_scores_by_class[name] = context_drift_score_by_class(reps, k=args.k)

    for cls in ("stable", "drift", "bifurc"):
        print(f"\n  Classe: {cls}")
        print(f"  {'época':<8}", end="")
        for name in reps_by_model:
            print(f"  {name:<10}", end="")
        print()
        for t in range(10):
            print(f"  t{t:<7}", end="")
            for name in reps_by_model:
                v = drift_scores_by_class[name][cls].get(t, float("nan"))
                print(f"  {v:<10.3f}", end="")
            print()

    # ── 2. Trajectory distance ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"2. Trajectory distance — cos_dist(h(S,t), h(S,t0)) por classe")
    print(f"{'─'*70}")

    traj_by_model: dict[str, dict] = {}
    for name, reps in reps_by_model.items():
        traj_by_model[name] = trajectory_distance(reps)

    for cls in ("stable", "drift", "bifurc"):
        print(f"\n  Classe: {cls}")
        print(f"  {'época':<8}", end="")
        for name in reps_by_model:
            print(f"  {name:<10}", end="")
        print()
        for t in range(10):
            print(f"  t{t:<7}", end="")
            for name in reps_by_model:
                v = traj_by_model[name][cls].get(t, float("nan"))
                print(f"  {v:<10.4f}", end="")
            print()

    # ── 3. Context coherence ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"3. Context coherence — fração k-NN com mesmo ctx (k={args.k})")
    print(f"{'─'*70}")

    coh_by_model: dict[str, dict] = {}
    for name, reps in reps_by_model.items():
        coh_by_model[name] = context_coherence(reps, k=args.k)

    for cls in ("stable", "drift", "bifurc"):
        print(f"\n  Classe: {cls}")
        print(f"  {'época':<8}", end="")
        for name in reps_by_model:
            print(f"  {name:<10}", end="")
        print()
        for t in range(10):
            print(f"  t{t:<7}", end="")
            for name in reps_by_model:
                v = coh_by_model[name][cls].get(t, float("nan"))
                print(f"  {v:<10.3f}", end="")
            print()

    # ── 4. Tabela qualitativa ──────────────────────────────────────────────
    qualitative_table(reps_by_model, k=5, target_subject=args.subject)

    # ── Salva resultados ───────────────────────────────────────────────────
    out = {
        "run_id":          run.run_id,
        "k":               args.k,
        # Backward-compatible key: drift subjects only.
        "drift_score":     {
            m: {str(t): v for t, v in d["drift"].items()}
            for m, d in drift_scores_by_class.items()
        },
        "drift_score_by_class": {
            m: {cls: {str(t): v for t, v in cls_scores.items()}
                for cls, cls_scores in d.items()}
            for m, d in drift_scores_by_class.items()
        },
        "trajectory_dist": {m: {cls: {str(t): v for t, v in d.items()}
                                for cls, d in td.items()}
                            for m, td in traj_by_model.items()},
        "coherence":       {m: {cls: {str(t): v for t, v in d.items()}
                                for cls, d in cd.items()}
                            for m, cd in coh_by_model.items()},
    }
    out_path = Path(f"outputs/runs/{run.run_id}/results/neighbor_analysis.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n\nResultados salvos em {out_path}")


if __name__ == "__main__":
    main()
