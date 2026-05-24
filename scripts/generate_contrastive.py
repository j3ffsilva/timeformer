"""
Gera o conjunto contrastivo para avaliação da Fase B.

Para cada sujeito em deriva (S11-S20), seleciona um par de épocas onde
true_context difere com alta confiança:
  t_early: P(ctx=A | S) > P_HIGH  →  true_context = A
  t_late:  P(ctx=A | S) < 1 - P_HIGH  →  true_context = B

Para cada par (S, O) disponível no corpus de treino nessas épocas,
gera dois itens com verbo mascarado — a única diferença é a época.

Saída: data/contrastive_set.tsv
Colunas: pair_id, subject, obj, epoch_idx, true_context
"""

from pathlib import Path
import csv
import json

CORPUS_PATH    = Path("data/corpus.tsv")
PARAMS_PATH    = Path("data/corpus.params.json")
OUTPUT_PATH    = Path("data/contrastive_set.tsv")

# Sujeitos em deriva: P(ctx=A) decresce monotonicamente
DRIFT_SUBJECTS = [f"S{i}" for i in range(11, 21)]

# Threshold de confiança para selecionar épocas do par
P_HIGH = 0.80


def find_epoch_pair(fractions: list[float], p_high: float) -> tuple[int, int] | None:
    """
    Encontra o par (t_early, t_late) de maior separação onde:
      fractions[t_early] > p_high  (ctx=A confiante)
      fractions[t_late]  < 1 - p_high  (ctx=B confiante)

    Retorna None se o sujeito não tem épocas suficientemente separadas.
    """
    early_candidates = [t for t, p in enumerate(fractions) if p > p_high]
    late_candidates  = [t for t, p in enumerate(fractions) if p < 1 - p_high]

    if not early_candidates or not late_candidates:
        return None

    # Maximizar separação temporal entre os dois extremos
    t_early = early_candidates[0]   # época mais cedo com alta confiança em A
    t_late  = late_candidates[-1]   # época mais tarde com alta confiança em B

    if t_early >= t_late:
        return None

    return t_early, t_late


def load_corpus_objects(corpus_path: Path, subjects: list[str]) -> dict[str, dict[int, list[str]]]:
    """
    Para cada sujeito em subjects, coleta os objetos disponíveis por época
    no split='train'. Retorna {subject: {epoch_idx: [obj1, obj2, ...]}}
    """
    result: dict[str, dict[int, list[str]]] = {s: {} for s in subjects}

    with open(corpus_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["split"] != "train":
                continue
            tokens = row["sentence"].split()
            subj, obj = tokens[0], tokens[2]
            if subj not in result:
                continue
            epoch_idx = int(row["epoch"][1:])
            result[subj].setdefault(epoch_idx, []).append(obj)

    return result


def generate(p_high: float = P_HIGH) -> list[dict]:
    params = json.loads(PARAMS_PATH.read_text())
    fractions: dict[str, list[float]] = params["context_a_fractions"]

    objects_by_subject_epoch = load_corpus_objects(CORPUS_PATH, DRIFT_SUBJECTS)

    rows: list[dict] = []
    pair_id = 0

    for subj in DRIFT_SUBJECTS:
        traj = fractions[subj]
        epoch_pair = find_epoch_pair(traj, p_high)
        if epoch_pair is None:
            print(f"  {subj}: sem par de épocas com confiança > {p_high} — ignorado")
            continue

        t_early, t_late = epoch_pair
        p_early = traj[t_early]
        p_late  = traj[t_late]

        # Objetos disponíveis em ambas as épocas para este sujeito
        objs_early = set(objects_by_subject_epoch[subj].get(t_early, []))
        objs_late  = set(objects_by_subject_epoch[subj].get(t_late,  []))
        shared_objs = sorted(objs_early & objs_late)

        if not shared_objs:
            # Sem objetos compartilhados: usar todos os objetos de cada época
            # O par não tem a mesma superfície, mas ainda é contrastivo
            all_objs = sorted(objs_early | objs_late)
        else:
            all_objs = shared_objs

        # Gera um par por objeto (até 4 para não sobrecarregar)
        for obj in all_objs[:4]:
            true_ctx_early = "A"
            true_ctx_late  = "B"
            rows.append({
                "pair_id":      pair_id,
                "subject":      subj,
                "obj":          obj,
                "epoch_idx":    t_early,
                "true_context": true_ctx_early,
            })
            rows.append({
                "pair_id":      pair_id,
                "subject":      subj,
                "obj":          obj,
                "epoch_idx":    t_late,
                "true_context": true_ctx_late,
            })
            pair_id += 1

        print(f"  {subj}: t_early=t{t_early} (P(A)={p_early:.2f}) "
              f"× t_late=t{t_late} (P(A)={p_late:.2f}) "
              f"→ {min(len(all_objs), 4)} pares (objs compartilhados={len(shared_objs)})")

    return rows


def main() -> None:
    print(f"Gerando conjunto contrastivo (P_HIGH={P_HIGH})...")
    rows = generate()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "subject", "obj",
                                               "epoch_idx", "true_context"],
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    n_pairs = len(rows) // 2
    subjects_covered = len({r["subject"] for r in rows})
    print(f"\nSalvo em {OUTPUT_PATH}")
    print(f"  {n_pairs} pares | {subjects_covered}/{len(DRIFT_SUBJECTS)} sujeitos cobertos")
    print(f"  {len(rows)} linhas totais (2 por par)")


if __name__ == "__main__":
    main()
