"""
Geração e verificação do corpus artificial do Timeformer.

Gera:
  data/corpus.tsv         — corpus principal (train / test / hard_verb / hard_both)
  data/corpus.params.json — parâmetros de trajetória (seed, frações plantadas)
  data/corpus_ambiguous.tsv — avaliação com p_canon=0.50 (ambiguous_test)
"""

from pathlib import Path
from collections import defaultdict

from src.corpus_generator import (
    SUBJECTS,
    SUBJECT_CLASSES,
    CONTEXT_A,
    CONTEXT_B,
    generate_corpus_v2,
    generate_ambiguous_eval,
)

CORPUS_PATH    = Path("data/corpus.tsv")
AMBIGUOUS_PATH = Path("data/corpus_ambiguous.tsv")


# ─── Verificações ─────────────────────────────────────────────────────────────

def _split_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["split"]] += 1
    return dict(counts)


def verify_main_corpus(rows: list[dict]) -> None:
    total = len(rows)
    counts = _split_counts(rows)
    all_splits = ["train", "test", "hard_verb", "hard_both"]

    print("\n── Distribuição de splits ──────────────────────────────────────")
    for sp in all_splits:
        n = counts.get(sp, 0)
        print(f"  {sp:<12}: {n:>5}  ({100*n/total:.1f}%)")

    # split × classe
    print("\n── split × classe ──────────────────────────────────────────────")
    classes = ["stable", "drift", "bifurcation"]
    header = f"  {'split':<12}" + "".join(f"  {c:>13}" for c in classes)
    print(header)
    for sp in all_splits:
        sp_rows = [r for r in rows if r["split"] == sp]
        row_str = f"  {sp:<12}"
        for cls in classes:
            n = sum(1 for r in sp_rows if SUBJECT_CLASSES.get(r["sentence"].split()[0]) == cls)
            row_str += f"  {n:>13}"
        print(row_str)

    # split × época (apenas hard_verb e hard_both)
    print("\n── hard_verb e hard_both por época ─────────────────────────────")
    epochs = sorted({r["epoch"] for r in rows}, key=lambda e: int(e[1:]))
    header = f"  {'época':<7}" + "".join(f"  {'hard_verb':>10}  {'hard_both':>10}")
    print(header)
    for ep in epochs:
        ep_rows = [r for r in rows if r["epoch"] == ep]
        hv = sum(1 for r in ep_rows if r["split"] == "hard_verb")
        hb = sum(1 for r in ep_rows if r["split"] == "hard_both")
        print(f"  {ep:<7}  {hv:>10}  {hb:>10}")

    # split × true_context
    print("\n── split × true_context ────────────────────────────────────────")
    for sp in ["test", "hard_verb", "hard_both"]:
        sp_rows = [r for r in rows if r["split"] == sp]
        n_a = sum(1 for r in sp_rows if r["true_context"] == "A")
        n_b = sum(1 for r in sp_rows if r["split"] == sp and r["true_context"] == "B")
        total_sp = len(sp_rows)
        if total_sp:
            print(f"  {sp:<12}: ctx_A={n_a} ({100*n_a/total_sp:.0f}%)  ctx_B={n_b} ({100*n_b/total_sp:.0f}%)")

    # tamanho mínimo de hard_both por sujeito
    print("\n── hard_both por sujeito (mínimo) ──────────────────────────────")
    hb_per_subj = defaultdict(int)
    for r in rows:
        if r["split"] == "hard_both":
            hb_per_subj[r["sentence"].split()[0]] += 1
    if hb_per_subj:
        min_subj = min(hb_per_subj, key=hb_per_subj.get)
        max_subj = max(hb_per_subj, key=hb_per_subj.get)
        print(f"  mínimo: {min_subj} → {hb_per_subj[min_subj]} frases")
        print(f"  máximo: {max_subj} → {hb_per_subj[max_subj]} frases")
        print(f"  sujeitos sem hard_both: "
              f"{[s for s in SUBJECTS if hb_per_subj[s] == 0] or 'nenhum'}")

    # integridade: verificar classificação correta de hard splits
    errors_hv, errors_hb = 0, 0
    for r in rows:
        verb = r["sentence"].split()[1]
        obj  = r["sentence"].split()[2]
        tc   = r["true_context"]
        v_cross = (tc == "A" and verb in CONTEXT_B["verbs"]) or (tc == "B" and verb in CONTEXT_A["verbs"])
        o_cross = (tc == "A" and obj  in CONTEXT_B["objects"]) or (tc == "B" and obj  in CONTEXT_A["objects"])
        if r["split"] == "hard_verb" and not (v_cross and not o_cross):
            errors_hv += 1
        if r["split"] == "hard_both" and not (v_cross and o_cross):
            errors_hb += 1
    print(f"\n── Integridade de splits ───────────────────────────────────────")
    print(f"  hard_verb erros: {errors_hv}")
    print(f"  hard_both erros: {errors_hb}")


def verify_ambiguous(rows: list[dict]) -> None:
    total = len(rows)
    print(f"\n── ambiguous_test: {total} frases ───────────────────────────────")

    # proporção de verbo e objeto canônicos por true_context
    for tc in ["A", "B"]:
        tc_rows = [r for r in rows if r["true_context"] == tc]
        if not tc_rows:
            continue
        canon_v = CONTEXT_A["verbs"] if tc == "A" else CONTEXT_B["verbs"]
        canon_o = CONTEXT_A["objects"] if tc == "A" else CONTEXT_B["objects"]
        pct_v = 100 * sum(1 for r in tc_rows if r["sentence"].split()[1] in canon_v) / len(tc_rows)
        pct_o = 100 * sum(1 for r in tc_rows if r["sentence"].split()[2] in canon_o) / len(tc_rows)
        print(f"  ctx={tc}  verbo canônico={pct_v:.1f}%  objeto canônico={pct_o:.1f}%  (esperado≈50%)")

    # distribuição por classe
    counts_cls: dict[str, int] = defaultdict(int)
    for r in rows:
        cls = SUBJECT_CLASSES.get(r["sentence"].split()[0], "?")
        counts_cls[cls] += 1
    print(f"\n  por classe: " + ", ".join(f"{c}={counts_cls[c]}" for c in ["stable", "drift", "bifurcation"]))


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Gerando corpus v2...")
    rows, fractions, _ = generate_corpus_v2(
        output_path=CORPUS_PATH,
        p_canon=0.75,
        seed=42,
    )
    print(f"\nCorpus salvo em {CORPUS_PATH}")
    verify_main_corpus(rows)

    print(f"\nGerando ambiguous_test (p_canon=0.50)...")
    amb_rows = generate_ambiguous_eval(
        output_path=AMBIGUOUS_PATH,
        fractions=fractions,
        seed=42,
    )
    print(f"Salvo em {AMBIGUOUS_PATH}")
    verify_ambiguous(amb_rows)


if __name__ == "__main__":
    main()
