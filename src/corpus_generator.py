"""
Gerador do corpus artificial para o Timeformer.

v2 (atual): 30 sujeitos, 10 épocas, trajetórias paramétricas.
  Sujeitos: S1-S10 (estável), S11-S20 (deriva), S21-S30 (bifurcação)
  Contexto A: verbos {V1-V4}, objetos {O1-O4}
  Contexto B: verbos {V5-V8}, objetos {O5-O8}

Trajetórias paramétricas:
  Estável:      P(A) = c,           c ~ Uniform(0.75, 1.0)
  Deriva:       linear / sigmoide / onset tardio, start→end
  Bifurcação:   plateau ~0.5 após onset, transição de 2-4 épocas

Splits:
  train     (80%) — frases para treinamento dos embeddings
  test      (20%) — frases de avaliação; contexto local informativo
  hard_test       — subconjunto de test onde o verbo observado contradiz
                    o true_context (marcador cruzado); época necessária

v1 (legado, mantido para reprodutibilidade de run_phase_a4.py):
  generate_corpus() e generate_corpus_probabilistic() usam
  CONTEXT_A_FRACTIONS_V1 com 6 sujeitos e 6 épocas.
"""

import math
import random
import csv
import json
from collections import defaultdict
from pathlib import Path

# ─── Constantes v2 ────────────────────────────────────────────────────────────

SEED = 42
SENTENCES_PER_EPOCH = 500
NUM_EPOCHS = 10
TEST_FRACTION = 0.20
# Distâncias mínimas por classe: stable tem range menor (só varia o nível),
# drift e bifurcação têm espaço de forma muito maior.
MIN_TRAJ_DIST: dict[str, float] = {"stable": 0.04, "drift": 0.10, "bifurcation": 0.10}

N_SUBJECTS     = 30
N_PER_CLASS    = 10

SUBJECTS: list[str] = [f"S{i+1}" for i in range(N_SUBJECTS)]
SUBJECT_CLASSES: dict[str, str] = {
    s: ("stable" if i < N_PER_CLASS else "drift" if i < 2 * N_PER_CLASS else "bifurcation")
    for i, s in enumerate(SUBJECTS)
}

NEIGH_1 = {"verbs": ["V1", "V2", "V3", "V4"], "objects": ["O1", "O2", "O3", "O4"]}
NEIGH_2 = {"verbs": ["V5", "V6", "V7", "V8"], "objects": ["O5", "O6", "O7", "O8"]}

# ─── Trajetórias paramétricas ─────────────────────────────────────────────────

def _traj_distance(t1: list[float], t2: list[float]) -> float:
    n = len(t1)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(t1, t2)) / n)


def _stable_traj(n: int, value: float) -> list[float]:
    return [value] * n


def _linear_drift_traj(n: int, start: float, end: float) -> list[float]:
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def _sigmoid_drift_traj(n: int, start: float, end: float, mid_frac: float, steepness: float) -> list[float]:
    result = []
    for i in range(n):
        x = (i / (n - 1) - mid_frac) * steepness * 8
        s = 1.0 / (1.0 + math.exp(-x))
        result.append(start + (end - start) * s)
    return result


def _late_drift_traj(n: int, start: float, end: float, onset: int) -> list[float]:
    result = [start] * onset
    remaining = n - onset
    for i in range(remaining):
        result.append(start + (end - start) * (i + 1) / remaining)
    return result


def _bifurcation_traj(n: int, start: float, plateau: float, onset: int, transition: int) -> list[float]:
    result = []
    for i in range(n):
        if i < onset:
            result.append(start)
        elif i < onset + transition:
            t = (i - onset + 1) / transition
            result.append(start + (plateau - start) * t)
        else:
            result.append(plateau)
    return result


def _sample_stable(rng: random.Random, n: int) -> tuple[list[float], dict]:
    value = rng.uniform(0.60, 1.0)
    return _stable_traj(n, value), {"family": "stable", "value": round(value, 4)}


def _generate_stable_values(rng: random.Random, n_subjects: int, lo: float = 0.60, hi: float = 1.0) -> list[float]:
    """
    Distribui n_subjects valores estáveis em [lo, hi] usando grade + jitter.
    Garante espaçamento mínimo sem risco de falha por amostragem aleatória.
    """
    step = (hi - lo) / n_subjects
    values = []
    for i in range(n_subjects):
        center = lo + step * (i + 0.5)
        jitter = rng.uniform(-step * 0.25, step * 0.25)
        values.append(round(max(lo + 0.01, min(hi - 0.01, center + jitter)), 4))
    rng.shuffle(values)
    return values


def _sample_drift(rng: random.Random, n: int) -> tuple[list[float], dict]:
    start = rng.uniform(0.85, 1.0)
    end   = rng.uniform(0.0, 0.25)
    family = rng.choice(["linear", "sigmoid", "late"])
    if family == "linear":
        traj = _linear_drift_traj(n, start, end)
        params = {"family": "linear_drift", "start": round(start, 4), "end": round(end, 4)}
    elif family == "sigmoid":
        mid   = rng.uniform(0.3, 0.7)
        steep = rng.uniform(0.8, 2.0)
        traj  = _sigmoid_drift_traj(n, start, end, mid, steep)
        params = {"family": "sigmoid_drift", "start": round(start, 4), "end": round(end, 4),
                  "mid_frac": round(mid, 4), "steepness": round(steep, 4)}
    else:
        onset = rng.randint(2, n // 2)
        traj  = _late_drift_traj(n, start, end, onset)
        params = {"family": "late_drift", "start": round(start, 4), "end": round(end, 4), "onset": onset}
    return traj, params


def _sample_bifurcation(rng: random.Random, n: int) -> tuple[list[float], dict]:
    start      = rng.uniform(0.9, 1.0)
    plateau    = rng.uniform(0.4, 0.6)
    onset      = rng.randint(2, n - 4)
    transition = rng.randint(2, min(4, n - onset))
    traj = _bifurcation_traj(n, start, plateau, onset, transition)
    params = {"family": "bifurcation", "start": round(start, 4), "plateau": round(plateau, 4),
              "onset": onset, "transition": transition}
    return traj, params


def generate_trajectories(
    n_epochs: int = NUM_EPOCHS,
    seed: int = SEED,
    min_dist: dict[str, float] = MIN_TRAJ_DIST,
    max_attempts: int = 1000,
) -> tuple[dict[str, list[float]], dict[str, dict]]:
    """
    Gera trajetórias paramétricas para todos os sujeitos com restrição
    de distância mínima por classe.

    Retorna (context_a_fractions, trajectory_params).
    """
    rng = random.Random(seed)
    samplers = {"stable": _sample_stable, "drift": _sample_drift, "bifurcation": _sample_bifurcation}

    by_class: dict[str, list[str]] = {"stable": [], "drift": [], "bifurcation": []}
    for s in SUBJECTS:
        by_class[SUBJECT_CLASSES[s]].append(s)

    fractions: dict[str, list[float]] = {}
    params_out: dict[str, dict] = {}

    for cls, subjects_in_class in by_class.items():
        sampler = samplers[cls]
        threshold = min_dist[cls]
        generated: list[list[float]] = []
        last_traj, last_params = None, None

        # Sujeitos estáveis: grade + jitter garante separação sem risco de falha
        if cls == "stable":
            stable_values = _generate_stable_values(rng, len(subjects_in_class))
            for subject, value in zip(subjects_in_class, stable_values):
                traj = [value] * n_epochs
                fractions[subject] = traj
                params_out[subject] = {"family": "stable", "value": value}
            continue

        for subject in subjects_in_class:
            chosen_traj, chosen_params = None, None
            for _ in range(max_attempts):
                traj, p = sampler(rng, n_epochs)
                last_traj, last_params = traj, p
                traj_c = [round(max(0.0, min(1.0, v)), 4) for v in traj]
                if all(_traj_distance(traj_c, existing) >= threshold for existing in generated):
                    chosen_traj, chosen_params = traj_c, p
                    break
            if chosen_traj is None:
                chosen_traj = [round(max(0.0, min(1.0, v)), 4) for v in last_traj]
                chosen_params = last_params

            fractions[subject] = chosen_traj
            params_out[subject] = chosen_params
            generated.append(chosen_traj)

    return fractions, params_out


# ─── Geração de frases ────────────────────────────────────────────────────────

def _make_sentence_probabilistic(
    subject: str, use_context_a: bool, rng: random.Random, p_canon: float
) -> tuple[str, str]:
    canon = NEIGH_1 if use_context_a else NEIGH_2
    cross = NEIGH_2 if use_context_a else NEIGH_1
    verb = rng.choice(canon["verbs"] if rng.random() < p_canon else cross["verbs"])
    obj  = rng.choice(canon["objects"] if rng.random() < p_canon else cross["objects"])
    true_ctx = "N1" if use_context_a else "N2"
    return f"{subject} {verb} {obj}", true_ctx


def _verb_is_cross(sentence: str, true_context: str) -> bool:
    verb = sentence.split()[1]
    return (true_context == "N1" and verb in NEIGH_2["verbs"]) or \
           (true_context == "N2" and verb in NEIGH_1["verbs"])


def _obj_is_cross(sentence: str, true_context: str) -> bool:
    obj = sentence.split()[2]
    return (true_context == "N1" and obj in NEIGH_2["objects"]) or \
           (true_context == "N2" and obj in NEIGH_1["objects"])


def _assign_splits(rows: list[dict], seed: int) -> list[dict]:
    """
    Estratifica splits por (época, sujeito, true_context).

    Splits de avaliação (mutuamente exclusivos):
      hard_both — verbo E objeto contradizem true_context (evidência local enganosa)
      hard_verb — só verbo contradiz; objeto ainda aponta corretamente (evidência local mista)
      test      — distribuição natural; contexto local parcialmente informativo
    """
    rng = random.Random(seed)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        subject = row["sentence"].split()[0]
        groups[(row["epoch"], subject, row["true_context"])].append(i)

    splits = ["train"] * len(rows)
    for indices in groups.values():
        n_test = max(1, round(len(indices) * TEST_FRACTION))
        rng.shuffle(indices)
        for idx in indices[:n_test]:
            splits[idx] = "test"

    for i, row in enumerate(rows):
        row["split"] = splits[i]
        if splits[i] == "test":
            v = _verb_is_cross(row["sentence"], row["true_context"])
            o = _obj_is_cross(row["sentence"], row["true_context"])
            if v and o:
                row["split"] = "hard_both"
            elif v:
                row["split"] = "hard_verb"

    return rows


# ─── Gerador principal v2 ─────────────────────────────────────────────────────

def generate_corpus_v2(
    output_path: Path,
    p_canon: float = 0.75,
    seed: int = SEED,
    n_epochs: int = NUM_EPOCHS,
    sentences_per_epoch: int = SENTENCES_PER_EPOCH,
) -> tuple[list[dict], dict[str, list[float]], dict[str, dict]]:
    """
    Gera corpus v2 com trajetórias paramétricas.

    Retorna (corpus_rows, context_a_fractions, trajectory_params).
    O TSV inclui colunas: epoch, sentence, true_context, split.
    Os parâmetros de trajetória são salvos em <output_path>.params.json.

    Splits de avaliação gerados:
      test      — distribuição natural (contexto local parcialmente informativo)
      hard_verb — só verbo contradiz true_context (evidência local mista)
      hard_both — verbo E objeto contradizem true_context (evidência local enganosa)
    """
    fractions, traj_params = generate_trajectories(n_epochs=n_epochs, seed=seed)

    rng = random.Random(seed)
    n_base    = sentences_per_epoch // len(SUBJECTS)
    remainder = sentences_per_epoch - n_base * len(SUBJECTS)
    counts = {s: n_base + (1 if i < remainder else 0) for i, s in enumerate(SUBJECTS)}

    rows: list[dict] = []

    for epoch_idx in range(n_epochs):
        epoch_label = f"t{epoch_idx}"
        epoch_sentences: list[dict] = []

        for subject in SUBJECTS:
            frac_a  = fractions[subject][epoch_idx]
            n_total = counts[subject]
            n_ctx_a = round(n_total * frac_a)
            n_ctx_b = n_total - n_ctx_a

            for _ in range(n_ctx_a):
                sent, true_ctx = _make_sentence_probabilistic(subject, True,  rng, p_canon)
                epoch_sentences.append({"epoch": epoch_label, "sentence": sent, "true_context": true_ctx})
            for _ in range(n_ctx_b):
                sent, true_ctx = _make_sentence_probabilistic(subject, False, rng, p_canon)
                epoch_sentences.append({"epoch": epoch_label, "sentence": sent, "true_context": true_ctx})

        rng.shuffle(epoch_sentences)
        rows.extend(epoch_sentences)

    _assign_splits(rows, seed=seed + 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "sentence", "true_context", "split"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    params_path = output_path.with_suffix(".params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed, "p_canon": p_canon, "n_epochs": n_epochs,
            "n_subjects": len(SUBJECTS), "sentences_per_epoch": sentences_per_epoch,
            "context_a_fractions": fractions,
            "trajectory_params": traj_params,
        }, f, indent=2)

    _print_stats_v2(rows, fractions)
    return rows, fractions, traj_params


def generate_ambiguous_eval(
    output_path: Path,
    fractions: dict[str, list[float]],
    seed: int = SEED,
    n_epochs: int = NUM_EPOCHS,
    sentences_per_epoch: int = SENTENCES_PER_EPOCH,
) -> list[dict]:
    """
    Gera corpus de avaliação com marcadores locais neutros (p_canon=0.50).

    Usa as mesmas trajetórias plantadas do corpus principal (fractions)
    para que P(ctx=A | sujeito, época) seja idêntico. A diferença é que
    verbo e objeto são amostrados com p=0.50 — cada um pertence ao contexto
    canônico com apenas 50% de probabilidade. A evidência local é não-informativa.

    Todas as frases recebem split='ambiguous_test'. Este corpus é exclusivo
    para avaliação — nunca deve ser incluído no treino.

    Uso: testa se o modelo usa sujeito+época (distribuição marginal temporal)
    quando o contexto local não carrega sinal. Não testa bifurcação intra-época
    instance-level, pois sem pista contextual nenhuma arquitetura pode superar
    a distribuição marginal do sujeito na época.
    """
    p_canon = 0.50
    rng = random.Random(seed + 99)  # seed distinto do treino para não vazar padrões

    n_base    = sentences_per_epoch // len(SUBJECTS)
    remainder = sentences_per_epoch - n_base * len(SUBJECTS)
    counts = {s: n_base + (1 if i < remainder else 0) for i, s in enumerate(SUBJECTS)}

    rows: list[dict] = []

    for epoch_idx in range(n_epochs):
        epoch_label = f"t{epoch_idx}"
        epoch_sentences: list[dict] = []

        for subject in SUBJECTS:
            frac_a  = fractions[subject][epoch_idx]
            n_total = counts[subject]
            n_ctx_a = round(n_total * frac_a)
            n_ctx_b = n_total - n_ctx_a

            for _ in range(n_ctx_a):
                sent, true_ctx = _make_sentence_probabilistic(subject, True,  rng, p_canon)
                epoch_sentences.append({"epoch": epoch_label, "sentence": sent,
                                        "true_context": true_ctx, "split": "ambiguous_test"})
            for _ in range(n_ctx_b):
                sent, true_ctx = _make_sentence_probabilistic(subject, False, rng, p_canon)
                epoch_sentences.append({"epoch": epoch_label, "sentence": sent,
                                        "true_context": true_ctx, "split": "ambiguous_test"})

        rng.shuffle(epoch_sentences)
        rows.extend(epoch_sentences)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "sentence", "true_context", "split"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return rows


def _print_stats_v2(rows: list[dict], fractions: dict[str, list[float]]) -> None:
    splits = ["train", "test", "hard_verb", "hard_both"]
    counts = {sp: sum(1 for r in rows if r["split"] == sp) for sp in splits}
    print(f"Corpus v2: {len(rows)} frases  "
          f"train={counts['train']}  test={counts['test']}  "
          f"hard_verb={counts['hard_verb']}  hard_both={counts['hard_both']}")
    print(f"\n{'Trajetórias plantadas (P(ctx=A) por época)':}")
    epochs = sorted({r['epoch'] for r in rows}, key=lambda e: int(e[1:]))
    header = f"  {'sujeito':<8} {'classe':<13}" + "".join(f"  {ep:>5}" for ep in epochs)
    print(header)
    print("  " + "-" * (8 + 13 + 7 * len(epochs)))
    for s in SUBJECTS:
        cls = SUBJECT_CLASSES[s]
        row_str = f"  {s:<8} {cls:<13}"
        for ep in epochs:
            ep_idx = int(ep[1:])
            row_str += f"  {fractions[s][ep_idx]:.2f}"
        # mark class
        print(row_str)


# ─── Legado v1 (6 sujeitos, 6 épocas) — mantido para reprodutibilidade ────────

_SUBJECTS_V1 = ["S1", "S2", "S3", "S4", "S5", "S6"]
_NUM_EPOCHS_V1 = 6
_SENTENCES_PER_EPOCH_V1 = 500

CONTEXT_A_FRACTIONS: dict[str, list[float]] = {
    "S1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "S4": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "S2": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    "S5": [1.0, 1.0, 0.9, 0.7, 0.5, 0.3],
    "S3": [1.0, 0.9, 0.7, 0.5, 0.5, 0.5],
    "S6": [1.0, 1.0, 1.0, 0.8, 0.5, 0.5],
}


def _make_sentence_v1(subject: str, use_context_a: bool, rng: random.Random) -> str:
    ctx = NEIGH_1 if use_context_a else NEIGH_2
    return f"{subject} {rng.choice(ctx['verbs'])} {rng.choice(ctx['objects'])}"


def _assign_splits_v1(rows: list[dict]) -> list[dict]:
    rng = random.Random(SEED + 1)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        tokens = row["sentence"].split()
        ctx = "N1" if tokens[1] in NEIGH_1["verbs"] else "N2"
        groups[(row["epoch"], tokens[0], ctx)].append(i)
    splits = ["train"] * len(rows)
    for indices in groups.values():
        n_test = max(1, round(len(indices) * TEST_FRACTION))
        rng.shuffle(indices)
        for idx in indices[:n_test]:
            splits[idx] = "test"
    for i, row in enumerate(rows):
        row["split"] = splits[i]
    return rows


def generate_corpus(output_path: Path) -> list[dict]:
    """Legado v1 — gera corpus determinístico com 6 sujeitos e 6 épocas."""
    rng = random.Random(SEED)
    n_base = _SENTENCES_PER_EPOCH_V1 // len(_SUBJECTS_V1)
    remainder = _SENTENCES_PER_EPOCH_V1 - n_base * len(_SUBJECTS_V1)
    counts = {s: n_base + (1 if i < remainder else 0) for i, s in enumerate(_SUBJECTS_V1)}
    rows: list[dict] = []
    for epoch_idx in range(_NUM_EPOCHS_V1):
        epoch_label = f"t{epoch_idx}"
        epoch_sentences: list[dict] = []
        for subject in _SUBJECTS_V1:
            frac_a  = CONTEXT_A_FRACTIONS[subject][epoch_idx]
            n_total = counts[subject]
            n_ctx_a = round(n_total * frac_a)
            n_ctx_b = n_total - n_ctx_a
            for _ in range(n_ctx_a):
                epoch_sentences.append({"epoch": epoch_label, "sentence": _make_sentence_v1(subject, True, rng)})
            for _ in range(n_ctx_b):
                epoch_sentences.append({"epoch": epoch_label, "sentence": _make_sentence_v1(subject, False, rng)})
        rng.shuffle(epoch_sentences)
        rows.extend(epoch_sentences)
    _assign_splits_v1(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "sentence", "split"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _assign_splits_by_true_context_v1(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        subject = row["sentence"].split()[0]
        groups[(row["epoch"], subject, row["true_context"])].append(i)
    splits = ["train"] * len(rows)
    for indices in groups.values():
        n_test = max(1, round(len(indices) * TEST_FRACTION))
        rng.shuffle(indices)
        for idx in indices[:n_test]:
            splits[idx] = "test"
    for i, row in enumerate(rows):
        row["split"] = splits[i]
    return rows


def generate_corpus_probabilistic(
    output_path: Path, p_canon: float = 0.75, seed: int = SEED,
) -> list[dict]:
    """Legado v1 — gera corpus probabilístico com 6 sujeitos e 6 épocas."""
    rng = random.Random(seed)
    n_base = _SENTENCES_PER_EPOCH_V1 // len(_SUBJECTS_V1)
    remainder = _SENTENCES_PER_EPOCH_V1 - n_base * len(_SUBJECTS_V1)
    counts = {s: n_base + (1 if i < remainder else 0) for i, s in enumerate(_SUBJECTS_V1)}
    rows: list[dict] = []
    for epoch_idx in range(_NUM_EPOCHS_V1):
        epoch_label = f"t{epoch_idx}"
        epoch_sentences: list[dict] = []
        for subject in _SUBJECTS_V1:
            frac_a  = CONTEXT_A_FRACTIONS[subject][epoch_idx]
            n_total = counts[subject]
            n_ctx_a = round(n_total * frac_a)
            n_ctx_b = n_total - n_ctx_a

            def _make_prob(use_a):
                canon = NEIGH_1 if use_a else NEIGH_2
                cross = NEIGH_2 if use_a else NEIGH_1
                verb = rng.choice(canon["verbs"] if rng.random() < p_canon else cross["verbs"])
                obj  = rng.choice(canon["objects"] if rng.random() < p_canon else cross["objects"])
                return f"{subject} {verb} {obj}", "N1" if use_a else "N2"

            for _ in range(n_ctx_a):
                sent, true_ctx = _make_prob(True)
                epoch_sentences.append({"epoch": epoch_label, "sentence": sent, "true_context": true_ctx})
            for _ in range(n_ctx_b):
                sent, true_ctx = _make_prob(False)
                epoch_sentences.append({"epoch": epoch_label, "sentence": sent, "true_context": true_ctx})

        rng.shuffle(epoch_sentences)
        rows.extend(epoch_sentences)

    _assign_splits_by_true_context_v1(rows, seed=seed + 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "sentence", "true_context", "split"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows
