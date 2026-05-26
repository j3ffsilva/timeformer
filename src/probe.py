"""
Probe de disambiguação contextual — Fase A do Timeformer.

Para cada época, testa se diferentes representações de sentença contendo S3
conseguem classificar o contexto (A vs B) via regressão logística linear.

Quatro representações por sentença:
  e(subj) — somente o embedding do sujeito
             Mesmo vetor para todas as frases do mesmo sujeito/época.
             Trivialmente insuficiente: acurácia = classe majoritária.

  e(verb) — somente o embedding do verbo (oráculo)
             V1-V4 sempre Contexto A, V5-V8 sempre Contexto B.
             Referência superior: deve ser ~100% em toda época.

  e(mean) — média dos embeddings dos três tokens da frase
             Incluir o verbo "contamina" o sinal do sujeito, mascarando
             a insuficiência de e(subj).

  e(ctx)  — média de verb + object, SEM o sujeito
             Teste não-trivial: a informação de bifurcação está distribuída
             nos co-ocorrentes (verbo + objeto), mas não está codificada
             no token embedding do próprio sujeito.
             Se e(ctx) >> e(subj), demonstra que o skip-gram aprende o
             sinal contextual — mas o agrega em verbos/objetos, não no
             sujeito que bifurca. Esse é o gap que o Timeformer preenche.

Resultado esperado para S3:
  e(subj): 100% em t0, degrada para ~50% em t3-t5
  e(verb): ~100% em todas as épocas (oráculo)
  e(mean): ~100% em todas as épocas (puxado pelo verbo)
  e(ctx):  deve manter acurácia alta (> e(subj)) — teste não-trivial

Protocolo:
  - 80% treino / 20% teste, estratificado, seed fixo
  - Quando só uma classe existe na época (e.g. S1 sempre A),
    accuracy = 1.0 por definição (não há erro possível)
"""

import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.train_embeddings import TOKEN_TO_IDX

NEIGH_1_VERBS = {"V1", "V2", "V3", "V4"}
NEIGH_2_VERBS = {"V5", "V6", "V7", "V8"}

# Backward-compatible aliases
CONTEXT_A_VERBS = NEIGH_1_VERBS
CONTEXT_B_VERBS = NEIGH_2_VERBS

SUBJECTS = ["S1", "S2", "S3", "S4", "S5", "S6"]
EPOCHS = [f"t{i}" for i in range(6)]
REP_TYPES = ["subj", "verb", "mean", "ctx"]
TEST_SIZE = 0.20
SEED = 42


def _label(verb: str) -> int:
    return 0 if verb in NEIGH_1_VERBS else 1


def _features(sentences: list[str], embeddings: np.ndarray, rep: str) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for sent in sentences:
        s, v, o = sent.split()
        e_s = embeddings[TOKEN_TO_IDX[s]]
        e_v = embeddings[TOKEN_TO_IDX[v]]
        e_o = embeddings[TOKEN_TO_IDX[o]]
        if rep == "subj":
            feat = e_s
        elif rep == "verb":
            feat = e_v
        elif rep == "mean":
            feat = (e_s + e_v + e_o) / 3.0
        else:  # ctx: verb + object, sem sujeito
            feat = (e_v + e_o) / 2.0
        X.append(feat)
        y.append(_label(v))
    return np.array(X), np.array(y)


def _probe_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    """Trains LogisticRegression on 80% and evaluates on 20%."""
    classes = np.unique(y)
    if len(classes) == 1:
        # Só uma classe — nenhum classificador pode errar
        return 1.0

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(X_tr, y_tr)
    return float(accuracy_score(y_te, clf.predict(X_te)))


def run_probe(
    corpus_rows: list[dict],
    all_embeddings: dict[str, np.ndarray],
) -> dict[str, dict[str, list[float]]]:
    """
    Roda o probe para cada (época × sujeito × representação).

    Retorna dict[rep_type][subject] → lista de acurácias em t0..t5.
    """
    # Usa apenas frases do split TEST — embeddings foram treinados sem elas
    by_epoch: dict[str, list[str]] = {}
    for row in corpus_rows:
        if row.get("split", "train") == "test":
            by_epoch.setdefault(row["epoch"], []).append(row["sentence"])

    results: dict[str, dict[str, list[float]]] = {
        rep: {s: [] for s in SUBJECTS} for rep in REP_TYPES
    }

    for epoch in EPOCHS:
        sentences = by_epoch[epoch]
        emb = all_embeddings[epoch]

        for subject in SUBJECTS:
            subj_sents = [s for s in sentences if s.startswith(subject + " ")]

            for rep in REP_TYPES:
                X, y = _features(subj_sents, emb, rep)
                acc = _probe_accuracy(X, y)
                results[rep][subject].append(acc)

        n_unique = len(set(sentences))
        print(f"  {epoch}: probe concluído ({n_unique} frases únicas)")

    return results


def print_probe_table(results: dict[str, dict[str, list[float]]]) -> None:
    """Imprime tabela de acurácias por época e representação."""
    print("\nProbe de disambiguação — acurácia por época")
    print(f"\n{'':8}", end="")
    for ep in EPOCHS:
        print(f"  {ep:>6}", end="")
    print()

    for subject in SUBJECTS:
        print(f"\n  {subject}:")
        for rep in REP_TYPES:
            accs = results[rep][subject]
            label = {"subj": "e(subj)", "verb": "e(verb)", "mean": "e(mean)", "ctx": "e(ctx)"}[rep]
            row = f"    {label:<8}"
            for acc in accs:
                row += f"  {acc:>6.1%}"
            print(row)
