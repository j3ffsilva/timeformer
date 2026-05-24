# Paper 1B: IBERAMIA — Timeformer: Explicit Temporal Conditioning for Traceable Semantic Representations

**Status:** paper plan  
**Target venue:** IBERAMIA or another international AI/NLP venue  
**Language:** English  
**Focus:** conceptual architecture, temporal traceability, and preliminary experiments  
**Relation to Paper 1A:** complementary paper using the same benchmark, but with a different scientific question.

---

## 1. Core Thesis

This paper should not claim that Timeformer is a general replacement for
Transformers. The contribution is narrower and more defensible:

> Timeformer introduces an explicit temporal interface for semantic
> representations, enabling tokens to be queried, compared, and traced as
> `token@time` rather than as timeless lexical types.

The emphasis is **semantic traceability**, not state-of-the-art performance.

---

## 2. Scientific Question

Can a Transformer-style architecture be extended so that the same lexical item
has temporally distinguishable contextual neighborhoods?

In concrete terms:

```text
S11@t2 should be closer to context A
S11@t8 should be closer to context B
```

even though both correspond to the same surface token `S11`.

---

## 3. Main Contributions

1. **Problem formulation:** temporal semantic traceability as the ability to
   represent and inspect `token@time`.
2. **Architecture proposal:** Timeformer, a Transformer extension with:
   - continuous TimeEncoding;
   - token-time interaction;
   - optional causal trajectory memory.
3. **Ablation chain:**
   - Static Transformer (`B1`);
   - Additive Time-Conditioned Transformer (`B2a`);
   - Token-Time Transformer (`B2b`);
   - Memory-Augmented Timeformer (`B3`).
4. **Contrastive evaluation:** same surface form, different epochs, expected
   semantic inversion.
5. **Diagnostic analysis:** why naive mean-pooled prototype memory is insufficient
   for intra-epoch bifurcation.
6. **Forward path:** sense-aware semantic canonization as the next step.

---

## 4. What This Paper Is Not

This paper should avoid overclaiming.

It is not:

- a full solution to lexical semantic change in natural language;
- a state-of-the-art comparison against all temporal language models;
- a complete solution to intra-epoch sense bifurcation;
- the final sense-aware Timeformer paper.

It is:

- a conceptual and experimental proposal for temporally traceable
  representations.

---

## 5. Architecture Narrative

### 5.1 Static Transformer (`B1`)

Input:

```text
[CLS] S V O [SEP]
```

No epoch information. This model learns contextual co-occurrence but cannot
represent that the same subject behaves differently in different epochs.

### 5.2 Additive Time-Conditioned Transformer (`B2a`)

```text
h(token,t) = TokenEmbedding(token) + TimeEncoding(t)
```

This tests whether simply telling the model the epoch is enough. The expected
answer is: partially, but not robustly.

### 5.3 Token-Time Transformer (`B2b`)

```text
h(token,t) = f(TokenEmbedding(token), TimeEncoding(t))
```

This is the main temporal representation. The epoch interacts with the lexical
identity of each token, enabling `S11@t2` and `S11@t8` to occupy different
semantic neighborhoods.

### 5.4 Memory-Augmented Timeformer (`B3`)

```text
M(S,t_k) = {m(S,t_0), ..., m(S,t_{k-1})}
```

The subject representation attends causally to historical prototypes. This
module tests whether explicit memory improves traceability, but current results
should be framed as preliminary/diagnostic unless the corrected `B3` run
produces stable gains.

---

## 6. Evaluation Plan

### 6.1 Main Probe

Train the models with MLM, without `true_context`. Then train a linear probe on
`h(subject)` to predict `true_context`.

Why `h(subject)`?

The subject is the token whose semantic trajectory was planted. Sentence-level
representations may solve the task for local reasons; `h(subject)` is closer to
the traceability claim.

### 6.2 Ambiguous Test

`ambiguous_test` uses `p_canon=0.50`, making local context non-informative.

Expected result:

```text
Token-Time Transformer > Static Transformer
```

Interpretation:

The model uses epoch information when local context does not help.

### 6.3 Contrastive Set

Pairs with the same surface form and different epochs:

```text
S11 [MASK] O3 @t2  → context A verbs
S11 [MASK] O3 @t8  → context B verbs
```

This is the cleanest evidence of semantic traceability.

### 6.4 Trajectory Continuation

Train on t0-t7, evaluate on t8-t9.

Important implementation requirement:

The continuation split must be truly held out during training. Previous runs
that trained on all `split=train` rows do not constitute a valid continuation
test.

### 6.5 Memory Controls

For the Memory-Augmented Timeformer (`B3`):

- B3-shuffled-subject;
- B3-shuffled-time;
- B3-nohistory.

These controls are necessary to distinguish trajectory use from extra
parameters.

---

## 7. Preliminary Findings to Report Carefully

The current evidence supports:

1. Time-aware representations can distinguish temporal neighborhoods.
2. Static/no-time models struggle on temporal marginal tasks.
3. The benchmark exposes failure modes of naive temporal memory.
4. Mean-pooled prototypes collapse coexisting senses.

The current evidence does **not yet** support:

1. Memory-Augmented Timeformer consistently improves over Token-Time Transformer.
2. PrototypeMemory solves trajectory continuation.
3. Timeformer solves intra-epoch bifurcation.

If the corrected `B3` run succeeds, upgrade the results section. If not, frame
Memory-Augmented Timeformer as a diagnostic module and motivate the sense-aware
extension.

---

## 8. Distinction From the KDMiLe Paper

This paper should not duplicate the KDMiLe paper.

| Aspect | KDMiLe Paper | IBERAMIA Paper |
|--------|--------------|----------------|
| Main question | How to build a controlled benchmark? | How to make representations temporally traceable? |
| Main contribution | Corpus, splits, methodology | Architecture and traceability evaluation |
| Language | Portuguese | English |
| Core artifact | Synthetic benchmark | Timeformer ablation chain |
| Results emphasis | Baselines and corpus validation | Timeformer ablation chain, contrastive evaluation |
| Future work | Timeformer | Sense-aware Timeformer |

If the KDMiLe paper is accepted first, cite it in the IBERAMIA paper as the
benchmark source.

---

## 9. Possible Title Options

1. **Timeformer: Explicit Temporal Conditioning for Traceable Semantic Representations**
2. **Tracing Semantic Drift with Time-Conditioned Transformer Representations**
3. **Toward Temporally Traceable Language Models**
4. **Token-at-Time Representations for Semantic Drift Analysis**

Preferred title:

> Timeformer: Explicit Temporal Conditioning for Traceable Semantic Representations

---

## 10. Suggested Structure

1. **Introduction**
   - semantic drift as traceability problem;
   - limitations of timeless token representations;
   - contribution summary.

2. **Related Work**
   - diachronic embeddings;
   - time-aware language models;
   - lexical semantic change detection;
   - multi-sense embeddings as future direction.

3. **Benchmark Summary**
   - short description only;
   - refer to the KDMiLe paper or technical report for full methodology.

4. **Timeformer Architecture**
   - Static Transformer / Additive Time-Conditioned Transformer / Token-Time Transformer / Memory-Augmented Timeformer;
   - continuous TimeEncoding;
   - token-time interaction;
   - causal memory.

5. **Evaluation**
   - MLM + probe;
   - ambiguous split;
   - contrastive set;
   - continuation;
   - shuffled controls.

6. **Results**
   - traceability metrics;
   - contrastive sign flip;
   - neighborhood preservation;
   - B3 diagnostics.

7. **Discussion**
   - what temporal encoding solves;
   - what prototype memory does not solve;
   - why sense-aware canonization is needed.

8. **Conclusion**
   - Timeformer as a traceability interface;
   - next step: sense-aware Timeformer.

---

## 11. Minimum Results Needed Before Submission

Required:

- Static Transformer vs Token-Time Transformer on `ambiguous_test`;
- contrastive sign-flip comparison;
- neighborhood/precision@k analysis;
- clear statement of B3 status.

Strongly recommended:

- corrected continuation split;
- memory aligned with best checkpoint;
- B3-shuffled and B3-nohistory controls;
- at least 3 seeds for key deltas.

If these are not ready, submit as conceptual/preliminary work rather than a
strong empirical claim.
