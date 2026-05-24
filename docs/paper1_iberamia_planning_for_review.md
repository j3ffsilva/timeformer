# IBERAMIA Paper Planning Brief

**Working title:** Timeformer: Explicit Temporal Conditioning for Traceable Semantic Representations  
**Target:** IBERAMIA or similar AI/NLP venue  
**Urgency:** high; target draft next week  
**Status:** planning only; not a manuscript draft  
**Language of final paper:** English  

---

## 0. Context for External Review

This project was originally imagined as one larger paper, but we are now
splitting it into two papers with different scientific roles.

### Paper A: KDMiLe Benchmark Paper

**Venue/language:** KDMiLe, Portuguese  
**Main contribution:** the controlled synthetic benchmark.

This paper owns:

- corpus construction;
- trajectory families;
- `true_context` ground truth;
- probabilistic markers;
- train/test/hard/ambiguous splits;
- methodological pitfalls in earlier corpus versions;
- baseline evidence that the benchmark exposes temporal behavior.

The KDMiLe paper answers:

> How can we build a controlled benchmark for studying temporal semantic drift
> with known ground truth?

### Paper B: IBERAMIA Timeformer Paper

**Venue/language:** IBERAMIA, English  
**Main contribution:** architecture and temporal traceability diagnostics.

This paper uses the benchmark but should not duplicate the full benchmark
methodology. It should summarize the corpus briefly and refer to the KDMiLe
paper or technical report for details when possible.

The IBERAMIA paper answers:

> How can a Transformer-style representation expose a queryable `token@time`
> interface for tracing semantic neighborhoods across epochs?

### Consequence for Review

When reviewing this plan, do not ask the IBERAMIA paper to carry the full burden
of the benchmark contribution. The IBERAMIA paper is the urgent paper and should
focus on Timeformer, temporal conditioning, and diagnostic evidence of
traceability.

---

## 1. IBERAMIA Positioning

This paper should be framed as an architecture and diagnostic evaluation paper,
not as a broad state-of-the-art paper on lexical semantic change.

**Central thesis:**

Timeformer provides an explicit temporal interface for semantic
representations, allowing a lexical item to be inspected as `token@time` rather
than only as a timeless token type.

**Defensible claim:**

Transformer-style representations can be extended with explicit temporal
conditioning so that the same surface token occupies measurably different
semantic neighborhoods at different epochs in a controlled benchmark.

**Operational definition:**

A representation is temporally traceable if the same surface token, evaluated
at different epochs, has measurably different and semantically appropriate
neighborhoods.

**Claims to avoid:**

- Timeformer solves lexical semantic change in natural language.
- Timeformer is a general replacement for existing Transformers.
- Memory-Augmented Timeformer is empirically superior unless `B3` wins robustly.
- Mean-pooled prototype memory solves intra-epoch sense bifurcation.
- The IBERAMIA paper introduces the benchmark as its primary contribution.

---

## 2. Scientific Question

Can a Transformer-style encoder represent the same lexical item differently
across epochs while preserving enough semantic structure to support temporal
inspection?

Concrete benchmark question:

```text
S11@t2 should be closer to context A
S11@t8 should be closer to context B
```

even though both examples contain the same surface token `S11`.

---

## 3. Contribution Set for IBERAMIA

1. **Temporal traceability formulation:** `token@time` as the object of
   inspection.
2. **Timeformer family:** a controlled ablation chain with increasing temporal
   capacity.
3. **Token-time interaction:** explicit interaction between lexical identity and
   continuous epoch encoding.
4. **Traceability diagnostics:** context drift score, contrastive sign flip,
   neighborhood analysis, and probe evaluation.
5. **Memory diagnostic:** evidence that naive mean-pooled historical prototypes
   are insufficient under the current MLM setup, motivating sense-aware memory.

The strongest positive result should be expected to come from the
**Token-Time Transformer (`B2b`)**, not necessarily from the
**Memory-Augmented Timeformer (`B3`)**.

---

## 4. Nomenclature

The experiment keeps short IDs internally but uses public names in the paper.

| Internal ID | Public name | Role |
|-------------|-------------|------|
| `B1` | Static Transformer | No time information |
| `B2a` | Additive Time-Conditioned Transformer | Adds a global epoch vector |
| `B2b` | Token-Time Transformer | Makes time interact with each token |
| `B3` | Memory-Augmented Timeformer | Adds causal historical prototype memory |

Preferred narrative:

```text
Static Transformer
→ Additive Time-Conditioned Transformer
→ Token-Time Transformer
→ Memory-Augmented Timeformer
```

Preferred delta names:

| Delta | Definition | Main interpretation |
|-------|------------|---------------------|
| Delta time-conditioning | `B2a - B1` | Gain from giving the model epoch information |
| Delta token-time interaction | `B2b - B2a` | Gain from token-specific temporal conditioning |
| Delta memory | `B3 - B2b` | Gain from causal historical prototypes |
| Delta spurious memory | `B3-shuffled - B2b` | Whether memory gains are trajectory-specific |

---

## 5. Current Evidence and Interpretation

### 5.1 Main Positive Evidence: Context Drift Score

The context drift score is the strongest evidence for the core traceability
claim. It measures the estimated context-A proportion among nearest neighbors of
drifting subjects across epochs.

Across runs `20260523_006` to `20260523_008`, for drifting subjects:

| Model | t0 mean | t9 mean | Delta |
|-------|---------|---------|-------|
| Static Transformer (`B1`) | 0.618 | 0.387 | -0.231 |
| Token-Time Transformer (`B2b`) | 0.847 | 0.233 | -0.614 |
| Memory-Augmented Timeformer (`B3`) | 0.836 | 0.263 | -0.573 |

Interpretation:

`B2b` produces a much stronger temporal neighborhood transition than `B1`.
This directly supports the claim that the same token can occupy different
semantic neighborhoods across epochs when token-time conditioning is present.

This result should become a central figure, not a secondary diagnostic.

Class-specific context drift has now been regenerated for all three seeds:

| Class | Model | t0 mean | t9 mean | Delta |
|-------|-------|---------|---------|-------|
| Stable | Static Transformer (`B1`) | 0.736 | 0.756 | +0.020 |
| Stable | Token-Time Transformer (`B2b`) | 0.843 | 0.641 | -0.203 |
| Stable | Memory-Augmented Timeformer (`B3`) | 0.810 | 0.658 | -0.153 |
| Drift | Static Transformer (`B1`) | 0.618 | 0.387 | -0.231 |
| Drift | Token-Time Transformer (`B2b`) | 0.847 | 0.233 | -0.614 |
| Drift | Memory-Augmented Timeformer (`B3`) | 0.836 | 0.263 | -0.573 |
| Bifurcation | Static Transformer (`B1`) | 0.794 | 0.682 | -0.112 |
| Bifurcation | Token-Time Transformer (`B2b`) | 0.895 | 0.521 | -0.374 |
| Bifurcation | Memory-Augmented Timeformer (`B3`) | 0.905 | 0.517 | -0.388 |

Interpretation:

The class-specific figure should not claim that stable subjects are perfectly
flat. Instead, it should show that the temporal neighborhood shift is strongest
for drifting subjects, weaker for bifurcating subjects, and weakest or least
directional for stable subjects. This is more honest and still supports the
traceability claim.

### 5.2 Probe Results Are Supportive but Not Sufficient

Probe accuracy is useful but should not carry the main claim alone.

Reason:

The difference between `B2a` and `B2b` on `ambiguous_test` is small and
seed-sensitive. In recent runs, Delta token-time interaction is positive but
not large:

```text
run 006: +0.0530
run 007: +0.0048
run 008: +0.0208
```

Implication:

The paper should use probe results as one diagnostic and use context drift score
and contrastive sign flip as the more direct evidence of temporal traceability.

### 5.3 Memory-Augmented Timeformer Is Diagnostic

`B3` does not currently provide a stable continuation gain over `B2b`.

Recent Delta memory values:

```text
run 006: +0.0126
run 007: +0.0050
run 008: -0.0176
```

Interpretation:

Memory-Augmented Timeformer should be reported honestly as a diagnostic
extension. The important finding is not "memory wins"; it is that naive
mean-pooled historical prototypes and the current MLM objective do not reliably
produce a trajectory-specific advantage.

### 5.4 Oracle Memory Diagnostic

The `B3-oracle` diagnostic has been rerun with the corrected continuation
protocol using reference run `20260523_006`.

Corrected comparison in run `20260524_001`:

| Model | Continuation accuracy | Contrastive sign flip |
|-------|-----------------------|-----------------------|
| `B2b_ref` | 0.7620 | 0.4483 |
| `B3_oracle` | 0.7544 | 0.5862 |
| `B3_learned` | 0.7746 | 0.4483 |

Interpretation:

Oracle memory does not improve the main continuation probe over `B2b_ref`, but
it modestly improves contrastive sign flip in this corrected run. The careful
claim is:

> Perfect historical memory alone does not solve the continuation/probe
> objective under this small SVO MLM setup, although oracle memory can affect
> contrastive temporal behavior.

This is useful as a bridge to the sense-aware memory paper, but it should not
be oversimplified as "oracle memory does nothing."

---

## 6. Proposed Paper Structure

The LaTeX planning skeleton has been created at:

```text
paper1_iberamia_timeformer.tex
```

Compile from the repository root with:

```text
TEXINPUTS=templates/LaTeX2e//: pdflatex paper1_iberamia_timeformer.tex
```

The target page limit is 12 pages in the LLNCS template. The current plan uses
three main figures and three compact tables. A fourth memory-specific figure
should be added only if the architecture section feels unclear after prose and
Figure 2 are in place.

### Page Budget

| Section | Target pages | Role |
|---------|--------------|------|
| Abstract | 0.25 | State traceability problem, architecture, and main finding |
| Introduction | 1.0--1.25 | Establish `token@time` and contribution set |
| Related Work | 1.25--1.5 | Situate against LSCD, diachronic embeddings, temporal conditioning |
| Controlled Benchmark Summary | 1.0 | Enough benchmark context without duplicating KDMiLe |
| Timeformer Architecture | 2.0 | Present ablation chain and token-time interaction |
| Evaluation Protocol | 1.25--1.5 | Define diagnostics and controls |
| Results | 2.5--3.0 | Lead with context drift, then contrastive/probe/memory |
| Discussion | 1.25--1.5 | Interpret limits, especially memory |
| Conclusion | 0.5 | Narrow restatement and next step |
| References/Credits | 0.75--1.0 | Keep compact |

### 1. Introduction

Goal: establish temporal traceability as the core problem.

Must include:

- lexical meaning changes across time;
- timeless token representations obscure temporal movement;
- `token@time` as the conceptual object;
- controlled benchmark as a diagnostic testbed, not as the main contribution;
- summary of Timeformer family and key findings.

Proposed contribution bullets:

- formulate temporal semantic traceability as a `token@time` representation
  problem;
- introduce a Timeformer ablation family with continuous time conditioning and
  token-time interaction;
- evaluate traceability using context drift score, contrastive sign flip, and
  probes;
- analyze why naive historical prototype memory is insufficient in the current
  setup.

### 2. Related Work

Goal: situate the paper in the right field.

Required clusters:

- diachronic word embeddings;
- Lexical Semantic Change Detection (LSCD);
- temporal conditioning in neural language models;
- probing/diagnostic evaluation, briefly;
- multi-sense or sense-aware representations, briefly as future direction.

References that should be included:

- Hamilton et al. (2016), diachronic word embeddings;
- SemEval 2020 Task 1, standard LSCD benchmark;
- Tahmasebi et al. (2021), survey on computational approaches to semantic
  change;
- TimeSformer or factorized attention reference if the temporal attention
  analogy is kept.

Positioning sentence:

Unlike classical LSCD pipelines that often compare or align representations
across independently modeled periods, this paper studies whether a single
architecture can expose a directly queryable `token@time` representation.

### 3. Controlled Benchmark Summary

Goal: provide only enough context to understand the experiment.

Must include:

- synthetic SVO corpus;
- subjects with planted temporal trajectories;
- contexts A/B as semantic neighborhoods;
- `true_context` as evaluation label only;
- train/test, ambiguous, contrastive, and continuation splits;
- pointer to the KDMiLe benchmark paper or technical report for full
  methodology.

Avoid:

- duplicating trajectory-generation details;
- presenting the corpus as the main IBERAMIA contribution;
- spending too much space on hard split details unless used in results.

### 4. Timeformer Architecture

Goal: present the ablation chain as the technical object.

Subsections:

1. Static Transformer.
2. Continuous TimeEncoding.
3. Additive Time-Conditioned Transformer.
4. Token-Time Transformer.
5. Memory-Augmented Timeformer.
6. PrototypeMemory and causality.

Emphasis:

The main positive architectural mechanism is Token-Time Interaction. Memory is
an extension and diagnostic.

### 5. Evaluation Protocol

Goal: explain what is measured and why.

Must include:

- all models trained with MLM;
- `true_context` not used during model training;
- probe over `h(subject)`;
- context drift score as primary geometric traceability metric;
- contrastive masked-token test;
- precision@k or neighborhood coherence;
- continuation split for held-out epochs;
- `B3` controls: shuffled/nohistory/oracle.

Key rationale:

The subject token is the planted drifting entity. Therefore, diagnostics over
`h(subject)` are closest to the traceability claim.

### 6. Results

Suggested ordering:

1. **Context drift score:** main showcase.
2. **Contrastive temporal inversion:** same surface form, different epochs.
3. **Probe accuracy:** supportive diagnostic.
4. **Memory diagnostics:** learned memory, shuffled/nohistory, oracle.

Recommended narrative:

- `B2b` shows stronger temporal neighborhood movement than `B1`.
- `B2a` may be close to `B2b` in probe accuracy, so probe alone is insufficient.
- Contrastive and drift diagnostics better reveal the traceability effect.
- `B3` is not a stable improvement over `B2b`, but its failure mode is
  informative and motivates sense-aware memory.

### 7. Discussion

Goal: interpret the results without overclaiming.

Points to cover:

- traceability vs raw predictive performance;
- why global time conditioning is not the same as token-time interaction;
- why memory may be hard to exploit under MLM in short SVO sequences;
- why mean-pooled prototypes collapse coexisting senses;
- why this motivates the next paper on sense-aware Timeformer.

### 8. Conclusion

Goal: restate the narrow contribution.

Conclude that explicit token-time conditioning can make semantic
representations temporally traceable in a controlled setting, while naive
historical prototype memory remains insufficient and motivates sense-aware
extensions.

---

## 7. Figures and Tables Needed

### Figure 1: Conceptual `token@time`

Status: created as a combined conceptual and ablation-chain figure.

Artifacts:

- `outputs/figures/figure1_ablation_chain.html`;
- `outputs/figures/figure1_ablation_chain.png`;
- `outputs/figures/figure1_ablation_chain.json`.

Show the same token at two epochs moving from context A toward context B, then
connect this idea to the Timeformer ablation chain.

Purpose:

Make traceability visually obvious early in the paper.

### Figure 2: Timeformer Ablation Chain

Status: Figure 1 covers the ablation chain. Figure 2 is now reserved for the
architecture detail.

Artifacts for architecture detail:

- `outputs/figures/figure2_architecture_detail.html`;
- `outputs/figures/figure2_architecture_detail.png`;
- `outputs/figures/figure2_architecture_detail.json`.

Show:

```text
Static → Additive Time → Token-Time → Memory-Augmented
```

Include the mechanism added at each step.

Updated role:

Use Figure 2 to explain the Token-Time Transformer core and the optional
Memory-Augmented Timeformer extension, including the corrected detail that
temporal attention updates the subject representation before the Transformer
encoder.

### Figure 3: Context Drift Score

This should be the central empirical figure.

Axes:

- x-axis: epoch;
- y-axis: context-A proportion among k-nearest neighbors;
- lines: Static Transformer, Token-Time Transformer, Memory-Augmented
  Timeformer;
- separate panels or line styles for stable, drift, and bifurcation classes;
- optional shaded region: seed variability.

Purpose:

Directly show that Token-Time Transformer traces the intended semantic drift
more strongly than Static Transformer, while revealing the smaller but nonzero
temporal movement in stable and bifurcating subjects.

### Figure 4: Memory-Augmented Timeformer

Show current subject representation attending to historical prototypes
`m(S,t<k)`.

Include even if memory results are diagnostic, because the architecture needs a
clear visual explanation.

### Table 1: Model Variants

Columns:

- internal ID;
- public name;
- temporal signal;
- memory;
- expected diagnostic.

### Table 2: Main Traceability Metrics

Columns:

- context drift delta;
- contrastive sign flip;
- ambiguous probe accuracy;
- continuation probe accuracy.

Rows:

- Static Transformer;
- Additive Time-Conditioned Transformer;
- Token-Time Transformer;
- Memory-Augmented Timeformer.

### Table 3: Memory Diagnostics

Rows:

- `B2b_ref`;
- `B3_learned`;
- `B3_shuffled`;
- `B3_nohistory`;
- `B3_oracle`.

Columns:

- continuation accuracy;
- contrastive sign flip;
- Delta against `B2b_ref`.

Purpose:

Show that memory is diagnostic rather than the main positive result.

---

## 8. Required Work Before Drafting

Already done or mostly done:

- corrected continuation split excludes t8-t9 from training;
- memory aligned with best checkpoint;
- gradient path for temporal cross-attention fixed;
- 3 seeds for main comparisons: runs `20260523_006` to `20260523_008`;
- public model labels added to result outputs.
- corrected `B3_oracle` rerun: run `20260524_001`;
- class-specific context drift regenerated for runs `20260523_006` to
  `20260523_008`.

Still needed:

- decide exact `B3` controls included in the final table;
- collect exact BibTeX/citations for LSCD/SemEval related work;
- decide whether `B3_oracle` belongs in main results or diagnostic table only.

Minimum acceptable IBERAMIA paper:

- context drift score clearly separates `B2b` from `B1`;
- contrastive evaluation supports temporal inversion;
- probe results are reported as auxiliary diagnostics;
- `B3` status is explicit and not overclaimed;
- benchmark is summarized but not duplicated from KDMiLe.

---

## 9. Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| `B3` does not improve over `B2b` | High | Frame memory as diagnostic; emphasize Token-Time Transformer |
| `B2a` close to `B2b` in probe accuracy | Medium | Use context drift score and contrastive sign flip as main evidence |
| Synthetic corpus seen as too artificial | Medium | Present it as controlled traceability testbed; point to KDMiLe paper |
| Reviewer expects SOTA comparison | Medium | State architecture/diagnostic scope clearly |
| MLM objective does not reward memory use | High | Discuss as finding; motivate auxiliary or sense-aware objectives |
| Mean-pool memory collapses senses | Medium | Use as bridge to sense-aware Timeformer |
| Two-paper split unclear to reviewers | Medium | In IBERAMIA, explicitly say benchmark details are in companion/technical report |

---

## 10. Decisions for the IBERAMIA Draft

Settled:

1. Main positive result: Token-Time Transformer.
2. `B3`: diagnostic extension, not main positive claim.
3. Continuation: secondary evidence, not the central showcase.
4. Context drift score: central figure/result.
5. Probe accuracy: auxiliary diagnostic.
6. Title: **Timeformer: Explicit Temporal Conditioning for Traceable Semantic
   Representations**.

Still open:

1. Whether to include `B3_oracle` in main results or appendix-style discussion.
2. Whether to mention KDMiLe as accepted/submitted/companion depending on timing.
3. How compact the benchmark section must be for page limits.

---

## 11. Questions for Second Opinion

Please review this plan assuming the benchmark contribution is being handled in
a separate KDMiLe paper. For IBERAMIA, focus on whether the architecture and
traceability story is strong enough.

Questions:

1. Is the IBERAMIA contribution sufficiently clear if the main positive result
   is Token-Time Transformer and `B3` is diagnostic?
2. Should the title use "Timeformer" for the whole family, or should it name
   "Token-Time Transformer" more directly?
3. Is context drift score strong enough to carry the central empirical claim?
4. Should `B3_oracle` be in the main results, given that it improves
   contrastive sign flip but not continuation accuracy?
5. How much benchmark detail is needed in IBERAMIA if KDMiLe owns the benchmark
   methodology?
6. Which related-work framing should be most prominent: LSCD, diachronic
   embeddings, or temporal conditioning in neural language models?

---

## 12. Segundo Parecer — Claude Code (2026-05-24)

### Avaliação geral

Versão substancialmente melhorada. A separação Paper A/B no §0 é a mudança
mais importante do documento — resolve o problema de escopo antes de qualquer
argumento científico. O context drift score está corretamente posicionado como
Figure 3 e primeiro no ordering de resultados. As decisões do §10 estão bem
tomadas. O documento pode guiar a escrita sem reformulação estrutural adicional.

### Problema de protocolo nos dados do B3-oracle

O §5.4 cita resultados de `B3_oracle` do run `20260523_005`. Esse run usou o
protocolo antigo com continuation leakage (t8/t9 no treino). Os runs
`20260523_006` a `20260523_008` corrigiram esse protocolo, mas o oracle
diagnostic não foi rerodado com a versão corrigida.

Consequência prática: os números do B3-oracle na Table 3 precisam ser
revalidados antes de entrar no paper. A interpretação do §5.4 pode estar
correta, mas os valores exatos (continuation=0.7670, sign_flip=0.7586) são
de um experimento com dados contaminados.

Ação: rodar `scripts/oracle_diagnostic.py --run-id 20260523_006` para
obter os números do B3-oracle com o protocolo correto. Só depois decidir
se B3-oracle vai para os resultados principais ou para discussão.

### Figure 3: adicionar sujeitos estáveis como controle

O plano atual mostra apenas sujeitos em deriva. A figura fica muito mais
convincente com três linhas por modelo: deriva (descende) e estável (plana).
Isso demonstra que B2b está capturando o drift específico plantado, não
simplesmente mudando todas as vizinhanças com o tempo.

Sugestão para o eixo y:
- linha sólida: sujeitos em deriva
- linha tracejada: sujeitos estáveis
- região sombreada: variância entre seeds

Com o corpus atual temos os dados — `neighbor_analysis.py` já calcula
o drift score por classe.

### Decisões ainda abertas são decidíveis agora

**Título:** das três opções, "Timeformer: Explicit Temporal Conditioning for
Traceable Semantic Representations" é a mais clara. "Token-at-Time" é inglês
incomum. A terceira opção repete "Token-Time" duas vezes. Recomendo fechar
com a primeira opção.

**B3-oracle em resultados principais ou discussão:** depende do resultado
após reexecução com protocolo correto. Se o sign_flip se confirmar (oracle
melhora contrastive mas não continuation), pertence a Table 3 com
interpretação explícita. Se os números mudarem substancialmente com o
protocolo corrigido, avaliar novamente.

**KDMiLe como referência:** se não estiver submetido no momento da submissão
do IBERAMIA, usar "technical report" ou deixar a seção de benchmark
autocontida o suficiente para revisores que não terão acesso ao KDMiLe.
Não assumir que companion paper estará disponível.

### Sobre as perguntas do §11

**Q1 (contribuição suficientemente clara sem B3 positivo?):** Sim, para
IBERAMIA. Token-time conditioning + context drift score + contrastive
evaluation é uma contribuição arquitetural e diagnóstica coerente. O campo
de semantic change tem muitos papers com corpus controlado e resultados
modestos — o que importa é que a evidência geométrica (drift score) seja
direta.

**Q2 (título com "Timeformer" ou "Token-Time Transformer"?):** Timeformer
como nome da família tem mais peso de branding. Usar no título; explicar no
abstract que Token-Time Transformer é o componente que carrega o resultado
principal.

**Q3 (context drift score suficiente para o claim central?):** Sim, com a
adição dos sujeitos estáveis como controle. A descida de 0.847→0.233 para
B2b versus 0.618→0.387 para B1, com std < 0.05 nas 3 seeds, é evidência
geométrica direta do claim. Mais forte do que qualquer delta de probe accuracy.

**Q4 (B3-oracle em resultados principais?):** Após reexecução com protocolo
correto, provavelmente sim — mas na Table 3 de diagnósticos de memória,
não na Table 2 de métricas principais. A interpretação cuidadosa do §5.4
("oracle memory does not solve continuation but affects contrastive behavior")
é o ponto mais interessante e deve ficar na seção de resultados, não só
na discussão.

**Q5 (quanto benchmark no IBERAMIA?):** O mínimo é: corpus SVO sintético,
30 sujeitos × 10 épocas, três classes de trajetória (estável/deriva/
bifurcação), true_context como label de avaliação apenas, splits listados
por nome. Uma tabela com as estatísticas do corpus e um parágrafo basta.
O resto aponta para KDMiLe ou technical report.

**Q6 (qual cluster de related work mais proeminente?):** LSCD/SemEval 2020
deve ser o mais proeminente — é o campo com que revisores especializados
vão comparar, e posicionar a diferença (representação diretamente consultável
vs. alinhamento de espaços independentes) é o argumento de novidade mais
claro. Diachronic embeddings é contexto histórico, uma subseção. Temporal
conditioning em LMs é opcional dado o escopo.

### Checklist atualizado

| Item | Status |
|------|--------|
| Continuation held-out real | ✅ feito (runs 006–008) |
| Memória alinhada ao melhor checkpoint | ✅ feito |
| Caminho de gradiente correto para B3 | ✅ feito |
| 3 seeds para comparações principais | ✅ feito |
| Context drift score calculado e agregado | ✅ feito (neighbor_analysis.py) |
| Contrastive summaries nas 3 seeds | ✅ disponível em results_full.json |
| B3-oracle com protocolo correto | ✅ feito (`20260524_001`) |
| Figure 3 com sujeitos estáveis como controle | ✅ feito (`outputs/figures/figure3_context_drift.png`) |
| Tabela final com nomenclatura pública | ✅ regenerada para runs 006–008 e 20260524_001 |
| Título fechado | ✅ opção 1 |
| LSCD/SemEval 2020 adicionado ao related work plan | ✅ feito; falta BibTeX |
| Decisão sobre KDMiLe como referência | ❌ verificar status de submissão |

---

## 13. Follow-up After Second Opinion

Actions completed after the second review:

- reran `B3_oracle` with corrected continuation protocol:
  `outputs/runs/20260524_001`;
- extended `scripts/neighbor_analysis.py` with `drift_score_by_class`;
- regenerated `neighbor_analysis.json` for runs `20260523_006`,
  `20260523_007`, and `20260523_008`;
- regenerated `results_table.csv` and `ablation_table.json` for runs
  `20260523_006`, `20260523_007`, `20260523_008`, and `20260524_001`;
- locked the IBERAMIA title:
  **Timeformer: Explicit Temporal Conditioning for Traceable Semantic
  Representations**.
- created Figure 3 with Plotly Express:
  - `outputs/figures/figure3_context_drift.html`;
  - `outputs/figures/figure3_context_drift.png`;
  - `outputs/figures/figure3_context_drift_data.csv`.
- created Figure 1 as a Plotly diagram:
  - `outputs/figures/figure1_ablation_chain.html`;
  - `outputs/figures/figure1_ablation_chain.png`;
  - `outputs/figures/figure1_ablation_chain.json`.
- created Figure 2 as a Plotly architecture diagram:
  - `outputs/figures/figure2_architecture_detail.html`;
  - `outputs/figures/figure2_architecture_detail.png`;
  - `outputs/figures/figure2_architecture_detail.json`.

Updated interpretation:

- `B3_oracle` remains diagnostic: it does not improve continuation over
  `B2b_ref`, but it improves contrastive sign flip in the corrected run.
- Class-specific context drift is useful, but stable subjects are not perfectly
  flat. The figure should therefore emphasize relative movement: drift shows
  the strongest transition, bifurcation an intermediate transition, and stable
  subjects weaker/non-central movement.

Plotting note:

The source figure is Plotly Express. Kaleido export was unstable on this macOS
environment, so the final PNG was rendered from the self-contained Plotly HTML
using Chrome headless. The HTML remains the editable/reviewable source artifact.

Immediate next artifact:

Create the final diagnostic memory table deciding whether `B3_oracle` stays in
main results or a focused diagnostic subsection.
