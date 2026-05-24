# Paper 2: Sense-Aware Timeformer — Ideias e Plano

**Status:** rascunho de conceito (pré-implementação)
**Data:** 2026-05-24
**Contexto:** continuação direta do Paper 1 (Timeformer / Semantic Traceability)

---

## 1. Motivação Central

O Paper 1 demonstra que `TokenTimeInteraction + TimeEncoding` cria vizinhanças
temporalmente discriminativas (context drift score B2b: 0.847 → 0.233, t0→t9).
Mas a `PrototypeMemory` do B3 usa:

```
m(S, t) = mean_pool(h(S) nas frases de S em época t)
```

Esse mean_pool **colapsa sentidos coexistentes numa mesma época**. Para sujeitos
bifurcantes (S21–S30), onde `P(ctx=A|S,t) ≈ 0.5`, o protótipo médio fica no
meio do espaço, equidistante de A e B — pouco informativo.

O oracle diagnostic confirma: mesmo com P(ctx=A|S,t) perfeito codificado na
memória, B3 não supera B2b (ΔA_cont = 0.000 ± 0.013). Isso sugere que o
problema não é só a qualidade do protótipo — é que mean_pool destrói a estrutura
multi-sentido antes de chegar à cross-attention.

**A pergunta do Paper 2:**

> Como rastrear `sentido(token)@tempo` em vez de apenas `token@tempo`?

---

## 2. Hipótese Principal

Se substituirmos a memória média por uma **memória multi-protótipo**:

```
M(S, t) = { m_1(S,t), m_2(S,t), ..., m_K(S,t) }
```

onde cada `m_k` representa um estado semântico induzido automaticamente, então:

1. Sujeitos bifurcantes terão protótipos separados para ctx=A e ctx=B.
2. A cross-attention do B3 poderá consultar trajetórias de sentido distintas.
3. Eventos semânticos (split, merge, drift, birth, death) tornam-se detectáveis.

Experimento de verificação (já possível com corpus atual):

```
B3-mean-memory:        m(S,t) = mean_pool       ← baseline atual
B3-oracle-sense:       m_A(S,t), m_B(S,t) usando true_context  ← upper bound
B3-auto-sense:         m_k(S,t) por clustering não supervisionado
```

Se `oracle-sense >> mean`, o problema é o colapso.
Se `auto-sense ≈ oracle-sense`, o Canonizer funciona.
Se nem `oracle-sense` melhora, o problema é a cross-attention em si (já vimos
esse risco no Paper 1 — tratar com cuidado).

---

## 3. Arquitetura Proposta

### 3.1 Pipeline geral

```
Transformer contextual (congelado ou fine-tuned)
    ↓
h(token, contexto_i, tempo_i)  — embeddings por ocorrência
    ↓
┌─────────────────────────────┐
│     Semantic Canonizer       │
│  clustering + alinhamento   │
└─────────────────────────────┘
    ↓
M(token, sense_k, tempo_t)  — memória multi-protótipo
    ↓
B3 cross-attention temporal  (agora sobre K×T vetores)
    ↓
h(token)@tempo atualizado
```

### 3.2 Semantic Canonizer

**Entrada:** todas as ocorrências contextualizadas de um token `S` até época `t`:

```
{ h(S, ctx_1, t_1), h(S, ctx_2, t_2), ..., h(S, ctx_n, t_n) }
```

**Saída:** K protótipos semânticos por época:

```
{ m_1(S,t), ..., m_K(S,t) }
```

**Passos internos:**

1. **Indução de sentidos (WSI):** clustering dos embeddings de ocorrência.
2. **Atribuição de sentido:** cada ocorrência → sense_k mais próximo.
3. **Construção de protótipos por (sentido, época):** mean_pool por cluster.
4. **Alinhamento temporal:** `m_k(S,t)` ↔ `m_k(S,t+1)` via Hungarian matching.
5. **Detecção de eventos:** split/merge/birth/death entre épocas adjacentes.

### 3.3 Integração com B3

A cross-attention atual recebe `memory: (batch, T, d_model)` — uma sequência de
protótipos por época. Com multi-protótipo, o shape passa a ser:

```
memory: (batch, T * K, d_model)
```

ou, com indexação explícita por sentido:

```
memory: (batch, T, K, d_model)  →  reshape → (batch, T*K, d_model)
```

O gated residual (`alpha=0` inicial) continua válido — B3 começa idêntico a
B2b e só usa memória se o gradiente justificar.

---

## 4. Métodos de Clustering Candidatos

| Método | Prós | Contras |
|--------|------|---------|
| K-Means (K fixo) | simples, reprodutível | K manual; collapsa forma |
| K-Means + BIC/AIC | K automático via modelo | assume gaussiano |
| HDBSCAN | K automático; ruído explícito | sensível a hiperparâmetros |
| Gaussian Mixture (GMM) | K automático; soft assignment | custo; K instável |
| DP-Means | online; K cresce com dados | convergência fraca |
| Spectral | captura forma não-convexa | custo quadrático em N |

**Recomendação para primeira implementação:** K-Means com K∈{2,3} avaliado por
silhouette + coerência com `true_context` (no corpus artificial). Depois HDBSCAN
como alternativa não-paramétrica.

---

## 5. Alinhamento Temporal de Protótipos

**Problema:** clustering independente por época produz permutações arbitrárias.
`m_1(S, t0)` pode corresponder a ctx=A mas `m_1(S, t1)` a ctx=B.

**Métodos de alinhamento:**

### 5.1 Hungarian Matching (recomendado para primeira versão)

Para cada par de épocas adjacentes (t, t+1):

```python
cost_matrix[k, j] = 1 - cosine_sim(m_k(S,t), m_j(S,t+1))
assignment = linear_sum_assignment(cost_matrix)  # scipy
```

Reatribui índices de forma a maximizar similaridade entre épocas consecutivas.
**Propagação:** alinhar t0→t1, depois manter índice de t1 para alinhar t1→t2, etc.

### 5.2 Nearest-neighbor com threshold

Mais simples: `m_k(S,t+1)` herda índice do vizinho mais próximo em `t`, se
`cos > θ`. Sentidos novos (dist < θ a todos) recebem índice novo → birth event.

### 5.3 Regularização temporal de suavidade

Se o Canonizer for uma camada treinável: adicionar loss de suavidade:

```
L_smooth = Σ_t ||m_k(S,t+1) - m_k(S,t)||²
```

Força trajetórias suaves. Pode suprimir eventos reais de split/merge — trade-off.

---

## 6. Detecção de Eventos Semânticos

Após alinhamento, comparar protótipos em t e t+1:

| Evento | Critério |
|--------|----------|
| **Drift** | `cos(m_k(S,t), m_k(S,t+1)) < θ_drift` |
| **Split** | K(S,t+1) > K(S,t) e novo centróide distante dos antigos |
| **Merge** | K(S,t+1) < K(S,t) e dois protótipos convergiram |
| **Birth** | K(S,t+1) > K(S,t) e novo centróide em região esparsa |
| **Death** | K(S,t+1) < K(S,t) por falta de suporte empírico |

No corpus artificial, os sujeitos bifurcantes (S21–S30) devem exibir **split**
em alguma época central. Isso é verificável contra `true_context`.

---

## 7. Conexão com Literatura

### 7.1 Word Sense Induction (WSI)

- Reisinger & Mooney (2010): multi-prototype word vectors — fundação teórica direta.
- Neelakantan et al. (2014): multiple embeddings per word.
- Shi et al. (2021): contextualized WSI com BERT.

### 7.2 Lexical Semantic Change Detection (LSCD)

- Hamilton et al. (2016): *Diachronic Word Embeddings* — baseline do campo.
- Kulkarni et al. (2015): *Statistically Significant Detection of Linguistic Change*.
- SemEval 2020 Task 1: benchmark padrão de LSCD (inglês, alemão, latim, sueco).
- Tahmasebi et al. (2021): survey abrangente de abordagens computacionais.

### 7.3 Temporal Sense Tracking

- Frermann & Lapata (2016): *A Bayesian Model of Diachronic Meaning Change* —
  modelo probabilístico com sentidos latentes ao longo do tempo. **O antecessor
  mais direto do Paper 2.** Deve ser citado e comparado.
- Kutuzov & Giulianelli (2020): probing temporal sense change.
- Rosin & Radinsky (2022): *Time Masking for Temporal Language Models*.

### 7.4 Posicionamento do Paper 2

O Paper 2 se diferencia de Frermann & Lapata (2016) por:
1. Usar Transformers contextualizados em vez de LDA/tópicos.
2. Ter ground truth controlado para validação (corpus artificial).
3. Integrar o Canonizer com uma arquitetura de downstream tracking (B3).
4. Tornar os eventos (split/merge/birth/death) explicitamente consultáveis.

---

## 8. Experimentos Planejados

### 8.1 Validação da Canonização (offline, com corpus artificial)

1. **Qualidade do clustering:** rodar K-Means/HDBSCAN nos embeddings de h_subj
   do B2b por (sujeito, época). Medir ARI/NMI contra `true_context`.
2. **Comparação de protótipos:** mean vs. multi-prototype, por classe de sujeito.
3. **Detecção de bifurcação:** para S21–S30, os clusters devem se separar.
   Medir F1 de detecção de split event vs. ground truth.

### 8.2 B3-oracle-sense vs B3-mean

Substituir `PrototypeMemory` por `MultiSenseMemory` onde `m_k(S,t)` vem do
`true_context` (oracle). Rodar B3 com essa memória e comparar:
- Probe accuracy (test, ambiguous_test, continuation)
- Context drift score
- Context coherence

Se `oracle-sense >> oracle-mean`, a bifurcação era de fato o gargalo.

### 8.3 B3-auto-sense (não supervisionado)

Usar K-Means nos embeddings do B2b para induzir sentidos. Medir gap entre
oracle-sense e auto-sense como proxy de qualidade do Canonizer.

### 8.4 Alinhamento temporal

Comparar:
- Sem alinhamento (permutação arbitrária por época)
- Hungarian matching
- Regularização de suavidade

Métrica: consistência temporal dos protótipos (ARI entre rótulos de sentido
em épocas adjacentes).

### 8.5 Detecção de eventos em dados reais (stretch goal)

Aplicar o Canonizer a um corpus real com marcação temporal (e.g., jornal, Reddit,
Twitter) e comparar eventos detectados com mudanças documentadas.

---

## 9. Corpus: Extensões Necessárias

O corpus atual (30 sujeitos × 10 épocas, 5000 frases) é pequeno para validar
clustering. Para Paper 2:

- **Mais frases por (sujeito, época):** pelo menos 50–100 (hoje ~17).
- **Mais sujeitos bifurcantes:** aumentar de 10 para 20 ou 30.
- **Gradientes de split mais variados:** splits em épocas diferentes (não só
  t4–t5), splits parciais (K=2 com pesos 0.3/0.7), etc.
- **Corpus v3** com essas extensões como baseline de Paper 2.

---

## 10. Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|-------|--------------|-----------|
| Oracle-sense não melhora B3 | Alta (já vimos com oracle-mean) | Focar na qualidade dos protótipos, não no ganho de probe; reformular claim |
| Clustering induz tópicos/gênero, não sentidos | Média | Adicionar baseline de "topic clustering" e comparar; usar contrastive loss por sentido |
| Alinhamento temporal falha em splits rápidos | Média | Testar com splits graduais vs. abruptos; usar regularização |
| K automático instável entre seeds | Alta | Fixar K=2 para corpus com 2 contextos; escalar gradualmente |
| Escopo muito grande para segundo paper | Alta | Separar em (a) corpus + canonização e (b) integração com B3 |

---

## 11. Claim Central do Paper 2 (rascunho)

> "Mean pooling destroys co-existing senses within an epoch. We propose Semantic
> Canonization — automatic induction of canonical sense prototypes from
> contextualized embeddings — and show that multi-prototype temporal memory
> captures bifurcation dynamics that single-prototype memory collapses. Using
> a controlled synthetic corpus with ground-truth sense annotations, we demonstrate
> that oracle sense-aware memory substantially improves temporal traceability for
> sense-bifurcating lexical items."

---

## 12. Divisão de Trabalho com Paper 1

| Componente | Paper 1 | Paper 2 |
|------------|---------|---------|
| TimeEncoding | contribuição | use as-is |
| TokenTimeInteraction | contribuição | use as-is |
| Context drift score | contribuição | use como baseline |
| PrototypeMemory (mean) | diagnóstico de limite | baseline a superar |
| Oracle-mean diagnostic | contribuição | motivação |
| Semantic Canonizer | futuro trabalho | contribuição central |
| Multi-prototype memory | mencionado | contribuição |
| Temporal event detection | não mencionado | contribuição |
| Corpus sintético v2 | contribuição | base |
| Corpus sintético v3 | não | contribuição |

---

## 13. Pendências Abertas

- [ ] Rodar context drift score por classe (stable/drift/bifurc) para confirmar
      que bifurcação é o único caso onde mean_pool claramente falha.
- [ ] Implementar `B3-oracle-sense` como experimento de prova de conceito.
- [ ] Definir critério de qualidade para o Canonizer além de ARI/NMI
      (coerência semântica interpretável por humanos?).
- [ ] Revisar Frermann & Lapata (2016) em detalhe para identificar o que reusar
      vs. o que superar.
- [ ] Decidir se Paper 2 usa corpus sintético (controlado, mais fácil) ou real
      (mais impacto, mais difícil de avaliar).
