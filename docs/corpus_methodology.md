# Corpus Artificial — Decisões de Construção

**Projeto:** Timeformer  
**Fase:** A (banco de prova para deriva semântica temporal)  
**Corpus:** `data/corpus.tsv` + `data/corpus_ambiguous.tsv`  
**Parâmetros:** `data/corpus.params.json`  
**Gerador:** `scripts/generate_corpus.py` → `src/corpus_generator.py`

---

## 1. Motivação e objetivo do corpus

O corpus artificial serve como banco de prova controlado para a hipótese do Timeformer: que uma arquitetura com dimensão temporal explícita aprende representações que preservam a vizinhança semântica de cada época. O corpus é artificial — não um corpus de linguagem natural — porque isso permite ground truth verificável: sabemos exatamente qual sujeito deveria estar associado a qual contexto em qual época, sem ambiguidade de anotação.

A deriva semântica é simulada via mudança nos padrões de co-ocorrência. Um sujeito S que deriva do contexto A para o contexto B muda seus vizinhos típicos no espaço de embeddings de {V1-V4, O1-O4} para {V5-V8, O5-O8} — análogo a um token de linguagem natural cujos co-textos típicos mudam ao longo de décadas.

---

## 2. Vocabulário

**Tamanho:** 46 tokens (30 sujeitos + 8 verbos + 8 objetos)

**Gramática:** SVO posicional estrita. Toda frase tem exatamente 3 tokens: sujeito verbo objeto. Não há ambiguidade posicional.

**Contextos semânticos:**
- Contexto A: verbos {V1, V2, V3, V4} e objetos {O1, O2, O3, O4}
- Contexto B: verbos {V5, V6, V7, V8} e objetos {O5, O6, O7, O8}

**Decisão:** vocabulário binário (cada verbo pertence exclusivamente a A ou B) para garantir que o ground truth seja verificável sem ambiguidade. Isso é uma simplificação em relação a linguagem natural, onde verbos são polissêmicos. Limitação reconhecida: um vocabulário com graus de associação (v2b) tornaria o experimento mais próximo do caso real, mas está fora do escopo da Fase A.

---

## 3. Sujeitos e fenômenos plantados

**30 sujeitos em 3 classes:**

| Classe | Sujeitos | Fenômeno |
|--------|----------|----------|
| Estável | S1–S10 | P(ctx=A \| S, t) = constante ao longo de todas as épocas |
| Deriva | S11–S20 | P(ctx=A \| S, t) decresce monotonicamente de ≈1.0 para ≈0.0 |
| Bifurcação | S21–S30 | P(ctx=A \| S, t) cai de ≈1.0 para plateau ≈0.5 e estabiliza |

**Decisão de usar 3 classes:** captura os três regimes qualitativamente distintos da mudança semântica temporal — estabilidade, deriva direcional e convergência para estado misto. A bifurcação em plateau ≈0.5 representa tokens cujo sentido se dividiu entre dois contextos coexistentes (análogo a "negro" com dois sentidos ativos simultaneamente).

**Decisão de usar 10 sujeitos por classe (em vez de 2 como no v1):** com 2 sujeitos por classe, qualquer resultado pode ser artefato da trajetória específica daquele par. Com 10, os resultados são atribuíveis à classe, não ao sujeito individual.

---

## 4. Trajetórias paramétricas

Em vez de trajetórias definidas manualmente, cada sujeito recebe parâmetros amostrados de uma família funcional com seed fixo.

**Famílias de trajetória:**

| Classe | Família | Parâmetros |
|--------|---------|------------|
| Estável | constante | valor P(A) ∈ [0.60, 1.0] |
| Deriva | linear | start ∈ [0.85, 1.0], end ∈ [0.0, 0.25] |
| Deriva | sigmoide | start, end + midpoint ∈ [0.3, 0.7] + steepness ∈ [0.8, 2.0] |
| Deriva | onset tardio | start, end + onset ∈ [t2, t5] |
| Bifurcação | degrau | start ∈ [0.9, 1.0], plateau ∈ [0.4, 0.6], onset ∈ [t2, t6], transition ∈ [2, 4 épocas] |

**Decisão de usar trajetórias paramétricas:** com trajetórias manuais seria impossível criar 30 trajetórias suficientemente distintas sem viés de seleção. Com famílias paramétricas, a diversidade emerge do espaço de parâmetros.

**Restrição de diversidade:** dentro de cada classe, nenhum par de sujeitos pode ter trajetórias com distância L2/√n < threshold (estável: 0.04; deriva/bifurcação: 0.10). Isso garante que os 10 sujeitos de cada classe sejam distinguíveis como instâncias separadas do fenômeno, não variações triviais uns dos outros. Para sujeitos estáveis, que variam apenas no nível de P(A), é usada uma grade uniforme em [0.60, 1.0] com jitter — a amostragem aleatória simples não consegue posicionar 10 valores com espaçamento mínimo em um range de 0.40.

**Reprodutibilidade:** todos os parâmetros são salvos em `data/corpus.params.json` junto com seed, p_canon e as frações geradas. O corpus pode ser regenerado deterministicamente.

---

## 5. Épocas

**10 épocas:** t0–t9

**Decisão:** 6 épocas (v1) eram insuficientes para distinguir deriva com onset tardio de bifurcação — ambas podiam ter trajetórias similares nas primeiras 6 posições. Com 10 épocas, a fase pós-onset (onde a bifurcação se estabiliza e a deriva continua) tem espaço suficiente.

---

## 6. Marcadores probabilísticos

**Parâmetro p_canon = 0.75** no corpus de treino e avaliação padrão.

Em uma frase cujo `true_context` é A, o verbo é amostrado de {V1-V4} com probabilidade 0.75 e de {V5-V8} com probabilidade 0.25. O objeto segue a mesma regra independentemente.

**Decisão:** marcadores determinísticos (v1) tornavam qualquer modelo que observasse o verbo um oráculo perfeito — a acurácia máxima teórica era 100% e a tarefa não testava nada além de lookup. Com p_canon=0.75, a acurácia máxima teórica de um modelo que usa apenas o verbo é ~75%, tornando a tarefa genuinamente informativa.

**`true_context`** é definido pelo fenômeno plantado (contexto A ou B para o par sujeito/época), não pelo verbo observado. O verbo é evidência ruidosa; o label é o contexto intencionado.

---

## 7. Splits de avaliação

O corpus tem quatro splits mutuamente exclusivos:

### 7.1 `train` (79%)
Frases usadas para treinar os modelos. Estratificado por (época, sujeito, true_context) para garantir proporções de contexto balanceadas.

### 7.2 `test` (15%)
Frases de avaliação com distribuição natural (p_canon=0.75). O contexto local é parcialmente informativo: na maioria das frases, verbo e objeto apontam na direção correta, mas com ruído.

### 7.3 `hard_verb` (4%)
Subconjunto de `test` onde o **verbo contradiz o `true_context`** mas o objeto ainda aponta corretamente. A evidência local é mista: um marcador erra, outro acerta. Testa robustez a ruído verbal.

### 7.4 `hard_both` (1.4%)
Subconjunto de `test` onde **verbo e objeto ambos contradizem o `true_context`**. A evidência local é enganosa na direção errada — nenhum marcador local aponta corretamente. Testa robustez contra evidência adversarial. Ocorrência natural com p_canon=0.75: (1-0.75)² ≈ 6% das frases de teste. **Limitação:** com ~69 frases e dois sujeitos sem representação (S17, S25), este split é pequeno demais para métricas por sujeito.

### 7.5 `ambiguous_test` (arquivo separado)
Corpus de avaliação gerado com **p_canon=0.50** usando as mesmas trajetórias plantadas do corpus principal. Com p=0.50, verbo e objeto de cada frase são amostrados independentemente com probabilidade igual de pertencer a A ou B — o contexto local é não-informativo. A única pista disponível é o sujeito e a época.

**Uso:** testa se o modelo usa a distribuição marginal `P(ctx=A | sujeito, época)` quando o contexto local não ajuda. O teto teórico de qualquer modelo que usa apenas contexto local neste split é 50%.

**Nota importante:** `ambiguous_test` com p=0.50 testa uso temporal marginal — não desambiguação contextual fine-grained intra-época. Para testar se o modelo distingue duas instâncias do mesmo token com contextos diferentes dentro de uma época, seria necessário evidência contextual sutil (não nula), o que está fora do escopo da Fase A.

---

## 8. Estatísticas do corpus gerado (seed=42, p_canon=0.75)

```
Corpus principal (data/corpus.tsv):
  Total de frases: 5.000
  Épocas × sujeitos: 10 × 30 ≈ 17 frases/sujeito/época
  Splits: train=3.953 | test=763 | hard_verb=215 | hard_both=69

Distribuição de splits por classe:
              estável   deriva  bifurcação
  train         1.350    1.329       1.274
  test            252      276         235
  hard_verb        70       77          68
  hard_both        28       18          23

Corpus ambíguo (data/corpus_ambiguous.tsv):
  Total de frases: 5.000 (apenas ambiguous_test)
  Proporção de verbo canônico: ≈51% (esperado 50%)
  Proporção de objeto canônico: ≈50%
```

---

## 9. Limitações para reportar na seção de metodologia

1. **Vocabulário binário:** verbos e objetos pertencem exclusivamente a um contexto. Não há polissemia lexical — a deriva é puramente distribucional (mudança de frequência de co-ocorrência). Isso simplifica a análise mas afasta o corpus de linguagem natural.

2. **Corpus pequeno:** ~17 frases/sujeito/época. Modelos com mais parâmetros são penalizados por overfitting. Os resultados devem ser interpretados como prova de conceito em regime de corpus pequeno, não como benchmark de escala.

3. **`hard_both` pequeno:** 69 frases totais, dois sujeitos sem representação. Métricas por sujeito neste split não são confiáveis.

4. **Bifurcação intra-época não totalmente testável:** o corpus captura bifurcação como mudança da distribuição marginal ao longo de épocas. Para testar que um modelo distingue instâncias A e B do mesmo token dentro de uma época, seriam necessárias representações contextualizadas por sentença — o que é responsabilidade da arquitetura Timeformer (Fase B), não do corpus.

5. **Sujeitos estáveis com P(A) < 1.0:** sujeitos estáveis têm P(A) ∈ [0.60, 1.0], não necessariamente 1.0. Isso é intencional (diversidade de trajetórias na classe estável), mas significa que "estável" pode ter P(A)=0.65, o que se sobrepõe ao plateau de alguns sujeitos bifurcados.
