# Timeformer — Decisões de Arquitetura

**Projeto:** Timeformer  
**Fase:** B (ablação de representação temporal)  
**Código:** `src/timeformer/models.py`, `src/timeformer/encoding.py`, `src/timeformer/memory.py`  
**Cadeia de ablação:** Static Transformer (`B1`) → Additive Time-Conditioned Transformer (`B2a`) → Token-Time Transformer (`B2b`) → Memory-Augmented Timeformer (`B3`)

---

## 1. Paradigma de treino: MLM + probe linear

O objetivo de treino de todos os modelos é Masked Language Modeling (MLM) sobre o corpus artificial. A métrica de avaliação é a acurácia de uma probe linear treinada sobre as representações de h(sujeito) para classificar o `true_context` ∈ {A, B}.

**Decisão:** a separação entre objetivo de treino (MLM) e métrica de avaliação (probe) é intencional e necessária por três razões:

1. **Não usa `true_context` durante treino.** O `true_context` é ground truth derivado do fenômeno plantado — se ele fosse sinal de treino, o modelo seria treinado diretamente para a tarefa que avaliamos. Com MLM, o modelo só vê tokens e épocas.

2. **Transferível para corpus natural.** MLM é auto-supervisionado — pode ser aplicado a qualquer corpus de texto sem anotação de deriva semântica. Um objetivo supervisionado direto (classificar A vs B) não teria equivalente em dados reais.

3. **A probe mede o que o modelo aprendeu implicitamente.** Se h(sujeito) é linealmente separável por `true_context`, o modelo aprendeu a codificar informação distribucional suficiente para distinguir épocas — sem nunca ter visto esse label.

**Alternativa considerada e descartada:** treinar B3 diretamente para prever `true_context` (classificação supervisionada). Descartada porque tornaria o experimento um benchmark fechado, não um modelo de linguagem.

---

## 2. Cadeia de ablação

A hipótese central é que representações temporalmente informadas são melhores para rastrear deriva semântica. Para isolar de onde vem o ganho, a arquitetura foi dividida em quatro modelos de complexidade crescente:

| Código | Nome público | O que acrescenta | Split principal de validação |
|--------|-------------|-----------------|------------------------------|
| B1 | Static Transformer | Transformer textual sem tempo — adversário base | — |
| B2a | Additive Time-Conditioned Transformer | TimeEncoding aditivo global | `ambiguous_test` |
| B2b | Token-Time Transformer | Interação token×época | `ambiguous_test` |
| B3 | Memory-Augmented Timeformer | Atenção temporal sobre memória histórica | `continuation` |

**Interpretação dos deltas:**
- **Delta time-conditioning = B2a − B1** em `ambiguous_test`: ganho de saber a época globalmente
- **Delta token-time interaction = B2b − B2a** em `ambiguous_test`: ganho de injetar época por token (vs. global)
- **Delta memory = B3 − B2b** em `continuation`: ganho de usar trajetória histórica específica do sujeito

**Decisão de usar `ambiguous_test` para B2 e `continuation` para B3:**

`ambiguous_test` (p_canon=0.50) remove toda pista local de contexto — o único sinal disponível é (sujeito, época). B2a e B2b exploram exatamente essa combinação. Mas `ambiguous_test` não distingue B2b de B3, porque ambos usam (sujeito, época) da sentença atual.

`continuation` (épocas t8–t9 com trajetória dos mesmos sujeitos) exige que o modelo generalize para épocas não vistas durante treino. B3 pode fazer isso via histórico de t0–t7; B2b não tem como — a interação token×época em t8 não foi vista antes.

---

## 3. TimeEncoding: features sinusoidais fixas + MLP

**Implementação:** `TimeEncoding(t) = MLP(sinusoidal(t / T))`

As features sinusoidais são vetores fixos (não aprendidos), computados por `register_buffer`:

```
freqs[i] = 1 / 10000^(2i / d_sin)
features = [sin(t/T * freqs), cos(t/T * freqs)]   → (d_sin,)
```

A MLP é `Linear(d_sin → d_sin) + GELU + Linear(d_sin → d_model)`.

**Decisão: features sinusoidais fixas em vez de `nn.Embedding(n_epochs, d_model)`:**

Embedding de época seria uma tabela de lookup com 10 linhas — funciona para as épocas vistas no treino, mas tem comportamento indefinido para t8/t9 no split `continuation`, pois essas épocas nunca foram amostradas como índices de embedding durante treino. Com features sinusoidais, a representação de t8/t9 é computada pela mesma fórmula analítica usada para t0–t7. O modelo vê t8/t9 como pontos numa curva contínua, não como índices desconhecidos.

**Decisão: normalização t/T (em vez de t bruto):**

Normalizar para [0, 1] garante que as frequências dos senos/cosenos sejam as mesmas independentemente de quantas épocas existem. Um corpus com 20 épocas produziria as mesmas features para t=0 e t=1 que um corpus com 10 épocas.

**Decisão: d_hidden = d_sin na MLP (não > d_sin):**

Se a MLP fosse larga (d_hidden >> d_sin), ela poderia memorizar os 10 valores de entrada específicos do treino como pontos isolados, em vez de aprender uma função contínua. Com d_hidden = d_sin, a capacidade da MLP é restrita para forçar a generalização pela estrutura sinusoidal.

---

## 4. B2a vs B2b: global vs token×época

**B2a:** o mesmo vetor `TimeEncoding(t)` é somado a todos os tokens da sentença atual:
```
embedding = TokenEmb(token) + PosEmb(pos) + TimeEncoding(t)
```

**B2b:** `TimeEncoding(t)` interage individualmente com cada token via `TokenTimeInteraction`:
```
tok_t = Linear([TokenEmb(token); TimeEncoding(t)])
embedding = tok_t + PosEmb(pos)
```

**Decisão de manter B2a como ablação intermediária:**

B2a não modula sujeito e verbo diferentemente — injeta época como ruído global. Se B2b > B2a em `ambiguous_test`, o ganho vem especificamente da interação token-específica, não de qualquer exposição à época. B2a isola a hipótese nula "basta dizer ao modelo em qual época estamos".

**Decisão de usar concatenação + projeção linear (em vez de soma ou produto):**

Soma `TokenEmb + TimeEncoding` (como em B2a) trata epoch como offset global no espaço de embeddings — qualquer relação entre token e época é mediada apenas pela atenção posterior. Produto `TokenEmb * TimeEncoding` seria um gating multiplicativo sem norma garantida. A concatenação + projeção linear permite que a rede aprenda qualquer combinação bilinear entre as duas dimensões, incluindo soma e produto como casos especiais.

---

## 5. B3: atenção temporal fatorada

**Arquitetura B3:**
1. Embedding com interação token×época (igual ao B2b)
2. Encoder textual Transformer sobre os tokens da sentença atual
3. Cross-attention temporal: `h(sujeito)` como query, `{m(S, t_0..t_{k-1})}` como key/value
4. Residual + LayerNorm sobre o h(sujeito) atualizado

**Decisão: atenção fatorada em vez de atenção completa 2D (texto × tempo):**

Atenção completa processaria todas as combinações (token, época) em um único mecanismo, resultando em uma sequência de tamanho `seq_len × n_epochs`. Para 5 tokens × 10 épocas, isso é 50 posições — viável para nosso corpus, mas não escalável. A atenção fatorada mantém as dimensões textual e temporal separadas, análoga à TimeSformer (atenção espacial + atenção temporal separadas). Isso também facilita a interpretação: o que h(sujeito) buscou no histórico fica isolado no módulo `_TemporalCrossAttention`.

**Decisão: apenas h(sujeito) faz cross-attention temporal:**

A atenção temporal é aplicada apenas ao token de sujeito, não a todos os tokens. O sujeito é a entidade cuja deriva semântica queremos rastrear — faz sentido que apenas ele consulte o histórico de si mesmo. Aplicar atenção temporal a verbos e objetos aumentaria parâmetros sem hipótese clara de benefício (verbos e objetos não têm trajetórias individuais no corpus).

**Decisão: memória histórica injetada externamente:**

O B3 não computa nem armazena a memória internamente — ela é injetada pelo `MLMTrainer` como tensor `(batch, hist_len, d_model)`. Isso separa a lógica de treino da arquitetura: o modelo não precisa saber de onde vêm os protótipos, apenas como usá-los.

---

## 6. PrototypeMemory: stop-gradient, causal, apenas split de treino

**Definição:** `m(S, t) = mean_pool(h(S) nas frases de S em época t, split='train')`

**Propriedades implementadas:**

| Propriedade | Implementação | Razão |
|-------------|--------------|-------|
| Stop-gradient | `@torch.no_grad()` no `update()` | Evita dependência cíclica: modelo → protótipos → loss → modelo |
| Causal | `get()` retorna apenas t < epoch_k | Garante que o modelo não vê o futuro durante treino ou avaliação |
| Split de treino | dataloader contém apenas split='train' | Protótipos de test/continuation seriam data leakage |
| Atualização por época | chamado ao fim de cada época de treino | Protótipos refletem o modelo atual, não o de épocas anteriores |
| t0 sem histórico | `get()` retorna tensor vazio (0, 0, d_model) | B3 degrada graciosamente para B2b na primeira época |

**Decisão: mean-pool em vez de representação mais sofisticada:**

Mean-pool é a operação mais simples que produz um protótipo de época. Alternativas (CLS token treinado, atenção sobre frases) adicionariam parâmetros e hipóteses. O objetivo aqui é ter um protótipo representativo de h(S) em cada época, sem enriquecer a memória com capacidade adicional que poderia mascarar o efeito da atenção temporal.

**Decisão: protótipos atualizados ao fim de cada época de treino (não por batch):**

Atualização por batch criaria um problema de bootstrap: os protótipos do começo da época foram computados com o modelo do fim da época anterior, mas seriam misturados com protótipos computados com o modelo corrente no mesmo loop. A atualização por época garante que todos os protótipos de uma época são computados com o mesmo snapshot do modelo.

---

## 7. Controles B3-shuffled e B3-nohistory

Para separar se o ganho de B3 vem do histórico específico (trajetória do sujeito correto) ou apenas dos parâmetros extras da atenção temporal:

**B3-shuffled-subject:** cada sujeito recebe o histórico de outro sujeito (permutação aleatória das trajetórias). A arquitetura é idêntica ao B3, mas a memória histórica pertence ao sujeito errado.
- Se B3 > B3-shuffled: o ganho vem da trajetória específica do sujeito, não dos parâmetros adicionais.
- Se B3 ≈ B3-shuffled: a atenção temporal não discrimina históricos — o ganho é espúrio.

**B3-nohistory:** memória toda zerada (`_valid = False`). O módulo de atenção temporal recebe apenas zeros como key/value e, pela lógica do `_TemporalCrossAttention`, não modifica h(sujeito).
- Se B3 > B3-nohistory: a atenção temporal faz algo mesmo além do B2b subjacente.

**Decisão de exigir B3 > B3-shuffled como critério de sucesso:**

O critério mínimo aceitável para o paper é que B3 supere B3-shuffled em `continuation`. Isso é a garantia de que o modelo aprendeu trajetórias específicas, não apenas que a arquitetura de atenção temporal tem mais capacidade.

---

## 8. Hiperparâmetros default

| Parâmetro | Valor | Razão |
|-----------|-------|-------|
| d_model | 64 | Suficiente para codificar 30 sujeitos × 10 épocas; evita overfitting com ~4k frases de treino |
| n_heads | 4 | 4 cabeças de atenção com d_k=16 — mínimo para atenção multi-head |
| n_layers | 2 | Com seq_len=5, profundidade 2 já captura interações de ordem 2; 3+ layers overfitam |
| d_ff | 128 | 2× d_model — regra padrão de Transformer pequeno |
| d_sin | 32 | Metade de d_model: features sinusoidais menores que a MLP de saída |
| dropout | 0.1 | Padrão; corpus pequeno não tolera dropout mais alto |

O corpus tem ~4k frases de treino para 30 sujeitos × 10 épocas = 300 combinações (sujeito, época). Com d_model=64, o modelo tem ~120k parâmetros para B3 — poucos parâmetros por exemplo positivo, o que favorece generalização.

---

## 9. Pre-LN vs Post-LN

O Transformer encoder usa `norm_first=True` (Pre-LN): LayerNorm é aplicado antes de self-attention e do FFN, não depois.

**Decisão:** Pre-LN é mais estável em corpus pequeno. Com Post-LN, o treinamento de Transformers profundos requer warmup de learning rate cuidadoso; com Pre-LN, a norma na entrada de cada sublayer garante gradientes mais controlados mesmo sem warmup. Em experimentos de ablação de Transformer BERT-like, Pre-LN converge mais rápido e com menor variância em baixo volume de dados.

---

## 10. Limitações para reportar

1. **Corpus artificial:** a cadeia de ablação foi validada em dados sintéticos onde o ground truth de deriva é perfeito. O comportamento em corpus de linguagem natural com deriva real (ex: Google Books Ngram, COHA) não foi testado nesta fase.

2. **Seq_len=5 (SVO estrito):** a atenção textual em seq_len=5 é quase redundante — há pouco contexto para distribuir. Em corpus real, com seq_len=512 e sentenças ricas, o papel relativo da atenção temporal vs textual pode ser muito diferente.

3. **Protótipos como mean-pool:** a memória histórica é um único vetor por (sujeito, época), perdendo a variância intra-época. Se um sujeito aparece em contextos muito diferentes dentro de uma época (ex: deriva dentro da época), o protótipo é a média e perde a distribuição.

4. **Causalidade na avaliação:** durante avaliação em `test` e `ambiguous_test`, o modelo usa protótipos computados com o modelo totalmente treinado, não com o modelo da época correspondente. Isso é uma pequena violação de causalidade — na prática não afeta os resultados porque os protótipos são stop-gradient e a avaliação não é sequencial, mas deve ser reconhecido.

5. **B3-shuffled não controla n_params:** B3 e B3-shuffled têm o mesmo número de parâmetros, mas as cabeças de atenção em B3-shuffled aprendem a usar histórico irrelevante durante treino. Uma comparação mais rigorosa seria B3 vs um modelo com o mesmo número de parâmetros extras mas sem acesso a nenhuma memória (já coberto por B3-nohistory).
