# Diagnósticos do Timeformer — Guia passo a passo

## Antes de começar: o que está sendo avaliado

O paper investiga uma propriedade chamada **traceabilidade temporal**: a ideia de que a representação matemática de uma palavra deveria mudar conforme o período histórico em que ela é usada. A palavra *vírus* em 1900 coexistia com palavras como *febre* e *contágio*; em 2020 passou a coexistir com *computador* e *software*. Um modelo com traceabilidade temporal deveria representar as duas de forma geometricamente distinta — em regiões diferentes do espaço de embeddings.

O problema central: o treinamento padrão de transformers (MLM — Masked Language Modeling) recompensa o modelo por prever tokens mascarados a partir dos vizinhos na mesma frase. Ele não tem nenhum incentivo direto para separar geometricamente o mesmo token em períodos diferentes.

O paper testa isso num ambiente controlado onde a resposta correta é conhecida de antemão.

---

## O ambiente controlado: o que existe nesse mundo artificial

### Vocabulário

O mundo do experimento tem três tipos de palavras:

- **30 sujeitos** (S1 a S30): as palavras cujas "trajetórias semânticas" vamos monitorar
- **8 verbos** (V1 a V8): divididos em dois grupos
- **8 objetos** (O1 a O8): divididos em dois grupos

### As duas vizinhanças semânticas: N1 e N2

Os verbos e objetos estão divididos em duas **vizinhanças semânticas**, como dois "campos de assunto" diferentes:

- **N1** (vizinhança 1): verbos V1, V2, V3, V4 e objetos O1, O2, O3, O4  
  → Imagine: linguagem de medicina (examinou, tratou, diagnosticou, curou)
- **N2** (vizinhança 2): verbos V5, V6, V7, V8 e objetos O5, O6, O7, O8  
  → Imagine: linguagem de tecnologia (processou, executou, transmitiu, armazenou)

### As frases (sentenças SVO)

Toda frase nesse mundo tem exatamente três palavras: **Sujeito – Verbo – Objeto**.

```
Exemplos de frases N1: S5 V2 O3 | S5 V4 O1 | S12 V1 O4
Exemplos de frases N2: S5 V6 O8 | S5 V7 O5 | S12 V8 O6
```

Note que o sujeito (S5) é o mesmo nos dois casos. O que muda são os verbos e objetos ao redor.

### Os 10 períodos

O tempo é dividido em **10 fatias temporais**: t0, t1, t2, ..., t9.  
Pense nelas como décadas: t0 = 1900, t1 = 1910, ..., t9 = 1990.

### A trajetória plantada: P(N1 | s, t)

Para cada sujeito `s` em cada período `t`, existe uma probabilidade plantada de ele aparecer em frases N1. Chamamos isso de **P(N1 | s, t)** — lê-se "a probabilidade de N1 dado o sujeito s no período t".

- P(N1 | S5, t0) = 0.95 → S5 aparece 95% das vezes com vocabulário N1 em t0
- P(N1 | S5, t5) = 0.50 → 50/50 entre N1 e N2 em t5
- P(N1 | S5, t9) = 0.05 → S5 aparece 95% das vezes com vocabulário N2 em t9

Isso é a **trajetória plantada** de S5: o valor de verdade que o avaliador conhece e que o modelo precisa capturar geometricamente.

### As três classes de trajetória

```
Stable:      P(N1|s,t) ≈ 0.8 constante de t0 a t9
             → a palavra não muda de "campo semântico"

Drift:       P(N1|s,t) cai de ≈1.0 em t0 para ≈0.0 em t9
             → a palavra migra completamente de N1 para N2

Bifurcating: P(N1|s,t) cai de ≈1.0 em t0 para ≈0.5 em t9
             → a palavra desenvolve dois sentidos coexistentes
```

O experimento tem **10 sujeitos por classe**: 10 Stable, 10 Drift, 10 Bifurcating.

### O ruído de 25%

Cada marcador local (o verbo e o objeto da frase) é sorteado da vizinhança **oposta** 25% do tempo. Isso é intencional: força o modelo a usar a identidade do sujeito + o período, não só os vizinhos locais.

```
Frase "ruidosa" de S5 em t1 (período em que S5 deveria estar em N1):
  S5 V6 O3
      ↑  ↑
      |  O3 ∈ N1 (coerente)
      V6 ∈ N2 (ruído — veio da vizinhança errada)
```

### O que o modelo produz

Cada palavra numa frase é processada pelo encoder e gera um **vetor de embedding** (uma lista de números) na última camada oculta. O vetor do sujeito na posição do sujeito é chamado de **h_s** — este é o objeto que todos os diagnósticos analisam.

```
Frase:   [CLS]  S5   V2   O3  [SEP]
          ↓     ↓    ↓    ↓     ↓
Vetores: h_cls  h_s  h_v  h_o  h_sep
                ↑
          Este é o que avaliamos
```

### Os quatro modelos comparados

- **Standard**: sem nenhum sinal de período. O modelo não sabe em que tempo está.
- **Additive**: o período é injetado somando o mesmo vetor τ(t) a todos os tokens.
- **Token-Time**: o período é concatenado com o embedding do token antes da atenção, criando uma projeção específica por token.
- **Memory-Augmented**: extensão do Token-Time com memória histórica dos protótipos passados.

---

## Diagnóstico D1 — Linear Probe

### O que quer saber

"A representação h_s do sujeito carrega informação sobre em qual vizinhança ele deveria estar nesse período?"

### Nomenclatura

- **Probe linear**: um classificador simples (regressão logística ou uma camada linear) treinado *por cima* de h_s com o encoder **congelado**. O encoder não é retreinado; só o probe aprende.
- **Split padrão (75% fidelidade)**: as frases de teste têm 75% dos marcadores coerentes. Verbos e objetos ajudam a identificar a vizinhança.
- **Split ambíguo (50% fidelidade)**: os marcadores locais são puro ruído — 50% N1, 50% N2. Só a identidade do sujeito e o período restam como evidência.
- **Split de continuação**: frases dos períodos t8 e t9, que o modelo **nunca viu** durante o treino MLM (treinou só em t0–t7).

### Passo a passo com exemplo

**Configuração:**  
Sujeito S5 (Drift), período t4.  
Ground truth plantado: P(N1|S5,t4) ≈ 0.5 (zona de transição).  
Label correto para o probe: N2 (porque a trajetória já passou do ponto médio nesse exemplo).

**Passo 1 — Extrair h_s.**  
Apresenta a frase `S5 V3 O2` ao encoder congelado. O encoder processa e devolve h_s: um vetor de 64 números (d=64 nesse experimento).

```
h_s = [0.23, -0.41, 0.87, 0.12, ..., -0.05]  ← 64 dimensões
```

**Passo 2 — Passar pelo probe linear.**  
O probe é uma matriz W ∈ R^{2×64} mais um bias. Multiplica W · h_s + b e aplica softmax.

```
logits = W · h_s + b = [1.2, 2.8]  ← score para [N1, N2]
após softmax: [0.18, 0.82]
predição: N2 (maior probabilidade) ✓ correto
```

**Passo 3 — Avaliar no split ambíguo.**  
Agora a frase é `S5 V6 O2` — V6 ∈ N2 e O2 ∈ N1, marcadores se cancelam.  
O probe agora tem que confiar em h_s ter capturado que S5 em t4 deveria estar em N2.

```
Token-Time: acerta 59.5% dos casos no split ambíguo
Additive:   acerta 57.9% dos casos no split ambíguo
```

Esses 1.6 pontos com CIs não sobrepostos são a **única vantagem reprodutível** do Token-Time sobre o Additive. Aparece exatamente quando os marcadores locais somem.

**O que D1 não mede (e por quê isso importa):**  
D1 mede se o label correto é *decodável* de h_s. Mas um modelo pode codificar o período como uma dimensão separada e ortogonal à semântica — "sei que é t4 mas S5 ainda está na vizinhança errada geometricamente". Probe alto com geometria errada é o cenário que D2 captura.

---

## Diagnóstico D2 — Context Drift Score

### O que quer saber

"Os vizinhos mais próximos de h_s no espaço geométrico são os tokens corretos para esse período?"

### Nomenclatura

- **Vizinhos mais próximos (k-NN)**: os k vetores mais similares a h_s no espaço de embeddings, medidos por **similaridade cosseno** (o ângulo entre vetores — 1.0 = idênticos, 0.0 = perpendiculares, -1.0 = opostos).
- **k=10**: os 10 vizinhos mais próximos são consultados.
- **Proporção N1**: quantos desses 10 vizinhos pertencem à vizinhança N1.
- **∆ (drift score)**: proporção em t9 menos proporção em t0. Negativo = movimento em direção a N2.

### Passo a passo com exemplo

**Configuração:**  
S5 (Drift). Esperamos que a proporção N1 caia de ~0.9 em t0 para ~0.1 em t9.

**Passo 1 — Extrair h_s em t0.**  
Apresenta várias frases com S5 em t0. Tira a média de h_s (ou usa cada ocorrência separadamente). O resultado é um vetor representando S5@t0.

**Passo 2 — Encontrar os 10 vizinhos mais próximos.**  
Calcula a similaridade cosseno de S5@t0 com os vetores de todos os verbos e objetos do vocabulário. Ordena por similaridade decrescente.

```
S5@t0 — 10 vizinhos mais próximos:
  1. V2 (sim=0.91) → N1
  2. O3 (sim=0.89) → N1
  3. V1 (sim=0.87) → N1
  4. O1 (sim=0.85) → N1
  5. V4 (sim=0.82) → N1
  6. O4 (sim=0.80) → N1
  7. V3 (sim=0.78) → N1
  8. O2 (sim=0.75) → N1
  9. V6 (sim=0.61) → N2  ← ocasional N2 por ruído
  10. O7 (sim=0.58) → N2

Proporção N1 em t0: 8/10 = 0.80
```

**Passo 3 — Repetir para t9.**

```
S5@t9 — 10 vizinhos mais próximos:
  1. V7 (sim=0.90) → N2
  2. O8 (sim=0.88) → N2
  3. V6 (sim=0.86) → N2
  4. O6 (sim=0.84) → N2
  5. V8 (sim=0.81) → N2
  6. O5 (sim=0.79) → N2
  7. V5 (sim=0.77) → N2
  8. O7 (sim=0.74) → N2
  9. V2 (sim=0.60) → N1  ← ocasional N1 por ruído
  10. O3 (sim=0.57) → N1

Proporção N1 em t9: 2/10 = 0.20
```

**Passo 4 — Calcular ∆.**

```
∆ = proporção_t9 − proporção_t0 = 0.20 − 0.80 = −0.60
```

**Comparação entre modelos:**

```
Standard:   ∆ = −0.22  (pouco movimento: sem sinal temporal, S5@t0 ≈ S5@t9)
Additive:   ∆ = −0.57  (movimento substancial)
Token-Time: ∆ = −0.56  (movimento substancial, indistinguível do Additive aqui)
```

**Controle com sujeito Stable (S12):**

```
S12@t0: proporção N1 = 0.84
S12@t9: proporção N1 = 0.67
∆ = −0.17  ← bem menor que o Drift, confirma que o modelo distingue as classes
```

**Por que D2 é fraco para separar Additive de Token-Time:**  
D2 colapsa 10 períodos em dois números (t0 e t9). Uma trajetória que cai linearmente de 0.80 para 0.20 e uma que fica em 0.80 até t7 e despenca até 0.20 em t8–t9 produzem exatamente o mesmo ∆. Isso é por que o protocolo propõe complementar com Spearman sobre os 10 pontos.

---

## Diagnóstico D3 — Contrastive Sign-Flip Rate

### O que quer saber

"Se eu mudar *só* o rótulo de período de uma frase, a representação muda de vizinhança dominante?"

Este é o diagnóstico mais cirúrgico: isola o efeito causal do período, mantendo a sentença idêntica.

### Nomenclatura

- **Par contrastivo**: a mesma frase apresentada duas vezes, com rótulos de período diferentes.
- **Sign-flip**: ocorre quando a vizinhança dominante (N1 vs N2) inverte entre os dois apresentações.
- **Sign-flip rate**: proporção de pares que flipam sobre todos os pares testados.

### Passo a passo com exemplo

**Configuração:**  
Frase: `S5 V2 O3` (marcadores N1, sujeito driftante).  
Dois períodos: t2 (S5 ainda em N1) e t8 (S5 deveria estar em N2).

**Passo 1 — Apresentar a frase com rótulo t2.**  
O modelo recebe `S5 V2 O3` e o período t2.

```
h_s(S5, "V2 O3", t2) — 10 vizinhos:
  N1: 8 vizinhos | N2: 2 vizinhos
  → Vizinhança dominante: N1
```

**Passo 2 — Apresentar a mesma frase com rótulo t8.**  
Mesma sentença, só o período muda.

```
h_s(S5, "V2 O3", t8) — 10 vizinhos:
  N1: 3 vizinhos | N2: 7 vizinhos
  → Vizinhança dominante: N2
```

**Passo 3 — Checar o flip.**

```
t2 → N1-dominante
t8 → N2-dominante
→ Flip: SIM ✓
```

**O mesmo par no Standard:**

```
Standard ignora o período.
"S5 V2 O3" @ t2: 7 N1  → N1-dominante
"S5 V2 O3" @ t8: 7 N1  → N1-dominante  (mesma resposta)
→ Flip: NÃO
```

**Passo 4 — Calcular a taxa sobre todos os pares.**

```
Standard:   0 flips em todos os pares → sign-flip rate = 0.000
Additive:   flipa em 60.2% dos pares  → sign-flip rate = 0.602
Token-Time: flipa em 62.4% dos pares  → sign-flip rate = 0.624
Mem-Aug:    flipa em 58.7% dos pares  → sign-flip rate = 0.587
```

**Interpretação:**  
Os CIs de Additive e Token-Time se sobrepõem: [0.549, 0.655] vs [0.576, 0.673]. D3 confirma que o período *influencia* a representação nos modelos condicionados, mas não consegue separar Additive de Token-Time. O mesmo problema de D2: o sinal temporal está sendo usado nos dois, mas de formas que produzem saídas quase idênticas nesse regime.

---

## Diagnóstico D4 — Continuation Diagnostic

### O que quer saber

"O modelo generalizou a direção da mudança semântica, ou apenas memorizou padrões de cada período?"

### Nomenclatura

- **Split de continuação**: frases dos períodos t8 e t9, **completamente excluídos do treino MLM**. O treino usou apenas t0–t7.
- **Generalização temporal**: capacidade de aplicar o padrão aprendido (S5 vai de N1 para N2) a períodos nunca vistos.
- **Acurácia de continuação**: acurácia do probe linear (treinado em t0–t7) aplicado às representações de t8–t9.

### Passo a passo com exemplo

**Configuração:**  
S5 (Drift). O modelo foi treinado apenas em frases de t0–t7.  
Avaliação: frases de t8 e t9.  
Ground truth: P(N1|S5,t8) ≈ 0.05, P(N1|S5,t9) ≈ 0.02 — S5 deveria estar quase completamente em N2.

**Passo 1 — Extrair h_s para S5 em t8.**  
Apresenta `S5 V7 O8` com rótulo t8. O modelo nunca processou o rótulo t8 durante o treino.

```
h_s(S5, t8) = [0.11, -0.73, 0.92, ...]  ← 64 dimensões
```

**Passo 2 — Aplicar o probe linear (treinado em t0–t7).**

```
logits = W · h_s + b = [0.8, 3.1]
após softmax: [0.08, 0.92]
predição: N2 ✓ correto
```

**Passo 3 — Interpretar o que permite isso.**

Um modelo que apenas memorizou "t4 → tal vizinhança, t6 → tal vizinhança" não consegue extrapolar para t8. Para acertar em t8, o modelo precisa ter aprendido a *direção* da mudança de S5, não só os valores pontuais.

**Resultados comparados:**

```
Standard:   0.730  (melhor que chance — consegue usar os marcadores locais
                    mesmo em t8, mas sem sinal temporal explícito)
Token-Time: 0.763  (gap de +3.3pp — generaliza um pouco melhor)
Mem-Aug:    0.755  (pior que Token-Time — memória causal não ajuda aqui)
```

**Por que o gap é pequeno:**  
A predição do protocolo (Fase 4A) é que esse gap cresce conforme o ruído aumenta. Em 75% de fidelidade, os marcadores locais bastam para boa parte das predições, mesmo em t8–t9. Quando os marcadores somem (split ambíguo, 50% fidelidade), o período se torna a única pista — e aí Token-Time deveria se destacar mais.

---

## Como os quatro diagnósticos se complementam

```
              Mede recoverability?  Mede geometria?  Separou A vs TT?
D1 probe      Sim                   Não              Sim (split ambíguo)
D2 drift      Parcialmente          Sim              Não (∆ similar)
D3 flip       Não                   Sim              Não (CIs sobrepostos)
D4 continuation Sim                 Não              Não (gap pequeno)
```

A tese central do paper vive na coluna do meio: **recoverability não é geometria**. Um modelo pode acertar D1 e D4 enquanto falha em D2 — e isso é precisamente o que mostra que o MLM não otimiza traceabilidade geométrica.

A única separação reprodutível entre Additive e Token-Time está em D1 no split ambíguo. Isso importa porque é o regime em que o modelo é *forçado* a usar a combinação sujeito+período sem bengala dos marcadores locais — que é o regime mais próximo do que você encontra em corpus real com vocabulário rico e polissemia genuína.