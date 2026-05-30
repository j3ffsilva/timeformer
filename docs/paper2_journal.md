# Planejamento Paper 2 — Timeformer: Framework Completo
## Versão revisada à luz do paper 1 final (IBERAMIA)

---

## O que o paper 1 já entrega (não repetir, partir daqui)

Leitura do paper 1 final revela que vários experimentos que constavam
como "future work" no planejamento anterior já foram executados:

**Já feito — CKA por camada:**
Additive vs. Token-Time: 0.726 → 0.887 ao longo da profundidade.
Standard vs. condicionados: ~0.22 (estável).
Conclusão já estabelecida: self-attention absorbe a diferença de input.

**Já feito — Ablação de heads:**
L0H3 mostra maior variância temporal de atenção (0.0075 vs. 0.0033).
Ablação individual: efeito parcial (L0H0: Δ=−0.026; L0H1: Δ=−0.021).
Conclusão já estabelecida: efeito distribuído, não localizado.

**Já feito — Pilot noise sweep:**
3 seeds, slope=+1.07, p=0.002.
Consistente com Token-Time retendo vantagem sob marcadores degradados.
Explicitamente marcado como "replicação necessária".

**Já feito — TOST de equivalência:**
Additive vs. Token-Time: diff=−0.003, 90% CI [−0.026, +0.019], pTOST<0.001.
Equivalência confirmada a δ=0.05 e δ=0.034.

**Já feito — Separabilidade algébrica (implícita):**
O paper 1 descreve W[e(w);τ(t)] como linear. A separabilidade
W_a·e(w) + W_b·τ(t) está implícita. Não é achado novo — é consequência
da equação. O paper 2 a nomeia explicitamente como motivação para FiLM.

**Implicação para o paper 2:**
As Fases 0, 1 e parte da Fase 3 do protocolo anterior já estão feitas.
O paper 2 parte desses resultados como motivação estabelecida, não os
reproduz como novidade. O paper 2 abre com: "paper 1 mostrou X, Y, Z;
a questão que resta é se um objetivo alinhado com trajetória resolve o
que a arquitetura não resolve sozinha."

---

## Reposicionamento: o que é o paper 2

**Não é:** extensão do paper 1 com mais experimentos.

**É:** o paper fundador do framework Timeformer — que nomeia,
formaliza e demonstra a solução completa. O paper 1 do IBERAMIA
é resultado preliminar absorvido; o paper 2 na revista é o registro
canônico do framework.

**Nome:** Timeformer estreia aqui, não no paper 1. O paper 2 define
Timeformer como framework com três pilares:
  1. Condicionamento temporal na arquitetura (paper 1, condensado)
  2. Objetivo alinhado com geometria de trajetória (contribuição nova)
  3. Suíte de diagnósticos controlada + avaliação em corpus natural

**Citação do paper 1 no paper 2:**
"Resultados diagnósticos preliminares foram reportados em [IBERAMIA];
este paper apresenta o framework Timeformer completo, incluindo
fundamentos formais, função objetivo de trajetória, e validação
em corpus natural."

---

## Contribuição central: a função objetivo

### O que o paper 1 estabelece como motivação

Três achados do paper 1 motivam diretamente o paper 2:

1. Qualquer condicionamento temporal mais que dobra o deslocamento de
   vizinhança vs. baseline (Δ≈−0.56 vs. −0.221). O sinal temporal
   ajuda — mas o teto é limitado.

2. Global shift e projeção conjunta são equivalentes em regime de
   marcadores confiáveis (TOST pTOST<0.001). O encoder absorve
   a diferença de input (CKA 0.726→0.887). O gargalo não é a
   arquitetura — é o objetivo.

3. Pilot noise sweep (slope=+1.07, p=0.002) sugere que Token-Time
   retém vantagem sob degradação, mas precisa de replicação.
   O paper 2 é essa replicação — e a extensão para corpus natural.

### Design da função objetivo: L_traj

**Escolha principal: Design C — loss auxiliar conjunta.**

Motivação: é o teste mais direto da tese. Mantém o MLM (preserva
competência de linguagem), adiciona sinal de trajetória, e a ablação
λ=0 vs. λ>0 é o experimento limpo que o paper 1 não pôde fazer.

```
L_total = L_MLM + λ · L_traj

L_traj = Σ_{s} Σ_{t} Σ_{t'≠t} max(0, sim(h_s(t), proto(t')) 
                                    − sim(h_s(t), proto(t)) + m)
```

onde proto(t) é o centroide das representações no período t,
m é a margem (hiperparâmetro), e a soma é sobre pares de períodos
para sujeitos com trajetória conhecida.

**Derivação formal (obrigatória para paper fundador de linha):**

Ponto de partida: definição de traceabilidade do paper 1 (D2).
Um sujeito s é traceable em t se a proporção de vizinhos N1 de h_s(t)
acompanha P(N1|s,t). Isso implica:

  sim(h_s(t), proto_N1) > sim(h_s(t), proto_N2)  quando P(N1|s,t) > 0.5
  sim(h_s(t), proto_N2) > sim(h_s(t), proto_N1)  quando P(N1|s,t) < 0.5

Generalizando para trajetória: para qualquer par (t, t') com
P(N1|s,t) > P(N1|s,t'), a representação em t deve estar mais próxima
do centroide do período t do que do período t'. Isso é exatamente
a condição que L_traj maximiza via margin ranking temporal.

Formalmente: L_traj é condição suficiente para traceabilidade no
sentido de D2, e necessária para que a representação seja monotônica
ao longo da trajetória plantada. A derivação vai na Seção 4 do paper 2.

**Fonte de proto(t):**

Opção A (sintético): centroides dos sujeitos por período, computados
a partir de um modelo congelado do passo 1. Teacher-student: o modelo
de passo 1 é o teacher fixo; o modelo de passo 2 é treinado para que
suas representações satisfaçam L_traj em relação aos centroides do teacher.
Evita circularidade porque o teacher não é atualizado.

Opção B (corpus natural): embeddings diacrônicos externos por fatia
temporal (Hamilton et al., word2vec por década em COHA). Não depende
de bootstrap; conecta diretamente com literatura LSCD. Risco: estamos
destilando um modelo mais simples num mais complexo — defensável como
"injetar conhecimento de estrutura temporal", mas o revisor vai perguntar.

Recomendação: usar Opção B no corpus natural (mais limpo, mais conectado
com related work), Opção A no sintético (ground truth disponível permite
proto(t) exato). Isso vira uma ablação: proto(t) exato vs. word2vec vs.
bootstrapped — a qual fonte de trajetória o modelo é sensível?

### Variante arquitetural: FiLM

A separabilidade algébrica do Token-Time (W[e(w);τ(t)] = W_a·e(w) + W_b·τ(t))
é nomeada explicitamente no paper 2 como motivação para uma variante
genuinamente conjunta. Isso transforma um potencial ponto fraco do
paper 1 numa motivação para contribuição nova.

FiLM (Feature-wise Linear Modulation):
```
z_i = γ(τ(t)) ⊙ e(w_i) + β(τ(t)) + p_i
```
onde γ, β são projeções aprendidas de τ(t) para R^d.

Propriedades:
- Cria interação multiplicativa genuína entre token e período
- Ablação γ=1, β=τ(t) recupera exatamente o Additive — família paramétrica
- Interpretável: γ modula quais dimensões de e(w) importam neste período;
  β é o deslocamento de base
- Literatura de suporte: FiLM (Perez et al., 2018) em condicionamento visual

O design FiLM + L_traj é a combinação que o paper 2 apresenta como
Timeformer completo. As ablações isolam contribuições separadas:

| Arquitetura | Objetivo    | Papel                              |
|-------------|-------------|------------------------------------|
| Standard    | MLM         | Baseline paper 1 (reproduzido)     |
| Token-Time  | MLM         | Resultado paper 1 (reproduzido)    |
| FiLM        | MLM         | Ablação: arquitetura sem objetivo  |
| Token-Time  | MLM+L_traj  | Ablação: objetivo sem arquitetura  |
| FiLM        | MLM+L_traj  | Timeformer completo                |
| Standard    | MLM+L_traj  | Ablação: objetivo compensa arq.?   |

A última linha é a mais interessante teoricamente: se Standard + L_traj
chega perto de FiLM + MLM, o objetivo é mais importante que a arquitetura —
fortalece a tese do paper 1. Se não chega, os dois são complementares.

---

## Corpus e dados

### Sintético como controle

Mesmo setup do paper 1: SVO, 30 sujeitos, 3 classes, 10 períodos.
Garante comparabilidade direta.

Adicionar a varredura de fidelidade completa (25%→50%, 7 níveis,
31 seeds por nível) — o pilot do paper 1 com 3 seeds vira experimento
completo. É a replicação que o paper 1 sinaliza como necessária.
Predição registrada: gap FiLM+L_traj vs. Token-Time+MLM cresce
monotonicamente com a degradação dos marcadores.

### Corpus natural: COHA como principal

**Por que COHA:**
- Granularidade decadal (1810–2000): 19 fatias temporais, comparação
  direta entre décadas
- Balanceado por gênero (ficção, não-ficção, revistas, jornais)
- Padrão para diacronia em inglês; bem documentado
- Permite construir proto(t) via word2vec por década (Hamilton et al.
  usaram exatamente este setup — conexão direta com related work)

**Seleção de palavras-alvo:**
Usar as palavras-alvo do SemEval-2020 Task 1 (inglês) como subconjunto
para avaliação extrínseca. Isso conecta os dois corpora: treina em COHA,
avalia em SemEval-2020. A predição é que Timeformer completo melhora
o ranking de mudança semântica sobre o baseline MLM.

**Seleção de âncoras para L_traj (se usar Opção A no natural):**
Selecionar top-50 palavras mais estáveis no COHA por variação de
vizinhança entre décadas. Ablação obrigatória: âncoras estáveis vs.
aleatórias vs. instáveis.

### SemEval-2020 como avaliação extrínseca

Tarefa de ranking: dado conjunto de palavras-alvo, rankear por
magnitude de mudança semântica entre dois períodos. Correlação de
Spearman com ground truth anotado (padrão da tarefa).

Comparação com Hamilton et al. como baseline natural — eles usam
word2vec+alinhamento, que é exatamente a Opção B de proto(t). Isso
fecha um ciclo: Timeformer aprende a reproduzir estrutura que o
word2vec diacrônico captura, mas de forma diretamente consultável
(token@time como objeto inspecionável, contribuição do paper 1).

---

## Protocolo de avaliação

### Diagnósticos do paper 1 (mantidos, comparabilidade direta)

D1–D4 todos mantidos. Expectativa: FiLM+L_traj melhora D2 e D4
sobre Token-Time+MLM; D3 (contrastive sign-flip) é o mais sensível
a L_traj, porque testa diretamente se mudar o período muda a
representação.

### Diagnósticos novos (dos protocolos discutidos)

**Spearman de trajetória (corrigido):**
Correlação entre similaridade ao centroide (contínua, sem discretização
kNN) e P(N1|s,t) plantado, por sujeito. Reportar por classe e por
modelo. Saturação esperada para Drift monotônico — documentar
explicitamente.

**Contraste de classe Drift−Stable:**
Operacionaliza o "global displacement" observado na Tabela 1 do paper 1.
Token-Time: Δ(Drift)=−0.563, Δ(Stable)=−0.180, contraste=0.383.
Predição para FiLM+L_traj: contraste maior (L_traj penaliza movimento
de sujeitos Stable, recompensa movimento correto de Drift).

**Bimodalidade Bifurcating:**
Silhouette entre ocorrências N1-contexto e N2-contexto nos períodos
tardios. Mede se h_s preserva dois modos ou colapsa em centroide.
Conecta diretamente com o failure mode de mean-prototype do paper 1.

**Varredura de fidelidade completa:**
Reportar gap FiLM+L_traj vs. Token-Time+MLM nos diagnósticos D2,
Spearman e contraste, ao longo de 7 níveis de fidelidade (25%→50%).
Testar monotonicidade (regressão + Jonckheere-Terpstra).

### Análise de mecanismo (parte do paper 2, não future work)

**CKA por camada para FiLM+L_traj:**
Predição: colapso CKA de FiLM+L_traj vs. Standard é menor do que
Token-Time+MLM vs. Standard ao longo das camadas — L_traj força o
encoder a preservar diferença de condicionamento que sem a loss
seria neutralizada.

**Comparação de variância temporal de atenção:**
O paper 1 já reporta L0H3: 0.0075 (Token-Time) vs. 0.0033 (Additive).
Adicionar FiLM+L_traj: predição é que a variância temporal de atenção
é ainda maior, especialmente nas heads que o paper 1 identificou.

---

## Estrutura do paper 2

```
1. Introduction
   - Problema: token@time como objeto inspecionável
   - O que paper 1 estabeleceu (diagnóstico)
   - A lacuna que paper 2 fecha: objetivo + arquitetura genuinamente conjunta
   - Contribuições: Timeformer como framework completo

2. Background
   - LSCD diacrônico (Hamilton et al.) — baseline natural e fonte de proto(t)
   - Temporal LMs (Dhingra, TimeLMs, Rosin & Radinsky) — sem teste geométrico
   - Condicionamento contextual (FiLM, Perez et al. 2018) — base para variante
   - Objetivos relacionais (data2vec, contrastive learning) — parentesco de L_traj
   - [IBERAMIA] como trabalho anterior direto

3. Temporal Traceability: Diagnóstico e Motivação (condensado do paper 1)
   - Definição formal de traceabilidade (D2 como operacionalização)
   - Resultado central: arquitetura condiciona, mas objetivo limita
   - CKA e noise sweep como motivação para paper 2 (2 parágrafos, não seção)

4. O Framework Timeformer
   4.1 Arquitetura: FiLM e separabilidade do Token-Time
   4.2 Objetivo: L_traj — derivação formal a partir de traceabilidade
   4.3 Fontes de proto(t): sintético (exato), natural (word2vec/teacher)
   4.4 Família de ablações (tabela 2×3)

5. Controlled Diagnostic Suite
   - Setup sintético (do paper 1, condensado)
   - Varredura de fidelidade completa (31 seeds × 7 níveis)
   - Tabela principal: todos os modelos × todos os diagnósticos
   - Análise de mecanismo: CKA e variância de atenção

6. Corpus Natural: COHA
   - Setup, períodos, palavras-alvo, âncoras
   - Diagnósticos D2, Spearman, contraste de classe, bimodalidade
   - Ablação de fonte de proto(t): exato vs. word2vec vs. bootstrapped

7. Avaliação Extrínseca: SemEval-2020
   - Ranking de mudança semântica
   - Comparação com Hamilton et al. como baseline

8. Análise e Discussão
   - Quando o objetivo é suficiente vs. quando a arquitetura importa
   - Escolha de âncoras: guidelines práticos
   - Re-treino vs. fine-tuning: resultados da ablação
   - Limitações: esparsidade temporal, custo de proto(t), vocabulário controlado

9. Timeformer como Framework: Agenda de Pesquisa
   - Design B (mascarar período) como próximo passo natural
   - Extensão multilíngue
   - Aplicações downstream: QA temporal, análise histórica
   - Conexão com interpretabilidade temporal

10. Conclusion
```

### Contribuições (versão para submissão)

1. Introduzimos Timeformer, um framework para representação semântica
   temporal com três pilares: condicionamento arquitetural, objetivo
   alinhado com trajetória, e suíte de diagnósticos controlada.

2. Propomos L_traj, uma loss de margem temporal derivada formalmente
   da definição de traceabilidade, que opera como termo auxiliar ao MLM
   e é comprovadamente suficiente para traceabilidade no sentido de D2.

3. Introduzimos FiLM-Timeformer, uma variante com condicionamento
   genuinamente conjunto que supera a separabilidade algébrica do
   Token-Time original.

4. Demonstramos melhora empírica em corpus natural (COHA) e benchmark
   extrínseco (SemEval-2020), com análise de mecanismo mostrando que
   L_traj reduz o colapso CKA que o MLM sozinho não evita.

5. Mostramos que na varredura de fidelidade completa (31 seeds × 7 níveis)
   o gap FiLM+L_traj vs. Token-Time+MLM cresce monotonicamente com a
   degradação dos marcadores — confirmando e estendendo o pilot do paper 1.

---

## Cronograma

### Semana 1–2: Fundamentos (desbloqueadores)

**Verificação algébrica FiLM vs. Token-Time:**
Nomear a separabilidade formalmente. Escrever a derivação de L_traj.
Implementar FiLM. Custo: alguns dias.

**Varredura de fidelidade completa no sintético:**
Reutiliza o código do paper 1. 31 seeds × 7 níveis × 5 modelos.
É a replicação do pilot que o paper 1 pede explicitamente.
Resultado esperado em 1 semana de compute.

**Ponto de decisão:** se FiLM+L_traj não melhora D2 no sintético
em relação a Token-Time+MLM, rever design de L_traj antes de avançar.
O sintético é o caso mais fácil (ground truth perfeito). Falha aqui
= problema no design, não no corpus.

### Semana 3–6: COHA

Preprocessar corpus por década. Treinar word2vec por fatia (ou usar
embeddings Hamilton et al. disponíveis publicamente — economiza semanas).
Selecionar âncoras. Rodar ablação de fonte de proto(t).
Rodar diagnósticos principais.

### Semana 7–8: SemEval-2020 e análise de mecanismo

Avaliação extrínseca de ranking. CKA por camada para todos os modelos.
Variância temporal de atenção.

### Semana 9–12: Escrita

Escrita paralela ao longo das semanas anteriores.
Rascunho completo na semana 9.
Revisão e polimento nas semanas 10–12.

### Marcos de decisão

**Fim da semana 2:**
FiLM+L_traj > Token-Time+MLM no sintético em pelo menos D2 e Spearman?
Se sim: avança para COHA.
Se não: rever L_traj (margem? fonte de proto? λ?).

**Fim da semana 6:**
Ganho em COHA real e consistente entre diagnósticos?
Se ganho < 5% em Spearman: avaliar se narrativa sustenta paper de revista
ou se é mais honesto como paper de conferência com contribuição mais modesta.

**Antes da submissão:**
Confirmar que os diagnósticos de avaliação são independentes de proto(t)
— não estamos avaliando com a mesma estrutura que usamos para treinar.
No sintético: proto(t) vem do teacher congelado; diagnóstico usa P(N1|s,t)
plantado diretamente. São independentes por construção.
No COHA: proto(t) vem de word2vec; diagnóstico usa vizinhança do modelo
nos embeddings finais. São independentes se word2vec e diagnóstico não
forem calculados sobre o mesmo conjunto de sentenças.

---

## Pré-registro (antes de rodar Semanas 3–8)

Após confirmação no sintético (fim da semana 2), registrar:

1. FiLM+L_traj > Token-Time+MLM em D2 no COHA
2. Gap cresce monotonicamente com degradação (varredura completa)
3. CKA collapse é menor em FiLM+L_traj vs. Token-Time+MLM

Registro em commit datado ou OSF antes de rodar o corpus natural.