# Paper 1A: KDMiLe — Benchmark Sintetico para Rastreabilidade de Deriva Semantica

**Status:** plano de artigo  
**Venue alvo:** KDMiLe 2026  
**Idioma:** portugues  
**Recorte:** metodologia, corpus, baselines e metricas  
**Relacao com a linha Timeformer:** primeiro artigo derivado do Paper 1; foca no banco de prova, nao na arquitetura completa.

---

## 1. Tese do artigo

Este artigo deve defender que avaliar deriva semantica temporal exige um banco
de prova controlado em que o fenomeno de mudanca seja conhecido, mensuravel e
separavel de atalhos triviais de contexto.

A tese central:

> Propomos um corpus sintetico parametrico para estudar rastreabilidade de deriva
> semantica temporal, com ground truth por sentenca, trajetorias conhecidas e
> splits de avaliacao que separam contexto local informativo, enganoso e neutro.

O artigo nao deve vender o Timeformer como contribuicao principal. A contribuicao
principal e o **instrumento experimental**.

---

## 2. Pergunta cientifica

Como construir um benchmark artificial que permita testar se uma representacao
vetorial captura:

1. estabilidade semantica;
2. deriva gradual;
3. bifurcacao para estado misto;
4. dependencia temporal quando o contexto local nao resolve a tarefa?

---

## 3. Contribuicoes

1. **Corpus sintetico parametrico** com 30 sujeitos, 10 epocas e trajetorias
   controladas de estabilidade, deriva e bifurcacao.
2. **Ground truth por sentenca** (`true_context`) que permite avaliar
   representacoes sem anotacao humana.
3. **Marcadores probabilisticos** (`p_canon=0.75`) para evitar oraculos
   deterministas baseados em verbo/objeto.
4. **Splits semanticos controlados**:
   - `test`: contexto local parcialmente informativo;
   - `hard_verb`: verbo enganoso, objeto correto;
   - `hard_both`: verbo e objeto enganosos;
   - `ambiguous_test`: marcadores locais neutros, p=0.50.
5. **Baselines iniciais** para mostrar que o sinal temporal e recuperavel e que
   modelos sem tempo colapsam distribuicoes dependentes de epoca.
6. **Analise metodologica** das armadilhas encontradas em versoes anteriores:
   probes triviais, oraculos verbais, Procrustes instavel e snapshots que escondem
   trajetorias.

---

## 4. O que fica fora deste artigo

Para manter o recorte etico e claro, este paper nao deve tentar cobrir tudo.

Ficam fora:

- a arquitetura Timeformer completa;
- memoria temporal B3 como resultado principal;
- Semantic Canonizer / memoria multi-prototipo;
- aplicacao em corpus real da Folha;
- claims de superioridade sobre Transformers.

Esses pontos ficam para o paper internacional.

---

## 5. Narrativa sugerida

### 5.1 Problema

Palavras mudam de vizinhanca semantica ao longo do tempo. Modelos sem dimensao
temporal podem misturar estados historicos diferentes de um mesmo token. Antes de
propor arquiteturas mais complexas, e necessario um ambiente controlado em que a
mudanca seja conhecida.

### 5.2 Solucao metodologica

Apresentar um corpus SVO artificial em que a semantica de cada sujeito e definida
pela distribuicao de coocorrencia com dois contextos latentes, A e B. A deriva
semantica e simulada por mudancas parametrizadas em `P(ctx=A | sujeito, epoca)`.

### 5.3 Avaliacao

Mostrar que:

- modelos sem acesso a epoca nao recuperam trajetorias temporais;
- modelos com informacao temporal recuperam a distribuicao marginal plantada;
- `ambiguous_test` separa modelos que usam tempo de modelos que dependem apenas
  de contexto local;
- `hard_verb` e `hard_both` expõem robustez contra evidencia local enganosa.

---

## 6. Estrutura proposta

1. **Introducao**
   - deriva semantica como problema de rastreabilidade;
   - lacuna: falta de ground truth controlado para avaliar trajetorias;
   - resumo das contribuicoes.

2. **Trabalhos Relacionados**
   - embeddings diacronicos;
   - lexical semantic change detection;
   - benchmarks sinteticos;
   - modelos temporais de linguagem.

3. **Construcao do Corpus**
   - vocabulario;
   - gramatica SVO;
   - contextos A/B;
   - trajetorias parametricas;
   - marcadores probabilisticos;
   - splits.

4. **Metricas e Baselines**
   - correlacao `P_real` vs `P_pred`;
   - acuracia/F1/AUROC em `true_context`;
   - avaliacao em `ambiguous_test`;
   - contrastive set, se couber.

5. **Resultados**
   - estatisticas do corpus;
   - recuperabilidade do sinal temporal;
   - desempenho por split;
   - analise dos casos dificeis.

6. **Discussao**
   - por que probes anteriores eram triviais;
   - por que contexto deterministico era problema;
   - limites do corpus binario;
   - como o benchmark prepara a avaliacao do Timeformer.

7. **Conclusao**
   - corpus como ferramenta para estudar rastreabilidade semantica temporal;
   - proximos passos: arquitetura temporal e memoria multi-prototipo.

---

## 7. Tabelas e figuras prioritarias

1. Tabela de trajetorias exemplo:
   - estavel;
   - deriva linear;
   - deriva sigmoide;
   - bifurcacao.

2. Tabela de splits:
   - `train`, `test`, `hard_verb`, `hard_both`, `ambiguous_test`.

3. Figura de `P(ctx=A | S,t)` por classe.

4. Tabela de baselines:
   - sem tempo;
   - por epoca;
   - TimeEncoding simples;
   - interacao token-tempo, se couber.

5. Tabela de resultados por split.

---

## 8. Claims permitidos

Claims seguros:

- O corpus permite testar deriva semantica com ground truth conhecido.
- Marcadores probabilisticos evitam oraculos triviais.
- Splits controlados separam cenarios em que o contexto local ajuda, engana ou e
  neutro.
- O benchmark e adequado para avaliar arquiteturas temporais futuras.

Claims a evitar:

- "Timeformer supera Transformers tradicionais."
- "Resolvemos bifurcacao intra-epoca."
- "O corpus artificial representa plenamente linguagem natural."

---

## 9. Relacao com o artigo internacional

Este paper deve ser citado futuramente como a descricao metodologica do benchmark.
O artigo internacional deve usar o benchmark, mas responder outra pergunta:

> Como uma arquitetura temporal pode tornar representacoes semanticas
> consultaveis e rastreaveis ao longo do tempo?

Para evitar auto-plagio:

- nao copiar blocos textuais entre papers;
- no paper internacional, resumir o corpus e citar este paper se ja estiver
  aceito;
- reservar detalhes de arquitetura para o paper internacional;
- reservar detalhes completos do corpus para este paper.

