# 08 — Registro de Decisões (ADR) e Questões Abertas

Formato: cada decisão tem contexto, decisão, alternativas consideradas e consequências.
Decisões **não são apagadas**; se revertidas, adicionar nova entrada referenciando a anterior.

---

## ADR-001 — Linguagem e stack principal
**Data**: 27/08/2026
**Contexto**: Necessidade de backtest vetorizado rápido, ecossistema de indicadores e dados.
**Decisão**: Python ≥ 3.11 + polars/pandas + DuckDB + vectorbt (inicial).
*(Ajuste 27/08/2026: o ambiente de desenvolvimento tem Python 3.11.9; `requires-python = ">=3.11"`,
CI testa 3.11 e 3.12. Nada no projeto exige recursos exclusivos do 3.12.)*
**Alternativas**: TypeScript/Node (mesma stack do `quedas-do-topo`, mas ecossistema quant fraco);
Rust/Go (performance desnecessária para dados diários).
**Consequências**: Cloudflare Worker existente pode continuar como orquestrador/webhook,
mas o core quant roda em Python (GitHub Actions ou serviço próprio).

## ADR-002 — Fonte primária de dados
**Data**: 27/08/2026
**Decisão**: yfinance como primária; COTAHIST como verificação; Alpha Vantage como backup EUA.
**Motivo**: Custo zero, cobertura B3 + EUA, preços ajustados. Risco de instabilidade mitigado
pela validação e pela fonte secundária.
**Revisar quando**: falhas de yfinance ultrapassarem 2 dias/mês, ou o projeto justificar dado pago.

## ADR-003 — Execução em D+1 abertura
**Data**: 27/08/2026
**Decisão**: Todo sinal calculado no fechamento de D executa na abertura de D+1.
**Alternativa**: fechamento de D+1 (menos gap risk, mas exige acompanhar o pregão).
**Consequência**: Backtest incorpora gaps de abertura; operacionalmente compatível com ordem
enviada na noite anterior ou no pré-abertura.

## ADR-004 — Priorizar B3, validar nos EUA
**Data**: 27/08/2026
**Decisão**: Universo operacional inicial é IBrX-100; S&P 500 usado para validação cruzada.
**Motivo**: Operação real será na B3; EUA fornece histórico longo e independente para checar robustez.

## ADR-005 — Sem ML na geração de sinal no MVP
**Data**: 27/08/2026
**Decisão**: Estratégias baseadas em regras explícitas. ML só em fase 6, como filtro (meta-labeling).
**Motivo**: Interpretabilidade, menor risco de overfitting, baseline mensurável antes de adicionar complexidade.

## ADR-006 — Estratégias plugáveis via contrato único
**Data**: 27/08/2026
**Decisão**: Interface `Strategy.generate(df) -> df[entry, exit, stop, score, max_hold]`.
**Consequência**: Backtester, screener e testes tratam todas as estratégias uniformemente;
adicionar estratégia = adicionar um arquivo + teste.

## ADR-007 — Reparo auditável de barras `high < low`
**Data**: 27/08/2026
**Contexto**: A carga inicial revelou barras do yfinance com `high` ≈ metade do preço em
2012-10-10 (AZZA3, CSAN3, EMBJ3) — erro da fonte, não do mercado.
**Decisão**: `MarketStore.repair_high_low()` roda após cada `update_prices`: substitui high/low
pelo envelope de open/high/low/close e acrescenta `+repair` em `source`.
**Alternativas**: descartar a barra (cria gap falso); manter e falhar o cron todo dia (inútil).
**Consequências**: dado reparado é rastreável por `source LIKE '%+repair'`; indicadores baseados
em range (ATR) ficam levemente subestimados nesses dias, impacto desprezível.

## ADR-008 — Buraco contíguo de histórico ≠ corrupção (`relisting_gap`)
**Data**: 27/08/2026
**Contexto**: NATU3 tem 2017 dias sem dados (deslistagem 2019 → relistagem 2025). O check de
gaps classificava como `critical` (33,8% faltando) e derrubaria o cron para sempre.
**Decisão**: se ≥ 90% dos pregões faltantes formam um único bloco contíguo, emitir
`relisting_gap` (warning) com a data em que o histórico útil recomeça. `missing_days` critical
fica reservado a faltas espalhadas (> 5%).
**Consequências**: estratégias devem tratar o ticker como se o histórico começasse após o gap
(a Fase 2 usará `min_history_rows` a partir do último bloco).

## ADR-009 — Semântica de `close` vs `adj_close` e verificação contra COTAHIST
**Data**: 27/08/2026
**Contexto**: yfinance com `auto_adjust=False` devolve `close` **ajustado por splits e
bonificações** (não por dividendos) e `adj_close` ajustado por tudo. O COTAHIST é bruto.
**Decisão**:
- Sinais e indicadores usam **`adj_close`** (série contínua). Preços de referência para ordens
  e sizing usam `close` do último dia (igual ao negociado).
- `verify-cotahist` classifica divergência com razão constante por trecho no ticker como
  `adjusted_split_or_bonus`; só razão inconsistente é `mismatch`.
**Resultado da verificação (2025, 25 tickers × 10 datas)**: 200 exatos (±0,5%), 50 ajustes
reconhecidos (COGN3, ENGI11, EGIE3, ITUB3 — bonificações), **0 divergências reais**.
**Consequências**: como `adj_close` de todo o histórico muda a cada provento, o cron semanal
deve rodar `update-data --full` (ver Q8).

## ADR-010 — Engine de backtest próprio (resolve Q3)
**Data**: 27/08/2026
**Contexto**: O protocolo exige carteira multi-ativo com limite de posições, ranking por score,
sizing por ATR com lote, stop intradiário, stop por tempo e filtro de regime por data.
Em `vectorbt` isso exige contorções (`from_signals` é por ativo; ranking cross-sectional e
caixa compartilhado não são nativos) e a licença/manutenção da versão open-source é incerta.
**Decisão**: engine próprio (`backtest/engine.py`), loop diário em numpy sobre um `Panel`
(datas × tickers). Regras explícitas e testadas com painéis sintéticos determinísticos
(`tests/test_engine.py`: execução D+1, stop com gap, custos por perna, ranking, lotes, regime).
**Alternativas**: vectorbt (descartado acima); backtesting.py (single-asset); zipline (pesado, morto).
**Consequências**: ~1–3 s por simulação de 16 anos × 100 tickers — suficiente para grids ≤ 30
combinações e walk-forward. Se precisar de grids grandes, vetorizar a camada de sinais e
paralelizar por combinação (multiprocessing), não trocar de engine.

## ADR-011 — Seleção de parâmetros e aprovação
**Data**: 27/08/2026
**Decisão**: parâmetros escolhidos por **Sharpe no treino** (exigindo ≥ `min_test_trades`
trades), robustez medida como média dos vizinhos ±1 passo ÷ ótimo; walk-forward rolling
(3a treino / 1a teste) em treino+validação; conjunto de teste tocado uma única vez por
`run_protocol`. Métricas de trade usam retorno sobre o custo de aquisição.
**Consequência**: `backtest_runs` guarda toda execução (incluindo reprovadas) para o log
de estratégias avaliadas — data snooping fica visível.

## ADR-012 — Regras de carteira dentro do engine; regime como séries por data
**Data**: 27/08/2026
**Contexto**: docs/03 exige limites de setor, correlação, por estratégia, redução por DD mensal
e circuit breaker. Aplicá-los *depois* do backtest (pós-processamento) não altera o caminho da
carteira; precisam agir no momento da decisão.
**Decisão**:
- As regras vivem no `Backtester` como campos do `RiskModel` (0 = desligado) e são avaliadas
  na abertura de D+1 com o patrimônio marcado à abertura. Caps de setor/estratégia **reduzem a
  quantidade**; correlação e subjacente repetido **bloqueiam**; circuit breaker bloqueia novas
  entradas até o DD voltar a menos da metade do limite (histerese); DD mensal > limite
  multiplica o sizing por 0,5 até o patrimônio recuperar o nível de início do mês.
- Regime é externo ao engine: `allow_entries[D]` e `size_factor[D]` (séries por data) vindas de
  `risk/regime.py` — assim o mesmo engine serve single-strategy e carteira, e o screener
  reaproveita as séries.
- Carteira multi-estratégia = painel combinado com colunas `TICKER@estrategia` e `underlying`
  para garantir uma posição por ticker.
**Consequências**: cada decisão de risco é contada em `risk_events`, e o relatório de carteira
traz uma **ablação** (com/sem regime × com/sem regras) para provar que cada camada ajuda —
regra que não melhora Sharpe/MDD na ablação é candidata a remoção.

**Correção 27/08/2026 (cooldown do circuit breaker)**: a primeira versão desarmava só quando o
DD voltava a < metade do limite, medido contra o pico histórico. Na carteira EUA isso produziu
6% de exposição e 424 trades em 16 anos (vs 2.730 sem a regra): uma vez acionado, um sistema
que não opera nunca recupera o pico e fica bloqueado para sempre. Agora o breaker também
desarma após `circuit_breaker_cooldown` (21) pregões, **redefinindo o pico de referência** para
o patrimônio atual (evento `circuit_breaker_reset`). Em operação real, esse é o momento da
revisão humana prevista em docs/03 §2.

## ADR-013 — Risco 0,5%/trade; filtro de tendência do benchmark desligado (resolve Q1)
**Data**: 27/08/2026
**Contexto**: Donchian B3 a 1%/trade tem sinal aprovado mas MC p95 −42% e P(DD>15%)=76%.
A ablação da carteira Donchian-B3 mostrou: regras de carteira **ajudam** (Sharpe 0,71→0,82,
MDD −25%→−13%); filtro IBOV>SMA200 **prejudica** (Sharpe →0,34, MDD →−23%: IBOV lateral por
anos deixa o filtro desligado e a estratégia perde as tendências individuais que o filtro por
ticker já captura); fator de vol é quase neutro (Sharpe −0,05, MDD −1,3 p.p.).
**Decisão**: `risk_per_trade = 0,5%` (MC p95 → −15,5%, P(DD>15%) → 8,5%, MDD realizado
−12,8%); `regime.trend_filter = false`, `regime.vol_filter = true`; só `donchian` habilitada.
**Resultado com os padrões novos (carteira Donchian B3, 2010–2026)**: CAGR 5,1%, Sharpe 0,79,
MDD −10,8%, PF 1,68, 615 trades; último terço: Sharpe 1,36, MDD −6,9%. IBOV no período:
CAGR 5,7%, Sharpe 0,36, MDD −48,6%. Carteira com as três estratégias: Sharpe 0,50, MDD −19%
(as duas reprovadas diluem) → confirmam ficar fora.
**Consequências**: retorno absoluto abaixo do IBOV com um quarto do drawdown; a tese passa a
ser "IBOV-like com risco de renda fixa", e o próximo ganho de CAGR virá de mais estratégias
não correlacionadas (momentum cross-sectional B3, pullback B2), não de alavancar o Donchian.

## ADR-014 — Screener espelha o engine; journal deriva posições das execuções; DuckDB no cache do CI
**Data**: 27/08/2026
**Decisão**:
- **Paridade por construção**: sizing (`position_size`) e ranking (`rank_key`) vivem em
  `risk/sizing.py` e são chamados tanto pelo `Backtester` quanto por `select_entries`. O teste
  de paridade roda o engine em `panel.slice(D, D+1)` e exige tickers e quantidades idênticos ao
  screener em D (dado o mesmo preço). Em produção o screener só conhece o close de D — a
  quantidade sugerida é indicativa; o conjunto e a ordem são exatos.
- **Journal**: sinal ≠ execução. `signals` guarda tudo que o screener emitiu (auditoria e
  aderência); `executions` guarda o que foi feito, com `side`. Posição aberta = compra executada
  ainda não zerada por vendas do mesmo `signal_id`. Patrimônio para sizing = capital inicial +
  P&L realizado (posições abertas ao custo) — conservador e reproduzível sem cotação intradiária.
- **Persistência no CI**: o DuckDB (~150 MB) não entra no repositório; vive no `actions/cache`
  (restaura o mais recente por prefixo, salva com `run_id`). Se o cache expirar (7 dias sem uso),
  `update-data` reconstrói do zero (~5 min) — aceitável.
- **Resolve Q8**: `update-data --full` aos sábados.
**Consequências**: o screener é re-executável para qualquer data (`--as-of`) e o journal não
duplica sinais do mesmo dia; a Fase 5 (aderência real × esperada) já tem os dados de que precisa.

## ADR-015 — Acompanhamento: marcação a mercado pelo journal e desligamento automático
**Data**: 27/08/2026
**Decisão**:
- **Curva real** (`monitoring/performance.py`): caixa + posições × fechamento, reconstruída
  só a partir de `executions` e dos preços do store — sem depender de extrato de corretora.
  Execuções em dias fora do calendário vão para o próximo pregão. Ledger por sinal com P&L,
  retorno sobre custo, `bars_held` e **slippage** (execução ÷ preço de referência do sinal);
  aderência = fração de sinais executados.
- **Retorno por estratégia**: P&L de cada trade fechado distribuído uniformemente entre entrada
  e saída, dividido pelo patrimônio do dia anterior — aproximação suficiente para Sharpe 6m.
- **Regra de desligamento** (`monitoring/health.py`, docs/04 §4): mensal; alerta se Sharpe 6m
  < 0 **ou** DD atual pior que o MC p95 do último backtest (`backtest_runs`) — desde o **ADR-017**
  a comparação usa o p95 do bootstrap diário de 1 ano (`Expected.dd_p95`); 2 alertas
  consecutivos → `paused`; o screener não gera entradas para estratégias pausadas (saídas
  continuam); reativação é manual (`health --resume`). Menos de 40 observações → sem veredito.
- **Relatório mensal** (`monitoring/monthly.py`): realizado × esperado, aderência, slippage,
  trades e sinais do mês; publicado como artifact no job do dia 1 (`daily.yml`).
- **Dashboard** Streamlit (`dashboard/app.py`): leitura direta do DuckDB, sem servidor extra.
**Consequências**: o sistema passa a ter o ciclo completo sinal → execução → medição → veto.
Bug encontrado e corrigido pelo teste de marcação a mercado: `-(a).where(...)` negava também as
vendas (precedência do unário) — o teste de consistência de caixa pegou.

## ADR-016 — Duas sleeves por mercado: Donchian-B3 e Momentum-EUA
**Data**: 27/08/2026
**Contexto**: Fase 6 buscava uma segunda perna não correlacionada. Pullback (B2) foi descartada;
Momentum reprovou na B3 mas passou 9/10 critérios nos EUA (Sharpe OOS 1,84, bootstrap IC
[0,60; 2,98], platô 0,85, WF 0,71, 3× custos OK), falhando apenas o MC p95 — mesmo padrão do
Donchian-B3. Duas estratégias, dois mercados, duas moedas: correlação estrutural baixa.
**Decisão**:
- Estratégias passam a ter `markets:` no `config.yaml`; `Config.enabled_strategies(mkt)` filtra.
  `donchian → [b3]`, `momentum → [us]`; rsi2/drops_ibs/pullback desabilitadas.
- Capital por sleeve: `capital.initial_by_market` (B3 R$ 100 k, EUA US$ 20 k inicial);
  screener, health e relatório mensal usam `capital.for_market(mkt)`.
- Ambas em **paper trading** (journal) até acumular OOS real; formalmente nenhuma está aprovada.
  <br>↳ **Superado em 28/08/2026**: as duas foram aprovadas 10/10 (ADR-017 e ADR-018). Seguem em
  paper trading por falta de execução real, não por falta de validação.
**Carteira Momentum-EUA (2010–2026, regras + vol)**: CAGR 12,9%, Sharpe 1,14, MDD −17,2%,
PF 1,64, 3.676 trades; último terço Sharpe 1,74 / MDD −16,8%. SPY: 14,2% / 0,86 / −33,7%.
Ablação: regime neutro; regras de carteira reduzem MDD (−21%→−17%) sem custo de Sharpe.
**Observação operacional**: com US$ 20 k e 0,5% de risco, o screener sugere 1–3 ações por
posição em papéis de US$ 300–500 — funciona só com fracionário (corretoras EUA permitem) ou com
sleeve maior. Registrado em Q11.
**Consequências**: o sistema opera dois mercados com um único código; o dashboard/health já são
por mercado. O próximo ganho de diversificação real seria uma estratégia de perfil diferente
(mean reversion **longa**, ex.: 3–5 dias com filtro de tendência, que ainda não foi testada).

## ADR-019 — Alvo de preço (`target`) no contrato de sinais

**Contexto**: até aqui uma posição só fechava por sinal booleano, stop ou tempo. Alvo como
"+Y% sobre a entrada" não cabia nesse contrato: `exit` é uma condição por barra, calculada sem
saber a que preço a posição foi aberta. A Pullback contornou isso usando um alvo que é uma
série de preço (máxima de N dias), mas um alvo percentual não tem como ser expresso assim.

**Decisão**: `target` vira a sexta coluna de `SIGNAL_COLUMNS`, simétrica ao `stop` — um preço
absoluto calculado no dia do sinal, NaN quando a estratégia não usa alvo. O engine passa a
checá-lo intradiário junto com o stop, com a mesma regra de gap (abertura acima do alvo executa
na abertura, a favor). **Quando a mesma barra toca stop e alvo, assume-se o stop**: sem dado
intradiário não há como saber a ordem, e o pessimista é o único que não infla o resultado.
`exit_reason` ganha o valor `target`.

**Consequências**: painéis e estratégias anteriores continuam válidos — `Panel.target` é
opcional e vira um frame de NaN quando ausente, e `empty_signals` já devolve a coluna. O alvo é
medido sobre o **fechamento do dia do sinal**, não sobre o preço de execução; num gap de
abertura o ganho realizado difere do `target_pct` nominal, para os dois lados.

**Pendência**: a produção ainda não carrega o alvo. A tabela `signals` do journal tem
`stop_price` e não `target_price`, e acrescentar coluna exige migrar o banco que vive no cache
do Actions. Fica para quando (e se) uma estratégia com alvo for aprovada — hoje a única é a
`dip`, com `enabled: false`.

## ADR-017 — Gate de drawdown: bootstrap em blocos dos retornos diários, horizonte de 1 ano
**Data**: 28/08/2026
**Contexto**: Q10. Donchian-B3 e Momentum-EUA passavam 9 dos 10 critérios e reprovavam sempre no
mesmo: o MC p95 do drawdown. A suspeita registrada era que o Monte Carlo por **embaralhamento de
trades** fosse pessimista demais para estratégias de giro alto. Ao medir, o diagnóstico ficou
diferente — e mais interessante — do que a suspeita.

**O que o MC de trades mede**: soma o P&L nominal dos trades, em ordem aleatória, sobre o capital
**inicial fixo**. Isso tem três defeitos: (a) ignora **sobreposição** — num pregão a carteira tem
até 6 posições, e o efeito líquido delas é o retorno do dia, não uma fila sequencial de trades;
(b) ignora **composição** — um P&L nominal do fim da série, quando o patrimônio já triplicou,
sorteado para o começo vira um drawdown percentual absurdo; (c) o resultado **escala com o
capital inicial** e com o giro, então os números de Donchian (637 trades) e Momentum (3.842)
não são comparáveis entre si nem com um alvo fixo de 15%.

**O achado que mudou a decisão**: max drawdown **cresce com o horizonte**. Medido sobre os 16
anos inteiros, o bootstrap diário dá p95 −27,5% (Donchian) e −33,6% (Momentum) — *pior* que o MC
de trades, o oposto do que a Q10 supunha. Não porque o método seja ruim, mas porque "pior queda
em 16 anos" e "pior queda em 1 ano" são perguntas diferentes, e o alvo de 15% (que é o nível do
circuit breaker, uma régua anual) só faz sentido na segunda.

**Decisão**:
- O gate é `block_bootstrap_drawdown` sobre os **retornos diários da carteira**, blocos de 20
  pregões (circular, preserva autocorrelação), horizonte de **252 pregões**, p95 ≥ −15%.
- O relatório mostra três colunas — bootstrap 1 ano (gate), bootstrap horizonte completo e MC de
  trades — mais uma linha de **calibração**: a distribuição do drawdown realizado nas janelas
  móveis de 1 ano do próprio backtest.
- `health` (ADR-015) passa a comparar o DD realizado com esse p95 (`Expected.dd_p95`), com
  fallback para o MC nos runs gravados antes desta data.

**Evidência de calibração** (é o que sustenta a troca — no mesmo horizonte, o simulado reproduz
o realizado, um pouco mais conservador, como se espera de uma cauda simulada):

| | bootstrap 1a p50 | realizado 1a mediana | bootstrap 1a p95 | realizado 1a p95 | realizado 1a pior |
|---|---|---|---|---|---|
| Donchian/B3 | −6,6% | −6,2% | −12,8% | −10,0% | −10,3% |
| Momentum/EUA | −6,7% | −7,1% | −14,2% | −13,1% | −15,3% |

**Consequências**: com o gate correto, **Donchian-B3 é aprovada 10/10** (o único critério que
faltava). Momentum-EUA continuava reprovada a 0,5% de risco (p95 −19,7%) — não por falta de
edge, mas por consumir orçamento de risco demais; ver ADR-018. **Resolve Q10.**
Ressalva honesta: o bootstrap supõe blocos i.i.d., então não reproduz regimes persistentes
(um 2008 encadeado). A linha de calibração existe justamente para flagrar quando essa hipótese
estiver sendo violada.

## ADR-018 — Risco por trade é por sleeve, calibrado pelo orçamento de drawdown
**Data**: 28/08/2026
**Contexto**: com o gate do ADR-017, Momentum-EUA dava DD p95 de 1 ano de −19,7% a 0,5% de risco
por trade — reprovada com Sharpe OOS 1,85 e bootstrap IC [0,69; 2,97]. O problema não é o sinal,
é o tamanho da aposta: mercados diferentes têm volatilidade diferente e não cabem no mesmo
número. Varredura (mesmo sinal, só mudando o risco por trade):

| risco/trade | Sharpe | CAGR | MDD realizado | DD p95 1 ano | exposição |
|---|---|---|---|---|---|
| 0,50% | 1,14 | 14,4% | −21,2% | −19,6% | 42,8% |
| 0,40% | 1,14 | 11,6% | −17,3% | −16,0% | 34,3% |
| **0,35%** | **1,16** | **10,3%** | **−15,2%** | **−13,9%** | **30,0%** |
| 0,30% | 1,17 | 8,9% | −13,2% | −11,8% | 25,6% |

A relação é linear (os caps de posição e de nº de posições não estão amarrando) e o Sharpe é
plano — reduzir risco custa retorno, não qualidade.

**Decisão**: `risk.risk_per_trade_by_market` no `config.yaml` (mesmo padrão do
`capital.initial_by_market` do ADR-016), com `b3: 0,5%` e `us: 0,35%`; `risk_for_market(mkt)`
alimenta backtest, `portfolio` e screener. O orçamento de drawdown de 15% em 1 ano é o que
define o número — não a preferência por mercado.
**Consequências**: **Momentum-EUA aprovada 10/10** (DD p95 1a −14,2%, Sharpe OOS 1,85, CAGR OOS
21,0%). Carteira EUA passa a Sharpe 1,14 / MDD −12,9% / CAGR 9,5% (era 1,14 / −17,2% / 12,9%).
As duas sleeves ficam sob o mesmo orçamento de risco, que é o que torna somá-las defensável.
Q11 (capital da sleeve EUA) fica **mais aguda**: a 0,35% de US$ 20 k, o risco por trade é US$ 70
e a maioria dos papéis do S&P 500 exige fracionário.

---

## Questões abertas

| # | Questão | Opções | Prazo para decidir |
|---|---|---|---|
| ~~Q1~~ | ~~% de risco por trade~~ | **Resolvida em ADR-013: 0,5%.** Capital inicial segue 100 k (config) | — |
| Q2 | Agendador: GitHub Actions vs Worker Cloudflare chamando serviço | GA é mais simples; Worker reaproveita infra existente | Fase 4 |
| ~~Q3~~ | ~~vectorbt é suficiente para carteira multi-ativo?~~ | **Resolvida em ADR-010: engine própria.** | — |
| Q4 | Bloquear entradas antes de resultado/ex-data? | Testar impacto no backtest | Fase 3 |
| Q5 | Corretora para execução futura | MT5 (Clear/XP/Genial) vs API própria da corretora | Fase 6 |
| Q6 | Incluir BDRs e ETFs no universo B3? | Aumenta universo; liquidez menor | Fase 3 |
| ~~Q7~~ | ~~Onde hospedar dashboard~~ | **Local por enquanto** (`swing-quant dashboard`); Streamlit Cloud exigiria expor o DuckDB — reavaliar se houver multiusuário | — |
| ~~Q8~~ | ~~Frequência do `--full`~~ | **Resolvida em ADR-014: semanal (sábado 12:00 UTC).** | — |
| Q9 | 2020-11-20 ausente em toda a B3 no yfinance (B3 operou) | ignorar (0,1%) vs preencher via COTAHIST | Fase 2 |
| ~~Q10~~ | ~~MC p95 por embaralhamento de trades é pessimista?~~ | **Resolvida em ADR-017: bootstrap em blocos dos retornos diários, horizonte de 1 ano.** O diagnóstico virou: o problema não era pessimismo, era horizonte indefinido | — |
| Q11 | Capital da sleeve EUA — a 0,35%/trade (ADR-018) o risco é US$ 70 por posição, papéis de US$ 300–500 exigem fracionário | fracionário (corretoras EUA permitem) vs sleeve maior vs universo de preço menor | Antes de executar de verdade |
| Q12 | Reavaliar o bootstrap do Sharpe do Donchian: IC [0,03; 2,44] exclui zero por muito pouco, com 147 trades OOS | acumular OOS real (paper trading) vs aceitar como está | Após ~6 meses de paper trading |
| Q13 | Dip/EUA para 9/10 no profit factor por 0,009 (1,391 vs 1,40). O gate de PF ≥ 1,4 faz sentido para uma estratégia de alvo fixo, com 21% de exposição e DD p95 1a de −9,6%? | manter o gate e descartar vs revisar o critério **por princípio** (como o ADR-017 fez com o de drawdown) vs avaliar a dip só como perna de carteira, junto do momentum (**testado em 29/08: inconclusivo — a carteira `momentum+dip` piorou o Sharpe, mas por causa da régua de ranking, ver Q14**) | Antes de qualquer novo run da dip |
| Q14 | Carteira multi-estratégia ordena candidatos por `score` **bruto** (`risk/sizing.py:rank_key`), mas cada estratégia produz score em escala própria — momentum (retorno 12-1) tem mediana 1,09 e máximo 15,0; dip (profundidade da queda) tem mediana 0,53 e teto ~0,9. Com vagas fixas (`max_positions: 6`), a de número maior monopoliza a carteira: em `momentum+dip` (29/08) a dip levou 234 trades e o momentum perdeu 246 — substituição, não diversificação, e o Sharpe caiu de 1,14 para 1,10 | normalizar o score dentro de cada estratégia (percentil ou z-score da própria distribuição) vs manter bruto e aceitar que carteira multi-estratégia só funciona com scores de escala parecida | Antes de qualquer carteira com 2+ estratégias de perfis diferentes |

---

## Log de estratégias avaliadas

| Estratégia | Mercado | Data | Sharpe IS | Sharpe OOS | PF OOS | Trades | Decisão | Relatório |
|---|---|---|---|---|---|---|---|---|
| RSI2 Connors (`rsi_entry=5, exit_sma=5`, sem stop) | B3 | 27/08/2026 | 0,44 | −0,80 | 0,82 | 2539 (694 OOS) | **Reprovada** — só platô (0,84) e nº de trades passam; WF eff. 0,07; positiva apenas a 0× custos (Sharpe 0,55) → edge bruto de ~0,3%/trade não cobre ~0,26% de custos. Cruzado EUA: Sharpe 0,41 | `reports/rsi2_b3_20260827_180755.md` |
| RSI2 Connors (`rsi_entry=5, exit_sma=5`, sem stop) | EUA | 27/08/2026 | 0,66 | 0,71 | 1,13 | 5436 (1146 OOS) | **Reprovada** — robusta (WF 0,68, platô 0,84, lucrativa a 2× custos) mas edge fraco: **baseline aleatória com mesma frequência tem Sharpe 0,83 > 0,71** e SPY buy-and-hold CAGR 14% vs 11%; exposição 95% = beta disfarçado. Cruzado B3: Sharpe −0,33 | `reports/rsi2_us_20260827_181257.md` |
| Donchian B1 (`entry=40, exit=10`, vol>1,5×, stop 2×ATR) | B3 | 27/08/2026 | 0,76 | **1,20** | **1,96** | 637 (147 OOS) | **Reprovada por dimensionamento, sinal aprovado** — 8/10: falha bootstrap (IC [−0,08; 2,39], P(Sharpe≤0)=3,5%, poucos trades OOS) e MC p95 −42% (P(DD>15%)=76%). Edge real: baseline aleatória 0,22; IBOV CAGR 5,7%/MDD −49% vs 9,3%/−25%; robusta a custos 2× (PF 1,45); cruzado EUA Sharpe 0,85. Win rate 37%, payoff 2,6, hold 23d. Ação: testar risco 0,5%/trade (Q1) | `reports/donchian_b3_20260827_212300.md` |
| Quedas+IBS A2 (`drops=3, ibs<0,2`) | B3 | 27/08/2026 | — | −0,36 | 0,92 | 1028 OOS | **Reprovada** — mesmo padrão do RSI2: reversão de 1–3 dias não paga ~0,26% de custos; WF 0,51 e platô 0,82 (robusta em ser fraca) | `reports/drops_ibs_b3_20260827_212341.md` |
| Donchian B1 — **risco 0,5%/trade** | B3 | 27/08/2026 | 0,76 | **1,25** | **2,08** | 637 (147 OOS) | **Reprovada por margem mínima** — 8/10: MC p95 −15,5% (alvo −15%), bootstrap IC [−0,09; 2,45]. MDD realizado −12,8% (OOS −7,6%), CAGR 5,4%. **Candidata nº 1 para paper trading na Fase 4** — reavaliar bootstrap com mais OOS | `reports/donchian_b3_20260827_212809.md` |
| Momentum B3 (`lookback=126, skip=21, exit_sma=100`) | B3 | 27/08/2026 | 0,94 | 0,42 | 1,20 | 2028 (436 OOS) | **Reprovada** — abaixo da baseline aleatória (0,82) no OOS 2023–26; platô 0,56 (parâmetros instáveis); WF 0,54. **Cruzado EUA forte: Sharpe 1,37, CAGR 16,5%, PF 2,27** → protocolo rodado com EUA como principal (linha abaixo) | `reports/momentum_b3_20260827_220709.md` |
| Pullback B2 (`fast=20, mid=50, slow=200`, stop mín−0,5×ATR) | B3 | 27/08/2026 | −0,16 | −0,59 | 0,88 | 3558 (883 OOS) | **Reprovada** — negativa até no treino; stop apertado estourado o tempo todo (holding 3,8d, win 52%, payoff 0,85). Descartada | `reports/pullback_b3_20260827_220754.md` |
| Momentum (`lookback=189, exit_sma=50`, skip 21, stop 3×ATR) | **EUA** | 27/08/2026 | 1,20 | **1,84** | **1,91** | 3842 (859 OOS) | **9/10 — reprovada só por MC p95 (−34,8%)**. CAGR OOS 30%, MDD −16,7%; WF 0,71; platô 0,85; **bootstrap IC [0,60; 2,98] exclui zero**; baseline aleatória 0,89; SPY Sharpe 0,86; lucrativa a 3× custos (Sharpe 0,91). Sizing dominado pelo cap de 20%/posição → teste com cap 10% (linha abaixo). Cruzado B3: 0,25 | `reports/momentum_us_20260827_221121.md` |
| Momentum — **cap 10%/posição** | EUA | 27/08/2026 | 1,20 | 1,84 | 1,91 | 3842 (859 OOS) | Idêntica (exposição 44%→41%): o cap não era o limitante, é o risco por ATR. MC p95 −32,5% vs MDD realizado −21%. **Adotada como sleeve EUA em paper trading** (ADR-016); MC por embaralhamento questionado em Q10 | `reports/momentum_us_20260827_221456.md` |
| **Donchian B1** (`entry=40, exit=10`) — gate do ADR-017 | B3 | 28/08/2026 | 0,68 | **1,25** | **2,08** | 637 (147 OOS) | ✅ **APROVADA 10/10** — o critério que faltava passou com o gate correto: DD p95 1a **−12,8%** (alvo −15%); realizado 1a p95 −10,0%. Baseline aleatória 0,12; IBOV Sharpe 0,36 / MDD −48,6% vs −12,8%. Ressalva: bootstrap IC [0,03; 2,44] passa raspando (Q12) | `reports/donchian_b3_20260828_224441.md` |
| **Momentum** (`lookback=189, exit_sma=50`) — **risco 0,35%** (ADR-018) | EUA | 28/08/2026 | 1,20 | **1,85** | **1,98** | 3842 (859 OOS) | ✅ **APROVADA 10/10** — a 0,5% de risco reprovava só no DD (p95 1a −19,7%); a 0,35% dá **−14,2%** sem perder qualidade (Sharpe OOS 1,85, CAGR OOS 21,0%, MDD −11,9%). Bootstrap IC [0,69; 2,97]; baseline aleatória 0,97; SPY 0,86 / −33,7% | `reports/momentum_us_20260828_225140.md` |
| **Dip A4** (`drop=20%, alvo=15%, topo 60d`, stop 2×ATR, sem filtro) | **EUA** | 29/08/2026 | 0,71 | **1,07** | 1,39 | 1407 (323 OOS) | **9/10 — reprovada só pelo profit factor (1,391 vs 1,40)**. Passa tudo o mais com o protocolo completo: WF 0,78, platô 0,70, bootstrap IC [0,09; 2,04] (P(Sharpe≤0)=1,9%), DD p95 1a −9,6% (o melhor de todas), 2× custos, cruzado B3 > 0. Exposição 21%, perm. 16d, 49% das saídas no alvo (+14,9%) contra 46% no stop (−11,6%). O treino **rejeitou** o filtro de tendência (escolheu `trend_sma=0`). Ver Q13 | `reports/dip_us_20260829_233532.md` |
| Dip A4 (`drop=20%, alvo=15%`, `trend_sma=200` escolhido no treino) | B3 | 29/08/2026 | — | −0,18 | 0,90 | 149 OOS | **Reprovada** — 5/10. O filtro de tendência, que o treino da B3 escolheu, cortou os trades pela metade e piorou tudo (WF 0,09, platô 0,36). Sem filtro dá Sharpe 0,18 / PF 1,05: fraca dos dois jeitos. Mesmo padrão das outras de reversão na B3 | `reports/dip_b3_20260829_232939.md` |
