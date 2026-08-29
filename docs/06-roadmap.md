# 06 — Roadmap

Fases sequenciais; cada uma tem **critério de conclusão** verificável. Datas são estimativas
a partir de 27/08/2026, assumindo dedicação parcial (~10h/semana).

---

## Fase 0 — Fundação (semana 1: até 06/09/2026)

**Entregáveis**
- [x] Repositório com `pyproject.toml` (uv), `ruff`, `mypy`, `pytest` configurados.
- [x] Estrutura de pastas conforme `05-arquitetura.md`.
- [x] `config.yaml` com capital, limites de risco, universo (+ `config.py` com validação pydantic).
- [x] CI: lint + tipos + testes em cada push (`.github/workflows/ci.yml`, matriz 3.11/3.12).
- [x] CLI `swing-quant` com comandos stub (`show-config`, `update-data`, `backtest`, `screen`).

**Conclusão**: ✅ 27/08/2026 — `uv run pytest` 6/6, ruff e mypy strict limpos.
CI verde depende do primeiro push para o GitHub.

---

## Fase 1 — Dados (semanas 2–3: até 20/09/2026)

**Entregáveis**
- [x] Loader yfinance com download incremental para lista de tickers (`data/loader.py`, lookback de 7 dias, `--full` para refazer histórico).
- [x] Persistência em DuckDB — schema completo de `05-arquitetura.md` §5 (`data/store.py`, upserts idempotentes).
- [x] Snapshot da composição atual do IBrX-100 (API B3, com setor) e S&P 500 (Wikipedia, GICS) — `data/universe.py`; `universe_at()` point-in-time. Rotina mensal: o snapshot é gravado a cada `update-data`, então basta o cron diário.
- [x] Calendário de pregões B3 e NYSE (`data/calendar.py`, pandas_market_calendars).
- [x] Validação de qualidade: integridade OHLC, volume zero, retornos extremos (close **e** adj_close), gaps vs. calendário, dados defasados, histórico curto → relatório por severidade (`data/quality.py`).
- [x] Verificação amostral contra COTAHIST (`data/cotahist.py`, CLI `verify-cotahist`): 2025, 25 tickers × 10 datas → 200 exatos, 50 ajustes corporativos reconhecidos, **0 divergências** (ADR-009).
- [x] CLI `update-data --market b3|us|all [--full] [--skip-universe]` e `quality`; exit code 1 quando há problema crítico.
- [x] Reparo auditável de barras `high<low` (ADR-007) e classificação `relisting_gap` (ADR-008).

**Conclusão**: ✅ 27/08/2026 — banco com **2,31 M linhas / 603 tickers** (IBrX-100: 98 + ^BVSP,
S&P 500: 503 + SPY) desde 2010-01-04; 0 falhas de download; qualidade **0 critical** nos dois
mercados após reparo; incremental idempotente (2ª rodada: "99 já em dia", 0 requisições).
Gate: 50 testes offline + 3 de rede, ruff/mypy strict limpos.

---

## Fase 2 — Backtester e primeira estratégia (semanas 4–6: até 11/10/2026)

**Entregáveis**
- [x] Indicadores: SMA, EMA, RSI (Wilder), ATR, Donchian (exclui barra atual), Bollinger, IBS, quedas consecutivas, volume financeiro — testes com valores conhecidos e teste anti-look-ahead (`indicators/core.py`).
- [x] Contrato `Strategy` (pydantic params, grid, `validate_signals`) + estratégia **A1 RSI(2)** com registro (`strategies/`).
- [x] Engine de carteira (`backtest/engine.py`, ADR-010): execução D+1 abertura, stop intradiário com gap, stop por tempo, custos por perna, sizing ATR com lote e caps, ranking por score, filtro de regime/liquidez — 11 testes determinísticos.
- [x] Painel ajustado por proventos (`backtest/panel.py`, ADR-009).
- [x] Métricas (`backtest/metrics.py`) + relatório Markdown com PNGs (equity/drawdown, WF) e CSV de trades; persistência em `backtest_runs` com `git_sha`.
- [x] Walk-forward rolling/anchored e grid de robustez com razão de platô (`backtest/validation.py`).
- [x] Monte Carlo (ordem dos trades), bootstrap em blocos do Sharpe, baseline aleatória.
- [x] Sensibilidade a custos (0×–3×).
- [x] CLI `backtest --strategy rsi2 -m b3|us [--cross] [--quick]` → protocolo completo em ~65 s (B3).

**Conclusão**: ✅ 27/08/2026 — RSI2 avaliado nos dois mercados com o protocolo completo
(B3 em 65 s, EUA em 5 min). **Reprovado em ambos**, por motivos distintos e documentados em
`08-decisoes.md`: na B3 o edge bruto (~0,3%/trade) não cobre custos; nos EUA é robusto mas
perde da baseline aleatória e do buy-and-hold (beta com 95% de exposição). O pipeline
funciona ponta a ponta e o protocolo rejeita o que deve rejeitar — objetivo da fase cumprido.
Relatórios em `reports/` e linhas em `backtest_runs`.

---

## Fase 3 — Estratégias adicionais e motor de risco (semanas 7–8: até 25/10/2026)

**Entregáveis**
- [x] Estratégias **B1 Donchian** (`strategies/donchian.py`) e **A2 Quedas+IBS** (`strategies/drops_ibs.py`) implementadas, registradas e testadas; validadas pelo protocolo (resultados no log de `08-decisoes.md`).
- [x] Filtros de regime (`risk/regime.py`): tendência do benchmark (SMA200 → bloqueia entradas) e volatilidade realizada acima do percentil 90 rolante (→ sizing × 0,5), sem look-ahead; liquidez já no engine.
- [x] Regras de carteira no engine (ADR-012): uma posição por subjacente, cap setorial, cap por estratégia, filtro de correlação 60d, redução por DD mensal, circuit breaker com histerese — contadores em `risk_events`; 8 testes determinísticos.
- [x] Painel combinado multi-estratégia (`backtest/portfolio.py`, colunas `TICKER@estrategia`), relatório com atribuição por estratégia e **ablação** (com/sem regime, com/sem regras); CLI `portfolio -m b3|us [--strategies a,b] [--no-regime]`.

**Conclusão** (critério original: ≥ 2 estratégias aprovadas; carteira com Sharpe OOS ≥ 1,0 e MDD ≤ 20%):
✅ **Atingida em 28/08/2026** — Donchian/B3 e Momentum/EUA aprovadas 10/10 depois que o gate de
drawdown passou a ter horizonte definido (ADR-017) e o risco por trade passou a ser por sleeve
(ADR-018). Carteiras: B3 Sharpe 0,79 / MDD −10,8%; EUA Sharpe 1,14 / MDD −12,9%. O histórico do
diagnóstico de 27/08 fica abaixo, porque explica por que os números mudaram sem que nenhum sinal
mudasse.

<details><summary>Status em 27/08/2026 (antes do ADR-017/018)</summary>

⚠️ **Parcialmente atingida.** Nenhuma estratégia aprovada formalmente:
Quedas+IBS reprovada (mesmo padrão do RSI2); **Donchian reprovada por margem mínima** (8/10,
MC p95 −15,5% vs −15%, bootstrap com 147 trades OOS) mas com sinal claramente real (baseline
aleatória 0,22 vs 1,20; cruzado EUA 0,85). Carteira só-Donchian com os padrões novos
(ADR-013): completo Sharpe 0,79 / MDD −10,8%; último terço Sharpe 1,36 / MDD −6,9% —
cumpre o numérico, mas com uma única estratégia. A ablação derrubou o filtro de tendência
do benchmark e confirmou as regras de carteira; o circuit breaker ganhou cooldown (ADR-012).
**Decisão**: seguir para a Fase 4 com o Donchian em **paper trading** (journal) enquanto se
acumula OOS para o bootstrap, e trazer B3 cross-sectional/B2 pullback como candidatas a
segunda perna não correlacionada.

</details>

---

## Fase 4 — Screener em produção (semanas 9–10: até 08/11/2026)

**Entregáveis**
- [x] CLI `screen -m b3|us [--as-of] [--dry-run] [--no-alert]` — estratégias habilitadas no dado mais recente: saídas das posições abertas (sinal, tempo, stop tocado) + entradas rankeadas com sizing; reprocessa datas passadas com `--as-of`.
- [x] **Sizing e ranking compartilhados** (`risk/sizing.py`) entre engine e screener; **teste de paridade** (`tests/test_screener.py`): em 20+ dias com sinal, tickers e quantidades do screener == compras do engine em D+1. Verificado também nos dados reais: 03/08 e 17/08/2026 reproduzem as entradas do backtest (GOAU4, CBAV3, UGPA3).
- [x] Journal em DuckDB (`journal/core.py`): `signals` (idempotente por dia/mercado/ticker/estratégia/lado) e `executions` com `side`; posições abertas derivadas das execuções; P&L realizado e patrimônio estimado alimentam o sizing do dia seguinte.
- [x] Telegram (`alerts/telegram.py`): resumo diário em Markdown (regime, saídas, entradas com preço/qty/stop/score), alerta de falha; sem credenciais vira no-op.
- [x] GitHub Actions `daily.yml`: seg–sex 22:30 UTC (`update-data` → `screen`), sábado `update-data --full` (Q8 resolvida: semanal); DuckDB persistido via `actions/cache`; `workflow_dispatch` com `market`/`dry_run`; passo de falha chama `notify-failure`.
- [x] Alerta de dado defasado (> 5 dias sem pregão novo → nota no resumo e exit 1) e de exceção (job failure).
- [x] CLI `record-execution --signal-id --price --qty [--side] [--fees]` e `positions [--signals]`.

**Conclusão**: 20 pregões consecutivos com execução automática sem intervenção;
sinais gerados coincidem com o que o backtester geraria para as mesmas datas (teste de paridade).
Status 29/08/2026: paridade ✅ (teste + reprocessamento real). **Repositório publicado**
(`github.com/jeffev/swing-quant`, privado), CI verde em 3.11 e 3.12, workflow "Screener diário"
ativo — a contagem dos 20 pregões começa no primeiro cron. O `MARKET` do cron passou a `all`:
em execução agendada não há `github.event.inputs`, então o padrão anterior (`b3`) deixava a
sleeve dos EUA sem atualização de dados e sem screener.
Pendente: `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` (`gh secret set`) e
`alerts.telegram.enabled: true` no config — sem eles o screener roda e grava no journal
normalmente, só não envia alerta. Atenção ao primeiro cron: o cache do Actions está vazio,
então o `update-data` reconstrói o banco inteiro (dois mercados, desde 2010) dentro do
`timeout-minutes: 45`.

---

## Fase 5 — Dashboard e acompanhamento (semanas 11–12: até 22/11/2026)

**Entregáveis**
- [x] Streamlit (`dashboard/app.py`, `swing-quant dashboard`): sinais mais recentes, posições abertas, equity real × benchmark, drawdown, Sharpe rolling 3m, ledger, realizado × esperado, saúde, backtests registrados.
- [x] Relatório mensal (`monthly-report`): realizado × esperado por estratégia, aderência sinal → execução, slippage médio/mediano/pior, trades fechados, sinais do mês; job do dia 1 no `daily.yml` publica como artifact.
- [x] Regra de desligamento (`health`, ADR-015): Sharpe 6m < 0 ou DD pior que MC p95 → alerta; 2 seguidos → pausa respeitada pelo screener; `health --resume` reativa.
- [x] Marcação a mercado da carteira real a partir do journal (`monitoring/performance.py`), com teste de consistência de caixa (que encontrou um bug real de sinal nas vendas).

**Conclusão**: ✅ 27/08/2026 — `monthly-report -m b3 --month 2026-08` gerou `reports/monthly_b3_2026-08.md`
(curva plana: ainda sem execuções); `health -m b3` → "dados insuficientes (0 obs)"; dashboard roda
(`uv sync --extra dashboard`). O conteúdo real só aparece quando houver execuções registradas.

---

## Fase 6 — Evolução (iniciada em 27/08/2026, sob demanda)

**Feito**
- [x] **B3 Momentum** (`strategies/momentum.py`): 12-1 adaptado (126 pregões pulando 21); a seleção
  cross-sectional usa o ranking de score do engine (vagas preenchidas pelos maiores momentums do
  dia); rotação por saída (momentum < 0, perda da SMA100, reavaliação a cada 63 pregões).
- [x] **B2 Pullback** (`strategies/pullback.py`): SMA20>50>200 ascendentes, toque na SMA20 com
  candle de reversão, alvo = máxima de 10 dias, stop = mínima − 0,5×ATR.
- [x] Ambas registradas, com testes (lógica, anti-look-ahead) e entradas no `config.yaml`.
- [x] **Resultados** (log em `08-decisoes.md`): Pullback reprovada e descartada; Momentum
  reprovada na B3 mas **9/10 nos EUA** (Sharpe OOS 1,84, bootstrap exclui zero) — falha só o
  MC p95, como o Donchian. **Adotada como sleeve EUA em paper trading** (ADR-016).
- [x] Habilitação de estratégia **por mercado** (`markets:` no config) e **capital por sleeve**
  (`capital.initial_by_market`); screener/health/mensal por mercado. Verificado ao vivo:
  `screen -m us --dry-run` gera 6 entradas de momentum; carteira Momentum-EUA 2010–2026:
  Sharpe 1,14 / CAGR 12,9% / MDD −17% (SPY 0,86 / 14,2% / −34%).
- Questões novas: Q10 (metodologia do MC p95), Q11 (capital da sleeve EUA).

**Feito em 28/08/2026 — as duas sleeves aprovadas**
- [x] **Q10 resolvida (ADR-017)**: o gate de drawdown passou a ser o bootstrap em blocos dos
  retornos diários com **horizonte de 1 ano**, com calibração contra as janelas móveis de 1 ano
  do próprio backtest. O achado que mudou a decisão: o MDD cresce com o horizonte, então o
  problema do MC antigo não era pessimismo e sim horizonte indefinido — sobre 16 anos, o
  bootstrap diário é ainda *mais* pesado (−27,5% / −33,6%).
- [x] **ADR-018**: `risk.risk_per_trade_by_market` — cada sleeve calibrada para caber no mesmo
  orçamento de 15% em 1 ano (B3 0,5%, EUA 0,35%).
- [x] **Donchian/B3 aprovada 10/10** (DD p95 1a −12,8%) e **Momentum/EUA aprovada 10/10**
  (0,35% de risco → −14,2%; Sharpe OOS 1,85). Relatórios de 28/08 em `reports/`.
- Questões novas: Q12 (bootstrap do Sharpe do Donchian passa raspando, IC [0,03; 2,44]).

**Backlog**
- Estratégia **B3 cross-sectional pura** (rebalanceamento explícito por rank) se o Momentum via
  ranking do engine se mostrar insuficiente.
- Integração de execução: MetaTrader 5 (B3) e/ou Alpaca (EUA), começando em **paper trading**.
- Combinar fundamentos (Fundamentus) como filtro.
- Explorar ML apenas como **filtro** sobre sinais de regras (meta-labeling), nunca como gerador.
- Multiusuário / produto, se fizer sentido.

---

## Marcos

| Marco | Data alvo | Evidência |
|---|---|---|
| M1 Dados confiáveis | 20/09/2026 | ✅ 27/08/2026 — qualidade 0 critical, 2,31 M linhas |
| M2 Primeira estratégia validada | 11/10/2026 | ✅ 28/08/2026 — Donchian/B3 10/10 (ADR-017) |
| M3 Carteira aprovada | 25/10/2026 | ✅ 28/08/2026 — 2 sleeves aprovadas; B3 Sharpe 0,79 / MDD −10,8%, EUA 1,14 / −12,9% |
| M4 Produção estável | 08/12/2026 | 20 pregões automáticos + paridade |
| M5 Primeiro relatório mensal | jan/2027 | Relatório realizado × esperado |
