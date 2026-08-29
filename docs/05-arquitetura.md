# 05 — Arquitetura

## 1. Visão geral

```
┌──────────────┐   cron diário   ┌──────────────────┐
│ Fontes dados │ ──────────────► │ Ingestão + cache │  DuckDB / Parquet
│ yfinance     │                 │ + validação      │
│ brapi        │                 └────────┬─────────┘
│ Alpha Vantage│                          │
│ Fundamentus  │                 ┌────────▼─────────┐
│ COTAHIST B3  │                 │ Indicadores      │  polars / pandas-ta
└──────────────┘                 └────────┬─────────┘
                                          │
                                 ┌────────▼─────────┐
                                 │ Motor de sinais  │  estratégias plugáveis
                                 └────────┬─────────┘
                                          │
                     ┌────────────────────┼─────────────────┐
              ┌──────▼──────┐   ┌─────────▼────────┐  ┌─────▼──────┐
              │ Backtester  │   │ Motor de risco   │  │ Journal    │
              │ + validação │   │ + Screener       │  │ (SQLite)   │
              └──────┬──────┘   └─────────┬────────┘  └─────▲──────┘
                     │                    │                 │
              ┌──────▼──────┐   ┌─────────▼────────┐        │
              │ Relatórios  │   │ Alertas Telegram │────────┘
              │ (HTML/MD)   │   └──────────────────┘
              └─────────────┘
                     │
              ┌──────▼──────┐
              │ Dashboard   │  Streamlit (MVP)
              └─────────────┘
```

## 2. Componentes

| Componente | Responsabilidade | Tecnologia |
|---|---|---|
| `data/` | Download incremental, ajuste por proventos, validação de qualidade, calendário | Python, yfinance, DuckDB |
| `indicators/` | Funções puras vetorizadas (RSI, ATR, MMA, Donchian, Bollinger, IBS) | polars ou pandas + numpy |
| `strategies/` | Uma classe por estratégia seguindo o contrato de `02-estrategias.md` | Python |
| `backtest/` | Simulação, custos, sizing, walk-forward, Monte Carlo, relatório | vectorbt (ou motor próprio simples) |
| `risk/` | Sizing ATR, limites de exposição, correlação, circuit breaker | Python |
| `screener/` | Roda estratégias no dado mais recente, aplica risco, rankeia | Python |
| `alerts/` | Formata e envia mensagens | Telegram Bot API |
| `journal/` | Persistência de sinais e execuções | SQLite |
| `dashboard/` | Visualização de sinais, carteira, métricas | Streamlit |
| `cli/` | Entrypoints: `update-data`, `backtest`, `screen`, `report` | Typer |

## 3. Stack

- **Linguagem**: Python 3.12+ (ecossistema quant maduro).
- **Gerenciador**: `uv` (rápido, lockfile).
- **Dados tabulares**: `polars` para ingestão/indicadores; `pandas` onde bibliotecas exigirem.
- **Armazenamento**: DuckDB (arquivo único `data/market.duckdb`) + Parquet particionado por ano
  para histórico bruto.
- **Backtest**: começar com `vectorbt` (vetorizado, rápido para grids). Se a lógica de carteira
  multi-ativo ficar limitante, escrever motor próprio event-driven simples.
- **Indicadores**: `pandas-ta` ou implementação própria (preferível — menos dependências, controle total).
- **Testes**: `pytest`, com fixtures de dados sintéticos.
- **Qualidade**: `ruff`, `mypy` (strict nos módulos core).
- **Agendamento**: GitHub Actions cron (19h30 BRT, após fechamento e consolidação) — free tier.
  Alternativa: Cloudflare Worker existente (`quedas-do-topo`) disparando webhook.
- **Alertas**: Telegram Bot.
- **Dashboard**: Streamlit Community Cloud (grátis) ou local.

## 4. Fontes de dados

| Fonte | Uso | Limitações |
|---|---|---|
| yfinance | OHLCV diário B3 (`.SA`) e EUA, ajustado | Não oficial; falhas ocasionais; validar |
| brapi.dev | OHLCV B3, fundamentos básicos | Free tier limitado |
| Alpha Vantage | EUA, backup | 25 req/dia (já usado em `quedas-do-topo`) |
| COTAHIST (B3) | Histórico oficial B3, **não ajustado** | Precisa aplicar ajustes de proventos manualmente |
| Fundamentus | Fundamentos B3 (ROE, P/L, dívida) | Scraping; usar cache diário |
| Composição de índices | IBrX-100 / S&P 500 históricos | Necessário para evitar survivorship bias; salvar snapshots mensais |

Estratégia: **yfinance como fonte primária**, COTAHIST como verificação de integridade
(comparar fechamentos não ajustados em datas amostrais).

### 4.1 Problemas conhecidos das fontes (observados na carga de 27/08/2026)

| Problema | Onde | Tratamento |
|---|---|---|
| Barras com `high < low` (`high` ≈ metade do preço) | yfinance B3, 2012-10-10 (AZZA3, CSAN3, EMBJ3) | Reparo automático auditável — ADR-007 |
| Pregão 2020-11-20 ausente em toda a B3 | yfinance (B3 operou nesse dia) | Warning `missing_days` 0,1%; decidir preenchimento via COTAHIST — Q9 |
| `open`/`close` fora de `[low, high]` em datas antigas (2012–2016) | yfinance B3, alguns tickers | Warning; diferenças de centavos por arredondamento de ajuste; sem ação |
| Barra do dia corrente inconsistente (`close_out_of_range`) | yfinance US quando rodado no mesmo dia | Auto-corrige no dia seguinte (lookback incremental de 7 dias) |
| Buraco de anos por deslistagem/relistagem | NATU3 (2019 → 2025) | `relisting_gap` warning — ADR-008 |
| `close` ajustado por splits **e bonificações**, `adj_close` por tudo; COTAHIST bruto | yfinance vs B3 | Semântica definida em ADR-009; `verify-cotahist` reconhece razões constantes por trecho |
| `adj_close` de todo o histórico muda a cada provento | yfinance | `update-data --full` periódico — Q8 |
| Volume zero em índices (^BVSP) e dias ilíquidos | yfinance | Info apenas; filtro de liquidez das estratégias exclui esses casos |

## 5. Modelo de dados (DuckDB)

```sql
-- Preços diários ajustados
CREATE TABLE prices (
  ticker      VARCHAR,
  date        DATE,
  open        DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
  adj_close   DOUBLE,
  volume      BIGINT,
  source      VARCHAR,
  PRIMARY KEY (ticker, date)
);

-- Universo por data (evita survivorship bias)
CREATE TABLE universe (
  index_name  VARCHAR,      -- 'IBRX100', 'SP500'
  as_of       DATE,
  ticker      VARCHAR,
  sector      VARCHAR,
  PRIMARY KEY (index_name, as_of, ticker)
);

-- Eventos corporativos
CREATE TABLE corporate_events (
  ticker VARCHAR, date DATE, type VARCHAR, value DOUBLE   -- split, dividend, jcp
);

-- Sinais gerados (journal)
CREATE TABLE signals (
  id          INTEGER PRIMARY KEY,
  generated_at TIMESTAMP,
  strategy    VARCHAR,
  ticker      VARCHAR,
  side        VARCHAR,      -- 'buy' | 'sell'
  ref_price   DOUBLE,
  stop_price  DOUBLE,
  qty         INTEGER,
  score       DOUBLE,
  max_hold    INTEGER,
  regime      VARCHAR
);

-- Execuções reais (preenchidas manualmente ou via corretora)
CREATE TABLE executions (
  signal_id   INTEGER REFERENCES signals(id),
  executed_at TIMESTAMP,
  price       DOUBLE,
  qty         INTEGER,
  fees        DOUBLE
);

-- Resultados de backtest (para comparação histórica)
CREATE TABLE backtest_runs (
  run_id      VARCHAR PRIMARY KEY,
  strategy    VARCHAR,
  params      JSON,
  period_start DATE, period_end DATE,
  metrics     JSON,
  created_at  TIMESTAMP,
  git_sha     VARCHAR
);
```

## 6. Estrutura de pastas proposta

```
swing-quant/
├── README.md
├── pyproject.toml
├── docs/                    # este planejamento
├── src/swing_quant/
│   ├── data/                # loaders, ajustes, validação, calendário
│   ├── indicators/
│   ├── strategies/          # base.py + uma por arquivo
│   ├── backtest/            # engine, walkforward, montecarlo, report
│   ├── risk/
│   ├── screener/
│   ├── alerts/
│   ├── journal/
│   └── cli.py
├── dashboard/               # streamlit app
├── notebooks/               # exploração (não é código de produção)
├── tests/
├── data/                    # .gitignore — duckdb + parquet
└── .github/workflows/       # cron diário
```

## 7. Fluxo diário (produção) — implementado na Fase 4

1. `22:30 UTC (19:30 BRT)`, seg–sex — `daily.yml` restaura o DuckDB do `actions/cache` e roda
   `swing-quant update-data --market b3` (incremental, 7 dias de lookback, reparo `high<low`,
   validação de qualidade; exit 1 se houver problema crítico → job falha → alerta).
2. `swing-quant screen --market b3`:
   - lê posições abertas do journal e patrimônio estimado (capital + P&L realizado);
   - monta os painéis das estratégias habilitadas (`config.yaml`) e o regime do benchmark;
   - **saídas**: sinal de saída, stop por tempo ou stop tocado em cada posição aberta;
   - **entradas**: candidatos do dia rankeados por score, filtrados por liquidez/regime/
     subjacente, dimensionados com a **mesma função do backtest**, limitados às vagas;
   - grava em `signals` (idempotente) e envia o resumo ao Telegram.
3. Sábado 12:00 UTC — `update-data --market all --full` refaz o histórico (adj_close).
4. `D+1` manhã: operador executa na abertura e registra `record-execution --signal-id … --side buy`;
   ao sair, `--side sell`. `positions` mostra a carteira e o P&L realizado.
5. Qualquer exceção no job → `notify-failure` envia 🚨 ao Telegram.

## 8. Segurança e configuração

- Segredos (token Telegram, chaves API) em variáveis de ambiente / GitHub Secrets; nunca no repo.
- `config.yaml` versionado com parâmetros de estratégias, capital, limites de risco.
- Toda execução de backtest grava `git_sha` + `params` → reprodutibilidade.
