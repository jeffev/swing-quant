# Swing Quant

Plataforma de **swing trade quantitativo** (horizonte 2–20 dias) para ações da B3 e dos EUA.

O **planejamento completo** está em `docs/`; o código é construído seguindo o roadmap
de `docs/06-roadmap.md` (status atual: **Fases 0–5 implementadas** — dados, backtester com
protocolo de validação, estratégias e risco de carteira, screener com paridade provada, journal,
Telegram, cron, dashboard, relatório mensal e regra de desligamento. **Fase 6 em curso**: duas
sleeves **aprovadas 10/10** em 28/08/2026 — Donchian na B3 e Momentum nos EUA (ADR-016), cada uma
com seu capital (`capital.initial_by_market`), seu risco por trade (`risk_per_trade_by_market`,
ADR-018) e habilitação por mercado (`markets:`). Repositório publicado em 29/08/2026
(`github.com/jeffev/swing-quant`, privado) com CI verde em 3.11/3.12; o cron diário está ativo.
Falta só configurar os secrets do Telegram para os alertas saírem).

> O critério de drawdown do protocolo mudou em 28/08/2026 (**ADR-017**): o gate é o p95 do
> bootstrap em blocos dos retornos diários com **horizonte de 1 ano**, calibrado contra as
> janelas móveis de 1 ano do próprio backtest. O Monte Carlo por embaralhamento de trades
> continua no relatório, como referência.

## Índice da documentação

| Arquivo | Conteúdo |
|---|---|
| [docs/01-visao-e-escopo.md](docs/01-visao-e-escopo.md) | Objetivo, público, escopo do MVP, o que fica fora |
| [docs/02-estrategias.md](docs/02-estrategias.md) | Estratégias quant candidatas, regras exatas, filtros de regime |
| [docs/03-gestao-de-risco.md](docs/03-gestao-de-risco.md) | Position sizing, stops, limites de exposição |
| [docs/04-metricas-e-validacao.md](docs/04-metricas-e-validacao.md) | Métricas de backtest, walk-forward, critérios de aprovação |
| [docs/05-arquitetura.md](docs/05-arquitetura.md) | Componentes, stack, fontes de dados, modelo de dados |
| [docs/06-roadmap.md](docs/06-roadmap.md) | Fases, entregáveis e critérios de conclusão |
| [docs/07-armadilhas.md](docs/07-armadilhas.md) | Vieses e erros comuns a evitar (checklist) |
| [docs/08-decisoes.md](docs/08-decisoes.md) | Registro de decisões (ADR) e questões abertas |

## Como rodar

```bash
# requisitos: Python >= 3.11 e uv (pip install uv)
uv sync --group dev              # cria .venv e instala dependências
uv run swing-quant --version
uv run swing-quant show-config   # valida e exibe config.yaml

# dados (Fase 1)
uv run swing-quant update-data --market all     # universo + OHLCV incremental + qualidade
uv run swing-quant update-data -m b3 --full     # refaz o histórico (recalcula adj_close)
uv run swing-quant quality -m us                # só a validação sobre o que já está no banco
uv run swing-quant verify-cotahist --year 2025  # compara fechamentos com o arquivo oficial da B3
uv run swing-quant update-riskfree -m all        # CDI (BCB) e T-bills EUA (BIL): baseline das comparações

# backtest (Fase 2) — protocolo completo: grid, platô, walk-forward, teste OOS, MC, bootstrap,
# custos 0-3x, baseline aleatória, mercado cruzado -> reports/<estrategia>_<mercado>_<data>.md
uv run swing-quant backtest --strategy rsi2 -m b3
uv run swing-quant backtest --strategy rsi2 -m us --quick --no-cross   # mais rápido

# vários de uma vez: fila de combinações estratégia x mercado + tabela comparativa no fim.
# O padrão é --quick/--no-cross (triagem); para decidir de verdade, refaça com `backtest`.
uv run swing-quant bench -s donchian,momentum -m all
uv run swing-quant bench                                # todas as estratégias habilitadas

# carteira combinada (Fase 3): estratégias habilitadas + regime + regras de risco, com ablação
uv run swing-quant portfolio -m b3
uv run swing-quant portfolio -m us --strategies donchian,drops_ibs --no-regime

# produção (Fase 4)
uv run swing-quant screen -m b3 --dry-run            # resumo do dia sem gravar/enviar
uv run swing-quant screen -m b3 --as-of 2026-08-17   # reprocessa um dia passado
uv run swing-quant screen -m b3                      # grava no journal + Telegram (se configurado)
uv run swing-quant record-execution --signal-id 12 --price 33.90 --qty 200 --fees 1.5
uv run swing-quant positions --signals               # posições abertas + sinais recentes

# acompanhamento (Fase 5)
uv run swing-quant health -m b3                      # saúde por estratégia (mensal; 2 alertas -> pausa)
uv run swing-quant health -m b3 --resume donchian    # reativação manual
uv run swing-quant monthly-report -m b3 --month 2026-08
uv sync --extra dashboard && uv run swing-quant dashboard   # Streamlit local
# duas páginas: "app" (o que está valendo: sinais, posições, realizado) e "Backtests"
# (compara os runs registrados, retorno por ano x índice x renda fixa, e o resultado ação
# por ação com os trades de cada uma)
```

Produção automática: `.github/workflows/daily.yml` (seg–sex 19:30 BRT: dados + screener;
sábado: histórico completo). Requer os secrets `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID`
e `alerts.telegram.enabled: true` no `config.yaml`.

```bash

# qualidade
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest --cov
```

Segredos ficam em `.env` (copiar de `.env.example`); nunca no `config.yaml`.

## Resumo em uma frase

Pipeline diário de dados → motor de sinais com estratégias plugáveis → backtester com
walk-forward → screener com sizing por volatilidade → alertas (Telegram) e dashboard.

## Princípios

1. **Nenhuma estratégia entra em produção sem out-of-sample positivo.**
2. **Poucos parâmetros, robustez acima de retorno.**
3. **Risco primeiro**: sizing e limites são parte do motor, não um passo opcional.
4. **Reprodutibilidade**: todo backtest deve ser reexecutável a partir de dados versionados.
