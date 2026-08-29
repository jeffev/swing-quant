# 01 — Visão e Escopo

## Objetivo

Construir uma aplicação que identifique, valide e acompanhe oportunidades de **swing trade**
(posições mantidas de 2 a 20 pregões) usando regras quantitativas testadas em histórico,
com gestão de risco embutida e operação diária automatizada.

## Problema que resolve

- Sinais discricionários não têm edge mensurável; regras quant permitem medir expectativa.
- Screeners prontos não permitem backtest com custos reais nem controle de risco por carteira.
- Falta de disciplina operacional (tamanho de posição, stop por tempo, exposição setorial).

## Público

Uso pessoal inicialmente (operador único). Arquitetura deve permitir evoluir para multiusuário,
mas isso **não** é requisito do MVP.

## Mercados

| Mercado | Universo inicial | Motivo |
|---|---|---|
| B3 | IBrX-100 (ações com volume médio 20d > R$ 10 mi) | Liquidez suficiente, dados gratuitos via yfinance (`.SA`) / brapi / COTAHIST |
| EUA | S&P 500 | Dados abundantes, ótimo para validar estratégias com histórico longo |

Prioridade: **B3 primeiro**, EUA como universo de validação cruzada (uma estratégia que só funciona
em um mercado é suspeita de overfitting).

## Escopo do MVP (o que entra)

1. Pipeline de dados OHLCV diário ajustado, atualização incremental, armazenado em DuckDB/Parquet.
2. Biblioteca de indicadores e **motor de sinais com estratégias plugáveis** (interface comum).
3. Backtester vetorizado com custos, slippage, sizing e walk-forward.
4. Relatório de métricas padronizado (ver `04-metricas-e-validacao.md`).
5. Screener diário pós-fechamento com ranking + sizing + alerta via Telegram.
6. Journal de sinais gerados (para comparar sinal vs. execução real depois).

## Fora do escopo do MVP (fases posteriores)

- Execução automática em corretora (MT5 / Alpaca) — Fase 4.
- Dados intraday, opções, futuros.
- Machine learning para geração de sinal (só depois de ter baseline de regras simples).
- Dashboard web elaborado — MVP usa Streamlit simples.
- Multiusuário, autenticação, billing.

## Critérios de sucesso do MVP

- Pelo menos **2 estratégias aprovadas** pelos critérios de validação (`04`).
- Screener rodando **diariamente sem intervenção manual** por 4 semanas seguidas.
- Journal mostrando aderência entre sinais gerados e o que o backtest teria gerado
  (sem divergência = sem look-ahead no pipeline de produção).

## Restrições

- Custo de infraestrutura próximo de zero (free tiers: GitHub Actions, Cloudflare Workers,
  Alpha Vantage 25 req/dia já em uso no projeto `quedas-do-topo`).
- Dados gratuitos podem ter falhas; o pipeline precisa de validação de qualidade
  (gaps, splits não ajustados, outliers).
