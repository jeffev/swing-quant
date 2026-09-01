# Estudo: swing trade x investir

Análise de dados comparando o swing trade quantitativo do projeto com as alternativas passivas
— quatro baselines de ações/caixa, **todas as classes de ativo** que um investidor brasileiro ou
americano podia ter comprado (seção 13) e as **carteiras** que ele montaria com elas (seções
14-17) — sobre os 16,6 anos de histórico do `data/market.duckdb` (2010-01 a 2026-08).

| Arquivo | O que é |
|---|---|
| `swing_vs_investing.ipynb` | O estudo: 12 achados, tabelas e figuras, com as saídas já executadas |
| `study_lib.py` | Os helpers do estudo (baselines passivas, juros sobre o caixa, impostos, DCA) |
| `asset_classes.py` | A comparação entre classes de ativo (seção 13): curvas mensais, retorno real, correlação |
| `investor.py` | As quatro seções do investidor (14-17): carteiras, janelas móveis, imposto por classe, regime de juros |
| `study_results.json` | Todos os números exportados pelo notebook — fonte única do artigo |
| `article/swing_vs_investing.html` | O artigo publicado (gráficos em SVG, dados embutidos) |
| `figures/` | PNGs das figuras (regeneráveis; fora do git, já estão dentro do `.ipynb`) |

## Rodar

O `uv` não está no PATH desta máquina; use o `.venv` direto.

```bash
# dependências do notebook (matplotlib, nbformat, nbconvert, ipykernel)
.venv\Scripts\python.exe -m pip install -e ".[research]"   # ou: uv sync --group research

# reexecutar o estudo inteiro (~2 min) e regravar as saídas no próprio .ipynb
cd notebooks
..\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace swing_vs_investing.ipynb
```

O notebook reusa o motor de produção (`Backtester`, `build_panel`, `run_portfolio`, `metrics`)
com os mesmos custos, sizing e regras de risco do `config.yaml`, então o lado "swing" da
comparação é exatamente o que o `swing-quant portfolio` roda.

## Os dados da seção 13 (classes de ativo)

Os proxies vivem no mesmo `market.duckdb`, ingeridos por `swing-quant update-assets` e
`swing-quant update-macro`. Três não vêm do yfinance de propósito:

- **FIIs**: IFIX oficial da B3. Uma cesta reconstruída por cotação de fundos individuais erra a
  classe em mais de quatro pontos ao ano — o `adj_close` do yfinance só conhece os proventos
  recentes, e um FII distribui quase todo o retorno em dinheiro. O mesmo vale para o SMLL.
- **Imóvel físico**: IVG-R do Banco Central (SGS 21340), que é avaliação, não preço de mercado —
  mede quanto o imóvel *vale*, não quanto ele *rende*, então há uma segunda linha com 4% ao ano
  de aluguel líquido somado.
- **Títulos públicos**: arquivo de preços do próprio Tesouro Direto, marcados a mercado e rolados
  num vencimento-alvo fixo, em vez de uma hipótese de cupom.

A comparação é mensal porque imóvel e poupança só existem mensalmente, e o retorno reportado é
**real** (deflacionado por IPCA no Brasil, CPI nos EUA) — em 16 anos a inflação decide o ranking.

## O que o `investor.py` responde (seções 14-17)

Uma tabela de CAGR de 16 anos não decide nada; estas quatro contas decidem:

1. **Carteiras** (`portfolio`, `blend_returns`): rebalanceamento anual com drift entre as datas.
   O achado: 20% de S&P em reais numa 60/40 brasileira levou o retorno real de 2,2% para 5,5%
   a.a. **e** cortou a queda máxima de 22% para 14%.
2. **Distribuições** (`window_stats`, `time_to_recover`): todas as janelas de 1/3/5/10 anos, não
   um CAGR único. 37% das janelas de 5 anos do Ibovespa terminaram abaixo da inflação, e o índice
   passou 15,8 dos 16,6 anos abaixo do pico real.
3. **Imposto por classe** (`TaxProfile`, `after_tax_cagr`): renda tributada quando chega e
   reinvestida **eleva o custo de aquisição** — ignorar isso bitributaria um FII isento. Cerca de
   3/4 do imposto pago no CDI incide sobre inflação, não sobre ganho real.
4. **Regime de juros** (`by_rate_regime`): no terço de juro real mais baixo, todo ativo de risco
   doméstico teve seu pior resultado; só o S&P em reais pagou.

As premissas de imposto (alíquotas, yields, custos de transação do imóvel) estão declaradas em
`TAX_BR` / `TAX_US`, e as que mais movem o resultado estão nos caveats do artigo.

## O que o `study_lib` acrescenta ao motor

O backtester deixa três coisas de fora de propósito — o que é certo num protocolo de validação
e errado numa comparação contra investir:

1. **Juros sobre o caixa parado.** O motor paga 0% (ver `src/swing_quant/data/riskfree.py`).
   Como a carteira fica ~72% em caixa, `cash_yield_detail()` credita o CDI/T-bill sobre a parte
   não investida e separa a perna de juros da perna de trades — no Brasil, **56% do CAGR era
   juros**.
2. **Baselines com a mesma exposição.** `blended_curve()` monta uma carteira passiva com o mesmo
   percentual médio em bolsa. É a única comparação honesta: contra ela a vantagem cai de 6,7
   para 3,2 pontos ao ano.
3. **Impostos.** `monthly_tax_on_trades()` (15% sobre o ganho mensal, prejuízo compensado,
   isenção de R$20k em vendas) e `interest_tax_schedule()` (regressiva sobre os juros, com
   variante come-cotas).

## Aviso

Resultados de backtest, um único caminho histórico, universo com viés de sobrevivência
(composição atual dos índices) e impostos simplificados. A comparação entre classes de ativo é
bruta de imposto e de taxa de administração, e usa índices — que ninguém compra diretamente.
Não é recomendação de investimento.
