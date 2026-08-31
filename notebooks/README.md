# Estudo: swing trade x investir

Análise de dados comparando o swing trade quantitativo do projeto com quatro alternativas
passivas, sobre os 16,6 anos de histórico do `data/market.duckdb` (2010-01 a 2026-08).

| Arquivo | O que é |
|---|---|
| `swing_vs_investing.ipynb` | O estudo: 7 achados, tabelas e figuras, com as saídas já executadas |
| `study_lib.py` | Os helpers do estudo (baselines passivas, juros sobre o caixa, impostos, DCA) |
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
(composição atual dos índices) e impostos simplificados. Não é recomendação de investimento.
