# Estudo: swing trade x investir

Análise de dados comparando o swing trade quantitativo do projeto com as alternativas passivas
— quatro baselines de ações/caixa, **todas as classes de ativo** que um investidor brasileiro ou
americano podia ter comprado (seção 13), as **carteiras** que ele montaria com elas (seções
14-17) e os **ETFs** com que ele compraria bolsa americana de verdade (seção 18) — sobre os
16,6 anos de histórico do `data/market.duckdb` (2010-01 a 2026-08).

| Arquivo | O que é |
|---|---|
| `swing_vs_investing.ipynb` | O estudo: 12 achados, tabelas e figuras, com as saídas já executadas |
| `study_lib.py` | Os helpers do estudo (baselines passivas, juros sobre o caixa, impostos, DCA) |
| `asset_classes.py` | A comparação entre classes de ativo (seção 13): curvas mensais, retorno real, correlação |
| `investor.py` | As quatro seções do investidor (14-17): carteiras, janelas móveis, imposto por classe, regime de juros |
| `etf_routes.py` | A seção 18: as duas rotas de ETF (B3 x EUA), decomposição câmbio/ativo, custos, impostos e projeção |
| `cycle_portfolio.py` | A carteira que rebalanceia pelo ciclo de mercado (relógio crescimento × inflação), com controle estático, oráculo e teste de rotação |
| `annual_comparison.py` | Tudo na mesma tabela ano a ano: as seis estratégias de swing, as carteiras e todas as classes de ativo |
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

## O que o `etf_routes.py` responde (seção 18)

A seção 13 diz que o S&P 500 em reais foi uma das três coisas que bateram o CDI. Ela diz isso
com um índice: sem taxa, sem spread, sem imposto e sem ticker. A seção 18 pergunta o que sobra
quando a pessoa compra um ETF de verdade, por uma das duas rotas:

- **B3** (`IVVB11`, `SPXI11`, `SPXB11`, `WRLD11`, `ACWI11`, `NASD11`): fundo de índice em reais,
  que acumula o dividendo. 15% sobre o ganho na venda, sem a isenção de R$20 mil das ações.
- **EUA** (`VOO`, `IVV`, `VTI`, `VT`, `VXUS`, `QQQ`, `SCHD`, `BND`): a cota americana direto.
  IOF de 1,1% mais spread na ida e 0,38% mais spread na volta, 30% retidos pelo IRS no
  dividendo e 15% sobre o ganho em reais (Lei 14.754/2023), variação cambial inclusa.

Três decisões de método sustentam os números:

1. **A taxa do fundo já está dentro da cota.** Descontá-la de novo contaria duas vezes. O que o
   módulo mede é o atraso observado de cada fundo brasileiro contra o próprio ETF que ele tem na
   carteira, convertido pelo câmbio do dia — taxa, tracking error, caixa parado e o câmbio que o
   fundo conseguiu, tudo junto. Deu **0,43 ponto ao ano** nos fundos de S&P.
2. **Retorno em reais é produto, não soma.** "O dólar subiu 6% e o ETF 10%" dá 16,6%, não 16%.
   A decomposição mantém o termo cruzado visível.
3. **A perna do câmbio não é dinheiro de graça.** Pela paridade de juros, o câmbio esperado é o
   diferencial CDI menos T-bill — exatamente o que se abre mão ao sair do caixa brasileiro. No
   período o dólar rendeu 6,3% ao ano contra 8,4% implícitos: ficou **2,1 pontos ao ano abaixo
   da paridade**.

Duas correções de dados moram aqui e aparecem impressas no notebook: desdobramentos que o Yahoo
não ajusta (o SPXI11 desdobrou 8 para 1 em jan/2026 e viraria uma queda de 88%) e o piso de
liquidez de R$100 mil/dia, que joga fora os meses em que um ETF recém-lançado é cotado mas não
negociado.

## O que o `cycle_portfolio.py` responde

O `investor.py` mostra que rebalanceamento anual vale pontos de retorno. A pergunta seguinte é
se rebalancear **pelo ciclo** vale mais: quatro alocações, uma por quadrante do relógio de
crescimento × inflação, trocadas quando o quadrante vira em vez de quando o ano vira.

```bash
..\.venv\Scripts\python.exe notebooks\cycle_portfolio.py b3 us   # -> reports/cycle_portfolio_*.md
```

O relatório traz a fase de hoje com os pesos-alvo, a comparação, a grade de sensibilidade e o
teste de rotação. Quatro decisões de método sustentam os números:

1. **A fase é um nowcast, nunca um rótulo com hindsight.** Crescimento sai do retorno das ações
   contra o caixa em 12 meses (disponível no mesmo dia); inflação sai do IPCA/CPI com a
   defasagem de publicação. A fase lida no fim de *t* rege o retorno de *t+1* — um único
   `shift(1)`, que é a garantia inteira de que nada é alimentado pelo retorno que vai ganhar.
2. **O giro é cobrado.** Trocar de quadrante move ~1/3 da carteira; a 15 bps por perna isso é
   custo de verdade, e uma regra que só ganha no bruto não ganha.
3. **O controle tem o mesmo cardápio.** Não é contra uma 60/40: é contra a **média das quatro
   alocações**, rebalanceada no calendário, com o mesmo custo. O que separa as duas linhas é o
   timing e mais nada.
4. **O timing é testado contra sorte.** `rotation_test` desliza a mesma sequência de fases no
   tempo, offset por offset. Preserva a frequência e a persistência das fases e destrói só o
   alinhamento com o mercado — o percentil que devolve diz quanto foi sinal e quanto foi época.

O achado é diferente nos dois mercados, e a grade de sensibilidade é quem decide:

- **B3**: a regra bate o próprio controle em 22% das 18 células da grade, no Sharpe em 11%, e
  **em nenhuma** sofre menos no drawdown (−12,5% contra −8,1% no ajuste padrão). O percentil
  mediano contra as rotações é 51% — ou seja, o calendário verdadeiro das fases não valeu nada.
  A vantagem de +0,4 ponto do ajuste padrão é a melhor célula da grade, não um platô.
- **EUA**: bate o controle no retorno real em **100%** das células (mediana +1,8 ponto ao ano),
  no Sharpe em 83% e no drawdown em 72%, e melhora de forma monótona quanto mais lenta é a
  regra — isso é platô, não célula sortuda. O percentil mediano de rotação é 71%: sugestivo,
  longe de conclusivo. E a 60/40 americana (6,6% real) ainda ganha das duas, porque o cardápio
  é defensivo num período que foi só bolsa subindo.

O oráculo (a melhor das quatro cestas a cada mês, com look-ahead) faz ~20% reais ao ano nos dois
mercados: o teto do cardápio é alto, o que sobra é o sinal.

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

## Os dados da seção 18 (ETFs)

`swing-quant update-etfs` baixa os 14 ETFs do catálogo (`src/swing_quant/data/etfs.py`) com
preço **e** proventos na mesma chamada do yfinance (`Ticker.history`), porque o retorno líquido
da rota americana depende de saber quanto do retorno chegou como dividendo — é sobre ele que o
IRS retém 30%. O câmbio não é rebaixado por esse comando: ele vem do `update-assets`, e as
outras seções já foram calculadas com aquela série.

## O que o `annual_comparison.py` responde

As outras seções comparam uma coisa de cada vez. Esta põe todo mundo numa tabela só — uma linha
por candidato, uma coluna por ano-calendário, retorno real — porque um CAGR diz que algo ganhou
sem nunca dizer *quando*.

```bash
..\.venv\Scripts\python.exe notebooksnnual_comparison.py b3 us   # -> reports/annual_returns_*.md
```

Entram as seis estratégias de swing (cada uma sozinha, **nos dois mercados**, mais o sleeve de
produção combinado), as carteiras (ciclo, controle estático, 60/40, permanente, diversificada) e
todas as classes de ativo, da poupança ao bitcoin. Três decisões sustentam a comparação:

1. **As linhas de swing recebem juros sobre o caixa parado.** O motor paga 0% de propósito e a
   carteira fica ~72% em caixa; comparar essa curva com o CDI seria comparar uma carteira que é
   quase toda caixa contra caixa, com a perna de juros apagada de um lado só.
2. **Toda estratégia roda nos dois mercados.** O `config.yaml` sabe qual é sleeve de produção
   onde (marcado com ★); a tabela mostra as seis contra os dois, porque estratégia que só
   funciona no mercado em que foi ajustada é a forma mais comum de um backtest mentir, e aqui
   isso fica visível de relance.
3. **Cada linha guarda a própria história; o CAGR não.** As células de ano começam quando o dado
   começa (a coluna `desde` diz quando), mas a coluna CAGR usa uma janela única, senão quem
   nasceu num período melhor ganharia a tabela por isso. O bitcoin é a exceção que confirma a
   regra: é 4,5 anos mais curto que o resto, então fica fora do cálculo da janela em vez de
   cobrar de todas as outras linhas um quarto da amostra.

## Aviso

Resultados de backtest, um único caminho histórico, universo com viés de sobrevivência
(composição atual dos índices) e impostos simplificados. A comparação entre classes de ativo é
bruta de imposto e de taxa de administração, e usa índices — que ninguém compra diretamente.
Não é recomendação de investimento.
