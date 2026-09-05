<!--
  Post para o Kaggle (Discussions). Copiar do título para baixo.
  Versão em português no fim do arquivo.
-->

# I backtested a swing-trading system for 16 years. Most of its "edge" was interest on idle cash.

I spent the last few weeks building a quantitative swing-trading system for Brazilian and US equities, validating it properly, and then trying to answer a question that turned out to be much harder than building the system: **was it worth running at all?**

The short version: the system beat the index in Brazil by 6.7 points a year. After I fixed the comparison, the real edge was 3.2 points — and more than half of what was left came from a money-market account, not from trading.

Full write-up, with interactive charts: **https://jeffev.github.io/swing-quant/** — code: https://github.com/jeffev/swing-quant

---

## The setup

- **Data:** 2.3M daily bars, Jan 2010 – Aug 2026, B3 (IBrX-100) and S&P 500. Prices adjusted for corporate actions across the whole OHLC bar.
- **Trading side:** two production sleeves, each passed a ten-point validation protocol — walk-forward, out-of-sample, Monte Carlo, block bootstrap, cost stress at 3× the modelled rate, and a random-entry baseline. Donchian breakout on Brazilian large caps, cross-sectional momentum on the S&P 500. Signals form at the close of D, fill at the open of D+1.
- **Investing side:** index buy-and-hold, equal weight, the risk-free rate (CDI / T-bills), and asset-class proxies rebuilt from official sources — B3's own IFIX for property funds, the central bank's appraisal index for physical property, the Treasury's daily price file for government bonds.

## The thing that broke the comparison

A swing book sized by risk — 0.5% of equity divided by 2 ATRs of stop distance — does not hold stocks. It holds a little bit of stocks and **72% cash, on average, in both markets.**

Comparing a 28%-invested portfolio to a 100%-invested index is not a test of trading skill. It is a test of what equities happened to do. If stocks rose, the fully invested side wins by construction; if they fell, the other one does.

The fix is a control that carries the same risk: put 28% in the index and the rest at the risk-free rate, rebalanced daily. Same average exposure, no timing. Against **that**, the Brazilian edge drops from 6.7 points a year to 3.2 — and stops correlating with the interest-rate level (−0.03, against a wild swing from −0.3% to +10.7% when measured against the index).

## Five things I did not expect

**1. In Brazil, 56% of the strategy's return was interest.** The backtest engine pays 0% on idle cash — correct inside a validation protocol, wrong when the question is "trade or invest", since a real broker pays the overnight rate on 72% of the book. Of 12.35% a year, 6.92 points were interest and 5.08 were the trades. A risk-sized swing book in a high-rate country is a money-market fund with an equity overlay.

**2. The environment decided, not the system.** Identical engine, identical risk rules: a clear win in Brazil and a clear loss in the US. The difference is almost entirely CDI at 9.8% against T-bills at 1.3%.

**3. The Brazilian equity risk premium was negative for sixteen years.** Real returns above inflation: cash 3.9%, the main index 0.5%, small caps −1.6%, listed property funds 3.0%, physical property 0.0% before rent. Taking the full risk of the stock market paid three and a half points a year *less* than taking none. Only three things beat cash, and two of them were the dollar in disguise.

**4. The largest single improvement was an allocation, not a rule.** Moving 20% of a 60/40 Brazilian portfolio into the S&P 500 raised the real return from 2.2% to 5.5% a year *and* cut the worst drawdown from 22% to 14%. More return and less risk, from holding *less* of the best-performing asset — the mechanical consequence of a −0.62 correlation between the dollar and the local index.

**5. Timing the cycle did not survive its own control.** I built a portfolio that switches allocation on the growth/inflation quadrant — the investment clock. Against the *average of its own four allocations*, it took a worse drawdown in all 18 parameter settings tested, and its entire return edge lived in a single year. In the US it beat the control in all 18 settings, but a rotation test — sliding the same phase sequence through time, preserving how many phases there were and how long each lasted, destroying only the alignment with the market — put the true calendar at a median 71st percentile. Suggestive; nowhere near proof.

## The part that transfers

If there is one method takeaway, it is that **every finding here only became useful once it had a fair opponent**:

- the same-exposure blend, for the trading book
- the average of the four phase allocations, for the market-timing rule
- the ETF a Brazilian wrapper actually holds, for measuring wrapper drag (0.43 points a year — and note that subtracting the published expense ratio would double-count it, since a fund's quote is already net of its fee)
- the rotated phase sequence, for the timing itself

Without those four controls, the same spreadsheets tell a story of skill in all four cases.

The second takeaway is smaller and cost me a real bug: **in pandas, a comparison against `NaN` returns `False`, not `NaN`.** My market-phase indicator was silently emitting confident "growth is down" readings during the warm-up window of a 12-month rolling calculation, because `NaN > threshold` is `False` and `False` is a valid state. A unit test caught it; nothing in the output looked wrong.

## Caveats, stated up front

Survivorship bias (the universe is today's index membership walked backwards). One historical path — every point estimate is a single draw. In-sample parameters on the trading side. Simplified tax model. Fifteen years is a thin sample for a question whose natural unit is a full market cycle. The write-up has a whole section on this before the conclusion, deliberately.

---

Happy to go deeper on any of it — especially the validation protocol or the rotation test, which is the cheapest way I know to ask "was it the signal, or was it the era?"

---
---

<!-- =========================== EDIÇÃO EM PORTUGUÊS =========================== -->

# Backtestei um sistema de swing trade por 16 anos. Quase toda a "vantagem" era juro sobre caixa parado.

Passei as últimas semanas construindo um sistema quantitativo de swing trade para ações brasileiras e americanas, validando direito, e depois tentando responder uma pergunta que se mostrou bem mais difícil que construir o sistema: **valia a pena rodar isso?**

Resumo: o sistema bateu o índice no Brasil por 6,7 pontos ao ano. Depois de consertar a comparação, a vantagem real era 3,2 pontos — e mais da metade do que sobrou vinha de uma conta remunerada, não de trade.

Artigo completo, com gráficos interativos: **https://jeffev.github.io/swing-quant/pt/**

## O que quebrava a comparação

Uma carteira de swing dimensionada por risco fica **72% em caixa, em média, nos dois mercados**. Comparar uma carteira 28% investida com um índice 100% investido não mede habilidade de trade — mede o que a bolsa fez no período. O conserto é um controle com a mesma exposição: 28% no índice, o resto na taxa livre de risco. Contra ele, a vantagem brasileira cai de 6,7 para 3,2 pontos ao ano e para de correlacionar com o nível de juro.

## Cinco coisas que eu não esperava

1. **56% do retorno da estratégia brasileira era juro.** De 12,35% ao ano, 6,92 pontos vieram do CDI sobre o caixa parado e 5,08 dos trades.
2. **Quem decidiu foi o ambiente, não o sistema.** Mesmo motor e mesmas regras: vitória clara no Brasil, derrota clara nos EUA. A diferença é CDI a 9,8% contra T-bill a 1,3%.
3. **O prêmio de risco de bolsa brasileiro foi negativo em dezesseis anos.** Real acima do IPCA: CDI 3,9%, Ibovespa 0,5%, small caps −1,6%, IFIX 3,0%, imóvel 0,0% sem aluguel.
4. **A maior melhora foi uma alocação, não uma regra.** 20% de uma 60/40 brasileira no S&P 500 levou o retorno real de 2,2% para 5,5% ao ano *e* cortou a pior queda de 22% para 14%.
5. **Acertar o ciclo não sobreviveu ao próprio controle.** Uma carteira que troca de alocação pelo quadrante de crescimento × inflação sofreu queda pior que a média das próprias alocações dela em todas as 18 variações testadas, e toda a vantagem de retorno morava num único ano.

## O que transfere

Todo achado só virou útil quando ganhou um adversário justo: a mistura de mesma exposição, a média das quatro fases, o ETF que o fundo brasileiro carrega, a rotação da sequência de fases. Sem esses quatro controles, as mesmas planilhas contam uma história de habilidade nos quatro casos.

E um detalhe técnico que me custou um bug de verdade: **no pandas, comparação com `NaN` devolve `False`, não `NaN`.** O indicador de fase do ciclo estava emitindo leituras confiantes de "crescimento em baixa" durante o aquecimento de uma janela móvel de 12 meses, porque `NaN > limiar` é `False` e `False` é um estado válido. Um teste pegou; nada na saída parecia errado.
