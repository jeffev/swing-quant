# 02 — Estratégias Candidatas

Todas as estratégias seguem a mesma interface: recebem um DataFrame OHLCV (+ indicadores) e
devolvem colunas `entry`, `exit`, `stop`, `score`. O `score` é usado pelo screener para rankear.

**Convenção anti-look-ahead**: sinal calculado com o fechamento do dia D é executado na
**abertura de D+1** (ou no fechamento de D+1, configurável). Nunca no fechamento de D.

---

## A. Mean Reversion (curto prazo, 1–5 dias)

### A1. RSI(2) — Connors
| Item | Regra |
|---|---|
| Filtro de tendência | Fechamento > MMA200 |
| Entrada | RSI(2) < 10 (agressivo: < 5) |
| Saída | Fechamento > MMA5 **ou** 5 pregões decorridos |
| Stop | Sem stop de preço (Connors original); versão com stop = 2×ATR14 para comparar |
| Score | 100 − RSI(2) (quanto mais sobrevendido, maior) |
| Parâmetros a testar | RSI limite ∈ {5, 10, 15}; saída MMA ∈ {3, 5, 10} |

### A2. Quedas consecutivas + IBS
| Item | Regra |
|---|---|
| Filtro | Fechamento > MMA200; volume médio 20d acima do mínimo de liquidez |
| Entrada | ≥ 3 fechamentos consecutivos em queda **e** IBS = (C−L)/(H−L) < 0,2 |
| Saída | Fechamento > máxima do dia anterior **ou** 3 pregões |
| Score | nº de quedas consecutivas × (1 − IBS) |

### A3. Bollinger Reversion
| Item | Regra |
|---|---|
| Filtro | Fechamento > MMA200 |
| Entrada | Fechamento < banda inferior (20, 2σ) |
| Saída | Fechamento > média da banda (MMA20) **ou** 7 pregões |
| Stop | 2,5×ATR14 abaixo da entrada |
| Score | (MMA20 − Close) / desvio-padrão |

---

## B. Momentum / Trend-Following (médio prazo, 5–20 dias)

### B1. Breakout Donchian
| Item | Regra |
|---|---|
| Filtro | Fechamento > MMA50 > MMA200; volume do dia > 1,5× volume médio 20d |
| Entrada | Fechamento > máxima dos últimos 20 pregões |
| Saída | Fechamento < mínima dos últimos 10 pregões (trailing) |
| Stop inicial | 2×ATR14 |
| Score | (Close − máx20 anterior) / ATR14 (força do rompimento) |

### B2. Pullback em tendência
| Item | Regra |
|---|---|
| Filtro | MMA20 > MMA50 > MMA200, todas ascendentes |
| Entrada | Mínima do dia toca a MMA20 **e** fechamento > abertura (candle de reversão) |
| Saída | Fechamento > máxima dos últimos 10 pregões (alvo) **ou** 10 pregões |
| Stop | Abaixo da mínima do candle de entrada − 0,5×ATR14 |
| Score | Inclinação da MMA50 (retorno 20d da média) |

### B3. Momentum cross-sectional (rotação)
| Item | Regra |
|---|---|
| Universo | Todo o IBrX-100 / S&P 500 com filtro de liquidez |
| Ranking | Retorno acumulado de 126 pregões **excluindo** os últimos 21 (12-1 adaptado) |
| Carteira | Top 10% do ranking, pesos iguais ou inverso da volatilidade |
| Rebalanceamento | Semanal (sexta-feira) |
| Filtro de regime | Só aloca se benchmark > MMA200; caso contrário fica em caixa |
| Score | Percentil do momentum |

---

## C. Filtros de Regime (aplicados a todas)

| Filtro | Regra | Efeito |
|---|---|---|
| Tendência do mercado | IBOV (ou SPY) > MMA200 | Habilita entradas compradas |
| Volatilidade | VIX > 30 (EUA) ou vol realizada 20d do IBOV > percentil 90 | Reduz sizing pela metade |
| Liquidez | Volume médio financeiro 20d > R$ 10 mi (B3) / US$ 20 mi (EUA) | Exclui do universo |
| Eventos | Ex-data de proventos / resultado nos próximos 2 dias | Opcional: bloqueia entrada (testar impacto) |

---

### Resultado A1 (27/08/2026 — ver `08-decisoes.md`, log de estratégias)
Reprovada na forma original de Connors nos dois mercados.
- **B3**: edge bruto real mas pequeno (~0,3%/trade, Sharpe 0,55 a custo zero) e holding de
  3,4 pregões → custos de ida-e-volta (~0,26%) anulam.
- **EUA**: robusta (WF 0,68, platô 0,84) mas fraca: Sharpe OOS 0,71 < baseline aleatória 0,83
  e < SPY buy-and-hold; exposição 95% ⇒ o retorno é beta, não alpha.

Implicações para as próximas variantes:
- Exigir **mais seletividade** por trade (ex.: RSI(2) < 5 **e** IBS < 0,2, ou 2 fechamentos abaixo
  do RSI limite) para elevar o ganho médio acima dos custos, aceitando menos trades.
- Testar **saída mais paciente** (fechamento > SMA10 ou máxima de 2 dias) para aumentar payoff.
- Combinar com filtro de regime do IBOV (Fase 3) — os anos ruins do WF (2013–15, 2020–22)
  coincidem com IBOV lateral/baixista.

### Resultado B1 e A2 (27/08/2026 — ver `08-decisoes.md`)
- **A2 Quedas+IBS (B3)**: reprovada; repete o diagnóstico do RSI2 — reversão de 1–3 pregões
  não paga os custos na B3. Conclusão para o backlog: mean reversion curta só volta como
  **filtro de timing de entrada** de outra estratégia, não como estratégia autônoma.
- **B1 Donchian (B3, `entry=40, exit=10`)**: sinal real (win rate 37%, payoff 2,6, holding
  23 pregões; baseline aleatória 0,22 vs Sharpe OOS 1,20; cruzado EUA 0,85; robusta a custos).
  Reprovada por margem mínima só no dimensionamento (MC/bootstrap). Com risco 0,5%/trade e
  regras de carteira: Sharpe 0,79, MDD −10,8% em 16 anos vs IBOV 0,36 / −49%. **Vai para
  paper trading na Fase 4.** O filtro de regime IBOV>SMA200 foi **removido** para ela (ADR-013).

### Resultado B2 e B3 na B3 (27/08/2026, Fase 6 — ver `08-decisoes.md`)
- **B2 Pullback**: reprovada e descartada — negativa até no treino. O stop na mínima do candle
  (−0,5×ATR) é estourado pelo ruído intradiário da B3 (holding 3,8d, payoff 0,85). Se voltar,
  só com stop ≥ 2×ATR e alvo mais longo — mas aí vira um Donchian pior.
- **B3 Momentum (via ranking do engine)**: reprovada na B3 (OOS 2023–26 abaixo da baseline
  aleatória; IBOV lateral pune momentum). **Nos EUA, como mercado principal: 9/10 critérios**
  (Sharpe OOS 1,84, CAGR 30%, PF 1,91, bootstrap IC [0,60; 2,98], platô 0,85, WF 0,71, 3× custos
  OK); falha só o MC p95, como o Donchian. **Sleeve EUA em paper trading** (ADR-016) com
  `lookback=189, exit_sma=50`.

### Aprovação das duas sleeves (28/08/2026 — ADR-017 e ADR-018)
Nenhum sinal mudou; mudou o critério de drawdown e o dimensionamento.
- **B1 Donchian (B3)**: ✅ **aprovada 10/10**. Com o gate de horizonte definido, DD p95 de 1 ano
  = −12,8% (alvo −15%), coerente com o realizado nas janelas de 1 ano (p95 −10,0%). Ressalva:
  o IC do bootstrap do Sharpe passa raspando, [0,03; 2,44], com 147 trades OOS — Q12.
- **B3 Momentum (EUA)**: ✅ **aprovada 10/10** com **risco 0,35%/trade** (ADR-018). A 0,5% dava
  DD p95 de −19,7%; a 0,35% dá −14,2% com Sharpe OOS 1,85 (era 1,84) e CAGR OOS 21%. O sinal
  sempre esteve aprovado — o que não cabia era o tamanho da aposta.

## D. Ordem de implementação

1. **A1 (RSI2)** — mais simples, referência clássica, valida o pipeline inteiro.
2. **B1 (Donchian)** — comportamento oposto (trend), bom para diversificar.
3. **A2 (Quedas + IBS)** — conecta com o projeto `quedas-do-topo` já existente.
4. **B3 (Cross-sectional)** — exige lógica de carteira; deixar para quando o backtester suportar multi-ativo.
5. A3 e B2 — variações para comparação.

---

## E. Interface de estratégia (contrato)

```python
class Strategy(Protocol):
    name: str
    params: dict

    def required_indicators(self) -> list[str]: ...
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recebe OHLCV + indicadores; devolve df com colunas:
        entry (bool), exit (bool), stop (float | NaN), score (float), max_hold (int)."""
```

Regras:
- `generate` **não** pode olhar linhas futuras (usar apenas `shift()` para trás).
- Parâmetros vêm de `params`; nada hardcoded, para permitir grid/robustez.
- Cada estratégia tem um teste unitário com um DataFrame sintético onde o sinal esperado é conhecido.

---

## F. Ideias para depois (backlog)

- Combinar mean reversion + fundamentos (Fundamentus: só operar reversão em empresas com ROE > X, dívida controlada).
- Pares / cointegração (long-short intra-setor na B3).
- Sazonalidade (fim de mês, virada de trimestre).
- Sentimento via fluxo estrangeiro (dados B3).
