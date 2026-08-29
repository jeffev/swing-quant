# 03 — Gestão de Risco

Regra de ouro: **o motor de risco é obrigatório**; nenhuma estratégia gera ordem sem passar por ele.

## 1. Position sizing por volatilidade (ATR)

```
risco_por_trade   = capital × r            (r = 0,5% a 1,0%)
distancia_stop    = k × ATR14              (k = 2,0 padrão; por estratégia)
quantidade        = floor(risco_por_trade / distancia_stop)
valor_posicao     = quantidade × preço
```

Limites adicionais:
- `valor_posicao ≤ capital × 20%` (nenhuma posição domina a carteira).
- `valor_posicao ≤ 1% do volume médio financeiro 20d` (não mover o mercado / slippage).
- Arredondar para lote padrão (100 na B3) quando o valor permitir; senão usar fracionário.

Para estratégias **sem stop de preço** (RSI2 original), usar `distancia_stop = 2×ATR14`
como proxy para o cálculo, mesmo que a ordem de stop não seja enviada.

> **Calibração 27/08/2026 (ADR-013)**: `r = 0,5%`. Com 1% o Donchian sozinho tinha MC p95 de
> −42% e 76% de chance de furar o circuit breaker; com 0,5%, −15,5% e 8,5%.
>
> **Por sleeve, 28/08/2026 (ADR-018)**: `r` passa a ser por mercado
> (`risk.risk_per_trade_by_market`), calibrado para caber no **mesmo orçamento de drawdown** —
> p95 de 1 ano ≤ 15% pelo bootstrap diário (ADR-017). B3 (Donchian) 0,5% → −12,8%;
> EUA (Momentum) 0,35% → −14,2%. O mesmo `r` nos dois mercados levaria a sleeve EUA a −19,7%:
> volatilidades diferentes não cabem no mesmo número.

## 2. Stops

| Tipo | Uso |
|---|---|
| Stop inicial | k×ATR14 abaixo da entrada (definido por estratégia) |
| Stop por tempo | Sair após N pregões sem atingir alvo — **essencial em mean reversion** |
| Trailing | Donchian 10 (trend) ou ATR trailing |
| Stop de carteira | Drawdown mensal > 6% → reduzir sizing em 50% até recuperar |
| Circuit breaker | Drawdown desde o pico > 15% → parar de abrir posições, revisar. No backtest: desarma ao recuperar metade do DD **ou** após 21 pregões de cooldown (redefinindo o pico) — sem isso o sistema fica bloqueado para sempre (ADR-012) |

## 3. Limites de exposição

| Limite | Valor inicial |
|---|---|
| Posições simultâneas | 5–8 |
| Exposição bruta | ≤ 100% do capital (sem alavancagem no MVP) |
| Exposição por setor | ≤ 30% |
| Correlação | Não abrir posição com correlação 60d > 0,8 com posição já aberta |
| Por estratégia | ≤ 50% do capital em uma única estratégia |

## 4. Alocação entre estratégias

- Início: pesos iguais entre estratégias aprovadas. Hoje as duas aprovadas estão em mercados e
  moedas diferentes, com sleeves de capital separadas (ADR-016) — não há rebalanceamento entre
  elas; cada uma respeita o próprio orçamento de risco.
- Evolução: **vol targeting** por estratégia (alocar inversamente à volatilidade dos retornos
  da estratégia em backtest, alvo de vol anual ~10–12% da carteira).
- Rebalancear pesos mensalmente; não reagir a uma semana ruim.

## 5. Custos assumidos no backtest (conservadores)

| Mercado | Corretagem | Emolumentos/taxas | Slippage por perna |
|---|---|---|---|
| B3 | R$ 0 (corretoras zero) — testar também R$ 5/ordem | 0,03% | 0,10% (líquidos) / 0,20% (menos líquidos) |
| EUA | US$ 0 | ~0,001% (SEC/FINRA) | 0,05% |

Impostos (IR 15% sobre ganho líquido em swing na B3) ficam **fora** do backtest,
mas o relatório deve exibir o retorno bruto e uma estimativa líquida.

## 6. Regras operacionais

- Ordens de entrada como **limite na abertura** (ou a mercado na abertura para papéis muito líquidos).
- Nunca aumentar posição perdedora.
- Se o sinal de saída e de entrada coincidirem no mesmo papel, prevalece a saída.
- Feriados e leilões: o pipeline deve conhecer o calendário da B3 (usar `pandas_market_calendars`
  ou lista própria).
- Quando o screener gera mais sinais do que vagas disponíveis, escolher pelo `score`
  e, em empate, pelo menor custo estimado de slippage (mais líquido).
