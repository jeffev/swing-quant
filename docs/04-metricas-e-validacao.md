# 04 — Métricas e Validação

## 1. Métricas padrão do relatório de backtest

### Retorno
| Métrica | Definição |
|---|---|
| CAGR | Retorno anualizado composto |
| Retorno total | Bruto e líquido de custos |
| Alpha vs. benchmark | CAGR − CAGR do IBOV/SPY no mesmo período |
| Exposição média | % do tempo/capital investido (retorno deve ser lido junto com isto) |

### Risco
| Métrica | Definição / alvo |
|---|---|
| Sharpe | (retorno − CDI/T-bill) / vol anualizada — alvo ≥ 1,0 out-of-sample |
| Sortino | Só penaliza vol negativa |
| Max drawdown | Pior queda pico-vale — alvo ≤ 20% |
| Duração do MDD | Pregões até recuperar o pico |
| Calmar | CAGR / MDD |
| Ulcer Index | Profundidade × duração dos drawdowns |

### Por trade
| Métrica | Definição |
|---|---|
| Nº de trades | Mínimo 200 para significância razoável |
| Win rate | % trades positivos |
| Payoff | Ganho médio / perda média |
| Profit factor | Soma ganhos / soma perdas — alvo ≥ 1,5 |
| Expectancy | Win% × ganho médio − Loss% × perda médio (em R e em %) |
| Holding médio | Pregões por trade |
| Maior sequência de perdas | Para calibrar psicologia e circuit breaker |

## 2. Protocolo de validação (obrigatório)

### 2.1 Divisão temporal
```
|------------- In-sample (60%) -------------|--- Validação (20%) ---|--- Teste (20%) ---|
```
- O conjunto de **teste é tocado uma única vez**, ao final. Se falhar, a estratégia é descartada
  (não reotimizada).
- Período mínimo: 10 anos para B3 (≥ 2014), 15+ para EUA.

### 2.2 Walk-forward
- Janela de otimização: 3 anos → janela de aplicação: 1 ano, deslizando (anchored ou rolling).
- Métrica de eficiência WF = Sharpe out-of-sample / Sharpe in-sample. Alvo ≥ 0,5.

### 2.3 Robustez de parâmetros
- Grid pequeno (≤ 3 parâmetros, ≤ 5 valores cada).
- Plotar superfície de Sharpe: deve ser um **platô**, não um pico isolado.
- Critério: os vizinhos imediatos do parâmetro escolhido devem manter ≥ 70% do Sharpe.

### 2.4 Validação cruzada entre mercados
- Estratégia otimizada na B3 deve ter Sharpe > 0 nos EUA sem reotimização (e vice-versa).

### 2.5 Testes estatísticos
- **Drawdown simulado** (ADR-017) — bootstrap circular em blocos de 20 pregões dos **retornos
  diários da carteira**, 1.000 simulações, horizonte de **252 pregões**: o p95 é o MDD esperado
  em 1 ano e é o que vale como critério. Fixar o horizonte é essencial: o MDD cresce com o
  comprimento do caminho, então um p95 medido sobre 16 anos não se compara a um alvo fixo.
  O relatório também traz, como referência, o bootstrap sobre o histórico inteiro e o Monte
  Carlo por embaralhamento de trades (que ignora sobreposição e composição, e escala com o
  capital inicial — por isso saiu do checklist).
- **Calibração obrigatória**: comparar o p95 simulado de 1 ano com o drawdown realizado nas
  janelas móveis de 1 ano do próprio backtest. O simulado deve ficar próximo e um pouco mais
  conservador; divergência grande indica que a hipótese de blocos i.i.d. não vale ali.
- **Bootstrap** do Sharpe: intervalo de confiança 95% deve excluir zero.
- Comparar com **estratégia aleatória** com mesma frequência e holding (baseline nulo).

### 2.6 Sensibilidade a custos
- Rodar com custos 0×, 1×, 2× e 3×. A estratégia deve continuar lucrativa em 2×.

## 3. Critérios de aprovação para produção

Uma estratégia é **aprovada** quando, no conjunto de teste + walk-forward:

- [ ] Sharpe OOS ≥ 0,8
- [ ] Profit factor ≥ 1,4
- [ ] Nº de trades ≥ 200 (total) e ≥ 30 no conjunto de teste
- [ ] Eficiência walk-forward ≥ 0,5
- [ ] Lucrativa com custos 2×
- [ ] Platô de parâmetros (vizinhos ≥ 70%)
- [ ] Sharpe > 0 no mercado cruzado
- [ ] IC 95% do Sharpe (bootstrap) não inclui zero
- [ ] MDD p95 **em 1 ano** (bootstrap diário) ≤ 15%, o nível do circuit breaker

O último critério é um **orçamento de risco**, não uma propriedade fixa do sinal: se uma
estratégia só falha nele, o caminho é reduzir o risco por trade da sleeve até caber
(`risk.risk_per_trade_by_market`, ADR-018) e revalidar — foi o que aprovou a sleeve EUA.
Falhar em qualquer outro critério é problema do sinal, e sizing não conserta.

**Aprovadas até 28/08/2026**: Donchian/B3 (0,5% de risco) e Momentum/EUA (0,35%).

## 4. Monitoramento em produção

> **Implementado em 27/08/2026 (ADR-015)**: `swing-quant health` (mensal, dia 1 no CI) aplica
> exatamente as regras abaixo com o DD p95 do último `backtest_runs`; `monthly-report` gera o
> comparativo realizado × esperado; o screener ignora estratégias `paused`.

- Comparar mensalmente métricas realizadas × esperadas (backtest).
- Alerta se Sharpe rolling 6 meses < 0 ou drawdown pior que o p95 simulado de 1 ano (ADR-017).
- **Regra de desligamento**: 2 alertas consecutivos → estratégia pausada e reavaliada.
- Journal: todo sinal gerado é gravado com timestamp, preço de referência, sizing e motivo;
  execução real (se houver) é registrada ao lado para medir slippage efetivo.
