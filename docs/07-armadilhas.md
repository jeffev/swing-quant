# 07 — Armadilhas e Checklist Anti-Viés

Lista de erros clássicos em sistemas quant e como este projeto se protege de cada um.
Usar como **checklist de revisão** antes de aprovar qualquer estratégia.

## 1. Vieses de dados

| Armadilha | Sintoma | Proteção neste projeto |
|---|---|---|
| **Look-ahead bias** | Backtest ótimo, produção medíocre | Sinal em D → execução em D+1; teste de paridade backtest × screener; indicadores só usam `shift()` para trás |
| **Survivorship bias** | Universo atual aplicado ao passado infla retornos (só sobreviventes) | Tabela `universe` com snapshots por data; backtest usa a composição vigente em cada data |
| **Ajuste de proventos incorreto** | Gaps falsos em ex-dividendo geram sinais fantasmas | Usar `adj_close` para sinais; validar contra COTAHIST; registrar `corporate_events` |
| **Dados sujos** | Outliers, volume zero, dias faltantes | Validação automática no pipeline; aborta se crítico |
| **Restatement** | Fundamentos revisados depois da divulgação | Se usar fundamentos, armazenar snapshot na data de coleta (point-in-time) |

## 2. Vieses de modelagem

| Armadilha | Sintoma | Proteção |
|---|---|---|
| **Overfitting** | Parâmetros “perfeitos” que quebram com ±1 | ≤ 3 parâmetros; platô de robustez; walk-forward; teste tocado uma vez |
| **Data snooping / múltiplos testes** | Testar 50 variações e escolher a melhor | Registrar **todas** as variações testadas em `backtest_runs`; aplicar desconto (Deflated Sharpe) se necessário |
| **Poucos trades** | Métricas bonitas com 30 trades | Mínimo 200 trades; bootstrap do Sharpe |
| **Ignorar exposição** | CAGR 15% mas 20% do tempo investido → alavancagem implícita | Reportar exposição média; comparar retorno sobre capital alocado |
| **Regime único** | Só testado em bull market | Período ≥ 10 anos incluindo 2015–16 e 2020 (B3), 2008 e 2022 (EUA) |
| **Mercado único** | Funciona só na B3 | Validação cruzada B3 ↔ EUA |

## 3. Vieses de execução

| Armadilha | Sintoma | Proteção |
|---|---|---|
| **Custos subestimados** | Estratégia de alta frequência de giro morre em produção | Custos conservadores; teste 2× e 3× |
| **Slippage em ilíquidos** | Preenchimento longe do preço de referência | Filtro de liquidez; limite de 1% do volume financeiro; registrar slippage real no journal |
| **Preço de execução irreal** | Backtest executa no fechamento que gerou o sinal | Execução na abertura de D+1 (gap incluso) |
| **Leilões e halts** | Ordem não executa | Calendário; tratar dias sem execução como “sinal expirado” |
| **Lote padrão** | Quantidade não múltipla de 100 | Arredondamento explícito no sizing |

## 4. Vieses psicológicos / operacionais

| Armadilha | Proteção |
|---|---|
| Desligar a estratégia após semana ruim | Regra objetiva de desligamento (`04` §4); revisar só mensalmente |
| Aumentar sizing após sequência boa | Sizing determinístico por ATR e capital; sem discretionaridade |
| “Só mais um filtro” | Todo filtro novo passa pelo protocolo de validação completo |
| Confundir sorte com edge | Baseline aleatória com mesma frequência/holding; IC do Sharpe |
| Não registrar | Journal obrigatório; sinal sem registro não existe |

## 5. Checklist de aprovação (copiar para cada estratégia)

```
Estratégia: ______________________  Data: ____/____/______

Dados
[ ] Universo point-in-time (sem survivorship)
[ ] Preços ajustados, validados contra fonte secundária
[ ] Período ≥ 10 anos com ≥ 2 crises

Sinal
[ ] Nenhum indicador usa dados futuros (revisão de código + teste sintético)
[ ] Execução em D+1
[ ] ≤ 3 parâmetros

Validação
[ ] Split 60/20/20 respeitado; teste tocado 1×
[ ] Walk-forward eficiência ≥ 0,5
[ ] Platô de parâmetros (vizinhos ≥ 70%)
[ ] ≥ 200 trades
[ ] Lucrativa com custos 2×
[ ] Sharpe > 0 no mercado cruzado
[ ] Bootstrap IC95 do Sharpe exclui zero
[ ] Monte Carlo MDD p95 ≤ 15%

Risco
[ ] Passa pelo motor de risco no backtest (não só no screener)
[ ] Exposição média reportada

Decisão: [ ] Aprovada  [ ] Rejeitada  — registrar em 08-decisoes.md
```
