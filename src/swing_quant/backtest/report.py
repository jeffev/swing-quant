"""Relatório do protocolo: Markdown (+ PNG opcional via matplotlib) e persistência no store."""

from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from swing_quant.backtest.metrics import Metrics, drawdown_series, rolling_drawdowns
from swing_quant.backtest.protocol import ProtocolResult
from swing_quant.data.store import MarketStore


def _f(x: object, fmt: str = ".2f") -> str:
    if x is None:
        return "-"
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(x)
    return "-" if math.isnan(v) else format(v, fmt)


def _pct(x: object) -> str:
    return _f(x, ".1%")


def _metrics_table(cols: dict[str, Metrics]) -> str:
    rows: list[tuple[str, Callable[[Metrics], str]]] = [
        ("Período", lambda m: f"{m.start} → {m.end} ({m.years}a)"),
        ("Retorno total", lambda m: _pct(m.total_return)),
        ("CAGR", lambda m: _pct(m.cagr)),
        ("Volatilidade", lambda m: _pct(m.volatility)),
        ("Sharpe", lambda m: _f(m.sharpe)),
        ("Sortino", lambda m: _f(m.sortino)),
        ("Max drawdown", lambda m: _pct(m.max_drawdown)),
        ("Duração MDD (pregões)", lambda m: str(m.max_drawdown_days)),
        ("Calmar", lambda m: _f(m.calmar)),
        ("Renda fixa no período", lambda m: _pct(m.rf_cagr)),
        ("CAGR − renda fixa", lambda m: _pct(m.excess_over_rf)),
        ("Exposição média", lambda m: _pct(m.exposure_avg)),
        ("Nº trades", lambda m: str(m.n_trades)),
        ("Win rate", lambda m: _pct(m.win_rate)),
        ("Payoff", lambda m: _f(m.payoff)),
        ("Profit factor", lambda m: _f(m.profit_factor)),
        ("Expectancy / trade", lambda m: _pct(m.expectancy_pct)),
        ("Holding médio (pregões)", lambda m: _f(m.avg_hold_bars, ".1f")),
        ("Máx. perdas seguidas", lambda m: str(m.max_consecutive_losses)),
        ("Custos totais", lambda m: _f(m.fees_total, ",.0f")),
    ]
    head = "| Métrica | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    body = "".join(
        f"| {name} | " + " | ".join(fn(m) for m in cols.values()) + " |\n" for name, fn in rows
    )
    return head + body


def _df_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    if df.empty:
        return "_vazio_\n"
    cols = list(df.columns)
    out = "| " + " | ".join(str(c) for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(_f(v, floatfmt))
            else:
                cells.append(str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out


def render_markdown(r: ProtocolResult, plots: dict[str, Path] | None = None) -> str:
    status = "✅ APROVADA" if r.approved else "❌ REPROVADA"
    lines = [
        f"# Backtest — {r.strategy_name} / {r.market.upper()}",
        "",
        f"**Status**: {status}  ",
        f"**Parâmetros escolhidos (treino)**: `{r.params}`  ",
        "**Timeframe**: candles diários (D1) — sinal no fechamento, execução na abertura do "
        "pregão seguinte; `bars_held` e `max_hold` contam pregões  ",
        f"**Gerado em**: {dt.datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Checklist de aprovação",
        "",
        "| Critério | Resultado |",
        "|---|---|",
    ]
    lines += [f"| {k} | {'✅' if v else '❌'} |" for k, v in r.checklist.items()]
    lines += [
        "",
        "## Split temporal",
        "",
        f"- Treino: {r.split.train}",
        f"- Validação: {r.split.val}",
        f"- Teste (tocado 1×): {r.split.test}",
        "",
    ]
    lines += [
        "## Métricas",
        "",
        _metrics_table(
            {"Treino": r.metrics_train, "Teste OOS": r.metrics_test, "Completo": r.metrics_full}
        ),
    ]
    if r.benchmark:
        lines += [
            "",
            f"Benchmark buy-and-hold (período completo): CAGR {_pct(r.benchmark.get('cagr'))}, "
            f"Sharpe {_f(r.benchmark.get('sharpe'))}, MDD {_pct(r.benchmark.get('max_drawdown'))}",
        ]
    if r.blend:
        w = r.blend.get("weight", float("nan"))
        lines += [
            "",
            f"Carteira passiva de mesma exposição ({w:.0%} índice + renda fixa): "
            f"CAGR {_pct(r.blend.get('cagr'))}, Sharpe {_f(r.blend.get('sharpe'))}, "
            f"MDD {_pct(r.blend.get('max_drawdown'))} — é contra ela que o gate do ADR-020 mede, "
            "não contra o índice cheio.",
        ]
    if plots:
        lines += ["", "## Gráficos", ""]
        lines += [f"![{k}]({p.name})" for k, p in plots.items()]
    lines += [
        "",
        "## Robustez de parâmetros (treino)",
        "",
        f"Razão de platô (vizinhos/ótimo): **{_f(r.plateau)}** (alvo ≥ {r.thresholds.plateau_min})",
        "",
        _df_table(r.grid.sort_values("sharpe", ascending=False)),
    ]
    lines += [
        "",
        "## Walk-forward (treino + validação)",
        "",
        f"Eficiência (Sharpe OOS encadeado / média Sharpe IS): **{_f(r.walk_forward.efficiency)}** "
        f"(alvo ≥ {r.thresholds.wf_efficiency_min})",
        "",
        _df_table(r.walk_forward.windows),
    ]
    roll = rolling_drawdowns(r.full.equity, window=r.thresholds.dd_horizon)
    lines += [
        "",
        "## Drawdown simulado (ADR-017)",
        "",
        "O **gate** é o bootstrap em blocos dos retornos diários com horizonte de 1 ano: preserva",
        "sobreposição de posições, composição e autocorrelação, e fixa o horizonte — sem isso o",
        "p95 do MDD cresce com o tamanho da amostra e deixa de ser comparável ao alvo. As outras",
        "duas colunas ficam como referência: medem o MDD do histórico inteiro, não de 1 ano.",
        "",
        "| Percentil | **Bootstrap 1 ano (gate)** | Bootstrap horizonte completo | MC ordem dos trades |",
        "|---|---|---|---|",
        f"| MDD mediano | {_pct(r.dd_bootstrap.get('mdd_p50'))} "
        f"| {_pct(r.dd_bootstrap_full.get('mdd_p50'))} | {_pct(r.monte_carlo.get('mdd_p50'))} |",
        f"| **MDD p95** | **{_pct(r.dd_bootstrap.get('mdd_p95'))}** "
        f"(alvo ≥ -{r.thresholds.dd_p95_max:.0%}) "
        f"| {_pct(r.dd_bootstrap_full.get('mdd_p95'))} | {_pct(r.monte_carlo.get('mdd_p95'))} |",
        f"| MDD p99 | {_pct(r.dd_bootstrap.get('mdd_p99'))} "
        f"| {_pct(r.dd_bootstrap_full.get('mdd_p99'))} | {_pct(r.monte_carlo.get('mdd_p99'))} |",
        f"| P(DD > circuit breaker) | {_pct(r.dd_bootstrap.get('prob_dd_gt_ruin'))} "
        f"| {_pct(r.dd_bootstrap_full.get('prob_dd_gt_ruin'))} "
        f"| {_pct(r.monte_carlo.get('prob_dd_gt_ruin'))} |",
        "",
        "**Calibração** — drawdown realizado nas janelas móveis de 1 ano do próprio backtest "
        f"({roll['n']:.0f} janelas): mediana {_pct(roll['p50'])}, p95 {_pct(roll['p95'])}, "
        f"pior {_pct(roll['worst'])}. O bootstrap de 1 ano deve ficar próximo (e um pouco mais "
        "conservador) desses números; se divergir muito, a hipótese de i.i.d. entre blocos "
        f"está sendo violada. MDD realizado no histórico inteiro: {_pct(r.metrics_full.max_drawdown)}.",
        "",
    ]
    lines += [
        "## Bootstrap do Sharpe (teste OOS)",
        "",
        f"- IC 95%: [{_f(r.bootstrap.get('sharpe_lo'))}, {_f(r.bootstrap.get('sharpe_hi'))}]",
        f"- P(Sharpe ≤ 0): {_pct(r.bootstrap.get('p_sharpe_le_0'))}",
        "",
    ]
    lines += ["## Sensibilidade a custos (período completo)", "", _df_table(r.costs)]
    lines += [
        "",
        "## Baseline aleatória (teste OOS, mesma frequência/holding)",
        "",
        f"- Sharpe médio do aleatório: {_f(r.baseline.get('random_sharpe_mean'))}; "
        f"p95: {_f(r.baseline.get('random_sharpe_p95'))}; estratégia: {_f(r.metrics_test.sharpe)}",
        "",
    ]
    if r.cross_market is not None:
        lines += [
            "## Mercado cruzado (mesmos parâmetros, período completo)",
            "",
            f"- Sharpe {_f(r.cross_market.get('sharpe'))}, CAGR {_pct(r.cross_market.get('cagr'))}, "
            f"PF {_f(r.cross_market.get('profit_factor'))}, trades {_f(r.cross_market.get('n_trades'), '.0f')}",
            "",
        ]
    if r.notes:
        lines += ["## Observações", ""] + [f"- {n}" for n in r.notes] + [""]
    lines += [
        "## Saídas por motivo (período completo)",
        "",
        _df_table(
            r.full.trades["exit_reason"].value_counts().rename_axis("motivo").reset_index(name="n")
        ),
    ]
    return "\n".join(lines) + "\n"


def make_plots(r: ProtocolResult, out_dir: Path, stem: str) -> dict[str, Path]:
    """Equity + drawdown e curva OOS do walk-forward. Silencioso se matplotlib não existir."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    plots: dict[str, Path] = {}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    eq = r.full.equity / r.full.equity.iloc[0]
    ax1.plot(eq.index, eq, label=f"{r.strategy_name} (completo)")
    for w, color in ((r.split.train, "#dddddd"), (r.split.test, "#ffe4b3")):
        ax1.axvspan(w.start, w.end, color=color, alpha=0.5)
    ax1.set_ylabel("Equity (base 1)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    dd = drawdown_series(r.full.equity)
    ax2.fill_between(dd.index, dd, 0, color="crimson", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"{stem}_equity.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    plots["equity"] = p

    fig, ax = plt.subplots(figsize=(10, 3.5))
    oos = r.walk_forward.oos_equity
    ax.plot(oos.index, oos, color="darkgreen")
    ax.set_title("Walk-forward — equity OOS encadeada")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"{stem}_walkforward.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    plots["walk_forward"] = p
    return plots


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def save_report(r: ProtocolResult, out_dir: Path, store: MarketStore | None = None) -> Path:
    """Escreve <stem>.md (+PNGs, +trades.csv) e grava a linha em `backtest_runs`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{r.strategy_name}_{r.market}_{stamp}"
    plots = make_plots(r, out_dir, stem)
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(render_markdown(r, plots), encoding="utf-8")
    r.full.trades.to_csv(out_dir / f"{stem}_trades.csv", index=False)

    if store is not None:
        run_id = f"{stem}_{uuid.uuid4().hex[:6]}"
        metrics = {
            "test": r.metrics_test.to_dict(),
            "full": r.metrics_full.to_dict(),
            "checklist": r.checklist,
            "approved": r.approved,
            "plateau": r.plateau,
            "wf_efficiency": r.walk_forward.efficiency,
            "monte_carlo": r.monte_carlo,
            "dd_bootstrap": r.dd_bootstrap,
            "dd_bootstrap_full": r.dd_bootstrap_full,
            "bootstrap": r.bootstrap,
        }
        store.con.execute(
            "INSERT INTO backtest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                f"{r.strategy_name}/{r.market}",
                json.dumps(r.params, default=str),
                r.split.train.start.date(),
                r.split.test.end.date(),
                json.dumps(metrics, default=str),
                dt.datetime.now(),
                _git_sha(),
            ],
        )
    return md_path


def make_plots_basic(
    equity: pd.Series, out_dir: Path, stem: str, title: str = ""
) -> dict[str, Path]:
    """Equity + drawdown de uma série; usado pelo relatório de carteira."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    eq = equity / equity.iloc[0]
    ax1.plot(eq.index, eq, label=title or "equity")
    ax1.set_ylabel("Equity (base 1)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    dd = drawdown_series(equity)
    ax2.fill_between(dd.index, dd, 0, color="crimson", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"{stem}_equity.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return {"equity": p}
