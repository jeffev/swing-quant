"""Entrypoints de linha de comando: `swing-quant <comando>`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from swing_quant import __version__
from swing_quant.backtest.metrics import Metrics
from swing_quant.config import DEFAULT_CONFIG_PATH, Config, load_config
from swing_quant.data.calendar import Market

if TYPE_CHECKING:
    from swing_quant.backtest.protocol import ProtocolResult

app = typer.Typer(
    name="swing-quant",
    help="Swing trade quantitativo — dados, backtest, screener e alertas.",
    no_args_is_help=True,
)
console = Console()

ConfigOpt = Annotated[
    Path,
    typer.Option("--config", "-c", help="Caminho do config.yaml", exists=True, dir_okay=False),
]


def _print_version(value: bool) -> None:
    if value:
        console.print(f"swing-quant {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Exibe a versão e sai",
            callback=_print_version,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Swing trade quantitativo — dados, backtest, screener e alertas."""


@app.command("show-config")
def show_config(config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Valida e exibe a configuração carregada."""
    cfg = load_config(config)
    table = Table(title=f"Configuração ({config})")
    table.add_column("Seção")
    table.add_column("Valor")
    table.add_row("capital", f"{cfg.capital.initial:,.0f} {cfg.capital.currency}")
    per_market = cfg.risk.risk_per_trade_by_market
    risk_txt = f"{cfg.risk.risk_per_trade:.2%}"
    if per_market:
        detail = ", ".join(f"{m}: {v:.2%}" for m, v in sorted(per_market.items()))
        risk_txt += f" (por mercado — {detail})"
    table.add_row("risco/trade", risk_txt)
    table.add_row("max posições", str(cfg.risk.max_positions))
    table.add_row("execução", cfg.data.execution)
    table.add_row("db", str(cfg.data.db_path))
    enabled = [k for k, v in cfg.strategies.items() if v.get("enabled")]
    table.add_row("estratégias ativas", ", ".join(enabled) or "-")
    console.print(table)


MarketOpt = Annotated[str, typer.Option("--market", "-m", help="b3 | us | all")]


def _markets(value: str) -> list[Market]:
    if value == "all":
        return ["b3", "us"]
    if value not in ("b3", "us"):
        console.print(f"[red]Mercado inválido:[/red] {value} (use b3, us ou all)")
        raise typer.Exit(code=1)
    return [value]  # type: ignore[list-item]


def _print_issues(issues: pd.DataFrame, limit: int = 30) -> None:
    from swing_quant.data.quality import summarize

    counts = summarize(issues)
    console.print(
        f"Qualidade: [red]{counts['critical']} critical[/red], "
        f"[yellow]{counts['warning']} warning[/yellow], {counts['info']} info"
    )
    if issues.empty:
        return
    shown = issues[issues["severity"] != "info"].head(limit)
    if shown.empty:
        return
    table = Table(title=f"Problemas (top {len(shown)})")
    for col in ("severity", "ticker", "date", "check", "detail"):
        table.add_column(col)
    for row in shown.to_dict("records"):
        d = "" if pd.isna(row["date"]) else str(pd.Timestamp(row["date"]).date())
        table.add_row(
            str(row["severity"]), str(row["ticker"]), d, str(row["check"]), str(row["detail"])
        )
    console.print(table)


@app.command("update-data")
def update_data(
    market: MarketOpt = "b3",
    full: Annotated[bool, typer.Option("--full", help="Rebaixa todo o histórico")] = False,
    skip_universe: Annotated[
        bool, typer.Option("--skip-universe", help="Não atualiza a composição do índice")
    ] = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Atualiza composição do índice e OHLCV do universo configurado."""
    import datetime as dt

    from swing_quant.data.loader import update_prices
    from swing_quant.data.quality import has_critical, run_checks
    from swing_quant.data.store import MarketStore
    from swing_quant.data.universe import INDEX_BY_MARKET, fetch_index, to_yf_symbol

    cfg = load_config(config)
    today = dt.date.today()
    exit_code = 0

    with MarketStore(cfg.data.db_path) as store:
        for mkt in _markets(market):
            index_name = INDEX_BY_MARKET[mkt]
            uni = cfg.market_universe(mkt)
            console.rule(f"[bold]{mkt.upper()} — {index_name}")

            # 1) universo
            if not skip_universe:
                try:
                    members = fetch_index(mkt)
                    n = store.upsert_universe(index_name, today, members)
                    console.print(f"Universo: {n} membros gravados (snapshot {today})")
                except Exception as exc:
                    console.print(
                        f"[yellow]Falha ao buscar universo ({exc}); usando snapshot.[/yellow]"
                    )
            members = store.universe_at(index_name)
            if members.empty:
                console.print("[red]Sem universo disponível; pulando mercado.[/red]")
                exit_code = 1
                continue

            # 2) preços
            tickers = [to_yf_symbol(t, mkt) for t in members["ticker"]] + [uni.benchmark]
            res = update_prices(store, tickers, cfg.data.history_start, full=full, as_of=today)
            console.print(
                f"Preços: {res.downloaded_rows} linhas, {res.tickers_updated} tickers atualizados, "
                f"{res.up_to_date} já em dia, {len(res.tickers_failed)} falhas"
            )
            if res.tickers_failed:
                console.print(f"[yellow]Falhas:[/yellow] {', '.join(res.tickers_failed[:20])}")

            # 3) qualidade
            prices = store.get_prices(tickers)
            issues = run_checks(prices, mkt, as_of=today)
            _print_issues(issues)
            if has_critical(issues):
                exit_code = 1

        console.print(
            f"Total no banco: {store.price_count():,} linhas / {len(store.tickers())} tickers"
        )
    raise typer.Exit(code=exit_code)


@app.command()
def quality(
    market: MarketOpt = "b3",
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Roda apenas a validação de qualidade sobre os dados já armazenados."""
    import datetime as dt

    from swing_quant.data.quality import has_critical, run_checks
    from swing_quant.data.store import MarketStore
    from swing_quant.data.universe import INDEX_BY_MARKET, to_yf_symbol

    cfg = load_config(config)
    exit_code = 0
    with MarketStore(cfg.data.db_path) as store:
        for mkt in _markets(market):
            members = store.universe_at(INDEX_BY_MARKET[mkt])
            tickers = [to_yf_symbol(t, mkt) for t in members["ticker"]]
            issues = run_checks(store.get_prices(tickers or None), mkt, as_of=dt.date.today())
            console.rule(f"[bold]{mkt.upper()}")
            _print_issues(issues)
            exit_code |= int(has_critical(issues))
    raise typer.Exit(code=exit_code)


@app.command("verify-cotahist")
def verify_cotahist(
    year: Annotated[int, typer.Option(help="Ano do arquivo COTAHIST_A{year}")],
    sample: Annotated[int, typer.Option(help="Nº de tickers amostrados")] = 20,
    dates: Annotated[int, typer.Option(help="Nº de datas por ticker")] = 10,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Compara fechamentos do store (B3) com o arquivo oficial COTAHIST da B3."""
    from swing_quant.data.cotahist import compare_with_store, download_cotahist
    from swing_quant.data.store import MarketStore
    from swing_quant.data.universe import INDEX_BY_MARKET, to_yf_symbol

    cfg = load_config(config)
    console.print(f"Baixando COTAHIST_A{year}…")
    cot = download_cotahist(year)
    console.print(f"{len(cot):,} registros (lote padrão, à vista)")

    with MarketStore(cfg.data.db_path) as store:
        members = store.universe_at(INDEX_BY_MARKET["b3"])
        if members.empty:
            console.print("[red]Universo B3 vazio; rode update-data primeiro.[/red]")
            raise typer.Exit(code=1)
        tickers = sorted(to_yf_symbol(t, "b3") for t in members["ticker"])
        step = max(1, len(tickers) // sample)
        chosen = tickers[::step][:sample]
        out = compare_with_store(store, cot, chosen, n_dates=dates)

    counts = out["status"].value_counts()
    table = Table(title=f"COTAHIST {year} vs store — {len(chosen)} tickers x {dates} datas")
    table.add_column("status")
    table.add_column("n", justify="right")
    for status, n in counts.items():
        table.add_row(str(status), str(n))
    console.print(table)

    bad = out[out["status"] == "mismatch"]
    if not bad.empty:
        t2 = Table(title="Divergências")
        for col in ("ticker", "date", "store_close", "b3_close", "rel_diff"):
            t2.add_column(col)
        for row in bad.head(30).to_dict("records"):
            t2.add_row(
                str(row["ticker"]),
                str(pd.Timestamp(row["date"]).date()),
                f"{row['store_close']:.2f}",
                f"{row['b3_close']:.2f}",
                f"{row['rel_diff']:.2%}",
            )
        console.print(t2)
    ok_ratio = counts.get("ok", 0) / max(len(out), 1)
    console.print(f"Aderência exata: {ok_ratio:.1%}")
    raise typer.Exit(code=0 if bad.empty else 1)


@app.command("update-riskfree")
def update_riskfree(
    market: MarketOpt = "all",
    start: Annotated[str, typer.Option(help="Data inicial (YYYY-MM-DD)")] = "2010-01-01",
    end: Annotated[str | None, typer.Option(help="Data final (YYYY-MM-DD)")] = None,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Baixa a renda fixa de referência de cada mercado: CDI (BCB) e T-bills dos EUA (BIL).

    Só alimenta as comparações (dashboard e relatórios) — o backtest continua sem remunerar o
    caixa das carteiras, então nenhum resultado histórico muda por causa deste comando.
    """
    import datetime as dt

    from swing_quant.data.riskfree import RISK_FREE_LABEL, update_risk_free
    from swing_quant.data.store import MarketStore

    cfg = load_config(config)
    ini = dt.date.fromisoformat(start)
    fim = dt.date.fromisoformat(end) if end else None
    with MarketStore(cfg.data.db_path) as store:
        for mkt in _markets(market):
            try:
                n = update_risk_free(store, mkt, ini, fim)
            except Exception as exc:
                console.print(f"[red]{RISK_FREE_LABEL[mkt]}: falhou[/red] — {exc}")
                raise typer.Exit(code=1) from exc
            console.print(f"{RISK_FREE_LABEL[mkt]} ({mkt}): {n} dias gravados")


@app.command()
def backtest(
    strategy: Annotated[str, typer.Option(help="Nome da estratégia (ex.: rsi2)")],
    market: Annotated[str, typer.Option("--market", "-m", help="b3 | us")] = "b3",
    start: Annotated[str | None, typer.Option(help="Data inicial (YYYY-MM-DD)")] = None,
    end: Annotated[str | None, typer.Option(help="Data final (YYYY-MM-DD)")] = None,
    cross: Annotated[
        bool, typer.Option("--cross/--no-cross", help="Valida no outro mercado")
    ] = True,
    quick: Annotated[bool, typer.Option("--quick", help="Menos simulações MC/bootstrap")] = False,
    out: Annotated[Path, typer.Option(help="Pasta de relatórios")] = Path("reports"),
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Roda o protocolo completo de validação (docs/04) e gera relatório em Markdown."""
    cfg = load_config(config)
    mkts = _markets(market)
    if len(mkts) != 1:
        console.print("[red]Escolha um único mercado para o backtest.[/red]")
        raise typer.Exit(code=1)
    mkt = mkts[0]
    result, path = _protocol_run(cfg, strategy, mkt, start, end, cross, quick, out)

    m = result.metrics_test
    table = Table(title=f"{strategy}/{mkt} — teste OOS ({result.split.test})")
    table.add_column("Métrica")
    table.add_column("Valor", justify="right")
    for k, val in (
        ("Sharpe", f"{m.sharpe:.2f}"),
        ("CAGR", f"{m.cagr:.1%}"),
        ("Max DD", f"{m.max_drawdown:.1%}"),
        ("Profit factor", f"{m.profit_factor:.2f}"),
        ("Trades", str(m.n_trades)),
        ("Exposição média", f"{m.exposure_avg:.1%}"),
        ("Permanência média (pregões)", f"{m.avg_hold_bars:.1f}"),
        ("Platô", f"{result.plateau:.2f}"),
        ("Eficiência WF", f"{result.walk_forward.efficiency:.2f}"),
        ("DD p95 em 1 ano", f"{result.dd_bootstrap.get('mdd_p95', float('nan')):.1%}"),
        (
            "DD p95 horizonte completo",
            f"{result.dd_bootstrap_full.get('mdd_p95', float('nan')):.1%}",
        ),
    ):
        table.add_row(k, val)
    console.print(table)
    for k, ok in result.checklist.items():
        console.print(f"  {'[green]OK[/green]' if ok else '[red]X [/red]'} {k}")
    verdict = "[green]APROVADA[/green]" if result.approved else "[red]REPROVADA[/red]"
    console.print(f"\nVeredito: {verdict} — relatório em {path}")
    raise typer.Exit(code=0 if result.approved else 3)


@app.command()
def bench(
    strategies: Annotated[
        str, typer.Option("--strategies", "-s", help="Lista separada por vírgula, ou 'all'")
    ] = "all",
    market: MarketOpt = "all",
    start: Annotated[str | None, typer.Option(help="Data inicial (YYYY-MM-DD)")] = None,
    end: Annotated[str | None, typer.Option(help="Data final (YYYY-MM-DD)")] = None,
    cross: Annotated[
        bool, typer.Option("--cross/--no-cross", help="Valida no outro mercado")
    ] = False,
    quick: Annotated[
        bool, typer.Option("--quick/--no-quick", help="Menos simulações MC/bootstrap")
    ] = True,
    out: Annotated[Path, typer.Option(help="Pasta de relatórios")] = Path("reports"),
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Roda o protocolo para várias estratégias/mercados de uma vez e compara os resultados.

    Cada combinação gera o mesmo run e o mesmo relatório que o comando `backtest` — a diferença
    é a fila e a tabela comparativa no fim. Como demora, o padrão aqui é `--quick` (menos
    simulações de Monte Carlo/bootstrap); para a decisão final, rode com `--no-quick`.
    O detalhe por ação fica no dashboard, página **Backtests**.
    """
    cfg = load_config(config)
    names = (
        [n for n, v in cfg.strategies.items() if v.get("enabled")]
        if strategies == "all"
        else [s.strip() for s in strategies.split(",") if s.strip()]
    )
    unknown = [n for n in names if n not in cfg.strategies]
    if unknown:
        console.print(f"[red]Estratégia não configurada em config.yaml:[/red] {', '.join(unknown)}")
        raise typer.Exit(code=1)
    if not names:
        console.print("[red]Nenhuma estratégia habilitada — use --strategies.[/red]")
        raise typer.Exit(code=1)

    combos = [(n, m) for m in _markets(market) for n in names]
    console.print(f"Fila: {len(combos)} backtest(s) — {', '.join(f'{n}/{m}' for n, m in combos)}")
    rows: list[tuple[str, ProtocolResult | None, str]] = []
    for i, (name, mkt) in enumerate(combos, start=1):
        console.rule(f"[{i}/{len(combos)}] {name}/{mkt}")
        try:
            result, _ = _protocol_run(cfg, name, mkt, start, end, cross, quick, out)
            rows.append((f"{name}/{mkt}", result, ""))
        except Exception as exc:  # uma combinação sem dados não pode derrubar a fila inteira
            console.print(f"[red]Falhou:[/red] {exc}")
            rows.append((f"{name}/{mkt}", None, str(exc)[:60]))

    table = Table(title="Comparação — teste OOS")
    # sem largura mínima o rich corta justamente o que identifica a linha ("momentum…")
    table.add_column("Estratégia", min_width=max(len(label) for label, _, _ in rows))
    for col in ("Veredito", "Sharpe", "CAGR", "Max DD", "DD p95 1a", "Trades", "Perm."):
        table.add_column(col, justify="right")
    for label, done, err in sorted(
        rows, key=lambda r: -(r[1].metrics_test.sharpe if r[1] is not None else float("-inf"))
    ):
        if done is None:
            table.add_row(label, f"[red]erro[/red] {err}", "-", "-", "-", "-", "-", "-")
            continue
        m = done.metrics_test
        table.add_row(
            label,
            "[green]APROVADA[/green]" if done.approved else "[red]REPROVADA[/red]",
            f"{m.sharpe:.2f}",
            f"{m.cagr:.1%}",
            f"{m.max_drawdown:.1%}",
            f"{done.dd_bootstrap.get('mdd_p95', float('nan')):.1%}",
            str(m.n_trades),
            f"{m.avg_hold_bars:.0f}",
        )
    console.print(table)
    if quick or not cross:
        atalhos = [
            x
            for x in (
                "menos simulações (--quick)" if quick else "",
                "sem validação cruzada (--no-cross)" if not cross else "",
            )
            if x
        ]
        console.print(
            f"[yellow]Triagem[/yellow]: rodada com {' e '.join(atalhos)} — o veredito aqui é "
            "indicativo. Antes de adotar, refaça com `backtest --no-quick --cross`."
        )
    console.print("\nResultado por ação: `swing-quant dashboard` → página [bold]Backtests[/bold].")


def _protocol_run(
    cfg: Config,
    strategy: str,
    mkt: Market,
    start: str | None,
    end: str | None,
    cross: bool,
    quick: bool,
    out: Path,
) -> tuple[ProtocolResult, Path]:
    """Protocolo completo (docs/04) de uma estratégia num mercado: roda, salva e registra.

    Compartilhado por `backtest` (uma combinação) e `bench` (uma fila delas), para que os dois
    produzam exatamente o mesmo run — custos, risco e painel montados do mesmo jeito.
    """
    from swing_quant.backtest.engine import CostModel, RiskModel
    from swing_quant.backtest.protocol import run_protocol
    from swing_quant.backtest.report import save_report
    from swing_quant.backtest.validation import default_panel_factory
    from swing_quant.data.store import MarketStore
    from swing_quant.data.universe import INDEX_BY_MARKET, to_yf_symbol
    from swing_quant.strategies import make_strategy

    if strategy not in cfg.strategies:
        console.print(f"[red]Estratégia não configurada em config.yaml:[/red] {strategy}")
        raise typer.Exit(code=1)
    other: Market = "us" if mkt == "b3" else "b3"
    strat = make_strategy(strategy, cfg.strategies[strategy])

    def costs_for(m: Market) -> CostModel:
        c = cfg.market_costs(m)
        return CostModel(c.commission_per_order, c.fees_pct, c.slippage_pct_liquid)

    risk = RiskModel(
        initial_capital=cfg.capital.initial,
        risk_per_trade=cfg.risk.risk_for_market(mkt),
        atr_multiple=cfg.risk.atr_multiple_default,
        max_position_pct=cfg.risk.max_position_pct,
        max_positions=cfg.risk.max_positions,
        max_volume_participation=cfg.risk.max_volume_participation,
        board_lot=cfg.risk.board_lot if mkt == "b3" else 1,
        min_dollar_volume=cfg.market_universe(mkt).min_avg_dollar_volume_20d,
    )

    with MarketStore(cfg.data.db_path) as store:

        def load(m: Market) -> pd.DataFrame:
            members = store.universe_at(INDEX_BY_MARKET[m])
            tickers = [to_yf_symbol(t, m) for t in members["ticker"]]
            return store.get_prices(tickers, start=start, end=end)

        console.print(f"Carregando preços {mkt.upper()}…")
        prices = load(mkt)
        bench_prices = store.get_prices([cfg.market_universe(mkt).benchmark], start=start, end=end)
        bench_close = (
            bench_prices.set_index("date")["adj_close"] if not bench_prices.empty else None
        )
        cross_factory = None
        if cross:
            console.print(f"Carregando preços {other.upper()} (mercado cruzado)…")
            cross_factory = default_panel_factory(load(other))

        console.print(
            f"Rodando protocolo para {strat!r} em {len(prices['ticker'].unique())} tickers…"
        )
        v = cfg.validation
        result = run_protocol(
            strat,
            default_panel_factory(prices),
            market=mkt,
            costs=costs_for(mkt),
            risk=risk,
            split_fractions=(v.split[0], v.split[1], v.split[2]),
            train_years=v.walkforward.train_years,
            test_years=v.walkforward.test_years,
            anchored=v.walkforward.anchored,
            cost_multipliers=tuple(v.cost_multipliers),
            mc_runs=200 if quick else v.monte_carlo_runs,
            boot_runs=200 if quick else 1000,
            baseline_runs=10 if quick else 30,
            benchmark_close=bench_close,
            cross_panel_factory=cross_factory,
            cross_costs=costs_for(other),
            min_trades_select=v.min_test_trades,
        )
        path = save_report(result, out, store)
    return result, path


@app.command()
def portfolio(
    market: Annotated[str, typer.Option("--market", "-m", help="b3 | us")] = "b3",
    strategies: Annotated[
        str | None, typer.Option(help="Lista separada por vírgula (padrão: enabled no config)")
    ] = None,
    start: Annotated[str | None, typer.Option(help="Data inicial (YYYY-MM-DD)")] = None,
    end: Annotated[str | None, typer.Option(help="Data final (YYYY-MM-DD)")] = None,
    regime: Annotated[bool, typer.Option("--regime/--no-regime", help="Filtro de regime")] = True,
    out: Annotated[Path, typer.Option(help="Pasta de relatórios")] = Path("reports"),
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Backtest da carteira combinada: estratégias + regime + regras de risco (Fase 3)."""
    from swing_quant.backtest.engine import CostModel, RiskModel
    from swing_quant.backtest.portfolio import combine_panels, run_portfolio, save_portfolio_report
    from swing_quant.backtest.validation import default_panel_factory
    from swing_quant.data.store import MarketStore
    from swing_quant.data.universe import INDEX_BY_MARKET, to_yf_symbol
    from swing_quant.risk.regime import RegimeConfig, build_regime
    from swing_quant.strategies import make_strategy

    cfg = load_config(config)
    mkts = _markets(market)
    if len(mkts) != 1:
        console.print("[red]Escolha um único mercado.[/red]")
        raise typer.Exit(code=1)
    mkt = mkts[0]
    names = (
        [s.strip() for s in strategies.split(",") if s.strip()]
        if strategies
        else cfg.enabled_strategies(mkt)
    )
    if not names:
        console.print(f"[red]Nenhuma estratégia habilitada para {mkt}.[/red]")
        raise typer.Exit(code=1)
    c = cfg.market_costs(mkt)
    costs = CostModel(c.commission_per_order, c.fees_pct, c.slippage_pct_liquid)
    rk = cfg.risk
    risk = RiskModel(
        initial_capital=cfg.capital.for_market(mkt),
        risk_per_trade=rk.risk_for_market(mkt),
        atr_multiple=rk.atr_multiple_default,
        max_position_pct=rk.max_position_pct,
        max_positions=rk.max_positions,
        max_volume_participation=rk.max_volume_participation,
        board_lot=rk.board_lot if mkt == "b3" else 1,
        min_dollar_volume=cfg.market_universe(mkt).min_avg_dollar_volume_20d,
        max_sector_pct=rk.max_sector_pct,
        max_strategy_pct=rk.max_strategy_pct,
        max_correlation=rk.max_correlation,
        monthly_dd_reduce=rk.monthly_dd_reduce,
        circuit_breaker_dd=rk.circuit_breaker_dd,
    )

    with MarketStore(cfg.data.db_path) as store:
        members = store.universe_at(INDEX_BY_MARKET[mkt])
        tickers = [to_yf_symbol(t, mkt) for t in members["ticker"]]
        sectors = {
            to_yf_symbol(t, mkt): str(s)
            for t, s in zip(members["ticker"], members["sector"], strict=True)
            if pd.notna(s)
        }
        console.print(f"Carregando preços {mkt.upper()} ({len(tickers)} tickers)…")
        prices = store.get_prices(tickers, start=start, end=end)
        bench = store.get_prices([cfg.market_universe(mkt).benchmark], start=start, end=end)
        bench_close = bench.set_index("date")["adj_close"] if not bench.empty else None

    factory = default_panel_factory(prices)
    panels = {n: factory(make_strategy(n, cfg.strategies.get(n, {}))) for n in names}
    combined = combine_panels(panels, sectors)
    reg = None
    if regime and bench_close is not None:
        reg = build_regime(
            bench_close,
            RegimeConfig(
                trend_sma=cfg.regime.benchmark_sma,
                vol_percentile=cfg.regime.high_vol_percentile,
                high_vol_size_factor=cfg.regime.high_vol_size_factor,
                use_trend=cfg.regime.trend_filter,
                use_vol=cfg.regime.vol_filter,
            ),
        )
    console.print(f"Rodando carteira {names} com {len(combined.tickers)} colunas…")
    result = run_portfolio(
        combined,
        market=mkt,
        costs=costs,
        risk=risk,
        regime=reg,
        benchmark_close=bench_close,
        split_fractions=(cfg.validation.split[0], cfg.validation.split[1], cfg.validation.split[2]),
    )
    path = save_portfolio_report(result, out)

    table = Table(title=f"Carteira {mkt.upper()} — {', '.join(names)}")
    table.add_column("Métrica")
    table.add_column("Completo", justify="right")
    table.add_column("Teste", justify="right")
    full_fmt = _fmt_metrics(result.metrics_full)
    test_fmt = _fmt_metrics(result.metrics_test)
    for label in full_fmt:
        table.add_row(label, full_fmt[label], test_fmt[label])
    console.print(table)
    console.print(_df_to_table(result.with_vs_without, "Ablação"))
    console.print(f"Relatório: {path}")


def _fmt_metrics(m: Metrics) -> dict[str, str]:
    return {
        "CAGR": f"{m.cagr:.1%}",
        "Sharpe": f"{m.sharpe:.2f}",
        "Max DD": f"{m.max_drawdown:.1%}",
        "Profit factor": f"{m.profit_factor:.2f}",
        "Trades": str(m.n_trades),
        "Exposição": f"{m.exposure_avg:.1%}",
    }


def _df_to_table(df: pd.DataFrame, title: str) -> Table:
    t = Table(title=title)
    for col in df.columns:
        t.add_column(str(col))
    for row in df.to_dict("records"):
        t.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()])
    return t


def _risk_from_config(cfg: object, mkt: Market) -> object:
    from swing_quant.backtest.engine import RiskModel
    from swing_quant.config import Config

    assert isinstance(cfg, Config)
    rk = cfg.risk
    return RiskModel(
        initial_capital=cfg.capital.for_market(mkt),
        risk_per_trade=rk.risk_for_market(mkt),
        atr_multiple=rk.atr_multiple_default,
        max_position_pct=rk.max_position_pct,
        max_positions=rk.max_positions,
        max_volume_participation=rk.max_volume_participation,
        board_lot=rk.board_lot if mkt == "b3" else 1,
        min_dollar_volume=cfg.market_universe(mkt).min_avg_dollar_volume_20d,
        max_sector_pct=rk.max_sector_pct,
        max_strategy_pct=rk.max_strategy_pct,
        max_correlation=rk.max_correlation,
        monthly_dd_reduce=rk.monthly_dd_reduce,
        circuit_breaker_dd=rk.circuit_breaker_dd,
    )


@app.command()
def screen(
    market: MarketOpt = "b3",
    as_of: Annotated[str | None, typer.Option(help="Data de referência (padrão: última)")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Não grava nem envia")] = False,
    no_alert: Annotated[bool, typer.Option("--no-alert", help="Grava mas não envia")] = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Screener diário: saídas das posições abertas + novas entradas para a próxima abertura."""
    import datetime as dt

    from swing_quant.alerts.telegram import format_message, send
    from swing_quant.backtest.engine import CostModel, RiskModel
    from swing_quant.data.store import MarketStore
    from swing_quant.data.universe import INDEX_BY_MARKET, to_yf_symbol
    from swing_quant.journal.core import Journal
    from swing_quant.risk.regime import RegimeConfig, build_regime
    from swing_quant.screener.core import run_screener
    from swing_quant.strategies import make_strategy

    cfg = load_config(config)
    exit_code = 0
    for mkt in _markets(market):
        names = cfg.enabled_strategies(mkt)
        if not names:
            console.print(f"[red]Nenhuma estratégia habilitada para {mkt} em config.yaml.[/red]")
            raise typer.Exit(code=1)
        with MarketStore(cfg.data.db_path) as _store:
            from swing_quant.monitoring.health import HealthStore

            paused = HealthStore(_store).paused(mkt)
        if paused:
            console.print(
                f"[yellow]Estratégias pausadas pela regra de saúde:[/yellow] {sorted(paused)}"
            )
            names = [n for n in names if n not in paused]
            if not names:
                console.print("[red]Todas as estratégias estão pausadas; nada a fazer.[/red]")
                continue
        c = cfg.market_costs(mkt)
        costs = CostModel(c.commission_per_order, c.fees_pct, c.slippage_pct_liquid)
        risk = _risk_from_config(cfg, mkt)
        assert isinstance(risk, RiskModel)

        with MarketStore(cfg.data.db_path) as store:
            journal = Journal(store)
            members = store.universe_at(INDEX_BY_MARKET[mkt])
            tickers = [to_yf_symbol(t, mkt) for t in members["ticker"]]
            sectors = {
                to_yf_symbol(t, mkt): str(s)
                for t, s in zip(members["ticker"], members["sector"], strict=True)
                if pd.notna(s)
            }
            ref = pd.Timestamp(as_of) if as_of else None
            end = ref.date() if ref is not None else None
            start = (ref or pd.Timestamp.today()) - pd.Timedelta(days=900)
            prices = store.get_prices(tickers, start=start.date(), end=end)
            bench = store.get_prices(
                [cfg.market_universe(mkt).benchmark], start=start.date(), end=end
            )
            regime = None
            if not bench.empty:
                regime = build_regime(
                    bench.set_index("date")["adj_close"],
                    RegimeConfig(
                        trend_sma=cfg.regime.benchmark_sma,
                        vol_percentile=cfg.regime.high_vol_percentile,
                        high_vol_size_factor=cfg.regime.high_vol_size_factor,
                        use_trend=cfg.regime.trend_filter,
                        use_vol=cfg.regime.vol_filter,
                    ),
                )
            positions = journal.open_positions(mkt)
            equity = journal.equity_estimate(cfg.capital.for_market(mkt), mkt)
            cash = equity - journal.invested_at_cost(mkt)
            result = run_screener(
                prices,
                {n: make_strategy(n, cfg.strategies[n]) for n in names},
                market=mkt,
                risk=risk,
                costs=costs,
                equity=equity,
                cash=cash,
                positions=positions,
                regime=regime,
                sectors=sectors,
                as_of=ref,
            )
            last_day = prices["date"].max().date() if not prices.empty else None
            stale = as_of is None and last_day is not None and (dt.date.today() - last_day).days > 5
            if stale:
                result.notes.append(f"⚠️ dados defasados: último pregão no banco = {last_day}")
                exit_code = 1
            text = format_message(
                result, top_n=cfg.alerts.telegram.top_n, currency="R$" if mkt == "b3" else "US$"
            )
            console.rule(f"[bold]{mkt.upper()} — {result.as_of.date()}")
            console.print(text)
            if not dry_run:
                ids = journal.record_screen(result)
                console.print(f"[green]{len(ids)} sinais gravados no journal.[/green]")
                if not no_alert and cfg.alerts.telegram.enabled:
                    sent = send(text)
                    console.print("Telegram: " + ("enviado" if sent else "sem credenciais"))
    raise typer.Exit(code=exit_code)


@app.command("record-execution")
def record_execution(
    signal_id: Annotated[int, typer.Option(help="id do sinal (ver `positions --signals`)")],
    price: Annotated[float, typer.Option(help="Preço executado")],
    qty: Annotated[int, typer.Option(help="Quantidade executada")],
    side: Annotated[str, typer.Option(help="buy | sell")] = "buy",
    fees: Annotated[float, typer.Option(help="Custos totais da ordem")] = 0.0,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Registra uma execução real ligada a um sinal do journal."""
    from swing_quant.data.store import MarketStore
    from swing_quant.journal.core import ExecutionRecord, Journal

    cfg = load_config(config)
    with MarketStore(cfg.data.db_path) as store:
        Journal(store).record_execution(ExecutionRecord(signal_id, side, price, qty, fees))
    console.print(f"[green]Execução registrada:[/green] sinal {signal_id} {side} {qty} @ {price}")


@app.command()
def positions(
    signals: Annotated[bool, typer.Option("--signals", help="Lista sinais recentes")] = False,
    days: Annotated[int, typer.Option(help="Janela de sinais (dias)")] = 7,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Posições abertas (e, opcionalmente, sinais recentes) do journal."""
    import datetime as dt

    from swing_quant.data.store import MarketStore
    from swing_quant.journal.core import Journal

    cfg = load_config(config)
    with MarketStore(cfg.data.db_path) as store:
        j = Journal(store)
        pos = j.open_positions()
        table = Table(title=f"Posições abertas ({len(pos)})")
        for col in ("sinal", "ticker", "estratégia", "qty", "entrada", "preço", "stop", "max_hold"):
            table.add_column(col)
        for p in pos:
            table.add_row(
                str(p.signal_id),
                p.ticker,
                p.strategy,
                str(p.qty),
                str(p.entry_date),
                f"{p.entry_price:.2f}",
                "-" if p.stop_price is None else f"{p.stop_price:.2f}",
                str(p.max_hold),
            )
        console.print(table)
        console.print(
            f"P&L realizado: {j.realized_pnl():,.2f} · patrimônio estimado: "
            f"{j.equity_estimate(cfg.capital.initial):,.2f}"
        )
        if signals:
            df = j.signals()
            if not df.empty:
                df = df[
                    pd.to_datetime(df["as_of"])
                    >= pd.Timestamp(dt.date.today() - dt.timedelta(days=days))
                ]
            cols = [
                "id",
                "as_of",
                "market",
                "side",
                "ticker",
                "strategy",
                "ref_price",
                "qty",
                "stop_price",
            ]
            console.print(_df_to_table(df[cols] if not df.empty else df, f"Sinais ({days}d)"))


@app.command()
def health(
    market: MarketOpt = "b3",
    as_of: Annotated[str | None, typer.Option(help="Data de referência (padrão: hoje)")] = None,
    resume: Annotated[str | None, typer.Option(help="Reativa a estratégia informada")] = None,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Saúde por estratégia (Sharpe 6m, drawdown vs DD p95 de 1 ano); 2 alertas seguidos → pausa."""
    import datetime as dt

    from swing_quant.data.store import MarketStore
    from swing_quant.data.universe import INDEX_BY_MARKET, to_yf_symbol
    from swing_quant.journal.core import Journal
    from swing_quant.monitoring.health import HealthStore, run_health
    from swing_quant.monitoring.performance import (
        mark_to_market,
        strategy_daily_returns,
        trade_ledger,
    )

    cfg = load_config(config)
    day = pd.Timestamp(as_of).date() if as_of else dt.date.today()
    for mkt in _markets(market):
        with MarketStore(cfg.data.db_path) as store:
            hs = HealthStore(store)
            if resume:
                hs.resume(resume, mkt, day)
                console.print(f"[green]{resume} reativada em {mkt}.[/green]")
                continue
            j = Journal(store)
            members = store.universe_at(INDEX_BY_MARKET[mkt])
            tickers = [to_yf_symbol(t, mkt) for t in members["ticker"]]
            ex = j.executions()
            ex = ex[ex["market"] == mkt] if not ex.empty else ex
            start = (
                (pd.to_datetime(ex["executed_at"]).min() - pd.Timedelta(days=5)).date()
                if not ex.empty
                else day - dt.timedelta(days=30)
            )
            prices = store.get_prices(tickers, start=start, end=day)
            rets: dict[str, pd.Series] = {}
            if not prices.empty:
                mtm = mark_to_market(ex, prices, cfg.capital.for_market(mkt))
                rets = strategy_daily_returns(
                    trade_ledger(j.signals(market=mkt), ex), mtm["equity"]
                )
            names = cfg.enabled_strategies(mkt)
            reports = run_health(store, mkt, rets, names, day)
        table = Table(title=f"Saúde {mkt.upper()} — {day}")
        for col in ("estratégia", "status", "motivo", "Sharpe 6m", "DD", "DD p95", "alertas"):
            table.add_column(col)
        for r in reports:
            table.add_row(
                r.strategy,
                r.status,
                r.reason,
                f"{r.sharpe_6m:.2f}",
                f"{r.drawdown:.1%}",
                f"{r.expected.dd_p95:.1%}",
                str(r.consecutive_alerts),
            )
        console.print(table)


@app.command("monthly-report")
def monthly_report(
    market: MarketOpt = "b3",
    month: Annotated[str | None, typer.Option(help="YYYY-MM (padrão: mês anterior)")] = None,
    out: Annotated[Path, typer.Option(help="Pasta de relatórios")] = Path("reports"),
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Relatório mensal: realizado × esperado, aderência, slippage, saúde."""
    import datetime as dt

    from swing_quant.data.store import MarketStore
    from swing_quant.data.universe import INDEX_BY_MARKET, to_yf_symbol
    from swing_quant.journal.core import Journal
    from swing_quant.monitoring.health import HealthReport, HealthStore, latest_expected
    from swing_quant.monitoring.monthly import build_monthly, save_monthly
    from swing_quant.monitoring.performance import mark_to_market

    cfg = load_config(config)
    if month is None:
        first = dt.date.today().replace(day=1)
        month = (first - dt.timedelta(days=1)).strftime("%Y-%m")
    for mkt in _markets(market):
        with MarketStore(cfg.data.db_path) as store:
            j = Journal(store)
            hs = HealthStore(store)
            members = store.universe_at(INDEX_BY_MARKET[mkt])
            tickers = [to_yf_symbol(t, mkt) for t in members["ticker"]]
            end = pd.Timestamp(month + "-01") + pd.offsets.MonthEnd(1)
            ex = j.executions()
            ex = ex[ex["market"] == mkt] if not ex.empty else ex
            start = (
                pd.to_datetime(ex["executed_at"]).min()
                if not ex.empty
                else pd.Timestamp(month + "-01")
            )
            prices = store.get_prices(tickers, start=start.date(), end=end.date())
            if prices.empty:
                console.print(f"[red]Sem preços para {mkt} em {month}.[/red]")
                continue
            mtm = mark_to_market(ex, prices, cfg.capital.for_market(mkt))
            hist = hs.history(mkt)
            health: list[HealthReport] = []
            for name in cfg.enabled_strategies(mkt):
                row = hist[hist["strategy"] == name].head(1)
                exp = latest_expected(store, name, mkt)
                if row.empty:
                    health.append(
                        HealthReport(
                            name,
                            mkt,
                            end.date(),
                            "active",
                            "sem avaliação",
                            float("nan"),
                            float("nan"),
                            0,
                            exp,
                            0,
                        )
                    )
                else:
                    r0 = row.iloc[0]
                    health.append(
                        HealthReport(
                            name,
                            mkt,
                            r0["as_of"],
                            str(r0["status"]),
                            str(r0["reason"]),
                            float(r0["sharpe_6m"]) if pd.notna(r0["sharpe_6m"]) else float("nan"),
                            float(r0["drawdown"]) if pd.notna(r0["drawdown"]) else float("nan"),
                            int(r0["consecutive_alerts"] or 0),
                            exp,
                            0,
                        )
                    )
            rep = build_monthly(mkt, month, mtm, j.signals(market=mkt), ex, health)
        path = save_monthly(rep, out)
        console.print(f"Relatório mensal {mkt.upper()} {month}: {path}")


@app.command()
def dashboard() -> None:
    """Abre o dashboard Streamlit (requer `uv sync --extra dashboard`)."""
    import subprocess
    import sys

    app_path = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
    raise typer.Exit(
        code=subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])
    )


@app.command("notify-failure")
def notify_failure(
    step: Annotated[str, typer.Option(help="Etapa que falhou")],
    market: Annotated[str, typer.Option(help="Mercado")] = "",
    error: Annotated[str, typer.Option(help="Mensagem de erro")] = "ver logs do workflow",
) -> None:
    """Envia alerta de falha do pipeline (usado pelo GitHub Actions)."""
    from swing_quant.alerts.telegram import format_failure, send

    sent = send(format_failure(step, error, market))
    console.print("Alerta de falha: " + ("enviado" if sent else "sem credenciais Telegram"))


if __name__ == "__main__":
    app()
