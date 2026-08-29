"""Leitura do COTAHIST (série histórica oficial da B3) e comparação com o store.

Layout de registro tipo 01 (posições 1-based, largura fixa 245):
  03-10 DATA  13-24 CODNEG  11-12 CODBDI  25-27 TPMERC
  57-69 PREABE  70-82 PREMAX  83-95 PREMIN  109-121 PREULT  153-170 QUATOT  171-188 VOLTOT
Preços têm 2 casas decimais implícitas.

Uso: verificação de integridade amostral dos dados do yfinance (docs/06 Fase 1).
Atenção: o COTAHIST é *bruto* (sem ajuste por splits); o `close` do yfinance é ajustado por
splits mas não por proventos. Divergências grandes em datas anteriores a um desdobramento são
esperadas e o relatório as marca como `adjusted_split_or_bonus` (razão constante por ticker).
"""

from __future__ import annotations

import io
import zipfile

import httpx
import pandas as pd

from swing_quant.data.store import MarketStore

COTAHIST_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"
_HEADERS = {"User-Agent": "Mozilla/5.0 (swing-quant)"}

_FIELDS: dict[str, tuple[int, int]] = {  # nome -> (início 0-based, fim exclusivo)
    "tipreg": (0, 2),
    "date": (2, 10),
    "codbdi": (10, 12),
    "ticker": (12, 24),
    "tpmerc": (24, 27),
    "open": (56, 69),
    "high": (69, 82),
    "low": (82, 95),
    "close": (108, 121),
    "quantity": (152, 170),
    "volume_fin": (170, 188),
}
_PRICE_FIELDS = ("open", "high", "low", "close")


def parse_cotahist(text: str, only_standard_lot: bool = True) -> pd.DataFrame:
    """Converte o conteúdo texto do COTAHIST em DataFrame(ticker, date, open, high, low, close,
    quantity, volume_fin). Mantém apenas mercado à vista (TPMERC=010) e, por padrão,
    lote padrão (CODBDI=02)."""
    lines = [ln for ln in text.splitlines() if ln.startswith("01")]
    if not lines:
        return pd.DataFrame(columns=["ticker", "date", *_PRICE_FIELDS, "quantity", "volume_fin"])
    raw = pd.DataFrame({name: [ln[a:b] for ln in lines] for name, (a, b) in _FIELDS.items()})
    mask = raw["tpmerc"] == "010"
    if only_standard_lot:
        mask &= raw["codbdi"] == "02"
    raw = raw[mask]
    out = pd.DataFrame(
        {
            "ticker": raw["ticker"].str.strip(),
            "date": pd.to_datetime(raw["date"], format="%Y%m%d"),
        }
    )
    for f in _PRICE_FIELDS:
        out[f] = pd.to_numeric(raw[f], errors="coerce") / 100.0
    out["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce").astype("int64")
    out["volume_fin"] = pd.to_numeric(raw["volume_fin"], errors="coerce") / 100.0
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def download_cotahist(year: int, timeout: float = 120.0) -> pd.DataFrame:
    """Baixa e parseia o arquivo anual COTAHIST_A{year}.ZIP da B3."""
    url = COTAHIST_URL.format(year=year)
    with httpx.Client(headers=_HEADERS, timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.upper().endswith(".TXT"))
        text = zf.read(name).decode("latin-1")
    return parse_cotahist(text)


def compare_with_store(
    store: MarketStore,
    cotahist: pd.DataFrame,
    tickers: list[str],
    *,
    n_dates: int = 10,
    tolerance: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    """Compara `close` do store (símbolos yfinance, ex.: PETR4.SA) com PREULT do COTAHIST.

    Retorna DataFrame(ticker, date, store_close, b3_close, rel_diff, status) onde status é
    `ok`, `mismatch`, `adjusted_split_or_bonus`, `missing_in_store` ou `missing_in_cotahist`.
    """
    rows: list[dict[str, object]] = []
    b3 = cotahist.set_index(["ticker", "date"])["close"]
    for yf_t in tickers:
        local = yf_t.removesuffix(".SA")
        sub = cotahist[cotahist["ticker"] == local]
        if sub.empty:
            rows.append(_row(yf_t, pd.NaT, None, None, "missing_in_cotahist"))
            continue
        dates = sub["date"].sample(n=min(n_dates, len(sub)), random_state=seed)
        stored = store.get_prices([yf_t], start=dates.min(), end=dates.max())
        stored_close = (
            stored.set_index("date")["close"] if not stored.empty else pd.Series(dtype=float)
        )
        ticker_rows: list[dict[str, object]] = []
        for d in sorted(dates):
            b3_close = float(b3.loc[(local, d)])
            if d not in stored_close.index:
                ticker_rows.append(_row(yf_t, d, None, b3_close, "missing_in_store"))
                continue
            sc = float(stored_close.loc[d])
            rel = abs(sc - b3_close) / b3_close if b3_close else float("nan")
            status = "ok" if rel <= tolerance else "mismatch"
            ticker_rows.append(_row(yf_t, d, sc, b3_close, status, rel))
        _classify_adjustments(ticker_rows, tolerance)
        rows.extend(ticker_rows)
    return pd.DataFrame(
        rows, columns=["ticker", "date", "store_close", "b3_close", "rel_diff", "status"]
    )


def _classify_adjustments(rows: list[dict[str, object]], tolerance: float) -> None:
    """Reclassifica `mismatch` como `adjusted_split_or_bonus` quando a razão store/B3 é
    constante entre as datas divergentes do ticker (split ou bonificação ajustados pelo
    yfinance, não pelo COTAHIST) ou corresponde a um fator de split usual."""
    bad = [r for r in rows if r["status"] == "mismatch"]
    if not bad:
        return
    store_px = [float(r["store_close"]) for r in bad]  # type: ignore[arg-type]
    b3_px = [float(r["b3_close"]) for r in bad]  # type: ignore[arg-type]
    ratios = [s / b for s, b in zip(store_px, b3_px, strict=True)]
    # Preços têm 2 casas: em papéis baratos (R$ 1 a 3) o arredondamento sozinho gera ~0,5 a 1%
    # de variação na razão, então a tolerância de "constante" cresce com 1/preço.
    tol = tolerance + 0.01 / min(min(store_px), min(b3_px))
    # Razões constantes por trecho (um evento -> um patamar; dois eventos no ano -> dois
    # patamares). Agrupa razões próximas; grupo com >= 2 datas = ajuste corporativo.
    order = sorted(range(len(ratios)), key=ratios.__getitem__)
    clusters: list[list[int]] = [[order[0]]]
    for i in order[1:]:
        if (ratios[i] - ratios[clusters[-1][0]]) / ratios[clusters[-1][0]] <= tol:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    for cluster in clusters:
        for i in cluster:
            if len(cluster) >= 2 or _looks_like_split(ratios[i]):
                bad[i]["status"] = "adjusted_split_or_bonus"


def _looks_like_split(ratio: float, tol: float = 0.02) -> bool:
    """Razões usuais de split (2, 3, 5, 10…) e de bonificação (1.05, 1.1, 1.2, 1.25, 1.4…)."""
    if ratio <= 0:
        return False
    ratio = max(ratio, 1 / ratio)
    candidates = (1.05, 1.1, 1.2, 1.25, 1.4, 1.5, 2, 2.5, 3, 4, 5, 8, 10, 20, 25, 50, 100)
    return any(abs(ratio - c) / c <= tol for c in candidates)


def _row(
    ticker: str,
    date: object,
    store_close: float | None,
    b3_close: float | None,
    status: str,
    rel_diff: float | None = None,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "date": date,
        "store_close": store_close,
        "b3_close": b3_close,
        "rel_diff": rel_diff,
        "status": status,
    }
