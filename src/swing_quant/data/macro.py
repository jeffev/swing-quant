"""Séries macro e de renda fixa que não são preço de ativo: IPCA, poupança, imóveis, CPI e
o Tesouro Direto.

São o que falta para comparar *classes de ativo* e não só ações contra caixa: um índice de
preços para converter retorno nominal em real, um índice de valor de imóveis residenciais para
representar o imóvel físico (que não tem cotação diária), a poupança — ainda o destino da maior
parte da poupança das famílias — e os títulos públicos com marcação a mercado.

Como em `riskfree.py`, nada aqui muda backtest: a tabela `macro` só alimenta comparações.

Unidades (coluna `unit` da tabela, e `MACRO_CATALOG` aqui):

- `index`        nível de um índice (IVG-R, CPI) — só a variação tem significado
- `pct_month`    variação percentual do mês, em pontos percentuais (IPCA 0,75 = +0,75%)
- `daily_return` retorno diário já em decimal (0,0004 = +0,04%)
"""

from __future__ import annotations

import datetime as dt
import io
import json
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

from swing_quant.data.store import MACRO_COLUMNS, MarketStore

_HEADERS = {"User-Agent": "swing-quant/1.0"}


# --------------------------------------------------------------------------- catálogo
@dataclass(frozen=True)
class MacroSeries:
    key: str
    label: str
    unit: str
    source: str
    note: str


MACRO_CATALOG: dict[str, MacroSeries] = {
    "ipca": MacroSeries(
        "ipca",
        "IPCA",
        "pct_month",
        "bcb_sgs_433",
        "inflação oficial — deflator do lado brasileiro",
    ),
    "tr": MacroSeries("tr", "TR", "pct_month", "bcb_sgs_7811", "usada na regra da poupança"),
    "poupanca": MacroSeries(
        "poupanca",
        "Poupança",
        "pct_month",
        "bcb_sgs_196",
        "rendimento mensal; antes de jun/2012 reconstruído pela regra antiga (0,5% + TR)",
    ),
    "ivgr": MacroSeries(
        "ivgr",
        "Imóveis residenciais (IVG-R)",
        "index",
        "bcb_sgs_21340",
        "valor de garantia de imóveis financiados — só a valorização, sem aluguel",
    ),
    "cpi_us": MacroSeries(
        "cpi_us",
        "CPI",
        "index",
        "bls_CUUR0000SA0",
        "inflação oficial dos EUA — deflator do lado americano",
    ),
    "tesouro_ipca": MacroSeries(
        "tesouro_ipca",
        "Tesouro IPCA+ (~10 anos, rolado)",
        "daily_return",
        "tesouro_direto",
        "NTN-B Principal marcada a mercado, rolando no vencimento mais próximo de 10 anos",
    ),
    "tesouro_prefixado": MacroSeries(
        "tesouro_prefixado",
        "Tesouro Prefixado (~4 anos, rolado)",
        "daily_return",
        "tesouro_direto",
        "LTN marcada a mercado, rolando no vencimento mais próximo de 4 anos",
    ),
    "tesouro_selic": MacroSeries(
        "tesouro_selic",
        "Tesouro Selic",
        "daily_return",
        "tesouro_direto",
        "LFT — o piso pós-fixado, praticamente sem marcação a mercado",
    ),
}


def _cols(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if df.empty:
        return pd.Series(dtype="datetime64[ns]"), pd.Series(dtype=float)
    return df["date"], df["value"]


def _frame(key: str, dates: pd.Series, values: pd.Series) -> pd.DataFrame:
    meta = MACRO_CATALOG[key]
    return pd.DataFrame(
        {
            "series": key,
            "date": pd.to_datetime(dates),
            "value": pd.to_numeric(values, errors="coerce").astype(float),
            "unit": meta.unit,
            "source": meta.source,
        },
        columns=list(MACRO_COLUMNS),
    ).dropna(subset=["value"])


# --------------------------------------------------------------------------- BCB / SGS
_SGS_URL = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados"
    "?formato=json&dataInicial={ini}&dataFinal={fim}"
)
_SGS_MAX_YEARS = 10  # a API recusa janelas maiores (HTTP 406)


def fetch_sgs(serie: int, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Série do SGS do Banco Central, em blocos de 10 anos. Colunas: `date`, `value`."""
    rows: list[dict[str, str]] = []
    ini = start
    while ini <= end:
        fim = min(end, dt.date(ini.year + _SGS_MAX_YEARS - 1, 12, 31))
        url = _SGS_URL.format(serie=serie, ini=f"{ini:%d/%m/%Y}", fim=f"{fim:%d/%m/%Y}")
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8-sig").strip()
        if body:  # janela sem dados devolve corpo vazio, não erro
            rows += json.loads(body)
        ini = fim + dt.timedelta(days=1)
    if not rows:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(rows).drop_duplicates(subset="data")
    return pd.DataFrame(
        {
            "date": pd.to_datetime(df["data"], format="%d/%m/%Y"),
            "value": pd.to_numeric(df["valor"], errors="coerce"),
        }
    ).dropna()


def fetch_ipca(start: dt.date, end: dt.date) -> pd.DataFrame:
    return _frame("ipca", *_cols(fetch_sgs(433, start, end)))


def fetch_ivgr(start: dt.date, end: dt.date) -> pd.DataFrame:
    return _frame("ivgr", *_cols(fetch_sgs(21340, start, end)))


def fetch_poupanca(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Rendimento mensal da poupança, emendando a regra antiga na nova.

    A série 196 (regra pós-mai/2012) começa em jun/2012. Antes disso valia a regra antiga —
    0,5% ao mês mais TR — e reconstruí-la com a série 7811 não é aproximação: no período a
    Selic estava bem acima de 8,5%, então o teto de 0,5% era o que valia de fato.
    """
    nova = fetch_sgs(196, start, end)
    tr = fetch_sgs(7811, start, end)
    antiga = (tr[tr["date"] < nova["date"].min()] if not nova.empty else tr).copy()
    antiga["value"] = antiga["value"] + 0.5
    both = pd.concat([antiga, nova]).sort_values("date").drop_duplicates("date", keep="last")
    return _frame("poupanca", *_cols(both))


# --------------------------------------------------------------------------- BLS (CPI dos EUA)
_BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/{sid}?startyear={a}&endyear={b}"
_BLS_MAX_YEARS = 10  # limite da API pública sem chave


def fetch_cpi_us(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Nível do CPI-U (todos os itens, sem ajuste sazonal) pela API pública do BLS."""
    frames: list[pd.DataFrame] = []
    a = start.year
    while a <= end.year:
        b = min(end.year, a + _BLS_MAX_YEARS - 1)
        req = urllib.request.Request(_BLS_URL.format(sid="CUUR0000SA0", a=a, b=b), headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read())
        if payload.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS recusou a requisição: {payload.get('message')}")
        for serie in payload["Results"]["series"]:
            # M13 é a média anual, e um mês ainda não publicado vem com valor "-".
            rows = [
                r
                for r in serie["data"]
                if str(r["period"]).startswith("M")
                and str(r["period"]) != "M13"
                and str(r["value"]).strip() not in {"", "-"}
            ]
            if rows:
                frames.append(
                    pd.DataFrame(
                        {
                            "date": pd.to_datetime(
                                [f"{r['year']}-{str(r['period'])[1:]}-01" for r in rows]
                            ),
                            "value": [float(r["value"]) for r in rows],
                        }
                    )
                )
        a = b + 1
    if not frames:
        return pd.DataFrame(columns=list(MACRO_COLUMNS))
    df = pd.concat(frames).drop_duplicates("date").sort_values("date")
    return _frame("cpi_us", *_cols(df))


# --------------------------------------------------------------------------- Tesouro Direto
_TD_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/"
    "resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv"
)
# Vencimento-alvo de cada curva rolada, em anos.
TD_SERIES: dict[str, tuple[str, float]] = {
    "tesouro_ipca": ("Tesouro IPCA+", 10.0),
    "tesouro_prefixado": ("Tesouro Prefixado", 4.0),
    "tesouro_selic": ("Tesouro Selic", 5.0),
}


def fetch_tesouro_direto(timeout: float = 300.0) -> pd.DataFrame:
    """Histórico completo de preços e taxas do Tesouro Direto (CSV público, ~15 MB)."""
    import httpx  # certifi: o urllib desta máquina não valida a cadeia deste host

    resp = httpx.get(
        _TD_URL, timeout=timeout, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}
    )
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content), sep=";", decimal=",", encoding="latin1")
    return pd.DataFrame(
        {
            "kind": df["Tipo Titulo"].astype(str).str.strip(),
            "date": pd.to_datetime(df["Data Base"], format="%d/%m/%Y"),
            "maturity": pd.to_datetime(df["Data Vencimento"], format="%d/%m/%Y"),
            "pu": pd.to_numeric(df["PU Base Manha"], errors="coerce"),
        }
    ).dropna(subset=["pu"])


def rolled_bond_returns(td: pd.DataFrame, kind: str, target_years: float) -> pd.DataFrame:
    """Retorno diário de uma posição que rola sempre no vencimento mais próximo do alvo.

    O retorno de cada dia vem do PU do papel que já estava em carteira na véspera, nunca de
    dois papéis diferentes — assim a troca de vencimento não inventa ganho nem perda, que é o
    erro clássico de quem monta série de título público pelo papel "corrente" de cada dia.
    """
    d = td[td["kind"] == kind]
    wide = d.pivot_table(index="date", columns="maturity", values="pu").sort_index()
    if len(wide) < 2:
        return pd.DataFrame(columns=["date", "value"])
    dates = pd.DatetimeIndex(wide.index)
    mats = pd.DatetimeIndex(wide.columns)
    days_to_maturity = (mats.to_numpy()[None, :] - dates.to_numpy()[:, None]) / np.timedelta64(
        1, "D"
    )
    years_to_maturity = days_to_maturity / 365.25
    distance = pd.DataFrame(
        np.abs(years_to_maturity - target_years), index=dates, columns=wide.columns
    ).where(wide.notna())
    # Papel carregado hoje é o que foi escolhido ontem: a rolagem não gera retorno.
    held = distance.idxmin(axis=1, skipna=True).shift(1)
    values = wide.to_numpy()
    col_of = {m: i for i, m in enumerate(wide.columns)}
    rets: list[float] = []
    for i in range(1, len(dates)):
        m = held.iloc[i]
        if pd.isna(m):
            rets.append(float("nan"))
            continue
        j = col_of[m]
        prev, cur = values[i - 1, j], values[i, j]
        rets.append(cur / prev - 1.0 if prev > 0 and cur > 0 else float("nan"))
    return pd.DataFrame({"date": dates[1:], "value": rets}).dropna()


def tesouro_curves(td: pd.DataFrame, start: dt.date | None = None) -> pd.DataFrame:
    """Todas as curvas de `TD_SERIES` no formato da tabela `macro`."""
    frames = []
    for key, (kind, years) in TD_SERIES.items():
        curve = rolled_bond_returns(td, kind, years)
        if start is not None and not curve.empty:
            curve = curve[curve["date"] >= pd.Timestamp(start)]
        frames.append(_frame(key, *_cols(curve)))
    if not frames:
        return pd.DataFrame(columns=list(MACRO_COLUMNS))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- orquestração
_SIMPLE_FETCHERS = {
    "ipca": fetch_ipca,
    "poupanca": fetch_poupanca,
    "ivgr": fetch_ivgr,
    "cpi_us": fetch_cpi_us,
}


def update_macro(
    store: MarketStore,
    start: dt.date,
    end: dt.date | None = None,
    *,
    only: list[str] | None = None,
) -> dict[str, int]:
    """Baixa e grava as séries macro; devolve quantas linhas cada uma gravou.

    Uma fonte fora do ar não derruba as outras: a série entra no resultado com -1.
    """
    end = end or dt.date.today()
    wanted = set(only) if only else set(MACRO_CATALOG) - {"tr"}
    out: dict[str, int] = {}

    for key, fetch in _SIMPLE_FETCHERS.items():
        if key not in wanted:
            continue
        try:
            out[key] = store.upsert_macro(fetch(start, end))
        except Exception:
            out[key] = -1

    td_keys = [k for k in TD_SERIES if k in wanted]
    if td_keys:
        try:
            df = tesouro_curves(fetch_tesouro_direto(), start)
            for key in td_keys:
                out[key] = store.upsert_macro(df[df["series"] == key])
        except Exception:
            for key in td_keys:
                out[key] = -1
    return out
