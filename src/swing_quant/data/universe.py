"""Composição de índices (IBrX-100 via B3, S&P 500 via Wikipedia).

Cada fetcher devolve um DataFrame com colunas `ticker` (símbolo local) e `sector`.
A conversão para símbolo yfinance fica em `to_yf_symbol`.
"""

from __future__ import annotations

import base64
import json
from io import StringIO
from typing import Any

import httpx
import pandas as pd

from swing_quant.data.calendar import Market

_B3_INDEX_URL = "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/{}"
_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "Mozilla/5.0 (swing-quant; +https://github.com)"}

INDEX_BY_MARKET: dict[Market, str] = {"b3": "IBRX100", "us": "SP500"}
_B3_INDEX_CODES: dict[str, str] = {
    "IBRX100": "IBXX",
    "IBOV": "IBOV",
    "SMLL": "SMLL",
    "IFIX": "IFIX",
}


# --------------------------------------------------------------------------- símbolos
def to_yf_symbol(ticker: str, market: Market) -> str:
    """Converte símbolo local para o formato yfinance (PETR4 -> PETR4.SA; BRK.B -> BRK-B)."""
    t = ticker.strip().upper()
    if market == "b3":
        return t if t.endswith(".SA") else f"{t}.SA"
    return t.replace(".", "-")


def from_yf_symbol(symbol: str, market: Market) -> str:
    s = symbol.strip().upper()
    if market == "b3":
        return s.removesuffix(".SA")
    return s.replace("-", ".")


# --------------------------------------------------------------------------- B3
def _b3_payload(index_code: str, segment: str, page: int = 1, page_size: int = 200) -> str:
    body = {
        "language": "pt-br",
        "pageNumber": page,
        "pageSize": page_size,
        "index": index_code,
        "segment": segment,
    }
    return base64.b64encode(json.dumps(body).encode()).decode()


def parse_b3_portfolio(payload: dict[str, Any]) -> pd.DataFrame:
    """Transforma a resposta JSON da B3 em DataFrame(ticker, sector, weight)."""
    rows = payload.get("results") or []
    if not rows:
        return pd.DataFrame(columns=["ticker", "sector", "weight"])
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "ticker": df["cod"].astype(str).str.strip(),
            "sector": df["segment"].astype(str).str.strip() if "segment" in df else pd.NA,
            "weight": pd.to_numeric(
                df["part"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )
            if "part" in df
            else pd.NA,
        }
    )
    # A B3 devolve linhas de subtotal por setor sem código de ativo; remover.
    out = out[out["ticker"].str.match(r"^[A-Z0-9]{5,6}$")]
    return out.drop_duplicates("ticker").reset_index(drop=True)


def fetch_b3_index(index_name: str = "IBRX100", timeout: float = 30.0) -> pd.DataFrame:
    """Baixa a carteira teórica do dia (com setor) do índice na B3."""
    code = _B3_INDEX_CODES.get(index_name, index_name)
    url = _B3_INDEX_URL.format(_b3_payload(code, segment="2"))
    with httpx.Client(headers=_HEADERS, timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        df = parse_b3_portfolio(resp.json())
    if df.empty:  # fallback: lista plana sem setor
        url = _B3_INDEX_URL.format(_b3_payload(code, segment="1"))
        with httpx.Client(headers=_HEADERS, timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            df = parse_b3_portfolio(resp.json())
    if df.empty:
        raise RuntimeError(f"B3 devolveu carteira vazia para {index_name}")
    return df


# --------------------------------------------------------------------------- S&P 500
def parse_sp500_html(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    table = next(t for t in tables if "Symbol" in t.columns)
    sector_col = next((c for c in table.columns if "Sector" in str(c)), None)
    out = pd.DataFrame(
        {
            "ticker": table["Symbol"].astype(str).str.strip(),
            "sector": table[sector_col].astype(str).str.strip() if sector_col else pd.NA,
        }
    )
    return out.drop_duplicates("ticker").reset_index(drop=True)


def fetch_sp500(timeout: float = 30.0) -> pd.DataFrame:
    with httpx.Client(headers=_HEADERS, timeout=timeout, follow_redirects=True) as client:
        resp = client.get(_SP500_URL)
        resp.raise_for_status()
    df = parse_sp500_html(resp.text)
    if len(df) < 480:
        raise RuntimeError(f"S&P 500 com apenas {len(df)} membros — página mudou?")
    return df


# --------------------------------------------------------------------------- dispatcher
def fetch_index(market: Market) -> pd.DataFrame:
    if market == "b3":
        return fetch_b3_index("IBRX100")
    return fetch_sp500()


# --------------------------------------------------------------------------- séries de índice
_B3_STATS_URL = (
    "https://sistemaswebb3-listados.b3.com.br/indexStatisticsProxy/IndexCall/GetPortfolioDay/{}"
)


def parse_b3_index_year(payload: dict[str, Any], year: int) -> pd.DataFrame:
    """Transforma a matriz dia x mês da B3 numa série diária (colunas `date`, `close`)."""
    rows = payload.get("results") or []
    out: list[tuple[pd.Timestamp, float]] = []
    for row in rows:
        day = int(row.get("day", 0))
        for month in range(1, 13):
            raw = row.get(f"rateValue{month}")
            if not raw:
                continue
            value = float(str(raw).replace(".", "").replace(",", "."))
            try:
                out.append((pd.Timestamp(year=year, month=month, day=day), value))
            except ValueError:  # dia 31 em mês de 30: a B3 devolve a grade cheia
                continue
    if not out:
        return pd.DataFrame(columns=["date", "close"])
    df = pd.DataFrame(out, columns=["date", "close"]).sort_values("date")
    return df.reset_index(drop=True)


def fetch_b3_index_series(
    index_name: str, start_year: int, end_year: int, timeout: float = 40.0
) -> pd.DataFrame:
    """Série diária de fechamento de um índice da B3 (IFIX, SMLL, IBOV...), ano a ano.

    É a fonte certa para índices que o yfinance não tem — e, no caso do IFIX, a única correta:
    reconstruir FIIs por cotação de fundos individuais erra os proventos (que são o retorno
    quase inteiro da classe) e ainda carrega o viés de só enxergar quem sobreviveu.
    """
    code = _B3_INDEX_CODES.get(index_name, index_name)
    frames: list[pd.DataFrame] = []
    with httpx.Client(headers=_HEADERS, timeout=timeout, follow_redirects=True) as client:
        for year in range(start_year, end_year + 1):
            payload = _b3_payload_year(code, year)
            resp = client.get(_B3_STATS_URL.format(payload))
            resp.raise_for_status()
            frames.append(parse_b3_index_year(resp.json(), year))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "close"])
    if df.empty:
        raise RuntimeError(f"B3 não devolveu série para {index_name} em {start_year}-{end_year}")
    return df.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)


def _b3_payload_year(index_code: str, year: int) -> str:
    body = {"language": "pt-br", "index": index_code, "year": str(year)}
    return base64.b64encode(json.dumps(body).encode()).decode()
