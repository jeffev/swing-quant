"""Catálogo dos ETFs que um brasileiro usa para comprar bolsa americana, e como baixá-los.

`assets.py` responde "em que classe de ativo colocar o dinheiro" com índices. Este catálogo
responde a pergunta seguinte, que é a que custa dinheiro: **por qual veículo comprar a classe**.
São duas rotas para o mesmo ativo:

- **B3** (`IVVB11`, `WRLD11`, ...): um fundo de índice brasileiro, cotado em reais, que compra o
  ETF americano lá fora. Não distribui rendimento — acumula — então o imposto só aparece na
  venda.
- **EUA** (`VTI`, `VOO`, ...): a cota americana comprada direto numa corretora no exterior. Paga
  dividendo em dólar, com 30% retidos na fonte pelo IRS, e cobra câmbio na ida e na volta.

A diferença entre as duas rotas não é a taxa de administração — é o câmbio, o diferimento do
imposto e a retenção sobre dividendos. Para medir isso é preciso ter, na mesma base, o preço
**e** os proventos de cada cota: `fetch_etf_history` usa `Ticker.history`, que devolve preço e
dividendo já ajustados pelo mesmo fator de desdobramento, em vez de `yf.download`, onde os dois
podem vir em bases diferentes e inventar um provento no dia do split.

Os ETFs ficam fora da tabela `universe`, como os proxies de classe: são régua, não candidatos a
trade.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

USDBRL = "USDBRL=X"

# Retenção do IRS sobre dividendos pagos a um residente brasileiro. Não há tratado entre Brasil
# e EUA: a alíquota cheia de 30% é retida na fonte, e é a mesma que o fundo brasileiro sofre
# quando o ETF lá fora paga dividendo para ele.
US_DIVIDEND_WITHHOLDING = 0.30


@dataclass(frozen=True)
class EtfVehicle:
    """Um ETF concreto e o que ele custa ao investidor brasileiro que o compra."""

    ticker: str
    label: str
    venue: str  # b3 (cotado em R$) | us (cotado em US$)
    exposure: str  # chave da exposição: sp500, world, nasdaq, us_total, intl, dividend, bonds
    exposure_label: str
    manager: str
    expense_ratio: float  # taxa anual informada pelo gestor — já embutida na cota, ver nota
    distributes: bool  # paga rendimento (EUA) ou acumula (B3)
    peer: str = ""  # o ETF da outra rota com a mesma exposição, para comparar as duas
    note: str = ""

    @property
    def currency(self) -> str:
        return "BRL" if self.venue == "b3" else "USD"


# ------------------------------------------------------------------ ETFs listados na B3 (R$)
# Todos são fundos de índice de renda variável: 15% sobre o ganho na venda, sem a isenção de
# R$20 mil das ações, sem come-cotas e sem distribuição de rendimento — o dividendo do ativo lá
# fora entra na cota. As taxas são as informadas nos regulamentos; note que elas **já estão
# dentro do preço da cota**, então o estudo não pode descontá-las de novo. O que o estudo mede é
# o atraso observado da cota contra o índice em reais, que inclui a taxa e o resto do atrito.
B3_ETFS: tuple[EtfVehicle, ...] = (
    EtfVehicle(
        "IVVB11.SA",
        "IVVB11 (iShares S&P 500)",
        "b3",
        "sp500",
        "S&P 500",
        "BlackRock",
        0.0023,
        distributes=False,
        peer="IVV",
        note="compra cotas do IVV; o ETF mais antigo da rota B3, desde abr/2014",
    ),
    EtfVehicle(
        "SPXI11.SA",
        "SPXI11 (It Now S&P 500 TRN)",
        "b3",
        "sp500",
        "S&P 500",
        "Itaú",
        0.0021,
        distributes=False,
        peer="IVV",
        note="segue a versão total return net do índice",
    ),
    EtfVehicle(
        "SPXB11.SA",
        "SPXB11 (BTG S&P 500)",
        "b3",
        "sp500",
        "S&P 500",
        "BTG Pactual",
        0.0023,
        distributes=False,
        peer="IVV",
        note="o S&P 500 mais recente da B3, desde set/2021",
    ),
    EtfVehicle(
        "WRLD11.SA",
        "WRLD11 (Investo FTSE Global Equities)",
        "b3",
        "world",
        "Ações do mundo todo",
        "Investo",
        0.0038,
        distributes=False,
        peer="VT",
        note="bolsa global num único ticker em reais, desde out/2021",
    ),
    EtfVehicle(
        "ACWI11.SA",
        "ACWI11 (Trend Bloomberg All Countries)",
        "b3",
        "world",
        "Ações do mundo todo",
        "Trend/XP",
        0.0062,
        distributes=False,
        peer="VT",
        note="mesma ideia do WRLD11, índice diferente",
    ),
    EtfVehicle(
        "NASD11.SA",
        "NASD11 (Trend Nasdaq 100)",
        "b3",
        "nasdaq",
        "Nasdaq 100",
        "Trend/XP",
        0.0050,
        distributes=False,
        peer="QQQ",
        note="a aposta concentrada em tecnologia, em reais",
    ),
)

# ------------------------------------------------------------------ ETFs listados nos EUA (US$)
# Comprados direto numa corretora no exterior. Distribuem dividendo, com 30% retidos na fonte, e
# desde a Lei 14.754/2023 o ganho é tributado em 15% na declaração anual, sem a antiga isenção
# de R$35 mil por mês. A variação cambial faz parte do ganho tributável.
US_ETFS: tuple[EtfVehicle, ...] = (
    EtfVehicle(
        "VOO",
        "VOO (Vanguard S&P 500)",
        "us",
        "sp500",
        "S&P 500",
        "Vanguard",
        0.0003,
        distributes=True,
        peer="IVVB11.SA",
        note="o S&P 500 pela rota direta",
    ),
    EtfVehicle(
        "IVV",
        "IVV (iShares Core S&P 500)",
        "us",
        "sp500",
        "S&P 500",
        "BlackRock",
        0.0003,
        distributes=True,
        peer="IVVB11.SA",
        note="é exatamente o que o IVVB11 tem na carteira — a comparação mais limpa das rotas",
    ),
    EtfVehicle(
        "VTI",
        "VTI (Vanguard Total Stock Market)",
        "us",
        "us_total",
        "Bolsa americana inteira",
        "Vanguard",
        0.0003,
        distributes=True,
        note="S&P 500 mais as small e mid caps que ele deixa de fora",
    ),
    EtfVehicle(
        "VT",
        "VT (Vanguard Total World)",
        "us",
        "world",
        "Ações do mundo todo",
        "Vanguard",
        0.0006,
        distributes=True,
        peer="WRLD11.SA",
        note="todas as bolsas do mundo, EUA incluído, num ticker",
    ),
    EtfVehicle(
        "VXUS",
        "VXUS (Vanguard Total International)",
        "us",
        "intl",
        "Bolsa fora dos EUA",
        "Vanguard",
        0.0005,
        distributes=True,
        note="o mundo sem os EUA — o contrapeso de quem acha o S&P caro",
    ),
    EtfVehicle(
        "QQQ",
        "QQQ (Invesco Nasdaq 100)",
        "us",
        "nasdaq",
        "Nasdaq 100",
        "Invesco",
        0.0020,
        distributes=True,
        peer="NASD11.SA",
        note="o índice de tecnologia que puxou o retorno da década",
    ),
    EtfVehicle(
        "SCHD",
        "SCHD (Schwab US Dividend Equity)",
        "us",
        "dividend",
        "Ações americanas de dividendo",
        "Schwab",
        0.0006,
        distributes=True,
        note="o caso em que a retenção de 30% na fonte mais dói: quase todo o retorno é dividendo",
    ),
    EtfVehicle(
        "BND",
        "BND (Vanguard Total Bond Market)",
        "us",
        "bonds",
        "Renda fixa americana",
        "Vanguard",
        0.0003,
        distributes=True,
        note="a perna conservadora em dólar, para quem quer câmbio sem bolsa",
    ),
)

ETF_VEHICLES: tuple[EtfVehicle, ...] = B3_ETFS + US_ETFS
ETF_BY_TICKER: dict[str, EtfVehicle] = {e.ticker: e for e in ETF_VEHICLES}

EXPOSURE_LABEL: dict[str, str] = {e.exposure: e.exposure_label for e in ETF_VEHICLES}


def etfs_for(venue: str | None = None) -> tuple[EtfVehicle, ...]:
    if venue is None:
        return ETF_VEHICLES
    return tuple(e for e in ETF_VEHICLES if e.venue == venue)


def etf_tickers(venue: str | None = None) -> list[str]:
    """Todos os tickers do catálogo, mais o câmbio de que a conversão para reais depende."""
    return sorted({e.ticker for e in etfs_for(venue)} | {USDBRL})


def peers() -> list[tuple[EtfVehicle, EtfVehicle]]:
    """Pares (ETF da B3, ETF americano) com a mesma exposição, sem repetir o par invertido."""
    out = []
    for e in B3_ETFS:
        peer = ETF_BY_TICKER.get(e.peer)
        if peer is not None:
            out.append((e, peer))
    return out


# --------------------------------------------------------------------------- higiene da série
# Um ETF de índice amplo não cai 87% num pregão. Quando isso aparece, é desdobramento que o
# Yahoo não ajustou — acontece com ETF da B3, cujos eventos societários ele não recebe. O
# SPXI11 é o caso do estudo: desdobrou 8 para 1 em 22/01/2026 e a série ficou com uma queda de
# 87,6% que, se não for corrigida, transforma o retorno de 16 anos em prejuízo.
SPLIT_THRESHOLD = 0.40
SPLIT_RATIOS: tuple[float, ...] = (2, 3, 4, 5, 8, 10, 20, 25, 50, 100)
SPLIT_TOLERANCE = 0.05


def _nearest_split_factor(ratio: float) -> float | None:
    """O fator de desdobramento mais próximo da razão observada, ou None se não for um.

    `ratio` é o preço da véspera dividido pelo de hoje: 8,0 num desdobramento de 8 para 1, e
    0,1 num grupamento de 1 para 10. Só razões perto de um fator redondo são aceitas — uma
    queda de 45% num ETF de índice é suspeita, mas não é um split e não deve ser "corrigida".
    """
    for target in SPLIT_RATIOS:
        for factor in (float(target), 1.0 / float(target)):
            if abs(ratio / factor - 1.0) <= SPLIT_TOLERANCE:
                return factor
    return None


def detect_unadjusted_splits(close: pd.Series) -> list[tuple[pd.Timestamp, float]]:
    """Datas em que o preço pulou por fator de desdobramento, com o fator (>1 = desdobramento)."""
    s = close.dropna().sort_index()
    dates = pd.DatetimeIndex(s.index)
    values = s.to_numpy(dtype=float)
    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(1, len(values)):
        previous, current = values[i - 1], values[i]
        if current <= 0 or previous <= 0:
            continue
        if abs(current / previous - 1.0) < SPLIT_THRESHOLD:
            continue
        factor = _nearest_split_factor(previous / current)
        if factor is not None:
            out.append((pd.Timestamp(dates[i]), factor))
    return out


def repair_unadjusted_splits(
    close: pd.Series,
) -> tuple[pd.Series, list[tuple[pd.Timestamp, float]]]:
    """Reescala o histórico anterior a cada desdobramento não ajustado.

    Devolve a série corrigida e os eventos aplicados, para o estudo poder mostrar o que foi
    mexido em vez de esconder a correção dentro do pipeline.
    """
    events = detect_unadjusted_splits(close)
    if not events:
        return close, []
    fixed = close.copy()
    for date, factor in events:
        fixed.loc[fixed.index < date] = fixed.loc[fixed.index < date] / factor
        log.warning("split não ajustado em %s: fator %.3f corrigido", date.date(), factor)
    return fixed, events


# --------------------------------------------------------------------------- download
def fetch_etf_history(
    ticker: str, start: dt.date | str, end: dt.date | str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baixa preços e proventos de um ETF na mesma base de desdobramento.

    Devolve `(prices, events)`: o primeiro no formato de `PRICE_COLUMNS`, o segundo com as
    colunas de `corporate_events` (`type` em {"dividend", "split"}). Preço e dividendo saem da
    mesma chamada de propósito — é o que permite reconstruir o retorno total líquido da retenção
    de 30% sem supor um dividend yield.
    """
    import yfinance as yf  # import tardio: pesado e só usado com rede

    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    hist = yf.Ticker(ticker).history(
        start=str(pd.Timestamp(start).date()),
        end=str((end_ts + pd.Timedelta(days=1)).date()),  # yfinance: `end` é exclusivo
        auto_adjust=False,
        actions=True,
    )
    if hist.empty:
        return pd.DataFrame(), pd.DataFrame()

    hist = hist.copy()
    hist.index = pd.DatetimeIndex(hist.index).tz_localize(None).normalize()
    hist.index.name = "date"

    prices = pd.DataFrame(
        {
            "ticker": ticker,
            "date": hist.index,
            "open": hist["Open"].to_numpy(),
            "high": hist["High"].to_numpy(),
            "low": hist["Low"].to_numpy(),
            "close": hist["Close"].to_numpy(),
            "adj_close": hist.get("Adj Close", hist["Close"]).to_numpy(),
            "volume": hist["Volume"].fillna(0).astype("int64").to_numpy(),
            "source": "yfinance:history",
        }
    ).dropna(subset=["close"])

    rows: list[pd.DataFrame] = []
    for col, kind in (("Dividends", "dividend"), ("Stock Splits", "split")):
        if col not in hist:
            continue
        s = hist[col]
        s = s[s > 0]
        if s.empty:
            continue
        rows.append(
            pd.DataFrame({"ticker": ticker, "date": s.index, "type": kind, "value": s.to_numpy()})
        )
    events = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["ticker", "date", "type", "value"])
    )
    return prices, events
