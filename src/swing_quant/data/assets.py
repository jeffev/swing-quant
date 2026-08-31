"""Catálogo dos proxies de classe de ativo usados nas comparações.

O universo do projeto (IBrX-100, S&P 500) responde "qual ação comprar". Este catálogo responde
uma pergunta anterior: **em que classe de ativo colocar o dinheiro**. São séries de referência,
não candidatas a trade — nada aqui entra no screener nem no portfólio.

Cada proxy vira uma curva de retorno total na moeda do investidor:

- `ticker`   série do yfinance, já com dividendos no `adj_close`
- `fx`       série em moeda estrangeira convertida pelo câmbio (ouro e S&P 500 em reais)
- `b3_index` índice publicado pela B3 (IFIX, SMLL) — ver `data/universe.py`
- `macro`    série da tabela `macro` (imóveis, poupança, Tesouro) — ver `data/macro.py`
- `riskfree` a série de renda fixa do mercado, que já existe na tabela `risk_free`

Por que FIIs e small caps vêm da B3 e não do yfinance: o `adj_close` de ativos brasileiros
subestima proventos (o yfinance só conhece os pagamentos recentes). Num FII, que distribui quase
todo o retorno em dinheiro, isso não é um detalhe — a diferença entre a cesta reconstruída por
cotação e o IFIX oficial passa de quatro pontos ao ano.

Os preços dos proxies ficam na mesma tabela `prices` do resto do projeto; quem separa universo
de referência é a tabela `universe`, então não há risco de um FII virar candidato a swing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

USDBRL = "USDBRL=X"


@dataclass(frozen=True)
class AssetProxy:
    """Uma classe de ativo e como reconstruir sua curva de retorno total."""

    key: str
    label: str
    asset_class: str  # equity | real_estate | fixed_income | cash | commodity | crypto | fx
    market: str  # sleeve a que pertence: b3 (em BRL) ou us (em USD)
    kind: str  # ticker | fx | b3_index | macro | riskfree
    symbols: tuple[str, ...] = ()
    series: str = ""  # para kind="macro"
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- Brasil (em R$)
BR_PROXIES: tuple[AssetProxy, ...] = (
    AssetProxy(
        "acoes_br",
        "Ações Brasil (Ibovespa)",
        "equity",
        "b3",
        "ticker",
        ("^BVSP",),
        note="índice de retorno total: reinveste dividendos por construção",
    ),
    AssetProxy(
        "small_caps_br",
        "Small caps Brasil (SMLL)",
        "equity",
        "b3",
        "b3_index",
        ("^SMLL",),
        note="índice oficial de retorno total das small caps",
    ),
    AssetProxy(
        "fiis",
        "FIIs (IFIX)",
        "real_estate",
        "b3",
        "b3_index",
        ("^IFIX",),
        note="índice oficial de retorno total: os rendimentos mensais entram reinvestidos",
    ),
    AssetProxy(
        "imoveis",
        "Imóvel físico (IVG-R)",
        "real_estate",
        "b3",
        "macro",
        series="ivgr",
        note="só a valorização do imóvel; sem aluguel, sem custos e sem liquidez diária",
        tags=("monthly", "no_income"),
    ),
    AssetProxy(
        "sp500_brl",
        "S&P 500 em reais",
        "equity",
        "b3",
        "fx",
        ("SPY", USDBRL),
        note="o mesmo SPY do sleeve americano, convertido pelo câmbio do dia",
    ),
    AssetProxy(
        "ouro_brl",
        "Ouro em reais",
        "commodity",
        "b3",
        "fx",
        ("GC=F", USDBRL),
        note="futuro do ouro em dólar, convertido pelo câmbio",
    ),
    AssetProxy(
        "dolar",
        "Dólar (sem remuneração)",
        "fx",
        "b3",
        "ticker",
        (USDBRL,),
        note="dólar guardado, sem render juros — o pior jeito de ter dólar",
    ),
    AssetProxy(
        "bitcoin_brl",
        "Bitcoin em reais",
        "crypto",
        "b3",
        "fx",
        ("BTC-USD", USDBRL),
        note="série começa em set/2014",
    ),
    AssetProxy(
        "cdi",
        "CDI",
        "cash",
        "b3",
        "riskfree",
        note="o caixa remunerado — a régua de qualquer investimento brasileiro",
    ),
    AssetProxy(
        "poupanca",
        "Poupança",
        "cash",
        "b3",
        "macro",
        series="poupanca",
        note="rendimento mensal oficial; isenta de IR",
        tags=("monthly",),
    ),
    AssetProxy(
        "tesouro_selic",
        "Tesouro Selic",
        "fixed_income",
        "b3",
        "macro",
        series="tesouro_selic",
    ),
    AssetProxy(
        "tesouro_prefixado",
        "Tesouro Prefixado (~4 anos)",
        "fixed_income",
        "b3",
        "macro",
        series="tesouro_prefixado",
        note="marcado a mercado, rolado no vencimento mais próximo de 4 anos",
    ),
    AssetProxy(
        "tesouro_ipca",
        "Tesouro IPCA+ (~10 anos)",
        "fixed_income",
        "b3",
        "macro",
        series="tesouro_ipca",
        note="marcado a mercado, rolado no vencimento mais próximo de 10 anos",
    ),
)

# --------------------------------------------------------------------------- EUA (em US$)
US_PROXIES: tuple[AssetProxy, ...] = (
    AssetProxy("acoes_us", "Ações EUA (S&P 500)", "equity", "us", "ticker", ("SPY",)),
    AssetProxy("nasdaq", "Nasdaq 100 (QQQ)", "equity", "us", "ticker", ("QQQ",)),
    AssetProxy("small_caps_us", "Small caps EUA (IWM)", "equity", "us", "ticker", ("IWM",)),
    AssetProxy("intl_dev", "Desenvolvidos ex-EUA (EFA)", "equity", "us", "ticker", ("EFA",)),
    AssetProxy("emergentes", "Emergentes (EEM)", "equity", "us", "ticker", ("EEM",)),
    AssetProxy("reits_us", "REITs (VNQ)", "real_estate", "us", "ticker", ("VNQ",)),
    AssetProxy(
        "bonds_us",
        "Renda fixa agregada (AGG)",
        "fixed_income",
        "us",
        "ticker",
        ("AGG",),
        note="o mercado de bonds inteiro, duration ~6 anos",
    ),
    AssetProxy(
        "treasuries_longas",
        "Treasuries longas (TLT)",
        "fixed_income",
        "us",
        "ticker",
        ("TLT",),
        note="20 anos ou mais — o ativo que a alta de juros de 2022 destruiu",
    ),
    AssetProxy(
        "tips", "Títulos indexados à inflação (TIP)", "fixed_income", "us", "ticker", ("TIP",)
    ),
    AssetProxy("ouro_usd", "Ouro (GLD)", "commodity", "us", "ticker", ("GLD",)),
    AssetProxy("commodities", "Commodities (DBC)", "commodity", "us", "ticker", ("DBC",)),
    AssetProxy("bitcoin_usd", "Bitcoin", "crypto", "us", "ticker", ("BTC-USD",)),
    AssetProxy("tbills", "T-bills", "cash", "us", "riskfree"),
)

ASSET_PROXIES: tuple[AssetProxy, ...] = BR_PROXIES + US_PROXIES
PROXY_BY_KEY: dict[str, AssetProxy] = {p.key: p for p in ASSET_PROXIES}

ASSET_CLASS_LABEL: dict[str, str] = {
    "equity": "Ações",
    "real_estate": "Imóveis",
    "fixed_income": "Renda fixa",
    "cash": "Caixa",
    "commodity": "Commodities",
    "crypto": "Cripto",
    "fx": "Câmbio",
}


def proxies_for(market: str | None = None) -> tuple[AssetProxy, ...]:
    if market is None:
        return ASSET_PROXIES
    return tuple(p for p in ASSET_PROXIES if p.market == market)


def proxy_tickers(market: str | None = None) -> list[str]:
    """Todos os símbolos do yfinance que os proxies precisam, sem repetição e ordenados."""
    out: set[str] = set()
    for p in proxies_for(market):
        if p.kind in {"ticker", "fx"}:
            out.update(p.symbols)
    return sorted(out)


def b3_index_symbols(market: str | None = None) -> list[str]:
    """Símbolos dos índices que vêm da B3, no formato `^IFIX`."""
    return sorted({s for p in proxies_for(market) if p.kind == "b3_index" for s in p.symbols})


def index_name_of(symbol: str) -> str:
    """`^IFIX` -> `IFIX`, o nome que `fetch_b3_index_series` espera."""
    return symbol.lstrip("^").upper()


def macro_series_needed(market: str | None = None) -> list[str]:
    return sorted({p.series for p in proxies_for(market) if p.kind == "macro" and p.series})
