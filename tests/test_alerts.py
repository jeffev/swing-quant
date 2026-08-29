import pandas as pd

from swing_quant.alerts.telegram import MAX_LEN, format_failure, format_message, send
from swing_quant.screener.core import ENTRY_COLUMNS, EXIT_COLUMNS, ScreenResult


def _result(n_entries: int = 2) -> ScreenResult:
    entries = pd.DataFrame(
        [
            {
                "ticker": f"T{i}.SA",
                "strategy": "donchian",
                "ref_price": 10.0 + i,
                "qty": 100,
                "notional": 1000.0 + 100 * i,
                "stop_price": 9.0 if i % 2 else None,
                "max_hold": 0,
                "score": 2.0 - i * 0.1,
                "atr": 0.5,
                "dollar_volume": 1e7,
            }
            for i in range(n_entries)
        ],
        columns=ENTRY_COLUMNS,
    )
    exits = pd.DataFrame(
        [
            {
                "ticker": "X.SA",
                "strategy": "donchian",
                "reason": "stop",
                "ref_price": 8.0,
                "entry_date": pd.Timestamp("2026-08-01").date(),
                "bars_held": 10,
                "qty": 200,
            }
        ],
        columns=EXIT_COLUMNS,
    )
    return ScreenResult(
        as_of=pd.Timestamp("2026-08-27"),
        market="b3",
        entries=entries,
        exits=exits,
        equity=123_456.0,
        cash=100_000.0,
        open_positions=1,
        slots=5,
        regime={"allow_entries": True, "size_factor": 0.5, "trend_on": True, "high_vol": True},
        notes=["Regime: volatilidade alta — sizing × 0.50."],
    )


def test_format_message_content() -> None:
    msg = format_message(_result())
    assert "27/08/2026" in msg and "123.456" in msg
    assert "Saídas (1)" in msg and "`X.SA`" in msg and "stop" in msg
    assert "Entradas (2)" in msg and "`T0.SA`" in msg and "stop 9.00" in msg
    assert "vol alta" in msg and "×0.50" in msg
    assert "_Regime: volatilidade alta" in msg


def test_format_message_empty_and_truncation() -> None:
    r = _result(0)
    r.exits = r.exits.iloc[0:0]
    msg = format_message(r)
    assert "Entradas:* nenhuma" in msg and "Saídas:* nenhuma" in msg
    big = format_message(_result(400), top_n=400)
    assert len(big) <= MAX_LEN


def test_format_failure() -> None:
    msg = format_failure("update-data", "Traceback ...", "b3")
    assert "falhou" in msg and "B3" in msg and "update-data" in msg


def test_send_without_credentials_returns_false(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert send("oi") is False
