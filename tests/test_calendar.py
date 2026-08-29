import datetime as dt

from swing_quant.data.calendar import (
    is_trading_day,
    last_trading_day,
    next_trading_day,
    trading_days,
)


def test_b3_carnival_is_holiday() -> None:
    # Carnaval 2024: 12 e 13 de fevereiro
    days = trading_days("b3", dt.date(2024, 2, 9), dt.date(2024, 2, 15))
    assert [d.date() for d in days] == [
        dt.date(2024, 2, 9),
        dt.date(2024, 2, 14),
        dt.date(2024, 2, 15),
    ]


def test_nyse_independence_day() -> None:
    assert not is_trading_day("us", dt.date(2024, 7, 4))
    assert is_trading_day("us", dt.date(2024, 7, 5))


def test_weekend_not_trading() -> None:
    assert not is_trading_day("b3", dt.date(2024, 6, 1))  # sábado


def test_last_and_next_trading_day() -> None:
    # 2024-11-15 (sexta) é feriado na B3 (Proclamação da República)
    assert last_trading_day("b3", dt.date(2024, 11, 17)) == dt.date(2024, 11, 14)
    assert next_trading_day("b3", dt.date(2024, 11, 14)) == dt.date(2024, 11, 18)


def test_trading_days_normalized() -> None:
    days = trading_days("us", dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    assert days.tz is None
    assert all(d.hour == 0 for d in days)
