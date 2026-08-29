"""Formatação e envio do resumo diário via Telegram Bot API."""

from __future__ import annotations

import os

import httpx
import pandas as pd

from swing_quant.screener.core import ScreenResult

_API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_LEN = 4000  # limite do Telegram é 4096


def _money(v: float) -> str:
    return f"{v:,.0f}".replace(",", ".")


def format_message(r: ScreenResult, top_n: int = 10, currency: str = "R$") -> str:
    lines = [
        f"📈 *Swing Quant — {r.market.upper()}* · {r.as_of.date():%d/%m/%Y}",
        f"Patrimônio {currency} {_money(r.equity)} · posições {r.open_positions} · vagas {r.slots}",
    ]
    if r.regime:
        trend = "✅" if r.regime.get("trend_on") else "⚠️"
        vol = "🔥 vol alta" if r.regime.get("high_vol") else "vol normal"
        lines.append(
            f"Regime: tendência {trend} · {vol} · sizing ×{r.regime.get('size_factor', 1):.2f}"
        )
    lines.append("")
    if r.exits.empty:
        lines.append("*Saídas:* nenhuma")
    else:
        lines.append(f"*Saídas ({len(r.exits)}) — vender na abertura:*")
        for row in r.exits.head(top_n).to_dict("records"):
            px = "" if pd.isna(row["ref_price"]) else f" @ {row['ref_price']:.2f}"
            lines.append(
                f"• `{row['ticker']}` {row['qty']} un{px} — {row['reason']} "
                f"({row['strategy']}, {row['bars_held']}d)"
            )
    lines.append("")
    if r.entries.empty:
        lines.append("*Entradas:* nenhuma")
    else:
        lines.append(f"*Entradas ({len(r.entries)}) — comprar na abertura:*")
        for row in r.entries.head(top_n).to_dict("records"):
            stop = (
                ""
                if row["stop_price"] is None or pd.isna(row["stop_price"])
                else f" · stop {row['stop_price']:.2f}"
            )
            hold = f" · máx {row['max_hold']}d" if row["max_hold"] else ""
            lines.append(
                f"• `{row['ticker']}` {row['qty']} un @ ~{row['ref_price']:.2f} "
                f"({currency} {_money(row['notional'])}){stop}{hold} — {row['strategy']} "
                f"score {row['score']:.2f}"
            )
        if len(r.entries) > top_n:
            lines.append(f"… +{len(r.entries) - top_n} candidatas com score menor")
    if r.notes:
        lines.append("")
        lines += [f"_{n}_" for n in r.notes]
    text = "\n".join(lines)
    return text if len(text) <= MAX_LEN else text[: MAX_LEN - 1] + "…"


def format_failure(step: str, error: str, market: str = "") -> str:
    return f"🚨 *Swing Quant falhou* {market.upper()} — etapa `{step}`\n```\n{error[:1500]}\n```"


def send(
    text: str, token: str | None = None, chat_id: str | None = None, timeout: float = 20.0
) -> bool:
    """Envia mensagem (Markdown). Sem token/chat_id (env TELEGRAM_BOT_TOKEN/CHAT_ID) → False."""
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    resp = httpx.post(
        _API.format(token=token),
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return True
