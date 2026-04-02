from analysis.signals import SignalResult
from portfolio.models import PortfolioSummary, Trade


SIGNAL_EMOJI = {
    "STRONG BUY": "🟢🟢",
    "BUY": "🟢",
    "HOLD": "🟡",
    "SELL": "🔴",
    "STRONG SELL": "🔴🔴",
}


def format_analysis(signal: SignalResult) -> str:
    emoji = SIGNAL_EMOJI.get(signal.overall_signal, "")
    lines = [
        f"<b>{signal.ticker}</b> Analysis",
        f"Price: <b>${signal.current_price:.2f}</b>",
        "",
        f"Signal: {emoji} <b>{signal.overall_signal}</b> (Score: {signal.score:+.2f})",
        "",
        "<b>Indicator Breakdown:</b>",
    ]

    for key, data in signal.details.items():
        name = key.upper()
        score = data["score"]
        detail = data["detail"]
        bar = _score_bar(score)
        lines.append(f"  {name}: {bar} {detail}")

    lines.append("")
    lines.append(f"<i>Updated: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}</i>")
    return "\n".join(lines)


def _score_bar(score: float) -> str:
    filled = int((score + 2) / 4 * 5)
    filled = max(0, min(5, filled))
    return "●" * filled + "○" * (5 - filled)


def format_portfolio(summary: PortfolioSummary) -> str:
    lines = [
        "<b>Portfolio Summary</b>",
        f"Cash: <b>${summary.cash:,.2f}</b>",
        f"Total Value: <b>${summary.total_value:,.2f}</b>",
    ]

    if summary.positions:
        pnl_sign = "+" if summary.total_unrealized_pnl >= 0 else ""
        lines.append(
            f"Unrealized P&L: <b>{pnl_sign}${summary.total_unrealized_pnl:,.2f}</b> "
            f"({pnl_sign}{summary.total_pnl_pct:.1f}%)"
        )
        lines.append("")
        lines.append("<b>Positions:</b>")

        for p in sorted(summary.positions, key=lambda x: x.market_value, reverse=True):
            pnl_sign = "+" if p.unrealized_pnl >= 0 else ""
            lines.append(
                f"  <b>{p.ticker}</b>: {p.shares:.4f} shares @ ${p.avg_cost:.2f}\n"
                f"    Value: ${p.market_value:,.2f} | "
                f"P&L: {pnl_sign}${p.unrealized_pnl:,.2f} ({pnl_sign}{p.pnl_pct:.1f}%)"
            )
    else:
        lines.append("\nNo open positions.")

    return "\n".join(lines)


def format_trade(trade: Trade) -> str:
    emoji = "📈" if trade.side == "BUY" else "📉"
    return (
        f"{emoji} <b>{trade.side}</b> {trade.ticker}\n"
        f"Shares: {trade.shares:.4f} @ ${trade.price:.2f}\n"
        f"Total: ${trade.total:,.2f}"
    )


def format_watchlist(tickers: list[str]) -> str:
    if not tickers:
        return "Your watchlist is empty.\nUse /watch <TICKER> to add stocks."
    lines = ["<b>Your Watchlist:</b>", ""]
    for i, t in enumerate(tickers, 1):
        lines.append(f"  {i}. {t}")
    lines.append(f"\n{len(tickers)} stocks tracked")
    return "\n".join(lines)


def format_alert(signal: SignalResult) -> str:
    emoji = SIGNAL_EMOJI.get(signal.overall_signal, "")
    return (
        f"🔔 <b>Alert: {signal.ticker}</b>\n"
        f"Signal: {emoji} {signal.overall_signal} (Score: {signal.score:+.2f})\n"
        f"Price: ${signal.current_price:.2f}"
    )


def format_help() -> str:
    return (
        "<b>FiftyOne Trading Bot</b>\n\n"
        "<b>Analysis:</b>\n"
        "  /analyze &lt;TICKER&gt; - Full technical analysis with chart\n\n"
        "<b>Watchlist:</b>\n"
        "  /watchlist - View your watchlist\n"
        "  /watch &lt;TICKER&gt; - Add stock to watchlist\n"
        "  /unwatch &lt;TICKER&gt; - Remove from watchlist\n\n"
        "<b>Portfolio:</b>\n"
        "  /portfolio - View portfolio &amp; P&amp;L\n"
        "  /buy &lt;TICKER&gt; &lt;$AMOUNT&gt; - Paper buy\n"
        "  /sell &lt;TICKER&gt; &lt;$AMOUNT&gt; - Paper sell\n\n"
        "<b>Settings:</b>\n"
        "  /alerts - Configure alert notifications\n"
        "  /help - Show this message"
    )
