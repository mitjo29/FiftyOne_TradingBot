from analysis.signals import SignalResult
from analysis.recommendation import TradeRecommendation
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
    ]

    if signal.ml_probability is not None:
        ml_dir = "Up" if signal.ml_probability > 0.5 else "Down"
        lines.append(f"ML Prediction: <b>{signal.ml_probability:.0%}</b> {ml_dir}")

    lines.append("")
    lines.append("<b>Indicator Breakdown:</b>")

    for key, data in signal.details.items():
        name = key.upper()
        score = data["score"]
        detail = data["detail"]
        weight_pct = data.get("weight", 0) * 100
        bar = _score_bar(score)
        lines.append(f"  {name} ({weight_pct:.0f}%): {bar} {detail}")

    # Trade recommendation
    if signal.recommendation:
        lines.append("")
        lines.append(_format_recommendation(signal.recommendation))

    lines.append("")
    lines.append(f"<i>Updated: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}</i>")
    return "\n".join(lines)


def _format_recommendation(rec: TradeRecommendation) -> str:
    """Format a trade recommendation for Telegram."""
    if rec.action == "HOLD":
        return (
            "<b>Recommendation: HOLD</b>\n"
            f"  No trade — signal is neutral\n"
            f"  Watch support: ${rec.stop_loss:,.2f}\n"
            f"  Watch resistance: ${rec.take_profit_1:,.2f}"
        )

    # Action header
    action_emoji = "🟢 BUY" if rec.action == "BUY" else "🔴 SELL"
    lines = [
        f"<b>Trade Recommendation:</b>",
        f"  {action_emoji} — {rec.confidence} confidence",
        "",
    ]

    # Order details
    lines.append(f"  <b>Order:</b> {rec.order_type}")
    if rec.order_type == "Market":
        lines.append(f"  Entry: ${rec.entry_price:,.2f} (at market)")
    elif rec.order_type == "Limit":
        lines.append(f"  Limit price: <b>${rec.limit_price:,.2f}</b>")
    elif rec.order_type == "Stop-Limit":
        lines.append(f"  Trigger: ${rec.entry_price:,.2f}")
        lines.append(f"  Limit: <b>${rec.limit_price:,.2f}</b>")

    # Position size
    lines.append(f"  <b>Size:</b> ${rec.position_size_usd:,.0f} ({rec.position_size_pct:.0f}% of portfolio)")

    # Stop-loss and take-profit
    lines.append("")
    lines.append(f"  🛑 Stop-loss: <b>${rec.stop_loss:,.2f}</b> (risk: ${rec.risk_per_share:.2f}/share)")
    lines.append(f"  🎯 Target 1: <b>${rec.take_profit_1:,.2f}</b> (conservative)")
    lines.append(f"  🎯 Target 2: <b>${rec.take_profit_2:,.2f}</b> (aggressive)")
    lines.append(f"  ⚖️ Risk/Reward: <b>{rec.risk_reward_ratio}:1</b>")

    # Notes
    if rec.notes:
        lines.append("")
        for note in rec.notes:
            lines.append(f"  ℹ️ {note}")

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
    lines = [
        f"🔔 <b>Alert: {signal.ticker}</b>",
        f"Signal: {emoji} {signal.overall_signal} (Score: {signal.score:+.2f})",
        f"Price: ${signal.current_price:.2f}",
    ]
    rec = signal.recommendation
    if rec and rec.action != "HOLD":
        lines.append("")
        lines.append(f"<b>{rec.action}</b> — {rec.order_type}")
        if rec.limit_price:
            lines.append(f"Limit: ${rec.limit_price:,.2f}")
        lines.append(f"Size: ${rec.position_size_usd:,.0f} | SL: ${rec.stop_loss:,.2f} | TP: ${rec.take_profit_1:,.2f}")
    return "\n".join(lines)


def format_overview(summary: PortfolioSummary, trades: list[dict], initial_cash: float) -> str:
    total_return = summary.total_value - initial_cash
    total_return_pct = (total_return / initial_cash * 100) if initial_cash > 0 else 0
    ret_sign = "+" if total_return >= 0 else ""

    lines = [
        "<b>Portfolio Overview</b>",
        "",
        f"Total Value:  <b>${summary.total_value:,.2f}</b>",
        f"Cash:         ${summary.cash:,.2f}",
        f"Invested:     ${summary.total_invested:,.2f}",
        f"Total Return: <b>{ret_sign}${total_return:,.2f} ({ret_sign}{total_return_pct:.1f}%)</b>",
        "",
    ]

    # Allocation breakdown
    if summary.positions:
        lines.append("<b>Allocation:</b>")
        total = summary.total_value
        for p in sorted(summary.positions, key=lambda x: x.market_value, reverse=True):
            pct = (p.market_value / total * 100) if total > 0 else 0
            bar = _alloc_bar(pct)
            lines.append(f"  {p.ticker}: {bar} {pct:.1f}% (${p.market_value:,.0f})")

        if summary.cash > 0:
            cash_pct = (summary.cash / total * 100) if total > 0 else 0
            bar = _alloc_bar(cash_pct)
            lines.append(f"  Cash: {bar} {cash_pct:.1f}% (${summary.cash:,.0f})")

        # Top / worst performers
        lines.append("")
        sorted_by_pnl = sorted(summary.positions, key=lambda x: x.pnl_pct, reverse=True)
        if sorted_by_pnl:
            best = sorted_by_pnl[0]
            worst = sorted_by_pnl[-1]
            lines.append(
                f"Best:  <b>{best.ticker}</b> {'+' if best.pnl_pct >= 0 else ''}{best.pnl_pct:.1f}%"
            )
            if len(sorted_by_pnl) > 1:
                lines.append(
                    f"Worst: <b>{worst.ticker}</b> {'+' if worst.pnl_pct >= 0 else ''}{worst.pnl_pct:.1f}%"
                )

    # Recent trades
    if trades:
        lines.append("")
        lines.append("<b>Recent Trades:</b>")
        for t in trades[:5]:
            emoji = "B" if t["side"] == "BUY" else "S"
            lines.append(
                f"  [{emoji}] {t['ticker']} {t['shares']:.2f} @ ${t['price']:.2f} = ${t['total']:,.2f}"
            )

    return "\n".join(lines)


def _alloc_bar(pct: float) -> str:
    filled = int(pct / 10)
    filled = max(0, min(10, filled))
    return "█" * filled + "░" * (10 - filled)


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
        "  /overview - Detailed portfolio overview\n"
        "  /dashboard - Visual performance dashboard\n"
        "  /buy &lt;TICKER&gt; &lt;$AMOUNT&gt; - Paper buy\n"
        "  /sell &lt;TICKER&gt; &lt;$AMOUNT&gt; - Paper sell\n\n"
        "<b>Settings:</b>\n"
        "  /alerts - Configure alert notifications\n"
        "  /clear - Reset AI conversation\n"
        "  /help - Show this message\n\n"
        "💬 <b>AI Chat:</b> Just type naturally!\n"
        "  \"What do you think about Tesla?\"\n"
        "  \"Buy $2000 of Apple\"\n"
        "  \"How's my portfolio doing?\"\n"
        "  \"Compare MSFT and GOOG\""
    )
