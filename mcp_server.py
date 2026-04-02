#!/usr/bin/env python3
"""MCP Server for FiftyOne Trading Bot.

Exposes all trading bot capabilities as MCP tools, resources, and prompts.
Connect from Claude Desktop, Claude Code, or any MCP client.

Usage:
    # stdio transport (Claude Desktop / Claude Code)
    python mcp_server.py

    # HTTP transport (for remote clients)
    python mcp_server.py --http --port 8080
"""

import asyncio
import base64
import json
import logging
import sys
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

# Bootstrap project imports
sys.path.insert(0, ".")
from config import DB_PATH, VIRTUAL_CASH
from storage.database import Database
from data.market import MarketData
from portfolio.manager import PortfolioManager
from analysis.pipeline import run_full_analysis
from portfolio.charts import generate_dashboard
from bot.formatters import format_overview

logger = logging.getLogger(__name__)

# Default user ID for MCP (single-user context since MCP has no user sessions)
MCP_USER_ID = 1


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize shared resources on startup, clean up on shutdown."""
    db = Database(DB_PATH)
    await db.init()
    await db.ensure_user(MCP_USER_ID)

    market = MarketData()
    portfolio_mgr = PortfolioManager(db, market)

    ctx = {
        "db": db,
        "market": market,
        "portfolio": portfolio_mgr,
    }
    try:
        yield ctx
    finally:
        await db.close()


mcp = FastMCP(
    "FiftyOne Trading Bot",
    instructions=(
        "A stock trading assistant with advanced analysis: 6 technical indicators "
        "(RSI, MACD, Moving Averages, Bollinger Bands, OBV Volume, Stochastic), "
        "news sentiment scoring, and ML-based price prediction (GradientBoosting). "
        "Includes Fibonacci retracement levels and ATR volatility. "
        "All trades are paper trading with virtual cash. Use the tools to analyze stocks, "
        "execute trades, and monitor portfolio performance."
    ),
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────


@mcp.tool()
async def analyze_stock(ticker: str) -> str:
    """Run full technical analysis on a stock with 8 indicators + sentiment + ML.

    Returns RSI, MACD, Moving Averages, Bollinger Bands, OBV Volume,
    Stochastic Oscillator, News Sentiment, and ML Price Prediction.
    Overall signal: Strong Buy / Buy / Hold / Sell / Strong Sell
    with a score from -2.0 to +2.0.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, TSLA, MSFT)
    """
    ctx = mcp.get_context()
    market: MarketData = ctx["market"]

    signal, _, _ = await run_full_analysis(market, ticker)

    details = []
    for key, data in signal.details.items():
        weight_pct = data.get("weight", 0) * 100
        details.append(f"  {key.upper()} ({weight_pct:.0f}%): score={data['score']}, {data['detail']}")

    text = (
        f"=== Analysis: {signal.ticker} ===\n"
        f"Price: ${signal.current_price:.2f}\n"
        f"Signal: {signal.overall_signal} (score: {signal.score:+.2f})\n"
    )
    if signal.ml_probability is not None:
        text += f"ML Prediction: {signal.ml_probability:.0%} up probability\n"
    text += f"\nIndicators:\n" + "\n".join(details)
    return text


@mcp.tool()
async def analyze_stock_with_chart(ticker: str) -> list:
    """Run full technical analysis with chart (Price+Fibonacci, RSI, Stochastic, Volume/OBV).

    Includes 8 indicators, news sentiment, and ML price prediction.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, TSLA, MSFT)
    """
    from mcp.types import TextContent, ImageContent

    ctx = mcp.get_context()
    market: MarketData = ctx["market"]

    signal, chart_buf, _ = await run_full_analysis(market, ticker)

    details = []
    for key, data in signal.details.items():
        weight_pct = data.get("weight", 0) * 100
        details.append(f"  {key.upper()} ({weight_pct:.0f}%): score={data['score']}, {data['detail']}")

    text = (
        f"=== Analysis: {signal.ticker} ===\n"
        f"Price: ${signal.current_price:.2f}\n"
        f"Signal: {signal.overall_signal} (score: {signal.score:+.2f})\n"
    )
    if signal.ml_probability is not None:
        text += f"ML Prediction: {signal.ml_probability:.0%} up probability\n"
    text += f"\nIndicators:\n" + "\n".join(details)

    chart_b64 = base64.b64encode(chart_buf.getvalue()).decode("utf-8")

    return [
        TextContent(type="text", text=text),
        ImageContent(type="image", data=chart_b64, mimeType="image/png"),
    ]


@mcp.tool()
async def get_stock_price(ticker: str) -> str:
    """Get the current market price of a stock.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, TSLA)
    """
    ctx = mcp.get_context()
    market: MarketData = ctx["market"]
    price = await market.get_current_price(ticker.upper())
    return f"{ticker.upper()}: ${price:.2f}"


@mcp.tool()
async def buy_stock(ticker: str, dollar_amount: float) -> str:
    """Paper-buy a stock with a given dollar amount.

    Deducts cash from virtual portfolio and adds shares at current market price.

    Args:
        ticker: Stock ticker symbol
        dollar_amount: Dollar amount to invest (e.g. 1000 for $1,000)
    """
    ctx = mcp.get_context()
    portfolio_mgr: PortfolioManager = ctx["portfolio"]
    trade = await portfolio_mgr.buy(MCP_USER_ID, ticker.upper(), dollar_amount)
    return (
        f"Bought {trade.shares:.4f} shares of {trade.ticker} "
        f"@ ${trade.price:.2f} for ${trade.total:,.2f}"
    )


@mcp.tool()
async def sell_stock(ticker: str, dollar_amount: float) -> str:
    """Paper-sell a stock position for a given dollar amount.

    Sells shares at current market price and returns cash to portfolio.

    Args:
        ticker: Stock ticker symbol
        dollar_amount: Dollar amount to sell (e.g. 500 for $500)
    """
    ctx = mcp.get_context()
    portfolio_mgr: PortfolioManager = ctx["portfolio"]
    trade = await portfolio_mgr.sell(MCP_USER_ID, ticker.upper(), dollar_amount)
    return (
        f"Sold {trade.shares:.4f} shares of {trade.ticker} "
        f"@ ${trade.price:.2f} for ${trade.total:,.2f}"
    )


@mcp.tool()
async def get_portfolio() -> str:
    """Get the full portfolio summary.

    Returns cash balance, all open positions with current market values,
    unrealized P&L per position, and total portfolio value.
    """
    ctx = mcp.get_context()
    portfolio_mgr: PortfolioManager = ctx["portfolio"]
    summary = await portfolio_mgr.get_summary(MCP_USER_ID)

    lines = [
        f"=== Portfolio Summary ===",
        f"Cash: ${summary.cash:,.2f}",
        f"Total Value: ${summary.total_value:,.2f}",
        f"Total Invested: ${summary.total_invested:,.2f}",
        f"Unrealized P&L: ${summary.total_unrealized_pnl:,.2f} ({summary.total_pnl_pct:+.1f}%)",
    ]
    if summary.positions:
        lines.append("\nPositions:")
        for p in sorted(summary.positions, key=lambda x: x.market_value, reverse=True):
            sign = "+" if p.unrealized_pnl >= 0 else ""
            lines.append(
                f"  {p.ticker}: {p.shares:.4f} shares @ avg ${p.avg_cost:.2f}, "
                f"now ${p.current_price:.2f}, value ${p.market_value:,.2f}, "
                f"P&L {sign}${p.unrealized_pnl:,.2f} ({sign}{p.pnl_pct:.1f}%)"
            )
    else:
        lines.append("\nNo open positions.")
    return "\n".join(lines)


@mcp.tool()
async def get_portfolio_overview() -> str:
    """Get a detailed portfolio overview with allocation breakdown,
    top/worst performers, and recent trade history."""
    ctx = mcp.get_context()
    db: Database = ctx["db"]
    portfolio_mgr: PortfolioManager = ctx["portfolio"]

    summary = await portfolio_mgr.get_summary(MCP_USER_ID)
    trades = await db.get_trades(MCP_USER_ID)
    return format_overview(summary, trades, VIRTUAL_CASH)


@mcp.tool()
async def get_dashboard_chart() -> list:
    """Generate a visual portfolio dashboard with performance chart,
    allocation pie chart, and P&L bar chart. Returns an image."""
    from mcp.types import TextContent, ImageContent

    ctx = mcp.get_context()
    db: Database = ctx["db"]
    portfolio_mgr: PortfolioManager = ctx["portfolio"]

    summary = await portfolio_mgr.get_summary(MCP_USER_ID)
    snapshots = await db.get_snapshots(MCP_USER_ID)
    chart_buf = generate_dashboard(summary, snapshots, VIRTUAL_CASH)

    total_return = summary.total_value - VIRTUAL_CASH
    ret_sign = "+" if total_return >= 0 else ""
    ret_pct = (total_return / VIRTUAL_CASH * 100) if VIRTUAL_CASH > 0 else 0

    chart_b64 = base64.b64encode(chart_buf.getvalue()).decode("utf-8")
    return [
        TextContent(
            type="text",
            text=(
                f"Portfolio Dashboard\n"
                f"Value: ${summary.total_value:,.2f} | "
                f"Return: {ret_sign}${total_return:,.2f} ({ret_sign}{ret_pct:.1f}%)\n"
                f"Positions: {len(summary.positions)} | Cash: ${summary.cash:,.2f}"
            ),
        ),
        ImageContent(type="image", data=chart_b64, mimeType="image/png"),
    ]


@mcp.tool()
async def get_watchlist() -> str:
    """Get the current stock watchlist."""
    ctx = mcp.get_context()
    db: Database = ctx["db"]
    tickers = await db.get_watchlist(MCP_USER_ID)
    if tickers:
        return f"Watchlist ({len(tickers)} stocks): {', '.join(tickers)}"
    return "Watchlist is empty."


@mcp.tool()
async def add_to_watchlist(ticker: str) -> str:
    """Add a stock to the watchlist.

    Args:
        ticker: Stock ticker symbol to watch
    """
    ctx = mcp.get_context()
    db: Database = ctx["db"]
    market: MarketData = ctx["market"]

    ticker = ticker.upper()
    valid = await market.validate_ticker(ticker)
    if not valid:
        return f"Ticker '{ticker}' not found."
    added = await db.add_to_watchlist(MCP_USER_ID, ticker)
    return f"Added {ticker} to watchlist." if added else f"{ticker} already in watchlist."


@mcp.tool()
async def remove_from_watchlist(ticker: str) -> str:
    """Remove a stock from the watchlist.

    Args:
        ticker: Stock ticker symbol to remove
    """
    ctx = mcp.get_context()
    db: Database = ctx["db"]
    removed = await db.remove_from_watchlist(MCP_USER_ID, ticker.upper())
    return f"Removed {ticker.upper()} from watchlist." if removed else f"{ticker.upper()} not in watchlist."


@mcp.tool()
async def get_trade_history(ticker: str | None = None) -> str:
    """Get recent trade history. Optionally filter by a specific stock.

    Args:
        ticker: Optional stock ticker to filter by
    """
    ctx = mcp.get_context()
    db: Database = ctx["db"]
    trades = await db.get_trades(MCP_USER_ID, ticker)
    if not trades:
        return "No trades found."
    lines = ["Recent trades:"]
    for t in trades[:15]:
        lines.append(
            f"  [{t['side']}] {t['ticker']} {t['shares']:.4f} shares "
            f"@ ${t['price']:.2f} = ${t['total']:,.2f} ({t['executed_at']})"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Resources (read-only data exposed to clients)
# ──────────────────────────────────────────────


@mcp.resource("trading://portfolio/summary")
async def portfolio_resource() -> str:
    """Current portfolio summary with positions and P&L."""
    return await get_portfolio()


@mcp.resource("trading://watchlist")
async def watchlist_resource() -> str:
    """Current watchlist of tracked stocks."""
    return await get_watchlist()


# ──────────────────────────────────────────────
# Prompts (reusable prompt templates)
# ──────────────────────────────────────────────


@mcp.prompt()
async def stock_analysis(ticker: str) -> str:
    """Analyze a stock and provide a trading recommendation.

    Args:
        ticker: Stock ticker to analyze
    """
    return (
        f"Please analyze {ticker.upper()} using the analyze_stock tool. "
        f"Then check my current portfolio with get_portfolio. "
        f"Based on the analysis and my current holdings, give me a clear recommendation: "
        f"should I buy, hold, or sell {ticker.upper()}? "
        f"If buying, suggest a dollar amount based on my available cash. "
        f"Keep the response concise."
    )


@mcp.prompt()
async def portfolio_review() -> str:
    """Review portfolio performance and suggest optimizations."""
    return (
        "Please review my portfolio:\n"
        "1. Use get_portfolio to see all my positions\n"
        "2. Use get_trade_history to see recent activity\n"
        "3. For each position, use analyze_stock to check current signals\n"
        "4. Identify any positions I should consider selling (weak signals)\n"
        "5. Suggest any new stocks to consider based on strong buy signals\n"
        "6. Provide an overall assessment of my portfolio health\n"
        "Keep it actionable and data-driven."
    )


@mcp.prompt()
async def market_scan(sector: str = "tech") -> str:
    """Scan popular stocks in a sector for trading opportunities.

    Args:
        sector: Market sector to scan (tech, healthcare, finance, energy, consumer)
    """
    sector_tickers = {
        "tech": "AAPL, MSFT, GOOG, AMZN, NVDA, META, TSLA",
        "healthcare": "JNJ, UNH, PFE, ABBV, MRK, TMO, LLY",
        "finance": "JPM, BAC, GS, MS, WFC, BLK, AXP",
        "energy": "XOM, CVX, COP, SLB, EOG, PXD, MPC",
        "consumer": "WMT, PG, KO, PEP, COST, NKE, MCD",
    }
    tickers = sector_tickers.get(sector.lower(), sector_tickers["tech"])
    return (
        f"Scan these {sector} stocks for trading opportunities: {tickers}\n"
        f"For each one, use analyze_stock to check the signal.\n"
        f"Then rank them from best to worst opportunity.\n"
        f"Highlight any Strong Buy or Strong Sell signals.\n"
        f"Suggest the top 2-3 stocks to consider trading."
    )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FiftyOne Trading Bot MCP Server")
    parser.add_argument("--http", action="store_true", help="Run HTTP transport instead of stdio")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    args = parser.parse_args()

    if args.http:
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
