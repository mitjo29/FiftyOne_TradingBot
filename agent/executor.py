"""Executes tool calls by delegating to existing trading bot modules."""

import json
import logging

from analysis.pipeline import run_full_analysis
from data.market import MarketData
from portfolio.manager import PortfolioManager
from storage.database import Database

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, db: Database, market: MarketData, portfolio_mgr: PortfolioManager):
        self.db = db
        self.market = market
        self.portfolio_mgr = portfolio_mgr

    async def execute(self, tool_name: str, tool_input: dict, user_id: int) -> dict:
        """Execute a tool and return result dict with 'text' and optional 'chart' (BytesIO)."""
        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if not handler:
                return {"text": f"Unknown tool: {tool_name}"}
            return await handler(tool_input, user_id)
        except ValueError as e:
            return {"text": f"Error: {e}"}
        except Exception as e:
            logger.exception(f"Tool execution error: {tool_name}")
            return {"text": f"Tool failed: {e}"}

    async def _tool_analyze_stock(self, inputs: dict, user_id: int) -> dict:
        ticker = inputs["ticker"].upper()
        signal, chart_buf, _ = await run_full_analysis(self.market, ticker)

        details = []
        for key, data in signal.details.items():
            weight_pct = data.get("weight", 0) * 100
            details.append(f"  {key.upper()} ({weight_pct:.0f}%): score={data['score']}, {data['detail']}")

        text = (
            f"Analysis for {signal.ticker}:\n"
            f"Price: ${signal.current_price:.2f}\n"
            f"Signal: {signal.overall_signal} (score: {signal.score:+.2f})\n"
        )
        if signal.ml_probability is not None:
            text += f"ML Prediction: {signal.ml_probability:.0%} up probability\n"
        text += f"Indicators:\n" + "\n".join(details)
        return {"text": text, "chart": chart_buf}

    async def _tool_buy_stock(self, inputs: dict, user_id: int) -> dict:
        ticker = inputs["ticker"].upper()
        amount = float(inputs["dollar_amount"])
        trade = await self.portfolio_mgr.buy(user_id, ticker, amount)
        return {
            "text": (
                f"Bought {trade.shares:.4f} shares of {trade.ticker} "
                f"@ ${trade.price:.2f} for ${trade.total:,.2f}"
            )
        }

    async def _tool_sell_stock(self, inputs: dict, user_id: int) -> dict:
        ticker = inputs["ticker"].upper()
        amount = float(inputs["dollar_amount"])
        trade = await self.portfolio_mgr.sell(user_id, ticker, amount)
        return {
            "text": (
                f"Sold {trade.shares:.4f} shares of {trade.ticker} "
                f"@ ${trade.price:.2f} for ${trade.total:,.2f}"
            )
        }

    async def _tool_get_portfolio(self, inputs: dict, user_id: int) -> dict:
        summary = await self.portfolio_mgr.get_summary(user_id)
        lines = [
            f"Cash: ${summary.cash:,.2f}",
            f"Total Value: ${summary.total_value:,.2f}",
            f"Unrealized P&L: ${summary.total_unrealized_pnl:,.2f} ({summary.total_pnl_pct:+.1f}%)",
        ]
        if summary.positions:
            lines.append("Positions:")
            for p in sorted(summary.positions, key=lambda x: x.market_value, reverse=True):
                sign = "+" if p.unrealized_pnl >= 0 else ""
                lines.append(
                    f"  {p.ticker}: {p.shares:.4f} shares @ avg ${p.avg_cost:.2f}, "
                    f"now ${p.current_price:.2f}, value ${p.market_value:,.2f}, "
                    f"P&L {sign}${p.unrealized_pnl:,.2f} ({sign}{p.pnl_pct:.1f}%)"
                )
        else:
            lines.append("No open positions.")
        return {"text": "\n".join(lines)}

    async def _tool_get_watchlist(self, inputs: dict, user_id: int) -> dict:
        tickers = await self.db.get_watchlist(user_id)
        if tickers:
            return {"text": f"Watchlist: {', '.join(tickers)} ({len(tickers)} stocks)"}
        return {"text": "Watchlist is empty."}

    async def _tool_add_to_watchlist(self, inputs: dict, user_id: int) -> dict:
        ticker = inputs["ticker"].upper()
        valid = await self.market.validate_ticker(ticker)
        if not valid:
            return {"text": f"Ticker '{ticker}' not found."}
        added = await self.db.add_to_watchlist(user_id, ticker)
        if added:
            return {"text": f"Added {ticker} to watchlist."}
        return {"text": f"{ticker} is already in the watchlist."}

    async def _tool_remove_from_watchlist(self, inputs: dict, user_id: int) -> dict:
        ticker = inputs["ticker"].upper()
        removed = await self.db.remove_from_watchlist(user_id, ticker)
        if removed:
            return {"text": f"Removed {ticker} from watchlist."}
        return {"text": f"{ticker} is not in the watchlist."}

    async def _tool_get_stock_price(self, inputs: dict, user_id: int) -> dict:
        ticker = inputs["ticker"].upper()
        price = await self.market.get_current_price(ticker)
        return {"text": f"{ticker} current price: ${price:.2f}"}

    async def _tool_get_trade_history(self, inputs: dict, user_id: int) -> dict:
        ticker = inputs.get("ticker")
        trades = await self.db.get_trades(user_id, ticker)
        if not trades:
            return {"text": "No trades found."}
        lines = ["Recent trades:"]
        for t in trades[:10]:
            lines.append(
                f"  [{t['side']}] {t['ticker']} {t['shares']:.4f} shares "
                f"@ ${t['price']:.2f} = ${t['total']:,.2f} ({t['executed_at']})"
            )
        return {"text": "\n".join(lines)}
