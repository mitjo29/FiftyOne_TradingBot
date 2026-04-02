from data.market import MarketData
from portfolio.models import Position, PortfolioSummary, Trade
from storage.database import Database


class PortfolioManager:
    def __init__(self, db: Database, market: MarketData):
        self.db = db
        self.market = market

    async def buy(self, user_id: int, ticker: str, dollar_amount: float) -> Trade:
        ticker = ticker.upper()

        cash = await self.db.get_cash_balance(user_id)
        if dollar_amount > cash:
            raise ValueError(
                f"Insufficient funds. Available: ${cash:,.2f}, Requested: ${dollar_amount:,.2f}"
            )

        price = await self.market.get_current_price(ticker)
        shares = dollar_amount / price
        total = shares * price

        # Update position with weighted average cost
        existing = await self.db.get_position(user_id, ticker)
        if existing:
            old_shares = existing["shares"]
            old_avg = existing["avg_cost"]
            new_shares = old_shares + shares
            new_avg = (old_shares * old_avg + shares * price) / new_shares
        else:
            new_shares = shares
            new_avg = price

        await self.db.update_position(user_id, ticker, new_shares, new_avg)
        new_cash = cash - total
        await self.db.update_cash_balance(user_id, new_cash)
        await self.db.record_trade(user_id, ticker, "BUY", shares, price, total)

        # Record portfolio snapshot after trade
        await self._record_snapshot(user_id)

        return Trade(ticker=ticker, side="BUY", shares=shares, price=price, total=total)

    async def sell(self, user_id: int, ticker: str, dollar_amount: float) -> Trade:
        ticker = ticker.upper()

        existing = await self.db.get_position(user_id, ticker)
        if not existing or existing["shares"] <= 0:
            raise ValueError(f"No position in {ticker}")

        price = await self.market.get_current_price(ticker)
        shares_to_sell = dollar_amount / price

        if shares_to_sell > existing["shares"]:
            raise ValueError(
                f"Insufficient shares. You have {existing['shares']:.4f} shares "
                f"(${existing['shares'] * price:,.2f}), tried to sell ${dollar_amount:,.2f}"
            )

        total = shares_to_sell * price
        new_shares = existing["shares"] - shares_to_sell

        await self.db.update_position(user_id, ticker, new_shares, existing["avg_cost"])
        cash = await self.db.get_cash_balance(user_id)
        await self.db.update_cash_balance(user_id, cash + total)
        await self.db.record_trade(user_id, ticker, "SELL", shares_to_sell, price, total)

        # Record portfolio snapshot after trade
        await self._record_snapshot(user_id)

        return Trade(ticker=ticker, side="SELL", shares=shares_to_sell, price=price, total=total)

    async def _record_snapshot(self, user_id: int):
        try:
            summary = await self.get_summary(user_id)
            positions_value = sum(p.market_value for p in summary.positions)
            await self.db.record_snapshot(
                user_id, summary.total_value, summary.cash, positions_value
            )
        except Exception:
            pass  # Don't fail trades if snapshot recording fails

    async def get_summary(self, user_id: int) -> PortfolioSummary:
        cash = await self.db.get_cash_balance(user_id)
        raw_positions = await self.db.get_all_positions(user_id)

        positions = []
        if raw_positions:
            tickers = [p["ticker"] for p in raw_positions]
            prices = await self.market.get_batch_prices(tickers)

            for p in raw_positions:
                pos = Position(
                    ticker=p["ticker"],
                    shares=p["shares"],
                    avg_cost=p["avg_cost"],
                )
                current_price = prices.get(p["ticker"], p["avg_cost"])
                pos.update_market_data(current_price)
                positions.append(pos)

        summary = PortfolioSummary(cash=cash, positions=positions)
        summary.calculate_totals()
        return summary
