import asyncio
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketData:
    async def get_stock_data(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        def _fetch():
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            return data

        df = await asyncio.to_thread(_fetch)
        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'")
        return df

    async def get_current_price(self, ticker: str) -> float:
        def _fetch():
            t = yf.Ticker(ticker)
            price = t.fast_info.get("lastPrice")
            if price is None:
                hist = t.history(period="1d")
                if hist.empty:
                    raise ValueError(f"Cannot get price for '{ticker}'")
                price = hist["Close"].iloc[-1]
            return float(price)

        return await asyncio.to_thread(_fetch)

    async def get_batch_prices(self, tickers: list[str]) -> dict[str, float]:
        if not tickers:
            return {}

        def _fetch():
            data = yf.download(
                " ".join(tickers), period="1d", interval="1d", progress=False
            )
            prices = {}
            if len(tickers) == 1:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                if not data.empty:
                    prices[tickers[0]] = float(data["Close"].iloc[-1])
            else:
                for t in tickers:
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
                            prices[t] = float(data["Close"][t].iloc[-1])
                        else:
                            prices[t] = float(data["Close"].iloc[-1])
                    except (KeyError, IndexError):
                        logger.warning(f"Could not get price for {t}")
            return prices

        return await asyncio.to_thread(_fetch)

    async def validate_ticker(self, ticker: str) -> bool:
        def _check():
            t = yf.Ticker(ticker)
            info = t.fast_info
            return info.get("lastPrice") is not None

        try:
            return await asyncio.to_thread(_check)
        except Exception:
            return False
