import aiosqlite
from config import DB_PATH, VIRTUAL_CASH


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._create_tables()

    async def close(self):
        if self._db:
            await self._db.close()

    async def _create_tables(self):
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS watchlist (
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, ticker)
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                shares REAL NOT NULL DEFAULT 0,
                avg_cost REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, ticker)
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
                shares REAL NOT NULL,
                price REAL NOT NULL,
                total REAL NOT NULL,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                cash_balance REAL NOT NULL,
                alert_enabled INTEGER DEFAULT 1,
                alert_threshold TEXT DEFAULT 'strong_buy,strong_sell'
            );
        """)
        await self._db.commit()

    # --- User Settings ---

    async def ensure_user(self, user_id: int):
        async with self._db.execute(
            "SELECT user_id FROM user_settings WHERE user_id = ?", (user_id,)
        ) as cursor:
            if not await cursor.fetchone():
                await self._db.execute(
                    "INSERT INTO user_settings (user_id, cash_balance) VALUES (?, ?)",
                    (user_id, VIRTUAL_CASH),
                )
                await self._db.commit()

    async def get_cash_balance(self, user_id: int) -> float:
        await self.ensure_user(user_id)
        async with self._db.execute(
            "SELECT cash_balance FROM user_settings WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row["cash_balance"]

    async def update_cash_balance(self, user_id: int, new_balance: float):
        await self._db.execute(
            "UPDATE user_settings SET cash_balance = ? WHERE user_id = ?",
            (new_balance, user_id),
        )
        await self._db.commit()

    async def get_alert_settings(self, user_id: int) -> dict:
        await self.ensure_user(user_id)
        async with self._db.execute(
            "SELECT alert_enabled, alert_threshold FROM user_settings WHERE user_id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return {
                "enabled": bool(row["alert_enabled"]),
                "threshold": row["alert_threshold"],
            }

    async def update_alert_settings(self, user_id: int, enabled: bool, threshold: str):
        await self._db.execute(
            "UPDATE user_settings SET alert_enabled = ?, alert_threshold = ? WHERE user_id = ?",
            (int(enabled), threshold, user_id),
        )
        await self._db.commit()

    # --- Watchlist ---

    async def add_to_watchlist(self, user_id: int, ticker: str) -> bool:
        try:
            await self._db.execute(
                "INSERT INTO watchlist (user_id, ticker) VALUES (?, ?)",
                (user_id, ticker.upper()),
            )
            await self._db.commit()
            return True
        except aiosqlite.IntegrityError:
            return False

    async def remove_from_watchlist(self, user_id: int, ticker: str) -> bool:
        cursor = await self._db.execute(
            "DELETE FROM watchlist WHERE user_id = ? AND ticker = ?",
            (user_id, ticker.upper()),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def get_watchlist(self, user_id: int) -> list[str]:
        async with self._db.execute(
            "SELECT ticker FROM watchlist WHERE user_id = ? ORDER BY added_at",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [row["ticker"] for row in rows]

    async def get_watchlist_count(self, user_id: int) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) as cnt FROM watchlist WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row["cnt"]

    async def get_all_watchlists(self) -> dict[int, list[str]]:
        result: dict[int, list[str]] = {}
        async with self._db.execute(
            "SELECT user_id, ticker FROM watchlist ORDER BY user_id"
        ) as cursor:
            async for row in cursor:
                uid = row["user_id"]
                if uid not in result:
                    result[uid] = []
                result[uid].append(row["ticker"])
        return result

    # --- Portfolio ---

    async def get_position(self, user_id: int, ticker: str):
        async with self._db.execute(
            "SELECT shares, avg_cost FROM portfolio WHERE user_id = ? AND ticker = ?",
            (user_id, ticker.upper()),
        ) as cursor:
            return await cursor.fetchone()

    async def get_all_positions(self, user_id: int) -> list[dict]:
        async with self._db.execute(
            "SELECT ticker, shares, avg_cost FROM portfolio WHERE user_id = ? AND shares > 0",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {"ticker": r["ticker"], "shares": r["shares"], "avg_cost": r["avg_cost"]}
                for r in rows
            ]

    async def update_position(self, user_id: int, ticker: str, shares: float, avg_cost: float):
        ticker = ticker.upper()
        if shares <= 0:
            await self._db.execute(
                "DELETE FROM portfolio WHERE user_id = ? AND ticker = ?",
                (user_id, ticker),
            )
        else:
            await self._db.execute(
                """INSERT INTO portfolio (user_id, ticker, shares, avg_cost)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, ticker) DO UPDATE SET shares = ?, avg_cost = ?""",
                (user_id, ticker, shares, avg_cost, shares, avg_cost),
            )
        await self._db.commit()

    # --- Trades ---

    async def record_trade(
        self, user_id: int, ticker: str, side: str, shares: float, price: float, total: float
    ):
        await self._db.execute(
            "INSERT INTO trades (user_id, ticker, side, shares, price, total) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, ticker.upper(), side, shares, price, total),
        )
        await self._db.commit()

    async def get_trades(self, user_id: int, ticker: str | None = None) -> list[dict]:
        if ticker:
            query = "SELECT * FROM trades WHERE user_id = ? AND ticker = ? ORDER BY executed_at DESC"
            params = (user_id, ticker.upper())
        else:
            query = "SELECT * FROM trades WHERE user_id = ? ORDER BY executed_at DESC LIMIT 20"
            params = (user_id,)
        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "ticker": r["ticker"],
                    "side": r["side"],
                    "shares": r["shares"],
                    "price": r["price"],
                    "total": r["total"],
                    "executed_at": r["executed_at"],
                }
                for r in rows
            ]
