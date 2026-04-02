from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    pnl_pct: float = 0.0

    def update_market_data(self, price: float):
        self.current_price = price
        self.market_value = self.shares * price
        cost_basis = self.shares * self.avg_cost
        self.unrealized_pnl = self.market_value - cost_basis
        self.pnl_pct = (self.unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0


@dataclass
class Trade:
    ticker: str
    side: str
    shares: float
    price: float
    total: float
    executed_at: datetime | None = None
    id: int | None = None


@dataclass
class PortfolioSummary:
    cash: float
    positions: list[Position] = field(default_factory=list)
    total_value: float = 0.0
    total_invested: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_pnl_pct: float = 0.0

    def calculate_totals(self):
        positions_value = sum(p.market_value for p in self.positions)
        self.total_invested = sum(p.shares * p.avg_cost for p in self.positions)
        self.total_value = self.cash + positions_value
        self.total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions)
        self.total_pnl_pct = (
            (self.total_unrealized_pnl / self.total_invested * 100)
            if self.total_invested > 0
            else 0.0
        )
