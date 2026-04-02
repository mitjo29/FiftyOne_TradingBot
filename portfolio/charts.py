from io import BytesIO
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from portfolio.models import PortfolioSummary


COLORS = [
    "#42a5f5", "#66bb6a", "#ffa726", "#ef5350", "#ab47bc",
    "#26c6da", "#ffee58", "#8d6e63", "#78909c", "#ec407a",
]


def generate_dashboard(
    summary: PortfolioSummary,
    snapshots: list[dict],
    initial_cash: float,
) -> BytesIO:
    plt.style.use("dark_background")

    has_snapshots = len(snapshots) >= 2
    has_positions = len(summary.positions) > 0
    num_plots = 1 + int(has_positions) + int(has_snapshots)

    if num_plots == 3:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        ax_value, ax_alloc, ax_pnl = axes
    elif num_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        if has_snapshots and has_positions:
            ax_value, ax_alloc = axes
            ax_pnl = None
        elif has_snapshots:
            ax_value = axes[0]
            ax_alloc = None
            ax_pnl = axes[1]
        else:
            ax_value = None
            ax_alloc, ax_pnl = axes
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        if has_snapshots:
            ax_value = ax
        elif has_positions:
            ax_alloc = ax
        else:
            ax_value = ax
        ax_alloc = ax_alloc if has_positions else None
        ax_pnl = None
        ax_value = ax_value if has_snapshots else None

    fig.suptitle(
        f"Portfolio Dashboard  |  Total: ${summary.total_value:,.2f}",
        fontsize=15, fontweight="bold", color="#42a5f5",
    )

    # --- Panel 1: Portfolio Value Over Time ---
    if has_snapshots and ax_value is not None:
        dates = []
        values = []
        for s in snapshots:
            try:
                dt = datetime.fromisoformat(s["recorded_at"])
            except (ValueError, TypeError):
                dt = datetime.strptime(str(s["recorded_at"]), "%Y-%m-%d %H:%M:%S")
            dates.append(dt)
            values.append(s["total_value"])

        ax_value.fill_between(dates, values, alpha=0.2, color="#42a5f5")
        ax_value.plot(dates, values, color="#42a5f5", linewidth=2)
        ax_value.axhline(y=initial_cash, color="#9e9e9e", linestyle="--", linewidth=0.8, alpha=0.5)

        # Color the area based on profit/loss relative to initial
        for i in range(len(values)):
            if values[i] >= initial_cash:
                ax_value.fill_between(
                    dates[max(0, i - 1):i + 1],
                    initial_cash,
                    values[max(0, i - 1):i + 1],
                    alpha=0.15, color="#66bb6a"
                )
            else:
                ax_value.fill_between(
                    dates[max(0, i - 1):i + 1],
                    values[max(0, i - 1):i + 1],
                    initial_cash,
                    alpha=0.15, color="#ef5350"
                )

        ax_value.set_title("Portfolio Value", fontsize=11)
        ax_value.set_ylabel("Value ($)")
        ax_value.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax_value.tick_params(axis="x", rotation=45)
        ax_value.grid(alpha=0.2)

        # Show total return
        total_return = ((summary.total_value - initial_cash) / initial_cash) * 100
        ret_color = "#66bb6a" if total_return >= 0 else "#ef5350"
        ret_sign = "+" if total_return >= 0 else ""
        ax_value.text(
            0.02, 0.95, f"{ret_sign}{total_return:.1f}%",
            transform=ax_value.transAxes, fontsize=14, fontweight="bold",
            color=ret_color, va="top",
        )

    # --- Panel 2: Allocation Pie Chart ---
    if has_positions and ax_alloc is not None:
        labels = []
        sizes = []
        colors = []

        for i, p in enumerate(sorted(summary.positions, key=lambda x: x.market_value, reverse=True)):
            labels.append(f"{p.ticker}\n${p.market_value:,.0f}")
            sizes.append(p.market_value)
            colors.append(COLORS[i % len(COLORS)])

        if summary.cash > 0:
            labels.append(f"Cash\n${summary.cash:,.0f}")
            sizes.append(summary.cash)
            colors.append("#9e9e9e")

        wedges, texts = ax_alloc.pie(
            sizes, labels=labels, colors=colors,
            startangle=90, textprops={"fontsize": 8},
        )
        ax_alloc.set_title("Allocation", fontsize=11)

    # --- Panel 3: P&L Bar Chart ---
    if has_positions and num_plots == 3 and ax_pnl is not None:
        positions_sorted = sorted(summary.positions, key=lambda x: x.unrealized_pnl, reverse=True)
        tickers = [p.ticker for p in positions_sorted]
        pnls = [p.unrealized_pnl for p in positions_sorted]
        bar_colors = ["#66bb6a" if pnl >= 0 else "#ef5350" for pnl in pnls]

        bars = ax_pnl.barh(tickers, pnls, color=bar_colors, height=0.6)
        ax_pnl.axvline(x=0, color="#9e9e9e", linewidth=0.8)
        ax_pnl.set_title("Unrealized P&L", fontsize=11)
        ax_pnl.set_xlabel("P&L ($)")
        ax_pnl.grid(alpha=0.2, axis="x")

        for bar, pnl, pos in zip(bars, pnls, positions_sorted):
            sign = "+" if pnl >= 0 else ""
            ax_pnl.text(
                bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {sign}${pnl:,.0f} ({sign}{pos.pnl_pct:.1f}%)",
                va="center", fontsize=8,
                color="#66bb6a" if pnl >= 0 else "#ef5350",
            )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf
