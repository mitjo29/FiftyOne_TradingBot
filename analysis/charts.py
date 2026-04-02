from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from analysis.signals import SignalResult


SIGNAL_COLORS = {
    "STRONG BUY": "#00e676",
    "BUY": "#66bb6a",
    "HOLD": "#ffa726",
    "SELL": "#ef5350",
    "STRONG SELL": "#d50000",
}

FIB_COLORS = {
    0.236: "#ffeb3b",
    0.382: "#ff9800",
    0.5: "#f44336",
    0.618: "#e91e63",
    0.786: "#9c27b0",
}


def generate_chart(
    ticker: str,
    df: pd.DataFrame,
    indicators: dict,
    signal_result: SignalResult,
) -> BytesIO:
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(12, 12),
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    dates = df.index

    # ── Panel 1: Price + Bollinger + MAs + Fibonacci ──

    ax1.plot(dates, df["Close"], color="#42a5f5", linewidth=1.5, label="Close")

    bb = indicators.get("bb", {})
    if bb.get("upper") is not None and not bb["upper"].empty:
        ax1.fill_between(
            dates, bb["lower"], bb["upper"], alpha=0.1, color="#7e57c2", label="Bollinger Bands"
        )
        ax1.plot(dates, bb["upper"], color="#7e57c2", linewidth=0.5, alpha=0.6)
        ax1.plot(dates, bb["lower"], color="#7e57c2", linewidth=0.5, alpha=0.6)

    ma = indicators.get("ma", {})
    if ma.get("sma_20") is not None and not ma["sma_20"].empty:
        ax1.plot(dates, ma["sma_20"], color="#ffeb3b", linewidth=0.8, alpha=0.7, label="SMA 20")
    if ma.get("sma_50") is not None and not ma["sma_50"].empty:
        ax1.plot(dates, ma["sma_50"], color="#ff9800", linewidth=0.8, alpha=0.7, label="SMA 50")

    # Fibonacci levels
    fib = indicators.get("fibonacci")
    if fib:
        for ratio, price in fib["levels"].items():
            if ratio in (0.0, 1.0):
                continue
            color = FIB_COLORS.get(ratio, "#9e9e9e")
            ax1.axhline(y=price, color=color, linestyle="--", linewidth=0.6, alpha=0.5)
            ax1.text(
                dates[0], price, f" Fib {ratio:.1%}", fontsize=7,
                color=color, alpha=0.7, va="center",
            )

    signal_color = SIGNAL_COLORS.get(signal_result.overall_signal, "#ffffff")
    title = (
        f"{ticker.upper()} - ${signal_result.current_price:.2f}  |  "
        f"Signal: {signal_result.overall_signal} ({signal_result.score:+.2f})"
    )
    if signal_result.ml_probability is not None:
        title += f"  |  ML: {signal_result.ml_probability:.0%} up"

    ax1.set_title(title, fontsize=13, fontweight="bold", color=signal_color)
    ax1.legend(loc="upper left", fontsize=7)
    ax1.set_ylabel("Price ($)")
    ax1.grid(alpha=0.2)

    # ── Panel 2: RSI ──

    rsi = indicators.get("rsi")
    if rsi is not None and not rsi.empty:
        ax2.plot(dates, rsi, color="#42a5f5", linewidth=1.2)
        ax2.axhline(y=70, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.axhline(y=30, color="#66bb6a", linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.axhline(y=50, color="#9e9e9e", linestyle=":", linewidth=0.5, alpha=0.5)
        ax2.fill_between(dates, 70, 100, alpha=0.05, color="#ef5350")
        ax2.fill_between(dates, 0, 30, alpha=0.05, color="#66bb6a")
        ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", fontsize=9)
    ax2.grid(alpha=0.2)

    # ── Panel 3: Stochastic ──

    stoch = indicators.get("stochastic", {})
    if stoch.get("slowk") is not None and not stoch["slowk"].empty:
        ax3.plot(dates, stoch["slowk"], color="#42a5f5", linewidth=1.2, label="%K")
        ax3.plot(dates, stoch["slowd"], color="#ffa726", linewidth=1.0, alpha=0.8, label="%D")
        ax3.axhline(y=80, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.7)
        ax3.axhline(y=20, color="#66bb6a", linestyle="--", linewidth=0.8, alpha=0.7)
        ax3.fill_between(dates, 80, 100, alpha=0.05, color="#ef5350")
        ax3.fill_between(dates, 0, 20, alpha=0.05, color="#66bb6a")
        ax3.set_ylim(0, 100)
        ax3.legend(loc="upper left", fontsize=7)
    ax3.set_ylabel("Stoch", fontsize=9)
    ax3.grid(alpha=0.2)

    # ── Panel 4: Volume + OBV ──

    volume = indicators.get("volume")
    if volume is not None and not volume.empty:
        close_vals = df["Close"].values
        colors = [
            "#66bb6a" if i == 0 or close_vals[i] >= close_vals[i - 1] else "#ef5350"
            for i in range(len(close_vals))
        ]
        ax4.bar(dates, volume, color=colors, alpha=0.6, width=0.8)
        ax4.set_ylabel("Volume", fontsize=9, color="#9e9e9e")

        # OBV overlay on secondary axis
        obv = indicators.get("obv")
        if obv is not None and not obv.empty:
            ax4b = ax4.twinx()
            ax4b.plot(dates, obv, color="#ab47bc", linewidth=1.0, alpha=0.8, label="OBV")
            ax4b.set_ylabel("OBV", fontsize=9, color="#ab47bc")
            ax4b.tick_params(axis="y", colors="#ab47bc")
            ax4b.legend(loc="upper left", fontsize=7)

    ax4.set_xlabel("Date")
    ax4.grid(alpha=0.2)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax4.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf
