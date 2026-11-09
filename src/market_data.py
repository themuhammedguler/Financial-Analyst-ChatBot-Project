"""Borsa İstanbul Canlı Piyasa ve Fiyat Grafiği Modülü.

Bu modül, Yahoo Finance (BIST .IS sembolleri) üzerinden hisse senetlerinin
anlık fiyatlarını, piyasa değerini, değerleme rasyolarını (F/K, PD/DD),
teknik hareketli ortalamalarını (SMA 20/50/200) ve RSI göstergesini çeker;
Plotly tabanlı kurumsal karanlık temalı interaktif Mum Grafiği (Candlestick) üretir.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.network import ensure_ascii_ca_bundle

logger = logging.getLogger(__name__)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Göreceli Güç Endeksi (RSI) hesaplar."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def fetch_ticker_market_data(
    ticker: str,
    period: str = "1y"
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Belirtilen BIST hissesi için Yahoo Finance'den geçmiş fiyat serisini ve piyasa metriklerini çeker.
    """
    ensure_ascii_ca_bundle()
    import yfinance as yf

    clean_sym = ticker.upper().replace(".IS", "").strip()
    norm_sym = f"{clean_sym}.IS"

    metrics: Dict[str, Any] = {
        "ticker": clean_sym,
        "symbol": norm_sym,
        "current_price": None,
        "previous_close": None,
        "change_val": None,
        "change_pct": None,
        "day_low": None,
        "day_high": None,
        "fifty_two_week_low": None,
        "fifty_two_week_high": None,
        "volume": None,
        "market_cap_str": "Veri Yok",
        "pe_ratio": None,
        "pb_ratio": None,
        "rsi_14": None,
        "sma_50": None,
        "sma_200": None,
        "status": "ok"
    }

    try:
        t = yf.Ticker(norm_sym)
        df = t.history(period=period)
        if df is None or df.empty:
            metrics["status"] = "no_data"
            return None, metrics

        # İndeksi datetime formatına garanti et
        df.index = pd.to_datetime(df.index)

        # Teknik göstergeler
        df["SMA20"] = df["Close"].rolling(window=20).mean()
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        df["SMA200"] = df["Close"].rolling(window=200).mean()
        df["RSI"] = calculate_rsi(df["Close"], 14)

        # Son gün değerleri
        last_row = df.iloc[-1]
        prev_close = df["Close"].iloc[-2] if len(df) > 1 else last_row["Close"]
        curr_price = float(last_row["Close"])
        change_val = curr_price - float(prev_close)
        change_pct = (change_val / float(prev_close)) * 100 if prev_close else 0.0

        metrics["current_price"] = curr_price
        metrics["previous_close"] = float(prev_close)
        metrics["change_val"] = change_val
        metrics["change_pct"] = change_pct
        metrics["day_low"] = float(last_row["Low"])
        metrics["day_high"] = float(last_row["High"])
        metrics["volume"] = int(last_row["Volume"])

        # 52 Haftalık aralık
        metrics["fifty_two_week_low"] = float(df["Low"].min())
        metrics["fifty_two_week_high"] = float(df["High"].max())

        # Son RSI ve SMA değerleri
        metrics["rsi_14"] = float(df["RSI"].dropna().iloc[-1]) if not df["RSI"].dropna().empty else None
        metrics["sma_50"] = float(df["SMA50"].dropna().iloc[-1]) if not df["SMA50"].dropna().empty else None
        metrics["sma_200"] = float(df["SMA200"].dropna().iloc[-1]) if not df["SMA200"].dropna().empty else None

        # Ticker info (Piyasa değeri, F/K, PD/DD)
        info = getattr(t, "info", {}) or {}
        raw_mcap = info.get("marketCap")
        if isinstance(raw_mcap, (int, float)) and raw_mcap > 0:
            if raw_mcap >= 1e9:
                metrics["market_cap_str"] = f"{raw_mcap / 1e9:.2f} Milyar TL"
            else:
                metrics["market_cap_str"] = f"{raw_mcap / 1e6:.1f} Milyon TL"

        pe = info.get("trailingPE") or info.get("forwardPE")
        if isinstance(pe, (int, float)) and pe > 0:
            metrics["pe_ratio"] = round(float(pe), 2)

        pb = info.get("priceToBook")
        if isinstance(pb, (int, float)) and pb > 0:
            metrics["pb_ratio"] = round(float(pb), 2)

        return df, metrics

    except Exception as e:
        logger.error(f"Piyasa verisi çekilirken hata ({ticker}): {e}")
        metrics["status"] = "error"
        metrics["error"] = str(e)
        return None, metrics


def create_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    company_name: str
) -> go.Figure:
    """
    Fiyat mumları, hareketli ortalamalar ve işlem hacmini içeren interaktif Plotly grafiği üretir.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # 1. Mum Grafiği (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Fiyat",
            increasing_line_color="#10B981",  # Zümrüt Yeşili
            increasing_fillcolor="#10B981",
            decreasing_line_color="#EF4444",  # Kırmızı
            decreasing_fillcolor="#EF4444",
        ),
        row=1, col=1
    )

    # 2. Hareketli Ortalamalar (SMA)
    if "SMA20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SMA20"],
                mode="lines", name="SMA 20",
                line=dict(color="#38BDF8", width=1.4)  # Camgöbeği
            ),
            row=1, col=1
        )
    if "SMA50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SMA50"],
                mode="lines", name="SMA 50",
                line=dict(color="#F59E0B", width=1.6)  # Kehribar
            ),
            row=1, col=1
        )
    if "SMA200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SMA200"],
                mode="lines", name="SMA 200 (Ana Trend)",
                line=dict(color="#A855F7", width=1.8, dash="dot")  # Mor
            ),
            row=1, col=1
        )

    # 3. İşlem Hacmi (Volume)
    colors = [
        "#10B981" if c >= o else "#EF4444"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Hacim",
            marker=dict(color=colors, opacity=0.75),
        ),
        row=2, col=1
    )

    # Düzen ve Kurumsal Dark Tema
    fig.update_layout(
        title=dict(
            text=f"<b>{company_name} ({ticker})</b> Fiyat & Hacim Grafiği",
            font=dict(size=15, color="#F8FAFC", family="Inter, sans-serif")
        ),
        paper_bgcolor="#080C14",
        plot_bgcolor="#0D1422",
        font=dict(color="#94A3B8"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#CBD5E1")
        ),
        margin=dict(l=40, r=40, t=50, b=30),
        height=520,
        xaxis=dict(
            gridcolor="rgba(30, 41, 59, 0.6)",
            showgrid=True,
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1 Ay", step="month", stepmode="backward"),
                    dict(count=3, label="3 Ay", step="month", stepmode="backward"),
                    dict(count=6, label="6 Ay", step="month", stepmode="backward"),
                    dict(count=1, label="1 Yıl", step="year", stepmode="backward"),
                    dict(step="all", label="Tümü")
                ],
                bgcolor="#1E293B",
                activecolor="#2563EB",
                font=dict(color="#F8FAFC", size=10)
            )
        ),
        xaxis2=dict(
            gridcolor="rgba(30, 41, 59, 0.6)",
            showgrid=True
        ),
        yaxis=dict(
            title="Fiyat (TL)",
            gridcolor="rgba(30, 41, 59, 0.6)",
            showgrid=True,
            side="right"
        ),
        yaxis2=dict(
            title="Hacim",
            gridcolor="rgba(30, 41, 59, 0.3)",
            showgrid=True,
            side="right"
        ),
    )

    return fig
