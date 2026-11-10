"""BIST Isı Haritası, Sektörel Screener ve Kâr Sürprizi Modülleri Birim Testleri."""

import pytest
import pandas as pd
import plotly.graph_objects as go
from src.market_heatmap import get_heatmap_dataset, create_bist_treemap
from src.screener import build_screener_universe, filter_and_rank_screener
from src.earnings_surprise import analyze_earnings_surprise, create_growth_comparison_chart


def test_get_heatmap_dataset():
    """Isı haritası veri kümesi doğru kolonlarla ve verilerle üretilmeli."""
    df = get_heatmap_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {"ticker", "sector", "market_cap_billion", "change_pct"}
    assert expected_cols.issubset(set(df.columns))
    assert (df["market_cap_billion"] > 0).all()


def test_create_bist_treemap():
    """Plotly Treemap grafiği hatasız üretilmeli."""
    df = get_heatmap_dataset()
    fig = create_bist_treemap(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert fig.data[0].type == "treemap"


def test_build_screener_universe():
    """Screener evreni tüm temel ve rasyo kolonlarını içermeli."""
    df = build_screener_universe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 10
    required_cols = {"ticker", "sector", "roe_pct", "z_score", "composite_score", "decision"}
    assert required_cols.issubset(set(df.columns))


def test_filter_and_rank_screener():
    """Screener filtreleme ve sıralama fonksiyonu beklendiği gibi çalışmalı."""
    df = build_screener_universe()
    # Min ROE %10 ve Z-Score 1.0 filtrele
    filtered = filter_and_rank_screener(
        df,
        sector="Tümü",
        min_roe=10.0,
        min_z_score=1.0,
        sort_by="roe_pct",
        ascending=False,
        top_n=5
    )
    assert len(filtered) <= 5
    if not filtered.empty:
        assert (filtered["roe_pct"] >= 10.0).all()
        assert (filtered["z_score"] >= 1.0).all()
        # Azalan sırada olmalı
        roes = list(filtered["roe_pct"])
        assert roes == sorted(roes, reverse=True)


def test_analyze_earnings_surprise():
    """Kâr sürprizi analizi kâr büyümesini ve sürpriz kategorisini hesaplamalı."""
    res = analyze_earnings_surprise("THYAO")
    assert res["ticker"] == "THYAO"
    assert "net_profit_growth_pct" in res
    assert "revenue_growth_pct" in res
    assert "surprise_category" in res
    assert "analyst_takeaway" in res
    assert len(res["analyst_takeaway"]) > 0


def test_create_growth_comparison_chart():
    """YoY büyüme karşılaştırma bar grafiği geçerli bir Figure üretmeli."""
    data = {
        "current_net_profit_million": 15000.0,
        "prev_net_profit_million": 10000.0,
        "current_revenue_million": 90000.0,
        "prev_revenue_million": 65000.0
    }
    fig = create_growth_comparison_chart(data, "TUPRS")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].type == "bar"
