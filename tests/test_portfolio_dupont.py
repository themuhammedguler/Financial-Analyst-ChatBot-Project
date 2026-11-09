"""Portföy Simülatörü ve DuPont / Altman Z-Skoru Birim Testleri."""

import pytest
import plotly.graph_objects as go
from src.portfolio import (
    analyze_portfolio,
    create_sector_donut_chart,
    create_portfolio_radar_chart,
)
from src.dupont_analysis import (
    calculate_dupont_model,
    calculate_altman_z_score,
    create_dupont_breakdown_chart,
)


def test_analyze_portfolio_empty():
    """Boş portföy verildiğinde hata mesajı döndürülmeli."""
    res = analyze_portfolio([])
    assert "error" in res
    assert "bulunamadı" in res["error"]


def test_analyze_portfolio_normalization():
    """Ağırlıklar toplamı 100 olmasa bile doğru normalize edilmeli."""
    items = [
        {"ticker": "THYAO", "weight": 20.0},
        {"ticker": "TUPRS", "weight": 30.0},
    ]
    res = analyze_portfolio(items)
    assert "error" not in res
    holdings = res.get("holdings", [])
    assert len(holdings) == 2
    
    # 20 ve 30 -> %40 ve %60 olmalı
    assert pytest.approx(holdings[0]["weight_pct"], 0.1) == 40.0
    assert pytest.approx(holdings[1]["weight_pct"], 0.1) == 60.0
    assert pytest.approx(sum(h["weight_pct"] for h in holdings), 0.1) == 100.0


def test_analyze_portfolio_composite_scores():
    """Portföy birleşik skoru, sektör dağılımı ve 5D ağırlıkları doğru hesaplanmalı."""
    items = [
        {"ticker": "THYAO", "weight": 50.0},
        {"ticker": "AKBNK", "weight": 50.0},
    ]
    res = analyze_portfolio(items)
    assert "composite_health_score" in res
    assert 0 <= res["composite_health_score"] <= 100
    assert "weighted_sentiment" in res
    assert -1.0 <= res["weighted_sentiment"] <= 1.0
    assert "sector_distribution" in res
    assert len(res["sector_distribution"]) >= 1
    assert "weighted_5d" in res
    assert len(res["weighted_5d"]) == 5
    assert "rebalance_suggestions" in res
    assert len(res["rebalance_suggestions"]) == 2


def test_create_sector_donut_chart():
    """Sektör dağılım donut grafiği geçerli bir Plotly Figure üretmeli."""
    sec_dist = {"Havacılık": 40.0, "Enerji": 35.0, "Bankacılık": 25.0}
    fig = create_sector_donut_chart(sec_dist)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].hole == 0.55


def test_create_portfolio_radar_chart():
    """Portföy 5D Finansal Sağlık Radarı geçerli bir Plotly Figure üretmeli."""
    profile = {
        "Kârlılık": 75.0,
        "Finansal Sağlık": 80.0,
        "Büyüme Katalizörleri": 65.0,
        "Piyasa Duygusu": 70.0,
        "Risk Dayanıklılığı": 85.0
    }
    fig = create_portfolio_radar_chart(profile)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatterpolar"


def test_calculate_dupont_model():
    """DuPont 3 aşamalı model (ROE, marj, aktif devir, kaldıraç) doğru hesaplanmalı."""
    dupont = calculate_dupont_model("THYAO")
    assert dupont["ticker"] == "THYAO"
    assert "net_profit_margin_pct" in dupont
    assert "asset_turnover" in dupont
    assert "equity_multiplier" in dupont
    assert "roe_pct" in dupont
    assert "driver" in dupont
    assert "quality_assessment" in dupont
    assert len(dupont["quality_assessment"]) > 0


def test_calculate_altman_z_score():
    """Altman Z-Skoru ve risk bölgesi (Safe, Grey, Distress) geçerli olmalı."""
    dupont = calculate_dupont_model("AKBNK")
    altman = calculate_altman_z_score("AKBNK", dupont)
    assert altman["ticker"] == "AKBNK"
    assert isinstance(altman["z_score"], (int, float))
    assert altman["zone"] in [
        "Güvenli Bölge (Safe Zone)",
        "Gri Bölge (Grey Zone)",
        "Stres Bölgesi (Distress Zone)"
    ]
    assert "badge_html" in altman
    assert "components" in altman
    assert len(altman["components"]) == 5


def test_create_dupont_breakdown_chart():
    """DuPont yatay bar grafiği geçerli bir Plotly Figure üretmeli."""
    dupont_data = {
        "roe_pct": 22.5,
        "net_profit_margin_pct": 12.0,
        "asset_turnover": 0.95,
        "equity_multiplier": 1.97
    }
    fig = create_dupont_breakdown_chart(dupont_data, "TUPRS")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
