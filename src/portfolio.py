"""BIST Portföy Risk & Temel Sağlık Simülatörü Modülü.

Kullanıcının seçtiği BIST hisselerini ve portföy ağırlıklarını alarak:
1. Portföyün ağırlıklı Temel Sağlık Puanı ve Duygu Skorunu hesaplar,
2. Sektörel çeşitlendirme ve yoğunlaşma riskini (Plotly Donut) analiz eder,
3. Birleşik 5 boyutlu Finansal Sağlık Radarını üretir,
4. Kelly Kriteri ve temel skorlara dayalı dinamik ağırlık optimizasyon önerisi sunar.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import plotly.graph_objects as go

from src.data_collection.kap_downloader import BIST_100_COMPANIES
from src.integration.models import FundamentalReport
from src.comparison import get_5d_profile_scores
from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)


def analyze_portfolio(
    portfolio_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Portföy hisselerini ve ağırlıklarını analiz ederek birleşik sağlık skorunu,
    sektör dağılımını ve Kelly optimizasyon önerilerini hesaplar.
    
    portfolio_items: [{"ticker": "THYAO", "weight": 40.0}, ...]
    """
    if not portfolio_items:
        return {"error": "Portföyde hisse bulunamadı."}

    # Ağırlıkları normalize et (toplam = 100)
    total_raw_weight = sum(item.get("weight", 0.0) for item in portfolio_items)
    if total_raw_weight <= 0:
        norm_factor = 100.0 / len(portfolio_items)
        normalized_items = [
            {"ticker": item["ticker"].upper().strip(), "weight": norm_factor}
            for item in portfolio_items
        ]
    else:
        normalized_items = [
            {
                "ticker": item["ticker"].upper().strip(),
                "weight": (item.get("weight", 0.0) / total_raw_weight) * 100.0
            }
            for item in portfolio_items
        ]

    sector_distribution: Dict[str, float] = {}
    weighted_sentiment = 0.0
    weighted_5d: Dict[str, float] = {
        "Kârlılık": 0.0,
        "Finansal Sağlık": 0.0,
        "Büyüme Katalizörleri": 0.0,
        "Piyasa Duygusu": 0.0,
        "Risk Dayanıklılığı": 0.0
    }

    holdings_details = []

    for item in normalized_items:
        ticker = item["ticker"]
        weight_pct = item["weight"]
        weight_ratio = weight_pct / 100.0

        comp_info = BIST_100_COMPANIES.get(ticker, {"name": ticker, "sector": "Genel"})
        sector = comp_info.get("sector", "Diğer")
        sector_distribution[sector] = sector_distribution.get(sector, 0.0) + weight_pct

        # Raporu yükle
        report_file = REPORTS_DIR / f"{ticker}_latest.json"
        if report_file.exists():
            try:
                rep = FundamentalReport.load_from_file(report_file)
            except Exception:
                rep = FundamentalReport(
                    ticker=ticker,
                    company_name=comp_info["name"],
                    report_date="2026-11-09",
                    fundamental_summary="Veri analizi devam ediyor.",
                    sentiment_score=0.0,
                    sentiment_label="Nötr",
                    key_metrics={}
                )
        else:
            rep = FundamentalReport(
                ticker=ticker,
                company_name=comp_info["name"],
                report_date="2026-11-09",
                fundamental_summary="Veri analizi devam ediyor.",
                sentiment_score=0.0,
                sentiment_label="Nötr",
                key_metrics={}
            )

        scores_5d = get_5d_profile_scores(rep)
        stock_avg_score = sum(scores_5d.values()) / len(scores_5d)

        # Ağırlıklı toplamlara ekle
        weighted_sentiment += rep.sentiment_score * weight_ratio
        for k in weighted_5d:
            weighted_5d[k] += scores_5d[k] * weight_ratio

        holdings_details.append({
            "ticker": ticker,
            "company_name": comp_info["name"],
            "sector": sector,
            "weight_pct": round(weight_pct, 1),
            "sentiment_score": rep.sentiment_score,
            "sentiment_label": rep.sentiment_label,
            "stock_avg_score": round(stock_avg_score, 1),
            "scores_5d": scores_5d
        })

    composite_health_score = sum(weighted_5d.values()) / len(weighted_5d)

    # Kelly Kriteri Rebalancing Önerisi:
    # Yüksek skora ve pozitif duyguya sahip hisselere daha yüksek sermaye tahsisi
    total_attractiveness = 0.0
    attractiveness_scores = {}
    for h in holdings_details:
        # Çekicilik Skoru = (5D Skor / 100) * (1 + Sentiment * 0.5)
        raw_att = (h["stock_avg_score"] / 100.0) * max(0.2, 1.0 + h["sentiment_score"] * 0.5)
        attractiveness_scores[h["ticker"]] = raw_att
        total_attractiveness += raw_att

    rebalance_suggestions = []
    for h in holdings_details:
        ticker = h["ticker"]
        opt_weight = (attractiveness_scores[ticker] / total_attractiveness) * 100.0 if total_attractiveness > 0 else h["weight_pct"]
        diff = opt_weight - h["weight_pct"]
        rebalance_suggestions.append({
            "ticker": ticker,
            "current_weight": h["weight_pct"],
            "optimal_weight": round(opt_weight, 1),
            "diff": round(diff, 1),
            "action": f"Ağırlığı %{abs(diff):.1f} {'ARTIR' if diff > 1.0 else ('AZALT' if diff < -1.0 else 'KORU')}"
        })

    # Duygu Etiketi
    if weighted_sentiment >= 0.20:
        port_sentiment_label = "Pozitif"
        port_badge_color = "#10B981"
    elif weighted_sentiment <= -0.20:
        port_sentiment_label = "Negatif"
        port_badge_color = "#EF4444"
    else:
        port_sentiment_label = "Dengeli / Nötr"
        port_badge_color = "#F59E0B"

    return {
        "holdings": holdings_details,
        "composite_health_score": round(composite_health_score, 1),
        "weighted_sentiment": round(weighted_sentiment, 2),
        "port_sentiment_label": port_sentiment_label,
        "port_badge_color": port_badge_color,
        "weighted_5d": {k: round(v, 1) for k, v in weighted_5d.items()},
        "sector_distribution": {k: round(v, 1) for k, v in sector_distribution.items()},
        "rebalance_suggestions": rebalance_suggestions
    }


def create_sector_donut_chart(sector_data: Dict[str, float]) -> go.Figure:
    """Sektörel yoğunlaşma ve çeşitlendirme dağılımını gösteren Plotly Donut grafiği."""
    labels = list(sector_data.keys())
    values = list(sector_data.values())

    palette = [
        "#38BDF8", "#10B981", "#F59E0B", "#A855F7", "#EC4899",
        "#06B6D4", "#F97316", "#84CC16", "#6366F1", "#14B8A6"
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                textinfo="label+percent",
                marker=dict(colors=palette[:len(labels)], line=dict(color="#0D1422", width=2)),
                hoverinfo="label+value+percent",
                textfont=dict(size=11, color="#F8FAFC")
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="<b>Sektörel Portföy Dağılımı</b>",
            font=dict(size=14, color="#F8FAFC", family="Inter, sans-serif")
        ),
        paper_bgcolor="#080C14",
        plot_bgcolor="#080C14",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=320
    )
    return fig


def create_portfolio_radar_chart(weighted_5d: Dict[str, float]) -> go.Figure:
    """Portföyün ağırlıklı 5 boyutlu Finansal Sağlık Radarını üretir."""
    categories = ["Kârlılık", "Finansal Sağlık", "Büyüme Katalizörleri", "Piyasa Duygusu", "Risk Dayanıklılığı"]
    r_cats = categories + [categories[0]]
    r_vals = [weighted_5d.get(c, 50.0) for c in categories] + [weighted_5d.get(categories[0], 50.0)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r_vals,
            theta=r_cats,
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.28)',
            line=dict(color='#38BDF8', width=2.6),
            name="Portföy Ortalaması"
        )
    )

    fig.update_layout(
        polar=dict(
            bgcolor="#0D1422",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                gridcolor="rgba(30, 41, 59, 0.6)",
                linecolor="rgba(30, 41, 59, 0.6)",
                tickfont=dict(color="#64748B", size=8)
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#CBD5E1"),
                gridcolor="rgba(30, 41, 59, 0.6)",
                linecolor="rgba(30, 41, 59, 0.6)",
            )
        ),
        paper_bgcolor="#080C14",
        plot_bgcolor="#080C14",
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=320
    )
    return fig

