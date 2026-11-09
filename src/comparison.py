"""BIST Çift Şirket Kıyaslama ve Peer-to-Peer Matris Modülü.

İki BIST şirketinin KAP temel analiz verilerini, finansal rasyolarını,
duygu skorlarını ve 5 boyutlu finansal sağlık profillerini tek ekranda
yan yana ve çift radar grafiği (Dual Radar Chart) ile kıyaslar.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import plotly.graph_objects as go

from src.integration.models import FundamentalReport


def _calculate_dimension_score(val_level: str, default: float = 60.0) -> float:
    """Metin tabanlı göstergeleri 0-100 radar skoruna dönüştürür."""
    lowered = str(val_level).lower()
    if any(w in lowered for w in ["artış", "yüksek", "güçlü", "pozitif", "düşük/yönetilebilir"]):
        return 88.0
    if any(w in lowered for w in ["orta", "dengeli", "nötr", "yatay", "yönetilebilir"]):
        return 65.0
    if any(w in lowered for w in ["azalış", "zayıf", "negatif", "yüksek"]):
        return 42.0
    return default


def get_5d_profile_scores(report: FundamentalReport) -> Dict[str, float]:
    """Rapor nesnesinden 5 boyutlu finansal karne skorlarını çıkarır."""
    # 1. Kârlılık (Net profit margin trend)
    profitability = _calculate_dimension_score(report.key_metrics.net_profit_margin_trend)
    
    # 2. Finansal Sağlık (Borç/Özkaynak)
    # Düşük borç -> Yüksek sağlık puanı
    debt_raw = report.key_metrics.debt_to_equity.lower()
    if "düşük" in debt_raw or "yönetilebilir" in debt_raw:
        financial_health = 90.0
    elif "orta" in debt_raw:
        financial_health = 68.0
    else:
        financial_health = 45.0

    # 3. Büyüme Katalizörleri
    cat_count = len(report.catalysts)
    growth = min(95.0, 50.0 + cat_count * 15.0)

    # 4. Piyasa Duygusu (-1.0 ile +1.0 -> 0-100)
    sentiment = max(10.0, min(95.0, (report.sentiment_score + 1.0) * 45.0 + 10.0))

    # 5. Risk Dayanıklılığı (Risk faktör sayısı azaldıkça puan yükselir)
    risk_count = len(report.risk_factors)
    risk_resilience = max(20.0, 95.0 - risk_count * 12.0)

    return {
        "Kârlılık": profitability,
        "Finansal Sağlık": financial_health,
        "Büyüme Katalizörleri": growth,
        "Piyasa Duygusu": sentiment,
        "Risk Dayanıklılığı": risk_resilience
    }


def create_comparison_radar_chart(
    report1: FundamentalReport,
    report2: FundamentalReport
) -> go.Figure:
    """
    İki şirketin 5 boyutlu finansal profilini aynı radar üzerinde çakıştırır.
    """
    scores1 = get_5d_profile_scores(report1)
    scores2 = get_5d_profile_scores(report2)

    categories = ["Kârlılık", "Finansal Sağlık", "Büyüme Katalizörleri", "Piyasa Duygusu", "Risk Dayanıklılığı"]
    # Radar döngüsünü kapatmak için ilk elemanı sona ekle
    r_cats = categories + [categories[0]]
    r_vals1 = [scores1[c] for c in categories] + [scores1[categories[0]]]
    r_vals2 = [scores2[c] for c in categories] + [scores2[categories[0]]]

    fig = go.Figure()

    # 1. Şirket (Camgöbeği)
    fig.add_trace(
        go.Scatterpolar(
            r=r_vals1,
            theta=r_cats,
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.22)',
            line=dict(color='#38BDF8', width=2.4),
            name=f"{report1.company_name} ({report1.ticker})"
        )
    )

    # 2. Şirket (Kehribar / Turuncu)
    fig.add_trace(
        go.Scatterpolar(
            r=r_vals2,
            theta=r_cats,
            fill='toself',
            fillcolor='rgba(245, 158, 11, 0.20)',
            line=dict(color='#F59E0B', width=2.4),
            name=f"{report2.company_name} ({report2.ticker})"
        )
    )

    fig.update_layout(
        polar=dict(
            bgcolor="#0D1422",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticks="",
                gridcolor="rgba(30, 41, 59, 0.6)",
                linecolor="rgba(30, 41, 59, 0.6)",
                tickfont=dict(color="#64748B", size=9)
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#CBD5E1", family="Inter, sans-serif"),
                gridcolor="rgba(30, 41, 59, 0.6)",
                linecolor="rgba(30, 41, 59, 0.6)",
            )
        ),
        paper_bgcolor="#080C14",
        plot_bgcolor="#080C14",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#F8FAFC")
        ),
        margin=dict(l=60, r=60, t=30, b=50),
        height=450
    )

    return fig


def generate_peer_comparison_summary(
    report1: FundamentalReport,
    report2: FundamentalReport,
    metrics1: Optional[Dict[str, Any]] = None,
    metrics2: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    İki şirketin karşılaştırma matrisini ve analitik sentezini üretir.
    """
    scores1 = get_5d_profile_scores(report1)
    scores2 = get_5d_profile_scores(report2)

    total_score1 = sum(scores1.values()) / len(scores1)
    total_score2 = sum(scores2.values()) / len(scores2)

    # Güçlü yönler
    lead1 = [k for k in scores1 if scores1[k] > scores2[k]]
    lead2 = [k for k in scores2 if scores2[k] > scores1[k]]

    verdict_parts = []
    if lead1:
        verdict_parts.append(f"**{report1.ticker}**, {', '.join(lead1)} alanlarında daha güçlü.")
    if lead2:
        verdict_parts.append(f"**{report2.ticker}**, {', '.join(lead2)} alanlarında öne çıkıyor.")

    verdict_text = " ".join(verdict_parts) if verdict_parts else "İki şirketin genel finansal profili dengeli bir dağılım sergiliyor."

    return {
        "ticker1": report1.ticker,
        "ticker2": report2.ticker,
        "name1": report1.company_name,
        "name2": report2.company_name,
        "score1": round(total_score1, 1),
        "score2": round(total_score2, 1),
        "scores_dict1": scores1,
        "scores_dict2": scores2,
        "lead1": lead1,
        "lead2": lead2,
        "verdict_text": verdict_text
    }
