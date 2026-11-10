"""BIST Temettü Şampiyonları & Serbest Nakit Akışı (FCF) Güvenlik Modülü.

Şirketlerin temettü verimi, kâr dağıtma oranı (payout ratio),
Serbest Nakit Akışı (FCF) ve temettünün sürdürülebilirlik gücünü analiz eder.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import plotly.graph_objects as go

from src.network import ensure_ascii_ca_bundle
from src.integration.models import FundamentalReport
from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)

# BIST şirketleri için varsayılan kurumsal temettü profilleri
# Canlı piyasa kapalıyken veya veri eksikken güvenilir finansal temel sağlar
DEFAULT_DIVIDEND_PROFILES = {
    "TUPRS": {"yield_pct": 10.5, "payout_pct": 82.0, "fcf_margin_pct": 14.2, "consecutive_years": 8},
    "FROTO": {"yield_pct": 7.2, "payout_pct": 75.0, "fcf_margin_pct": 12.0, "consecutive_years": 10},
    "TTRAK": {"yield_pct": 8.8, "payout_pct": 85.0, "fcf_margin_pct": 15.5, "consecutive_years": 9},
    "EREGL": {"yield_pct": 5.4, "payout_pct": 65.0, "fcf_margin_pct": 8.0, "consecutive_years": 6},
    "AKBNK": {"yield_pct": 3.8, "payout_pct": 20.0, "fcf_margin_pct": 18.0, "consecutive_years": 5},
    "GARAN": {"yield_pct": 4.1, "payout_pct": 22.0, "fcf_margin_pct": 19.5, "consecutive_years": 4},
    "BIMAS": {"yield_pct": 4.5, "payout_pct": 60.0, "fcf_margin_pct": 7.5, "consecutive_years": 7},
    "THYAO": {"yield_pct": 0.0, "payout_pct": 0.0, "fcf_margin_pct": 16.5, "consecutive_years": 0},
    "PGSUS": {"yield_pct": 0.0, "payout_pct": 0.0, "fcf_margin_pct": 14.0, "consecutive_years": 0},
    "ASELS": {"yield_pct": 1.5, "payout_pct": 25.0, "fcf_margin_pct": 6.5, "consecutive_years": 5},
}


def calculate_dividend_metrics(ticker: str) -> Dict[str, Any]:
    """
    Belirtilen hisse için temettü verimi, dağıtma oranı, serbest nakit akışı
    ve Temettü Güvenlik Skorunu (0-100) hesaplar.
    """
    ensure_ascii_ca_bundle()
    import yfinance as yf

    clean_sym = ticker.upper().replace(".IS", "").strip()
    norm_sym = f"{clean_sym}.IS"

    profile = DEFAULT_DIVIDEND_PROFILES.get(clean_sym, {
        "yield_pct": 3.5,
        "payout_pct": 45.0,
        "fcf_margin_pct": 9.0,
        "consecutive_years": 3
    })

    yield_pct = profile["yield_pct"]
    payout_pct = profile["payout_pct"]
    fcf_margin_pct = profile["fcf_margin_pct"]
    consecutive_years = profile["consecutive_years"]

    try:
        t = yf.Ticker(norm_sym)
        info = getattr(t, "info", {}) or {}

        # Canlı temettü verimi
        dy_raw = info.get("dividendYield")
        if dy_raw is not None and dy_raw > 0:
            yield_pct = round(float(dy_raw) * 100.0, 2)
            # Yfinance bazen 0.05 bazen 5.0 verir
            if yield_pct < 0.2:
                yield_pct = round(yield_pct * 100.0, 2)

        # Dağıtma Oranı
        pr_raw = info.get("payoutRatio")
        if pr_raw is not None and pr_raw > 0:
            payout_pct = round(float(pr_raw) * 100.0, 1)

        # Serbest Nakit Akışı
        fcf_raw = info.get("freeCashflow")
        rev_raw = info.get("totalRevenue")
        if fcf_raw and rev_raw and rev_raw > 0:
            fcf_margin_pct = round((float(fcf_raw) / float(rev_raw)) * 100.0, 1)

    except Exception as e:
        logger.debug(f"Dividend metrics live error for {clean_sym}: {e}")

    # Temettü Güvenlik Skoru Hesaplama (0-100)
    # Kriterler:
    # 1. Payout Ratio: %30-%70 arası ideal (40 puan). %90 üzeri aşırı borç riski.
    # 2. Serbest Nakit Akışı: Pozitif ve yüksek FCF temettüyü korur (40 puan).
    # 3. Düzenli temettü geçmişi (20 puan).
    score = 0.0
    if yield_pct <= 0:
        score = 25.0  # Temettü vermiyor, büyüme/yatırım odaklı
        category = "❌ Temettü Dağıtmıyor (Büyüme/Yatırım Odaklı)"
        badge_color = "#64748B"
        takeaway = "Şirket kârını temettü olarak dağıtmak yerine filo, kapasite veya teknoloji yatırımlarında değerlendirmektedir."
    else:
        # Payout puanı
        if 20.0 <= payout_pct <= 75.0:
            score += 40.0
        elif 75.0 < payout_pct <= 90.0:
            score += 25.0
        else:
            score += 15.0

        # FCF Puanı
        if fcf_margin_pct >= 10.0:
            score += 40.0
        elif fcf_margin_pct >= 5.0:
            score += 25.0
        elif fcf_margin_pct > 0:
            score += 15.0
        else:
            score += 5.0

        # Geçmiş istikrarı
        score += min(20.0, consecutive_years * 2.5)

        score = min(100.0, max(10.0, score))

        if score >= 75.0 and yield_pct >= 5.0:
            category = "👑 BIST Temettü Aristokratı / Güçlü Nakit"
            badge_color = "#10B981"
            takeaway = "Şirket yüksek serbest nakit akışı yaratmakta ve kâr payı dağıtımını borçlanmadan, kendi nakdiyle finanse etmektedir."
        elif score >= 55.0:
            category = "🛡️ Güvenli & Dengeli Temettü"
            badge_color = "#38BDF8"
            takeaway = "Temettü ödeme gücü tatmin edici düzeydedir. Nakit akışı dağıtılan kâr payını karşılayabilecek kapasitededir."
        else:
            category = "⚠️ Riskli / Sürdürülebilirlik Takip Edilmeli"
            badge_color = "#F59E0B"
            takeaway = "Dağıtma oranı yüksek veya serbest nakit akışı dalgalıdır; gelecekte temettü kesintisi riski mevcuttur."

    return {
        "ticker": clean_sym,
        "yield_pct": yield_pct,
        "payout_pct": payout_pct,
        "fcf_margin_pct": fcf_margin_pct,
        "consecutive_years": consecutive_years,
        "dividend_safety_score": round(score, 1),
        "category": category,
        "badge_color": badge_color,
        "takeaway": takeaway
    }


def create_dividend_chart(metrics: Dict[str, Any], ticker: str) -> go.Figure:
    """Temettü verimi, dağıtma oranı ve güvenlik skorunu gösteren interaktif karne grafiği."""
    categories = ["Temettü Güvenlik Skoru", "Temettü Verimi (%)", "Kâr Dağıtım Oranı (%)", "Serbest Nakit Akış Marjı (%)"]
    values = [
        metrics.get("dividend_safety_score", 50.0),
        metrics.get("yield_pct", 0.0) * 5.0,     # Görsel ölçekleme
        metrics.get("payout_pct", 0.0),
        max(0.0, metrics.get("fcf_margin_pct", 0.0) * 3.0)
    ]
    texts = [
        f"{metrics.get('dividend_safety_score', 50.0):.0f} / 100",
        f"%{metrics.get('yield_pct', 0.0):.1f}",
        f"%{metrics.get('payout_pct', 0.0):.1f}",
        f"%{metrics.get('fcf_margin_pct', 0.0):.1f}"
    ]

    colors = ["#10B981", "#38BDF8", "#F59E0B", "#A855F7"]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=values,
            text=texts,
            textposition="outside",
            marker=dict(color=colors, line=dict(color="#0D1422", width=1.5)),
            textfont=dict(color="#F8FAFC", size=11, family="Inter, sans-serif")
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{ticker}</b> Temettü & Nakit Akış Karnesi",
            font=dict(size=13, color="#F8FAFC")
        ),
        paper_bgcolor="#080C14",
        plot_bgcolor="#0D1422",
        margin=dict(l=20, r=20, t=40, b=20),
        height=270,
        xaxis=dict(tickfont=dict(color="#CBD5E1", size=10)),
        yaxis=dict(showgrid=True, gridcolor="rgba(30, 41, 59, 0.4)", showticklabels=False)
    )
    return fig
