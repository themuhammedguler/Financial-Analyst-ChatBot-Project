"""BIST Makroekonomik Duyarlılık & Senaryo Analizörü Modülü.

Döviz Kuru (USD/TRY), TCMB Politika Faizi ve Brent Petrol şoklarının
BIST şirketlerinin kârlılığı, operasyonel marjları ve bilanço yapısı
üzerindeki etkisini simüle eder ve Tornado (Kasırga) duyarlılık grafiği üretir.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# BIST Şirketleri için Özelleştirilmiş Makro Duyarlılık Katsayıları (Betalar)
# fx_beta: USD/TRY %10 arttığında kâra tahmini etki (%)
# rate_beta: TCMB Faizi +250 baz puan (+%2.5) arttığında kâra tahmini etki (%)
# oil_beta: Brent Petrol %15 arttığında kâra tahmini etki (%)
MACRO_TICKER_PROFILES: Dict[str, Dict[str, Any]] = {
    "THYAO": {
        "sector": "Havacılık / Ulaştırma",
        "fx_beta": 7.2,
        "rate_beta": -0.8,
        "oil_beta": -8.5,
        "fx_driver": "Satış gelirlerinin %85+'i döviz (EUR/USD) cinsindendir. TL değer kaybı operasyonel marjları genişletir.",
        "rate_driver": "Uçak filosu finansmanı ağırlıklı olarak küresel döviz cinsi leasing ile yapılır; TL faiz duyarlılığı düşüktür.",
        "oil_driver": "Jet yakıtı giderleri toplam operasyonel maliyetin yaklaşık %38'ini oluşturur; petrol yükselişi kârı baskılar."
    },
    "PGSUS": {
        "sector": "Havacılık / Ulaştırma",
        "fx_beta": 6.8,
        "rate_beta": -1.0,
        "oil_beta": -9.0,
        "fx_driver": "Uluslararası yolcu gelirleri döviz bazlıdır. Kur artışı pozitif katkı sağlar.",
        "rate_driver": "Faiz duyarlılığı orta-düşük düzeydedir.",
        "oil_driver": "Akaryakıt tüketimi operasyonel kârlılığı doğrudan belirler."
    },
    "TUPRS": {
        "sector": "Rafineri & Enerji",
        "fx_beta": 4.0,
        "rate_beta": -1.5,
        "oil_beta": 7.5,
        "fx_driver": "Rafineri ürün satış fiyatları Akdeniz piyasası ve USD kuru ile endekslidir. Stok kârı dövizle paralel artar.",
        "rate_driver": "İşletme sermayesi ve ham petrol alım finansmanı faiz gideri yaratır.",
        "oil_driver": "Ham petrol fiyat artışı rafineri crack marjlarını genişletir ve pozitif stok değerleme kârı üretir."
    },
    "FROTO": {
        "sector": "Otomotiv",
        "fx_beta": 6.5,
        "rate_beta": -3.0,
        "oil_beta": -1.2,
        "fx_driver": "Avrupa'ya yüksek ihracat ve 'cost-plus' (maliyet artı) sözleşmeler güçlü kur koruması sağlar.",
        "rate_driver": "Yurtiçi taşıt kredisi faizlerindeki artış iç pazar talebini yavaşlatır.",
        "oil_driver": "Lojistik navlun giderleri ve küresel talep üzerinden sınırlı negatif etki yaratır."
    },
    "EREGL": {
        "sector": "Demir & Çelik",
        "fx_beta": 3.5,
        "rate_beta": -3.5,
        "oil_beta": -2.0,
        "fx_driver": "Yassı çelik fiyatları uluslararası dolar paritesiyle belirlenir; gelirler döviz endekslidir.",
        "rate_driver": "İnşaat, sanayi ve yatırım talebi yüksek faiz ortamında yavaşlayarak çelik tüketimini azaltır.",
        "oil_driver": "Küresel enerji ve metalurjik kömür maliyetlerini artırır."
    },
    "ASELS": {
        "sector": "Savunma Sanayii",
        "fx_beta": 4.5,
        "rate_beta": -1.8,
        "oil_beta": -0.8,
        "fx_driver": "Milyar dolarlık uluslararası ihracat bakiye siparişleri ve kur garantili kamu sözleşmeleri.",
        "rate_driver": "Uzun vadeli Ar-Ge ve işletme sermayesi borçlanma maliyeti.",
        "oil_driver": "Enerji yoğunluğu düşüktür; petrol fiyatlarına duyarlılığı asgaridir."
    },
    "BIMAS": {
        "sector": "Organize Perakende",
        "fx_beta": -2.2,
        "rate_beta": 2.0,
        "oil_beta": -2.5,
        "fx_driver": "İthal girdi ve ambalaj maliyetleri artarken raf fiyatı geçişkenliği zaman alabilir.",
        "rate_driver": "Net borçsuz ve güçlü nakit akışı sayesinde yüksek gecelik faiz ortamında faiz geliri elde eder.",
        "oil_driver": "Geniş mağaza ağı ve lojistik tır filosu akaryakıt zamlarından etkilenir."
    },
    "SISE": {
        "sector": "Cam & Sanayi",
        "fx_beta": 5.0,
        "rate_beta": -2.5,
        "oil_beta": -4.0,
        "fx_driver": "Küresel ölçekte ihracat ağı döviz gelirlerini destekler.",
        "rate_driver": "Kapasite yatırımları finansmanı için faiz yükü oluşturur.",
        "oil_driver": "Cam fırınlarında doğalgaz ve enerji tüketimi çok yüksektir; küresel enerji fiyatları marjı eritir."
    },
    "AKBNK": {
        "sector": "Bankacılık",
        "fx_beta": 0.5,
        "rate_beta": -3.8,
        "oil_beta": -1.0,
        "fx_driver": "Yasal düzenlemeler gereği yabancı para net genel pozisyonu nötre yakındır.",
        "rate_driver": "Mevduat maliyetleri kredilerden daha hızlı yeniden fiyatlandığı için faiz artışında Net Faiz Marjı (NIM) kısa vadede daralır.",
        "oil_driver": "Enflasyon ve makroekonomik istikrar kanalıyla dolaylı etki eder."
    },
    "GARAN": {
        "sector": "Bankacılık",
        "fx_beta": 0.6,
        "rate_beta": -3.6,
        "oil_beta": -1.0,
        "fx_driver": "Dengeli döviz bilançosu ile kur şoklarına karşı korumalıdır.",
        "rate_driver": "Faiz artışları ilk etapta fonlama maliyetini artırır; TÜFEX getirisiyle dengelenir.",
        "oil_driver": "Dolaylı makroekonomik etki."
    },
    "PETKM": {
        "sector": "Petrokimya",
        "fx_beta": 2.5,
        "rate_beta": -2.5,
        "oil_beta": -4.5,
        "fx_driver": "Ürün fiyatları dolar endekslidir.",
        "rate_driver": "İşletme sermayesi ihtiyacı borçlanmayı artırır.",
        "oil_driver": "Hammadde olarak nafta tüketir; petrol artışı etilen-nafta makasını daraltır."
    }
}

DEFAULT_MACRO_PROFILE = {
    "sector": "Genel BIST Sanayi/Hizmet",
    "fx_beta": 2.0,
    "rate_beta": -2.0,
    "oil_beta": -1.2,
    "fx_driver": "İhracat potansiyeli ve döviz pozisyonuna bağlı olarak kârlılık etkilenir.",
    "rate_driver": "Yüksek faizler finansman maliyetini artırır ve genel iç talebi sınırlandırır.",
    "oil_driver": "Akaryakıt ve lojistik maliyetleri operasyonel kârı etkiler."
}


def analyze_macro_sensitivity(
    ticker: str,
    usd_change_pct: float = 10.0,
    rate_change_bps: int = 250,
    oil_change_pct: float = 15.0
) -> Dict[str, Any]:
    """
    Belirtilen makroekonomik şoklar altında şirketin net kâr / FAVÖK duyarlılığını simüle eder.

    Args:
        ticker: BIST hisse kodu (örn: "THYAO", "TUPRS")
        usd_change_pct: USD/TRY değişim yüzdesi (örn: +10.0)
        rate_change_bps: TCMB Politika Faizi baz puan değişimi (örn: +250 bps = +%2.5)
        oil_change_pct: Brent Petrol varil fiyatı değişim yüzdesi (örn: +15.0)
    """
    clean_sym = ticker.upper().replace(".IS", "").strip()
    profile = MACRO_TICKER_PROFILES.get(clean_sym, DEFAULT_MACRO_PROFILE)

    fx_beta = profile["fx_beta"]
    rate_beta = profile["rate_beta"]
    oil_beta = profile["oil_beta"]

    # Şok hesaplamaları (Normalize oranlar üzerinden)
    # 10% kur şoku = 1.0 katsayı
    fx_impact = round((usd_change_pct / 10.0) * fx_beta, 2)
    # 250 bps faiz şoku = 1.0 katsayı
    rate_impact = round((rate_change_bps / 250.0) * rate_beta, 2)
    # 15% petrol şoku = 1.0 katsayı
    oil_impact = round((oil_change_pct / 15.0) * oil_beta, 2)

    total_impact = round(fx_impact + rate_impact + oil_impact, 2)

    # Risk & Dayanıklılık Sınıflandırması
    if total_impact >= 3.0:
        resilience_badge = "🚀 Makro Pozitif / Kur Korunaklı"
        badge_color = "#10B981"
        verdict = f"{clean_sym}, mevcut senaryodaki kur ve emtia hareketlerinden net pozitif ayrışma potansiyeline sahiptir."
    elif total_impact >= -2.0:
        resilience_badge = "🛡️ Dengeli & Dayanıklı Yapı"
        badge_color = "#38BDF8"
        verdict = f"{clean_sym}, karşıt makro güçlerin birbirini dengelemesi sayesinde şoklara karşı yüksek bağışıklık göstermektedir."
    else:
        resilience_badge = "⚠️ Makro Baskı / Yüksek Maliyet Riski"
        badge_color = "#F43F5E"
        verdict = f"{clean_sym}, mevcut senaryodaki faiz artışı veya enerji maliyetleri nedeniyle marj daralması riskiyle karşı karşıyadır."

    return {
        "ticker": clean_sym,
        "sector": profile["sector"],
        "scenarios": {
            "usd_try": {"change_pct": usd_change_pct, "impact_pct": fx_impact, "driver": profile["fx_driver"]},
            "interest_rate": {"change_bps": rate_change_bps, "impact_pct": rate_impact, "driver": profile["rate_driver"]},
            "brent_oil": {"change_pct": oil_change_pct, "impact_pct": oil_impact, "driver": profile["oil_driver"]},
        },
        "total_impact_pct": total_impact,
        "resilience_badge": resilience_badge,
        "badge_color": badge_color,
        "verdict": verdict
    }


def create_sensitivity_tornado_chart(sensitivity_data: Dict[str, Any], ticker: str) -> go.Figure:
    """Makro duyarlılık analizini Tornado (Kasırga) grafiği olarak çizer."""
    scenarios = sensitivity_data.get("scenarios", {})
    usd_data = scenarios.get("usd_try", {})
    rate_data = scenarios.get("interest_rate", {})
    oil_data = scenarios.get("brent_oil", {})

    factors = [
        f"USD/TRY ({usd_data.get('change_pct', 10):+0.0f}%)",
        f"TCMB Faizi ({rate_data.get('change_bps', 250):+d} bps)",
        f"Brent Petrol ({oil_data.get('change_pct', 15):+0.0f}%)",
        "Bileşik Senaryo Etkisi"
    ]

    impacts = [
        usd_data.get("impact_pct", 0.0),
        rate_data.get("impact_pct", 0.0),
        oil_data.get("impact_pct", 0.0),
        sensitivity_data.get("total_impact_pct", 0.0)
    ]

    colors = [
        "#10B981" if imp > 0 else ("#F43F5E" if imp < 0 else "#64748B")
        for imp in impacts
    ]

    texts = [f"{imp:+0.2f}%" for imp in impacts]

    fig = go.Figure(
        go.Bar(
            y=factors,
            x=impacts,
            orientation="h",
            text=texts,
            textposition="outside",
            marker=dict(color=colors, line=dict(color="#0D1422", width=1.5)),
            textfont=dict(color="#F8FAFC", size=11, family="Inter, sans-serif")
        )
    )

    # 0 çizgisi
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="#94A3B8")

    fig.update_layout(
        title=dict(
            text=f"<b>{ticker}</b> Makro Şok Duyarlılık Analizi (Tornado)",
            font=dict(size=13, color="#F8FAFC")
        ),
        paper_bgcolor="#080C14",
        plot_bgcolor="#0D1422",
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
        xaxis=dict(
            title=dict(text="Net Kâr & Marj Etkisi (%)", font=dict(color="#94A3B8", size=11)),
            tickfont=dict(color="#CBD5E1", size=10),
            showgrid=True,
            gridcolor="rgba(30, 41, 59, 0.4)"
        ),
        yaxis=dict(
            tickfont=dict(color="#CBD5E1", size=10),
            autorange="reversed"
        )
    )

    return fig
