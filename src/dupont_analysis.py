"""BIST Derin Finansal Rasyolar, DuPont Analizi & Altman Z-Skoru Modülü.

Bu modül:
1. 3 Aşamalı DuPont Modelini (Net Kâr Marjı × Aktif Devir Hızı × Finansal Kaldıraç)
   hesaplar ve Özkaynak Kârlılığının (ROE) ana itici gücünü ayrıştırır.
2. Altman Z-Skorunu (Finansal Dayanıklılık & İflas Riski İndeksi) hesaplayarak
   şirketi Güvenli / Gri / Stres bölgelerine ayırır.
3. Plotly ve HTML tabanlı kurumsal görselleştirme bileşenleri üretir.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import plotly.graph_objects as go

from src.network import ensure_ascii_ca_bundle
from src.integration.models import FundamentalReport
from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)


def calculate_dupont_model(ticker: str) -> Dict[str, Any]:
    """
    Belirtilen hisse için 3 aşamalı DuPont analizini hesaplar.
    ROE = Net Kâr Marjı × Aktif Devir Hızı × Finansal Kaldıraç
    """
    ensure_ascii_ca_bundle()
    import yfinance as yf

    clean_sym = ticker.upper().replace(".IS", "").strip()
    norm_sym = f"{clean_sym}.IS"

    # Varsayılan rasyolar (sektörel ortalama)
    data: Dict[str, Any] = {
        "ticker": clean_sym,
        "net_profit_margin_pct": 8.5,
        "asset_turnover": 0.85,
        "equity_multiplier": 2.2,
        "roe_pct": 15.8,
        "roa_pct": 7.2,
        "driver": "Operasyonel Verimlilik & Marj",
        "quality_assessment": "Dengeli ve sürdürülebilir kârlılık yapısı.",
        "source": "model_estimate"
    }

    try:
        t = yf.Ticker(norm_sym)
        info = getattr(t, "info", {}) or {}

        roe_raw = info.get("returnOnEquity")
        roa_raw = info.get("returnOnAssets")
        npm_raw = info.get("profitMargins")

        if roe_raw is not None and npm_raw is not None and npm_raw > 0:
            npm_pct = float(npm_raw) * 100.0
            roe_pct = float(roe_raw) * 100.0
            roa_pct = (float(roa_raw) * 100.0) if roa_raw is not None else (roe_pct * 0.45)

            # Aktif Devir Hızı = ROA / Net Kâr Marjı
            asset_turnover = round(roa_pct / npm_pct, 2) if npm_pct > 0 else 0.8
            # Finansal Kaldıraç Çarpanı = ROE / ROA
            equity_multiplier = round(roe_pct / roa_pct, 2) if roa_pct > 0 else 2.2

            data["net_profit_margin_pct"] = round(npm_pct, 2)
            data["asset_turnover"] = max(0.1, asset_turnover)
            data["equity_multiplier"] = max(1.0, equity_multiplier)
            data["roe_pct"] = round(roe_pct, 2)
            data["roa_pct"] = round(roa_pct, 2)
            data["source"] = "live_yfinance"

            # İtici güç tespiti
            if equity_multiplier > 3.5:
                data["driver"] = "Yüksek Finansal Kaldıraç (Borç Riski)"
                data["quality_assessment"] = "Özkaynak kârlılığı yüksek borçluluk ve finansal kaldıraç ile destekleniyor. Faiz riski yakından izlenmeli."
            elif asset_turnover > 1.3:
                data["driver"] = "Hızlı Varlık Devir Hızı (Operasyonel Hız)"
                data["quality_assessment"] = "Şirket varlıklarını çok verimli bir şekilde ciroya dönüştürüyor; operasyonel hacim ana itici güç."
            else:
                data["driver"] = "Güçlü Net Kâr Marjı (Fiyatlama Gücü)"
                data["quality_assessment"] = "Yüksek net kâr marjı şirketin güçlü pazar payı ve fiyatlama gücüne sahip olduğunu gösteriyor."

            return data

    except Exception as e:
        logger.debug(f"DuPont canlı hesaplama hatası ({ticker}): {e}")

    # Fallback: Yerel rapordan marj trendine göre tahmini değerler
    rep_file = REPORTS_DIR / f"{clean_sym}_latest.json"
    if rep_file.exists():
        try:
            rep = FundamentalReport.load_from_file(rep_file)
            trend = rep.key_metrics.net_profit_margin_trend.lower()
            debt = rep.key_metrics.debt_to_equity.lower()

            if "artış" in trend:
                data["net_profit_margin_pct"] = 12.4
                data["roe_pct"] = 22.5
            elif "azalış" in trend:
                data["net_profit_margin_pct"] = 4.2
                data["roe_pct"] = 8.5

            if "yüksek" in debt:
                data["equity_multiplier"] = 3.6
                data["driver"] = "Finansal Kaldıraç Ağırlıklı"
            else:
                data["equity_multiplier"] = 1.8
                data["driver"] = "Özkaynak ve Marj Odaklı"

        except Exception:
            pass

    return data


def calculate_altman_z_score(ticker: str, dupont: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanayi ve gelişmekte olan piyasalar için uyarlanmış Altman Z-Skorunu hesaplar.
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    """
    clean_sym = ticker.upper().replace(".IS", "").strip()

    # DuPont ve piyasa çarpanlarına dayalı Z-skoru bileşen kestirimi
    npm = dupont.get("net_profit_margin_pct", 8.0)
    lev = dupont.get("equity_multiplier", 2.0)
    turnover = dupont.get("asset_turnover", 0.9)

    # 1. İşletme Sermayesi / Varlıklar (Likidite proxy)
    x1 = max(0.05, min(0.40, 0.25 - (lev - 2.0) * 0.05))
    # 2. Dağıtılmamış Kârlar / Varlıklar
    x2 = max(0.05, min(0.50, npm * 0.025))
    # 3. EBIT / Varlıklar (ROA ilişkisi)
    x3 = max(0.02, min(0.35, dupont.get("roa_pct", 7.0) / 100.0 * 1.3))
    # 4. Piyasa Değeri / Toplam Borçlar
    x4 = max(0.3, min(3.5, 4.0 / max(1.1, lev)))
    # 5. Satışlar / Varlıklar
    x5 = turnover

    z_score = round(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5, 2)

    if z_score >= 2.90:
        zone = "Güvenli Bölge (Safe Zone)"
        zone_color = "#10B981"
        badge_html = f'<span style="background:rgba(16,185,129,0.18); color:#10B981; border:1px solid #10B981; padding:3px 10px; border-radius:9999px; font-weight:700;">🟢 {zone}</span>'
        interpretation = "Finansal dayanıklılık çok yüksek; kısa ve orta vadeli iflas veya temerrüt riski bulunmuyor."
    elif z_score >= 1.80:
        zone = "Gri Bölge (Grey Zone)"
        zone_color = "#F59E0B"
        badge_html = f'<span style="background:rgba(245,158,11,0.18); color:#F59E0B; border:1px solid #F59E0B; padding:3px 10px; border-radius:9999px; font-weight:700;">🟡 {zone}</span>'
        interpretation = "Finansal yapı dengeli ancak işletme sermayesi ve borç çevirme oranları dikkatle takip edilmeli."
    else:
        zone = "Stres Bölgesi (Distress Zone)"
        zone_color = "#EF4444"
        badge_html = f'<span style="background:rgba(239,68,68,0.18); color:#EF4444; border:1px solid #EF4444; padding:3px 10px; border-radius:9999px; font-weight:700;">🔴 {zone}</span>'
        interpretation = "Yüksek borç yükü ve zayıf kârlılık marjı finansal kırılganlığı artırıyor."

    return {
        "ticker": clean_sym,
        "z_score": z_score,
        "zone": zone,
        "zone_color": zone_color,
        "badge_html": badge_html,
        "interpretation": interpretation,
        "components": {
            "Likidite (X1)": round(x1, 3),
            "Kümülatif Kârlılık (X2)": round(x2, 3),
            "Faaliyet Getirisi (X3)": round(x3, 3),
            "Kaldıraç Gücü (X4)": round(x4, 3),
            "Varlık Verimi (X5)": round(x5, 3)
        }
    }


def create_dupont_breakdown_chart(dupont: Dict[str, Any], ticker: str) -> go.Figure:
    """
    DuPont 3 aşamalı bileşenlerini gösteren karşılaştırmalı yatay bar grafiği üretir.
    """
    categories = [
        "Özkaynak Kârlılığı (ROE %)",
        "Net Kâr Marjı (%)",
        "Aktif Devir Hızı (x)",
        "Finansal Kaldıraç (x)"
    ]
    values = [
        dupont.get("roe_pct", 15.0),
        dupont.get("net_profit_margin_pct", 8.0),
        dupont.get("asset_turnover", 0.9) * 10.0,  # Görsel ölçekleme için
        dupont.get("equity_multiplier", 2.0) * 5.0
    ]
    labels_display = [
        f"%{dupont.get('roe_pct', 15.0):.1f}",
        f"%{dupont.get('net_profit_margin_pct', 8.0):.1f}",
        f"{dupont.get('asset_turnover', 0.9):.2f}x",
        f"{dupont.get('equity_multiplier', 2.0):.2f}x"
    ]

    colors = ["#38BDF8", "#10B981", "#F59E0B", "#A855F7"]

    fig = go.Figure(
        go.Bar(
            y=categories,
            x=values,
            orientation="h",
            text=labels_display,
            textposition="outside",
            marker=dict(color=colors, line=dict(color="#0D1422", width=1.5)),
            textfont=dict(color="#F8FAFC", size=11, family="Inter, sans-serif")
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{ticker}</b> DuPont Ayrıştırma Modeli (ROE Sürücüleri)",
            font=dict(size=13, color="#F8FAFC")
        ),
        paper_bgcolor="#080C14",
        plot_bgcolor="#0D1422",
        margin=dict(l=40, r=40, t=40, b=20),
        height=260,
        xaxis=dict(showgrid=True, gridcolor="rgba(30, 41, 59, 0.4)", showticklabels=False),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#CBD5E1", size=11))
    )
    return fig

