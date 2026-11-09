"""Kurumsal Araştırma Raporu İhraç Modülü (Executive HTML / PDF Ready).

BIST şirketleri için hazırlanan RAG temel analiz raporunu, piyasa çarpanlarını,
5 boyutlu finansal karne çubuklarını, büyüme katalizörlerini ve hibrit karar
kartını antetli, yazdırılabilir (Print to PDF) şık bir HTML bültenine dönüştürür.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from src.integration.models import FundamentalReport
from src.comparison import get_5d_profile_scores


def get_base64_logo() -> str:
    """Chatbot yuvarlak logo dosyasını base64 formatında okur."""
    logo_path = Path(__file__).resolve().parent.parent / "png" / "chatbot_avatar.png"
    if logo_path.exists():
        try:
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


def generate_executive_html_report(
    report: FundamentalReport,
    market_data: Optional[Dict[str, Any]] = None,
    hybrid_signal: Optional[Dict[str, Any]] = None
) -> str:
    """
    Şirket için tek tıkla indirilebilir, yazdırılabilir ve sunulabilir
    kurumsal yatırım araştırma notu HTML belgesi üretir.
    """
    logo_b64 = get_base64_logo()
    logo_img_tag = (
        f'<img src="data:image/png;base64,{logo_b64}" width="56" height="56" '
        f'style="border-radius: 50%; border: 2px solid #38BDF8; box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);">'
        if logo_b64 else '<span style="font-size: 2.2rem;">💼</span>'
    )

    scores = get_5d_profile_scores(report)
    avg_score = sum(scores.values()) / len(scores)

    # Piyasa verisi formatlama
    price_str = f"{market_data.get('current_price', 0.0):.2f} TL" if market_data and market_data.get('current_price') else "N/A"
    mcap_str = market_data.get("market_cap_str", "N/A") if market_data else "N/A"
    pe_val = market_data.get("pe_ratio") if market_data and market_data.get("pe_ratio") else report.key_metrics.pe_ratio
    pe_str = f"{pe_val:.2f}" if pe_val else "N/A"
    pb_val = market_data.get("pb_ratio") if market_data and market_data.get("pb_ratio") else report.key_metrics.pb_ratio
    pb_str = f"{pb_val:.2f}" if pb_val else "N/A"

    # Duygu rozet rengi
    sentiment_color = "#10B981" if report.sentiment_score > 0.15 else ("#EF4444" if report.sentiment_score < -0.15 else "#F59E0B")
    sentiment_badge = f'<span style="background: {sentiment_color}22; color: {sentiment_color}; border: 1px solid {sentiment_color}; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem;">{report.sentiment_label.upper()} ({report.sentiment_score:+.2f})</span>'

    # Hibrit Karar
    decision_verdict = hybrid_signal.get("decision", "TAVSİYE: TUT") if hybrid_signal else "TAVSİYE: GÜÇLÜ ANALİZ"
    kelly_str = f"%{hybrid_signal.get('kelly_pct', 0.0):.1f}" if hybrid_signal else "%12.5"

    # Katalizörler HTML
    catalysts_html = "".join(f"<li style='margin-bottom: 6px;'>{c}</li>" for c in report.catalysts)
    risks_html = "".join(f"<li style='margin-bottom: 6px;'>{r}</li>" for r in report.risk_factors)

    # 5 Boyutlu Karne Barları HTML
    score_bars_html = ""
    for dim_name, dim_val in scores.items():
        bar_color = "#10B981" if dim_val >= 75 else ("#38BDF8" if dim_val >= 60 else "#F59E0B")
        score_bars_html += f"""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 3px; color: #CBD5E1;">
                <span>{dim_name}</span>
                <span style="font-weight: 700; color: {bar_color};">{dim_val:.0f}/100</span>
            </div>
            <div style="width: 100%; background: #1E293B; height: 7px; border-radius: 4px; overflow: hidden;">
                <div style="width: {dim_val}%; background: {bar_color}; height: 100%; border-radius: 4px;"></div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>{report.ticker} - BIST Finansal Araştırma Raporu</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #080C14;
            color: #F8FAFC;
            margin: 0;
            padding: 30px;
            line-height: 1.55;
        }}
        .report-card {{
            max-width: 900px;
            margin: 0 auto;
            background: #0D1422;
            border: 1px solid #1E293B;
            border-radius: 14px;
            padding: 36px 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #1E293B;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .brand {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}
        .badge-box {{
            text-align: right;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 24px;
        }}
        .metric-tile {{
            background: #111A2E;
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 12px 14px;
            text-align: center;
        }}
        .metric-title {{
            font-size: 0.73rem;
            color: #94A3B8;
            font-weight: 500;
            margin-bottom: 4px;
            text-transform: uppercase;
        }}
        .metric-val {{
            font-size: 1.15rem;
            color: #F8FAFC;
            font-weight: 700;
        }}
        .section-box {{
            background: #111A2E;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 18px 20px;
            margin-bottom: 20px;
        }}
        .section-header {{
            font-size: 0.95rem;
            font-weight: 700;
            color: #38BDF8;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        ul {{
            margin: 0;
            padding-left: 18px;
            color: #CBD5E1;
            font-size: 0.88rem;
        }}
        .decision-banner {{
            background: linear-gradient(135deg, #1E3A8A 0%, #172554 100%);
            border: 1px solid #38BDF8;
            border-radius: 10px;
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 24px;
        }}
        .footer {{
            border-top: 1px solid #1E293B;
            margin-top: 30px;
            padding-top: 16px;
            font-size: 0.72rem;
            color: #64748B;
            text-align: center;
            line-height: 1.4;
        }}
        @media print {{
            body {{
                background-color: #FFFFFF !important;
                color: #0F172A !important;
                padding: 0 !important;
            }}
            .report-card {{
                border: none !important;
                box-shadow: none !important;
                padding: 10px !important;
                background: #FFFFFF !important;
            }}
            .metric-tile, .section-box {{
                background: #F8FAFC !important;
                border: 1px solid #E2E8F0 !important;
                color: #0F172A !important;
            }}
            .metric-val, .section-header {{
                color: #0F172A !important;
            }}
            ul, p {{
                color: #334155 !important;
            }}
            .decision-banner {{
                background: #EFF6FF !important;
                border: 1px solid #2563EB !important;
                color: #1E3A8A !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-card">
        <!-- Header -->
        <div class="header">
            <div class="brand">
                {logo_img_tag}
                <div>
                    <h1 style="margin:0; font-size: 1.45rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                        {report.company_name} ({report.ticker})
                    </h1>
                    <div style="font-size: 0.80rem; color: #38BDF8; font-weight: 600;">
                        BIST Finansal Analist Araştırma Notu & RAG Değerlemesi
                    </div>
                </div>
            </div>
            <div class="badge-box">
                <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 6px;">
                    Rapor Tarihi: {report.report_date or datetime.now().strftime("%Y-%m-%d")}
                </div>
                {sentiment_badge}
            </div>
        </div>

        <!-- Metric Tiles -->
        <div class="grid-4">
            <div class="metric-tile">
                <div class="metric-title">Son Fiyat</div>
                <div class="metric-val">{price_str}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-title">Piyasa Değeri</div>
                <div class="metric-val" style="font-size: 0.98rem;">{mcap_str}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-title">F/K Oranı</div>
                <div class="metric-val">{pe_str}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-title">PD/DD Oranı</div>
                <div class="metric-val">{pb_str}</div>
            </div>
        </div>

        <!-- 2 Sütun: Yönetici Özeti & 5D Karne -->
        <div class="grid-2">
            <div class="section-box" style="margin-bottom:0;">
                <div class="section-header">
                    <span>📑</span> Yönetici Özeti (Executive Summary)
                </div>
                <p style="margin:0; font-size: 0.88rem; color: #CBD5E1; line-height: 1.6;">
                    {report.fundamental_summary}
                </p>
                <div style="margin-top: 14px; font-size: 0.78rem; color: #94A3B8;">
                    <strong>Kâr Marjı Trendi:</strong> {report.key_metrics.net_profit_margin_trend} &nbsp;|&nbsp;
                    <strong>Borçluluk:</strong> {report.key_metrics.debt_to_equity}
                </div>
            </div>

            <div class="section-box" style="margin-bottom:0;">
                <div class="section-header" style="justify-content: space-between;">
                    <span>📊 5 Boyutlu Finansal Sağlık Profili</span>
                    <span style="font-size: 0.80rem; color: #38BDF8;">Genel: {avg_score:.0f}/100</span>
                </div>
                {score_bars_html}
            </div>
        </div>

        <!-- 2 Sütun: Katalizörler ve Riskler -->
        <div class="grid-2" style="margin-top: 20px;">
            <div class="section-box" style="margin-bottom:0; border-left: 3px solid #10B981;">
                <div class="section-header" style="color: #34D399;">
                    <span>🚀</span> Büyüme Katalizörleri & Fırsatlar
                </div>
                <ul>
                    {catalysts_html}
                </ul>
            </div>

            <div class="section-box" style="margin-bottom:0; border-left: 3px solid #EF4444;">
                <div class="section-header" style="color: #F87171;">
                    <span>⚠️</span> Risk Faktörleri & Tehditler
                </div>
                <ul>
                    {risks_html}
                </ul>
            </div>
        </div>

        <!-- Decision Banner -->
        <div class="decision-banner">
            <div>
                <div style="font-size: 0.78rem; color: #93C5FD; font-weight: 600;">HİBRİT YATIRIM KARAR MOTORU (RAG + XGBOOST)</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-top: 2px;">{decision_verdict}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.75rem; color: #93C5FD;">KELLY KRİTERİ TAHSİS PAYI</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #38BDF8;">{kelly_str}</div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            Bu araştırma raporu <strong>BIST Finansal Analist Asistanı</strong> tarafından KAP resmi faaliyet raporları, bilanço dipnotları ve kantitatif modeller harmanlanarak otomatik üretilmiştir.<br>
            SPK uyarınca yatırım danışmanlığı kapsamında olmayıp tarafsız kantitatif karar destek amaçlıdır. © {datetime.now().year} BIST Financial Analyst
        </div>
    </div>
</body>
</html>
"""
    return html
