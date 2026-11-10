"""BIST Sektörel Liderlik & Rasyo Filtreleme Motoru (Screener / Leaderboard).

BIST 100 hisselerini; DuPont ROE, Altman Z-Skoru, F/K, PD/DD,
Temel Sağlık Skoru ve Piyasa Duygusuna göre tarar, filtreler ve sıralar.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from src.data_collection.kap_downloader import BIST_100_COMPANIES
from src.integration.models import FundamentalReport
from src.config import REPORTS_DIR
from src.dupont_analysis import calculate_dupont_model, calculate_altman_z_score
from src.dividend_analysis import DEFAULT_DIVIDEND_PROFILES

logger = logging.getLogger(__name__)


def build_screener_universe() -> pd.DataFrame:
    """
    Tüm BIST hisselerinin bilanço rasyolarını, DuPont kârlılığını,
    Altman Z-skorunu ve temel rapor metriklerini tek bir veri tablosunda birleştirir.
    """
    rows: List[Dict[str, Any]] = []

    for ticker, info in BIST_100_COMPANIES.items():
        comp_name = info.get("name", ticker)
        sector = info.get("sector", "Genel")

        # 1. Rapor dosyasından temel verileri al
        rep_file = REPORTS_DIR / f"{ticker}_latest.json"
        sentiment_score = 0.0
        sentiment_label = "Nötr"
        composite_score = 65.0
        pe_ratio = 7.5
        pb_ratio = 2.1
        decision = "TUT"

        if rep_file.exists():
            try:
                rep = FundamentalReport.load_from_file(rep_file)
                sentiment_score = rep.sentiment_score
                sentiment_label = rep.sentiment_label
                if rep.key_metrics:
                    if rep.key_metrics.pe_ratio:
                        pe_ratio = float(rep.key_metrics.pe_ratio)
                    if rep.key_metrics.pb_ratio:
                        pb_ratio = float(rep.key_metrics.pb_ratio)
            except Exception:
                pass

        # 2. DuPont ve Altman Z-Skoru hesapla
        dupont = calculate_dupont_model(ticker)
        altman = calculate_altman_z_score(ticker, dupont)

        roe = dupont.get("roe_pct", 15.0)
        net_margin = dupont.get("net_profit_margin_pct", 8.0)
        z_score = altman.get("z_score", 1.8)
        zone = altman.get("zone", "Gri Bölge (Grey Zone)")

        # Birleşik Kalite Skoru (ROE %30 + Z-Skor %30 + Temel Duygu %40)
        z_norm = min(100.0, max(0.0, (z_score / 3.5) * 100))
        roe_norm = min(100.0, max(0.0, (roe / 35.0) * 100))
        sent_norm = ((sentiment_score + 1.0) / 2.0) * 100
        composite_score = round(roe_norm * 0.35 + z_norm * 0.35 + sent_norm * 0.30, 1)

        # Karar Sinyali
        if composite_score >= 80 and z_score >= 2.0:
            decision = "GÜÇLÜ AL"
        elif composite_score >= 68:
            decision = "AL"
        elif composite_score <= 45 or z_score < 1.1:
            decision = "RİSKLİ / AZALT"
        else:
            decision = "TUT"

        div_prof = DEFAULT_DIVIDEND_PROFILES.get(ticker, {"yield_pct": 0.0})
        div_yield = div_prof.get("yield_pct", 0.0)

        rows.append({
            "ticker": ticker,
            "company_name": comp_name,
            "sector": sector,
            "roe_pct": round(roe, 1),
            "net_margin_pct": round(net_margin, 1),
            "dividend_yield_pct": round(div_yield, 1),
            "z_score": round(z_score, 2),
            "zone": zone,
            "pe_ratio": round(pe_ratio, 2),
            "pb_ratio": round(pb_ratio, 2),
            "sentiment_score": round(sentiment_score, 2),
            "sentiment_label": sentiment_label,
            "composite_score": composite_score,
            "decision": decision
        })

    return pd.DataFrame(rows)


def filter_and_rank_screener(
    df: pd.DataFrame,
    sector: str = "Tümü",
    min_roe: float = 0.0,
    min_z_score: float = 0.0,
    max_pe: float = 100.0,
    sort_by: str = "composite_score",
    ascending: bool = False,
    top_n: int = 25
) -> pd.DataFrame:
    """
    Belirlenen finansal kriterlere ve sektöre göre hisseleri filtreleyip sıralar.
    """
    filtered = df.copy()

    if sector != "Tümü":
        filtered = filtered[filtered["sector"] == sector]

    filtered = filtered[
        (filtered["roe_pct"] >= min_roe) &
        (filtered["z_score"] >= min_z_score) &
        (filtered["pe_ratio"] <= max_pe)
    ]

    if sort_by in filtered.columns:
        filtered = filtered.sort_values(by=sort_by, ascending=ascending)

    return filtered.head(top_n).reset_index(drop=True)
