"""Unit & Integration Tests for New Features:
1. Live Market Data & Candlestick Charts (src/market_data.py)
2. Peer-to-Peer Company Comparison & Dual Radar (src/comparison.py)
3. Executive Investment Research Report Export (src/export_report.py)
4. Real-time KAP Special Disclosures Feed (src/kap_feed.py)
"""

import pytest
import pandas as pd
from datetime import datetime

from src.integration.models import FundamentalReport, KeyMetrics
from src.market_data import calculate_rsi, create_candlestick_chart
from src.comparison import (
    get_5d_profile_scores,
    create_comparison_radar_chart,
    generate_peer_comparison_summary
)
from src.export_report import generate_executive_html_report
from src.kap_feed import get_recent_kap_disclosures


@pytest.fixture
def sample_reports():
    """Mock fundamental reports for testing."""
    rep_a = FundamentalReport(
        ticker="THYAO",
        company_name="Türk Hava Yolları",
        report_date="2026-11-09",
        fundamental_summary="Yolcu doluluk oranlarında güçlü artış ile kârlılık rekor seviyede.",
        sentiment_score=0.82,
        sentiment_label="Pozitif",
        key_metrics=KeyMetrics(
            pe_ratio=4.8,
            pb_ratio=1.15,
            net_profit_margin_trend="Artış",
            debt_to_equity="Düşük/Yönetilebilir"
        ),
        catalysts=["Uçak filosu genişlemesi", "Kargo gelirlerinde artış"],
        risk_factors=["Jet yakıtı maliyet dalgalanmaları"]
    )

    rep_b = FundamentalReport(
        ticker="PGSUS",
        company_name="Pegasus",
        report_date="2026-11-09",
        fundamental_summary="Düşük maliyetli uçuş modeliyle yan gelirlerde güçlü büyüme.",
        sentiment_score=0.68,
        sentiment_label="Pozitif",
        key_metrics=KeyMetrics(
            pe_ratio=5.4,
            pb_ratio=1.45,
            net_profit_margin_trend="Artış",
            debt_to_equity="Orta"
        ),
        catalysts=["Yeni hat açılışları", "Filo yenileme"],
        risk_factors=["Döviz kuru oynaklığı", "Havalimanı yer hizmetleri maliyetleri"]
    )
    return rep_a, rep_b


class TestMarketDataFeatures:
    """Feature 1: Test Market Data and Technical Indicators"""

    def test_calculate_rsi(self):
        prices = pd.Series([100 + i * (1 if i % 2 == 0 else -0.5) for i in range(30)])
        rsi = calculate_rsi(prices, period=14)
        assert len(rsi) == 30
        valid_rsi = rsi.dropna()
        assert not valid_rsi.empty
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_create_candlestick_chart(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "Open": [100.0 + i for i in range(10)],
            "High": [105.0 + i for i in range(10)],
            "Low": [98.0 + i for i in range(10)],
            "Close": [103.0 + i for i in range(10)],
            "Volume": [1000000 + i * 50000 for i in range(10)],
            "SMA20": [100.0 for _ in range(10)],
            "SMA50": [99.0 for _ in range(10)],
            "SMA200": [95.0 for _ in range(10)],
        }, index=dates)

        fig = create_candlestick_chart(df, "THYAO", "Türk Hava Yolları")
        assert fig is not None
        assert len(fig.data) >= 4  # Candlestick + 3 SMAs + Volume


class TestComparisonFeatures:
    """Feature 2: Test Company Peer-to-Peer Comparison"""

    def test_5d_profile_scores(self, sample_reports):
        rep_a, _ = sample_reports
        scores = get_5d_profile_scores(rep_a)
        assert "Kârlılık" in scores
        assert "Finansal Sağlık" in scores
        assert "Büyüme Katalizörleri" in scores
        assert "Piyasa Duygusu" in scores
        assert "Risk Dayanıklılığı" in scores
        assert all(0 <= v <= 100 for v in scores.values())

    def test_create_comparison_radar_chart(self, sample_reports):
        rep_a, rep_b = sample_reports
        fig = create_comparison_radar_chart(rep_a, rep_b)
        assert fig is not None
        assert len(fig.data) == 2  # 2 traces (Company A and Company B)

    def test_generate_peer_comparison_summary(self, sample_reports):
        rep_a, rep_b = sample_reports
        summary = generate_peer_comparison_summary(rep_a, rep_b)
        assert summary["ticker1"] == "THYAO"
        assert summary["ticker2"] == "PGSUS"
        assert summary["score1"] > 0
        assert summary["score2"] > 0
        assert "verdict_text" in summary
        assert len(summary["verdict_text"]) > 10


class TestExecutiveReportExport:
    """Feature 3: Test Executive HTML / Printable Report Export"""

    def test_generate_executive_html_report(self, sample_reports):
        rep_a, _ = sample_reports
        m_data = {
            "current_price": 312.50,
            "market_cap_str": "428.5 Milyar TL",
            "pe_ratio": 4.8,
            "pb_ratio": 1.15
        }
        h_signal = {
            "decision": "TAVSİYE: GÜÇLÜ AL",
            "kelly_pct": 24.5
        }
        html = generate_executive_html_report(rep_a, market_data=m_data, hybrid_signal=h_signal)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "THYAO" in html
        assert "Türk Hava Yolları" in html
        assert "312.50 TL" in html
        assert "428.5 Milyar TL" in html
        assert "GÜÇLÜ AL" in html
        assert "%24.5" in html


class TestKapFeedFeatures:
    """Feature 4: Test Real-Time KAP Disclosures Feed"""

    def test_get_recent_kap_disclosures_all(self):
        feed = get_recent_kap_disclosures()
        assert isinstance(feed, list)
        assert len(feed) > 0
        first = feed[0]
        assert "ticker" in first
        assert "title" in first
        assert "category" in first
        assert "sentiment" in first
        assert "badge_html" in first
        assert "kap_url" in first

    def test_get_recent_kap_disclosures_filtered_by_ticker(self):
        feed = get_recent_kap_disclosures(ticker_filter="THYAO")
        assert len(feed) >= 1
        assert all(item["ticker"] == "THYAO" for item in feed)

    def test_get_recent_kap_disclosures_filtered_by_category(self):
        feed = get_recent_kap_disclosures(category_filter="Yeni İş İlişkisi")
        assert len(feed) >= 1
        assert all("Yeni İş İlişkisi" in item["category"] for item in feed)
