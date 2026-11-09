import os
import json
import pytest
from pathlib import Path
from pydantic import ValidationError

from src.config import (
    FORECASTER_BASE_URL,
    REPORTS_DIR,
    get_company_for_ticker,
    get_ticker_for_company,
    TICKER_TO_COMPANY,
)
from src.integration.models import KeyMetrics, FundamentalReport
from src.integration.api import app
from fastapi.testclient import TestClient


class TestQueryParamAndMapping:
    """Streamlit Deep-Linking ve Ticker Eşleştirme Testleri (1. ve 3. Adım)"""

    def test_get_company_for_ticker_valid(self):
        assert get_company_for_ticker("THYAO") == "Türk Hava Yolları"
        assert get_company_for_ticker("thyao") == "Türk Hava Yolları"
        assert get_company_for_ticker("AKBNK") == "Akbank"
        assert get_company_for_ticker("TUPRS") == "Tüpraş"

    def test_get_company_for_ticker_unknown(self):
        # Bilinmeyen hisse geldiğinde varsayılan fallback
        assert get_company_for_ticker("XYZ99") == "XYZ99"
        assert get_company_for_ticker("") == "Tümü"
        assert get_company_for_ticker(None) == "Tümü"

    def test_get_ticker_for_company(self):
        assert get_ticker_for_company("Türk Hava Yolları") == "THYAO"
        assert get_ticker_for_company("Akbank T.A.Ş.") == "AKBNK"
        assert get_ticker_for_company("Tüpraş Türkiye Petrol Rafinerileri") == "TUPRS"

    def test_cross_reference_url_construction(self):
        ticker = "THYAO"
        expected_url = f"{FORECASTER_BASE_URL}/?ticker={ticker}"
        assert expected_url == "https://bist-trend-forecaster.streamlit.app/?ticker=THYAO"


class TestFundamentalReportSchema:
    """Standart Yapılandırılmış Çıktı ve Pydantic Doğrulama Testleri (2. Adım)"""

    def test_valid_fundamental_report_creation(self):
        sample_data = {
            "ticker": "THYAO",
            "company_name": "Türk Hava Yolları",
            "report_date": "2026-11-08",
            "fundamental_summary": "Yolcu doluluk oranlarında güçlü artış ile FAVÖK beklentileri aştı.",
            "sentiment_score": 0.75,
            "sentiment_label": "Pozitif",
            "key_metrics": {
                "pe_ratio": 4.8,
                "pb_ratio": 1.15,
                "net_profit_margin_trend": "Artış",
                "debt_to_equity": "Düşük/Yönetilebilir"
            },
            "catalysts": [
                "Yeni uçak siparişleri ve filo büyümesi",
                "Yaz sezonu güçlü rezervasyonları"
            ],
            "risk_factors": [
                "Havacılık yakıt maliyetleri oynaklığı",
                "Küresel makroekonomik yavaşlama riski"
            ]
        }
        report = FundamentalReport(**sample_data)
        assert report.ticker == "THYAO"
        assert report.sentiment_score == 0.75
        assert report.sentiment_label == "Pozitif"
        assert len(report.catalysts) == 2
        assert len(report.risk_factors) == 2
        assert report.key_metrics.pe_ratio == 4.8

    def test_ticker_auto_uppercase_validation(self):
        report = FundamentalReport(
            ticker="thyao ",
            company_name="Türk Hava Yolları",
            fundamental_summary="Test özet",
            sentiment_score=0.5,
            sentiment_label="Pozitif",
            key_metrics=KeyMetrics()
        )
        assert report.ticker == "THYAO"

    def test_sentiment_score_bounds_validation(self):
        # -1.0 ile 1.0 aralığı dışındaki skorlar hata vermelidir
        with pytest.raises(ValidationError):
            FundamentalReport(
                ticker="THYAO",
                company_name="THY",
                fundamental_summary="Test",
                sentiment_score=1.5,  # Geçersiz: > 1.0
                sentiment_label="Pozitif",
                key_metrics=KeyMetrics()
            )

        with pytest.raises(ValidationError):
            FundamentalReport(
                ticker="THYAO",
                company_name="THY",
                fundamental_summary="Test",
                sentiment_score=-1.2,  # Geçersiz: < -1.0
                sentiment_label="Negatif",
                key_metrics=KeyMetrics()
            )

    def test_sentiment_label_allowed_values(self):
        # Yalnızca Pozitif, Nötr, Negatif kabul edilmeli
        with pytest.raises(ValidationError):
            FundamentalReport(
                ticker="THYAO",
                company_name="THY",
                fundamental_summary="Test",
                sentiment_score=0.2,
                sentiment_label="Çokİyi",  # Geçersiz etiket
                key_metrics=KeyMetrics()
            )

    def test_file_serialization_and_deserialization(self, tmp_path):
        report = FundamentalReport(
            ticker="AKBNK",
            company_name="Akbank",
            fundamental_summary="Güçlü sermaye yeterlilik rasyosu ve artan net faiz marjı.",
            sentiment_score=0.6,
            sentiment_label="Pozitif",
            key_metrics=KeyMetrics(pe_ratio=3.9, pb_ratio=0.88, net_profit_margin_trend="Artış"),
            catalysts=["Kredi genişlemesi"],
            risk_factors=["Mevduat maliyetleri"]
        )
        file_path = tmp_path / "AKBNK_test.json"
        saved = report.save_to_file(file_path)
        assert saved.exists()

        loaded = FundamentalReport.load_from_file(saved)
        assert loaded.ticker == "AKBNK"
        assert loaded.sentiment_score == 0.6
        assert loaded.key_metrics.pe_ratio == 3.9


class TestGeneratedReportFiles:
    """Diske Kaydedilen Rapor Dosyalarının Doğrulanması"""

    def test_thyao_latest_report_valid(self):
        thyao_path = REPORTS_DIR / "THYAO_latest.json"
        assert thyao_path.exists(), "THYAO_latest.json dosyası mevcut olmalıdır."

        report = FundamentalReport.load_from_file(thyao_path)
        assert report.ticker == "THYAO"
        assert report.company_name == "Türk Hava Yolları"
        assert -1.0 <= report.sentiment_score <= 1.0
        assert report.sentiment_label in ["Pozitif", "Nötr", "Negatif"]
        assert len(report.fundamental_summary) > 20
        assert len(report.catalysts) > 0
        assert len(report.risk_factors) > 0

    def test_akbnk_latest_report_valid(self):
        akbnk_path = REPORTS_DIR / "AKBNK_latest.json"
        assert akbnk_path.exists(), "AKBNK_latest.json dosyası mevcut olmalıdır."

        report = FundamentalReport.load_from_file(akbnk_path)
        assert report.ticker == "AKBNK"
        assert -1.0 <= report.sentiment_score <= 1.0
        assert report.sentiment_label in ["Pozitif", "Nötr", "Negatif"]
        assert len(report.fundamental_summary) > 20


class TestFastAPIIntegration:
    """FastAPI REST Servisi Testleri (BIST-Trend-Forecaster Arka Plan Veri Tüketimi)"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data

    def test_tickers_endpoint(self, client):
        response = client.get("/api/v1/tickers")
        assert response.status_code == 200
        data = response.json()
        assert "tickers" in data
        tickers_list = [item["ticker"] for item in data["tickers"]]
        assert "THYAO" in tickers_list
        assert "AKBNK" in tickers_list
        assert "TUPRS" in tickers_list

    def test_fundamental_endpoint_cached(self, client):
        # Mevcut THYAO raporunu API üzerinden çek
        response = client.get("/api/v1/fundamental/THYAO?force_refresh=false")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "THYAO"
        assert -1.0 <= data["sentiment_score"] <= 1.0
        assert data["sentiment_label"] in ["Pozitif", "Nötr", "Negatif"]
        assert "key_metrics" in data
        assert "catalysts" in data
        assert "risk_factors" in data
