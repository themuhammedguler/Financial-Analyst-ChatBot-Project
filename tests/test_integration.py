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

    def test_company_filter_alphabetical_sorting(self):
        from src.config import COMPANIES
        _tr_order = "aAbBcCçÇdDeEfFgGğĞhHıIİijJkKlLmMnNoOöÖpPrRsSşŞtTuUüÜvVyYzZ"
        _tr_dict = {c: i for i, c in enumerate(_tr_order)}
        def _tr_sort_key(text: str):
            return [_tr_dict.get(c, ord(c) + 1000) for c in text]

        sorted_companies = sorted(list(COMPANIES.keys()), key=_tr_sort_key)
        company_options = ["Tümü"] + sorted_companies
        assert company_options[0] == "Tümü"
        # First company should start with 'A'
        assert company_options[1].startswith("A")
        # Ensure Turkish C / Ç ordering
        c_index = next(i for i, name in enumerate(company_options) if name.startswith("C"))
        ç_index = next(i for i, name in enumerate(company_options) if name.startswith("Ç"))
        assert c_index < ç_index



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
        assert hasattr(report, "recent_disclosures")
        assert len(report.recent_disclosures) >= 2
        for disc in report.recent_disclosures:
            assert hasattr(disc, "Tarih")
            assert hasattr(disc, "Kategori")
            assert hasattr(disc, "Başlık")
            assert hasattr(disc, "Etki")

    def test_garan_and_eregl_recent_disclosures_valid(self):
        for ticker in ["GARAN", "EREGL"]:
            rep_path = REPORTS_DIR / f"{ticker}_latest.json"
            assert rep_path.exists()
            report = FundamentalReport.load_from_file(rep_path)
            assert len(report.recent_disclosures) >= 2
            for disc in report.recent_disclosures:
                assert disc.Tarih is not None
                assert disc.Kategori is not None
                assert disc.Başlık is not None
                assert disc.Etki in ["Pozitif", "Nötr", "Negatif"]

    def test_recent_disclosures_catalog_and_fallback(self):
        from src.kap_feed import get_company_recent_disclosures
        thyao_disclosures = get_company_recent_disclosures("THYAO")
        assert len(thyao_disclosures) >= 2
        assert thyao_disclosures[0]["Tarih"] == "2026-11-05"
        assert "filo" in thyao_disclosures[0]["Başlık"].lower()

        fallback_disclosures = get_company_recent_disclosures("XYZ99")
        assert len(fallback_disclosures) >= 1
        assert "Tarih" in fallback_disclosures[0]
        assert "Kategori" in fallback_disclosures[0]
        assert "Başlık" in fallback_disclosures[0]
        assert "Etki" in fallback_disclosures[0]

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
        assert "recent_disclosures" in data
        assert len(data["recent_disclosures"]) >= 2
        assert "Tarih" in data["recent_disclosures"][0]
        assert "Kategori" in data["recent_disclosures"][0]
        assert "Başlık" in data["recent_disclosures"][0]
        assert "Etki" in data["recent_disclosures"][0]

    def test_sentiment_summary_endpoint(self, client):
        response = client.get("/api/v1/sentiment/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_reports" in data
        assert data["total_reports"] >= 1
        assert "average_sentiment_score" in data
        assert "sentiment_breakdown" in data
        assert "positive_ratio" in data
        assert "top_bullish_tickers" in data
        assert "top_bearish_tickers" in data
        assert "available_tickers" in data
        assert isinstance(data["available_tickers"], list)
        assert len(data["top_bullish_tickers"]) <= 3
        assert len(data["top_bearish_tickers"]) <= 3

    def test_sentiment_top_bullish_endpoint(self, client):
        response = client.get("/api/v1/sentiment/top-bullish?limit=3")
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "bullish"
        assert data["limit"] == 3
        assert len(data["rankings"]) <= 3
        if data["rankings"]:
            # Check descending order
            scores = [r["sentiment_score"] for r in data["rankings"]]
            assert scores == sorted(scores, reverse=True)
            first_rank = data["rankings"][0]
            assert first_rank["rank"] == 1
            assert "ticker" in first_rank
            assert "sentiment_label" in first_rank
            assert "key_metrics" in first_rank

    def test_sentiment_top_bearish_endpoint(self, client):
        response = client.get("/api/v1/sentiment/top-bearish?limit=3")
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "bearish"
        assert data["limit"] == 3
        assert len(data["rankings"]) <= 3
        if data["rankings"]:
            # Check ascending order
            scores = [r["sentiment_score"] for r in data["rankings"]]
            assert scores == sorted(scores)
            first_rank = data["rankings"][0]
            assert first_rank["rank"] == 1
            assert "ticker" in first_rank
            assert "sentiment_label" in first_rank
            assert "key_metrics" in first_rank



class TestHallucinationGuardAndBenchmark:
    """Halüsinasyon Savunması, 2026 Desteği ve RAG Benchmark Testleri"""

    def test_supported_years_includes_2026(self):
        from src.config import SUPPORTED_YEARS
        assert 2025 in SUPPORTED_YEARS
        assert 2026 in SUPPORTED_YEARS
        assert SUPPORTED_YEARS[-1] == 2026

    def test_hallucination_guard_claim_extraction(self):
        from src.rag.guard import HallucinationGuard
        guard = HallucinationGuard()
        sample_text = "Akbank 2024 yılında net kârını %57 artırarak 66.49 milyar TL seviyesine çıkarmıştır."
        claims = guard.extract_numerical_claims(sample_text)
        assert len(claims) >= 2
        assert any("66.49" in c or "milyar" in c for c in claims)
        assert any("%57" in c or "57" in c for c in claims)

    def test_hallucination_guard_verification(self):
        from src.rag.guard import HallucinationGuard
        from langchain_core.documents import Document

        guard = HallucinationGuard()
        sources = [
            Document(page_content="Akbank 2024 net kârı 66.49 milyar TL ve sermaye yeterlilik rasyosu %20.5 olmuştur.")
        ]
        answer_valid = "Şirketin 2024 net kârı 66.49 milyar TL olarak açıklanmıştır."
        res_valid = guard.verify_claims(answer_valid, sources)
        assert res_valid["groundedness_score"] == 1.0
        assert res_valid["is_safe"] is True

        answer_hallucinated = "Şirket uzay turizminden 999 milyar TL gelir elde etmiştir."
        res_hallucinated = guard.verify_claims(answer_hallucinated, sources)
        assert res_hallucinated["groundedness_score"] < 0.5
        assert len(res_hallucinated["unverified_claims"]) > 0

    def test_benchmark_report_content(self):
        bm_path = REPORTS_DIR / "rag_benchmark_report.json"
        assert bm_path.exists(), "rag_benchmark_report.json dosyası mevcut olmalıdır."

        with open(bm_path, "r", encoding="utf-8") as bf:
            bm_data = json.load(bf)

        sm = bm_data["summary_metrics"]
        assert "faithfulness_score" in sm
        assert "hallucination_rate" in sm
        assert "trap_defense_success_percent" in sm
        assert sm["faithfulness_score"] > 0.70
        assert bm_data["total_test_cases"] >= 8


class TestHybridSignalAndComparison:
    """Hibrit Karar Motoru ve Çift Şirket Karşılaştırmalı Analiz Testleri"""

    def test_get_hybrid_signal_api_default(self):
        client = TestClient(app)
        response = client.get("/api/v1/signal/THYAO")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "THYAO"
        assert "composite_score" in data
        assert 0.0 <= data["composite_score"] <= 100.0
        assert data["decision"] in ["GÜÇLÜ AL", "AL", "TUT / NÖTR", "SAT", "GÜÇLÜ SAT"]
        assert data["kelly_allocation_percent"] >= 0.0
        assert "technical_trend" in data
        assert "rationale" in data

    def test_get_hybrid_signal_with_forecaster_params(self):
        client = TestClient(app)
        response = client.get("/api/v1/signal/THYAO?technical_trend=Yükseliş&technical_confidence=0.90")
        assert response.status_code == 200
        data = response.json()
        assert data["technical_trend"] == "Yükseliş"
        assert data["technical_score"] >= 80.0
        assert data["decision"] in ["GÜÇLÜ AL", "AL"]
        assert data["kelly_allocation_percent"] > 0.0

    def test_peer_comparison_filter_detection(self):
        from src.rag.engine import FinancialRAGEngine
        engine = FinancialRAGEngine.__new__(FinancialRAGEngine)
        
        query = "THY ile Tüpraş'ın 2024 yılı kârlılıklarını ve net kâr marjlarını karşılaştır"
        filters = engine.detect_filters(query)
        assert filters["is_comparison"] is True
        assert len(filters["companies"]) >= 2
        assert any("Türk Hava Yolları" in c for c in filters["companies"])
        assert any("Tüpraş" in c for c in filters["companies"])
        assert filters["year"] == 2024
