from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import TICKER_TO_COMPANY, REPORTS_DIR, get_company_for_ticker
from src.integration.models import (
    FundamentalReport,
    SentimentRankItem,
    SentimentRankResponse,
    SentimentSummaryResponse,
    HybridSignalResponse,
)
from src.integration.report_generator import FundamentalReportGenerator
from src.tools.financial_tools import (
    tool_calculate_dupont,
    tool_calculate_altman_z,
    tool_calculate_dcf,
    tool_calculate_kelly_allocation,
    tool_calculate_macro_sensitivity,
    tool_compare_peers,
    tool_get_kap_disclosures,
    tool_get_multiyear_trend,
)
from src.dividend_analysis import calculate_dividend_metrics

app = FastAPI(
    title="BIST Temel Analiz & RAG Entegrasyon Servisi",
    description="BIST-Trend-Forecaster karar destek sistemi için KAP, hibrit RAG, DuPont, Altman Z-Skoru, DCF, Makro Stres Testi ve Kelly portföy tahsisini JSON formatında sağlayan API.",
    version="2.0.0",
)

# Cross-origin desteği (BIST-Trend-Forecaster Streamlit ve web istemcileri için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

report_generator = FundamentalReportGenerator()


def get_all_reports() -> List[FundamentalReport]:
    """Diskteki tüm *_latest.json raporlarını yükler."""
    reports: List[FundamentalReport] = []
    if not REPORTS_DIR.exists():
        return reports
    for p in REPORTS_DIR.glob("*_latest.json"):
        try:
            reports.append(FundamentalReport.load_from_file(p))
        except Exception:
            continue
    return reports


@app.get("/health", tags=["Status"])
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "BIST-Fundamental-RAG-API"}


@app.get("/api/v1/tickers", tags=["Tickers"])
def list_supported_tickers() -> Dict[str, Any]:
    """Sistemde desteklenen BIST hisselerinin listesini döner."""
    return {
        "count": len(TICKER_TO_COMPANY),
        "tickers": [
            {"ticker": k, "company_name": v}
            for k, v in TICKER_TO_COMPANY.items()
        ]
    }


@app.get(
    "/api/v1/fundamental/{ticker}",
    response_model=FundamentalReport,
    tags=["Fundamental Analysis"]
)
def get_fundamental_analysis(
    ticker: str,
    force_refresh: bool = Query(
        default=False, 
        description="Önbellekteki rapor yerine yeniden RAG analizi üretmeye zorla"
    )
) -> FundamentalReport:
    """
    Belirtilen BIST hissesi için KAP ve finansal rapor özetini,
    duygu skorunu (sentiment_score) ve temel göstergeleri döner.
    """
    ticker_clean = ticker.upper().strip()
    try:
        report = report_generator.generate_report(ticker=ticker_clean, force_refresh=force_refresh)
        return report
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"{ticker_clean} için temel analiz raporu üretilirken hata oluştu: {str(e)}"
        )


@app.get(
    "/api/v1/sentiment/summary",
    response_model=SentimentSummaryResponse,
    tags=["Sentiment Analysis"]
)
def get_sentiment_summary() -> SentimentSummaryResponse:
    """
    Piyasa genelindeki tüm üretilmiş raporlar üzerinden duygu (sentiment) analiz özetini döner.
    Ortalama skor, pozitif/nötr/negatif dağılımı ve en yüksek/en düşük hisseleri içerir.
    """
    reports = get_all_reports()
    if not reports:
        return SentimentSummaryResponse(
            total_reports=0,
            average_sentiment_score=0.0,
            sentiment_breakdown={"Pozitif": 0, "Nötr": 0, "Negatif": 0},
            positive_ratio=0.0,
            top_bullish_tickers=[],
            top_bearish_tickers=[],
            available_tickers=[]
        )

    total = len(reports)
    avg_score = round(sum(r.sentiment_score for r in reports) / total, 3)
    breakdown: Dict[str, int] = {"Pozitif": 0, "Nötr": 0, "Negatif": 0}
    for r in reports:
        breakdown[r.sentiment_label] = breakdown.get(r.sentiment_label, 0) + 1

    sorted_desc = sorted(reports, key=lambda x: x.sentiment_score, reverse=True)
    sorted_asc = sorted(reports, key=lambda x: x.sentiment_score)

    top_bullish = [r.ticker for r in sorted_desc[:3]]
    top_bearish = [r.ticker for r in sorted_asc[:3]]
    all_tickers = sorted([r.ticker for r in reports])
    pos_ratio = round(breakdown["Pozitif"] / total, 3) if total > 0 else 0.0

    return SentimentSummaryResponse(
        total_reports=total,
        average_sentiment_score=avg_score,
        sentiment_breakdown=breakdown,
        positive_ratio=pos_ratio,
        top_bullish_tickers=top_bullish,
        top_bearish_tickers=top_bearish,
        available_tickers=all_tickers
    )


@app.get(
    "/api/v1/sentiment/top-bullish",
    response_model=SentimentRankResponse,
    tags=["Sentiment Analysis"]
)
def get_top_bullish_stocks(
    limit: int = Query(default=5, ge=1, le=50, description="Döndürülecek maksimum hisse sayısı")
) -> SentimentRankResponse:
    """
    KAP ve bilanço analizlerine göre en pozitif duygu skoruna (sentiment_score) sahip hisseleri döner.
    """
    reports = get_all_reports()
    sorted_reports = sorted(reports, key=lambda x: x.sentiment_score, reverse=True)[:limit]

    rankings = [
        SentimentRankItem(
            rank=idx + 1,
            ticker=r.ticker,
            company_name=r.company_name,
            sentiment_score=r.sentiment_score,
            sentiment_label=r.sentiment_label,
            fundamental_summary=r.fundamental_summary,
            key_metrics=r.key_metrics,
            catalysts=r.catalysts,
            risk_factors=r.risk_factors,
            report_date=r.report_date
        )
        for idx, r in enumerate(sorted_reports)
    ]

    return SentimentRankResponse(
        category="bullish",
        count=len(rankings),
        limit=limit,
        rankings=rankings
    )


@app.get(
    "/api/v1/sentiment/top-bearish",
    response_model=SentimentRankResponse,
    tags=["Sentiment Analysis"]
)
def get_top_bearish_stocks(
    limit: int = Query(default=5, ge=1, le=50, description="Döndürülecek maksimum hisse sayısı")
) -> SentimentRankResponse:
    """
    KAP ve bilanço analizlerine göre en düşük veya negatif duygu skoruna (sentiment_score) sahip hisseleri döner.
    """
    reports = get_all_reports()
    sorted_reports = sorted(reports, key=lambda x: x.sentiment_score)[:limit]

    rankings = [
        SentimentRankItem(
            rank=idx + 1,
            ticker=r.ticker,
            company_name=r.company_name,
            sentiment_score=r.sentiment_score,
            sentiment_label=r.sentiment_label,
            fundamental_summary=r.fundamental_summary,
            key_metrics=r.key_metrics,
            catalysts=r.catalysts,
            risk_factors=r.risk_factors,
            report_date=r.report_date
        )
        for idx, r in enumerate(sorted_reports)
    ]

    return SentimentRankResponse(
        category="bearish",
        count=len(rankings),
        limit=limit,
        rankings=rankings
    )


@app.get(
    "/api/v1/signal/{ticker}",
    response_model=HybridSignalResponse,
    tags=["Hybrid Decision Engine"]
)
def get_hybrid_signal(
    ticker: str,
    technical_trend: Optional[str] = Query(default=None, description="BIST-Trend-Forecaster teknik trendi: Yükseliş | Yatay | Düşüş"),
    technical_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Teknik model güven skoru (0.0 - 1.0)"),
) -> HybridSignalResponse:
    """
    RAG Temel Analiz Duygusu (%60) ile BIST-Trend-Forecaster Teknik Analiz Tahminini (%40)
    birleştirerek hibrit yatırım kararı ve Kelly Sermaye Tahsis oranı üretir.
    """
    ticker_clean = ticker.upper().strip()
    try:
        report = report_generator.generate_report(ticker=ticker_clean, force_refresh=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{ticker_clean} raporu alınamadı: {e}")

    # 1. Temel Analiz Skoru (0-100 ölçeğine normalize)
    fund_score = round(((report.sentiment_score + 1.0) / 2.0) * 100, 1)

    # 2. Teknik Analiz Skoru
    if isinstance(technical_trend, str) and technical_trend.strip():
        t_lower = technical_trend.lower()
        conf = technical_confidence if isinstance(technical_confidence, (int, float)) else 0.75
        if "yüksel" in t_lower or "bull" in t_lower or "al" in t_lower:
            tech_score = round(60.0 + (conf * 35.0), 1)
            t_label = "Yükseliş"
        elif "düş" in t_lower or "bear" in t_lower or "sat" in t_lower:
            tech_score = round(40.0 - (conf * 30.0), 1)
            t_label = "Düşüş"
        else:
            tech_score = 50.0
            t_label = "Yatay"
    else:
        tech_score = round(fund_score * 0.9 + 5.0, 1)
        if tech_score >= 60.0:
            t_label = "Yükseliş"
        elif tech_score <= 42.0:
            t_label = "Düşüş"
        else:
            t_label = "Yatay"

    # 3. Bileşik Karar Skoru (%60 Temel + %40 Teknik)
    comp_score = round((0.60 * fund_score) + (0.40 * tech_score), 1)

    # 4. Tavsiye Sinyali ve Kelly Sermaye Tahsisi
    if comp_score >= 76.0:
        decision = "GÜÇLÜ AL"
        kelly = round(min(20.0, 8.0 + (comp_score - 76.0) * 0.5), 1)
        rationale = f"Güçlü temel göstergeler ({report.sentiment_label}, Skor: {fund_score}) ve pozitif teknik trend ({t_label}, Skor: {tech_score}) hisseyi yüksek güvenlik marjıyla desteklemektedir."
    elif comp_score >= 60.0:
        decision = "AL"
        kelly = round(min(12.0, 4.0 + (comp_score - 60.0) * 0.5), 1)
        rationale = f"Pozitif temel görünüm ve teknik ivme ({t_label}) hissede yukarı yönlü potansiyeli artırmaktadır."
    elif comp_score >= 45.0:
        decision = "TUT / NÖTR"
        kelly = 0.0
        rationale = f"Temel rasyolar dengeli olmakla birlikte teknik momentum yatay seyretmektedir. Yeni pozisyon açmadan önce katalizörler beklenmelidir."
    elif comp_score >= 30.0:
        decision = "SAT"
        kelly = 0.0
        rationale = f"Kârlılık veya borçluluk baskısı ve negatif teknik sinyaller ({t_label}) hissede aşağı yönlü risk oluşturmaktadır."
    else:
        decision = "GÜÇLÜ SAT"
        kelly = 0.0
        rationale = f"Zayıf temel finansal sağlık ve negatif teknik momentum hissede sermaye kaybı riskini artırmaktadır."

    from datetime import date
    return HybridSignalResponse(
        ticker=ticker_clean,
        company_name=report.company_name,
        fundamental_score=fund_score,
        fundamental_sentiment=report.sentiment_label,
        technical_score=tech_score,
        technical_trend=t_label,
        composite_score=comp_score,
        decision=decision,
        kelly_allocation_percent=kelly,
        rationale=rationale,
        report_date=date.today().isoformat()
    )


class AskRequest(BaseModel):
    """BIST-Trend-Forecaster üzerinden gönderilen doğal dil analiz sorgusu."""
    question: str = Field(..., description="Analist sorusu (Örn: THY 2024 uçak sayısı kaçtır?)")
    company_filter: Optional[str] = Field(default=None, description="Şirket filtresi (Örn: Türk Hava Yolları)")
    year_filter: Optional[int] = Field(default=None, description="Mali yıl filtresi (Örn: 2024)")
    provider: str = Field(default="gemini", description="LLM sağlayıcısı: gemini | colab | ollama")
    base_url: Optional[str] = Field(default=None, description="Colab / Özel sunucu API adresi")


@app.get(
    "/api/v1/quant/{ticker}",
    tags=["Quantitative & Valuation Engine"]
)
def get_quantitative_metrics(
    ticker: str,
    growth_rate_pct: float = Query(default=15.0, description="DCF büyüme beklentisi (%)"),
    discount_rate_pct: float = Query(default=24.0, description="DCF WACC iskonto oranı (%)"),
) -> Dict[str, Any]:
    """
    BIST-Trend-Forecaster için tek çağrıda tüm deterministik kantitatif metrikleri döner:
    - DuPont 3-Aşamalı ROE Ayrıştırması
    - Altman Z-Skoru İflas & Dayanıklılık Analizi
    - İndirgenmiş Nakit Akımları (DCF) Adil Değerleme
    - Temettü & Serbest Nakit Akışı (FCF) Güvenlik Skoru
    - Kelly Kriteri Optimal Sermaye Tahsisi
    - Resmi KAP Bildirimleri
    """
    sym = ticker.upper().replace(".IS", "").strip()
    return {
        "ticker": sym,
        "company_name": get_company_for_ticker(sym),
        "dupont": tool_calculate_dupont(sym),
        "altman_z": tool_calculate_altman_z(sym),
        "dcf_valuation": tool_calculate_dcf(sym, growth_rate_pct=growth_rate_pct, discount_rate_pct=discount_rate_pct),
        "dividend_safety": calculate_dividend_metrics(sym),
        "kelly_allocation": tool_calculate_kelly_allocation(sym),
        "kap_disclosures": tool_get_kap_disclosures(sym, limit=3),
    }


@app.get(
    "/api/v1/macro-stress/{ticker}",
    tags=["Quantitative & Valuation Engine"]
)
def get_macro_stress_test(
    ticker: str,
    usd_change_pct: float = Query(default=10.0, description="USD/TRY şok yüzdesi (%)"),
    rate_change_bps: float = Query(default=250.0, description="Politika faizi değişimi (baz puan)"),
    oil_change_pct: float = Query(default=15.0, description="Brent petrol değişimi (%)"),
) -> Dict[str, Any]:
    """
    Döviz kuru, TCMB politika faizi ve Brent petrol şoklarının hisse net kârı üzerindeki
    etkisini (Tornado duyarlılık simülasyonu) hesaplar.
    """
    sym = ticker.upper().replace(".IS", "").strip()
    return tool_calculate_macro_sensitivity(
        ticker=sym,
        usd_change_pct=usd_change_pct,
        rate_change_bps=rate_change_bps,
        oil_change_pct=oil_change_pct,
    )


@app.get(
    "/api/v1/compare/{ticker_a}/{ticker_b}",
    tags=["Quantitative & Valuation Engine"]
)
def compare_two_stocks(ticker_a: str, ticker_b: str) -> Dict[str, Any]:
    """
    İki BIST hissesini DuPont kârlılık marjları, finansal kaldıraç ve Altman Z-skoru
    açısından eşler arası (peer-to-peer) kıyaslar.
    """
    return tool_compare_peers(ticker_a, ticker_b)


@app.get(
    "/api/v1/trend/{ticker}",
    tags=["Quantitative & Valuation Engine"]
)
def get_multiyear_trend_analysis(
    ticker: str,
    start_year: int = Query(default=2020, ge=2020, le=2026, description="Başlangıç yılı (2020)"),
    end_year: int = Query(default=2026, ge=2020, le=2026, description="Bitiş yılı (2026)"),
) -> Dict[str, Any]:
    """
    Belirtilen BIST hissesinin 2020–2026 (7 Yıllık) KAP gerçekleşen verilerini ve
    2025T–2026T analist projeksiyon serisini (Hasılat, Net Kâr, ROE, Borç/Özkaynak ve CAGR) döner.
    """
    return tool_get_multiyear_trend(ticker=ticker, start_year=start_year, end_year=end_year)


@app.post(
    "/api/v1/ask",
    tags=["RAG & Hybrid Analyst Q&A"]
)
def ask_financial_analyst(req: AskRequest) -> Dict[str, Any]:
    """
    Hibrit RAG (FAISS + BM25 + Aynı Sayfa Tablo Birleştirme) ve Deterministik Finansal Motoru
    kullanarak doğal dildeki soruları kaynak kanıtlarıyla yanıtlar.
    """
    from src.rag.engine import FinancialRAGEngine
    engine = FinancialRAGEngine(
        provider=req.provider,
        base_url=req.base_url,
    )
    res = engine.answer_query(
        question=req.question,
        company_filter=req.company_filter,
        year_filter=req.year_filter,
    )
    return {
        "question": req.question,
        "answer": res.get("answer", ""),
        "guard_result": res.get("guard_result", {}),
        "sources": [
            {
                "company": d.metadata.get("company"),
                "year": d.metadata.get("year"),
                "file_name": d.metadata.get("file_name"),
                "page": d.metadata.get("page"),
            }
            for d in res.get("docs", [])
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.integration.api:app", host="0.0.0.0", port=8000, reload=True)
