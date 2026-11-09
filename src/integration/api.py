from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.config import TICKER_TO_COMPANY, REPORTS_DIR, get_company_for_ticker
from src.integration.models import (
    FundamentalReport,
    SentimentRankItem,
    SentimentRankResponse,
    SentimentSummaryResponse,
)
from src.integration.report_generator import FundamentalReportGenerator

app = FastAPI(
    title="BIST Temel Analiz & RAG Entegrasyon Servisi",
    description="BIST-Trend-Forecaster karar destek sistemi için KAP ve finansal rapor özetlerini JSON formatında sağlayan API.",
    version="1.0.0",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.integration.api:app", host="0.0.0.0", port=8000, reload=True)
