from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.config import TICKER_TO_COMPANY, get_company_for_ticker
from src.integration.models import FundamentalReport
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.integration.api:app", host="0.0.0.0", port=8000, reload=True)
