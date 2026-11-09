import os
import sys
import argparse
from pathlib import Path
from typing import List

# Proje kökünü sys.path'e ekle
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.integration.report_generator import FundamentalReportGenerator
from src.integration.models import FundamentalReport
from src.config import REPORTS_DIR


def generate_batch_reports(tickers: List[str], force_refresh: bool = False):
    """
    Belirtilen hisseler için BIST-Trend-Forecaster standartlarında
    yapılandırılmış JSON temel analiz raporları üretir ve kaydeder.
    """
    generator = FundamentalReportGenerator()
    print(f"\n=======================================================")
    print(f"📊 BIST Toplu Temel Analiz JSON Raporu Üretimi ({len(tickers)} Şirket)")
    print(f"=======================================================\n")

    summary_rows = []

    for idx, ticker in enumerate(tickers):
        ticker_clean = ticker.upper().strip()
        print(f"[{idx+1}/{len(tickers)}] {ticker_clean} için RAG analizi yürütülüyor...")
        
        try:
            report: FundamentalReport = generator.generate_report(ticker=ticker_clean, force_refresh=force_refresh)
            summary_rows.append({
                "ticker": report.ticker,
                "company": report.company_name[:24],
                "date": report.report_date,
                "sentiment_score": report.sentiment_score,
                "sentiment_label": report.sentiment_label,
                "catalysts_count": len(report.catalysts),
                "risks_count": len(report.risk_factors),
                "path": str(REPORTS_DIR / f"{report.ticker}_latest.json")
            })
            print(f"       ✓ {report.ticker} Raporu Hazır: {report.sentiment_label} ({report.sentiment_score})")
        except Exception as e:
            print(f"       ✗ {ticker_clean} üretilirken hata: {e}")

    # Özet Tablo
    print(f"\n=======================================================")
    print(f"📋 ÜRETİLEN RAPORLAR LİSTESİ (BIST-Trend-Forecaster Servisi)")
    print(f"=======================================================")
    print(f"{'Hisse':<8} | {'Şirket Adı':<25} | {'Duygu Skoru':<12} | {'Etiket':<10} | {'Katalizör':<10} | {'Riskler'}")
    print("-" * 80)
    for row in summary_rows:
        print(f"{row['ticker']:<8} | {row['company']:<25} | {row['sentiment_score']:<12} | {row['sentiment_label']:<10} | {row['catalysts_count']:<10} | {row['risks_count']}")
    print("-" * 80)
    print(f"\nTüm raporlar '{REPORTS_DIR}' dizininde hazır.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toplu Temel Analiz JSON Raporu Üretici")
    parser.add_argument(
        "--tickers", 
        nargs="+", 
        help="Rapor üretilecek hisse kodları", 
        default=["AKBNK", "THYAO", "TUPRS", "EREGL"]
    )
    parser.add_argument(
        "--force-refresh", 
        action="store_true", 
        help="Önbelleği yoksay ve sıfırdan LLM analizi üret"
    )
    args = parser.parse_args()

    generate_batch_reports(args.tickers, force_refresh=args.force_refresh)
