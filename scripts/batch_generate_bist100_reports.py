import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Proje kökünü sys.path'e ekle
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import REPORTS_DIR
from src.data_collection.kap_downloader import BIST_100_COMPANIES
from src.integration.report_generator import FundamentalReportGenerator
from src.integration.models import FundamentalReport

# Borsa İstanbul BIST 30 Çekirdek Endeks Hisseleri
BIST_30_TICKERS: List[str] = [
    "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", 
    "BIMAS", "BRSAN", "EKGYO", "ENKAI", "EREGL", 
    "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", 
    "KCHOL", "KONTR", "KOZAL", "KRDMD", "MGROS", 
    "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", 
    "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", 
    "YKBNK"
]


def run_batch_generation(
    target_tickers: List[str],
    force: bool = False,
    delay: float = 2.0,
) -> Dict[str, Any]:
    """
    BIST hisseleri için toplu temel analiz ve RAG raporları üretir.
    Hata toleranslı (fault-tolerant) ve önbellek dostudur (cache-friendly).
    """
    generator = FundamentalReportGenerator()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(target_tickers)
    processed = 0
    skipped = 0
    generated = 0
    failed = 0

    results_summary = []

    print(f"\n=======================================================")
    print(f"🚀 BIST Toplu Rapor Üretimi Başlatılıyor")
    print(f"● Hedef Hisse Sayısı : {total}")
    print(f"● Zorla Yenileme     : {'Açık (--force)' if force else 'Kapalı (Önbellek Korunur)'}")
    print(f"● İstekler Arası Bekleme: {delay} saniye")
    print(f"=======================================================\n")

    for idx, ticker in enumerate(target_tickers):
        ticker_clean = ticker.upper().strip()
        comp_info = BIST_100_COMPANIES.get(ticker_clean, {"name": ticker_clean, "sector": "Genel"})
        comp_name = comp_info.get("name", ticker_clean)
        report_file = REPORTS_DIR / f"{ticker_clean}_latest.json"

        processed += 1
        prefix = f"[{processed}/{total}] {ticker_clean} ({comp_name[:24]})"

        # 1. Önbellek Kontrolü (Cache-Hit)
        if not force and report_file.exists():
            try:
                cached_report = FundamentalReport.load_from_file(report_file)
                score_str = f"+{cached_report.sentiment_score}" if cached_report.sentiment_score > 0 else str(cached_report.sentiment_score)
                print(f"[SKIP] {prefix:<40} | Zaten mevcut | Duygu: {cached_report.sentiment_label} ({score_str})")
                skipped += 1
                results_summary.append({
                    "ticker": ticker_clean,
                    "company": comp_name[:22],
                    "status": "CACHE-HIT",
                    "sentiment": f"{cached_report.sentiment_label} ({score_str})",
                    "file": report_file.name
                })
                continue
            except Exception:
                # Önbellekteki dosya bozuksa yeniden üretmeye devam et
                pass

        # 2. Rapor Üretimi (Fault-Tolerant)
        try:
            report = generator.generate_report(ticker=ticker_clean, force_refresh=True)
            score_str = f"+{report.sentiment_score}" if report.sentiment_score > 0 else str(report.sentiment_score)
            print(f"[✓]    {prefix:<40} | Duygu: {report.sentiment_label} ({score_str}) | Kaydedildi")
            generated += 1
            results_summary.append({
                "ticker": ticker_clean,
                "company": comp_name[:22],
                "status": "OLUŞTURULDU",
                "sentiment": f"{report.sentiment_label} ({score_str})",
                "file": report_file.name
            })
        except Exception as e:
            print(f"[✗]    {prefix:<40} | HATA: {e}")
            failed += 1
            results_summary.append({
                "ticker": ticker_clean,
                "company": comp_name[:22],
                "status": "HATA",
                "sentiment": "-",
                "file": "-"
            })

        # 3. Rate-Limit Koruması (Hız Sınırı Beklemesi)
        if idx < total - 1 and delay > 0:
            time.sleep(delay)

    # 4. Sonuç Özeti
    print(f"\n=======================================================")
    print(f"📊 BİST TOPLU RAPOR ÜRETİM KARNESİ")
    print(f"=======================================================")
    print(f"● Toplam Hedef Hisse   : {total}")
    print(f"● Yeni Üretilen Rapor  : {generated}")
    print(f"● Önbellekten Alınan   : {skipped}")
    print(f"● Hata Alan Hisse      : {failed}")
    print(f"● Raporlar Klasörü     : {REPORTS_DIR}")
    print(f"=======================================================\n")

    return {
        "total": total,
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "results": results_summary
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIST 100 Şirketleri İçin Toplu Temel Analiz & RAG Rapor Üretici")
    parser.add_argument("--priority-bist30", action="store_true", help="Yalnızca en likit BIST 30 şirketleri için çalıştır")
    parser.add_argument("--all", action="store_true", help="Tüm 100 BIST şirketi için çalıştır")
    parser.add_argument("--tickers", nargs="+", help="Belirli hisse kodları (Örn: ASELS SISE FROTO)")
    parser.add_argument("--force", action="store_true", help="Var olan reports/{ticker}_latest.json dosyalarını ezerek yeniden üret")
    parser.add_argument("--delay", type=float, default=2.0, help="İstekler arası bekleme süresi (saniye, varsayılan: 2.0)")

    args = parser.parse_args()

    # Hedef hisse listesini belirle
    if args.tickers:
        targets = [t.upper().strip() for t in args.tickers]
    elif args.all:
        targets = list(BIST_100_COMPANIES.keys())
    elif args.priority_bist30:
        targets = [t for t in BIST_30_TICKERS if t in BIST_100_COMPANIES]
    else:
        # Varsayılan: BIST 30 öncelikli hisseler
        targets = [t for t in BIST_30_TICKERS if t in BIST_100_COMPANIES]

    run_batch_generation(targets, force=args.force, delay=args.delay)
