"""BIST 100 Finansal LLM Fine-Tuning Veri Kümesi Üretici Betiği.

Projedeki 100 BIST şirketinin resmi faaliyet raporları, bilanço dipnotları,
DuPont kârlılık modeli, Altman Z-skorları ve son KAP bildirimlerinden yararlanarak
Google Colab Pro'da Llama-3.1-8B veya Qwen-2.5-7B modellerini eğitmek için
yüksek kaliteli Alpaca ve ShareGPT formatında instruction veri kümesi üretir.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Proje kökünü sys.path'e ekle
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import logging
from typing import List, Dict, Any

from src.config import ROOT_DIR, REPORTS_DIR, TICKER_TO_COMPANY
from src.dupont_analysis import calculate_dupont_model, calculate_altman_z_score
from src.dividend_analysis import calculate_dividend_metrics
from src.macro_sensitivity import analyze_macro_sensitivity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_OUTPUT_DIR = ROOT_DIR / "data"
ALPACA_OUTPUT_FILE = DATA_OUTPUT_DIR / "finetune_bist_dataset.jsonl"
SHAREGPT_OUTPUT_FILE = DATA_OUTPUT_DIR / "finetune_bist_sharegpt.jsonl"


def create_instruction_entry(
    instruction: str,
    user_input: str,
    output: str,
    category: str,
    ticker: str
) -> Dict[str, Any]:
    """Alpaca ve ShareGPT uyumlu standart veri sözlüğü üretir."""
    return {
        "instruction": instruction.strip(),
        "input": user_input.strip(),
        "output": output.strip(),
        "category": category,
        "ticker": ticker
    }


def generate_dataset() -> List[Dict[str, Any]]:
    """Tüm BIST 100 verilerini tarayarak zengin bir finansal eğitim seti oluşturur."""
    dataset: List[Dict[str, Any]] = []

    report_files = list(REPORTS_DIR.glob("*_latest.json"))
    logger.info(f"Toplam {len(report_files)} şirket raporu taranıyor...")

    for rep_file in report_files:
        try:
            with open(rep_file, "r", encoding="utf-8") as f:
                rep = json.load(f)
        except Exception as e:
            logger.warning(f"Rapor okunamadı {rep_file.name}: {e}")
            continue

        ticker = rep.get("ticker", "").upper().strip()
        comp_name = rep.get("company_name", TICKER_TO_COMPANY.get(ticker, ticker))
        summary = rep.get("fundamental_summary", "")
        sent_label = rep.get("sentiment_label", "Nötr")
        sent_score = rep.get("sentiment_score", 0.0)
        catalysts = rep.get("catalysts", [])
        risks = rep.get("risk_factors", [])
        disclosures = rep.get("recent_disclosures", [])

        if not ticker or not summary:
            continue

        # 1. TEMEL ANALİZ VE YÖNETİCİ ÖZETİ EĞİTİM ÖRNEĞİ
        cat_text = "\n".join(f"● {c}" for c in catalysts[:3]) if catalysts else "● Güçlü operasyonel yapı"
        risk_text = "\n".join(f"● {r}" for r in risks[:3]) if risks else "● Genel makroekonomik dalgalanmalar"

        summary_ans = (
            f"**{comp_name} ({ticker}) Temel Analiz & Yönetici Özeti:**\n\n"
            f"{summary}\n\n"
            f"**Genel Piyasa & Bilanço Duygusu:** {sent_label} (Skor: {sent_score:+.2f})\n\n"
            f"**Öne Çıkan Büyüme Katalizörleri:**\n{cat_text}\n\n"
            f"**İzlenen Temel Riskler:**\n{risk_text}"
        )

        dataset.append(create_instruction_entry(
            instruction="Sen Borsa İstanbul (BIST) uzmanı bir kıdemli finansal analistsin. Verilen şirketin son dönem finansal performansını, büyüme katalizörlerini ve temel risklerini özetle.",
            user_input=f"{comp_name} ({ticker}) son dönem finansal durumu ve operasyonel performansı nasıldır?",
            output=summary_ans,
            category="fundamental_summary",
            ticker=ticker
        ))

        # 2. DUPONT 3-AŞAMALI ROE VE ALTMAN Z-SKORU ANALİZİ
        try:
            dupont = calculate_dupont_model(ticker)
            altman = calculate_altman_z_score(ticker, dupont)

            dupont_ans = (
                f"**{ticker} DuPont 3-Aşamalı ROE Ayrıştırması & Finansal Sağlık:**\n\n"
                f"Şirketin Özkaynak Kârlılığı (ROE) **%{dupont['roe_pct']:.1f}** seviyesindedir. "
                f"Kârlılığın ana belirleyicisi **{dupont['driver']}** bileşenidir.\n\n"
                f"● **Net Kâr Marjı:** %{dupont['net_profit_margin_pct']:.1f}\n"
                f"● **Aktif Devir Hızı:** {dupont['asset_turnover']:.2f}x\n"
                f"● **Finansal Kaldıraç (Özkaynak Çarpanı):** {dupont['equity_multiplier']:.2f}x\n\n"
                f"**Altman Z-Skoru (Finansal Stres / İflas Riski):**\n"
                f"Hesaplanan Z-Skoru **{altman['z_score']:.2f}** olup, şirket **{altman['zone']}** konumundadır. "
                f"{altman['interpretation']}"
            )

            dataset.append(create_instruction_entry(
                instruction="Verilen BIST hissesi için 3 aşamalı DuPont ROE ayrıştırmasını ve Altman Z-skoru iflas/stres dayanıklılık değerlendirmesini yap.",
                user_input=f"{ticker} hissesinin DuPont kârlılık sürücüleri ve finansal sağlamlık Z-skoru nedir?",
                output=dupont_ans,
                category="dupont_altman",
                ticker=ticker
            ))
        except Exception:
            pass

        # 3. TEMETTÜ VE SERBEST NAKİT AKIŞI (FCF) GÜVENLİK ANALİZİ
        try:
            div = calculate_dividend_metrics(ticker)
            div_ans = (
                f"**{comp_name} ({ticker}) Temettü & Nakit Akış Güvenlik Karnesi:**\n\n"
                f"● **Temettü Verimi:** %{div['yield_pct']:.2f}\n"
                f"● **Kâr Dağıtım Oranı (Payout Ratio):** %{div['payout_pct']:.1f}\n"
                f"● **Serbest Nakit Akışı (FCF) Marjı:** %{div['fcf_margin_pct']:.1f}\n"
                f"● **Temettü Güvenlik Skoru:** {div['dividend_safety_score']:.0f} / 100\n"
                f"● **Kategori & Durum:** {div['category']}\n\n"
                f"**Analist Değerlendirmesi:** {div['takeaway']}"
            )

            dataset.append(create_instruction_entry(
                instruction="BIST şirketinin temettü verimi, serbest nakit akışı gücü ve temettü ödeme sürdürülebilirliğini değerlendir.",
                user_input=f"{ticker} temettü verimi ve temettü ödeme güvenliği ne durumdadır?",
                output=div_ans,
                category="dividend_safety",
                ticker=ticker
            ))
        except Exception:
            pass

        # 4. MAKROEKONOMİK DUYARLILIK (TORNADO SENARYOSU)
        try:
            macro = analyze_macro_sensitivity(ticker, usd_change_pct=10.0, rate_change_bps=250, oil_change_pct=15.0)
            u_sc = macro["scenarios"]["usd_try"]
            r_sc = macro["scenarios"]["interest_rate"]
            o_sc = macro["scenarios"]["brent_oil"]

            macro_ans = (
                f"**{ticker} Makroekonomik Şok Duyarlılık Analizi (USD/TRY, Faiz, Petrol):**\n\n"
                f"● **Dolar Kuru (%10 Artış):** Net Kâr Etkisi %{u_sc['impact_pct']:+.2f} ({u_sc['driver']})\n"
                f"● **TCMB Faizi (+250 bps Artış):** Net Kâr Etkisi %{r_sc['impact_pct']:+.2f} ({r_sc['driver']})\n"
                f"● **Brent Petrol (%15 Artış):** Net Kâr Etkisi %{o_sc['impact_pct']:+.2f} ({o_sc['driver']})\n\n"
                f"**Bileşik Senaryo Etkisi:** %{macro['total_impact_pct']:+.2f}\n"
                f"**Dayanıklılık Derecesi:** {macro['resilience_badge']}\n"
                f"{macro['verdict']}"
            )

            dataset.append(create_instruction_entry(
                instruction="Döviz kuru (USD/TRY), TCMB politika faizi ve emtia fiyatlarındaki olası şokların şirketin net kârı ve marjları üzerindeki etkisini simüle et.",
                user_input=f"Döviz kuru ve faizlerdeki yükseliş {ticker} kârlılığını nasıl etkiler?",
                output=macro_ans,
                category="macro_sensitivity",
                ticker=ticker
            ))
        except Exception:
            pass

        # 5. KAP BİLDİRİMLERİ VE GÜNCEL GELİŞMELER
        if disclosures:
            disc_lines = []
            for d in disclosures[:3]:
                tarih = d.get("Tarih", "")
                kat = d.get("Kategori", "")
                baslik = d.get("Başlık", "")
                etki = d.get("Etki", "Nötr")
                disc_lines.append(f"● **[{tarih}] {kat} ({etki}):** {baslik}")

            disc_ans = (
                f"**{comp_name} ({ticker}) Tarafından Yayımlanan Son Dönem KAP Bildirimleri:**\n\n"
                + "\n".join(disc_lines) +
                "\n\nBu bildirimler şirketin operasyonel hacmi, sermaye yönetimi ve yatırım takvimi açısından pozitif katalizör oluşturmaktadır."
            )

            dataset.append(create_instruction_entry(
                instruction="Kamuyu Aydınlatma Platformu'nda (KAP) yayımlanan son bildirimleri listele ve şirket üzerindeki etkilerini özetle.",
                user_input=f"{ticker} şirketinin son KAP bildirimleri ve önemli gelişmeleri nelerdir?",
                output=disc_ans,
                category="kap_disclosures",
                ticker=ticker
            ))

    return dataset


def save_datasets(dataset: List[Dict[str, Any]]) -> None:
    """Alpaca ve ShareGPT JSONL formatlarında diske kaydeder."""
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Alpaca Format
    with open(ALPACA_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Alpaca dataset kaydedildi: {ALPACA_OUTPUT_FILE} ({len(dataset)} örnek)")

    # 2. ShareGPT Format (Modern SFT / Unsloth uyumlu)
    with open(SHAREGPT_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in dataset:
            system_msg = item["instruction"]
            user_msg = item["input"]
            asst_msg = item["output"]

            sharegpt_entry = {
                "conversations": [
                    {"from": "system", "value": system_msg},
                    {"from": "human", "value": user_msg},
                    {"from": "gpt", "value": asst_msg}
                ]
            }
            f.write(json.dumps(sharegpt_entry, ensure_ascii=False) + "\n")
    logger.info(f"ShareGPT dataset kaydedildi: {SHAREGPT_OUTPUT_FILE}")


if __name__ == "__main__":
    generated = generate_dataset()
    save_datasets(generated)
    print(f"\n✅ Toplam {len(generated)} yüksek kaliteli BIST finansal eğitim çifti başarıyla üretildi!")
