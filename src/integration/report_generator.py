import os
import re
import json
from datetime import date
from pathlib import Path
from typing import Optional, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from src.config import (
    DEFAULT_LLM_MODEL,
    REPORTS_DIR,
    GOOGLE_API_KEY,
    get_company_for_ticker,
    extract_text_from_content,
)
from src.rag.engine import FinancialRAGEngine
from src.integration.models import FundamentalReport, KeyMetrics, RecentDisclosure


STRUCTURED_REPORT_PROMPT_TEMPLATE = """Sen, Borsa İstanbul (BIST) şirketleri için kantitatif ve temel analiz raporları hazırlayan uzman bir Finansal Araştırma Direktörüsün.
Aşağıda {company_name} ({ticker}) şirketinin KAP'ta yayımlanmış resmi finansal raporlarından ve bilançolarından derlenen kanıt metinleri bulunmaktadır.

BAĞLAM (KAP RAPORLARI VE BİLANÇO DİPNOTLARI):
----------------------------------------------
{context}
----------------------------------------------

GÖREV:
Yukarıdaki resmi raporlara dayanarak makine öğrenimi tabanlı tahmin sistemimiz (BIST-Trend-Forecaster) için yapılandırılmış, standart bir Temel Analiz JSON nesnesi oluştur.

KURALLAR:
1. SADECE aşağıdaki JSON şemasına birebir uyan geçerli bir JSON çıktısı üret. Ekstra markdown (` ```json ` etiketleri hariç) veya sohbet metni yazma.
2. `fundamental_summary`: Şirketin son dönem finansal sağlığını, kârlılığını, gelir trendini ve operasyonel performansını özetleyen 3-4 cümlelik net profesyonel analist özeti.
3. `sentiment_score`: Şirketin temel göstergelerine göre genel duygu skoru (-1.0 ile +1.0 arasında bir float sayı, örn: 0.65).
4. `sentiment_label`: "Pozitif", "Nötr" veya "Negatif" (sentiment_score ile tutarlı olmalı).
5. `key_metrics`:
   - `pe_ratio`: Varsa F/K rasyosu (float veya bilinmiyorsa null)
   - `pb_ratio`: Varsa PD/DD rasyosu (float veya bilinmiyorsa null)
   - `net_profit_margin_trend`: "Artış", "Yatay" veya "Azalış"
   - `debt_to_equity`: "Düşük/Yönetilebilir", "Orta" veya "Yüksek"
6. `catalysts`: Şirketin gelecekteki hisse performansını yukarı taşıyabilecek en az 2 adet somut büyüme/fırsat faktörü (liste).
7. `risk_factors`: Şirketin kârlılığını veya operasyonlarını tehdit eden en az 2 adet finansal/makro risk faktörü (liste).
8. `recent_disclosures`: Şirketin son dönemde yayımladığı dikkat çeken en az 2 adet KAP bildirimini içeren liste. Her bildirimde "Tarih" (YYYY-MM-DD), "Kategori", "Başlık" ve "Etki" ("Pozitif", "Nötr" veya "Negatif") alanları bulunmalıdır.

JSON FORMATI:
{{
  "ticker": "{ticker}",
  "company_name": "{company_name}",
  "report_date": "{today_date}",
  "fundamental_summary": "...",
  "sentiment_score": 0.75,
  "sentiment_label": "Pozitif",
  "key_metrics": {{
    "pe_ratio": 4.8,
    "pb_ratio": 1.15,
    "net_profit_margin_trend": "Artış",
    "debt_to_equity": "Düşük/Yönetilebilir"
  }},
  "catalysts": [
    "Katalizör 1",
    "Katalizör 2"
  ],
  "risk_factors": [
    "Risk 1",
    "Risk 2"
  ],
  "recent_disclosures": [
    {{
      "Tarih": "2026-11-05",
      "Kategori": "Özel Durum Açıklaması",
      "Başlık": "Yeni uçak filosu alımı ve finansman anlaşması tamamlandı.",
      "Etki": "Pozitif"
    }},
    {{
      "Tarih": "2026-10-28",
      "Kategori": "Finansal Rapor Bildirimi",
      "Başlık": "2024 yılı 9 aylık konsolide finansal tablolar açıklandı.",
      "Etki": "Pozitif"
    }}
  ]
}}
"""

STRUCTURED_PROMPT = PromptTemplate(
    template=STRUCTURED_REPORT_PROMPT_TEMPLATE,
    input_variables=["ticker", "company_name", "today_date", "context"]
)


class FundamentalReportGenerator:
    """
    BIST hisseleri için KAP raporlarını RAG ile analiz edip
    BIST-Trend-Forecaster uyumlu standart JSON raporu üreten servis.
    """

    def __init__(self, rag_engine: Optional[FinancialRAGEngine] = None):
        self.rag_engine = rag_engine or FinancialRAGEngine()
        self.llm = ChatGoogleGenerativeAI(
            model=DEFAULT_LLM_MODEL,
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY,
        )

    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """LLM yanıtından temiz JSON bloğunu çıkarır."""
        cleaned = text.strip()
        # Markdown kod bloklarını kaldır
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        # Doğrudan JSON parse dene
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Süslü parantez arasını regex ile bul
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError(f"Geçerli bir JSON bulunamadı: {text[:200]}")

    def generate_report(
        self,
        ticker: str,
        force_refresh: bool = False,
    ) -> FundamentalReport:
        """
        Belirtilen hisse için RAG tabanlı temel analiz raporunu üretir ve kaydeder.
        """
        ticker_clean = ticker.upper().strip()
        report_file = REPORTS_DIR / f"{ticker_clean}_latest.json"

        # Eğer önceden üretilmişse ve zorla yenileme istenmemişse diskten oku
        if not force_refresh and report_file.exists():
            try:
                rep = FundamentalReport.load_from_file(report_file)
                if not rep.recent_disclosures:
                    from src.kap_feed import get_company_recent_disclosures
                    rep.recent_disclosures = [
                        RecentDisclosure(**d) for d in get_company_recent_disclosures(ticker_clean)
                    ]
                    rep.save_to_file(report_file)
                    try:
                        data_rep_dir = Path("data/reports")
                        data_rep_dir.mkdir(parents=True, exist_ok=True)
                        rep.save_to_file(data_rep_dir / f"{ticker_clean}.json")
                    except Exception:
                        pass
                return rep
            except Exception as e:
                print(f"[ReportGenerator] Önbellekteki rapor okunamadı, yeniden üretiliyor: {e}")

        company_name = get_company_for_ticker(ticker_clean)

        # 1. Şirketin en kritik rapor bölümlerini getir
        query = f"{company_name} {ticker_clean} finansal tablolar kâr marjı borçluluk büyüme beklentileri ve riskler"
        docs = self.rag_engine.retrieve_documents(
            query=query,
            company_filter=company_name,
            top_k=6,
            use_mmr=True
        )

        context_text = ""
        if docs:
            context_text = "\n\n".join(
                f"[Kaynak {i+1} | {d.metadata.get('file_name', '')} | Sayfa: {d.metadata.get('page', '')}]:\n{d.page_content}"
                for i, d in enumerate(docs)
            )
        else:
            context_text = f"{company_name} ({ticker_clean}) şirketine ait finansal tablolar ve faaliyet raporları incelenmiştir."

        # 2. LLM ile yapılandırılmış analizi üret
        formatted_prompt = STRUCTURED_PROMPT.format(
            ticker=ticker_clean,
            company_name=company_name,
            today_date=date.today().isoformat(),
            context=context_text,
        )

        response = self.llm.invoke(formatted_prompt)
        raw_c = response.content if hasattr(response, "content") else response
        parsed_data = self._extract_json_from_response(extract_text_from_content(raw_c))

        # 3. Son dönem KAP bildirimlerini doğrula ve entegre et
        if not parsed_data.get("recent_disclosures"):
            from src.kap_feed import get_company_recent_disclosures
            parsed_data["recent_disclosures"] = get_company_recent_disclosures(ticker_clean)

        # 4. Pydantic ile şemayı doğrula
        report = FundamentalReport(**parsed_data)

        # 5. Diske kaydet (reports/{ticker}_latest.json)
        report.save_to_file(report_file)

        # data/reports/{ticker}.json hedefini de senkronize et
        try:
            data_rep_dir = Path("data/reports")
            data_rep_dir.mkdir(parents=True, exist_ok=True)
            report.save_to_file(data_rep_dir / f"{ticker_clean}.json")
        except Exception:
            pass

        print(f"[ReportGenerator] {ticker_clean} temel analiz raporu kaydedildi: {report_file}")

        return report


# Kolay erişim için tekil fonksiyon
def get_or_create_fundamental_report(ticker: str, force_refresh: bool = False) -> FundamentalReport:
    generator = FundamentalReportGenerator()
    return generator.generate_report(ticker=ticker, force_refresh=force_refresh)
