import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document


class HallucinationGuard:
    """
    Finansal RAG için Sayısal ve İddia Doğrulama Katmanı (Hallucination Defense).
    Modelin ürettiği finansal rakamları (kâr, gelir, borç, FAVÖK, oran vb.)
    resmi KAP raporu kaynaklarıyla karşılaştırarak halüsinasyonları önler ve skorlar.
    """

    # Finansal rakam, yüzde ve para birimi yakalayan regex deseni
    NUMBER_PATTERN = re.compile(
        r"(?:%?\s*\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b\s*%?|\b\d+(?:[.,]\d+)?\s*(?:milyar|milyon|bin|tl|usd|eur|dolar|avro)\b)",
        re.IGNORECASE
    )

    def extract_numerical_claims(self, text: str) -> List[str]:
        """Üretilen yanıttaki tüm sayısal ve oransal finansal iddiaları çıkarır."""
        raw_matches = self.NUMBER_PATTERN.findall(text)
        cleaned = []
        for m in raw_matches:
            item = m.strip()
            # Yıl sayılarını (2020-2026) finansal iddia sayma
            if re.match(r"^202[0-6]$", item):
                continue
            # Çok kısa tek haneli maddeleri filtrele
            if len(item) == 1 and item.isdigit():
                continue
            if item not in cleaned:
                cleaned.append(item)
        return cleaned

    def normalize_number(self, val: str) -> str:
        """Karşılaştırma için sayıları temizler (nokta/virgül normalizasyonu)."""
        clean = val.lower().replace("tl", "").replace("usd", "").replace("%", "").strip()
        clean = clean.replace(".", "").replace(",", ".")
        return clean

    def verify_claims(
        self,
        answer: str,
        sources: List[Document]
    ) -> Dict[str, Any]:
        """
        Üretilen yanıttaki iddiaları kaynak belgelerle çapraz doğrular.
        
        Döner:
        - groundedness_score: 0.0 ile 1.0 arasında doğruluk skoru
        - verified_claims: Doğrulanan iddialar
        - unverified_claims: Kaynakta bulunamayan (olası halüsinasyon) iddialar
        - is_safe: Yanıt güvenli mi?
        - status_badge: Arayüz için şık durum rozeti
        """
        claims = self.extract_numerical_claims(answer)
        if not claims:
            return {
                "groundedness_score": 1.0,
                "verified_claims": [],
                "unverified_claims": [],
                "total_claims": 0,
                "is_safe": True,
                "status_badge": "● Sayısal İddia İçermiyor / Niteliksel Analiz"
            }

        # Tüm kaynak metinleri tek bir havuzda topla
        combined_sources = " ".join([d.page_content.lower() for d in sources])

        verified = []
        unverified = []

        for claim in claims:
            claim_clean = claim.lower().strip()
            # 1. Doğrudan metin eşleşmesi
            if claim_clean in combined_sources:
                verified.append(claim)
                continue

            # 2. Sayısal çekirdek eşleşmesi (örn: 66,49 -> 66.49 veya 66490)
            digits_only = re.sub(r"[^\d]", "", claim_clean)
            if len(digits_only) >= 2 and digits_only in re.sub(r"[^\d]", "", combined_sources):
                verified.append(claim)
            else:
                unverified.append(claim)

        total = len(claims)
        verified_count = len(verified)
        score = round(verified_count / total, 2) if total > 0 else 1.0
        is_safe = score >= 0.85

        if score == 1.0:
            badge = f"🛡️ %100 Resmi KAP Raporuyla Doğrulandı ({verified_count}/{total} Finansal Veri)"
        elif score >= 0.80:
            badge = f"✓ Yüksek Güvenilirlik ({verified_count}/{total} Doğrulandı)"
        else:
            badge = f"⚠️ Dikkat: Bazı Rakamlar Doğrulanamadı ({len(unverified)} Şüpheli Veri)"

        return {
            "groundedness_score": score,
            "verified_claims": verified,
            "unverified_claims": unverified,
            "total_claims": total,
            "is_safe": is_safe,
            "status_badge": badge
        }
