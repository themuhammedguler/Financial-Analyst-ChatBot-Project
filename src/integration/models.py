import json
from pathlib import Path
from datetime import date
from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class KeyMetrics(BaseModel):
    """BIST-Trend-Forecaster ile paylaşılan temel finansal rasyolar."""
    pe_ratio: Optional[Union[float, str]] = Field(default=None, description="Fiyat/Kazanç (F/K) Oranı")
    pb_ratio: Optional[Union[float, str]] = Field(default=None, description="Piyasa Değeri / Defter Değeri (PD/DD)")
    net_profit_margin_trend: str = Field(default="Yatay", description="Net kâr marjı trendi: Artış | Yatay | Azalış")
    debt_to_equity: str = Field(default="Yönetilebilir", description="Borçluluk durumu: Düşük/Yönetilebilir | Orta | Yüksek")
    extra_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Ek rasyolar (FAVÖK marjı, ROE, cari oran vb.)")


class FundamentalReport(BaseModel):
    """
    BIST-Trend-Forecaster entegrasyonu için standart yapılandırılmış temel analiz çıktısı.
    Teknik analiz ve kantitatif tahmin modelleriyle birleştirilmek üzere tasarlanmıştır.
    """
    ticker: str = Field(..., description="BIST Hisse Kodu (Örn: THYAO, AKBNK)")
    company_name: str = Field(..., description="Şirket Ticari Unvanı")
    report_date: str = Field(default_factory=lambda: date.today().isoformat(), description="Rapor üretim tarihi (YYYY-MM-DD)")
    fundamental_summary: str = Field(..., description="KAP ve finansal rapor özet analizi")
    sentiment_score: float = Field(
        ..., 
        ge=-1.0, 
        le=1.0, 
        description="Niteliksel duygu skoru (-1.0: Çok Negatif, 0.0: Nötr, +1.0: Çok Pozitif)"
    )
    sentiment_label: Literal["Pozitif", "Nötr", "Negatif"] = Field(
        ..., 
        description="Duygu etiketi: Pozitif | Nötr | Negatif"
    )
    key_metrics: KeyMetrics = Field(..., description="Temel finansal göstergeler")
    catalysts: List[str] = Field(default_factory=list, description="Hisse için yukarı yönlü katalizörler ve büyüme faktörleri")
    risk_factors: List[str] = Field(default_factory=list, description="Faaliyet ve finansal risk faktörleri")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        return v.upper().strip()

    def save_to_file(self, target_path: Union[str, Path]) -> Path:
        """Raporu JSON dosyası olarak diske kaydeder."""
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load_from_file(cls, path: Union[str, Path]) -> "FundamentalReport":
        """Diskteki JSON dosyasından FundamentalReport nesnesi yükler."""
        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class SentimentRankItem(BaseModel):
    """Hisse duygu sıralaması öğesi."""
    rank: int = Field(..., description="Sıralama pozisyonu (1'den başlar)")
    ticker: str = Field(..., description="BIST Hisse Kodu")
    company_name: str = Field(..., description="Şirket Ticari Unvanı")
    sentiment_score: float = Field(..., description="Duygu skoru (-1.0 ile +1.0 arası)")
    sentiment_label: Literal["Pozitif", "Nötr", "Negatif"] = Field(..., description="Duygu etiketi")
    fundamental_summary: str = Field(..., description="Kısa temel analiz özeti")
    key_metrics: KeyMetrics = Field(..., description="Temel rasyolar")
    catalysts: List[str] = Field(default_factory=list, description="Katalizörler")
    risk_factors: List[str] = Field(default_factory=list, description="Risk faktörleri")
    report_date: str = Field(..., description="Rapor üretim tarihi")


class SentimentRankResponse(BaseModel):
    """Duygu sıralaması yanıt modeli (Top-Bullish veya Top-Bearish)."""
    category: Literal["bullish", "bearish"] = Field(..., description="Sıralama kategorisi")
    count: int = Field(..., description="Dönen hisse sayısı")
    limit: int = Field(..., description="İstenen limit")
    rankings: List[SentimentRankItem] = Field(default_factory=list, description="Sıralı hisse listesi")


class SentimentSummaryResponse(BaseModel):
    """Piyasa geneli duygu özeti modeli."""
    total_reports: int = Field(..., description="İncelenen toplam şirket raporu sayısı")
    average_sentiment_score: float = Field(..., description="Ortalama duygu skoru")
    sentiment_breakdown: Dict[str, int] = Field(..., description="Etiket dağılımı: Pozitif, Nötr, Negatif sayıları")
    positive_ratio: float = Field(..., description="Pozitif duyguya sahip şirketlerin oranı (0.0 - 1.0)")
    top_bullish_tickers: List[str] = Field(default_factory=list, description="En yüksek skora sahip hisseler")
    top_bearish_tickers: List[str] = Field(default_factory=list, description="En düşük skora sahip hisseler")
    available_tickers: List[str] = Field(default_factory=list, description="Mevcut raporu bulunan tüm hisseler")

