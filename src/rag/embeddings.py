import time
import re
from typing import List
import google.generativeai as genai
from langchain_core.embeddings import Embeddings
from src.config import GOOGLE_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM


class GeminiEmbeddings(Embeddings):
    """
    Google Gemini Embedding Modeli için optimize edilmiş LangChain uyumlu Embeddings sınıfı.
    output_dimensionality=768 kullanarak hem hız hem de yüksek anlamsal temsil sağlar.
    429 Rate Limit (Free Tier kota) durumlarında otomatik olarak bekler ve güvenli şekilde devam eder.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIM,
        batch_size: int = 50,
        delay_between_batches: float = 1.0,
        max_retries: int = 6,
    ):
        self.api_key = api_key or GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyanızı kontrol edin.")
        
        genai.configure(api_key=self.api_key)
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.max_retries = max_retries

    def _extract_retry_delay(self, error_str: str) -> float:
        """Hata mesajındaki 'retry in X seconds' bilgisini çeker."""
        match = re.search(r"retry in ([0-9.]+)s", str(error_str), re.IGNORECASE)
        if match:
            delay = float(match.group(1)) + 2.0
            if delay > 120:
                raise RuntimeError(f"Google Generative AI günlük ücretsiz kota sınırına ulaşıldı. Lütfen daha sonra tekrar deneyin veya faturalandırmalı anahtar kullanın. ({delay:.0f}s)")
            return delay
        match_seconds = re.search(r"seconds:\s*([0-9]+)", str(error_str))
        if match_seconds:
            delay = float(match_seconds.group(1)) + 2.0
            if delay > 120:
                raise RuntimeError(f"Google Generative AI günlük ücretsiz kota sınırına ulaşıldı. Lütfen daha sonra tekrar deneyin veya faturalandırmalı anahtar kullanın. ({delay:.0f}s)")
            return delay
        return 35.0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Birden fazla metni toplu olarak (batch) vektörleştirir."""
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start_i = batch_idx * self.batch_size
            end_i = min(start_i + self.batch_size, len(texts))
            batch = texts[start_i:end_i]
            
            # Boş veya sadece boşluktan oluşan metinleri güvenli hale getir
            safe_batch = [t if t and t.strip() else " " for t in batch]
            
            for attempt in range(self.max_retries):
                try:
                    res = genai.embed_content(
                        model=self.model,
                        content=safe_batch,
                        output_dimensionality=self.dimension,
                    )
                    embeddings = res.get("embedding", [])
                    all_embeddings.extend(embeddings)
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg.lower():
                        sleep_time = self._extract_retry_delay(err_msg)
                        print(f"[GeminiEmbeddings] Kota / Hız sınırı (429) aşıldı. {sleep_time:.1f} saniye bekleniyor... (Deneme {attempt+1}/{self.max_retries})")
                        time.sleep(sleep_time)
                    elif attempt < self.max_retries - 1:
                        sleep_time = (2 ** attempt) + 1
                        time.sleep(sleep_time)
                    else:
                        raise RuntimeError(f"Embedding oluşturulurken kalıcı hata: {e}")
                        
            if self.delay_between_batches > 0 and batch_idx < total_batches - 1:
                time.sleep(self.delay_between_batches)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Tek bir sorgu metnini vektörleştirir."""
        safe_text = text if text and text.strip() else " "
        for attempt in range(self.max_retries):
            try:
                res = genai.embed_content(
                    model=self.model,
                    content=safe_text,
                    output_dimensionality=self.dimension,
                )
                return res.get("embedding", [])
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg.lower():
                    sleep_time = self._extract_retry_delay(err_msg)
                    time.sleep(sleep_time)
                elif attempt < self.max_retries - 1:
                    time.sleep((2 ** attempt) + 1)
                else:
                    raise RuntimeError(f"Sorgu embedding hatası: {e}")


class LocalHuggingFaceEmbeddings(Embeddings):
    """
    Yerel veya on-premise çalışan HuggingFace tabanlı embedding motoru (KVKK / Air-Gapped Uyumlu).
    İnternet veya harici API anahtarı gerektirmeden BAAI/bge-m3 veya multilingual-e5 ile çalışır.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self._hf_embed = None

    def _get_hf(self):
        if self._hf_embed is None:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._hf_embed = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    encode_kwargs={"normalize_embeddings": True}
                )
            except Exception as e:
                raise RuntimeError(f"HuggingFace embedding modeli yüklenemedi: {e}")
        return self._hf_embed

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        safe_texts = [t if t and t.strip() else " " for t in texts]
        return self._get_hf().embed_documents(safe_texts)

    def embed_query(self, text: str) -> List[float]:
        safe_text = text if text and text.strip() else " "
        return self._get_hf().embed_query(safe_text)


def get_embedding_provider(provider: str = "auto") -> Embeddings:
    """
    Kullanıcının tercihine veya ortam değişkenine göre uygun embedding motorunu döner.
    - 'gemini': Google Gemini Embeddings (768-D)
    - 'local' / 'huggingface': Yerel HuggingFace Embeddings (On-Premises / Air-Gapped)
    - 'auto': GOOGLE_API_KEY varsa Gemini, yoksa Yerel
    """
    prov = (provider or "auto").lower().strip()
    if prov in ("local", "huggingface"):
        return LocalHuggingFaceEmbeddings()
    elif prov == "gemini":
        return GeminiEmbeddings()
    else:
        if GOOGLE_API_KEY:
            try:
                return GeminiEmbeddings()
            except Exception:
                return LocalHuggingFaceEmbeddings()
        return LocalHuggingFaceEmbeddings()
