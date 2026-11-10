"""Çoklu LLM Sağlayıcı & Çıkarım Motoru (Universal LLM Provider Hub).

Bulut API bağımlılığını kırarak;
1. Google Gemini (Bulut / Cloud)
2. Ollama (Tamamen Yerel / On-Premises localhost:11434)
3. Google Colab Pro / vLLM / Özel GPU Sunucusu (OpenAI Uyumlu Endpoint)
4. LM Studio / LocalAI

arasında kesintisiz geçiş ve gerçek zamanlı token akışı (streaming) sağlar.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Tuple
import requests

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOpenAI

from src.config import (
    GOOGLE_API_KEY,
    DEFAULT_LLM_MODEL,
    ADVANCED_LLM_MODEL,
)

logger = logging.getLogger(__name__)

# Desteklenen Model Sağlayıcıları
PROVIDER_GEMINI = "gemini"
PROVIDER_OLLAMA = "ollama"
PROVIDER_COLAB = "colab"
PROVIDER_CUSTOM = "custom"

# Varsayılan Yerel ve Uzak Sunucu Adresleri
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_COLAB_URL = "http://localhost:8000/v1"
DEFAULT_LOCAL_MODEL = "qwen2.5:7b"


class LLMProviderHub:
    """Farklı model sağlayıcılarını tek çatı altında yöneten fabrika sınıfı."""

    @staticmethod
    def get_chat_model(
        provider: str = PROVIDER_GEMINI,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        streaming: bool = True,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 2,
    ) -> BaseChatModel:
        """
        Seçilen sağlayıcıya (Gemini, Ollama, Colab/vLLM) göre LangChain uyumlu
        sohbet modeli nesnesi üretir.
        """
        provider_clean = (provider or PROVIDER_GEMINI).lower().strip()

        # 1. YEREL OLLAMA (Local On-Premises)
        if provider_clean == PROVIDER_OLLAMA:
            url = base_url or DEFAULT_OLLAMA_URL
            model = model_name or DEFAULT_LOCAL_MODEL
            logger.info(f"[LLMProvider] Yerel Ollama modeli bağlanıyor: {model} ({url})")
            return ChatOpenAI(
                base_url=url,
                api_key=api_key or "ollama",
                model=model,
                temperature=temperature,
                streaming=streaming,
                max_retries=max_retries,
            )

        # 2. GOOGLE COLAB PRO / ÖZEL GPU SUNUCUSU (vLLM / ngrok)
        elif provider_clean in (PROVIDER_COLAB, PROVIDER_CUSTOM):
            url = base_url or DEFAULT_COLAB_URL
            model = model_name or "bist-financial-llm"
            logger.info(f"[LLMProvider] Colab/Özel GPU sunucusu bağlanıyor: {model} ({url})")
            return ChatOpenAI(
                base_url=url,
                api_key=api_key or "colab-secret-key",
                model=model,
                temperature=temperature,
                streaming=streaming,
                max_retries=max_retries,
            )

        # 3. GOOGLE GEMINI (Varsayılan Bulut)
        else:
            selected_model = model_name or DEFAULT_LLM_MODEL
            logger.info(f"[LLMProvider] Google Gemini modeli bağlanıyor: {selected_model}")
            return ChatGoogleGenerativeAI(
                model=selected_model,
                temperature=temperature,
                google_api_key=GOOGLE_API_KEY,
                streaming=streaming,
                max_retries=max_retries,
            )

    @staticmethod
    def test_connection(
        provider: str,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Seçilen sağlayıcı ile bağlantıyı doğrular ve durum mesajı döner.
        """
        provider_clean = (provider or PROVIDER_GEMINI).lower().strip()

        if provider_clean == PROVIDER_GEMINI:
            if not GOOGLE_API_KEY:
                return False, "Google API Key tanımlı değil."
            return True, "Google Gemini Bulut API hazır."

        url = base_url
        if provider_clean == PROVIDER_OLLAMA:
            url = url or DEFAULT_OLLAMA_URL
        elif provider_clean in (PROVIDER_COLAB, PROVIDER_CUSTOM):
            url = url or DEFAULT_COLAB_URL

        # OpenAI uyumlu /models endpoint kontrolü
        test_url = url.rstrip("/")
        if not test_url.endswith("/v1"):
            models_endpoint = f"{test_url}/v1/models"
        else:
            models_endpoint = f"{test_url}/models"

        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            resp = requests.get(models_endpoint, headers=headers, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                model_count = len(data.get("data", []))
                return True, f"Bağlantı başarılı! {model_count} adet model aktif ({url})."
            else:
                return False, f"Sunucu yanıt verdi ancak hata kodu döndü: {resp.status_code}"
        except Exception as e:
            return False, f"Sunucuya erişilemedi ({url}): {e}"
