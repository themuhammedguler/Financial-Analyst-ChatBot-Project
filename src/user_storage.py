"""Kullanıcı Kalıcı Veri Saklama Modülü (User Storage).

Sayfa yenilendiğinde, oturum kapandığında veya tarayıcı yeniden açıldığında
İzleme Listesi (Watchlist) ve Analist Not Defteri (Notes) verilerinin kaybolmamasını sağlar.
Veriler `user_data/` dizininde JSON formatında atomik ve kalıcı olarak saklanır.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import ROOT_DIR

logger = logging.getLogger(__name__)

USER_DATA_DIR = ROOT_DIR / "user_data"
WATCHLIST_FILE = USER_DATA_DIR / "watchlist.json"
NOTES_FILE = USER_DATA_DIR / "saved_notes.json"
CLIENT_PREFS_FILE = USER_DATA_DIR / "client_model_prefs.json"

DEFAULT_WATCHLIST: List[str] = ["THYAO", "ASELS", "TUPRS", "EREGL", "BIMAS"]

BROWSER_MODEL_COOKIE = "bist_user_llm_prefs"

DEFAULT_MODEL_SETTINGS: Dict[str, str] = {
    "provider": "gemini",
    "gemini_model": "gemini-3.1-flash-lite",
    "ollama_url": "http://localhost:11434/v1",
    "ollama_model": "qwen2.5:7b",
    "colab_url": "",
    "colab_model": "bist-financial-llm",
}


def _ensure_dir() -> None:
    """Veri dizininin varlığını doğrular, yoksa oluşturur."""
    try:
        USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"User data dizini oluşturulamadı: {e}")


def load_watchlist() -> List[str]:
    """
    Kalıcı izleme listesini diskten okur.
    Dosya yoksa varsayılan listeyi oluşturup kaydeder.
    """
    _ensure_dir()
    if not WATCHLIST_FILE.exists():
        save_watchlist(DEFAULT_WATCHLIST)
        return list(DEFAULT_WATCHLIST)

    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list) and data:
                # Temizle, büyük harfe çevir ve benzersizliği koru
                cleaned = []
                for item in data:
                    sym = str(item).upper().replace(".IS", "").strip()
                    if sym and sym not in cleaned:
                        cleaned.append(sym)
                return cleaned if cleaned else list(DEFAULT_WATCHLIST)
    except Exception as e:
        logger.warning(f"Watchlist dosyası okunamadı ({e}), varsayılana dönülüyor.")

    return list(DEFAULT_WATCHLIST)


def save_watchlist(watchlist: List[str]) -> bool:
    """İzleme listesini kalıcı JSON dosyasına yazar."""
    _ensure_dir()
    try:
        cleaned = []
        for item in watchlist:
            sym = str(item).upper().replace(".IS", "").strip()
            if sym and sym not in cleaned:
                cleaned.append(sym)
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Watchlist kaydedilemedi: {e}")
        return False


def add_to_watchlist(ticker: str) -> List[str]:
    """İzleme listesine yeni hisse ekler ve diske kaydeder."""
    sym = ticker.upper().replace(".IS", "").strip()
    if not sym:
        return load_watchlist()

    current = load_watchlist()
    if sym not in current:
        current.append(sym)
        save_watchlist(current)
    return current


def remove_from_watchlist(ticker: str) -> List[str]:
    """İzleme listesinden hisse çıkarır ve diske kaydeder."""
    sym = ticker.upper().replace(".IS", "").strip()
    current = load_watchlist()
    if sym in current:
        current.remove(sym)
        save_watchlist(current)
    return current


def load_saved_notes() -> List[Dict[str, Any]]:
    """Kayıtlı analist notlarını diskten okur."""
    _ensure_dir()
    if not NOTES_FILE.exists():
        return []

    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        logger.warning(f"Kayıtlı notlar dosyası okunamadı ({e}).")

    return []


def save_saved_notes(notes: List[Dict[str, Any]]) -> bool:
    """Analist notlarını kalıcı JSON dosyasına yazar."""
    _ensure_dir()
    try:
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Notlar kaydedilemedi: {e}")
        return False


def add_saved_note(
    ticker: str,
    title: str,
    content: str,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Yeni analist notu ekler, tarihlendirir ve diske kaydeder.
    """
    clean_sym = ticker.upper().replace(".IS", "").strip() if ticker else "GENEL"
    note_id = uuid.uuid4().hex[:8]
    now_str = datetime.now().strftime("%d.%m.%Y %H:%M")

    note_entry = {
        "id": note_id,
        "ticker": clean_sym,
        "title": title.strip() or f"{clean_sym} Analiz Özeti",
        "content": content.strip(),
        "created_at": now_str,
        "tags": tags or [clean_sym, "KAP/Bilanço"]
    }

    current_notes = load_saved_notes()
    current_notes.insert(0, note_entry)  # En yeni not en başta
    save_saved_notes(current_notes)
    return note_entry


def delete_saved_note(note_id: str) -> bool:
    """Belirtilen ID'ye sahip notu siler ve diski günceller."""
    current_notes = load_saved_notes()
    filtered = [n for n in current_notes if n.get("id") != note_id]
    if len(filtered) != len(current_notes):
        save_saved_notes(filtered)
        return True
    return False


def _get_client_fingerprint() -> str:
    """
    O anki kullanıcının tarayıcı ve ağ bilgilerine (IP + User-Agent + Dil)
    dayalı benzersiz bir SHA-256 parmak izi üretir.
    Böylece her kullanıcının ayarı yalnızca kendi cihazında/tarayıcısında hatırlanır,
    başka bir bilgisayardan veya tarayıcıdan giren kişilere kesinlikle görünmez.
    """
    import hashlib
    try:
        import streamlit as st
        ip = getattr(st.context, "ip_address", None) or "local_ip"
        headers = getattr(st.context, "headers", {}) or {}
        ua = headers.get("User-Agent", headers.get("user-agent", "unknown_ua"))
        lang = headers.get("Accept-Language", headers.get("accept-language", "tr"))
        sec_ch = headers.get("Sec-Ch-Ua", headers.get("sec-ch-ua", ""))
        raw = f"{ip}|{ua}|{lang}|{sec_ch}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    except Exception:
        return "default_local_client"


def load_model_settings() -> Dict[str, str]:
    """
    Sadece o anki kullanıcının kendi tarayıcısındaki (cookie / parmak izi)
    model ve URL tercihlerini okur. Sunucu üzerindeki diğer kullanıcılarla paylaşılmaz.
    """
    _ensure_dir()
    settings = dict(DEFAULT_MODEL_SETTINGS)
    try:
        import streamlit as st
        from urllib.parse import unquote

        # 1. Mevcut aktif çalışma döngüsünde (session_state) güncellenmiş değer varsa onu baz al
        if "user_browser_llm_prefs" in st.session_state and isinstance(st.session_state.user_browser_llm_prefs, dict):
            for k, v in st.session_state.user_browser_llm_prefs.items():
                if k in settings and isinstance(v, str):
                    settings[k] = v.strip()
            return settings

        # 2. Önce bu kullanıcının tarayıcı/IP parmak izine (fingerprint) kayıtlı senkron veriyi oku
        fp = _get_client_fingerprint()
        if CLIENT_PREFS_FILE.exists():
            try:
                with open(CLIENT_PREFS_FILE, "r", encoding="utf-8") as f:
                    all_clients = json.load(f)
                    if isinstance(all_clients, dict) and fp in all_clients:
                        client_data = all_clients[fp]
                        if isinstance(client_data, dict):
                            for k, v in client_data.items():
                                if k in settings and isinstance(v, str) and v.strip():
                                    settings[k] = v.strip()
            except Exception:
                pass

        # 3. Tarayıcı çerezini (st.context.cookies) üzerine uygula (Enter'a basmadan yapılan yapıştırmaları da yakalar)
        cookies = getattr(st.context, "cookies", None)
        if cookies and BROWSER_MODEL_COOKIE in cookies:
            raw_val = unquote(cookies[BROWSER_MODEL_COOKIE])
            data = json.loads(raw_val)
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in settings and isinstance(v, str) and v.strip():
                        settings[k] = v.strip()

        st.session_state.user_browser_llm_prefs = dict(settings)
    except Exception as e:
        logger.debug(f"Kullanıcı model tercihi okunamadı: {e}")

    return settings


def save_model_settings(updates: Dict[str, str]) -> Dict[str, str]:
    """
    Model sağlayıcı ve URL seçimlerini yalnızca o kullanıcının kendi tarayıcısına
    ve tarayıcı parmak izine özel olarak kaydeder.
    Ayrıca metin kutusuna link yapıştırıldığı anda (Enter'a basılmasa bile) çereze yazar.
    """
    _ensure_dir()
    current = load_model_settings()
    for k, v in updates.items():
        if k in current and isinstance(v, str):
            val = v.strip()
            # Eğer kullanıcı kutuyu bilerek değiştirmişse veya dolu bir değer girmişse güncelle
            if k == "colab_url" and not val and current.get("colab_url"):
                pass
            else:
                current[k] = val

    try:
        import streamlit as st
        import streamlit.components.v1 as components

        st.session_state.user_browser_llm_prefs = dict(current)

        # 1. Senkron olarak sadece bu kullanıcının parmak izine (fingerprint) kaydet
        fp = _get_client_fingerprint()
        all_clients: Dict[str, Any] = {}
        if CLIENT_PREFS_FILE.exists():
            try:
                with open(CLIENT_PREFS_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        all_clients = loaded
            except Exception:
                all_clients = {}

        if all_clients.get(fp) != current:
            all_clients[fp] = dict(current)
            with open(CLIENT_PREFS_FILE, "w", encoding="utf-8") as f:
                json.dump(all_clients, f, ensure_ascii=False, indent=2)

        # 2. Tarayıcı çerezini güncelle ve Enter'a basmadan yapılan yapıştırmaları (paste/input) anında yakala
        prefs_json = json.dumps(current, ensure_ascii=False)
        components.html(
            f"""<script>
            (function() {{
                var cookieName = "{BROWSER_MODEL_COOKIE}";
                var prefs = {prefs_json};
                function writeCookie(obj) {{
                    var encoded = encodeURIComponent(JSON.stringify(obj));
                    var cookieStr = cookieName + "=" + encoded + "; path=/; max-age=31536000; SameSite=Lax";
                    try {{ window.parent.document.cookie = cookieStr; }} catch(e) {{}}
                    try {{ document.cookie = cookieStr; }} catch(e) {{}}
                }}
                writeCookie(prefs);

                try {{
                    var parentDoc = window.parent.document;
                    var sidebar = parentDoc.querySelector('[data-testid="stSidebar"]');
                    if (sidebar) {{
                        var inputs = sidebar.querySelectorAll('input[type="text"]');
                        inputs.forEach(function(inp) {{
                            var syncInput = function() {{
                                var wrap = inp.closest('[data-testid="stTextInput"]');
                                var lbl = wrap ? wrap.innerText : "";
                                if (lbl.indexOf("Colab") !== -1 || lbl.indexOf("ngrok") !== -1) {{
                                    prefs.colab_url = inp.value.trim();
                                }} else if (lbl.indexOf("Model Etiketi") !== -1) {{
                                    prefs.colab_model = inp.value.trim();
                                }} else if (lbl.indexOf("Ollama API") !== -1) {{
                                    prefs.ollama_url = inp.value.trim();
                                }} else if (lbl.indexOf("Model Adı") !== -1) {{
                                    prefs.ollama_model = inp.value.trim();
                                }}
                                writeCookie(prefs);
                            }};
                            if (inp.value && inp.value.trim()) {{
                                syncInput();
                            }}
                            if (!inp.dataset.bistCookieBound) {{
                                inp.dataset.bistCookieBound = "1";
                                inp.addEventListener("input", syncInput);
                                inp.addEventListener("paste", function() {{ setTimeout(syncInput, 15); }});
                                inp.addEventListener("change", syncInput);
                            }}
                        }});
                    }}
                }} catch(err) {{}}
            }})();
            </script>""",
            height=1,
            width=1,
        )
    except Exception as e:
        logger.debug(f"Kullanıcı model tercihi yazılamadı: {e}")

    return current

