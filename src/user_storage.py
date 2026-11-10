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

DEFAULT_WATCHLIST: List[str] = ["THYAO", "ASELS", "TUPRS", "EREGL", "BIMAS"]


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
