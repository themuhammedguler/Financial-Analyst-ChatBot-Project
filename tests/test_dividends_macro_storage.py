"""Unit tests for Dividend Analysis, Macro Sensitivity, User Storage, and Chat Export."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.dividend_analysis import calculate_dividend_metrics, create_dividend_chart, DEFAULT_DIVIDEND_PROFILES
from src.macro_sensitivity import analyze_macro_sensitivity, create_sensitivity_tornado_chart, MACRO_TICKER_PROFILES
from src.user_storage import (
    load_watchlist,
    save_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    load_saved_notes,
    save_saved_notes,
    add_saved_note,
    delete_saved_note,
    WATCHLIST_FILE,
    NOTES_FILE,
)
from src.chat_export import export_chat_history_markdown, export_chat_history_html


# ==========================================
# 1. Dividend Analysis Tests
# ==========================================

def test_dividend_metrics_tuprs():
    metrics = calculate_dividend_metrics("TUPRS")
    assert metrics["ticker"] == "TUPRS"
    assert metrics["yield_pct"] >= 0
    assert "dividend_safety_score" in metrics
    assert 0 <= metrics["dividend_safety_score"] <= 100
    assert "category" in metrics
    assert "takeaway" in metrics


def test_dividend_metrics_non_dividend():
    # THYAO typically does not pay dividend (0% yield in default profile)
    metrics = calculate_dividend_metrics("THYAO")
    assert metrics["ticker"] == "THYAO"
    assert "Temettü Dağıtmıyor" in metrics["category"]


def test_dividend_chart_generation():
    metrics = calculate_dividend_metrics("FROTO")
    fig = create_dividend_chart(metrics, "FROTO")
    assert fig is not None
    assert len(fig.data) > 0
    assert "FROTO" in fig.layout.title.text


# ==========================================
# 2. Macro Sensitivity Tests
# ==========================================

def test_macro_sensitivity_thyao():
    res = analyze_macro_sensitivity("THYAO", usd_change_pct=10.0, rate_change_bps=250, oil_change_pct=15.0)
    assert res["ticker"] == "THYAO"
    assert "Havacılık" in res["sector"]
    # FX positive (+7.2), Oil negative (-8.5)
    assert res["scenarios"]["usd_try"]["impact_pct"] > 0
    assert res["scenarios"]["brent_oil"]["impact_pct"] < 0
    assert "verdict" in res
    assert "badge_color" in res


def test_macro_sensitivity_tuprs():
    # Oil positive for TUPRS refinery crack margin
    res = analyze_macro_sensitivity("TUPRS", usd_change_pct=10.0, rate_change_bps=250, oil_change_pct=15.0)
    assert res["ticker"] == "TUPRS"
    assert res["scenarios"]["brent_oil"]["impact_pct"] > 0


def test_macro_sensitivity_default():
    res = analyze_macro_sensitivity("UNKNOWN_TICKER", usd_change_pct=5.0, rate_change_bps=-100, oil_change_pct=10.0)
    assert "total_impact_pct" in res
    assert "resilience_badge" in res


def test_sensitivity_tornado_chart():
    res = analyze_macro_sensitivity("THYAO", usd_change_pct=15.0, rate_change_bps=500, oil_change_pct=20.0)
    fig = create_sensitivity_tornado_chart(res, "THYAO")
    assert fig is not None
    assert len(fig.data) > 0
    assert "Tornado" in fig.layout.title.text


# ==========================================
# 3. User Storage (Watchlist & Notes) Tests
# ==========================================

def test_user_storage_watchlist(tmp_path, monkeypatch):
    test_wl_file = tmp_path / "test_watchlist.json"
    monkeypatch.setattr("src.user_storage.WATCHLIST_FILE", test_wl_file)
    monkeypatch.setattr("src.user_storage.USER_DATA_DIR", tmp_path)

    # Initial load creates default
    wl = load_watchlist()
    assert "THYAO" in wl
    assert test_wl_file.exists()

    # Add ticker
    updated = add_to_watchlist("KCHOL")
    assert "KCHOL" in updated
    assert "KCHOL" in load_watchlist()

    # Add duplicate (should not duplicate)
    updated_dup = add_to_watchlist("KCHOL")
    assert updated_dup.count("KCHOL") == 1

    # Remove ticker
    after_remove = remove_from_watchlist("KCHOL")
    assert "KCHOL" not in after_remove
    assert "KCHOL" not in load_watchlist()


def test_user_storage_notes(tmp_path, monkeypatch):
    test_notes_file = tmp_path / "test_notes.json"
    monkeypatch.setattr("src.user_storage.NOTES_FILE", test_notes_file)
    monkeypatch.setattr("src.user_storage.USER_DATA_DIR", tmp_path)

    # Empty initially
    notes = load_saved_notes()
    assert notes == []

    # Add note
    new_note = add_saved_note(
        ticker="THYAO",
        title="3Ç24 Bilanço Notu",
        content="Yolcu doluluk oranı güçlü, FCF marjı korundu.",
        tags=["THYAO", "Bilanço"]
    )
    assert new_note["ticker"] == "THYAO"
    assert new_note["id"] is not None

    loaded = load_saved_notes()
    assert len(loaded) == 1
    assert loaded[0]["title"] == "3Ç24 Bilanço Notu"

    # Delete note
    deleted = delete_saved_note(new_note["id"])
    assert deleted is True
    assert len(load_saved_notes()) == 0


# ==========================================
# 4. Chat Export Tests
# ==========================================

def test_chat_export_markdown():
    messages = [
        {"role": "user", "content": "THYAO'nun kârlılığı nasıl?"},
        {"role": "assistant", "content": "THYAO'nun net kâr marjı %16.5 seviyesindedir."}
    ]
    md = export_chat_history_markdown(messages, current_ticker="THYAO")
    assert "# 📊 BIST Finansal Araştırma & Sohbet Raporu" in md
    assert "THYAO" in md
    assert "THYAO'nun kârlılığı nasıl?" in md
    assert "Yasal Uyarı" in md


def test_chat_export_html():
    messages = [
        {"role": "user", "content": "TUPRS temettü verimi nedir?"},
        {"role": "assistant", "content": "TUPRS temettü verimi yaklaşık **%10.5** seviyesindedir."}
    ]
    html_out = export_chat_history_html(messages, current_ticker="TUPRS")
    assert "<!DOCTYPE html>" in html_out
    assert "TUPRS" in html_out
    assert "TUPRS temettü verimi nedir?" in html_out
    assert "<strong>%10.5</strong>" in html_out
    assert "Yasal Uyarı" in html_out
