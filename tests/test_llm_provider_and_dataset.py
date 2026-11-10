"""Unit tests for Multi-Provider LLM Hub, Local Embeddings, and Fine-Tuning Dataset."""

import json
import pytest
from pathlib import Path

from src.llm_provider import (
    LLMProviderHub,
    PROVIDER_GEMINI,
    PROVIDER_OLLAMA,
    PROVIDER_COLAB,
)
from src.rag.embeddings import get_embedding_provider, LocalHuggingFaceEmbeddings, GeminiEmbeddings
from src.config import ROOT_DIR


def test_llm_provider_gemini():
    chat_model = LLMProviderHub.get_chat_model(
        provider=PROVIDER_GEMINI,
        model_name="gemini-3.1-flash-lite",
        temperature=0.1,
        streaming=False,
    )
    assert chat_model is not None


def test_llm_provider_ollama():
    chat_model = LLMProviderHub.get_chat_model(
        provider=PROVIDER_OLLAMA,
        model_name="qwen2.5:7b",
        base_url="http://localhost:11434/v1",
        streaming=True,
    )
    assert chat_model is not None


def test_llm_provider_colab():
    chat_model = LLMProviderHub.get_chat_model(
        provider=PROVIDER_COLAB,
        model_name="bist-financial-llm",
        base_url="https://mock-colab.ngrok-free.app/v1",
        streaming=True,
    )
    assert chat_model is not None


def test_llm_provider_test_connection_invalid():
    # Invalid host should return False
    ok, msg = LLMProviderHub.test_connection(
        provider="ollama",
        base_url="http://127.0.0.1:9999/v1"
    )
    assert ok is False
    assert "erişilemedi" in msg or "hata" in msg.lower()


def test_embedding_provider_factory():
    gem_emb = get_embedding_provider("gemini")
    assert isinstance(gem_emb, GeminiEmbeddings)

    loc_emb = get_embedding_provider("local")
    assert isinstance(loc_emb, LocalHuggingFaceEmbeddings)


def test_finetune_dataset_exists_and_valid():
    dataset_file = ROOT_DIR / "data" / "finetune_bist_dataset.jsonl"
    assert dataset_file.exists(), "finetune_bist_dataset.jsonl mevcut olmalıdır."

    with open(dataset_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) >= 100, f"En az 100 örnek bulunmalıdır, bulunan: {len(lines)}"

    first_item = json.loads(lines[0])
    assert "instruction" in first_item
    assert "input" in first_item
    assert "output" in first_item
    assert "category" in first_item
    assert "ticker" in first_item
    assert len(first_item["output"]) > 30


def test_sharegpt_dataset_valid():
    sharegpt_file = ROOT_DIR / "data" / "finetune_bist_sharegpt.jsonl"
    assert sharegpt_file.exists(), "finetune_bist_sharegpt.jsonl mevcut olmalıdır."

    with open(sharegpt_file, "r", encoding="utf-8") as f:
        first_line = f.readline()

    item = json.loads(first_line)
    assert "conversations" in item
    assert len(item["conversations"]) == 3
    assert item["conversations"][0]["from"] == "system"
    assert item["conversations"][1]["from"] == "human"
    assert item["conversations"][2]["from"] == "gpt"


def test_colab_notebook_valid_json():
    nb_file = ROOT_DIR / "notebooks" / "BIST_Finans_LLM_Colab_FineTuning.ipynb"
    assert nb_file.exists(), "Colab notebook dosyası mevcut olmalıdır."

    with open(nb_file, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert len(nb_data["cells"]) >= 10
    assert nb_data["nbformat"] == 4


def test_dpo_dataset_valid():
    dpo_file = ROOT_DIR / "data" / "finetune_bist_dpo.jsonl"
    assert dpo_file.exists(), "finetune_bist_dpo.jsonl mevcut olmalıdır."

    with open(dpo_file, "r", encoding="utf-8") as f:
        lines = [f.readline() for _ in range(5)]

    assert len(lines) == 5
    for l in lines:
        item = json.loads(l)
        assert "prompt" in item
        assert "chosen" in item
        assert "rejected" in item
        assert "<think>" in item["chosen"]
        assert "</think>" in item["chosen"]


def test_financial_tools_dispatch():
    from src.tools.financial_tools import execute_financial_tool, FINANCIAL_TOOL_DEFINITIONS
    assert len(FINANCIAL_TOOL_DEFINITIONS) >= 7

    res = execute_financial_tool("calculate_dupont", {"ticker": "THYAO"})
    assert "roe_pct" in res
    assert "net_profit_margin_pct" in res

    alt_res = execute_financial_tool("tool_calculate_altman_z", {"ticker": "THYAO"})
    assert "z_score" in alt_res
    assert "zone" in alt_res

    kelly_res = execute_financial_tool("calculate_kelly_allocation", {"ticker": "THYAO", "win_rate": 0.65, "win_loss_ratio": 2.0})
    assert "recommended_portfolio_allocation_pct" in kelly_res
    assert "full_kelly_pct" in kelly_res


def test_parse_cot_response_helper():
    from src.app import parse_cot_response
    sample = "<think>\n1. DuPont Hesabı: %12.5\n</think>\n**Sonuç Raporu:** DuPont harikadır."
    thought, clean = parse_cot_response(sample)
    assert thought == "1. DuPont Hesabı: %12.5"
    assert clean == "**Sonuç Raporu:** DuPont harikadır."

