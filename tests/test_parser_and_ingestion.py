"""Gelişmiş Finansal Parser, Tablo Bütünlüğü ve Contextual Chunking Testleri."""

import pytest
from langchain_core.documents import Document
from src.rag.parser import (
    detect_financial_section,
    extract_key_financial_terms,
    clean_text_preserving_tables,
)
from src.rag.ingestion import chunk_documents


def test_detect_financial_section_bilanco():
    text = "KONSOLİDE FİNANSAL DURUM TABLOSU (BİLANÇO)\nDönen Varlıklar: 154.230 Bin TL\nToplam Varlıklar"
    sec = detect_financial_section(text)
    assert sec == "Bilanço (Finansal Durum Tablosu)"


def test_detect_financial_section_gelir_tablosu():
    text = "KONSOLİDE KÂR VEYA ZARAR VE DİĞER KAPSAMLI GELİR TABLOSU\nSatış Gelirleri: 320.000\nEsas Faaliyet Kârı: 45.000"
    sec = detect_financial_section(text)
    assert sec == "Gelir Tablosu (Kâr/Zarar)"


def test_detect_financial_section_dipnot():
    text = "FİNANSAL TABLOLARA İLİŞKİN AÇIKLAYICI DİPNOTLAR\nDipnot 14 - Finansal Borçlar ve Kredi Vadeleri"
    sec = detect_financial_section(text)
    assert "Finansal Dipnotlar" in sec
    assert "14" in sec


def test_extract_key_financial_terms():
    sample_text = """
    Şirketimizin 2024 yılı hasılatı 185 milyar TL'ye ulaşırken, 
    FAVÖK rakamı %28 artışla 42 milyar TL olarak gerçekleşmiştir.
    Yönetim kurulu hisse başına net kâr üzerinden temettü dağıtılmasına karar vermiştir.
    """
    terms = extract_key_financial_terms(sample_text)
    assert "Net Kâr" in terms
    assert "FAVÖK/EBITDA" in terms
    assert "Hasılat/Satış" in terms
    assert "Temettü" in terms


def test_clean_text_preserving_tables():
    raw_table = """
    Dönen Varlıklar           154.230      112.450
    Nakit ve Nakit Benzerleri  45.120       38.900
    
    
    Ticari Alacaklar           62.300       45.100
    """
    cleaned = clean_text_preserving_tables(raw_table)
    assert "Dönen Varlıklar 154.230 112.450" in cleaned
    assert "Ticari Alacaklar 62.300 45.100" in cleaned
    assert "\n\n\n" not in cleaned


def test_contextual_chunking_headers():
    doc = Document(
        page_content="Şirketin 2024 yılı net kârı bir önceki yıla göre %45 artış kaydetmiştir.",
        metadata={
            "company": "Türk Hava Yolları",
            "ticker": "THYAO",
            "year": 2024,
            "doc_type": "Faaliyet Raporu",
            "section": "Yönetim & Operasyonel Rapor",
            "page": 18,
            "total_pages": 120
        }
    )
    chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50, add_context_header=True)
    assert len(chunks) >= 1
    # Chunk başında bağlamsal üstbilgi bulunmalı
    header_line = chunks[0].page_content.split("\n")[0]
    assert "Türk Hava Yolları" in header_line
    assert "THYAO" in header_line
    assert "2024" in header_line
    assert "Sayfa: 18" in header_line
