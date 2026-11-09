import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from src.config import (
    DATA_DIR,
    VECTORSTORE_DIR,
    VECTORSTORE_PATH_STR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)
from src.rag.parser import parse_pdf_file
from src.rag.embeddings import GeminiEmbeddings


def get_all_pdf_files(data_dir: Path = DATA_DIR) -> List[Path]:
    """Tüm PDF dosyalarını data klasörü altından rekürsif olarak bulur."""
    return sorted(list(data_dir.glob("**/*.pdf")))


def load_and_parse_all_pdfs(pdf_files: List[Path]) -> List[Document]:
    """Tüm PDF'leri pypdfium2 ile hızlıca temizleyip Document nesnelerine dönüştürür."""
    all_docs = []
    print(f"\n[Ingestion] Toplam {len(pdf_files)} PDF dosyası taranıyor...")
    
    for idx, pdf_path in enumerate(pdf_files):
        t0 = time.time()
        docs = parse_pdf_file(pdf_path)
        all_docs.extend(docs)
        print(f"[{idx+1}/{len(pdf_files)}] {pdf_path.name[:38]:38} | {len(docs)} sayfa | {time.time()-t0:.2f}s")
        
    print(f"[Ingestion] Toplam {len(all_docs)} sayfa metin başarıyla yüklendi.")
    return all_docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Dokümanları finansal tabloların ve paragrafların bütünlüğünü koruyarak böler."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[Ingestion] Dokümanlar {len(chunks)} adet anlamsal parçaya (chunk) bölündü.")
    return chunks


def build_vector_store(
    chunks: List[Document],
    output_dir: str = VECTORSTORE_PATH_STR,
    batch_size: int = 50,
) -> FAISS:
    """
    Chunk'ları Gemini Embeddings ile vektörleştirip FAISS veritabanına kaydeder.
    Kota aşımını önlemek için otomatik backoff ve pacing kullanır.
    """
    embeddings = GeminiEmbeddings(batch_size=batch_size, delay_between_batches=1.0)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Ingestion] Vektör veritabanı oluşturuluyor ({len(chunks)} chunk)...")
    t0 = time.time()

    # FAISS veritabanını chunk'lardan oluştur
    db = FAISS.from_documents(chunks, embeddings)
    
    # Yerel olarak kaydet (Windows Türkçe karakter sorunlarına karşı güvenli path formatı)
    db.save_local(output_dir)
    print(f"[Ingestion] Vektör veritabanı başarıyla kaydedildi: {output_dir} ({time.time()-t0:.2f}s)")
    return db


def filter_high_value_financial_pages(docs: List[Document], max_pages: int = 60) -> List[Document]:
    """
    Yüzlerce sayfalık kurumsal faaliyet raporlarında yüksek sinyalli finansal,
    operasyonel ve yönetim mesajı içeren sayfaları akıllıca filtreler.
    Böylece kota tasarrufu sağlanır ve RAG arama kalitesi yükselir.
    """
    if len(docs) <= max_pages:
        return docs

    keywords = [
        "özet finansal", "finansal göstergeler", "gelir tablosu", "bilanço",
        "nakit akış", "favök", "ebitda", "net kâr", "net kar", "temettü",
        "yönetim kurulu başkanı", "genel müdür", "faaliyet sonuçları",
        "kapasite kullanım", "satış gelirleri", "yatırımlar", "kâr dağıtım"
    ]

    # İlk 10 sayfa (Yönetici özeti ve genel bakış) daima dahil edilir
    selected_indices = set(range(min(10, len(docs))))

    for idx, d in enumerate(docs):
        txt = d.page_content.lower()
        if any(kw in txt for kw in keywords):
            selected_indices.add(idx)

    # Sıralı olarak seçilen sayfaları döndür
    chosen_docs = [docs[i] for i in sorted(selected_indices)]
    print(f"[Ingestion] Akıllı Sayfa Filtresi: {len(docs)} sayfadan en kritik {len(chosen_docs)} sayfa seçildi.")
    return chosen_docs


def add_single_pdf_to_vectorstore(
    pdf_path: Path,
    vectorstore_path_str: str = VECTORSTORE_PATH_STR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    smart_filter: bool = True,
) -> bool:
    """
    Yeni indirilen tek bir PDF raporunu mevcut FAISS veritabanına artırımlı (incremental) ekler.
    BIST 100 şirketlerinin raporlarını sisteme dinamik olarak dahil etmeyi sağlar.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"[Ingestion] Hata: Dosya bulunamadı -> {pdf_path}")
        return False

    print(f"[Ingestion] Yeni rapor ekleniyor: {pdf_path.name}")
    docs = parse_pdf_file(pdf_path)
    if not docs:
        print("[Ingestion] Dokümandan metin çıkarılamadı.")
        return False

    if smart_filter and len(docs) > 40:
        docs = filter_high_value_financial_pages(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    embeddings = GeminiEmbeddings()

    try:
        if os.path.exists(os.path.join(vectorstore_path_str, "index.faiss")):
            db = FAISS.load_local(vectorstore_path_str, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(chunks)
        else:
            db = FAISS.from_documents(chunks, embeddings)

        db.save_local(vectorstore_path_str)
        print(f"[Ingestion] {pdf_path.name} başarıyla indekslendi ({len(chunks)} chunk).")
        return True
    except Exception as e:
        print(f"[Ingestion] Vektör veritabanına eklenirken hata: {e}")
        return False


if __name__ == "__main__":
    pdf_files = get_all_pdf_files()
    if not pdf_files:
        print("Data klasöründe PDF dosyası bulunamadı!")
        sys.exit(1)

    docs = load_and_parse_all_pdfs(pdf_files)
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
