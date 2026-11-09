import os
import sys
import time
from pathlib import Path

project_root = Path(r"c:\Users\themu\OneDrive\Masaüstü\Kodex\Financial-Analyst-ChatBot-Project")
sys.path.insert(0, str(project_root))

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import DATA_DIR, VECTORSTORE_DIR, VECTORSTORE_PATH_STR
from src.rag.parser import parse_pdf_file
from src.rag.embeddings import GeminiEmbeddings

# Seçili 2023-2024 Raporları
target_pdfs = [
    DATA_DIR / "2024" / "31122024 Akbank Konsolide Finansal Tablolar ve Dipnotlar.pdf",
    DATA_DIR / "2024" / "THY A.O. Aralık 2024.pdf",
    DATA_DIR / "2024" / "tupras-konsolide-spk-31122024.pdf",
    DATA_DIR / "2024" / "THY A.O. YK Faaliyet Raporu Aralık 2024.pdf",
    DATA_DIR / "2023" / "31122023 Akbank Konsolide Finansal Tablolar ve Dipnotlar.pdf",
    DATA_DIR / "2023" / "THY A.O. Aralık 2023.pdf",
    DATA_DIR / "2023" / "Tupras-Konsolide-SPK-31122023.pdf",
    DATA_DIR / "2023" / "THY A.O. YK Faaliyet Raporu Aralık 2023.pdf",
]

# Mevcut olanları filtrele
target_pdfs = [p for p in target_pdfs if p.exists()]
print(f"Hedeflenen yüksek öncelikli rapor sayısı: {len(target_pdfs)}")

# Pacing: Free tier 100 RPM kuralına uygun (batch_size=20, delay=14s -> ~85 istek/dakika)
embeddings = GeminiEmbeddings(batch_size=20, delay_between_batches=14.0)
splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=150)

all_chunks = []
for p in target_pdfs:
    docs = parse_pdf_file(p)
    chunks = splitter.split_documents(docs)
    all_chunks.extend(chunks)
    print(f"- {p.name[:35]:35} -> {len(docs)} sayfa, {len(chunks)} chunk")

print(f"\nToplam chunk sayısı: {len(all_chunks)}. Vektör veritabanına indeksleniyor...")

db = None
batch_chunk_size = 60 # 3 adet 20'lik batch

for i in range(0, len(all_chunks), batch_chunk_size):
    sub_batch = all_chunks[i : i + batch_chunk_size]
    print(f"[{i+1}-{min(i+batch_chunk_size, len(all_chunks))}/{len(all_chunks)}] İndeksleniyor...")
    if db is None:
        db = FAISS.from_documents(sub_batch, embeddings)
    else:
        db.add_documents(sub_batch)
    
    # Ara kayıt al
    db.save_local(VECTORSTORE_PATH_STR)
    print(f"  -> Checkpoint kaydedildi ({min(i+batch_chunk_size, len(all_chunks))} chunk)")

print(f"\n[Başarılı] FAISS İndeksi {VECTORSTORE_PATH_STR} konumuna kaydedildi! Toplam chunk: {len(all_chunks)}")
