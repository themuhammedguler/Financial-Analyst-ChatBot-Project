import os
import sys
import time
import shutil
from pathlib import Path

project_root = Path(r"c:\Users\themu\OneDrive\Masaüstü\Kodex\Financial-Analyst-ChatBot-Project")
sys.path.insert(0, str(project_root))

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import DATA_DIR, VECTORSTORE_DIR
from src.rag.parser import parse_pdf_file
from src.rag.embeddings import GeminiEmbeddings

# Önce eski indeksi yedekle
old_index = VECTORSTORE_DIR / "index.faiss"
backup_dir = VECTORSTORE_DIR.parent / "db_faiss_legacy_backup"
if old_index.exists() and not backup_dir.exists():
    print(f"[IndexBuilder] Eski FAISS indeksi {backup_dir} konumuna yedekleniyor...")
    shutil.copytree(str(VECTORSTORE_DIR), str(backup_dir))

# En kritik finansal raporları öncelikli olarak seç:
# 1. Tüm Finansal Tablolar ve Dipnotlar (2020-2024: 15 adet) -> Rakamların, bilançoların, kâr/zarar tablolarının %100'ü buradadır!
# 2. En güncel Faaliyet Raporları (2023 ve 2024: 6 adet) -> Strateji, vizyon, yatırım, sürdürülebilirlik buradadır!
all_pdfs = sorted(list(DATA_DIR.glob("**/*.pdf")))

priority_pdfs = []
for p in all_pdfs:
    name_l = p.name.lower()
    # Finansal tablolar
    if "finansal" in name_l or "spk" in name_l or "tablo" in name_l or "aralık" in name_l:
        priority_pdfs.append(p)
    # 2023 ve 2024 faaliyet raporları
    elif ("2023" in str(p) or "2024" in str(p)) and ("faaliyet" in name_l or "entegre" in name_l or "frat" in name_l):
        priority_pdfs.append(p)

print(f"[IndexBuilder] Toplam {len(all_pdfs)} PDF'ten öncelikli {len(priority_pdfs)} dosya işleniyor...")

embeddings = GeminiEmbeddings(batch_size=40, delay_between_batches=1.0)
splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)

db = None
total_chunks_indexed = 0

for idx, pdf_path in enumerate(priority_pdfs):
    t0 = time.time()
    try:
        docs = parse_pdf_file(pdf_path)
        if not docs:
            continue
        chunks = splitter.split_documents(docs)
        if not chunks:
            continue

        print(f"[{idx+1}/{len(priority_pdfs)}] İndeksleniyor: {pdf_path.name[:35]:35} | {len(chunks)} chunk...")
        
        if db is None:
            db = FAISS.from_documents(chunks, embeddings)
        else:
            db.add_documents(chunks)

        total_chunks_indexed += len(chunks)
        # Her 3 dosyada bir ara kayıt al (checkpoint)
        if (idx + 1) % 3 == 0 or (idx + 1) == len(priority_pdfs):
            db.save_local(str(VECTORSTORE_DIR))
            print(f"  --> Ara kayıt alındı (Checkpoint): Toplam {total_chunks_indexed} chunk kaydedildi.")

    except Exception as e:
        print(f"  [Hata] {pdf_path.name}: {e}")

# Son kayıt
if db:
    db.save_local(str(VECTORSTORE_DIR))
    print(f"\n[IndexBuilder] İndeksleme BAŞARIYLA tamamlandı! Toplam chunk sayısı: {total_chunks_indexed}")
