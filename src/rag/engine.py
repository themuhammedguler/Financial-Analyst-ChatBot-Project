import os
import re
from typing import List, Dict, Any, Optional, Generator
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from src.config import (
    VECTORSTORE_DIR,
    VECTORSTORE_PATH_STR,
    DEFAULT_LLM_MODEL,
    ADVANCED_LLM_MODEL,
    TOP_K_RETRIEVAL,
    MMR_LAMBDA,
    COMPANIES,
    SUPPORTED_YEARS,
    GOOGLE_API_KEY,
)
from src.rag.embeddings import GeminiEmbeddings
from src.rag.rewriter import QueryRewriter
from src.rag.guard import HallucinationGuard


SYSTEM_PROMPT_TEMPLATE = """Sen, Borsa İstanbul (BIST) şirketlerinin finansal tablolarını, faaliyet raporlarını ve KAP bildirimlerini derinlemesine analiz eden **Kıdemli bir Finansal Analist ve Araştırma Uzmanısın**.

Aşağıda şirketlerin KAP'a sunduğu resmi finansal raporlarından alınmış kanıt metinleri (bağlam/context) yer almaktadır.

BAĞLAM (KAP RAPORLARI):
---------------------
{context}
---------------------

KULLANICININ SORUSU:
{question}

ANALİZ VE CEVAP TALİMATLARI:
1. **Veriye Dayalı Analiz:** Cevabını SADECE ve SADECE yukarıdaki bağlamda yer alan resmi finansal verilere dayandır. Tahmin yürütme veya bağlam dışı genel bilgi ekleme.
2. **Eksik Bilgi Kuralı:** Eğer sorulan bilgi bağlamdaki raporlarda yer almıyorsa veya net değilse; "İncelenen raporlarda bu konuya ilişkin net bir veri bulunamadı." şeklinde açıkça belirt.
3. **Finansal Birimler ve Sayısal Hassasiyet:** Rakamları verirken birimlere (TL, Bin TL, Milyon TL, Milyar TL, USD, EUR, %) azami dikkat et. Finansal dipnotlarda genellikle tutarlar 'Bin TL' bazında verilir; bunu doğru ifade et.
4. **Yapı ve Okunabilirlik:** 
   - Yanıtını kurumsal bir finansal bülten formatında, yönetici özeti (Executive Summary), temel göstergeler (Key Metrics - madde işaretleriyle) ve varsa risk/trend değerlendirmesi şeklinde organize et.
   - Önemli finansal rasyoları veya büyüme oranlarını kalın (bold) harflerle vurgula.
5. **Kaynak Şeffaflığı:** Rapor adını veya ilgili mali yılı cevabın içinde doğal olarak belirt.
6. **Şirket Karşılaştırması (Peer Comparison Matrisi):** Eğer bağlamda veya soruda birden fazla şirket karşılaştırılıyorsa (Örn: THY vs Pegasus, Akbank vs Garanti), cevabında mutlaka her iki şirketin temel finansal rasyolarını (F/K, PD/DD, Kâr Marjı, Borçluluk) ve operasyonel hacimlerini yan yana özetleyen şık bir **Markdown Kıyaslama Tablosu (Peer Matrix)** sun ve ardından analitik bir mukayese değerlendirmesi yap.
"""

FINANCIAL_PROMPT = PromptTemplate(
    template=SYSTEM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


class FinancialRAGEngine:
    """
    Sektörel seviyede Finansal RAG Analiz Motoru.
    - Çok turlu sohbet hafızası (Query Rewriting)
    - Hibrit Arama (Hybrid Search: FAISS Vektör + BM25 Anahtar Kelime)
    - Metadata filtreleme (BIST 100 Şirketleri & Yıl bazlı)
    - Hallucination Guard (Sayısal ve İddia Doğrulama Katmanı)
    - Gerçek zamanlı LLM Token Streaming
    """

    def __init__(
        self,
        vectorstore_path: Optional[str] = None,
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.2,
    ):
        self.vectorstore_path = vectorstore_path or VECTORSTORE_PATH_STR
        self.model_name = model_name
        self.temperature = temperature
        self.embeddings = GeminiEmbeddings()
        self.rewriter = QueryRewriter(model=DEFAULT_LLM_MODEL)
        self.guard = HallucinationGuard()
        self._vector_db = None
        self._bm25_retriever = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        """FAISS vektör veritabanını ve BM25 seyrek indeksini yükler."""
        if os.path.exists(os.path.join(self.vectorstore_path, "index.faiss")):
            try:
                self._vector_db = FAISS.load_local(
                    self.vectorstore_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                # BM25 Sparse İndeksini bellekteki doküman havuzundan otomatik oluştur
                all_docs = list(self._vector_db.docstore._dict.values())
                if all_docs:
                    self._bm25_retriever = BM25Retriever.from_documents(all_docs)
                    self._bm25_retriever.k = TOP_K_RETRIEVAL * 2
                    print(f"[RAGEngine] Hibrit Arama Aktif: {len(all_docs)} parça FAISS ve BM25 ile indekslendi.")
            except Exception as e:
                print(f"[RAGEngine] Vektör veritabanı yüklenemedi: {e}")
                self._vector_db = None
                self._bm25_retriever = None
        else:
            self._vector_db = None
            self._bm25_retriever = None

    def get_llm(
        self, 
        model: Optional[str] = None, 
        streaming: bool = True, 
        max_retries: int = 1
    ) -> ChatGoogleGenerativeAI:
        """İlgili LLM modelini yapılandırır."""
        return ChatGoogleGenerativeAI(
            model=model or self.model_name,
            temperature=self.temperature,
            google_api_key=GOOGLE_API_KEY,
            streaming=streaming,
            max_retries=max_retries,
        )

    def detect_filters(self, query: str) -> Dict[str, Any]:
        """Kullanıcının sorgusundan şirket(ler) ve yıl filtresini otomatik tespit eder."""
        query_lower = query.lower()
        detected_companies = []
        detected_year = None

        for comp, aliases in COMPANIES.items():
            if any(alias in query_lower for alias in aliases):
                if comp not in detected_companies:
                    detected_companies.append(comp)

        for y in SUPPORTED_YEARS:
            if str(y) in query_lower:
                detected_year = y
                break

        primary_company = detected_companies[0] if detected_companies else None
        return {
            "company": primary_company,
            "companies": detected_companies,
            "is_comparison": len(detected_companies) >= 2,
            "year": detected_year,
        }

    def retrieve_documents(
        self,
        query: str,
        company_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        top_k: int = TOP_K_RETRIEVAL,
        use_hybrid: bool = True,
        use_mmr: bool = False,
        **kwargs
    ) -> List[Document]:
        """
        Gelişmiş Hibrit Belge Getirme (Hybrid Retrieval: Dense FAISS + Sparse BM25).
        - FAISS: Anlamsal / semantik benzerlik
        - BM25: Rakam, oran, dipnot numarası ve finansal terim tam eşleşmesi
        - Metadata filtreleme ve Reciprocal Rank Fusion (RRF) harmanlaması
        """
        if not self._vector_db:
            return []

        # Otomatik filtre tespiti
        auto_filters = self.detect_filters(query)
        effective_company = company_filter if company_filter and company_filter != "Tümü" else auto_filters["company"]
        effective_year = year_filter if year_filter and year_filter != "Tümü" else auto_filters["year"]

        filter_dict = {}
        if effective_company and effective_company != "Tümü":
            filter_dict["company"] = effective_company
        if effective_year and effective_year != "Tümü":
            try:
                filter_dict["year"] = int(effective_year)
            except ValueError:
                pass

        # 1. Dense (FAISS) Arama
        dense_docs = []
        try:
            dense_docs = self._vector_db.similarity_search(query, k=top_k * 2)
        except Exception as e:
            print(f"[RAGEngine] FAISS arama hatası: {e}")

        # 2. Sparse (BM25) Arama
        sparse_docs = []
        if use_hybrid and self._bm25_retriever:
            try:
                sparse_docs = self._bm25_retriever.invoke(query)
            except Exception as e:
                print(f"[RAGEngine] BM25 arama hatası: {e}")

        # 3. Reciprocal Rank Fusion (RRF) ile Skorlama
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Dense skorları (Ağırlık: 0.6)
        for rank, doc in enumerate(dense_docs):
            key = f"{doc.metadata.get('file_name', '')}_{doc.metadata.get('page', '')}_{doc.page_content[:50]}"
            doc_scores[key] = doc_scores.get(key, 0.0) + (0.6 / (60 + rank))
            doc_map[key] = doc

        # Sparse skorları (Ağırlık: 0.4)
        for rank, doc in enumerate(sparse_docs):
            key = f"{doc.metadata.get('file_name', '')}_{doc.metadata.get('page', '')}_{doc.page_content[:50]}"
            doc_scores[key] = doc_scores.get(key, 0.0) + (0.4 / (60 + rank))
            doc_map[key] = doc

        # RRF skoruna göre azalan sırada sırala
        ranked_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        combined_docs = [doc_map[k] for k in ranked_keys]

        # 4. Çoklu Şirket Karşılaştırması (Peer Comparison) Modu
        if auto_filters.get("is_comparison") and len(auto_filters.get("companies", [])) >= 2 and (not company_filter or company_filter == "Tümü"):
            peer_docs = []
            per_comp_k = max(2, top_k // 2)
            for c_name in auto_filters["companies"][:2]:
                c_matches = [d for d in combined_docs if d.metadata.get("company") == c_name]
                if not c_matches:
                    try:
                        c_matches = self._vector_db.similarity_search(f"{c_name} {query}", k=per_comp_k)
                    except Exception:
                        c_matches = []
                peer_docs.extend(c_matches[:per_comp_k])

            if len(peer_docs) >= 2:
                remaining = [d for d in combined_docs if d not in peer_docs]
                return peer_docs + remaining[:max(0, top_k - len(peer_docs))]

        # 5. Metadata Filtresini Uygula
        if filter_dict:
            filtered_docs = []
            for doc in combined_docs:
                matches = True
                for fk, fv in filter_dict.items():
                    if doc.metadata.get(fk) != fv:
                        matches = False
                        break
                if matches:
                    filtered_docs.append(doc)

            if len(filtered_docs) >= 2:
                return filtered_docs[:top_k]
            elif filtered_docs:
                # Eşleşenleri en başa koy, kalanı havuzdan tamamla
                return filtered_docs + [d for d in combined_docs if d not in filtered_docs][:top_k - len(filtered_docs)]

        return combined_docs[:top_k] if combined_docs else dense_docs[:top_k]

    def answer_query(
        self,
        question: str,
        company_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Soruya tek seferde (senkron) yanıt üretir ve Hallucination Guard denetiminden geçirir.
        Evaluation benchmark'ları ve REST API uç noktaları tarafından kullanılır.
        """
        docs = self.retrieve_documents(
            query=question,
            company_filter=company_filter,
            year_filter=year_filter,
        )

        if not docs:
            return {
                "answer": "İlgili şirket veya konuya ilişkin veri tabanında resmi rapor bulunamadı.",
                "docs": [],
                "guard_result": {"groundedness_score": 1.0, "is_safe": True}
            }

        context_parts = []
        for idx, doc in enumerate(docs):
            src_name = doc.metadata.get("file_name", "Bilinmiyor")
            page = doc.metadata.get("page", "-")
            comp = doc.metadata.get("company", "Bilinmeyen")
            yr = doc.metadata.get("year", "")
            context_parts.append(f"--- [KAYNAK {idx+1}] {comp} {yr} (Sayfa {page}) ---\n{doc.page_content}\n")

        context_str = "\n".join(context_parts)
        prompt_text = FINANCIAL_PROMPT.format(context=context_str, question=question)

        fallback_candidates = [self.model_name, DEFAULT_LLM_MODEL, "gemini-3.5-flash", "gemini-3.8-flash"]
        models_to_try = []
        for m in fallback_candidates:
            if m and m not in models_to_try:
                models_to_try.append(m)

        answer_text = ""
        for model_to_use in models_to_try:
            try:
                llm = self.get_llm(model=model_to_use, streaming=False, max_retries=1)
                response = llm.invoke(prompt_text)
                answer_text = response.content if hasattr(response, "content") else str(response)
                if answer_text:
                    break
            except Exception as e:
                print(f"[RAGEngine] answer_query '{model_to_use}' hatası: {e}. Alternatif deneniyor...")
                continue

        if not answer_text:
            answer_text = "Finansal model şu anda yanıt üretemedi. Lütfen daha sonra tekrar deneyiniz."

        # Hallucination Guard ile sayısal doğrulama yap
        guard_result = self.guard.verify_claims(answer_text, docs)

        return {
            "answer": answer_text,
            "docs": docs,
            "guard_result": guard_result,
        }

    def stream_answer(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        company_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Soruya canlı streaming olarak token token yanıt üretir.
        Her yield adımında metin parçası (chunk), tamamlandığında ise kaynak belgeler
        ve Hallucination Guard doğrulama skoru döner.
        """
        chat_history = chat_history or []
        rewritten_query = self.rewriter.rewrite(question, chat_history)

        docs = self.retrieve_documents(
            query=rewritten_query,
            company_filter=company_filter,
            year_filter=year_filter,
        )

        if not docs:
            yield {
                "type": "content",
                "content": "Veritabanında ilgili soruya ilişkin herhangi bir finansal rapor veya doküman bulunamadı."
            }
            yield {
                "type": "done",
                "docs": [],
                "rewritten_query": rewritten_query,
                "guard_result": {"groundedness_score": 1.0, "is_safe": True}
            }
            return

        context_parts = []
        for idx, doc in enumerate(docs):
            src_name = doc.metadata.get("file_name", os.path.basename(doc.metadata.get("source", "Bilinmiyor")))
            page = doc.metadata.get("page", "Belirtilmemiş")
            comp = doc.metadata.get("company", "Bilinmeyen")
            yr = doc.metadata.get("year", "")
            header = f"--- [KAYNAK {idx+1}] Şirket: {comp} | Yıl: {yr} | Dosya: {src_name} | Sayfa: {page} ---"
            context_parts.append(f"{header}\n{doc.page_content}\n")

        context_str = "\n".join(context_parts)
        formatted_prompt = FINANCIAL_PROMPT.format(context=context_str, question=question)

        # Hata toleranslı model sırası (Kota/429 durumunda otomatik fallback)
        fallback_candidates = [self.model_name, DEFAULT_LLM_MODEL, "gemini-3.5-flash", "gemini-3.8-flash"]
        models_to_try = []
        for m in fallback_candidates:
            if m and m not in models_to_try:
                models_to_try.append(m)

        full_answer = ""
        stream_success = False

        for model_to_use in models_to_try:
            try:
                llm = self.get_llm(model=model_to_use, streaming=True, max_retries=1)
                full_answer = ""
                for chunk in llm.stream(formatted_prompt):
                    if chunk.content:
                        full_answer += chunk.content
                        yield {"type": "content", "content": chunk.content}
                stream_success = True
                break
            except Exception as e:
                print(f"[RAGEngine] Model '{model_to_use}' streaming hatası: {e}. Alternatif model deneniyor...")
                continue

        if not stream_success:
            yield {"type": "content", "content": "\n\n⚠️ Ücretsiz Gemini API kota sınırına ulaşıldı veya model yanıt veremedi. Lütfen birkaç dakika sonra tekrar deneyin."}

        # Hallucination Guard denetimi
        guard_result = self.guard.verify_claims(full_answer, docs)

        yield {
            "type": "done",
            "docs": docs,
            "rewritten_query": rewritten_query,
            "guard_result": guard_result,
        }
