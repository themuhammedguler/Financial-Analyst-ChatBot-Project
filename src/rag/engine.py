import os
import re
from typing import List, Dict, Any, Optional, Generator
from langchain_community.vectorstores import FAISS
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
"""

FINANCIAL_PROMPT = PromptTemplate(
    template=SYSTEM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


class FinancialRAGEngine:
    """
    Sektörel seviyede RAG Analiz Motoru.
    - Çok turlu sohbet hafızası (Query Rewriting)
    - Metadata filtreleme (Şirket & Yıl bazlı)
    - MMR (Maximal Marginal Relevance) çeşitlilik araması
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
        self._vector_db = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        """FAISS vektör veritabanını yükler."""
        if os.path.exists(os.path.join(self.vectorstore_path, "index.faiss")):
            try:
                self._vector_db = FAISS.load_local(
                    self.vectorstore_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"[RAGEngine] Vektör veritabanı yüklenemedi: {e}")
                self._vector_db = None
        else:
            self._vector_db = None

    def get_llm(self, streaming: bool = True) -> ChatGoogleGenerativeAI:
        """İlgili LLM modelini yapılandırır."""
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=GOOGLE_API_KEY,
            streaming=streaming,
        )

    def detect_filters(self, query: str) -> Dict[str, Any]:
        """Kullanıcının sorgusundan şirket ve yıl filtresini otomatik tespit eder."""
        query_lower = query.lower()
        detected_company = None
        detected_year = None

        for comp, aliases in COMPANIES.items():
            if any(alias in query_lower for alias in aliases):
                detected_company = comp
                break

        for y in SUPPORTED_YEARS:
            if str(y) in query_lower:
                detected_year = y
                break

        return {"company": detected_company, "year": detected_year}

    def retrieve_documents(
        self,
        query: str,
        company_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        top_k: int = TOP_K_RETRIEVAL,
        use_mmr: bool = True,
    ) -> List[Document]:
        """
        Gelişmiş belge getirme (Retrieval).
        Filtreleme kriterlerine göre FAISS üzerinden MMR veya Benzerlik araması yapar.
        """
        if not self._vector_db:
            return []

        # Otomatik filtre tespiti (eğer kullanıcı arayüzden seçmediyse)
        auto_filters = self.detect_filters(query)
        effective_company = company_filter if company_filter and company_filter != "Tümü" else auto_filters["company"]
        effective_year = year_filter if year_filter and year_filter != "Tümü" else auto_filters["year"]

        # FAISS metadata filter dictionary
        filter_dict = {}
        if effective_company and effective_company != "Tümü":
            filter_dict["company"] = effective_company
        if effective_year and effective_year != "Tümü":
            try:
                filter_dict["year"] = int(effective_year)
            except ValueError:
                pass

        try:
            if filter_dict:
                # Filtreli arama: önce daha geniş bir havuz çekip filtre uygula
                initial_k = top_k * 4
                raw_docs = self._vector_db.similarity_search(query, k=initial_k)
                filtered = []
                for d in raw_docs:
                    match = True
                    for k, v in filter_dict.items():
                        if d.metadata.get(k) != v:
                            match = False
                            break
                    if match:
                        filtered.append(d)
                
                # Eğer filtrelenmiş sonuç yeterliyse onu dön, yoksa geniş havuzdan devam et
                if len(filtered) >= 2:
                    return filtered[:top_k]
                elif filtered:
                    return filtered + [d for d in raw_docs if d not in filtered][:top_k - len(filtered)]
            
            # Filtresiz veya doğrudan MMR araması
            if use_mmr:
                retriever = self._vector_db.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": top_k, "lambda_mult": MMR_LAMBDA, "fetch_k": top_k * 3}
                )
                return retriever.invoke(query)
            else:
                return self._vector_db.similarity_search(query, k=top_k)

        except Exception as e:
            print(f"[RAGEngine] Retrieval hatası: {e}")
            return self._vector_db.similarity_search(query, k=top_k)

    def stream_answer(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        company_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Soruya canlı streaming olarak token token yanıt üretir.
        Her yield adımında metin parçası (chunk), tamamlandığında ise kaynak belgeler döner.
        """
        chat_history = chat_history or []

        # 1. Takip sorusu varsa geçmiş bağlama göre yeniden yaz
        rewritten_query = self.rewriter.rewrite(question, chat_history)

        # 2. Vektör veritabanından en alakalı belgeleri getir
        docs = self.retrieve_documents(
            query=rewritten_query,
            company_filter=company_filter,
            year_filter=year_filter,
        )

        if not docs:
            yield {
                "type": "content",
                "content": "Veritabanında ilgili soruya ilişkin herhangi bir finansal rapor veya doküman bulunamadı. Lütfen vektör indeksinin oluşturulduğundan ve raporların yüklendiğinden emin olun."
            }
            yield {"type": "done", "docs": [], "rewritten_query": rewritten_query}
            return

        # 3. Belgeleri bağlam metni olarak formatla
        context_parts = []
        for idx, doc in enumerate(docs):
            src_name = doc.metadata.get("file_name", os.path.basename(doc.metadata.get("source", "Bilinmiyor")))
            page = doc.metadata.get("page", "Belirtilmemiş")
            comp = doc.metadata.get("company", "Bilinmeyen")
            yr = doc.metadata.get("year", "")
            
            header = f"--- [KAYNAK {idx+1}] Şirket: {comp} | Yıl: {yr} | Dosya: {src_name} | Sayfa: {page} ---"
            context_parts.append(f"{header}\n{doc.page_content}\n")

        context_str = "\n".join(context_parts)

        # 4. Prompt'u oluştur ve LLM'e streaming gönder
        formatted_prompt = FINANCIAL_PROMPT.format(context=context_str, question=question)
        llm = self.get_llm(streaming=True)

        try:
            for chunk in llm.stream(formatted_prompt):
                if chunk.content:
                    yield {"type": "content", "content": chunk.content}
        except Exception as e:
            yield {"type": "content", "content": f"\n\n**Hata oluştu:** {e}"}

        # 5. Kaynakları ve sorgu detayını tamamlandı mesajı ile teslim et
        yield {
            "type": "done",
            "docs": docs,
            "rewritten_query": rewritten_query
        }
