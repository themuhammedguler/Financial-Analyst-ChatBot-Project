import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# .env dosyasından environment variable'ları yükle
load_dotenv()

# Google API anahtarını al
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Google API anahtarı bulunamadı. Lütfen .env dosyanızı kontrol edin.")
    st.stop()


# Vektör veritabanının yolu
DB_FAISS_PATH = 'vectorstore/db_faiss'

# --- RAG MİMARİSİNİN TEMEL FONKSİYONLARI ---

def get_vector_store():
    """
    Daha önce oluşturulmuş ve kaydedilmiş FAISS vektör veritabanını yükler.
    Embedding modeli olarak Gemini'yi kullanır.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_db
    except Exception as e:
        st.error(f"Vektör veritabanı yüklenirken bir hata oluştu: {e}")
        return None

def get_conversational_chain():
    """
    Soru-cevaplama zincirini (QA Chain) oluşturur. Bu zincir, LLM'in (Gemini Pro)
    nasıl davranacağını belirleyen bir prompt şablonu kullanır.
    """
    
    # LLM'e verilecek talimatları içeren prompt şablonu
    prompt_template = """
    Sen bir Finansal Analist Asistanısın. Görevin, sana verilen bağlamdaki (context) finansal raporları
    analiz ederek sorulan sorulara doğru ve net cevaplar vermektir.
    
    Bağlam (Context):
    {context}
    
    Soru:
    {question}
    
    Cevap:
    - Cevabını sadece ve sadece sana verilen bağlamdaki bilgilere dayanarak oluştur.
    - Eğer bağlamda sorunun cevabı yoksa, "Bu bilgiye sahip değilim." veya "Raporlarda bu konu hakkında bir bilgi bulamadım." şeklinde cevap ver. Kesinlikle bağlam dışı bilgi kullanma veya tahmin yürütme.
    - Cevaplarını madde işaretleri kullanarak veya kısa paragraflar halinde, kolay okunabilir bir formatta sun.
    - Finansal terimleri basit bir dille açıkla.
    """
    
    # LLM modelini başlat (Gemini Pro)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # Prompt şablonunu ve modeli kullanarak bir QA zinciri oluştur
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# --- STREAMLIT WEB ARAYÜZÜ ---

st.set_page_config(page_title="Finansal Analist Chatbot", page_icon="📈")
st.header("📈 Finansal Analist Chatbot")
st.write("Akbank, THY ve Tüpraş'ın finansal raporlarını kullanarak sorularınızı yanıtlar.")

# Vektör veritabanını yükle
vector_store = get_vector_store()

if vector_store:
    # Kullanıcıdan soru al
    user_question = st.text_input(
        "Lütfen sorunuzu buraya yazın:",
        placeholder="Akbank'ın dijitalleşme vizyonu hakkında bilgi verir misin?" # Örnek metni ekledik
    )

    if user_question:
        # Vektör veritabanından ilgili dokümanları bul (Similarity Search)
        # Bu aşama, RAG'in "Retrieval" (Getirme) kısmıdır.
        docs = vector_store.similarity_search(user_question, k=5) # En alakalı 5 parçayı getir
        
        # Soru-cevap zincirini al
        chain = get_conversational_chain()
        
        # Zinciri çalıştırarak cevabı üret
        # Bu aşama, RAG'in "Augmented Generation" (Zenginleştirilmiş Üretim) kısmıdır.
        with st.spinner("Cevap oluşturuluyor..."):
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("### Cevap")
            st.write(response["output_text"])

        # Cevabın hangi kaynaklardan üretildiğini göster
        with st.expander("Cevap için kullanılan kaynak metinleri görmek için tıklayın"):
            for i, doc in enumerate(docs):
                st.write(f"**Kaynak {i+1} (Sayfa: {doc.metadata.get('page', 'Bilinmiyor')})**")
                st.write(doc.page_content)
                st.write("---")
else:
    st.warning("Uygulamanın başlayabilmesi için vektör veritabanının başarıyla yüklenmesi gerekmektedir.")