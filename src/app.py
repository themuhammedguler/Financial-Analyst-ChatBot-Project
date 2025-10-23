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

# --- STREAMLIT CHATBOT ARAYÜZÜ ---

st.set_page_config(page_title="Finansal Analist Asistanı", page_icon="📈", layout="wide")
st.title("📈 Finansal Analist Asistanı")
st.divider()

# Vektör veritabanını yükle
vector_store = get_vector_store()

if not vector_store:
    st.warning("Uygulamanın başlayabilmesi için vektör veritabanının başarıyla yüklenmesi gerekmektedir.")
    st.stop()

# Adım 1: Sohbet geçmişini başlatma (DEĞİŞTİRİLMİŞ İLK MESAJ İLE)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", 
         "content": "Merhaba! Ben sizin Finansal Analist Asistanınızım. **Akbank, THY ve Tüpraş**'a ait **finansal tablolar ve yıllık faaliyet raporları** hakkında istediğiniz soruyu sorabilirsiniz."}
    ]

# Adım 2: Geçmişteki tüm mesajları ekrana yazdırma
# Bu döngü, her etkileşimde yeniden çalışarak geçmişi ekranda tutar.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Adım 3: Kullanıcıdan yeni bir girdi alma
# st.chat_input, sayfanın altında sabit bir giriş kutusu oluşturur.
if prompt := st.chat_input("Akbank'ın dijitalleşme vizyonu hakkında bilgi verir misin?"):
    
    # a. Kullanıcının mesajını sohbet geçmişine ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # b. Kullanıcının mesajını ekrana anında yazdır
    with st.chat_message("user"):
        st.markdown(prompt)

    # c. Asistanın cevabını oluşturma (RAG süreci)
    with st.chat_message("assistant"):
        with st.spinner("Analiz ediliyor..."):
            # RAG zincirini çalıştır (eski kodunuzdaki mantığın aynısı)
            docs = vector_store.similarity_search(prompt, k=5)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
            
            # Cevabı ve kaynakları ekrana yazdır
            assistant_response = response["output_text"]
            st.markdown(assistant_response)

            # Kaynakları cevabın altına gizlenmiş bir şekilde ekle
            with st.expander("Referans Alınan Kaynak Metinleri Gör"):
                for i, doc in enumerate(docs):
                    source_filename = os.path.basename(doc.metadata.get('source', 'Bilinmiyor'))
                    st.info(f"**Kaynak {i+1}** | Dosya: `{source_filename}` | Sayfa: `{doc.metadata.get('page', 'Bilinmiyor')}`")
                    st.text_area(label="", value=doc.page_content, height=150, key=f"expander_source_{i}")

    # d. Asistanın cevabını da sohbet geçmişine ekle
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})