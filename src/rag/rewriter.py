from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.config import DEFAULT_LLM_MODEL, GOOGLE_API_KEY


REWRITE_PROMPT_TEMPLATE = """Aşağıda bir finansal analist asistanı ile kullanıcı arasındaki önceki sohbet geçmişi ve kullanıcının son sorusu verilmiştir.
Görevin, eğer son soru önceki konuşmaya atıfta bulunuyorsa (zamirler, eksik şirket isimleri, "peki bu", "bir önceki yıl", "oranı nedir" gibi ifadeler içeriyorsa), soruyu tek başına anlaşılabilir, arama motorunda ve vektör veritabanında en doğru sonuçları getirecek bağımsız bir finansal arama sorgusuna dönüştürmektir.

Kurallar:
- Eğer soru zaten tek başına net ve bağımsızsa, aynen veya sadece hafifçe optimize ederek bırak.
- Şirket adı (Akbank, THY, Tüpraş vb.) ve ilgili mali yıl (2020, 2021, 2022, 2023, 2024 vb.) bağlamdan biliniyorsa mutlaka sorguya dahil et.
- Sadece yeniden yazılmış sorguyu döndür, açıklama veya ek metin yazma.

Sohbet Geçmişi:
{chat_history}

Son Soru:
{question}

Yeniden Yazılmış Bağımsız Arama Sorgusu:"""

REWRITE_PROMPT = PromptTemplate(
    template=REWRITE_PROMPT_TEMPLATE,
    input_variables=["chat_history", "question"]
)


class QueryRewriter:
    """
    Çok turlu konuşmalarda takip sorularını (follow-up questions)
    bağlamı koruyarak bağımsız arama sorgularına dönüştüren modül.
    """

    def __init__(self, api_key: str = None, model: str = DEFAULT_LLM_MODEL):
        self.api_key = api_key or GOOGLE_API_KEY
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            google_api_key=self.api_key,
        )
        self.chain = REWRITE_PROMPT | self.llm

    def rewrite(self, question: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Geçmiş konuşmayı analiz ederek soruyu yeniden yazar.
        Eğer sohbet geçmişi boşsa veya sadece ilk selamlama varsa doğrudan orijinal soruyu döner.
        """
        # Kullanıcı ve asistan mesajlarını filtrele (ilk sistem selamlaması hariç)
        meaningful_history = [
            m for m in chat_history 
            if m.get("role") in ["user", "assistant"] and len(m.get("content", "")) > 0
        ]
        
        # Eğer geçmişte yeterli mesaj yoksa yeniden yazmaya gerek yok
        if len(meaningful_history) <= 1:
            return question.strip()

        # Son 4 mesajı bağlam olarak al (aşırı token yükünden kaçınmak için)
        recent_turns = meaningful_history[-4:]
        history_str = "\n".join(
            f"{'Kullanıcı' if m['role'] == 'user' else 'Asistan'}: {m['content']}"
            for m in recent_turns
        )

        try:
            response = self.chain.invoke({
                "chat_history": history_str,
                "question": question
            })
            rewritten = response.content.strip()
            # Tırnak işaretleri varsa temizle
            rewritten = rewritten.strip('"\'')
            return rewritten if rewritten else question
        except Exception as e:
            # Herhangi bir API hatasında orijinal soruyu bozmadan devam et
            print(f"Query rewrite fallback: {e}")
            return question
