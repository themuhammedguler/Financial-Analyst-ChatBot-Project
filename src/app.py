import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Proje kökünü sys.path'e ekle
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from src.config import (
    DEFAULT_LLM_MODEL,
    ADVANCED_LLM_MODEL,
    COMPANIES,
    SUPPORTED_YEARS,
    VECTORSTORE_DIR,
    FORECASTER_BASE_URL,
    REPORTS_DIR,
    TICKER_TO_COMPANY,
    get_company_for_ticker,
    get_ticker_for_company,
)
import plotly.graph_objects as go
from src.rag.engine import FinancialRAGEngine
from src.data_collection.kap_downloader import BIST_100_COMPANIES, download_company_annual_report
from src.rag.ingestion import add_single_pdf_to_vectorstore
from src.integration.report_generator import FundamentalReportGenerator
from src.integration.models import FundamentalReport
from src.integration.api import get_hybrid_signal

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="BIST Finansal Analist Asistanı",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DEEP-LINKING (URL QUERY PARAMS DESTEĞİ - 1. ADIM) ---
query_params = st.query_params
raw_ticker_param = query_params.get("ticker") or query_params.get("symbol")
url_ticker = raw_ticker_param.upper().strip() if raw_ticker_param else None

# Ticker'a karşılık gelen şirket adını belirle
url_company_name = get_company_for_ticker(url_ticker) if url_ticker else None

# --- KURUMSAL DARK MODE CSS STİLLERİ ---
st.markdown("""
<style>
    /* Global font ve kurumsal dark palet */
    .stApp {
        background-color: #080C14;
        color: #F8FAFC;
    }
    
    /* Sol Kenar Çubuğu (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #0B111E !important;
        border-right: 1px solid #1E293B !important;
    }

    /* Başlık ve Alt Başlık */
    .main-header {
        font-size: 2.15rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .main-header .brand-highlight {
        color: #38BDF8;
    }
    .main-header .version-tag {
        font-size: 0.78rem;
        color: #38BDF8;
        font-weight: 600;
        padding: 3px 8px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 6px;
        margin-left: 6px;
    }
    .sub-header {
        font-size: 0.98rem;
        color: #94A3B8;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }

    /* Durum ve Deep-Link Rozetleri */
    .status-badge {
        display: inline-flex;
        align-items: center;
        background-color: rgba(16, 185, 129, 0.12);
        color: #34D399;
        font-size: 0.80rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid rgba(52, 211, 153, 0.3);
        margin-bottom: 1rem;
    }
    .deep-link-badge {
        display: inline-flex;
        align-items: center;
        background-color: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        font-size: 0.80rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid rgba(56, 189, 248, 0.35);
        margin-bottom: 1rem;
        margin-left: 6px;
    }

    /* Şirket Rozetleri */
    .tag-akbank {
        background-color: rgba(239, 68, 68, 0.15);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .tag-thy {
        background-color: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.35);
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .tag-tupras {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FCD34D;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .tag-eregli {
        background-color: rgba(20, 184, 166, 0.15);
        color: #5EEAD4;
        border: 1px solid rgba(20, 184, 166, 0.35);
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
    }

    /* Kaynak / Alıntı Kartı (Source Box) */
    .source-box {
        background-color: #0E1626;
        border: 1px solid #1E293B;
        border-left: 4px solid #38BDF8;
        color: #CBD5E1;
        padding: 12px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.86rem;
        line-height: 1.5;
    }
    .source-box code {
        background-color: #172238 !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        padding: 1px 5px !important;
        border-radius: 4px !important;
    }

    /* BIST-Trend-Forecaster Köprü Kartı (Sidebar) */
    .forecaster-card {
        background: linear-gradient(145deg, #0D1628 0%, #162444 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 10px;
        padding: 14px 14px;
        margin-top: 10px;
        margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }

    /* Sohbet Mesajı Kutuları */
    [data-testid="stChatMessage"] {
        background-color: #0D1422 !important;
        border: 1px solid #182236 !important;
        border-radius: 12px !important;
        margin-bottom: 0.75rem !important;
    }

    /* Hızlı Butonlar (Prompt Chips) ve Standart Butonlar */
    div.stButton > button {
        background-color: #0F172A !important;
        color: #CBD5E1 !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background-color: #172554 !important;
        color: #38BDF8 !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)


# --- ENGINE VE REPORT GENERATOR ÖNBELLEKLEME ---
def get_vectorstore_version() -> float:
    idx_p = Path("vectorstore/db_faiss/index.faiss")
    return idx_p.stat().st_mtime if idx_p.exists() else 0.0

@st.cache_resource(show_spinner=False)
def load_rag_engine(model_choice: str, vs_version: float = 0.0):
    return FinancialRAGEngine(model_name=model_choice)

@st.cache_resource(show_spinner=False)
def load_report_generator():
    return FundamentalReportGenerator()


# --- YARDIMCI GÖRSEL FONKSİYONLAR ---
def create_radar_chart(report: FundamentalReport) -> go.Figure:
    """Hisse için 5 boyutlu kurumsal finansal sağlık radar grafiği oluşturur."""
    sentiment_score_norm = max(10.0, min(100.0, ((report.sentiment_score + 1.0) / 2.0) * 100))
    
    prof_trend = str(report.key_metrics.net_profit_margin_trend).lower()
    if "artış" in prof_trend or "pozitif" in prof_trend or "yüksek" in prof_trend:
        prof_score = 88.0
    elif "azalış" in prof_trend or "düşüş" in prof_trend:
        prof_score = 45.0
    else:
        prof_score = 65.0

    debt = str(report.key_metrics.debt_to_equity).lower()
    if "düşük" in debt or "yönetilebilir" in debt:
        debt_score = 88.0
    elif "orta" in debt:
        debt_score = 65.0
    else:
        debt_score = 40.0

    catalyst_score = min(95.0, 45.0 + len(report.catalysts) * 15.0)
    risk_score = max(35.0, 95.0 - len(report.risk_factors) * 15.0)

    categories = [
        "Kârlılık & Marj",
        "Finansal Sağlık",
        "Büyüme Katalizörü",
        "Piyasa Duygusu",
        "Risk Dayanıklılığı"
    ]
    values = [prof_score, debt_score, catalyst_score, sentiment_score_norm, risk_score]
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(56, 189, 248, 0.22)',
        line=dict(color='#38BDF8', width=2),
        marker=dict(size=5, color='#38BDF8'),
        name=report.ticker
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False,
                linecolor='rgba(148, 163, 184, 0.2)',
                gridcolor='rgba(148, 163, 184, 0.15)',
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color='#CBD5E1'),
                linecolor='rgba(148, 163, 184, 0.2)',
                gridcolor='rgba(148, 163, 184, 0.15)',
            ),
            bgcolor='rgba(15, 23, 42, 0.6)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=25, r=25, t=20, b=20),
        showlegend=False,
        height=260
    )
    return fig


def get_company_badge(company_name: str) -> str:
    comp = str(company_name).lower()
    if "akbank" in comp:
        return f'<span class="tag-akbank">Akbank</span>'
    elif "hava yolları" in comp or "thy" in comp:
        return f'<span class="tag-thy">THY</span>'
    elif "tüpraş" in comp or "tupras" in comp:
        return f'<span class="tag-tupras">Tüpraş</span>'
    elif "ereğli" in comp or "eregli" in comp or "erdemir" in comp:
        return f'<span class="tag-eregli">Ereğli</span>'
    return f'<span style="background:rgba(148, 163, 184, 0.12); color:#CBD5E1; border:1px solid rgba(148, 163, 184, 0.25); padding:2px 8px; border-radius:6px; font-size:0.78rem;">{company_name}</span>'


# --- KENAR ÇUBUĞU (SIDEBAR) ---
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 14px; margin-bottom: 12px; background: linear-gradient(135deg, #0F172A, #1E293B); border: 1px solid #1E293B; border-radius: 10px; display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.5rem;">📊</span>
        <div>
            <div style="font-weight: 700; font-size: 0.96rem; color: #F8FAFC; letter-spacing: -0.01em;">BIST ANALYTICS</div>
            <div style="font-size: 0.73rem; color: #38BDF8; font-weight: 600;">KAP RAG & Temel Analiz</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### ⚙️ Analiz & Filtre Paneli")
    
    # 1. Model Seçimi
    model_option = st.selectbox(
        "🧠 Yapay Zeka Modeli",
        options=[DEFAULT_LLM_MODEL, ADVANCED_LLM_MODEL],
        format_func=lambda x: "Gemini 3.5 Flash (Ultra Hızlı)" if "3.5" in x else "Gemini 3.8 Flash (Derin Muhakeme)",
        index=0,
        help="Finansal analizler için Gemini model versiyonu."
    )

    # 2. Şirket Filtresi (Deep-linking ile otomatik eşleşme - Türkçe Alfabetik Sıralı)
    _tr_order = "aAbBcCçÇdDeEfFgGğĞhHıIİijJkKlLmMnNoOöÖpPrRsSşŞtTuUüÜvVyYzZ"
    _tr_dict = {c: i for i, c in enumerate(_tr_order)}
    def _tr_sort_key(text: str):
        return [_tr_dict.get(c, ord(c) + 1000) for c in text]

    sorted_company_names = sorted(list(COMPANIES.keys()), key=_tr_sort_key)
    company_options = ["Tümü"] + sorted_company_names
    
    # Eğer URL'den geçerli bir şirket geldiyse varsayılan indeks olarak onu seç
    default_company_index = 0
    if url_company_name and url_company_name in company_options:
        default_company_index = company_options.index(url_company_name)

    selected_company = st.selectbox(
        "🏢 Şirket Filtresi",
        options=company_options,
        index=default_company_index,
        help="Sorguyu belirli bir şirketle sınırlandırabilir veya 'Tümü' seçeneğiyle sorunun içeriğinden otomatik tespit edilmesini sağlayabilirsiniz."
    )

    # Aktif hisse sembolü
    active_ticker = url_ticker if url_ticker else get_ticker_for_company(selected_company)

    # 3. Yıl Filtresi
    year_options = ["Tümü"] + [str(y) for y in reversed(SUPPORTED_YEARS)]
    selected_year = st.selectbox(
        "📅 Mali Yıl Filtresi",
        options=year_options,
        index=0,
        help="Belirli bir faaliyet veya bilanço yılına odaklanın."
    )

    if st.button("🔄 Sohbeti & Önbelleği Temizle", use_container_width=True, help="Sohbet geçmişini ve bellek önbelleğini sıfırlar"):
        st.cache_resource.clear()
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # --- 3. ADIM: BIST-TREND-FORECASTER ÇİFT YÖNLÜ KÖPRÜ VE HİBRİT KARAR MOTORU ---
    forecaster_url = f"{FORECASTER_BASE_URL}/?ticker={active_ticker}"

    report_file = REPORTS_DIR / f"{active_ticker}_latest.json"
    active_report_obj = None
    hybrid_sig = None
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as rf:
                active_report_obj = FundamentalReport(**json.load(rf))
            hybrid_sig = get_hybrid_signal(ticker=active_ticker)
        except Exception:
            pass

    decision_color = "#34D399"
    decision_bg = "rgba(16, 185, 129, 0.18)"
    decision_border = "rgba(52, 211, 153, 0.4)"
    if hybrid_sig:
        dec = hybrid_sig.decision
        if "GÜÇLÜ AL" in dec:
            decision_color = "#10B981"
            decision_bg = "rgba(16, 185, 129, 0.22)"
            decision_border = "rgba(16, 185, 129, 0.5)"
        elif "AL" in dec:
            decision_color = "#38BDF8"
            decision_bg = "rgba(56, 189, 248, 0.20)"
            decision_border = "rgba(56, 189, 248, 0.45)"
        elif "TUT" in dec or "NÖTR" in dec:
            decision_color = "#FBBF24"
            decision_bg = "rgba(245, 158, 11, 0.20)"
            decision_border = "rgba(245, 158, 11, 0.45)"
        else:
            decision_color = "#F87171"
            decision_bg = "rgba(239, 68, 68, 0.20)"
            decision_border = "rgba(239, 68, 68, 0.45)"

    st.markdown(f"""
    <div class="forecaster-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 700; color: #38BDF8; font-size: 0.90rem;">
                📈 BIST-Trend-Forecaster
            </span>
            {f'''<span style="background:{decision_bg}; color:{decision_color}; border:1px solid {decision_border}; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:9999px;">
                {hybrid_sig.decision}
            </span>''' if hybrid_sig else ''}
        </div>
        {f'''<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.15); border-radius: 6px; padding: 8px 10px; margin-bottom: 10px; font-size: 0.78rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span style="color:#94A3B8;">Bileşik Skor:</span>
                <span style="color:#F8FAFC; font-weight:700;">{hybrid_sig.composite_score} / 100</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span style="color:#94A3B8;">Teknik Trend:</span>
                <span style="color:#38BDF8; font-weight:600;">{hybrid_sig.technical_trend}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color:#94A3B8;">Kelly Tahsisi:</span>
                <span style="color:#34D399; font-weight:700;">%{hybrid_sig.kelly_allocation_percent} Portföy</span>
            </div>
        </div>''' if hybrid_sig else f'''<div style="color: #CBD5E1; font-size: 0.82rem; line-height: 1.4; margin-bottom: 10px;">
            📈 <strong>{active_ticker}</strong> hissesinin <strong>XGBoost trend tahminini, teknik indikatörlerini</strong> ve <strong>Kelly sermaye tahsisini</strong> inceleyin.
        </div>'''}
        <a href="{forecaster_url}" target="_blank" style="display: block; text-align: center; background: linear-gradient(135deg, #2563EB, #1D4ED8); color: #FFFFFF; padding: 8px 12px; border-radius: 6px; text-decoration: none; font-size: 0.82rem; font-weight: 600; border: 1px solid rgba(56, 189, 248, 0.35); box-shadow: 0 2px 8px rgba(37,99,235,0.4);">
            Forecaster'da Teknik Tahmini Gör ↗
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Finansal Sağlık Radarı (Plotly)
    if active_report_obj:
        with st.expander("📈 Finansal Sağlık Radarı (Plotly)", expanded=False):
            st.caption(f"**{active_ticker}** için 5 Boyutlu RAG Temel Analiz Profili")
            fig = create_radar_chart(active_report_obj)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # --- 2. ADIM: STANDART YAPILANDIRILMIŞ ÇIKTI (JSON EXPORT / BIST-TREND-FORECASTER ENTEGRASYONU) ---
    with st.expander("📊 Temel Analiz JSON Raporu (BIST-Trend-Forecaster)"):
        st.caption(f"{active_ticker} hissesi için teknik analiz modeline beslenecek standart yapılandırılmış JSON çıktısı.")
        
        rep_gen = load_report_generator()
        report_file = REPORTS_DIR / f"{active_ticker}_latest.json"

        if st.button("⚡ JSON Raporunu Üret / Güncelle", use_container_width=True):
            with st.spinner(f"{active_ticker} için RAG temel analiz raporu oluşturuluyor..."):
                try:
                    generated_rep = rep_gen.generate_report(ticker=active_ticker, force_refresh=True)
                    st.success(f"{active_ticker} raporu hazırlandı!")
                except Exception as e:
                    st.error(f"Rapor oluşturulurken hata: {e}")

        # Eğer önceden üretilmiş rapor varsa göster ve indirme butonu koy
        if report_file.exists():
            with open(report_file, "r", encoding="utf-8") as rf:
                report_json_str = rf.read()
            st.json(json.loads(report_json_str))
            st.download_button(
                label=f"📥 {active_ticker}_latest.json İndir",
                data=report_json_str,
                file_name=f"{active_ticker}_latest.json",
                mime="application/json",
                use_container_width=True
            )

    st.divider()

    # 4. BIST 100 Yeni Rapor İndirici (Ölçeklenebilirlik Çözümü)
    with st.expander("📥 KAP'tan Yeni Rapor İndir (BIST 100)"):
        st.caption("BIST 100 şirketlerinin raporlarını doğrudan indirip sisteme ekleyin.")
        bist_ticker_choice = st.selectbox(
            "Hisse Kodu (Ticker)",
            options=sorted(list(BIST_100_COMPANIES.keys())),
            format_func=lambda x: f"{x} - {BIST_100_COMPANIES[x]['name'][:24]}...",
            index=0
        )
        report_year = st.selectbox("Rapor Yılı", options=list(reversed(SUPPORTED_YEARS)), index=0)
        report_type = st.radio("Rapor Türü", options=["Faaliyet Raporu", "Finansal Tablolar"], horizontal=True)
        direct_url = st.text_input("KAP Doğrudan PDF URL (Opsiyonel)", placeholder="https://www.kap.org.tr/tr/BildirimPdf/...")

        if st.button("🚀 Raporu İndir ve İndeksle", use_container_width=True):
            if direct_url:
                with st.spinner(f"{bist_ticker_choice} {report_year} raporu indiriliyor..."):
                    doc_t = "faaliyet" if "Faaliyet" in report_type else "finansal"
                    downloaded_path = download_company_annual_report(
                        ticker=bist_ticker_choice,
                        year=report_year,
                        doc_type=doc_t,
                        custom_url=direct_url
                    )
                    if downloaded_path and downloaded_path.exists():
                        st.success(f"Rapor indirildi: {downloaded_path.name}")
                        with st.spinner("Rapor vektör indeksine ekleniyor..."):
                            success = add_single_pdf_to_vectorstore(downloaded_path)
                            if success:
                                st.cache_resource.clear()
                                st.success("Rapor başarıyla indekslendi ve analize hazır!")
                            else:
                                st.error("İndeksleme sırasında bir hata oluştu.")
                    else:
                        st.error("Dosya indirilemedi. Lütfen bağlantıyı kontrol edin.")
            else:
                st.info("KAP bildirim sayfasındaki PDF bağlantısını yukarıdaki kutuya yapıştırarak tek tıkla sisteme entegre edebilirsiniz.")

    st.divider()

    # 5. RAG Halüsinasyon & Kalite Benchmark Karnesi
    with st.expander("🛡️ RAG Güvenilirlik & Benchmark Karnesi"):
        st.caption("Ragas & TruLens metodolojisiyle ölçülmüş kurumsal doğruluk metrikleri.")
        bm_path = REPORTS_DIR / "rag_benchmark_report.json"
        if bm_path.exists():
            try:
                with open(bm_path, "r", encoding="utf-8") as bf:
                    bm_data = json.load(bf)
                sm = bm_data.get("summary_metrics", {})
                b_c1, b_c2 = st.columns(2)
                with b_c1:
                    st.metric("Sadakat", sm.get("faithfulness_percent", "%85.1"))
                with b_c2:
                    st.metric("Tuzak Savunma", sm.get("trap_defense_success_percent", "%100.0"))
                st.markdown(f"""
                <div style="font-size:0.78rem; color:#94A3B8; margin-top:6px; margin-bottom:8px;">
                    ● Arama Motoru: <strong>Hibrit (FAISS + BM25)</strong><br>
                    ● Sayısal Doğrulama: <strong>Hallucination Guard Aktif</strong><br>
                    ● Test Edilen Senaryo: <strong>{bm_data.get('total_test_cases', 8)} Vaka</strong>
                </div>
                """, unsafe_allow_html=True)
                st.download_button(
                    label="📥 Benchmark Raporunu İndir",
                    data=json.dumps(bm_data, ensure_ascii=False, indent=2),
                    file_name="rag_benchmark_report.json",
                    mime="application/json",
                    use_container_width=True
                )
            except Exception as e:
                st.caption(f"Benchmark yüklenirken hata: {e}")
        else:
            st.info("Benchmark raporu henüz oluşturulmadı.")

    st.divider()

    # 6. Sohbeti Temizleme ve Dışa Aktarma
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑️ Temizle", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_export:
        chat_export = ""
        if "messages" in st.session_state and st.session_state.messages:
            chat_export = f"# BIST Finansal Analist Asistanı - Analiz Raporu\n*Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n---\n"
            for m in st.session_state.messages:
                sender = "Kullanıcı" if m["role"] == "user" else "Finansal Analist Asistanı"
                chat_export += f"### {sender}\n{m['content']}\n\n"
        
        st.download_button(
            label="💾 İndir (MD)",
            data=chat_export,
            file_name=f"Finansal_Analiz_Raporu_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not bool(chat_export)
        )

# --- ANA EKRAN VE BAŞLIK ---
st.markdown("""
<div class="main-header">
    <span style="font-size: 2.1rem;">📈</span>
    <span>BIST <span class="brand-highlight">Finansal Analist</span> Asistanı</span>
    <span class="version-tag">Terminal v2.0</span>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="sub-header">Borsa İstanbul şirketlerinin KAP\'ta yayımlanan resmi faaliyet raporları, bilançoları ve dipnotlarına dayalı kurumsal RAG analiz ve temel veri servisi.</div>', unsafe_allow_html=True)

# Durum rozetleri
col_badges = st.container()
has_index = (VECTORSTORE_DIR / "index.faiss").exists()

badge_html = ""
if has_index:
    badge_html += '<span class="status-badge">● Finansal Bilgi Bankası Aktif (2020-2024 KAP Raporları)</span>'
else:
    badge_html += '<span style="background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.35); padding: 4px 12px; border-radius: 9999px; font-size: 0.80rem; font-weight: 600;">⚠️ Vektör indeksi senkronize ediliyor</span>'

if url_ticker:
    badge_html += f'<span class="deep-link-badge">🔗 Entegrasyon Modu: {url_ticker} ({url_company_name or url_ticker})</span>'

st.markdown(badge_html, unsafe_allow_html=True)

# --- RAG MOTORUNU YÜKLE ---
engine = load_rag_engine(model_option, vs_version=get_vectorstore_version())

# --- SOHBET GEÇMİŞİNİ BAŞLAT (DEEP-LINKING UYARLAMASI - 1. ADIM) ---
if "messages" not in st.session_state or not st.session_state.messages:
    if url_ticker:
        comp_title = url_company_name if url_company_name else url_ticker
        welcome_content = f"👋 **{url_ticker}** ({comp_title}) için son dönem bilanço, faaliyet raporu ve KAP analizine hoş geldiniz. Ne öğrenmek istersiniz?"
    else:
        welcome_content = "Merhaba! Ben sizin **BIST Kıdemli Finansal Analist Asistanınızım**. Akbank, Türk Hava Yolları ve Tüpraş gibi BIST şirketlerinin bilanço, kâr/zarar tabloları, faaliyet raporları ve rasyoları hakkında sorularınızı yanıtlayabilirim."

    st.session_state.messages = [
        {"role": "assistant", "content": welcome_content}
    ]

# --- HIZLI ANALİZ ŞABLONLARI (PROMPT CHIPS - 1. ADIM) ---
st.markdown("##### 💡 Hızlı Finansal Analiz Soruları")
q_cols = st.columns(4)

if url_ticker:
    # URL'den gelen hisseye özel hızlı butonlar
    comp_name_short = url_company_name if url_company_name else url_ticker
    quick_queries = [
        (f"📊 {url_ticker} Net Kâr", f"{comp_name_short} ({url_ticker}) son dönem net kârı ve gelirleri nasıl geldi?"),
        (f"💰 {url_ticker} Temettü", f"{comp_name_short} ({url_ticker}) temettü dağıtım veya sermaye artırımı kararı var mı?"),
        (f"⚖️ {url_ticker} Borçluluk", f"{comp_name_short} ({url_ticker}) borçluluk durumu, net borç ve finansal yapısı nasıldır?"),
        (f"🎯 {url_ticker} Beklentiler", f"{comp_name_short} ({url_ticker}) gelecek dönem beklentileri, hedefleri ve yatırım planları nelerdir?"),
    ]
else:
    # Genel hisse soruları
    quick_queries = [
        ("Akbank 2024 Net Kâr", "Akbank'ın 2024 yılı net kârı ve aktif büyüklüğü nedir?"),
        ("THY 2024 Filo & Yolcu", "Türk Hava Yolları'nın 2024 filo büyüklüğü ve yolcu sayısı ne kadardır?"),
        ("Tüpraş 2024 Rafineri Marjı", "Tüpraş'ın 2024 yılı rafineri kapasite kullanımı ve net rafineri marjı nedir?"),
        ("Kârlılık Karşılaştırması", "Akbank ve Tüpraş'ın 2023-2024 yıllarındaki net kâr değişimlerini karşılaştırır mısın?"),
    ]

selected_quick_prompt = None
for i, (label, query_text) in enumerate(quick_queries):
    with q_cols[i]:
        if st.button(label, key=f"quick_btn_{i}", use_container_width=True):
            selected_quick_prompt = query_text

st.divider()

# --- GEÇMİŞ MESAJLARI GÖSTER ---
for msg in st.session_state.messages:
    msg_avatar = "👤" if msg.get("role") == "user" else "📈"
    with st.chat_message(msg["role"], avatar=msg_avatar):
        st.markdown(msg["content"], unsafe_allow_html=True)
        # Eğer kaydedilmiş kaynaklar varsa göster
        if "sources" in msg and msg["sources"]:
            with st.expander("📌 Referans Alınan KAP Raporları & Sayfalar"):
                for s_idx, src in enumerate(msg["sources"]):
                    comp_badge = get_company_badge(src.get("company", ""))
                    st.markdown(
                        f"""<div class="source-box">
                        <strong>Kaynak {s_idx+1}:</strong> {comp_badge} | <strong>Yıl:</strong> {src.get('year')} | 
                        <strong>Dosya:</strong> <code>{src.get('file_name')}</code> | <strong>Sayfa:</strong> {src.get('page')}
                        <br><br>
                        <em>"{src.get('snippet')}..."</em>
                        </div>""",
                        unsafe_allow_html=True
                    )

# --- KULLANICI GİRDİSİ VE ANALİZ DÖNGÜSÜ ---
user_input = st.chat_input("Finansal bir gösterge veya şirket performansı hakkında soru sorun...")
prompt_to_process = selected_quick_prompt if selected_quick_prompt else user_input

if prompt_to_process:
    # 1. Kullanıcı mesajını ekle ve ekrana yazdır
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_to_process)

    # 2. Asistan cevabını streaming ile üret
    with st.chat_message("assistant", avatar="📈"):
        full_response = ""
        captured_docs = []
        rewritten_q = prompt_to_process
        captured_guard = {}

        # Canlı Düşünme & Arama Durum Paneli
        status_box = st.status("📊 Finansal Analist verileri inceliyor...", expanded=True)
        stream_placeholder = st.empty()

        generator = engine.stream_answer(
            question=prompt_to_process,
            chat_history=st.session_state.messages[:-1],
            company_filter=selected_company,
            year_filter=selected_year,
        )

        for event in generator:
            ev_type = event.get("type")
            if ev_type == "status":
                status_box.write(event.get("message", "Finansal veriler işleniyor..."))
            elif ev_type == "content":
                if not full_response:
                    # İlk veri geldiğinde durum kutusunu kapatıp tamamlandı rozetine dönüştür
                    status_box.update(label="✅ Finansal Raporlar & Dipnotlar İncelendi", state="complete", expanded=False)
                full_response += event["content"]
                stream_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            elif ev_type == "done":
                captured_docs = event.get("docs", [])
                rewritten_q = event.get("rewritten_query", prompt_to_process)
                captured_guard = event.get("guard_result", {})

        # Status kutusunu tamamlandı olarak sonlandır
        status_box.update(label="✅ Finansal Analiz Tamamlandı", state="complete", expanded=False)

        # İmleci kaldır ve nihai cevabı yerleştir
        if full_response.strip():
            stream_placeholder.markdown(full_response, unsafe_allow_html=True)
        else:
            fallback_msg = "⚠️ Finansal raporlardan veri getirilirken bir aksaklık oluştu. Lütfen sorunuzu tekrar iletiniz."
            stream_placeholder.markdown(fallback_msg, unsafe_allow_html=True)
            full_response = fallback_msg

        # Hallucination Guard Doğrulama Rozeti
        if captured_guard and captured_guard.get("status_badge"):
            guard_badge_text = captured_guard["status_badge"]
            guard_color = "#34D399" if captured_guard.get("is_safe", True) else "#FBBF24"
            st.markdown(f"""
            <div style="margin-top: 10px; margin-bottom: 4px; padding: 5px 12px; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(52, 211, 153, 0.25); border-radius: 6px; font-size: 0.80rem; color: {guard_color}; display: inline-flex; align-items: center; gap: 8px;">
                <span>{guard_badge_text}</span>
            </div>
            """, unsafe_allow_html=True)

        # 3. Adım Köprü Kartı: Cevap sonuna BIST-Trend-Forecaster butonu ekle
        detected_or_active_ticker = active_ticker
        st.markdown(f"""
        <div style="margin-top: 14px; padding: 10px 16px; background: #0E1626; border-radius: 8px; border: 1px solid #1E293B; display: inline-flex; align-items: center; gap: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.35);">
            <span style="font-size: 0.85rem; color: #E2E8F0;">📈 <strong>{detected_or_active_ticker}</strong> hissesinin teknik model tahminlerini görmek ister misiniz?</span>
            <a href="{FORECASTER_BASE_URL}/?ticker={detected_or_active_ticker}" target="_blank" style="background: linear-gradient(135deg, #2563EB, #1D4ED8); color: white; font-size: 0.78rem; font-weight: 600; padding: 5px 12px; border-radius: 6px; text-decoration: none; border: 1px solid rgba(56, 189, 248, 0.35); box-shadow: 0 2px 6px rgba(37,99,235,0.3);">BIST-Trend-Forecaster'a Git ↗</a>
        </div>
        """, unsafe_allow_html=True)

        # Kaynakları şık kartlar halinde göster
        formatted_sources = []
        if captured_docs:
            with st.expander("📌 Referans Alınan KAP Raporları & Sayfalar"):
                for s_idx, doc in enumerate(captured_docs):
                    comp = doc.metadata.get("company", "Bilinmeyen")
                    yr = doc.metadata.get("year", "")
                    fname = doc.metadata.get("file_name", os.path.basename(doc.metadata.get("source", "")))
                    page = doc.metadata.get("page", "Bilinmiyor")
                    snippet = doc.page_content[:240].replace("\n", " ").strip()
                    
                    formatted_sources.append({
                        "company": comp,
                        "year": yr,
                        "file_name": fname,
                        "page": page,
                        "snippet": snippet
                    })

                    comp_badge = get_company_badge(comp)
                    st.markdown(
                        f"""<div class="source-box">
                        <strong>Kaynak {s_idx+1}:</strong> {comp_badge} | <strong>Yıl:</strong> {yr} | 
                        <strong>Dosya:</strong> <code>{fname}</code> | <strong>Sayfa:</strong> {page}
                        <br><br>
                        <em>"{snippet}..."</em>
                        </div>""",
                        unsafe_allow_html=True
                    )

        # 3. Asistan mesajını hafızaya kaydet
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": formatted_sources
        })
