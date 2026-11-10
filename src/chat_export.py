"""Sohbet Seansını Rapor Olarak İndirme Modülü (Chat Export).

Kullanıcının finansal analist ile yaptığı derinlemesine sohbet oturumunu,
şirket analizlerini, KAP referanslarını ve soru-cevap geçmişini
profesyonel Markdown (.md) ve yazdırılabilir/PDF'e dönüştürülebilir HTML (.html)
formatlarına dönüştürür.
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from typing import List, Dict, Any


def _format_markdown_text(text: str) -> str:
    """Metin içindeki gereksiz boşlukları ve biçimlendirmeyi temizler."""
    return text.strip()


def export_chat_history_markdown(
    messages: List[Dict[str, Any]],
    current_ticker: str = "BIST"
) -> str:
    """Sohbet geçmişini profesyonel bir finansal araştırma notu (Markdown) formatına çevirir."""
    now_str = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    clean_ticker = current_ticker.upper().replace(".IS", "").strip() or "BIST"

    total_turns = sum(1 for m in messages if m.get("role") == "user")

    lines = [
        f"# 📊 BIST Finansal Araştırma & Sohbet Raporu",
        f"**Oluşturulma Tarihi:** {now_str}  ",
        f"**Odak Şirket / Kapsam:** {clean_ticker}  ",
        f"**Soru & Yanıt Seansı:** {total_turns} Adet İnceleme  ",
        f"**Üretim Motoru:** BIST Finansal Analist Chatbot (Gemini 3.5 Flash-Lite / Hibrit RAG)",
        "",
        "---",
        "",
        "## 📑 Sohbet & Analiz Akışı",
        ""
    ]

    turn_idx = 1
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()

        if role == "user":
            lines.append(f"### 💬 Soru {turn_idx}: Yatırımcı / Kullanıcı")
            lines.append(f"> **{content}**")
            lines.append("")
        elif role == "assistant":
            lines.append(f"### 🤖 BIST Finansal Analist Değerlendirmesi")
            lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")
            turn_idx += 1

    lines.extend([
        "## ⚖️ Yasal Uyarı & Metodoloji",
        "Bu rapor **BIST Finansal Analist ChatBot** tarafından, Kamuyu Aydınlatma Platformu (KAP) verileri, "
        "finansal dipnotlar ve yfinance piyasa verileri kullanılarak otomatik araştırma amacıyla derlenmiştir. "
        "Burada yer alan analiz, yorum ve değerlendirmeler **kesinlikle yatırım danışmanlığı kapsamında değildir**. "
        "Yatırım kararları öncesinde yetkili yatırım uzmanlarına danışılması tavsiye olunur.",
        "",
        f"*Rapor Sonu — BIST Finansal Analist Sistem Kaydı ({now_str})*"
    ])

    return "\n".join(lines)


def export_chat_history_html(
    messages: List[Dict[str, Any]],
    current_ticker: str = "BIST"
) -> str:
    """Sohbet geçmişini profesyonel kurumsal araştırma notu formatında (HTML/CSS) sunar."""
    now_str = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    clean_ticker = current_ticker.upper().replace(".IS", "").strip() or "BIST 100"
    total_turns = sum(1 for m in messages if m.get("role") == "user")

    # Basit markdown -> HTML çevirici (Kalın, satır sonları, maddeler)
    def simple_md_to_html(text: str) -> str:
        escaped = html.escape(text)
        # Bold: **metin**
        escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
        # Lists: * veya - ile başlayan satırlar
        paragraphs = []
        for line in escaped.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("- ") or line.startswith("* "):
                paragraphs.append(f"<li>{line[2:]}</li>")
            elif line.startswith("#"):
                clean_h = line.lstrip("#").strip()
                paragraphs.append(f"<h4 style='color: #38BDF8; margin-top: 14px; margin-bottom: 6px;'>{clean_h}</h4>")
            else:
                paragraphs.append(f"<p style='margin: 6px 0; line-height: 1.6;'>{line}</p>")
        return "".join(paragraphs)

    html_blocks = []
    turn_idx = 1
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue

        if role == "user":
            html_blocks.append(f"""
            <div class="message-card user-card">
                <div class="card-header">
                    <span class="user-badge">👤 Soru #{turn_idx}</span>
                    <span class="timestamp">Yatırımcı / Kullanıcı</span>
                </div>
                <div class="card-body">
                    <strong>{html.escape(content)}</strong>
                </div>
            </div>
            """)
        elif role == "assistant":
            body_html = simple_md_to_html(content)
            html_blocks.append(f"""
            <div class="message-card assistant-card">
                <div class="card-header">
                    <span class="analyst-badge">🤖 BIST Finansal Analist</span>
                    <span class="timestamp">KAP & Finansal RAG Motoru</span>
                </div>
                <div class="card-body">
                    {body_html}
                </div>
            </div>
            """)
            turn_idx += 1

    messages_joined = "\n".join(html_blocks)

    html_template = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>BIST Finansal Analiz Raporu - {clean_ticker}</title>
    <style>
        :root {{
            --bg: #080C14;
            --surface: #0D1422;
            --card-border: #1E293B;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --accent: #38BDF8;
            --accent-green: #10B981;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-primary);
            margin: 0;
            padding: 32px 20px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 900px;
            width: 100%;
        }}
        .header {{
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }}
        .badge-bar {{
            display: flex;
            gap: 10px;
            margin-bottom: 12px;
        }}
        .badge {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 22px;
            color: #FFFFFF;
        }}
        .meta-info {{
            font-size: 13px;
            color: var(--text-secondary);
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .message-card {{
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 18px;
            border: 1px solid var(--card-border);
        }}
        .user-card {{
            background-color: #111827;
            border-left: 4px solid var(--accent);
        }}
        .assistant-card {{
            background-color: #0D1422;
            border-left: 4px solid var(--accent-green);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-size: 12px;
        }}
        .user-badge {{
            color: var(--accent);
            font-weight: 600;
        }}
        .analyst-badge {{
            color: var(--accent-green);
            font-weight: 600;
        }}
        .timestamp {{
            color: var(--text-secondary);
            font-size: 11px;
        }}
        .card-body {{
            font-size: 14px;
            line-height: 1.65;
            color: #E2E8F0;
        }}
        .card-body strong {{
            color: #FFFFFF;
        }}
        .disclaimer {{
            background: rgba(15, 23, 42, 0.7);
            border: 1px dashed var(--card-border);
            border-radius: 8px;
            padding: 16px;
            margin-top: 32px;
            font-size: 11.5px;
            color: var(--text-secondary);
            line-height: 1.5;
        }}
        @media print {{
            body {{
                background-color: #FFFFFF;
                color: #000000;
            }}
            .header, .message-card, .disclaimer {{
                background-color: #FFFFFF !important;
                color: #000000 !important;
                border: 1px solid #CCCCCC !important;
                box-shadow: none !important;
            }}
            .card-body strong, h1, .badge {{
                color: #000000 !important;
            }}
            .meta-info, .timestamp {{
                color: #555555 !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="badge-bar">
                <span class="badge">BIST RESEARCH MEMO</span>
                <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #10B981;">{clean_ticker}</span>
            </div>
            <h1>BIST Finansal Araştırma & Sohbet Raporu</h1>
            <div class="meta-info">
                <span>📅 <strong>Tarih:</strong> {now_str}</span>
                <span>🏢 <strong>Kapsam:</strong> {clean_ticker}</span>
                <span>💬 <strong>İnceleme:</strong> {total_turns} Soru-Cevap</span>
            </div>
        </div>

        {messages_joined}

        <div class="disclaimer">
            <strong>Yasal Uyarı:</strong> Bu araştırma raporu BIST Finansal Analist Chatbot sistemi tarafından Kamuyu Aydınlatma Platformu (KAP) duyuruları ve finansal tablolar baz alınarak otomatik olarak oluşturulmuştur. Burada yer alan hiçbir bilgi ve değerlendirme yatırım danışmanlığı kapsamında değildir.
        </div>
    </div>
</body>
</html>
"""
    return html_template
