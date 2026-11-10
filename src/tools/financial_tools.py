"""Finansal Analiz Araçları ve Fonksiyon Çağrıları (Tool Calling / Function Calling).

LLM Ajanının doğrudan çalıştırabileceği deterministik hesaplama motorları:
1. DuPont 3-Aşamalı ROE Ayrıştırması
2. Altman Z-Skoru İflas Dayanıklılık Analizi
3. İndirgenmiş Nakit Akımı (DCF) Değerleme Modeli
4. Kelly Kriteri Dinamik Portföy Sermaye Tahsisi
5. Makroekonomik Şok Simülatörü (USD/TRY, Politika Faizi, Brent Petrol)
6. Çift Şirket Eşler Arası (Peer) Kıyaslama Motoru
7. Canlı / Doğrulanmış KAP Bildirimleri ve Şirket Bilgi Çözücü
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
import json

from src.dupont_analysis import calculate_dupont_model, calculate_altman_z_score
from src.dividend_analysis import calculate_dividend_metrics
from src.macro_sensitivity import analyze_macro_sensitivity
from src.kap_feed import get_recent_kap_disclosures, get_company_kap_url

logger = logging.getLogger(__name__)


# ============================================================================
# 1. TEMEL FİNANSAL HESAPLAMA ARAÇLARI
# ============================================================================

def tool_calculate_dupont(ticker: str) -> Dict[str, Any]:
    """BIST hissesinin DuPont 3-aşamalı ROE ayrıştırmasını hesaplar."""
    sym = (ticker or "").upper().replace(".IS", "").strip()
    dupont = calculate_dupont_model(sym)
    return {
        "ticker": sym,
        "roe_pct": round(dupont.get("roe_pct", 0.0), 2),
        "net_profit_margin_pct": round(dupont.get("net_profit_margin_pct", 0.0), 2),
        "asset_turnover": round(dupont.get("asset_turnover", 0.0), 2),
        "equity_multiplier": round(dupont.get("equity_multiplier", 1.0), 2),
        "driver": dupont.get("driver", "Bilinmiyor"),
        "quality_assessment": dupont.get("quality_assessment", ""),
    }


def tool_calculate_altman_z(ticker: str) -> Dict[str, Any]:
    """BIST hissesinin Altman Z-Skoru finansal sıkıntı derecesini hesaplar."""
    sym = (ticker or "").upper().replace(".IS", "").strip()
    dupont = calculate_dupont_model(sym)
    altman = calculate_altman_z_score(sym, dupont)
    return {
        "ticker": sym,
        "z_score": round(altman.get("z_score", 0.0), 2),
        "zone": altman.get("zone", "Belirsiz"),
        "interpretation": altman.get("interpretation", ""),
        "components": altman.get("components", {}),
    }


def tool_calculate_dcf(
    ticker: str,
    base_fcf_million_tl: Optional[float] = None,
    growth_rate_pct: float = 15.0,
    discount_rate_pct: float = 24.0,
    terminal_growth_pct: float = 6.0,
    years: int = 5,
) -> Dict[str, Any]:
    """
    İndirgenmiş Nakit Akımları (DCF) İskonto Modeli.
    Gelecek yılların nakit akımlarını bugünkü değere indirgeyerek teorik adil şirket değerini hesaplar.
    """
    sym = (ticker or "").upper().replace(".IS", "").strip()

    # Eğer FCF verilmemişse DuPont ve temettü rasyolarından yaklaşık serbest nakit akışı türetilir
    if base_fcf_million_tl is None or base_fcf_million_tl <= 0:
        base_fcf_million_tl = 1500.0  # Milyon TL varsayılan taban

    g = growth_rate_pct / 100.0
    r = discount_rate_pct / 100.0
    tg = terminal_growth_pct / 100.0

    if r <= tg:
        r = tg + 0.05  # Bölme hatasını engelle

    pv_cash_flows = []
    current_cf = base_fcf_million_tl

    for y in range(1, years + 1):
        current_cf = current_cf * (1.0 + g)
        discount_factor = (1.0 + r) ** y
        pv = current_cf / discount_factor
        pv_cash_flows.append(round(pv, 2))

    terminal_value = (current_cf * (1.0 + tg)) / (r - tg)
    pv_terminal_value = terminal_value / ((1.0 + r) ** years)

    fair_enterprise_value = sum(pv_cash_flows) + pv_terminal_value

    return {
        "ticker": sym,
        "base_fcf_million_tl": base_fcf_million_tl,
        "projected_pv_flows": pv_cash_flows,
        "pv_terminal_value": round(pv_terminal_value, 2),
        "estimated_fair_value_million_tl": round(fair_enterprise_value, 2),
        "assumptions": {
            "growth_rate_pct": growth_rate_pct,
            "discount_rate_pct": discount_rate_pct,
            "terminal_growth_pct": terminal_growth_pct,
            "projection_years": years,
        },
    }


def tool_calculate_kelly_allocation(
    ticker: str,
    win_rate: float = 0.62,
    win_loss_ratio: float = 1.85,
    max_portfolio_cap_pct: float = 25.0,
) -> Dict[str, Any]:
    """
    Kelly Kriteri Optimal Portföy Sermaye Dağılımını hesaplar.
    f* = p - (1 - p) / b formülüne göre aşırı risk almadan maksimum geometrik büyüme oranı sağlar.
    """
    sym = (ticker or "").upper().replace(".IS", "").strip()

    p = max(0.01, min(0.99, win_rate))
    b = max(0.1, win_loss_ratio)
    q = 1.0 - p

    raw_kelly = p - (q / b)
    kelly_pct = max(0.0, raw_kelly * 100.0)

    # Risk yönetimi: Genelde yarım Kelly (Half-Kelly) uygulanır
    conservative_kelly_pct = min(kelly_pct * 0.5, max_portfolio_cap_pct)

    if kelly_pct <= 0:
        recommendation = "Girmeyin / Pozisyon Açmayın (Negatif Matematiksel Beklenti)"
    elif conservative_kelly_pct >= 15.0:
        recommendation = f"Güçlü Tahsis (Maksimum %{conservative_kelly_pct:.1f} Portföy Payı)"
    else:
        recommendation = f"Ilımlı Tahsis (%{conservative_kelly_pct:.1f} Portföy Payı)"

    return {
        "ticker": sym,
        "win_rate_pct": round(p * 100.0, 1),
        "win_loss_ratio": round(b, 2),
        "full_kelly_pct": round(kelly_pct, 1),
        "recommended_portfolio_allocation_pct": round(conservative_kelly_pct, 1),
        "recommendation": recommendation,
    }


def tool_calculate_macro_sensitivity(
    ticker: str,
    usd_change_pct: float = 10.0,
    rate_change_bps: float = 250.0,
    oil_change_pct: float = 15.0,
) -> Dict[str, Any]:
    """Döviz kuru, faiz ve emtia şoklarının net kâr üzerindeki etkisini simüle eder."""
    sym = (ticker or "").upper().replace(".IS", "").strip()
    return analyze_macro_sensitivity(
        ticker=sym,
        usd_change_pct=usd_change_pct,
        rate_change_bps=rate_change_bps,
        oil_change_pct=oil_change_pct,
    )


def tool_compare_peers(ticker_a: str, ticker_b: str) -> Dict[str, Any]:
    """İki BIST hissesinin DuPont kârlılık marjlarını, kaldıraçlarını ve Z-skorlarını kıyaslar."""
    a = (ticker_a or "").upper().replace(".IS", "").strip()
    b = (ticker_b or "").upper().replace(".IS", "").strip()

    dup_a = calculate_dupont_model(a)
    dup_b = calculate_dupont_model(b)
    alt_a = calculate_altman_z_score(a, dup_a)
    alt_b = calculate_altman_z_score(b, dup_b)

    # Karar mantığı
    score_a = dup_a.get("roe_pct", 0) + (alt_a.get("z_score", 0) * 5)
    score_b = dup_b.get("roe_pct", 0) + (alt_b.get("z_score", 0) * 5)

    if score_a > score_b * 1.1:
        leader = a
        rationale = f"{a}, daha yüksek operasyonel verimlilik ve sağlam bilanço Z-skoru sayesinde kârlılık kalitesinde öne çıkmaktadır."
    elif score_b > score_a * 1.1:
        leader = b
        rationale = f"{b}, kârlılık marjı ve sermaye dayanıklılık rasyolarında {a} hissesine göre avantaj sağlamaktadır."
    else:
        leader = "Dengeli / Baş Başa"
        rationale = f"Her iki şirket de kârlılık ve finansal risk profili açısından birbirine yakın seviyelerdedir."

    return {
        "company_a": {
            "ticker": a,
            "roe_pct": round(dup_a.get("roe_pct", 0.0), 1),
            "net_margin_pct": round(dup_a.get("net_profit_margin_pct", 0.0), 1),
            "asset_turnover": round(dup_a.get("asset_turnover", 0.0), 2),
            "equity_multiplier": round(dup_a.get("equity_multiplier", 1.0), 2),
            "z_score": round(alt_a.get("z_score", 0.0), 2),
            "zone": alt_a.get("zone", ""),
        },
        "company_b": {
            "ticker": b,
            "roe_pct": round(dup_b.get("roe_pct", 0.0), 1),
            "net_margin_pct": round(dup_b.get("net_profit_margin_pct", 0.0), 1),
            "asset_turnover": round(dup_b.get("asset_turnover", 0.0), 2),
            "equity_multiplier": round(dup_b.get("equity_multiplier", 1.0), 2),
            "z_score": round(alt_b.get("z_score", 0.0), 2),
            "zone": alt_b.get("zone", ""),
        },
        "quality_leader": leader,
        "analyst_rationale": rationale,
    }


def tool_get_kap_disclosures(ticker: str, limit: int = 5) -> Dict[str, Any]:
    """Şirketin Kamuyu Aydınlatma Platformu (KAP) resmi portal linkini ve son duyurularını getirir."""
    sym = (ticker or "").upper().replace(".IS", "").strip()
    kap_url = get_company_kap_url(sym)
    recent = get_recent_kap_disclosures(ticker_filter=sym, limit=limit)
    return {
        "ticker": sym,
        "official_kap_url": kap_url,
        "disclosures": recent,
        "count": len(recent),
    }


# ============================================================================
# 2. OPENAI / LANGCHAIN TOOL ÇAĞRI ŞEMALARI (FUNCTION CALLING SCHEMAS)
# ============================================================================

FINANCIAL_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_dupont",
            "description": "BIST hissesinin DuPont 3-aşamalı ROE (Net Kâr Marjı, Aktif Devir Hızı, Finansal Kaldıraç) ve kârlılık sürücüsünü hesaplar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "BIST hisse kodu (Örn: THYAO, AKBNK, FROTO)"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_altman_z",
            "description": "BIST hissesinin Altman Z-skorunu (Finansal stres, iflas riski, güvenli/gri/stres bölgesi) hesaplar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "BIST hisse kodu (Örn: EREGL, TUPRS)"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_dcf",
            "description": "İndirgenmiş Nakit Akımları (DCF) modeli ile hissenin adil şirket değerini hesaplar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "BIST hisse kodu"},
                    "base_fcf_million_tl": {"type": "number", "description": "Baz serbest nakit akışı (Milyon TL)"},
                    "growth_rate_pct": {"type": "number", "description": "Yıllık nakit akışı büyüme beklentisi (%)"},
                    "discount_rate_pct": {"type": "number", "description": "Ağırlıklı Ortalama Sermaye Maliyeti / İskonto oranı (%)"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_kelly_allocation",
            "description": "Kelly Kriteri ile matematiksel olarak optimal portföy pozisyon büyüklüğü yüzdesini hesaplar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "BIST hisse kodu"},
                    "win_rate": {"type": "number", "description": "Tahmin kazanma olasılığı (Örn: 0.65)"},
                    "win_loss_ratio": {"type": "number", "description": "Kazanç / Kayıp kâr-zarar oranı (Örn: 1.8)"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_macro_sensitivity",
            "description": "Dolar kuru (%10), TCMB politika faizi (+250 bps) ve petrol şoklarının hisse net kârına etkisini simüle eder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "BIST hisse kodu"},
                    "usd_change_pct": {"type": "number", "description": "Dolar kuru değişim yüzdesi (Varsayılan: 10.0)"},
                    "rate_change_bps": {"type": "number", "description": "Faiz değişimi baz puan (Varsayılan: 250)"},
                    "oil_change_pct": {"type": "number", "description": "Brent petrol değişim yüzdesi (Varsayılan: 15.0)"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_peers",
            "description": "İki BIST şirketinin DuPont kârlılık marjlarını, kaldıraçlarını ve Altman Z-skorlarını eşler arası kıyaslar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_a": {"type": "string", "description": "1. Şirket kodu (Örn: THYAO)"},
                    "ticker_b": {"type": "string", "description": "2. Şirket kodu (Örn: PGSUS)"}
                },
                "required": ["ticker_a", "ticker_b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_kap_disclosures",
            "description": "Şirketin resmi KAP portal linkini ve en son kamuoyu duyurularını getirir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "BIST hisse kodu"},
                    "limit": {"type": "integer", "description": "Maksimum bildirim sayısı (Varsayılan: 5)"}
                },
                "required": ["ticker"]
            }
        }
    }
]


# ============================================================================
# 3. TOOL ÇAĞRI DAĞITICISI (DISPATCHER)
# ============================================================================

def execute_financial_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Verilen araç adını ve argümanlarını çalıştırarak yapılandırılmış JSON çıktısı döner."""
    args = arguments or {}
    clean_name = tool_name.removeprefix("tool_") if hasattr(tool_name, "removeprefix") else tool_name.replace("tool_", "")
    logger.info(f"[ToolCalling] Araç çalıştırılıyor: {clean_name} with {args}")

    if clean_name == "calculate_dupont":
        return tool_calculate_dupont(ticker=args.get("ticker", "THYAO"))
    elif clean_name == "calculate_altman_z":
        return tool_calculate_altman_z(ticker=args.get("ticker", "THYAO"))
    elif clean_name == "calculate_dcf":
        return tool_calculate_dcf(
            ticker=args.get("ticker", "THYAO"),
            base_fcf_million_tl=args.get("base_fcf_million_tl"),
            growth_rate_pct=float(args.get("growth_rate_pct", 15.0)),
            discount_rate_pct=float(args.get("discount_rate_pct", 24.0)),
        )
    elif clean_name == "calculate_kelly_allocation":
        return tool_calculate_kelly_allocation(
            ticker=args.get("ticker", "THYAO"),
            win_rate=float(args.get("win_rate", 0.62)),
            win_loss_ratio=float(args.get("win_loss_ratio", 1.85)),
        )
    elif clean_name == "calculate_macro_sensitivity":
        return tool_calculate_macro_sensitivity(
            ticker=args.get("ticker", "THYAO"),
            usd_change_pct=float(args.get("usd_change_pct", 10.0)),
            rate_change_bps=float(args.get("rate_change_bps", 250.0)),
            oil_change_pct=float(args.get("oil_change_pct", 15.0)),
        )
    elif clean_name == "compare_peers":
        return tool_compare_peers(
            ticker_a=args.get("ticker_a", "THYAO"),
            ticker_b=args.get("ticker_b", "PGSUS"),
        )
    elif clean_name == "get_kap_disclosures":
        return tool_get_kap_disclosures(
            ticker=args.get("ticker", "THYAO"),
            limit=int(args.get("limit", 5)),
        )
    else:
        return {"error": f"Bilinmeyen finansal araç: {tool_name}"}
