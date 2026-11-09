import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from src.config import REPORTS_DIR
from src.rag.engine import FinancialRAGEngine
from src.evaluation.dataset import GOLDEN_FINANCIAL_DATASET


class RAGBenchmark:
    """
    Finansal RAG Kalite ve Halüsinasyon Benchmark Motoru (Ragas & TruLens Metodolojisi).
    - Sadakat Skoru (Faithfulness / Groundedness)
    - Halüsinasyon Oranı (Hallucination Rate)
    - Cevap Alakalılığı (Answer Relevance)
    - Halüsinasyon Tuzağı Direnci (Trap Resistance)
    - Uçtan Uca Gecikme (Latency)
    """

    def __init__(self, engine: FinancialRAGEngine = None):
        self.engine = engine or FinancialRAGEngine()
        self.output_path = REPORTS_DIR / "rag_benchmark_report.json"

    def evaluate_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Tek bir test senaryosunu RAG motorunda çalıştırıp puanlar."""
        q = test_case["question"]
        ticker = test_case["ticker"]
        is_trap = test_case["is_hallucination_trap"]

        t0 = time.time()
        res = self.engine.answer_query(q)
        latency = round(time.time() - t0, 2)

        answer = res.get("answer", "")
        docs = res.get("docs", [])
        guard_result = res.get("guard_result", {})

        groundedness = guard_result.get("groundedness_score", 1.0)
        verified_claims = guard_result.get("verified_claims", [])
        unverified_claims = guard_result.get("unverified_claims", [])

        # 1. Alakalılık (Relevance) Kontrolü
        expected_topics = test_case.get("expected_topics", [])
        answer_lower = answer.lower()
        topic_matches = sum(1 for top in expected_topics if top.lower() in answer_lower)
        relevance_score = round(topic_matches / len(expected_topics), 2) if expected_topics else 1.0

        # 2. Halüsinasyon Tuzağı Direnci (Trap Resistance)
        trap_success = True
        if is_trap:
            # Tuzak sorularda model net olarak "bilgi bulunamadı/yer almıyor" demelidir
            refusal_keywords = ["bulunamadı", "yer almamaktadır", "bilgi yok", "mevcut değil", "ulaşılamamıştır"]
            trap_success = any(kw in answer_lower for kw in refusal_keywords) and len(unverified_claims) == 0

        # 3. Bağlam İsabeti (Context Precision)
        context_has_target = any(ticker.lower() in str(d.metadata).lower() for d in docs) if docs else False

        return {
            "id": test_case["id"],
            "ticker": ticker,
            "category": test_case["category"],
            "question": q,
            "answer_snippet": answer[:180] + ("..." if len(answer) > 180 else ""),
            "latency_seconds": latency,
            "groundedness_score": groundedness,
            "verified_claims_count": len(verified_claims),
            "unverified_claims_count": len(unverified_claims),
            "unverified_claims": unverified_claims,
            "relevance_score": relevance_score,
            "is_trap": is_trap,
            "trap_defended": trap_success if is_trap else None,
            "context_precision": 1.0 if context_has_target or is_trap else 0.5,
            "docs_count": len(docs),
        }

    def run_benchmark(self, dataset: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Tüm benchmark veri setini çalıştırır ve kurumsal kalite raporu üretir."""
        test_dataset = dataset or GOLDEN_FINANCIAL_DATASET
        print(f"\n=======================================================")
        print(f"🚀 BIST Finansal RAG Benchmark Başlatılıyor ({len(test_dataset)} Test Senaryosu)")
        print(f"=======================================================\n")

        results = []
        for idx, tc in enumerate(test_dataset):
            print(f"[{idx+1}/{len(test_dataset)}] Test Koşuluyor: {tc['id']} ({tc['ticker']}) - {tc['category']}...")
            eval_res = self.evaluate_test_case(tc)
            results.append(eval_res)
            print(f"       ↳ Sadakat: %{eval_res['groundedness_score']*100:.0f} | Gecikme: {eval_res['latency_seconds']}s")

        # Toplu Metrik Hesaplamaları
        total_tests = len(results)
        avg_groundedness = round(sum(r["groundedness_score"] for r in results) / total_tests, 3)
        avg_relevance = round(sum(r["relevance_score"] for r in results) / total_tests, 3)
        avg_latency = round(sum(r["latency_seconds"] for r in results) / total_tests, 2)

        # Halüsinasyon Oranı (doğrulanamayan iddiaların toplam iddialara oranı)
        total_claims = sum(r["verified_claims_count"] + r["unverified_claims_count"] for r in results)
        total_unverified = sum(r["unverified_claims_count"] for r in results)
        hallucination_rate = round(total_unverified / total_claims, 3) if total_claims > 0 else 0.0

        # Tuzak Savunma Başarısı
        trap_tests = [r for r in results if r["is_trap"]]
        trap_success_count = sum(1 for r in trap_tests if r["trap_defended"])
        trap_defense_rate = round(trap_success_count / len(trap_tests), 2) if trap_tests else 1.0

        report = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_version": "1.0.0",
            "model_tested": self.engine.model_name,
            "total_test_cases": total_tests,
            "summary_metrics": {
                "faithfulness_score": avg_groundedness,
                "faithfulness_percent": f"%{avg_groundedness * 100:.1f}",
                "hallucination_rate": hallucination_rate,
                "hallucination_percent": f"%{hallucination_rate * 100:.1f}",
                "answer_relevance_percent": f"%{avg_relevance * 100:.1f}",
                "trap_defense_success_percent": f"%{trap_defense_rate * 100:.1f}",
                "average_latency_seconds": avg_latency,
            },
            "detailed_results": results
        }

        # Diske Kaydet
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n=======================================================")
        print(f"📊 BENCHMARK SONUÇLARI (HALÜSİNASYON & GÜVENİLİRLİK KARNESİ)")
        print(f"=======================================================")
        print(f"● Sadakat / Doğruluk Oranı (Faithfulness): {report['summary_metrics']['faithfulness_percent']}")
        print(f"● Halüsinasyon Oranı (Hallucination Rate): {report['summary_metrics']['hallucination_percent']}")
        print(f"● Cevap Alakalılığı (Answer Relevance)  : {report['summary_metrics']['answer_relevance_percent']}")
        print(f"● Tuzak Soru Savunması (Trap Defense)   : {report['summary_metrics']['trap_defense_success_percent']}")
        print(f"● Ortalama Yanıt Gecikmesi (Latency)   : {avg_latency}s")
        print(f"Rapor Kaydedildi: {self.output_path}\n")

        return report


if __name__ == "__main__":
    bm = RAGBenchmark()
    bm.run_benchmark()
