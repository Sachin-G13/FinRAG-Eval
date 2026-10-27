import os
import sys
import json
import time
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

import run_eval_suite
from run_eval_suite import (
    FACTUAL_QUERIES,
    vector_store,
    synthesizer_chain,
    evaluate_output,
    agent_llm,
    judge_llm
)

run_eval_suite.MODELS_POOL = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
agent_llm.request_timeout = 30.0
agent_llm.model_name = "openai/gpt-oss-120b"
judge_llm.request_timeout = 30.0
judge_llm.model_name = "openai/gpt-oss-120b"

# Retriever for Baseline with k=6
baseline_k6_retriever = vector_store.as_retriever(search_kwargs={"k": 6})

def run_baseline_k6(question: str) -> Dict[str, Any]:
    """Pipeline: Naive dense retrieval with top-k=6 directly to synthesizer (no rewrite, no grader)."""
    start_t = time.perf_counter()
    docs = baseline_k6_retriever.invoke(question)
    doc_contents = [d.page_content for d in docs]
    context_str = "\n\n".join(doc_contents)
    res = synthesizer_chain.invoke({"context": context_str, "question": question})
    duration = round(time.perf_counter() - start_t, 3)
    return {
        "answer": res.content.strip(),
        "retrieved_docs": doc_contents,
        "latency": duration
    }

def main():
    print("=" * 80)
    print("ABLATION STUDY: NAIVE BASELINE (k=6) ACROSS 28 FACTUAL SEC QUERIES")
    print("=" * 80)

    checkpoint_file = "ablation_checkpoint.json"
    cached = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"Loaded {len(cached)} existing samples from '{checkpoint_file}'.")
        except Exception:
            cached = {}

    results_baseline_k6: List[Dict[str, Any]] = []

    for i, item in enumerate(FACTUAL_QUERIES, 1):
        qid = item["id"]
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\n[{i}/28] [{qid}] Q: {q}")

        if qid in cached:
            rec = cached[qid]
            print(f"  -> [Cached] Lat: {rec['latency']}s | Hit: {rec['recall_hit']} | Refusal: {rec['false_refusal']}")
        else:
            out = run_baseline_k6(q)
            eval_res = evaluate_output(q, gt, out["retrieved_docs"], out["answer"], is_adversarial=False)
            rec = {
                "id": qid,
                "category": "factual",
                "question": q,
                "ground_truth": gt,
                "pipeline": "baseline_k6",
                "latency": out["latency"],
                "answer": out["answer"],
                "recall_hit": eval_res["recall_hit"],
                "false_refusal": eval_res["is_refusal"],
                "is_hallucination": eval_res["is_hallucination"],
                "judge_reasoning": eval_res["reasoning"]
            }
            cached[qid] = rec
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(cached, f, indent=2)
            print(f"     [Base k=6] Lat: {out['latency']}s | Hit: {eval_res['recall_hit']} | Refusal: {eval_res['is_refusal']}")
            time.sleep(1.0)
        results_baseline_k6.append(rec)

    # Load Baseline (k=3) and Agentic LangGraph (k=6) from eval_comparison.json
    with open("eval_comparison.json", "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    # 1. Baseline k=3
    base_k3_items = [q for q in comp_data["query_details"]["baseline"] if q["category"] == "factual"]
    hits_k3 = sum(q["recall_hit"] for q in base_k3_items)
    fr_k3 = sum(q["false_refusal"] for q in base_k3_items)
    lat_k3 = round(sum(q["latency"] for q in base_k3_items) / len(base_k3_items), 2)

    # 2. Baseline k=6 (Ablation)
    hits_k6 = sum(q["recall_hit"] for q in results_baseline_k6)
    fr_k6 = sum(q["false_refusal"] for q in results_baseline_k6)
    lat_k6 = round(sum(q["latency"] for q in results_baseline_k6) / len(results_baseline_k6), 2)

    # 3. Agentic LangGraph k=6
    agent_items = [q for q in comp_data["query_details"]["agentic_langgraph"] if q["category"] == "factual"]
    hits_agent = sum(q["recall_hit"] for q in agent_items)
    fr_agent = sum(q["false_refusal"] for q in agent_items)
    lat_agent = round(sum(q["latency"] for q in agent_items) / len(agent_items), 2)

    total_q = len(FACTUAL_QUERIES)

    # Output table
    table_md = f"""
| Metric | Baseline (k=3) | Baseline (k=6) | Agentic LangGraph (k=6) | Delta / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Recall (Factual)** | {hits_k3} / {total_q} ({hits_k3/total_q*100:.1f}%) | {hits_k6} / {total_q} ({hits_k6/total_q*100:.1f}%) | {hits_agent} / {total_q} ({hits_agent/total_q*100:.1f}%) | +{hits_k6-hits_k3} from k=6, +{hits_agent-hits_k6} from Agentic |
| **False Refusal Rate** | {fr_k3} / {total_q} ({fr_k3/total_q*100:.1f}%) | {fr_k6} / {total_q} ({fr_k6/total_q*100:.1f}%) | {fr_agent} / {total_q} ({fr_agent/total_q*100:.1f}%) | {fr_k3-fr_k6} reduction (k=6), {fr_k6-fr_agent} further (Agentic) |
| **Avg Latency (sec)** | {lat_k3:.1f}s | {lat_k6:.1f}s | {lat_agent:.1f}s | +{lat_k6-lat_k3:.1f}s (k=6), +{lat_agent-lat_k6:.1f}s (Agentic loop) |
"""
    print("\n" + "=" * 80)
    print("3-WAY COMPARATIVE ABLATION RESULTS (28 FACTUAL SEC QUERIES)")
    print("=" * 80)
    print(table_md)

    # Save full ablation output
    ablation_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_factual_queries": total_q,
            "description": "Ablation study comparing naive dense baseline at k=3 vs k=6 vs full Agentic LangGraph at k=6"
        },
        "metrics": {
            "recall_factual": {
                "baseline_k3": {"hits": hits_k3, "pct": round(hits_k3/total_q*100, 1)},
                "baseline_k6": {"hits": hits_k6, "pct": round(hits_k6/total_q*100, 1)},
                "agentic_k6": {"hits": hits_agent, "pct": round(hits_agent/total_q*100, 1)}
            },
            "false_refusal_rate": {
                "baseline_k3": {"count": fr_k3, "pct": round(fr_k3/total_q*100, 1)},
                "baseline_k6": {"count": fr_k6, "pct": round(fr_k6/total_q*100, 1)},
                "agentic_k6": {"count": fr_agent, "pct": round(fr_agent/total_q*100, 1)}
            },
            "avg_latency_seconds": {
                "baseline_k3": lat_k3,
                "baseline_k6": lat_k6,
                "agentic_k6": lat_agent
            }
        },
        "query_details": {
            "baseline_k6": results_baseline_k6
        }
    }

    with open("ablation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)
    print("Ablation results saved to 'ablation_comparison.json'.")

if __name__ == "__main__":
    main()
