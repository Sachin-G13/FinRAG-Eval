import os
import sys
import json
import re
import time
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_groq import ChatGroq
from agent import app

load_dotenv()


def load_dataset(dataset_path: str = "eval_dataset.json") -> List[Dict[str, Any]]:
    """Load the evaluation dataset from disk."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation dataset '{dataset_path}' not found. Run generate_eval_dataset.py first.")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test cases from '{dataset_path}'.")
    return data


def get_judge_llm(model_name: str = None) -> ChatGroq:
    """Initialize the LLM-as-a-Judge using ChatGroq in JSON mode."""
    if not model_name:
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return ChatGroq(
        model_name=model_name,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )


def judge_sample(
    judge_llm: ChatGroq,
    question: str,
    ground_truth: str,
    retrieved_chunks: List[str],
    agent_answer: str,
    is_unanswerable: bool = False
) -> Dict[str, Any]:
    """Score retrieval_relevance and groundedness using LLM-as-a-Judge."""
    context_str = "\n\n--- CHUNK BREAK ---\n\n".join(retrieved_chunks) if retrieved_chunks else "[No chunks retrieved]"

    prompt = (
        "You are an expert impartial AI evaluation auditor assessing an SEC Financial RAG Agent.\n"
        "Evaluate the following RAG system outputs against the ground truth reference and retrieved context.\n\n"
        f"Question:\n{question}\n\n"
        f"Ground Truth Reference:\n{ground_truth}\n\n"
        f"Is Unanswerable/Tricky Query: {is_unanswerable}\n\n"
        f"Retrieved Chunks:\n{context_str}\n\n"
        f"Agent's Generated Answer:\n{agent_answer}\n\n"
        "Scoring Guidelines:\n"
        "1. 'retrieval_relevance' (Score 1 or 0):\n"
        "   - For answerable factual queries: Score 1 if the retrieved chunks contain the key data/numbers needed to answer the question; Score 0 if the chunks lack the required info.\n"
        "   - For unanswerable/tricky queries: Score 1 if the retriever fetched relevant filing sections of the requested entity (even if the specific nonexistent fact is absent), or Score 0 if completely unrelated company documents were retrieved.\n"
        "2. 'groundedness' (Score 1 or 0):\n"
        "   - Score 1 if the answer is completely faithful to the context and facts without fabricating information. If the query is unanswerable, score 1 if the agent correctly admits it doesn't know / the information is not provided.\n"
        "   - Score 0 if the agent hallucinates, fabricates financial metrics, or presents unverified speculation as fact.\n\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "retrieval_relevance": 1,\n'
        '  "groundedness": 1,\n'
        '  "reasoning": "Brief explanation of scoring decisions"\n'
        "}"
    )

    try:
        res = judge_llm.invoke(prompt)
        content = res.content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                raise ValueError(f"Could not parse JSON: {content}")

        relevance = 1 if int(parsed.get("retrieval_relevance", 0)) == 1 else 0
        groundedness = 1 if int(parsed.get("groundedness", 0)) == 1 else 0
        reasoning = str(parsed.get("reasoning", "")).strip()

        return {
            "retrieval_relevance": relevance,
            "groundedness": groundedness,
            "reasoning": reasoning
        }
    except Exception as e:
        print(f"   [Judge Error]: {e}")
        return {
            "retrieval_relevance": 0,
            "groundedness": 1 if "don't know" in agent_answer.lower() or "not provided" in agent_answer.lower() else 0,
            "reasoning": f"Judge fallback evaluation due to parsing error: {e}"
        }


def run_evaluation(
    dataset_path: str = "eval_dataset.json",
    results_path: str = "eval_results.json",
    model_name: str = None
):
    if not model_name:
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    """Execute end-to-end evaluation pipeline and print summary benchmark table."""
    print("=" * 75)
    print("FINANCIAL RAG AGENT OFFLINE BENCHMARK EVALUATION")
    print("=" * 75)

    dataset = load_dataset(dataset_path)
    judge_llm = get_judge_llm(model_name)

    evaluated_records = []
    latencies = []
    relevance_scores = []
    groundedness_scores = []

    print(f"\nEvaluating {len(dataset)} queries through RAG graph and LLM-as-a-Judge...\n")

    for i, item in enumerate(dataset, 1):
        q_id = item.get("id", f"sample_{i:02d}")
        question = item["question"]
        ground_truth = item["ground_truth"]
        is_unanswerable = item.get("is_unanswerable", False)
        category = item.get("category", "general")

        print("-" * 75)
        print(f"[{i}/{len(dataset)}] [{q_id}] ({category})")
        print(f"Q: {question}")

        # 1. Run RAG agent and measure latency
        start_time = time.perf_counter()
        agent_output = app.invoke({"question": question})
        latency = round(time.perf_counter() - start_time, 2)
        latencies.append(latency)

        generated_answer = agent_output.get("generation", "").strip()
        rewritten_query = agent_output.get("rewritten_query", "")
        retrieved_docs = agent_output.get("retrieved_documents", agent_output.get("documents", []))

        print(f"-> Latency: {latency:.2f}s | Rewritten: \"{rewritten_query}\"")
        print(f"-> Agent Answer: {generated_answer[:140]}..." if len(generated_answer) > 140 else f"-> Agent Answer: {generated_answer}")

        # 2. LLM-as-a-Judge scoring
        judge_res = judge_sample(
            judge_llm=judge_llm,
            question=question,
            ground_truth=ground_truth,
            retrieved_chunks=retrieved_docs,
            agent_answer=generated_answer,
            is_unanswerable=is_unanswerable
        )

        rel = judge_res["retrieval_relevance"]
        grd = judge_res["groundedness"]
        reasoning = judge_res["reasoning"]

        relevance_scores.append(rel)
        groundedness_scores.append(grd)

        print(f"-> Scores: Retrieval Relevance = {rel} | Groundedness = {grd}")
        print(f"-> Judge Reasoning: {reasoning}")

        record = {
            "id": q_id,
            "category": category,
            "is_unanswerable": is_unanswerable,
            "question": question,
            "ground_truth": ground_truth,
            "rewritten_query": rewritten_query,
            "source_reference": item.get("source", ""),
            "agent_answer": generated_answer,
            "latency_seconds": latency,
            "retrieval_relevance": rel,
            "groundedness": grd,
            "judge_reasoning": reasoning,
            "retrieved_chunks_preview": [d[:150].strip() + "..." for d in retrieved_docs]
        }
        evaluated_records.append(record)
        time.sleep(0.3)  # Brief pause between samples

    # 3. Aggregate Metrics
    total_cases = len(evaluated_records)
    mean_retrieval_pct = round((sum(relevance_scores) / total_cases) * 100, 2) if total_cases else 0.0
    faithfulness_pct = round((sum(groundedness_scores) / total_cases) * 100, 2) if total_cases else 0.0
    hallucination_rate_pct = round(100.0 - faithfulness_pct, 2) if total_cases else 0.0
    avg_latency = round(sum(latencies) / total_cases, 2) if total_cases else 0.0

    summary = {
        "total_test_cases": total_cases,
        "mean_retrieval_accuracy_pct": mean_retrieval_pct,
        "faithfulness_groundedness_rate_pct": faithfulness_pct,
        "hallucination_rate_pct": hallucination_rate_pct,
        "average_query_latency_seconds": avg_latency,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    final_results = {
        "summary": summary,
        "results": evaluated_records
    }

    # 4. Save to disk
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("DETAILED RESULTS SAVED TO:", os.path.abspath(results_path))
    print("=" * 75)

    # 5. Print Formatted Benchmark Summary Table
    print("\n" + "+" + "-" * 42 + "+" + "-" * 20 + "+")
    print(f"| {'METRIC':<40} | {'VALUE':<18} |")
    print("+" + "-" * 42 + "+" + "-" * 20 + "+")
    print(f"| {'Total Test Cases':<40} | {total_cases:<18} |")
    print(f"| {'Mean Retrieval Accuracy (%)':<40} | {mean_retrieval_pct:>17.2f}% |")
    print(f"| {'Faithfulness / Groundedness Rate (%)':<40} | {faithfulness_pct:>17.2f}% |")
    print(f"| {'Hallucination Rate (%)':<40} | {hallucination_rate_pct:>17.2f}% |")
    print(f"| {'Average Query Latency (seconds)':<40} | {avg_latency:>16.2f}s |")
    print("+" + "-" * 42 + "+" + "-" * 20 + "+")
    print()

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline evaluation benchmark for Financial RAG Agent.")
    parser.add_argument("--dataset", default="eval_dataset.json", help="Path to evaluation dataset JSON")
    parser.add_argument("--output", default="eval_results.json", help="Path to save evaluation results JSON")
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), help="Model name for LLM judge")

    args = parser.parse_args()
    run_evaluation(
        dataset_path=args.dataset,
        results_path=args.output,
        model_name=args.model
    )
