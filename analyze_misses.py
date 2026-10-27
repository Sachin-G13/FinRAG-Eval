import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("ablation_comparison.json", encoding="utf-8") as f:
    ab_data = json.load(f)
with open("eval_comparison.json", encoding="utf-8") as f:
    ev_data = json.load(f)

base_k6 = {q["id"]: q for q in ab_data["query_details"]["baseline_k6"]}
base_k3 = {q["id"]: q for q in ev_data["query_details"]["baseline"] if q["category"] == "factual"}
agent_k6 = {q["id"]: q for q in ev_data["query_details"]["agentic_langgraph"] if q["category"] == "factual"}

print("================================================================================")
print("MISS DIAGNOSTICS & ROOT CAUSE CLASSIFICATION")
print("================================================================================")

all_qids = [f"F{i:02d}" for i in range(1, 29)]

categories = {
    "Table Boundary Severance (Chunking Artifact)": [],
    "Semantic Period Mismatch (Temporal Drift)": [],
    "Partial Token / Header Omission": []
}

# Known root causes from qualitative analysis of retrieved chunks:
# F01: Par value retrieved, but trading symbol on cover page header wasn't co-located. -> Partial Token / Header Omission
# F02: Chunk cut off mid-balance-sheet liabilities -> Table Boundary Severance
# F03: Retrieved FY23/24/25 instead of Q2 2026 -> Semantic Period Mismatch
# F05: Cash flow table severed between operating and financing rows -> Table Boundary Severance
# F07: EBITDA note footnote not retrieved -> Semantic Period Mismatch
# F10: Q3 2024 EBITDA margin retrieved instead of dollar values in another table -> Semantic Period Mismatch
# F13: Statement of Operations severed between operating income and per-share items -> Table Boundary Severance
# F15: Connection error on agentic judge fallback -> Transient / Network
# F20: Segment revenue table chunk truncated before iPhone rows -> Table Boundary Severance
# F21: Condensed Consolidated Statements truncated before Operating Income line -> Table Boundary Severance
# F26: Q4 update summary chunk lacked full-year comparison table -> Semantic Period Mismatch
# F27: Quarterly margins table omitted; full year retrieved -> Semantic Period Mismatch
# F28: R&D expense line not in top-k chunks -> Semantic Period Mismatch

root_cause_map = {
    "F01": ("Partial Token / Header Omission", "Par value ($0.00001) retrieved from Notes, but Nasdaq symbol ('AAPL') on Form 10-K cover was not in top chunks."),
    "F02": ("Table Boundary Severance", "Q1 2026 balance sheet chunk cuts off right after 'Other current liabilities', severing 'Term debt' line items."),
    "F03": ("Semantic Period Mismatch", "Retriever fetched annual FY23/FY24/FY25 balance sheet tables instead of Q2 2026 quarterly equity statements."),
    "F05": ("Table Boundary Severance", "Cash flow financing section was severed across chunk boundaries; stock repurchases separated from debt repayments."),
    "F07": ("Semantic Period Mismatch", "Retrieved summary highlights mentioning EBITDA adjustments without the specific footnote defining digital asset treatment."),
    "F10": ("Semantic Period Mismatch", "Retriever pulled margin percentage tables rather than the non-GAAP dollar reconciliation table for Q3 2023 vs Q3 2024."),
    "F13": ("Table Boundary Severance", "Statement of operations chunk cut off diluted EPS rows away from net income."),
    "F15": ("Judge Fallback", "Transient judge connection timeout during initial agentic evaluation run."),
    "F20": ("Table Boundary Severance", "Q2 2026 segment table chunk truncated after Services, omitting iPhone revenue row."),
    "F21": ("Table Boundary Severance", "Condensed statement of operations truncated after total operating expenses, dropping Operating Income."),
    "F26": ("Semantic Period Mismatch", "Retrieved quarterly Q4 updates instead of the full-year 2025 vs 2024 summary table."),
    "F27": ("Semantic Period Mismatch", "Retrieved annual multi-year summary table (2021-2025) instead of Q3 2025 quarterly margin breakdown."),
    "F28": ("Semantic Period Mismatch", "Retrieved production volume highlights and webcast info rather than Q3 vs Q2 2025 R&D operating expense breakdown.")
}

for qid in all_qids:
    k3 = base_k3[qid]["recall_hit"]
    k6 = base_k6[qid]["recall_hit"]
    ag = agent_k6[qid]["recall_hit"]
    if k3 == 0 or k6 == 0 or ag == 0:
        cause, desc = root_cause_map.get(qid, ("Other", "N/A"))
        print(f"| {qid} | {'HIT' if k3 else 'MISS'} | {'HIT' if k6 else 'MISS'} | {'HIT' if ag else 'MISS'} | {cause} | {desc} |")


all_qids = [f"F{i:02d}" for i in range(1, 29)]
miss_categories = {
    "table_severance": [],
    "semantic_period_mismatch": [],
    "partial_entity_coverage": []
}

for qid in all_qids:
    k3_hit = base_k3[qid]["recall_hit"]
    k6_hit = base_k6[qid]["recall_hit"]
    ag_hit = agent_k6[qid]["recall_hit"]
    if k3_hit == 0 or k6_hit == 0 or ag_hit == 0:
        print(f"\n[{qid}] Base(k=3): {'HIT' if k3_hit else 'MISS'} | Base(k=6): {'HIT' if k6_hit else 'MISS'} | Agentic(k=6): {'HIT' if ag_hit else 'MISS'}")
        print(f"Question: {base_k6[qid]['question']}")
        print(f"Ground Truth: {base_k6[qid]['ground_truth']}")
        if k6_hit == 0:
            print(f"  -> Base(k=6) Miss Reason: {base_k6[qid]['judge_reasoning']}")
        if ag_hit == 0:
            print(f"  -> Agent(k=6) Miss Reason: {agent_k6[qid]['judge_reasoning']}")

print("\n" + "=" * 80)
