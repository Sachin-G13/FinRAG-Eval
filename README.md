# FinRAG-Eval

FinRAG-Eval is an agentic Corrective RAG (CRAG) system and evaluation framework for financial SEC filings built with LangGraph, ChromaDB, and Groq.

## Key findings

| Setup | Recall | False refusals | Latency |
|---|---|---|---|
| Plain k=3 | 14/28 | 14/28 | 7.3s (~0.5x) |
| Plain k=6 | 18/28 | 9/28 | 13.6s (1x) |
| Agentic (CRAG) | 17/28 | 9/28 | 64.7s (~5x) |

- **Scaling top-k over agentic complexity**: Scaling retrieval from $k=3 \to 6$ cut false refusals significantly ($14/28 \to 9/28$) and improved recall ($14/28 \to 18/28$).
- **Latency vs. Lift**: Plain $k=6$ matched agentic recall ($18/28$ vs $17/28$) at **~1/5th the latency** (13.6s vs 64.7s; Plain $k=3$ at 7.3s), while the agentic loop added negligible measured lift.

## Error analysis

**12 of 13 misses: period mismatch / severed table chunks**

*(Note on the "13 misses": Plain $k=6$ had 10 retrieval misses and Agentic $k=6$ had 11 misses; the 13 audited cases represent the unique union of distinct queries that experienced retrieval misses across both $k=6$ configurations.)*

Root-cause classification across these 13 retrieval misses attributed 12 failures to two core ingestion and embedding challenges:
- **Severed Table Chunks (6/13 misses)**: Standard token-based text splitting broke multi-row financial tables mid-section (e.g., cutting off debt line items from liabilities or cash flow financing activities), preventing the LLM from synthesizing related figures.
- **Semantic Period Mismatch (6/13 misses)**: Dense embeddings favored semantically similar multi-year summaries or alternate fiscal years (e.g., pulling FY23–FY25 annual balance sheets rather than Q2 2026 statements).
- **Partial Header / Metadata Omission (1/13 misses)**: Cover page metadata (e.g., ticker symbol) was not co-located with notes chunks.

## Limitations

- **28-query eval set over 6 filings**: Tested on Apple and Tesla SEC filings (10-K, 10-Q, Shareholder Updates). Broader generalization requires scaling across additional industry sectors and document layouts.
- **LLM-as-judge**: Evaluated using `openai/gpt-oss-20b` (via Groq) in JSON mode at `temperature=0` across retrieval relevance and groundedness metrics; validated through qualitative reasoning audits and manual error inspection.
- **Agentic loop gave negligible lift**: Multi-step query rewriting and document grading in the CRAG loop could not repair severed table chunks or temporal drift, incurring a ~5x latency penalty without improving factual answer accuracy.

## Setup & usage

### 1. Installation

```bash
git clone https://github.com/Sachin-G13/FinRAG-Eval.git
cd FinRAG-Eval
python -m venv venv
venv\Scripts\activate   # On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

### 3. Ingestion & Benchmark Execution

- **Ingest SEC PDFs into ChromaDB**:
  ```bash
  python ingest.py
  ```
- **Run the evaluation suite**:
  ```bash
  python eval.py
  ```
- **Run 3-way ablation study (Plain k=3 vs Plain k=6 vs Agentic)**:
  ```bash
  python run_ablation_k6.py
  ```
- **Run retrieval error analysis**:
  ```bash
  python analyze_misses.py
  ```
