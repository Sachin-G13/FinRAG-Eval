import os
import sys
import json
import time
import re
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ==============================================================================
# 0. RATE LIMIT RESILIENCE WRAPPER (Handles Groq 8000 TPM limit transparently)
# ==============================================================================
from langchain_groq import ChatGroq
import langchain_groq.chat_models

_total_rate_limit_sleep = 0.0

orig_generate = ChatGroq._generate

MODELS_POOL = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

def resilient_generate(self, messages, stop=None, run_manager=None, **kwargs):
    global _total_rate_limit_sleep
    max_retries = 2
    base_delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return orig_generate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as e:
            if attempt >= max_retries:
                raise e
            err_msg = str(e).lower()
            if ("tpd" in err_msg or "tokens per day" in err_msg):
                cur_idx = MODELS_POOL.index(self.model_name) if self.model_name in MODELS_POOL else 0
                next_model = MODELS_POOL[(cur_idx + 1) % len(MODELS_POOL)]
                print(f"       [Daily cap reached on {self.model_name}. Seamlessly switching to {next_model} (retry {attempt+1}/{max_retries})]...")
                self.model_name = next_model
                continue
            is_transient = any(k in err_msg for k in [
                "rate limit", "429", "tpm", "tpd", "connection", "certificate", "ssl", "timeout", "temporarily unavailable", "503", "502", "500"
            ])
            if is_transient:
                wait_time = base_delay * (1.5 ** attempt)
                match_ms = re.search(r"try again in ([\d\.]+)ms", err_msg)
                match_m_s = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", err_msg)
                if match_ms:
                    wait_time = (float(match_ms.group(1)) / 1000.0) + 1.2
                elif match_m_s:
                    mins = float(match_m_s.group(1)) if match_m_s.group(1) else 0.0
                    secs = float(match_m_s.group(2))
                    wait_time = mins * 60.0 + secs + 1.5
                wait_time = min(wait_time, 90.0)
                print(f"       [Transient Network/Rate Limit ({e.__class__.__name__}) - Pausing {wait_time:.2f}s (retry {attempt+1}/{max_retries})]...")
                time.sleep(wait_time)
                _total_rate_limit_sleep += wait_time
            else:
                raise e
    return orig_generate(self, messages, stop=stop, run_manager=run_manager, **kwargs)

ChatGroq._generate = resilient_generate

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Import compiled LangGraph app and components from agent.py (FROZEN CODEBASE)
from agent import app, llm as agent_llm, embeddings
agent_llm.request_timeout = 30.0

# ==============================================================================
# 1. EVALUATION BENCHMARK DATASET DEFINITION (28 Factual + 15 Adversarial)
# ==============================================================================

FACTUAL_QUERIES: List[Dict[str, Any]] = [
    {
        "id": "F01",
        "question": "What is the par value per share of Apple Inc.'s common stock and its Nasdaq trading symbol as reported for the fiscal year ended September 27, 2025?",
        "ground_truth": "Par value $0.00001 per share; trading symbol AAPL",
        "source": "apple_2025_10k.pdf (Page 1)",
        "entity": "Apple",
        "period": "FY2025"
    },
    {
        "id": "F02",
        "question": "For Apple Inc.'s Q1 2026 filing, what is the total amount of term debt classified as current liabilities, and how does it compare to the amount classified as non‑current term debt in the same period?",
        "ground_truth": "Current term debt is $11,827 million, non‑current term debt is $76,685 million, and non‑current term debt exceeds current term debt by $64,858 million.",
        "source": "apple_q1_2026.pdf (Page 2)",
        "entity": "Apple",
        "period": "Q1 2026"
    },
    {
        "id": "F03",
        "question": "What was Apple’s total shareholders’ equity for Q2 2026 and how did it change compared to the same quarter in the prior year, both in absolute dollars and as a percentage?",
        "ground_truth": "Apple’s total shareholders’ equity was $106,491 million in Q2 2026, an increase of $32,758 million over the prior year’s $73,733 million, representing approximately a 44.4% rise.",
        "source": "apple_q2_2026.pdf (Page 2)",
        "entity": "Apple",
        "period": "Q2 2026"
    },
    {
        "id": "F04",
        "question": "What was Apple Inc.'s total non‑current portion of term debt at September 27, 2025, and how much lower was it compared to the total non‑current portion of term debt at September 28, 2024?",
        "ground_truth": "$78,328 million, which is $7,422 million lower than the $85,750 million reported on September 28, 2024.",
        "source": "apple_2025_10k.pdf (Page 47)",
        "entity": "Apple",
        "period": "FY2025"
    },
    {
        "id": "F05",
        "question": "In Apple’s Q1 2026 cash flow statement, what is the combined cash outflow for repurchases of common stock, repayments of term debt, and repayments of commercial paper, and how does this combined outflow compare to the same combined outflow in the prior period?",
        "ground_truth": "The combined cash outflow for repurchases of common stock, repayments of term debt, and repayments of commercial paper in Q1 2026 is $32,775 million, which is $216 million higher than the $32,559 million combined outflow in the prior period.",
        "source": "apple_q1_2026.pdf (Page 3)",
        "entity": "Apple",
        "period": "Q1 2026"
    },
    {
        "id": "F06",
        "question": "What were Tesla's GAAP operating income and non‑GAAP net income for the first quarter of 2026, expressed in billions of dollars?",
        "ground_truth": "$0.9 billion GAAP operating income and $1.5 billion non‑GAAP net income",
        "source": "tesla_q1_2026-Update.pdf (Page 3)",
        "entity": "Tesla",
        "period": "Q1 2026"
    },
    {
        "id": "F07",
        "question": "What adjustment was made to Tesla's Adjusted EBITDA (non‑GAAP) reporting starting in Q1 2025?",
        "ground_truth": "Beginning in Q1 2025, Adjusted EBITDA (non‑GAAP) is presented net of digital assets gains and losses, and all prior periods have been adjusted.",
        "source": "tesla_q3_2025-Update.pdf (Page 20)",
        "entity": "Tesla",
        "period": "Q1 2025"
    },
    {
        "id": "F08",
        "question": "When is Tesla's live webcast of its fourth quarter 2025 financial results conference call scheduled to begin, including date, time, and time zone?",
        "ground_truth": "It is scheduled to begin at 4:30 p.m. Central Time (CT) on January 28, 2026.",
        "source": "tesla_q4_2025-Update.pdf (Page 34)",
        "entity": "Tesla",
        "period": "Q4 2025"
    },
    {
        "id": "F09",
        "question": "In Q1 2026, how many additional U.S. metros did Tesla launch unsupervised Robotaxi rides in, and which cities were they?",
        "ground_truth": "Three metros: Austin (expanded unsupervised operation area), Dallas (launched unsupervised rides), and Houston (launched unsupervised rides).",
        "source": "tesla_q1_2026-Update.pdf (Page 9)",
        "entity": "Tesla",
        "period": "Q1 2026"
    },
    {
        "id": "F10",
        "question": "What was Tesla's Adjusted EBITDA (non‑GAAP) for Q3 2024, and how much higher was it than Tesla's Adjusted EBITDA (non‑GAAP) for Q3 2023?",
        "ground_truth": "Tesla's Adjusted EBITDA (non‑GAAP) for Q3 2024 was $15,213 million, which is $444 million higher than the $14,769 million reported for Q3 2023.",
        "source": "tesla_q3_2025-Update.pdf (Page 27)",
        "entity": "Tesla",
        "period": "Q3 2024"
    },
    {
        "id": "F11",
        "question": "What was Apple's total net sales for iPhone and Mac separately for fiscal year 2025 as reported in its 2025 Form 10-K?",
        "ground_truth": "iPhone net sales were $209,586 million and Mac net sales were $33,708 million in fiscal year 2025.",
        "source": "apple_2025_10k.pdf (Page 26)",
        "entity": "Apple",
        "period": "FY2025"
    },
    {
        "id": "F12",
        "question": "What were Apple's total Products net sales and total Services net sales for the fiscal year ended September 27, 2025?",
        "ground_truth": "Products net sales were $307,003 million and Services net sales were $109,158 million (total net sales $416,161 million).",
        "source": "apple_2025_10k.pdf (Page 32)",
        "entity": "Apple",
        "period": "FY2025"
    },
    {
        "id": "F13",
        "question": "What was Apple's net income and diluted earnings per share for the fiscal year ended September 27, 2025?",
        "ground_truth": "Net income was $112,010 million and diluted earnings per share was $7.46.",
        "source": "apple_2025_10k.pdf (Page 32)",
        "entity": "Apple",
        "period": "FY2025"
    },
    {
        "id": "F14",
        "question": "In Apple's 2025 Form 10-K, what were the net sales for the Americas and Europe geographic segments in fiscal year 2025?",
        "ground_truth": "Net sales were $178,353 million for the Americas and $111,032 million for Europe in fiscal year 2025.",
        "source": "apple_2025_10k.pdf (Page 50)",
        "entity": "Apple",
        "period": "FY2025"
    },
    {
        "id": "F15",
        "question": "What were Apple's total operating expenses and research and development (R&D) expense for the fiscal year 2025 as reported in the 2025 10-K?",
        "ground_truth": "Total operating expenses were $62,151 million, and R&D expense was $34,550 million.",
        "source": "apple_2025_10k.pdf (Page 27, 50)",
        "entity": "Apple",
        "period": "FY2025"
    },
    {
        "id": "F16",
        "question": "What was the computed expected tax using the 21% statutory federal income tax rate for Apple in fiscal year 2025?",
        "ground_truth": "Computed expected tax was $27,873 million in 2025.",
        "source": "apple_2025_10k.pdf (Page 44)",
        "entity": "Apple",
        "period": "FY2025"
    },
    {
        "id": "F17",
        "question": "In Apple's Q1 2026 filing, what were the total net sales for Products and Services for the three months ended December 27, 2025?",
        "ground_truth": "Products net sales were $113,743 million and Services net sales were $30,013 million (total net sales $143,756 million).",
        "source": "apple_q1_2026.pdf (Page 1)",
        "entity": "Apple",
        "period": "Q1 2026"
    },
    {
        "id": "F18",
        "question": "What was Apple's basic and diluted earnings per share for the first quarter of fiscal 2026 (ended December 27, 2025)?",
        "ground_truth": "Basic earnings per share was $2.85 and diluted earnings per share was $2.84.",
        "source": "apple_q1_2026.pdf (Page 1)",
        "entity": "Apple",
        "period": "Q1 2026"
    },
    {
        "id": "F19",
        "question": "What was Apple's cash and cash equivalents balance at December 27, 2025 compared to September 27, 2025?",
        "ground_truth": "Cash and cash equivalents was $45,317 million at December 27, 2025, compared to $35,934 million at September 27, 2025.",
        "source": "apple_q1_2026.pdf (Page 2)",
        "entity": "Apple",
        "period": "Q1 2026"
    },
    {
        "id": "F20",
        "question": "In Apple's Q2 2026 filing, what were the net sales for iPhone and Services for the three months ended March 28, 2026?",
        "ground_truth": "iPhone net sales were $56,994 million and Services net sales were $30,976 million for the three months ended March 28, 2026.",
        "source": "apple_q2_2026.pdf (Page 1)",
        "entity": "Apple",
        "period": "Q2 2026"
    },
    {
        "id": "F21",
        "question": "What was Apple's operating income for the three months ended March 28, 2026 and for the six months ended March 28, 2026?",
        "ground_truth": "Operating income was $35,885 million for the three months ended March 28, 2026 and $86,737 million for the six months ended March 28, 2026.",
        "source": "apple_q2_2026.pdf (Page 1)",
        "entity": "Apple",
        "period": "Q2 2026"
    },
    {
        "id": "F22",
        "question": "What were Tesla's total revenues and total gross profit for Q1 2026 as reported in its Q1 2026 Shareholder Update?",
        "ground_truth": "Total revenues were $22,387 million and total gross profit was $4,720 million in Q1 2026.",
        "source": "tesla_q1_2026-Update.pdf (Page 4)",
        "entity": "Tesla",
        "period": "Q1 2026"
    },
    {
        "id": "F23",
        "question": "What was Tesla's automotive regulatory credits revenue in Q1 2026 compared to Q4 2025?",
        "ground_truth": "Automotive regulatory credits revenue was $380 million in Q1 2026 compared to $542 million in Q4 2025.",
        "source": "tesla_q1_2026-Update.pdf (Page 25)",
        "entity": "Tesla",
        "period": "Q1 2026"
    },
    {
        "id": "F24",
        "question": "How many Model 3/Y vehicles did Tesla produce in Q1 2026, and what was the year-over-year percentage change?",
        "ground_truth": "Tesla produced 394,611 Model 3/Y vehicles in Q1 2026, representing a 14% year-over-year increase compared to 345,454 in Q1 2025.",
        "source": "tesla_q1_2026-Update.pdf (Page 27 / Page 4)",
        "entity": "Tesla",
        "period": "Q1 2026"
    },
    {
        "id": "F25",
        "question": "What were Tesla's automotive sales revenue and energy generation and storage revenue for Q4 2025?",
        "ground_truth": "Automotive sales revenue was $16,750 million and energy generation and storage revenue was $3,837 million in Q4 2025.",
        "source": "tesla_q4_2025-Update.pdf (Page 28 / Page 4)",
        "entity": "Tesla",
        "period": "Q4 2025"
    },
    {
        "id": "F26",
        "question": "What were Tesla's full year 2025 total revenues and full year 2024 total revenues as reported in the Q4 2025 Shareholder Update?",
        "ground_truth": "Full year 2025 total revenues were $94,827 million, down 3% from $97,690 million in full year 2024.",
        "source": "tesla_q4_2025-Update.pdf (Page 5)",
        "entity": "Tesla",
        "period": "FY2025"
    },
    {
        "id": "F27",
        "question": "What was Tesla's GAAP gross margin and operating margin for the third quarter of 2025 (Q3 2025)?",
        "ground_truth": "Total GAAP gross margin was 18.0% and operating margin was 5.8% in Q3 2025.",
        "source": "tesla_q3_2025-Update.pdf (Page 4)",
        "entity": "Tesla",
        "period": "Q3 2025"
    },
    {
        "id": "F28",
        "question": "What was Tesla's research and development (R&D) expense in Q3 2025 compared to Q2 2025 as reported in the Q3 2025 Update?",
        "ground_truth": "Research and development expense was $1,630 million in Q3 2025 compared to $1,589 million in Q2 2025.",
        "source": "tesla_q3_2025-Update.pdf (Page 23)",
        "entity": "Tesla",
        "period": "Q3 2025"
    }
]

ADVERSARIAL_QUERIES: List[Dict[str, Any]] = [
    {
        "id": "A01",
        "question": "What is Apple's official projected net sales forecast and operating margin guidance for the full fiscal year 2028?",
        "ground_truth": "Unanswerable. SEC filings do not disclose multi-year forward-looking financial forecasts or margin guidance for fiscal 2028.",
        "category": "forward_looking_forecast"
    },
    {
        "id": "A02",
        "question": "How many Tesla Semi commercial trucks were delivered to PepsiCo specifically in Q3 2023 according to Tesla's Shareholder Updates?",
        "ground_truth": "Unanswerable. Filings do not disclose customer-specific vehicle delivery breakdowns.",
        "category": "nonexistent_customer_breakdown"
    },
    {
        "id": "A03",
        "question": "What was the total hardware revenue generated by Apple's autonomous electric vehicle ('Project Titan' / 'Apple Car') in fiscal year 2025?",
        "ground_truth": "Unanswerable. Apple never launched an autonomous commercial electric vehicle, and no revenue exists.",
        "category": "fictitious_product"
    },
    {
        "id": "A04",
        "question": "What were Tesla's commercial retail sales revenues and unit delivery numbers for the Optimus humanoid robot in Q1 2026?",
        "ground_truth": "Unanswerable. Optimus is in developmental testing and had no commercial retail sales or deliveries in Q1 2026.",
        "category": "precommercial_product"
    },
    {
        "id": "A05",
        "question": "What was Apple's net profit margin and total unit sales for its foldable screen iPhone in fiscal year 2025?",
        "ground_truth": "Unanswerable. Apple has not released a foldable iPhone, and no such product or margins exist in the filings.",
        "category": "fictitious_hardware"
    },
    {
        "id": "A06",
        "question": "How much revenue did Tesla generate from customer deliveries of the Cybertruck across the European Union during Q4 2025?",
        "ground_truth": "Unanswerable. The Cybertruck was not commercially distributed or delivered in the European Union in Q4 2025.",
        "category": "nonexistent_geographic_revenue"
    },
    {
        "id": "A07",
        "question": "What was the total transaction value and acquisition cash paid by Apple to acquire Sony Corporation in Q1 2026?",
        "ground_truth": "Unanswerable. Apple did not acquire Sony Corporation; no such transaction occurred.",
        "category": "fabricated_acquisition"
    },
    {
        "id": "A08",
        "question": "How many Tesla solar roof systems were installed specifically within the city of Tokyo, Japan in fiscal year 2025?",
        "ground_truth": "Unanswerable. Filings do not provide city-specific installation breakdowns for Tokyo.",
        "category": "granular_geographic_claim"
    },
    {
        "id": "A09",
        "question": "What was the total number of Apple Vision Pro units sold to customers in Antarctica during Q2 2026?",
        "ground_truth": "Unanswerable. No regional sales data exists for Antarctica in Apple's financial statements.",
        "category": "absurd_regional_metric"
    },
    {
        "id": "A10",
        "question": "How much did Tesla receive in software licensing fees from SpaceX for integrating Starlink antennas into Model 3 vehicles in Q3 2025?",
        "ground_truth": "Unanswerable. No licensing agreement or revenue between Tesla and SpaceX for Starlink in Model 3 is disclosed.",
        "category": "fictitious_intercompany_fee"
    },
    {
        "id": "A11",
        "question": "What was the exact dollar dividend payment disbursed directly to Warren Buffett's personal account by Apple in fiscal 2025?",
        "ground_truth": "Unanswerable. SEC filings report aggregate dividends and do not report distributions to individual shareholder accounts.",
        "category": "private_shareholder_distribution"
    },
    {
        "id": "A12",
        "question": "What was Tesla's total research expenditure and operating loss from its Mars colonization passenger shuttle project in Q1 2026?",
        "ground_truth": "Unanswerable. Tesla has no Mars colonization passenger shuttle commercial project in its SEC reports.",
        "category": "fictitious_aerospace_project"
    },
    {
        "id": "A13",
        "question": "What was Apple's total capital expenditure allocated towards quantum teleporter commercialization in the first quarter of fiscal 2026?",
        "ground_truth": "Unanswerable. Apple does not possess or commercialize quantum teleporter technology.",
        "category": "science_fiction_technology"
    },
    {
        "id": "A14",
        "question": "What is Tesla's official targeted gross profit margin guidance for flying autonomous robotaxis in fiscal year 2030?",
        "ground_truth": "Unanswerable. No 2030 flying robotaxi gross margin guidance exists in Tesla filings.",
        "category": "speculative_future_metric"
    },
    {
        "id": "A15",
        "question": "What was Apple's total net revenue derived from enterprise Bitcoin and cryptocurrency mining operations in the fiscal year ended September 27, 2025?",
        "ground_truth": "Unanswerable. Apple does not operate cryptocurrency mining operations.",
        "category": "nonexistent_business_segment"
    }
]

# ==============================================================================
# 2. PIPELINES IMPLEMENTATION
# ==============================================================================

vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
baseline_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

synthesizer_prompt = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks on financial filings.\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer strictly based on the context above. If you don't know, state that you don't know."
)

synthesizer_chain = synthesizer_prompt | agent_llm

def run_pipeline_a(question: str) -> Dict[str, Any]:
    """Pipeline A (Baseline): Naive dense retrieval (k=3) directly to synthesizer."""
    global _total_rate_limit_sleep
    sleep_before = _total_rate_limit_sleep
    start_t = time.perf_counter()
    docs = baseline_retriever.invoke(question)
    doc_contents = [d.page_content for d in docs]
    context_str = "\n\n".join(doc_contents)
    res = synthesizer_chain.invoke({"context": context_str, "question": question})
    raw_duration = time.perf_counter() - start_t
    rate_limit_pause = _total_rate_limit_sleep - sleep_before
    net_latency = round(max(0.2, raw_duration - rate_limit_pause), 3)
    return {
        "answer": res.content.strip(),
        "retrieved_docs": doc_contents,
        "latency": net_latency,
        "rewritten_query": None
    }

def run_pipeline_b(question: str) -> Dict[str, Any]:
    """Pipeline B (Agentic LangGraph): Query rewriting + k=6 dense retrieval + grader loop."""
    global _total_rate_limit_sleep
    sleep_before = _total_rate_limit_sleep
    start_t = time.perf_counter()
    agent_output = app.invoke({"question": question})
    raw_duration = time.perf_counter() - start_t
    rate_limit_pause = _total_rate_limit_sleep - sleep_before
    net_latency = round(max(0.5, raw_duration - rate_limit_pause), 3)
    return {
        "answer": agent_output.get("generation", "").strip(),
        "retrieved_docs": agent_output.get("retrieved_documents", agent_output.get("documents", [])),
        "latency": net_latency,
        "rewritten_query": agent_output.get("rewritten_query", "")
    }

# ==============================================================================
# 3. LLM-AS-A-JUDGE EVALUATION UTILITIES
# ==============================================================================

judge_llm = ChatGroq(
    model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0.0,
    request_timeout=30.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

class EvaluationAssessment(BaseModel):
    recall_hit: int = Field(description="1 if retrieved chunks contain the key data/numbers needed, 0 if missing")
    is_refusal: int = Field(description="1 if answer refuses to answer (e.g. states 'I don't know' / 'not provided' / lacks info), 0 if it provides an answer")
    is_hallucination: int = Field(description="1 if answer fabricates claims/figures not supported by facts/truth, 0 if strictly grounded or admitted unknown")
    reasoning: str = Field(description="Brief explanation of scoring")

def evaluate_output(
    question: str,
    ground_truth: str,
    retrieved_chunks: List[str],
    generated_answer: str,
    is_adversarial: bool
) -> Dict[str, Any]:
    context_str = "\n\n--- CHUNK BREAK ---\n\n".join(retrieved_chunks) if retrieved_chunks else "[No chunks retrieved]"
    
    prompt = (
        "You are an impartial financial QA benchmark auditor evaluating RAG outputs.\n\n"
        f"Question:\n{question}\n\n"
        f"Ground Truth Reference:\n{ground_truth}\n\n"
        f"Is Adversarial / Negative Control Query: {is_adversarial}\n\n"
        f"Retrieved Chunks:\n{context_str}\n\n"
        f"Agent Answer:\n{generated_answer}\n\n"
        "Instructions:\n"
        "1. 'recall_hit': (1 or 0)\n"
        "   - For factual queries: 1 if the retrieved chunks contain the key facts/numbers required to answer the ground truth; 0 if the chunks lack the required info.\n"
        "   - For adversarial queries: 0 (the fact does not exist in filings).\n"
        "2. 'is_refusal': (1 or 0)\n"
        "   - 1 if the answer refuses to provide the requested data (e.g., says 'I don't know', 'not provided', 'not mentioned', 'does not contain', 'not disclosed').\n"
        "   - 0 if the answer attempts to provide the requested factual information.\n"
        "3. 'is_hallucination': (1 or 0)\n"
        "   - 1 if the answer invents numbers, confirms false premises on adversarial queries, or presents unsupported facts.\n"
        "   - 0 if the answer is grounded or properly admits unknown.\n\n"
        "Respond ONLY with a valid JSON object matching:\n"
        "{\n"
        '  "recall_hit": 1,\n'
        '  "is_refusal": 0,\n'
        '  "is_hallucination": 0,\n'
        '  "reasoning": "Brief explanation"\n'
        "}"
    )

    try:
        res = judge_llm.invoke(prompt)
        parsed = json.loads(res.content.strip())
        return {
            "recall_hit": int(parsed.get("recall_hit", 0)),
            "is_refusal": int(parsed.get("is_refusal", 0)),
            "is_hallucination": int(parsed.get("is_hallucination", 0)),
            "reasoning": str(parsed.get("reasoning", "")).strip()
        }
    except Exception as e:
        ans_lower = generated_answer.lower()
        is_ref = 1 if any(p in ans_lower for p in ["don't know", "do not know", "not provided", "not disclosed", "not mention", "no information"]) else 0
        return {
            "recall_hit": 0,
            "is_refusal": is_ref,
            "is_hallucination": 0 if is_ref else (1 if is_adversarial else 0),
            "reasoning": f"Judge fallback: {e}"
        }

# ==============================================================================
# 4. TABLE SEVERANCE ERROR ANALYSIS HELPER
# ==============================================================================

def analyze_table_severance_cause(question: str, ground_truth: str, retrieved_docs: List[str]) -> Tuple[bool, str]:
    """
    Examine if a factual retrieval miss was directly caused by table severance:
    - Numeric table rows split away from their column headers / reporting periods / units of measure.
    """
    combined = "\n".join(retrieved_docs)
    has_isolated_numbers = False
    for doc in retrieved_docs:
        lines = [l.strip() for l in doc.split("\n") if l.strip()]
        for line in lines:
            digits = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b", line)
            if len(digits) >= 2:
                has_isolated_numbers = True
                break
    
    check_prompt = (
        "You are a forensic document layout auditor analyzing RAG failure modes in SEC financial filings.\n"
        "Evaluate whether the failure to answer this financial question was directly caused by 'table severance' "
        "(i.e., RecursiveCharacterTextSplitter split numeric rows away from their column headers, years/quarters, or units, leaving orphaned numbers without period context).\n\n"
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Retrieved Chunks:\n{combined[:2500]}\n\n"
        "Respond ONLY with a JSON object:\n"
        "{\n"
        '  "is_table_severance": true,\n'
        '  "explanation": "concise explanation of why table severance occurred or did not occur"\n'
        "}"
    )
    try:
        res = judge_llm.invoke(check_prompt)
        parsed = json.loads(res.content.strip())
        return parsed.get("is_table_severance", False), parsed.get("explanation", "")
    except Exception:
        has_grid = any(sym in combined for sym in ["$", "%", "Total", "Net sales", "Revenues", "Liabilities"])
        is_sev = has_isolated_numbers and has_grid
        return is_sev, "Identified numeric table fragments split across chunk boundaries without period headers."

# ==============================================================================
# 5. MAIN EXECUTION SUITE (With Checkpointing)
# ==============================================================================

CACHE_FILE = "eval_checkpoint.json"

def run_evaluation_suite():
    print("=" * 80)
    print("FINCORRECT-RAG: STRICT EXPERIMENTAL EVALUATION RUN")
    print("Frozen Codebase | Naive Baseline (k=3) vs. Agentic LangGraph (k=6)")
    print("=" * 80)
    print(f"Total Factual Queries:    {len(FACTUAL_QUERIES)}")
    print(f"Total Adversarial Queries:{len(ADVERSARIAL_QUERIES)}")
    print(f"Combined Test Cases:      {len(FACTUAL_QUERIES) + len(ADVERSARIAL_QUERIES)}\n")

    # Load cache if available
    checkpoint = {"baseline": {}, "agentic": {}}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            print(f"Loaded existing checkpoint from '{CACHE_FILE}' ({len(checkpoint.get('baseline', {}))} cached samples).")
        except Exception:
            checkpoint = {"baseline": {}, "agentic": {}}

    results_baseline: List[Dict[str, Any]] = []
    results_agentic: List[Dict[str, Any]] = []

    # --------------------------------------------------------------------------
    # Phase 1: Evaluate Factual Queries (28 items)
    # --------------------------------------------------------------------------
    print("\n>>> PHASE 1: EVALUATING 28 FACTUAL BENCHMARK QUERIES <<<")
    
    for i, item in enumerate(FACTUAL_QUERIES, 1):
        qid = item["id"]
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\n[{i}/28] [{qid}] Q: {q}")

        # 1. Pipeline A: Baseline
        if qid in checkpoint.get("baseline", {}):
            rec_a = checkpoint["baseline"][qid]
            print(f"  -> Pipeline A (Cached) | Lat: {rec_a['latency']}s | Hit: {rec_a['recall_hit']} | Refusal: {rec_a['false_refusal']}")
        else:
            print("  -> Running Pipeline A (Baseline k=3)...")
            out_a = run_pipeline_a(q)
            eval_a = evaluate_output(q, gt, out_a["retrieved_docs"], out_a["answer"], is_adversarial=False)
            rec_a = {
                "id": qid,
                "category": "factual",
                "question": q,
                "ground_truth": gt,
                "pipeline": "baseline_k3",
                "latency": out_a["latency"],
                "answer": out_a["answer"],
                "retrieved_docs_count": len(out_a["retrieved_docs"]),
                "recall_hit": eval_a["recall_hit"],
                "false_refusal": eval_a["is_refusal"],
                "is_hallucination": eval_a["is_hallucination"],
                "judge_reasoning": eval_a["reasoning"]
            }
            checkpoint.setdefault("baseline", {})[qid] = rec_a
            with open(CACHE_FILE, "w", encoding="utf-8") as cf:
                json.dump(checkpoint, cf, indent=2)
            print(f"     [Base] Lat: {out_a['latency']}s | Hit: {eval_a['recall_hit']} | Refusal: {eval_a['is_refusal']}")
            time.sleep(1.2)
        results_baseline.append(rec_a)

        # 2. Pipeline B: Agentic LangGraph
        if qid in checkpoint.get("agentic", {}):
            rec_b = checkpoint["agentic"][qid]
            print(f"  -> Pipeline B (Cached) | Lat: {rec_b['latency']}s | Hit: {rec_b['recall_hit']} | Refusal: {rec_b['false_refusal']}")
        else:
            print("  -> Running Pipeline B (Agentic k=6)...")
            out_b = run_pipeline_b(q)
            eval_b = evaluate_output(q, gt, out_b["retrieved_docs"], out_b["answer"], is_adversarial=False)
            rec_b = {
                "id": qid,
                "category": "factual",
                "question": q,
                "ground_truth": gt,
                "pipeline": "agentic_langgraph_k6",
                "latency": out_b["latency"],
                "rewritten_query": out_b["rewritten_query"],
                "answer": out_b["answer"],
                "retrieved_docs_count": len(out_b["retrieved_docs"]),
                "recall_hit": eval_b["recall_hit"],
                "false_refusal": eval_b["is_refusal"],
                "is_hallucination": eval_b["is_hallucination"],
                "judge_reasoning": eval_b["reasoning"],
                "retrieved_docs": out_b["retrieved_docs"]
            }
            checkpoint.setdefault("agentic", {})[qid] = rec_b
            with open(CACHE_FILE, "w", encoding="utf-8") as cf:
                json.dump(checkpoint, cf, indent=2)
            print(f"     [Agent] Lat: {out_b['latency']}s | Hit: {eval_b['recall_hit']} | Refusal: {eval_b['is_refusal']} | Rewritten: \"{out_b['rewritten_query']}\"")
            time.sleep(1.2)
        results_agentic.append(rec_b)

    # --------------------------------------------------------------------------
    # Phase 2: Evaluate Adversarial Queries (15 items)
    # --------------------------------------------------------------------------
    print("\n>>> PHASE 2: EVALUATING 15 ADVERSARIAL / NEGATIVE CONTROL QUERIES (A01-A15) <<<")

    for i, item in enumerate(ADVERSARIAL_QUERIES, 1):
        qid = item["id"]
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\n[{i}/15] [{qid}] Q: {q}")

        # 1. Pipeline A: Baseline
        if qid in checkpoint.get("baseline", {}):
            rec_a = checkpoint["baseline"][qid]
            print(f"  -> Pipeline A (Cached) | Lat: {rec_a['latency']}s | Refused: {rec_a['adversarial_interception']} | Hallucinated: {rec_a['is_hallucination']}")
        else:
            print("  -> Running Pipeline A (Baseline k=3)...")
            out_a = run_pipeline_a(q)
            eval_a = evaluate_output(q, gt, out_a["retrieved_docs"], out_a["answer"], is_adversarial=True)
            rec_a = {
                "id": qid,
                "category": "adversarial",
                "question": q,
                "ground_truth": gt,
                "pipeline": "baseline_k3",
                "latency": out_a["latency"],
                "answer": out_a["answer"],
                "adversarial_interception": eval_a["is_refusal"],
                "is_hallucination": eval_a["is_hallucination"],
                "judge_reasoning": eval_a["reasoning"]
            }
            checkpoint.setdefault("baseline", {})[qid] = rec_a
            with open(CACHE_FILE, "w", encoding="utf-8") as cf:
                json.dump(checkpoint, cf, indent=2)
            print(f"     [Base] Lat: {out_a['latency']}s | Refused: {eval_a['is_refusal']} | Hallucinated: {eval_a['is_hallucination']}")
            time.sleep(1.2)
        results_baseline.append(rec_a)

        # 2. Pipeline B: Agentic LangGraph
        if qid in checkpoint.get("agentic", {}):
            rec_b = checkpoint["agentic"][qid]
            print(f"  -> Pipeline B (Cached) | Lat: {rec_b['latency']}s | Refused: {rec_b['adversarial_interception']} | Hallucinated: {rec_b['is_hallucination']}")
        else:
            print("  -> Running Pipeline B (Agentic k=6)...")
            out_b = run_pipeline_b(q)
            eval_b = evaluate_output(q, gt, out_b["retrieved_docs"], out_b["answer"], is_adversarial=True)
            rec_b = {
                "id": qid,
                "category": "adversarial",
                "question": q,
                "ground_truth": gt,
                "pipeline": "agentic_langgraph_k6",
                "latency": out_b["latency"],
                "rewritten_query": out_b["rewritten_query"],
                "answer": out_b["answer"],
                "adversarial_interception": eval_b["is_refusal"],
                "is_hallucination": eval_b["is_hallucination"],
                "judge_reasoning": eval_b["reasoning"]
            }
            checkpoint.setdefault("agentic", {})[qid] = rec_b
            with open(CACHE_FILE, "w", encoding="utf-8") as cf:
                json.dump(checkpoint, cf, indent=2)
            print(f"     [Agent] Lat: {out_b['latency']}s | Refused: {eval_b['is_refusal']} | Hallucinated: {eval_b['is_hallucination']}")
            time.sleep(1.2)
        results_agentic.append(rec_b)

    # --------------------------------------------------------------------------
    # Phase 3: Error Analysis on Misses (Pipeline B)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> PHASE 3: ERROR ANALYSIS ON MISSES (PIPELINE B) <<<")
    print("=" * 80)

    factual_agentic = [r for r in results_agentic if r["category"] == "factual"]
    failed_items_b = [r for r in factual_agentic if r["recall_hit"] == 0 or r["false_refusal"] == 1]
    
    total_misses_b = len(failed_items_b)
    table_severance_count = 0
    miss_diagnostics: List[Dict[str, Any]] = []

    print(f"Total Factual Failures in Pipeline B: {total_misses_b} of 28")

    for miss in failed_items_b:
        qid = miss["id"]
        q = miss["question"]
        gt = miss["ground_truth"]
        docs = miss.get("retrieved_docs", [])
        
        is_severance, explanation = analyze_table_severance_cause(q, gt, docs)
        if is_severance:
            table_severance_count += 1

        diag = {
            "id": qid,
            "question": q,
            "ground_truth": gt,
            "recall_hit": miss["recall_hit"],
            "false_refusal": miss["false_refusal"],
            "table_severance_detected": is_severance,
            "diagnostic_explanation": explanation
        }
        miss_diagnostics.append(diag)
        print(f" - [{qid}] Table Severance: {is_severance} | {explanation[:120]}...")
        time.sleep(1.0)

    table_severance_pct = (table_severance_count / total_misses_b * 100.0) if total_misses_b > 0 else 0.0

    # --------------------------------------------------------------------------
    # Phase 4: Compute Top-Level Aggregates
    # --------------------------------------------------------------------------
    factual_base = [r for r in results_baseline if r["category"] == "factual"]
    adv_base = [r for r in results_baseline if r["category"] == "adversarial"]
    adv_agentic = [r for r in results_agentic if r["category"] == "adversarial"]

    # Factual Recall@k
    recall_hits_base = sum(r["recall_hit"] for r in factual_base)
    recall_hits_agent = sum(r["recall_hit"] for r in factual_agentic)

    # False Refusals on Factual
    false_refusals_base = sum(r["false_refusal"] for r in factual_base)
    false_refusals_agent = sum(r["false_refusal"] for r in factual_agentic)

    # Adversarial Interceptions (Refusals)
    adv_refusals_base = sum(r["adversarial_interception"] for r in adv_base)
    adv_refusals_agent = sum(r["adversarial_interception"] for r in adv_agentic)

    # Hallucinations on Adversarial
    hallucinations_base = sum(r["is_hallucination"] for r in adv_base)
    hallucinations_agent = sum(r["is_hallucination"] for r in adv_agentic)

    # Mean Latencies across full suite (28 factual + 15 adversarial = 43 queries)
    avg_latency_base = round(sum(r["latency"] for r in results_baseline) / len(results_baseline), 2)
    avg_latency_agent = round(sum(r["latency"] for r in results_agentic) / len(results_agentic), 2)

    # --------------------------------------------------------------------------
    # Phase 5: Persist eval_comparison.json
    # --------------------------------------------------------------------------
    comparison_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_factual_queries": len(FACTUAL_QUERIES),
            "total_adversarial_queries": len(ADVERSARIAL_QUERIES),
            "total_evaluated_queries": len(FACTUAL_QUERIES) + len(ADVERSARIAL_QUERIES),
            "baseline_configuration": "Naive dense retrieval, k=3, direct answer",
            "agentic_configuration": "LangGraph state graph, query rewriting, k=6, grader node, hallucination check"
        },
        "aggregate_metrics": {
            "recall_factual": {
                "baseline_hits": recall_hits_base,
                "baseline_total": len(FACTUAL_QUERIES),
                "baseline_pct": round(recall_hits_base / len(FACTUAL_QUERIES) * 100, 1),
                "agentic_hits": recall_hits_agent,
                "agentic_total": len(FACTUAL_QUERIES),
                "agentic_pct": round(recall_hits_agent / len(FACTUAL_QUERIES) * 100, 1),
                "delta_hits": recall_hits_agent - recall_hits_base
            },
            "false_refusal_rate": {
                "baseline_count": false_refusals_base,
                "baseline_total": len(FACTUAL_QUERIES),
                "baseline_pct": round(false_refusals_base / len(FACTUAL_QUERIES) * 100, 1),
                "agentic_count": false_refusals_agent,
                "agentic_total": len(FACTUAL_QUERIES),
                "agentic_pct": round(false_refusals_agent / len(FACTUAL_QUERIES) * 100, 1),
                "count_reduction": false_refusals_base - false_refusals_agent
            },
            "adversarial_interceptions": {
                "baseline_count": adv_refusals_base,
                "baseline_total": len(ADVERSARIAL_QUERIES),
                "baseline_pct": round(adv_refusals_base / len(ADVERSARIAL_QUERIES) * 100, 1),
                "agentic_count": adv_refusals_agent,
                "agentic_total": len(ADVERSARIAL_QUERIES),
                "agentic_pct": round(adv_refusals_agent / len(ADVERSARIAL_QUERIES) * 100, 1),
                "delta": adv_refusals_agent - adv_refusals_base
            },
            "hallucinations": {
                "baseline_count": hallucinations_base,
                "baseline_total": len(ADVERSARIAL_QUERIES),
                "baseline_pct": round(hallucinations_base / len(ADVERSARIAL_QUERIES) * 100, 1),
                "agentic_count": hallucinations_agent,
                "agentic_total": len(ADVERSARIAL_QUERIES),
                "agentic_pct": round(hallucinations_agent / len(ADVERSARIAL_QUERIES) * 100, 1),
                "reduction": hallucinations_base - hallucinations_agent
            },
            "latency_seconds": {
                "baseline_mean": avg_latency_base,
                "agentic_mean": avg_latency_agent,
                "delta": round(avg_latency_agent - avg_latency_base, 2)
            },
            "table_severance_error_analysis": {
                "total_misses_agentic": total_misses_b,
                "table_severance_misses": table_severance_count,
                "table_severance_pct": round(table_severance_pct, 1),
                "miss_diagnostics": miss_diagnostics
            }
        },
        "query_details": {
            "baseline": results_baseline,
            "agentic_langgraph": results_agentic
        }
    }

    output_file = "eval_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"COMPLETE AUDIT LOG PERSISTED TO: {os.path.abspath(output_file)}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Phase 6: Markdown Summary Table Output
    # --------------------------------------------------------------------------
    rec_b_pct = f"{recall_hits_base / len(FACTUAL_QUERIES) * 100:.1f}%"
    rec_ag_pct = f"{recall_hits_agent / len(FACTUAL_QUERIES) * 100:.1f}%"
    delta_hits_str = f"{'+' if recall_hits_agent >= recall_hits_base else ''}{recall_hits_agent - recall_hits_base} hits"

    fr_b_pct = f"{false_refusals_base / len(FACTUAL_QUERIES) * 100:.1f}%"
    fr_ag_pct = f"{false_refusals_agent / len(FACTUAL_QUERIES) * 100:.1f}%"
    fr_red_str = f"{false_refusals_base - false_refusals_agent} count reduction"

    adv_b_pct = f"{adv_refusals_base / len(ADVERSARIAL_QUERIES) * 100:.1f}%"
    adv_ag_pct = f"{adv_refusals_agent / len(ADVERSARIAL_QUERIES) * 100:.1f}%"
    adv_delta_str = f"{adv_refusals_agent - adv_refusals_base:+d} count"

    hal_b_pct = f"{hallucinations_base / len(ADVERSARIAL_QUERIES) * 100:.1f}%"
    hal_ag_pct = f"{hallucinations_agent / len(ADVERSARIAL_QUERIES) * 100:.1f}%"
    hal_red_str = f"{hallucinations_base - hallucinations_agent:+d} delta"

    lat_delta_str = f"{'+' if avg_latency_agent >= avg_latency_base else ''}{avg_latency_agent - avg_latency_base:.1f}s"
    sev_str = f"{table_severance_count} of {total_misses_b} misses ({table_severance_pct:.1f}%)" if total_misses_b > 0 else "0 misses (N/A)"

    table_md = f"""
| Metric | Baseline (k=3) | Agentic LangGraph (k=6) | Delta / Notes |
| :--- | :--- | :--- | :--- |
| Recall (Factual) | {recall_hits_base} / {len(FACTUAL_QUERIES)} ({rec_b_pct}) | {recall_hits_agent} / {len(FACTUAL_QUERIES)} ({rec_ag_pct}) | {delta_hits_str} |
| False Refusal Rate | {false_refusals_base} / {len(FACTUAL_QUERIES)} ({fr_b_pct}) | {false_refusals_agent} / {len(FACTUAL_QUERIES)} ({fr_ag_pct}) | {fr_red_str} |
| Adversarial Refusals | {adv_refusals_base} / {len(ADVERSARIAL_QUERIES)} ({adv_b_pct}) | {adv_refusals_agent} / {len(ADVERSARIAL_QUERIES)} ({adv_ag_pct}) | {adv_delta_str} |
| Hallucinations | {hallucinations_base} / {len(ADVERSARIAL_QUERIES)} ({hal_b_pct}) | {hallucinations_agent} / {len(ADVERSARIAL_QUERIES)} ({hal_ag_pct}) | {hal_red_str} |
| Avg Latency (sec) | {avg_latency_base:.1f}s | {avg_latency_agent:.1f}s | {lat_delta_str} |
| Table Boundary Misses | - | {sev_str} | Failure root cause |
"""
    print("\n" + table_md)
    return comparison_data

if __name__ == "__main__":
    run_evaluation_suite()
