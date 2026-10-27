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

import chromadb
from langchain_groq import ChatGroq

load_dotenv()


def get_chroma_collection(persist_dir: str = "./chroma_db", collection_name: str = "langchain"):
    """Connect to the existing ChromaDB persistent client and retrieve collection."""
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"ChromaDB directory not found at '{persist_dir}'. Run ingest.py first.")
    client = chromadb.PersistentClient(path=persist_dir)
    collections = client.list_collections()
    col_names = [c.name for c in collections]
    if collection_name not in col_names:
        if collections:
            collection_name = collections[0].name
        else:
            raise ValueError(f"No collections found in ChromaDB at '{persist_dir}'.")
    return client.get_collection(collection_name)


def sample_diverse_chunks(collection, total_chunks: int = 10) -> List[Dict[str, Any]]:
    """Sample diverse, high-quality text chunks from Apple and Tesla filings."""
    print("Fetching stored documents and metadata from ChromaDB...")
    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    print(f"Total chunks in vector database: {len(documents)}")

    # Group candidate chunks by filing
    # Filtering criteria: at least 350 chars and containing numbers/financial metrics
    filing_candidates: Dict[str, List[Dict[str, Any]]] = {}
    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "")
        # Require substantive text with digits (financial metrics)
        if len(doc.strip()) >= 350 and any(ch.isdigit() for ch in doc):
            if source not in filing_candidates:
                filing_candidates[source] = []
            filing_candidates[source].append({"text": doc, "metadata": meta})

    for f, cands in filing_candidates.items():
        print(f" - {f}: {len(cands)} candidate quality chunks")

    # Target: 5 Apple chunks, 5 Tesla chunks
    target_apple = total_chunks // 2
    target_tesla = total_chunks - target_apple

    apple_files = [f for f in filing_candidates.keys() if "apple" in f.lower()]
    tesla_files = [f for f in filing_candidates.keys() if "tesla" in f.lower()]

    def pick_from_group(files: List[str], count: int) -> List[Dict[str, Any]]:
        selected = []
        if not files:
            return selected
        
        # Sort chunks within each file by page number to sample diverse sections
        for f in files:
            filing_candidates[f].sort(key=lambda x: x["metadata"].get("page", 0))

        # Stratified sampling across files
        # Prioritize distributing evenly across filings
        file_idx = 0
        while len(selected) < count:
            current_file = files[file_idx % len(files)]
            candidates = filing_candidates.get(current_file, [])
            if candidates:
                # Pick evenly spaced across pages
                step = max(1, len(candidates) // max(1, (count // len(files)) + 1))
                pick_offset = (len(selected) * step * 3) % len(candidates)
                selected.append(candidates.pop(pick_offset))
            file_idx += 1
            # Check if all candidate lists empty
            if all(len(filing_candidates.get(f, [])) == 0 for f in files):
                break
        return selected

    sampled_apple = pick_from_group(apple_files, target_apple)
    sampled_tesla = pick_from_group(tesla_files, target_tesla)

    sampled = sampled_apple + sampled_tesla
    print(f"\nSuccessfully selected {len(sampled)} diverse chunks ({len(sampled_apple)} Apple, {len(sampled_tesla)} Tesla).")
    return sampled


def generate_auditor_qa(llm: ChatGroq, chunk_text: str, metadata: Dict[str, Any]) -> Dict[str, str]:
    """Prompt ChatGroq to act as a financial auditor and generate a question-ground_truth pair."""
    source_file = metadata.get("file_name", metadata.get("source", "SEC Filing"))
    page_num = metadata.get("page_number", metadata.get("page", "N/A"))

    prompt = (
        "You are a senior forensic financial auditor and financial QA dataset engineer evaluating corporate SEC filings.\n"
        f"Analyze the following filing excerpt from {source_file} (Page {page_num}):\n\n"
        "---\n"
        f"{chunk_text}\n"
        "---\n\n"
        "Your task is to generate one high-quality, rigorous evaluation QA pair based strictly and exclusively on the excerpt above.\n\n"
        "Requirements:\n"
        "1. 'question': A challenging, specific financial query that can be answered unambiguously using ONLY facts and figures from this chunk.\n"
        "   - Explicitly mention the company name, relevant fiscal period (e.g. Q1 2026, FY2025, or full year 2024), and exact metric names.\n"
        "   - Do NOT use meta-phrases such as 'according to the text', 'based on the excerpt', or 'in this document'. Make it sound like a real analyst inquiry.\n"
        "2. 'ground_truth': The exact, concise factual answer according to the chunk. Include specific dollar figures, percentages, or units of measure directly supported by the excerpt.\n\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "question": "Specific financial question",\n'
        '  "ground_truth": "Precise factual answer with exact numbers and units"\n'
        "}"
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Parse JSON
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                raise ValueError(f"Could not extract JSON from LLM response: {content}")

        question = parsed.get("question", "").strip()
        ground_truth = parsed.get("ground_truth", "").strip()
        return {"question": question, "ground_truth": ground_truth}

    except Exception as e:
        print(f"Error during LLM QA generation: {e}")
        return {
            "question": f"What financial performance metrics are reported in {source_file} (Page {page_num})?",
            "ground_truth": chunk_text[:200].strip() + "..."
        }


def get_tricky_hallucination_questions() -> List[Dict[str, Any]]:
    """Return 3 intentionally tricky/unanswerable questions to test hallucination resistance."""
    return [
        {
            "question": "What is Apple's official projected net sales forecast and operating margin guidance for the full fiscal year 2028?",
            "ground_truth": "The provided SEC filings do not contain full-year 2028 net sales forecasts or operating margin guidance, as Apple does not publish multi-year forward-looking financial projections in its historical 10-K or quarterly reports.",
            "source": "Tricky/Hallucination Test: Forward-looking multi-year forecast (unanswerable from historical filings)",
            "is_unanswerable": True,
            "category": "hallucination_resistance_forecast"
        },
        {
            "question": "How many Tesla Semi commercial trucks were delivered to PepsiCo specifically in Q3 2023 according to Tesla's Shareholder Updates?",
            "ground_truth": "The provided filings do not disclose customer-specific vehicle delivery numbers (such as commercial delivery counts broken down specifically for PepsiCo in Q3 2023).",
            "source": "Tricky/Hallucination Test: Nonexistent granular customer delivery count (unanswerable from filings)",
            "is_unanswerable": True,
            "category": "hallucination_resistance_nonexistent_metric"
        },
        {
            "question": "What was the total hardware revenue generated by Apple's autonomous electric vehicle ('Project Titan' / 'Apple Car') in fiscal year 2025?",
            "ground_truth": "Apple never launched an autonomous electric vehicle commercially, and no revenue figures for an 'Apple Car' or 'Project Titan' exist in Apple's fiscal year 2025 financial reports.",
            "source": "Tricky/Hallucination Test: Speculative/fictitious product hardware revenue (unanswerable)",
            "is_unanswerable": True,
            "category": "hallucination_resistance_fictitious_product"
        }
    ]


def build_eval_dataset(
    persist_dir: str = "./chroma_db",
    output_file: str = "eval_dataset.json",
    num_sampled: int = 10,
    model_name: str = "openai/gpt-oss-120b"
):
    """Main generation pipeline: samples chunks, invokes auditor LLM, appends tricky tests, and saves JSON."""
    print("=" * 70)
    print("AUTOMATED SYNTHETIC EVALUATION DATASET GENERATOR")
    print("=" * 70)

    # 1. Initialize LLM
    print(f"Initializing ChatGroq LLM (model='{model_name}')...")
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    # 2. Connect to ChromaDB and sample chunks
    collection = get_chroma_collection(persist_dir=persist_dir)
    sampled_chunks = sample_diverse_chunks(collection, total_chunks=num_sampled)

    dataset: List[Dict[str, Any]] = []

    # 3. Generate auditor QA pairs for each chunk
    print("\nGenerating auditor QA pairs from sampled chunks...")
    for idx, item in enumerate(sampled_chunks, 1):
        chunk_text = item["text"]
        metadata = item["metadata"]
        filename = metadata.get("file_name", metadata.get("source", "Unknown"))
        page = metadata.get("page_number", metadata.get("page", "N/A"))

        # Clean preview snippet
        clean_preview = " ".join(chunk_text.split())
        preview_snip = clean_preview[:120] + ("..." if len(clean_preview) > 120 else "")

        print(f"[{idx}/{len(sampled_chunks)}] Generating QA for {filename} (Page {page})...")
        qa = generate_auditor_qa(llm, chunk_text, metadata)

        # Source description according to requirements
        source_description = f"{filename} (Page {page}) | Preview: \"{preview_snip}\""

        dataset_entry = {
            "id": f"eval_factual_{idx:02d}",
            "question": qa["question"],
            "ground_truth": qa["ground_truth"],
            "source": source_description,
            "is_unanswerable": False,
            "category": "factual_chunk_audit",
            "metadata": {
                "file_name": filename,
                "page": page,
                "chunk_length": len(chunk_text)
            }
        }
        dataset.append(dataset_entry)
        print(f"   Q: {qa['question']}")
        print(f"   A: {qa['ground_truth']}")
        time.sleep(0.3)  # Brief polite delay between Groq calls

    # 4. Append 3 tricky / unanswerable hallucination questions
    print("\nAppending 3 intentionally tricky/unanswerable hallucination test queries...")
    tricky_questions = get_tricky_hallucination_questions()
    for t_idx, item in enumerate(tricky_questions, 1):
        dataset_entry = {
            "id": f"eval_tricky_{t_idx:02d}",
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "source": item["source"],
            "is_unanswerable": item["is_unanswerable"],
            "category": item["category"]
        }
        dataset.append(dataset_entry)
        print(f" [{t_idx}/3] Tricky Q: {item['question']}")

    # 5. Save final dataset to disk
    print(f"\nWriting {len(dataset)} evaluation items to '{output_file}'...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"SUCCESS: Evaluation dataset successfully saved to {os.path.abspath(output_file)}")
    print(f"Total test questions: {len(dataset)} (10 grounded + 3 hallucination test cases)")
    print("=" * 70)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation dataset for financial RAG agent.")
    parser.add_argument("--chroma_dir", default="./chroma_db", help="Path to ChromaDB directory")
    parser.add_argument("--output", default="eval_dataset.json", help="Output path for eval dataset JSON")
    parser.add_argument("--num_chunks", type=int, default=10, help="Number of chunks to sample from ChromaDB")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="ChatGroq model name")

    args = parser.parse_args()
    build_eval_dataset(
        persist_dir=args.chroma_dir,
        output_file=args.output,
        num_sampled=args.num_chunks,
        model_name=args.model
    )
