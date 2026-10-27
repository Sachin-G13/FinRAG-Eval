import os
import sys
from typing import List, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv()

# 1. Initialize Vector Store & LLM
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

llm = ChatGroq(model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=0)

# 2. Define State
class GraphState(TypedDict, total=False):
    question: str
    rewritten_query: str
    documents: List[str]
    retrieved_documents: List[str]
    grading_verdicts: List[dict]
    hallucination_grade: str
    generation: str
    loop_count: int

# 3. Pydantic Schemas for Structured Output
class DocumentRelevance(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Whether the document is relevant to the question ('yes' or 'no')"
    )

class HallucinationGrade(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Whether the answer is strictly grounded in the facts ('yes' or 'no')"
    )

# 4. Graph Nodes
def rewrite_query(state: GraphState):
    print("--- [NODE: REWRITE QUERY] Optimizing query for financial retrieval ---")
    question = state["question"]
    prompt = ChatPromptTemplate.from_template(
        "You are an expert financial research assistant. Your task is to transform the user's question into a clean, "
        "highly targeted semantic search query for SEC filings (10-K, 10-Q, and Shareholder Updates).\n\n"
        "Guidelines:\n"
        "- Output a search query containing ONLY the explicit company/entity names, fiscal quarters/years, and specific financial metrics or operational topics asked in the user question.\n"
        "- Do NOT append generic boilerplate keywords like 'revenues, net income, operating profit margins, cash flow, balance sheets' unless explicitly requested.\n"
        "- Respond ONLY with the rewritten search query. Do not include explanation, quotation marks, or preamble.\n\n"
        "User question: {question}\n\n"
        "Targeted search query:"
    )
    chain = prompt | llm
    res = chain.invoke({"question": question})
    rewritten = res.content.strip().strip('"').strip("'")
    print(f"-> Rewritten query: '{rewritten}'")
    return {"rewritten_query": rewritten}

def retrieve(state: GraphState):
    print("--- [NODE: RETRIEVE] Fetching documents ---")
    search_query = state.get("rewritten_query") or state["question"]
    docs = retriever.invoke(search_query)
    contents = [d.page_content for d in docs]
    return {
        "documents": contents,
        "retrieved_documents": contents,
        "loop_count": 0,
        "grading_verdicts": [],
        "hallucination_grade": "",
    }

def safe_structured_grade(chain, inputs, default="yes") -> str:
    try:
        res = chain.invoke(inputs)
        return res.binary_score
    except Exception as e:
        err_str = str(e).lower()
        if "failed_generation" in err_str:
            if "'no" in err_str or '"no' in err_str or "no." in err_str:
                return "no"
            if "'yes" in err_str or '"yes' in err_str or "yes." in err_str:
                return "yes"
        return default

def grade_documents(state: GraphState):
    print("--- [NODE: GRADE DOCUMENTS] Filtering relevant chunks ---")
    grader_llm = llm.with_structured_output(DocumentRelevance)
    
    prompt = ChatPromptTemplate.from_template(
        "Question: {question}\nDocument: {context}\n"
        "Evaluate if the document contains info relevant to answering the question. Output 'yes' or 'no'."
    )
    chain = prompt | grader_llm
    
    filtered_docs = []
    verdicts = []
    for idx, doc in enumerate(state.get("documents", [])):
        score = safe_structured_grade(chain, {"question": state["question"], "context": doc}, default="yes")
        verdicts.append({
            "chunk_id": idx + 1,
            "relevance": score,
            "content": doc,
        })
        if score == "yes":
            filtered_docs.append(doc)
            
    # Fallback: if all chunks get filtered out, retain original context
    if not filtered_docs:
        filtered_docs = state["documents"]
        
    return {"documents": filtered_docs, "grading_verdicts": verdicts}

def generate(state: GraphState):
    print("--- [NODE: GENERATE] Crafting answer ---")
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks on financial filings.\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer strictly based on the context above. If you don't know, state that you don't know."
    )
    chain = prompt | llm
    context_str = "\n\n".join(state["documents"])
    res = chain.invoke({"context": context_str, "question": state["question"]})
    return {"generation": res.content, "loop_count": state.get("loop_count", 0) + 1}

# 5. Hallucination Evaluation Node & Router
def evaluate_hallucination(state: GraphState):
    print("--- [NODE: EVALUATE HALLUCINATION] Fact-checking response ---")
    hallucination_llm = llm.with_structured_output(HallucinationGrade)
    
    prompt = ChatPromptTemplate.from_template(
        "Context Facts:\n{context}\n\n"
        "Generated Answer:\n{generation}\n\n"
        "Determine if the Generated Answer is strictly grounded in and supported by the Context Facts."
    )
    chain = prompt | hallucination_llm
    context_str = "\n\n".join(state["documents"])
    score = safe_structured_grade(chain, {"context": context_str, "generation": state["generation"]}, default="yes")
    return {"hallucination_grade": score}

def check_hallucination(state: GraphState) -> Literal["useful", "regenerate"]:
    score = state.get("hallucination_grade", "yes")
    if score == "yes":
        print("-> Decision: Grounded in source. Ready for output.")
        return "useful"
    
    # Self-correction loop (capped at 2 retries to prevent infinite loops)
    if state.get("loop_count", 0) < 2:
        print("-> Decision: Hallucination flagged! Retrying generation.")
        return "regenerate"
    else:
        print("-> Decision: Max retries reached. Outputting best effort.")
        return "useful"

# 6. Build LangGraph
builder = StateGraph(GraphState)

builder.add_node("rewrite_query", rewrite_query)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_documents", grade_documents)
builder.add_node("generate", generate)
builder.add_node("evaluate_hallucination", evaluate_hallucination)

builder.set_entry_point("rewrite_query")
builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("retrieve", "grade_documents")
builder.add_edge("grade_documents", "generate")
builder.add_edge("generate", "evaluate_hallucination")

builder.add_conditional_edges(
    "evaluate_hallucination",
    check_hallucination,
    {
        "useful": END,
        "regenerate": "generate"
    }
)

app = builder.compile()

# 7. Test Run
if __name__ == "__main__":
    sample_query = "What was Tesla's total automotive revenue or delivery numbers mentioned in the reports?"
    initial_state = {"question": sample_query}
    result = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("QUESTION:", sample_query)
    print("="*50)
    print("FINAL ANSWER:\n", result["generation"])
    print("="*50)