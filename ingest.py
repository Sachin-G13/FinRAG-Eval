import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma


def ingest_documents(data_dir: str = "./data", persist_dir: str = "./chroma_db"):
    print(f"Scanning for PDF documents in '{data_dir}'...")
    pdf_files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in directory '{data_dir}'.")

    print(f"Found {len(pdf_files)} PDF file(s):")
    for f in pdf_files:
        print(f" - {os.path.basename(f)}")

    all_docs = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\nLoading '{filename}' with PyPDFLoader...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Attach file metadata: source filename and 1-indexed page number
        for doc in docs:
            # PyPDFLoader assigns a 0-indexed integer to 'page'
            page_index = doc.metadata.get("page", 0)
            page_number = page_index + 1 if isinstance(page_index, int) else page_index

            doc.metadata["source"] = filename
            doc.metadata["file_name"] = filename
            doc.metadata["page"] = page_number
            doc.metadata["page_number"] = page_number

        print(f"Loaded {len(docs)} page(s) from '{filename}'.")
        all_docs.extend(docs)

    print(f"\nTotal loaded pages across all documents: {len(all_docs)}")

    # Chunk the documents using RecursiveCharacterTextSplitter
    print("\nChunking documents with RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=150)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Generated {len(chunks)} text chunks.")

    # Initialize FastEmbed embeddings
    model_name = "BAAI/bge-small-en-v1.5"
    print(f"\nInitializing FastEmbedEmbeddings with model '{model_name}'...")
    embeddings = FastEmbedEmbeddings(model_name=model_name)

    # Embed and persist into ChromaDB
    print(f"\nEmbedding chunks and persisting to ChromaDB at '{persist_dir}'...")
    # Chroma handles batching automatically or in batches
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    # Verify collection count
    collection_count = vectorstore._collection.count()
    print(f"\nSuccessfully stored {collection_count} vectors in ChromaDB collection!")
    print("Ingestion completed successfully.")
    return vectorstore


if __name__ == "__main__":
    ingest_documents()
