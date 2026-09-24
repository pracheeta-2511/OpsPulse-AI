"""
rag_store.py - Runbook Vector Store & Retrieval for OpsPulse AI
Uses SentenceTransformer ('all-MiniLM-L6-v2') and FAISS (IndexFlatIP) to index PDF runbooks.
"""

import os
import glob
from typing import List, Dict, Any, Tuple
import numpy as np
import pypdf
import faiss
from sentence_transformers import SentenceTransformer

# Lazy loaded embedding model instance
_EMBEDDING_MODEL = None

def get_embedding_model() -> SentenceTransformer:
    """Singleton loader for sentence transformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def extract_chunks_from_pdf(pdf_path: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF file page by page and creates overlapping chunks with metadata.
    """
    chunks = []
    reader = pypdf.PdfReader(pdf_path)
    file_name = os.path.basename(pdf_path)
    
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text() or ""
        lines = [line.strip() for line in page_text.split("\n") if line.strip()]
        if not lines:
            continue
            
        # Group lines into logical chunks
        current_chunk_lines = []
        current_length = 0
        
        for line in lines:
            current_chunk_lines.append(line)
            current_length += len(line)
            if current_length >= chunk_size:
                chunk_str = " ".join(current_chunk_lines)
                chunks.append({
                    "text": chunk_str,
                    "source": file_name,
                    "page": page_num,
                    "pdf_path": pdf_path
                })
                # Keep overlap
                overlap_lines = current_chunk_lines[-2:] if len(current_chunk_lines) > 2 else []
                current_chunk_lines = overlap_lines
                current_length = sum(len(l) for l in current_chunk_lines)
                
        if current_chunk_lines:
            chunk_str = " ".join(current_chunk_lines)
            chunks.append({
                "text": chunk_str,
                "source": file_name,
                "page": page_num,
                "pdf_path": pdf_path
            })
            
    return chunks


def build_runbook_index(runbooks_dir: str = "runbooks") -> Tuple[faiss.IndexFlatIP, List[str], List[Dict[str, Any]]]:
    """
    Scans the runbooks directory, parses all PDFs, computes embeddings, and builds a FAISS IndexFlatIP index.
    
    Returns:
        (index, chunks_text_list, metadata_list)
    """
    model = get_embedding_model()
    pdf_files = glob.glob(os.path.join(runbooks_dir, "*.pdf"))
    
    all_chunks = []
    for pdf_file in pdf_files:
        pdf_chunks = extract_chunks_from_pdf(pdf_file)
        all_chunks.extend(pdf_chunks)
        
    if not all_chunks:
        # Create empty index with dimension 384
        empty_index = faiss.IndexFlatIP(384)
        return empty_index, [], []
        
    chunk_texts = [c["text"] for c in all_chunks]
    
    # Compute normalized embeddings for cosine similarity with IndexFlatIP
    embeddings = model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index, chunk_texts, all_chunks


def search_runbooks(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Searches the FAISS index for relevant runbook chunks.
    
    Returns:
        List of dicts: [
            {
                "text": str,
                "source": str,
                "page": int,
                "score": float,
                "rank": int
            }
        ]
    """
    if index is None or len(chunks) == 0:
        return []
        
    model = get_embedding_model()
    query_vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    query_vector = np.array(query_vector, dtype=np.float32)
    
    effective_k = min(top_k, len(chunks))
    scores, indices = index.search(query_vector, effective_k)
    
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        meta = metadata[idx]
        results.append({
            "text": chunks[idx],
            "source": meta.get("source", "runbook.pdf"),
            "page": meta.get("page", 1),
            "score": float(score),
            "rank": rank
        })
        
    return results


def format_runbook_context_for_prompt(results: List[Dict[str, Any]]) -> str:
    """Formats retrieved runbook search results into a clean string for LLM prompting."""
    if not results:
        return "No specific runbook procedures found in knowledge base."
        
    formatted = []
    for r in results:
        formatted.append(
            f"[Source: {r['source']} | Page {r['page']} | Relevance: {r['score']:.2f}]\n{r['text']}"
        )
    return "\n\n---\n\n".join(formatted)


if __name__ == "__main__":
    print("Testing rag_store.py...")
    test_index, test_chunks, test_meta = build_runbook_index("runbooks")
    print(f"Indexed {len(test_chunks)} chunks from runbooks.")
    
    test_query = "Postgres connection pool exhausted remaining connection slots"
    matches = search_runbooks(test_query, test_index, test_chunks, test_meta, top_k=2)
    print(f"\nSearch results for '{test_query}':")
    for m in matches:
        print(f"- Rank {m['rank']} (Score: {m['score']:.4f}) [Source: {m['source']}, Page: {m['page']}]:")
        print(f"  {m['text'][:120]}...")
