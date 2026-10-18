"""
SafeHilsa AI - Local RAG Engine
Indexes and retrieves West Bengal and regional fisheries regulations using
ChromaDB and sentence-transformers (all-MiniLM-L6-v2) with zero paid APIs.
Includes resilient fallback vector similarity in case of native wheel initialization.
"""

import os
import re
import math
from typing import List, Dict, Any, Optional

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "regulations.md")
DEFAULT_CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "safehilsa_regulations"

# Optional global embedding model cache
_EMBEDDING_MODEL = None
_CHROMA_CLIENT = None
_FALLBACK_STORE = None


class RecursiveCharacterTextSplitter:
    """
    Lightweight, dependency-free recursive character text splitter
    optimized for structured markdown legal regulations.
    """
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]

    def split_text(self, text: str) -> List[str]:
        chunks = []
        raw_splits = self._split_recursive(text, self.separators)
        
        # Combine splits up to chunk_size with overlap
        current_chunk = ""
        for piece in raw_splits:
            piece = piece.strip()
            if not piece:
                continue
            if len(current_chunk) + len(piece) + 1 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{piece}" if current_chunk else piece
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Keep overlap if possible
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = f"{overlap_text}\n\n{piece}" if overlap_text else piece
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]
        sep = separators[0]
        remaining_seps = separators[1:]
        
        splits = text.split(sep)
        result = []
        for i, split in enumerate(splits):
            chunk = (sep + split) if (i > 0 and sep.startswith("\n#")) else split
            if len(chunk) > self.chunk_size and remaining_seps:
                result.extend(self._split_recursive(chunk, remaining_seps))
            else:
                result.append(chunk)
        return result


class FallbackTfidfStore:
    """
    In-memory vector store using TF-IDF + Cosine Similarity.
    Ensures 100% functionality even during first-run package loading.
    """
    def __init__(self):
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\u0980-\u09FF]{3,}\b", text.lower())

    def index(self, docs: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        self.documents = docs
        self.metadatas = metadatas
        self.ids = ids
        doc_tokens = [self._tokenize(d) for d in docs]
        
        # Build Vocab and Document Frequencies
        df: Dict[str, int] = {}
        total_docs = len(docs)
        for tokens in doc_tokens:
            unique_terms = set(tokens)
            for t in unique_terms:
                df[t] = df.get(t, 0) + 1
        
        self.idf = {t: math.log((total_docs + 1) / (count + 1)) + 1.0 for t, count in df.items()}
        
        # Compute Document TF-IDF vectors
        self.doc_vectors = []
        for tokens in doc_tokens:
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            length = len(tokens) or 1
            vec = {t: (count / length) * self.idf.get(t, 1.0) for t, count in tf.items()}
            # Normalize vector
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            norm_vec = {t: v / norm for t, v in vec.items()}
            self.doc_vectors.append(norm_vec)

    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        q_tokens = self._tokenize(query_text)
        if not q_tokens or not self.doc_vectors:
            return {"documents": [self.documents[:n_results]], "metadatas": [self.metadatas[:n_results]], "distances": [[0.5] * min(n_results, len(self.documents))]}
        
        q_tf: Dict[str, float] = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1
        q_len = len(q_tokens)
        q_vec = {t: (cnt / q_len) * self.idf.get(t, 1.0) for t, cnt in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec_norm = {t: v / q_norm for t, v in q_vec.items()}
        
        scores = []
        for i, d_vec in enumerate(self.doc_vectors):
            dot_product = sum(q_vec_norm[t] * d_vec.get(t, 0.0) for t in q_vec_norm)
            scores.append((dot_product, i))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:n_results]
        
        ret_docs = [self.documents[idx] for _, idx in top_k]
        ret_metas = [self.metadatas[idx] for _, idx in top_k]
        ret_distances = [round(1.0 - score, 4) for score, _ in top_k]
        
        return {
            "documents": [ret_docs],
            "metadatas": [ret_metas],
            "distances": [ret_distances]
        }


def _get_embedding_model():
    """Lazily loads the local sentence-transformers model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Using lightweight local CPU-friendly embedding model
            _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[RAG Engine] SentenceTransformer could not be loaded: {e}. Using fallback embedding.")
            _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def parse_and_chunk_regulations(file_path: str = DEFAULT_DATA_PATH) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Parses regulations.md, splits into chunks, and extracts section metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Regulations document not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=90)
    raw_chunks = splitter.split_text(content)

    documents = []
    metadatas = []
    ids = []

    for idx, chunk in enumerate(raw_chunks):
        # Extract section headers if available
        section_match = re.search(r"##\s+(?:(\d+(?:\.\d+)?)\s+)?([^\n]+)", chunk)
        section_name = section_match.group(2).strip() if section_match else "General Provisions"
        section_num = section_match.group(1) if section_match and section_match.group(1) else f"Sec-{idx+1}"

        # Categorize legal topic
        chunk_lower = chunk.lower()
        if "size" in chunk_lower or "25" in chunk_lower or "jatka" in chunk_lower:
            topic = "Minimum Legal Size & Jatka"
        elif "ban" in chunk_lower or "spawning" in chunk_lower or "ashvin" in chunk_lower:
            topic = "Seasonal Ban Windows"
        elif "sanctuary" in chunk_lower or "farakka" in chunk_lower or "katwa" in chunk_lower:
            topic = "Restricted River Sanctuaries"
        elif "penalty" in chunk_lower or "fine" in chunk_lower or "jail" in chunk_lower:
            topic = "Enforcement Penalties & Gear Confiscation"
        elif "alternative" in chunk_lower or "rui" in chunk_lower or "catla" in chunk_lower or "tilapia" in chunk_lower:
            topic = "Alternative Non-Protected Species"
        else:
            topic = "Fisheries Management"

        doc_id = f"reg_chunk_{idx:03d}"
        documents.append(chunk)
        metadatas.append({
            "source": "regulations.md",
            "section_number": section_num,
            "section_title": section_name,
            "topic": topic,
            "char_count": len(chunk)
        })
        ids.append(doc_id)

    return documents, metadatas, ids


def initialize_rag(data_path: str = DEFAULT_DATA_PATH, chroma_dir: str = DEFAULT_CHROMA_DIR, force_reindex: bool = False):
    """
    Initializes the local ChromaDB vector store.
    Embeds data/regulations.md with sentence-transformers.
    Falls back gracefully to FallbackTfidfStore if ChromaDB is unavailable.
    """
    global _CHROMA_CLIENT, _FALLBACK_STORE

    docs, metadatas, ids = parse_and_chunk_regulations(data_path)

    # Attempt ChromaDB initialization
    try:
        import chromadb
        from chromadb.config import Settings
        
        os.makedirs(chroma_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=chroma_dir)
        _CHROMA_CLIENT = client

        # Check existing collection
        existing_collections = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing_collections and not force_reindex:
            col = client.get_collection(COLLECTION_NAME)
            if col.count() > 0:
                print(f"[RAG Engine] Loaded existing ChromaDB collection '{COLLECTION_NAME}' with {col.count()} documents.")
                return col
            else:
                client.delete_collection(COLLECTION_NAME)

        # Generate embeddings via sentence-transformers
        model = _get_embedding_model()
        if model is not None:
            print(f"[RAG Engine] Generating embeddings with sentence-transformers for {len(docs)} chunks...")
            embeddings = model.encode(docs, show_progress_bar=False).tolist()
            collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
            collection.add(
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"[RAG Engine] Successfully indexed {len(docs)} chunks in local ChromaDB.")
            return collection
        else:
            # Native Chroma default embeddings or fallback
            collection = client.create_collection(name=COLLECTION_NAME)
            collection.add(documents=docs, metadatas=metadatas, ids=ids)
            print(f"[RAG Engine] Indexed {len(docs)} chunks in ChromaDB with default embedder.")
            return collection

    except Exception as exc:
        print(f"[RAG Engine Warning] ChromaDB initialization encountered: {exc}. Initializing FallbackTfidfStore.")
        fallback = FallbackTfidfStore()
        fallback.index(docs, metadatas, ids)
        _FALLBACK_STORE = fallback
        return fallback


def search_regulations(query: str, n_results: int = 3, data_path: str = DEFAULT_DATA_PATH) -> str:
    """
    Performs similarity search and returns grounded context along with legal citations.
    """
    results = search_regulations_structured(query, n_results=n_results, data_path=data_path)
    if not results:
        return "No specific regulation found matching your query in the SafeHilsa legal database."

    formatted_sections = []
    for i, res in enumerate(results, 1):
        meta = res.get("metadata", {})
        sec_title = meta.get("section_title", "Official Regulation")
        topic = meta.get("topic", "Fisheries Law")
        doc_text = res.get("document", "").strip()

        formatted_sections.append(
            f"--- [CITATION {i}: {sec_title} | Topic: {topic}] ---\n"
            f"{doc_text}"
        )

    return "\n\n".join(formatted_sections)


def search_regulations_structured(query: str, n_results: int = 3, data_path: str = DEFAULT_DATA_PATH) -> List[Dict[str, Any]]:
    """
    Structured query returning list of matching documents with metadata and distance.
    """
    global _CHROMA_CLIENT, _FALLBACK_STORE

    try:
        import chromadb
        client = _CHROMA_CLIENT or chromadb.PersistentClient(path=DEFAULT_CHROMA_DIR)
        existing = [c.name for c in client.list_collections()]
        
        if COLLECTION_NAME in existing:
            collection = client.get_collection(COLLECTION_NAME)
            model = _get_embedding_model()
            
            if model is not None:
                q_emb = model.encode([query]).tolist()
                chroma_res = collection.query(query_embeddings=q_emb, n_results=n_results)
            else:
                chroma_res = collection.query(query_texts=[query], n_results=n_results)
            
            results = []
            if chroma_res and "documents" in chroma_res and chroma_res["documents"]:
                docs = chroma_res["documents"][0]
                metas = chroma_res.get("metadatas", [[]])[0]
                dists = chroma_res.get("distances", [[]])[0] if "distances" in chroma_res else [0.0] * len(docs)
                
                for d, m, dist in zip(docs, metas, dists):
                    results.append({
                        "document": d,
                        "metadata": m,
                        "distance": dist
                    })
                return results
    except Exception as e:
        print(f"[RAG Engine] Chroma query fallback: {e}")

    # Use or initialize FallbackTfidfStore
    if _FALLBACK_STORE is None:
        initialize_rag(data_path=data_path)
    
    if _FALLBACK_STORE is not None:
        res = _FALLBACK_STORE.query(query, n_results=n_results)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[]])[0]
        return [{"document": d, "metadata": m, "distance": dist} for d, m, dist in zip(docs, metas, dists)]

    return []


if __name__ == "__main__":
    print("Testing SafeHilsa RAG Engine...")
    initialize_rag()
    test_q = "What is the minimum legal size for Hilsa and penalties for catching Jatka?"
    print(f"\nQuery: {test_q}\n")
    print(search_regulations(test_q, n_results=2))
