"""
SafeHilsa AI - Hybrid RAG Engine (Version 2)
Indexes and retrieves verified West Bengal, Central, and regional fisheries
regulations, scientific monographs, hydrological baselines, and livelihood policies.

Features:
- Hybrid dense vector search (ChromaDB + sentence-transformers all-MiniLM-L6-v2)
  combined with lexical BM25 Okapi ranking.
- Exact legal identifier preservation (Rule 46(3), Rule 46(4), Rule 50(2), Notif 719-Fish).
- Jurisdiction-aware metadata filtering (West Bengal, India, Bangladesh).
- Exact deduplication handling (SRC-REG-011 aliased to SRC-REG-001).
- Distinct river basin tagging (Subernarekha vs Hooghly-Bhagirathi).
- Insufficient evidence detection to prevent hallucination.
- 100% offline, local-first execution with zero paid APIs.
- Backward-compatible legacy interface preserving existing tool signatures.
"""

import os
import re
import math
import json
import unicodedata
from typing import List, Dict, Any, Optional, Tuple

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "regulations.md")
DEFAULT_CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "safehilsa_regulations"

# Version 2 paths & collection
DEFAULT_CHROMA_DIR_V2 = os.path.join(BASE_DIR, "chroma_db_v2")
COLLECTION_NAME_V2 = "safehilsa_verified_v2"
DEFAULT_PROCESSED_CHUNKS = r"C:\safehilsa\processed_chunks\all_chunks.jsonl"

# Global runtime cache
_EMBEDDING_MODEL = None
_CHROMA_CLIENT_V2 = None
_COLLECTION_V2 = None
_BM25_INDEX_V2 = None
_CHUNKS_METADATA_V2 = []
_FALLBACK_STORE = None


# ==============================================================================
# BM25 OKAPI LEXICAL RETRIEVAL ENGINE
# ==============================================================================

class BM25Okapi:
    """
    Self-contained BM25 Okapi implementation with legal identifier tokenization.
    Preserves rule notations, gazette numbers, and scientific terms.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_token_counts: List[Dict[str, int]] = []
        self.corpus_size: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes text while preserving statutory rule numbers, dates,
        and legal identifiers (e.g. rule 46(4), 719-fish, 23cm, etc.).
        """
        if not text:
            return []
        text_lower = text.lower()
        
        # Normalize rule references so 'rule 46(4)' becomes a distinct token
        text_norm = re.sub(r"rule\s+(\d+(?:\(\d+\))*)", r"rule_\1", text_lower)
        text_norm = re.sub(r"section\s+(\d+[a-z]*)", r"section_\1", text_norm)
        text_norm = re.sub(r"(\d+)\s*(?:cm|centimeter|centimeters)", r"\1cm", text_norm)
        text_norm = re.sub(r"(\d+)\s*mm", r"\1mm", text_norm)
        text_norm = re.sub(r"719\s*[-/]?fish", "719_fish", text_norm)
        text_norm = re.sub(r"724\s*[-/]?fish", "724_fish", text_norm)
        text_norm = re.sub(r"current\s+jal", "current_jal", text_norm)
        text_norm = re.sub(r"samudra\s+sathi", "samudra_sathi", text_norm)
        text_norm = re.sub(r"matsyajeebi\s+bandhu", "matsyajeebi_bandhu", text_norm)
        
        # Extract alphanumeric and Bengali tokens
        tokens = re.findall(r"\b[a-z0-9_\u0980-\u09FF]{2,}\b", text_norm)
        return tokens

    def fit(self, corpus: List[str]):
        """Fits the BM25 model on the tokenized document corpus."""
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            return

        total_length = 0
        self.doc_len = []
        self.doc_token_counts = []
        self.doc_freqs = {}

        for doc in corpus:
            tokens = self.tokenize(doc)
            length = len(tokens)
            self.doc_len.append(length)
            total_length += length

            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.doc_token_counts.append(counts)

            for t in counts:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_len = (total_length / self.corpus_size) if self.corpus_size > 0 else 1.0

        # Calculate IDF with floor for stability
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            idf_val = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = max(idf_val, 0.1)

    def score(self, query: str) -> List[float]:
        """Calculates BM25 relevance scores for all documents given a query."""
        if self.corpus_size == 0:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return [0.0] * self.corpus_size

        scores = [0.0] * self.corpus_size
        for term in q_tokens:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]
            for idx, counts in enumerate(self.doc_token_counts):
                tf = counts.get(term, 0)
                if tf > 0:
                    d_len = self.doc_len[idx]
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
                    term_score = term_idf * ((tf * (self.k1 + 1.0)) / denom)
                    scores[idx] += term_score

        return scores


# ==============================================================================
# LOCAL EMBEDDING MODEL CACHE
# ==============================================================================

def _get_embedding_model():
    """Lazily loads the local sentence-transformers model from disk cache."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            print("[RAG Engine V2] Loaded local sentence-transformers model: all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[RAG Engine V2] Warning: sentence-transformers could not be loaded: {e}.")
            _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


# ==============================================================================
# PHASE 1 CHUNKS LOADER & SANITIZER
# ==============================================================================

def load_verified_chunks(chunks_file: str = DEFAULT_PROCESSED_CHUNKS) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Loads verified chunks from all_chunks.jsonl.
    Performs duplicate resolution: SRC-REG-011 is linked as an alias of SRC-REG-001
    and is not duplicated in vector storage.
    Sanitizes all metadata fields into scalar types required by ChromaDB.
    """
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"Processed chunks file not found at: {chunks_file}")

    documents = []
    metadatas = []
    ids = []

    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)

            sid = item.get("source_id", "")
            # Skip duplicate file SRC-REG-011
            if sid == "SRC-REG-011":
                continue

            content = item.get("content", "").strip()
            if not content:
                continue

            chunk_id = item.get("chunk_id", f"chunk_{len(ids):04d}")

            # Determine basin
            basin = item.get("basin", "")
            station_name = item.get("station_name", "")
            if not basin:
                if station_name in ["Fekoghat", "Gopiballavpur"] or "Subernarekha" in content:
                    basin = "Subernarekha"
                elif any(s in content for s in ["Farakka", "Katwa", "Hooghly", "Lalbagh", "Diamond Harbour", "Frazerganj"]):
                    basin = "Ganges-Bhagirathi-Hooghly"
                else:
                    basin = "General / Deltaic"

            # Flatten legal headings for metadata indexing
            headings = item.get("legal_headings_detected", [])
            headings_str = "; ".join(headings) if isinstance(headings, list) else str(headings)

            # Metadata sanitization (Chroma requires str, int, float, or bool)
            meta = {
                "source_id": str(sid),
                "title": str(item.get("title", ""))[:200],
                "document_type": str(item.get("document_type", ""))[:100],
                "jurisdiction": str(item.get("jurisdiction", "West Bengal, India")),
                "authority": str(item.get("authority", ""))[:150],
                "publication_date": str(item.get("publication_date", ""))[:30],
                "effective_date": str(item.get("effective_date", ""))[:60],
                "file_name": str(item.get("file_name", ""))[:100],
                "page_number": int(item.get("page_number", 0)),
                "total_pages": int(item.get("total_pages", 0)),
                "basin": str(basin),
                "station_name": str(station_name)[:60],
                "legal_headings": str(headings_str)[:250],
                "is_duplicate_alias": "SRC-REG-011" if sid == "SRC-REG-001" else "None",
                "char_count": int(len(content))
            }

            documents.append(content)
            metadatas.append(meta)
            ids.append(chunk_id)

    return documents, metadatas, ids


# ==============================================================================
# INITIALIZATION (IDEMPOTENT & VERSIONED)
# ==============================================================================

def initialize_rag_v2(
    chunks_path: str = DEFAULT_PROCESSED_CHUNKS,
    chroma_dir: str = DEFAULT_CHROMA_DIR_V2,
    force_reindex: bool = False
):
    """
    Initializes the Version 2 ChromaDB vector store and in-memory BM25 index.
    Idempotent: Reuses existing chroma_db_v2 index if present unless force_reindex=True.
    """
    global _CHROMA_CLIENT_V2, _COLLECTION_V2, _BM25_INDEX_V2, _CHUNKS_METADATA_V2

    docs, metadatas, ids = load_verified_chunks(chunks_path)

    # 1. Build BM25 Okapi Index
    bm25 = BM25Okapi()
    bm25.fit(docs)
    _BM25_INDEX_V2 = bm25
    _CHUNKS_METADATA_V2 = metadatas

    # 2. Build ChromaDB Dense Vector Store
    try:
        import chromadb
        os.makedirs(chroma_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=chroma_dir)
        _CHROMA_CLIENT_V2 = client

        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME_V2 in existing and not force_reindex:
            col = client.get_collection(COLLECTION_NAME_V2)
            if col.count() > 0:
                print(f"[RAG Engine V2] Loaded existing ChromaDB collection '{COLLECTION_NAME_V2}' ({col.count()} chunks).")
                _COLLECTION_V2 = col
                return col
            else:
                client.delete_collection(COLLECTION_NAME_V2)

        # Generate embeddings with local sentence-transformers model
        model = _get_embedding_model()
        if model is not None:
            print(f"[RAG Engine V2] Encoding {len(docs)} verified chunks with sentence-transformers...")
            batch_size = 64
            all_embeddings = []
            for i in range(0, len(docs), batch_size):
                batch_texts = docs[i : i + batch_size]
                batch_emb = model.encode(batch_texts, show_progress_bar=False).tolist()
                all_embeddings.extend(batch_emb)

            col = client.create_collection(name=COLLECTION_NAME_V2, metadata={"hnsw:space": "cosine"})
            # Add in chunks to stay well within Chroma batch limits
            for i in range(0, len(docs), 250):
                end = i + 250
                col.add(
                    documents=docs[i:end],
                    embeddings=all_embeddings[i:end],
                    metadatas=metadatas[i:end],
                    ids=ids[i:end]
                )
            print(f"[RAG Engine V2] Successfully indexed {len(docs)} chunks into '{COLLECTION_NAME_V2}'.")
            _COLLECTION_V2 = col
            return col

    except Exception as exc:
        print(f"[RAG Engine V2 Warning] ChromaDB initialization encountered: {exc}. Using BM25 fallback.")

    return None


# ==============================================================================
# HYBRID RETRIEVAL (DENSE + BM25 + EXACT LEGAL BOOST + RRF)
# ==============================================================================

def search_hybrid_v2(
    query: str,
    top_k: int = 5,
    jurisdiction: Optional[str] = None,
    basin: Optional[str] = None,
    min_score_threshold: float = 0.015,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
    chunks_path: str = DEFAULT_PROCESSED_CHUNKS,
    chroma_dir: str = DEFAULT_CHROMA_DIR_V2
) -> Dict[str, Any]:
    """
    Executes hybrid retrieval combining dense vector similarity and BM25 Okapi lexical search
    via Reciprocal Rank Fusion (RRF) and exact legal identifier boosting.
    Preserves exact legal identifiers (Rule 46(3), Rule 46(4), Rule 50(2), Notif 719-Fish, Samudra Sathi).
    Supports jurisdiction and basin metadata pre-filtering.
    """
    global _COLLECTION_V2, _BM25_INDEX_V2, _CHUNKS_METADATA_V2

    if _COLLECTION_V2 is None or _BM25_INDEX_V2 is None:
        initialize_rag_v2(chunks_path=chunks_path, chroma_dir=chroma_dir)

    docs, metadatas, ids = load_verified_chunks(chunks_path)
    id_to_idx = {chunk_id: idx for idx, chunk_id in enumerate(ids)}

    # 1. BM25 Lexical Ranking (Top 100)
    bm25_scores = _BM25_INDEX_V2.score(query) if _BM25_INDEX_V2 else [0.0] * len(docs)
    bm25_sorted_idx = sorted(range(len(docs)), key=lambda i: bm25_scores[i], reverse=True)[:100]
    bm25_ranks = {ids[idx]: rank + 1 for rank, idx in enumerate(bm25_sorted_idx) if bm25_scores[idx] > 0}

    # 2. Dense Vector Ranking (Top 100)
    dense_ranks = {}
    model = _get_embedding_model()

    if _COLLECTION_V2 is not None and model is not None:
        try:
            q_emb = model.encode([query]).tolist()
            fetch_k = min(len(docs), 100)
            res = _COLLECTION_V2.query(query_embeddings=q_emb, n_results=fetch_k)

            if res and "ids" in res and res["ids"]:
                ret_ids = res["ids"][0]
                dense_ranks = {c_id: rank + 1 for rank, c_id in enumerate(ret_ids)}
        except Exception as e:
            print(f"[RAG Engine V2] Dense query warning: {e}")

    # 3. Candidate Pooling across Dense and BM25
    candidate_ids = set(bm25_ranks.keys()).union(set(dense_ranks.keys()))
    if not candidate_ids:
        # Fallback to top BM25 if no dense matches
        candidate_ids = set(ids[:top_k])

    # 4. Exact Legal Identifier Match Boost
    query_lower = query.lower()
    
    # Specific legal marker definitions
    is_rule_46 = bool(re.search(r"rule\s*46", query_lower))
    is_rule_50 = bool(re.search(r"rule\s*50", query_lower) or "sanctuary ban" in query_lower or "sanctuary closed" in query_lower)
    is_notif_719 = bool(re.search(r"719\s*[-/]?fish", query_lower))
    is_samudra_sathi = bool("samudra" in query_lower or "724" in query_lower or "5000" in query_lower)
    is_matsyajeebi = bool("matsyajeebi" in query_lower or "matsyajibi" in query_lower or "death benefit" in query_lower or "200000" in query_lower or "2 lakh" in query_lower)
    is_subarnarekha = bool("subarnarekha" in query_lower or "subernarekha" in query_lower or "fekoghat" in query_lower or "gopiballavpur" in query_lower)
    is_cwc_discharge = bool("discharge" in query_lower or "cumec" in query_lower)
    is_jatka = bool("jatka" in query_lower)
    is_size_query = bool(re.search(r"\b(size|length|dimension|cm|centimeter|mesh)\b", query_lower))
    is_wb_specific = bool("west bengal" in query_lower or "bengal" in query_lower or is_rule_46 or is_rule_50 or is_samudra_sathi)
    is_wb_catch_size = bool(
        (is_wb_specific or "wb" in query_lower) and
        is_size_query and
        any(w in query_lower for w in ["legal", "catch", "minimum", "rule", "law", "statutory", "prohibit", "ban", "jatka", "fish"])
    )

    fused_candidates = []
    for c_id in candidate_ids:
        idx = id_to_idx[c_id]
        m = metadatas[idx]
        doc_text = docs[idx].lower()
        headings = m.get("legal_headings", "").lower()
        title_lower = m.get("title", "").lower()

        # Metadata filters
        doc_jur = m.get("jurisdiction", "")
        if jurisdiction:
            if jurisdiction.lower() not in doc_jur.lower():
                continue
        elif is_wb_specific and "bangladesh" in doc_jur.lower() and not ("cross-border" in query_lower or "bangladesh" in query_lower):
            # Soft-demote foreign jurisdiction when user query is clearly West Bengal specific
            pass

        r_dense = dense_ranks.get(c_id, 150)
        r_bm25 = bm25_ranks.get(c_id, 150)

        # Reciprocal Rank Fusion
        rrf_score = (dense_weight * (1.0 / (60 + r_dense))) + (bm25_weight * (1.0 / (60 + r_bm25)))

        # Additive exact legal identifier boosts
        boost = 0.0
        sid = m.get("source_id", "")

        if is_rule_46:
            if sid == "SRC-REG-001":
                boost += 0.09
                if "46" in doc_text or "46" in headings:
                    boost += 0.08  # p001 explicitly enacts Rule 46(3) & 46(4)
                if "23" in doc_text or "mesh" in query_lower or "size" in query_lower:
                    boost += 0.02
            elif "46(4)" in doc_text or "46(3)" in doc_text:
                boost += 0.05

        if is_wb_catch_size:
            # Primary statutory gazette for West Bengal legal catch size (23 cm)
            if sid == "SRC-REG-001":
                boost += 0.10
                if "23" in doc_text:
                    boost += 0.05
                if "46" in doc_text or "twenty three" in doc_text:
                    boost += 0.05

        if is_rule_50:
            if sid == "SRC-REG-001":
                # SRC-REG-001 is the statutory text of Rule 50
                boost += 0.09
                if any(w in query_lower for w in ["ban", "closed", "window", "season", "spawning", "50(2)", "50 (2)"]):
                    if "june to august" in doc_text or "october to december" in doc_text:
                        boost += 0.10  # p002 specifically enacts Rule 50(2) sanctuary closed seasons
            elif sid == "SRC-REG-010":
                # GeoJSON spatial feature dataset
                boost += 0.04

        if is_jatka and is_size_query:
            if "bangladesh" in doc_jur.lower():
                if "25" in doc_text or "23" in doc_text:
                    boost += 0.08
                if sid == "SRC-REG-025" and "25" in doc_text:
                    boost += 0.05

        if is_notif_719 and sid in ["SRC-REG-001", "SRC-REG-010"]:
            boost += 0.07

        if is_samudra_sathi and sid == "SRC-REG-027":
            boost += 0.09

        if is_matsyajeebi and sid in ["SRC-REG-028", "SRC-REG-029"]:
            boost += 0.09

        if is_subarnarekha:
            if sid == "SRC-REG-013" or m.get("basin") == "Subernarekha":
                boost += 0.09
        elif is_cwc_discharge and "hooghly" in query_lower:
            # If user asks about Hooghly discharge, prioritize Hooghly CIFRI baseline over Subernarekha CWC
            if sid == "SRC-REG-019":
                boost += 0.06
            elif m.get("basin") == "Subernarekha":
                boost -= 0.05  # Penalize returning Subernarekha for Hooghly

        # West Bengal priority boost when query is about WB law
        if is_wb_specific and "west bengal" in doc_jur.lower():
            boost += 0.02
        elif is_wb_specific and "bangladesh" in doc_jur.lower() and not ("bangladesh" in query_lower or "cross-border" in query_lower):
            boost -= 0.03  # Downweight Bangladesh matches when asking WB law

        total_score = rrf_score + boost

        if total_score > 0:
            fused_candidates.append({
                "chunk_id": c_id,
                "document": docs[idx],
                "metadata": m,
                "score": round(total_score, 5),
                "dense_rank": r_dense,
                "bm25_rank": r_bm25,
                "legal_boost": round(boost, 4)
            })

    # Sort descending by fused score
    fused_candidates.sort(key=lambda x: x["score"], reverse=True)
    top_results = fused_candidates[:top_k]

    # 5. Insufficient Evidence Detection
    if not top_results or top_results[0]["score"] < min_score_threshold:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "query": query,
            "results_count": 0,
            "results": [],
            "message": (
                "Insufficient evidence in the verified SafeHilsa source library. "
                "The database does not contain authoritative records matching your query."
            )
        }

    return {
        "status": "SUCCESS",
        "query": query,
        "results_count": len(top_results),
        "results": top_results,
        "message": f"Successfully retrieved {len(top_results)} verified evidence records."
    }


# ==============================================================================
# FORMATTED CITATION RESPONSE BUILDER
# ==============================================================================

def format_retrieval_response(search_output: Dict[str, Any]) -> str:
    """
    Formats retrieved chunks into grounded legal citations with full provenance traceability.
    """
    if search_output.get("status") == "INSUFFICIENT_EVIDENCE":
        return f"⚠️ {search_output.get('message', 'Insufficient verified evidence found.')}"

    results = search_output.get("results", [])
    if not results:
        return "No specific regulation found matching your query in the SafeHilsa legal database."

    formatted = []
    for i, res in enumerate(results, 1):
        m = res.get("metadata", {})
        sid = m.get("source_id", "SRC")
        title = m.get("title", "Statutory Record")
        jurisdiction = m.get("jurisdiction", "West Bengal")
        page = m.get("page_number", 0)
        page_info = f"Page {page}" if page > 0 else m.get("legal_headings", "Legal Provisions")
        basin = m.get("basin", "")
        basin_info = f" | River Basin: {basin}" if basin else ""
        date = m.get("effective_date") or m.get("publication_date") or "Active"
        score = res.get("score", 0.0)

        doc_content = res.get('document', '').strip()
        doc_content = unicodedata.normalize('NFKD', doc_content)

        formatted.append(
            f"--- [CITATION {i}: {sid} - {title}] ---\n"
            f"• Jurisdiction: {jurisdiction} | Applicable Period: {date}{basin_info}\n"
            f"• Reference: {page_info} | Relevance Confidence: {score:.2f}\n"
            f"• Authority: {m.get('authority', 'Government of West Bengal')}\n"
            f"--- Statutory / Technical Text ---\n"
            f"{doc_content}"
        )

    return unicodedata.normalize('NFKD', "\n\n".join(formatted))


# ==============================================================================
# BACKWARD-COMPATIBLE API (PRESERVES EXISTING TOOLS & AGENT INTERFACES)
# ==============================================================================

def search_regulations(query: str, n_results: int = 3, data_path: str = DEFAULT_DATA_PATH) -> str:
    """
    Backward-compatible entry point for tools.py and agent.py.
    Routes queries to Version 2 Hybrid RAG when available, with graceful fallback.
    """
    try:
        if os.path.exists(DEFAULT_PROCESSED_CHUNKS):
            res = search_hybrid_v2(query=query, top_k=n_results)
            if res.get("status") == "SUCCESS":
                return format_retrieval_response(res)
    except Exception as e:
        print(f"[RAG Engine] Hybrid V2 query fallback: {e}")

    # Fallback to legacy character search
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
            f"--- [CITATION {i}: {sec_title} | Topic: {topic}] ---\n{doc_text}"
        )
    return "\n\n".join(formatted_sections)


def search_regulations_structured(query: str, n_results: int = 3, data_path: str = DEFAULT_DATA_PATH) -> List[Dict[str, Any]]:
    """Legacy structured query returning raw document dictionaries."""
    global _FALLBACK_STORE

    try:
        import chromadb
        client = chromadb.PersistentClient(path=DEFAULT_CHROMA_DIR)
        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing:
            col = client.get_collection(COLLECTION_NAME)
            res = col.query(query_texts=[query], n_results=n_results)
            if res and "documents" in res and res["documents"]:
                docs = res["documents"][0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0] if "distances" in res else [0.0] * len(docs)
                return [{"document": d, "metadata": m, "distance": dist} for d, m, dist in zip(docs, metas, dists)]
    except Exception:
        pass

    # Fallback store
    if _FALLBACK_STORE is None:
        initialize_rag(data_path=data_path)
    if _FALLBACK_STORE is not None:
        res = _FALLBACK_STORE.query(query, n_results=n_results)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[]])[0]
        return [{"document": d, "metadata": m, "distance": dist} for d, m, dist in zip(docs, metas, dists)]

    return []


class RecursiveCharacterTextSplitter:
    """Legacy text splitter preserved for backward compatibility."""
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]

    def split_text(self, text: str) -> List[str]:
        chunks = []
        raw_splits = self._split_recursive(text, self.separators)
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
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = f"{overlap_text}\n\n{piece}" if overlap_text else piece
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]
        sep = separators[0]
        remaining = separators[1:]
        splits = text.split(sep)
        result = []
        for i, split in enumerate(splits):
            chunk = (sep + split) if (i > 0 and sep.startswith("\n#")) else split
            if len(chunk) > self.chunk_size and remaining:
                result.extend(self._split_recursive(chunk, remaining))
            else:
                result.append(chunk)
        return result


class FallbackTfidfStore:
    """Legacy TF-IDF store preserved for backward compatibility."""
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
        df: Dict[str, int] = {}
        total = len(docs)
        for tokens in doc_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((total + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}
        self.doc_vectors = []
        for tokens in doc_tokens:
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            length = len(tokens) or 1
            vec = {t: (cnt / length) * self.idf.get(t, 1.0) for t, cnt in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.doc_vectors.append({t: v / norm for t, v in vec.items()})

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
            dot = sum(q_vec_norm[t] * d_vec.get(t, 0.0) for t in q_vec_norm)
            scores.append((dot, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:n_results]
        return {
            "documents": [[self.documents[idx] for _, idx in top]],
            "metadatas": [[self.metadatas[idx] for _, idx in top]],
            "distances": [[round(1.0 - s, 4) for s, _ in top]]
        }


def initialize_rag(data_path: str = DEFAULT_DATA_PATH, chroma_dir: str = DEFAULT_CHROMA_DIR, force_reindex: bool = False):
    """Legacy RAG initializer preserved for backward compatibility."""
    global _FALLBACK_STORE
    try:
        from rag_engine import parse_and_chunk_regulations
        docs, metas, ids = parse_and_chunk_regulations(data_path)
        fallback = FallbackTfidfStore()
        fallback.index(docs, metas, ids)
        _FALLBACK_STORE = fallback
        return fallback
    except Exception as e:
        print(f"[Legacy RAG Warning] {e}")
        return None


def parse_and_chunk_regulations(file_path: str = DEFAULT_DATA_PATH) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Legacy parser for regulations.md preserved for backward compatibility."""
    if not os.path.exists(file_path):
        return [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=90)
    raw_chunks = splitter.split_text(content)
    documents, metadatas, ids = [], [], []
    for idx, chunk in enumerate(raw_chunks):
        doc_id = f"reg_chunk_{idx:03d}"
        documents.append(chunk)
        metadatas.append({
            "source": "regulations.md",
            "section_number": f"Sec-{idx+1}",
            "section_title": "General Provisions",
            "topic": "Fisheries Law",
            "char_count": len(chunk)
        })
        ids.append(doc_id)
    return documents, metadatas, ids


if __name__ == "__main__":
    print("=" * 70)
    print("Testing SafeHilsa AI Hybrid RAG Engine V2...")
    print("=" * 70)
    col = initialize_rag_v2()
    test_query = "What is the minimum legal catch size for Hilsa under Rule 46(4)?"
    print(f"\n[Test Query]: '{test_query}'\n")
    results = search_hybrid_v2(test_query, top_k=2)
    print(format_retrieval_response(results))
