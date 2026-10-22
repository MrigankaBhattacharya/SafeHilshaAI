"""
SafeHilsa AI - Phase 2 Retrieval & Vector Store Verification Suite
Validates the Version 2 ChromaDB index and Hybrid RAG Engine.

Covers:
1. Exact Rule-Number Matching (Rule 46(4), Rule 46(3), Rule 50(2)).
2. West Bengal vs Bangladesh Regulation Separation (23cm vs 25cm).
3. Samudra Sathi Livelihood Scheme (₹5,000/mo cash DBT).
4. Source Citations & Provenance Traceability.
5. Duplicate Handling (SRC-REG-001 vs SRC-REG-011).
6. River Basin Disambiguation (Subarnarekha vs Hooghly-Bhagirathi).
7. Insufficient Evidence Detection.
"""

import os
import sys
import unittest
import io

# Ensure UTF-8 output handling on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure app directory is in path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from rag_engine import (
    initialize_rag_v2,
    search_hybrid_v2,
    format_retrieval_response,
    load_verified_chunks,
    COLLECTION_NAME_V2,
    DEFAULT_CHROMA_DIR_V2,
    DEFAULT_PROCESSED_CHUNKS
)


class TestSafeHilsaRetrievalV2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initializes RAG V2 once for all test methods."""
        cls.col = initialize_rag_v2()
        cls.docs, cls.metas, cls.ids = load_verified_chunks()

    def test_01_exact_rule_number_matching(self):
        """Rule 46(4) must return Gazette 719-Fish citing 23 cm minimum legal size."""
        query = "What is the minimum legal catch size for Hilsa under Rule 46(4)?"
        res = search_hybrid_v2(query, top_k=3)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreaterEqual(len(res["results"]), 1)

        top_match = res["results"][0]
        self.assertEqual(top_match["metadata"]["source_id"], "SRC-REG-001")
        self.assertIn("West Bengal", top_match["metadata"]["jurisdiction"])

        content = top_match["document"]
        self.assertIn("23", content)
        combined_text = (top_match["metadata"]["legal_headings"] + " " + content).lower()
        self.assertIn("rule 46", combined_text)
        print(f"\n[PASS] Test 1: Matched Rule 46(4) -> {top_match['metadata']['source_id']} ({top_match['metadata']['title'][:50]})")

    def test_02_sanctuary_ban_periods_rule_50(self):
        """Rule 50(2) must return sanctuary ban windows: June-Aug and Oct-Dec."""
        query = "What are the sanctuary closed ban windows under Rule 50(2)?"
        res = search_hybrid_v2(query, top_k=3)

        self.assertEqual(res["status"], "SUCCESS")
        top_match = res["results"][0]
        self.assertIn(top_match["metadata"]["source_id"], ["SRC-REG-001", "SRC-REG-010"])

        content = top_match["document"].lower()
        self.assertTrue("june to august" in content or "june-aug" in content)
        self.assertTrue("october to december" in content or "oct-dec" in content)
        print(f"[PASS] Test 2: Matched Rule 50(2) -> {top_match['metadata']['source_id']} (June-Aug & Oct-Dec)")

    def test_03_west_bengal_vs_bangladesh_separation(self):
        """West Bengal legal query must NOT return Bangladesh regulations as top hit."""
        # Query 3A: West Bengal size
        res_wb = search_hybrid_v2("What is the legal catch size for Hilsa in West Bengal?", top_k=3, jurisdiction="West Bengal, India")
        self.assertEqual(res_wb["status"], "SUCCESS")
        top_wb = res_wb["results"][0]
        self.assertIn("West Bengal", top_wb["metadata"]["jurisdiction"])
        self.assertEqual(top_wb["metadata"]["source_id"], "SRC-REG-001", "Must cite primary Gazette source SRC-REG-001, not a secondary research report")
        self.assertIn("23", top_wb["document"])

        # Query 3B: Bangladesh size
        res_bgd = search_hybrid_v2("What is the legal definition of Jatka size in Bangladesh?", top_k=3, jurisdiction="Bangladesh")
        self.assertEqual(res_bgd["status"], "SUCCESS")
        top_bgd = res_bgd["results"][0]
        self.assertIn("Bangladesh", top_bgd["metadata"]["jurisdiction"])
        self.assertIn("25", top_bgd["document"])
        print(f"[PASS] Test 3: Disambiguated jurisdictions -> WB: {top_wb['metadata']['source_id']} (23cm) vs BGD: {top_bgd['metadata']['source_id']} (25cm)")

    def test_04_samudra_sathi_livelihood_scheme(self):
        """Query about ban livelihood compensation must return Samudra Sathi (Rs 5,000/mo cash DBT)."""
        query = "What financial assistance is given to fishermen under Samudra Sathi scheme during the fishing ban?"
        res = search_hybrid_v2(query, top_k=2)

        self.assertEqual(res["status"], "SUCCESS")
        top_match = res["results"][0]
        self.assertEqual(top_match["metadata"]["source_id"], "SRC-REG-027")

        content = top_match["document"]
        self.assertIn("5000", content)
        self.assertIn("Samudra Sathi", content)
        self.assertIn("15th April to 14th June", content)
        print(f"[PASS] Test 4: Matched Samudra Sathi -> {top_match['metadata']['source_id']} (Rs 5,000/mo DBT, 15 Apr - 14 Jun)")

    def test_05_source_citations_traceability(self):
        """Retrieved formatted response must contain structured citation headers."""
        query = "What mesh size is banned for monofilament current jal in West Bengal?"
        res = search_hybrid_v2(query, top_k=2)
        formatted = format_retrieval_response(res)

        self.assertIn("CITATION 1", formatted)
        self.assertIn("Jurisdiction:", formatted)
        self.assertIn("Authority:", formatted)
        self.assertIn("90", formatted)  # 90mm mesh rule
        print("[PASS] Test 5: Formatted citations contain full provenance and authority metadata")

    def test_06_duplicate_handling_src_reg_011(self):
        """SRC-REG-011 must NOT be indexed as a duplicate vector; must be aliased to SRC-REG-001."""
        indexed_sids = [m["source_id"] for m in self.metas]
        self.assertNotIn("SRC-REG-011", indexed_sids, "SRC-REG-011 should not be indexed as a separate vector")

        # Verify alias is recorded in SRC-REG-001 metadata
        src_001_meta = next(m for m in self.metas if m["source_id"] == "SRC-REG-001")
        self.assertEqual(src_001_meta["is_duplicate_alias"], "SRC-REG-011")
        print("[PASS] Test 6: Deduplication verified -> SRC-REG-011 aliased without duplicate vectors")

    def test_07_river_basin_disambiguation(self):
        """Subarnarekha discharge data must NOT be conflated with Hooghly-Bhagirathi hydrology."""
        # Query Subarnarekha
        res_sub = search_hybrid_v2("What is the river discharge at Fekoghat on the Subarnarekha River?", top_k=2)
        self.assertEqual(res_sub["status"], "SUCCESS")
        top_sub = res_sub["results"][0]
        self.assertEqual(top_sub["metadata"]["basin"], "Subernarekha")
        self.assertEqual(top_sub["metadata"]["source_id"], "SRC-REG-013")

        # Query Hooghly
        res_hooghly = search_hybrid_v2("What is the river hydrology and dissolved oxygen at Farakka on the Hooghly-Bhagirathi?", top_k=2)
        self.assertEqual(res_hooghly["status"], "SUCCESS")
        top_hooghly = res_hooghly["results"][0]
        self.assertIn(top_hooghly["metadata"]["source_id"], ["SRC-REG-019", "SRC-REG-020"])
        self.assertEqual(top_hooghly["metadata"]["basin"], "Ganges-Bhagirathi-Hooghly")
        print(f"[PASS] Test 7: Disambiguated river basins -> Subarnarekha: {top_sub['metadata']['source_id']} vs Hooghly: {top_hooghly['metadata']['source_id']}")

    def test_08_insufficient_evidence_handling(self):
        """Unrelated non-fisheries query must return INSUFFICIENT_EVIDENCE status."""
        query = "What is the quantum computing algorithm for interstellar gravitational propulsion?"
        res = search_hybrid_v2(query, min_score_threshold=0.03)

        self.assertEqual(res["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(res["results_count"], 0)
        self.assertIn("Insufficient evidence", res["message"])
        print("[PASS] Test 8: Insufficient evidence handled cleanly without hallucination")


if __name__ == "__main__":
    unittest.main(verbosity=2)
