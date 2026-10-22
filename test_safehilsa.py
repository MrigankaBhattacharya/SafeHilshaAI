"""
SafeHilsa AI - Automated System Verification Suite
Validates geospatial polygons, RAG indexing, deterministic tools,
Ollama agent fallback loop, and syntax integrity.
"""

import os
import sys
import json
import unittest

# Ensure app directory is in path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


class TestSafeHilsaApp(unittest.TestCase):

    def test_01_data_files_exist_and_valid(self):
        """Verify regulations.md and sanctuaries.geojson exist and parse."""
        reg_path = os.path.join(APP_DIR, "data", "regulations.md")
        geo_path = os.path.join(APP_DIR, "data", "sanctuaries.geojson")
        
        self.assertTrue(os.path.exists(reg_path), "regulations.md does not exist")
        self.assertTrue(os.path.exists(geo_path), "sanctuaries.geojson does not exist")
        
        with open(reg_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("25 centimeters", content)
            self.assertIn("Jatka", content)
            self.assertIn("Farakka", content)
            self.assertIn("Rui", content)

        with open(geo_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)
            self.assertEqual(geojson.get("type"), "FeatureCollection")
            features = geojson.get("features", [])
            self.assertGreaterEqual(len(features), 3, "Expected at least 3 sanctuary features")
            for feat in features:
                self.assertIn("name", feat["properties"])
                self.assertEqual(feat["properties"]["status"], "RESTRICTED")
                self.assertEqual(feat["geometry"]["type"], "Polygon")

    def test_02_tools_geospatial_compliance(self):
        """Test Shapely Point-in-Polygon compliance logic."""
        from tools import check_location_compliance
        
        # Point inside Farakka Barrage sanctuary
        res_farakka = check_location_compliance(24.8100, 87.9250)
        self.assertEqual(res_farakka["status"], "RESTRICTED")
        self.assertFalse(res_farakka["is_legal"])
        self.assertIn("Farakka", res_farakka["zone_name"])
        self.assertIn("₹25,000", res_farakka["penalty"])

        # Point inside Katwa-Nabadwip sanctuary
        res_katwa = check_location_compliance(23.5000, 88.2500)
        self.assertEqual(res_katwa["status"], "RESTRICTED")
        self.assertFalse(res_katwa["is_legal"])

        # Point outside (Tribeni legal waters)
        res_safe = check_location_compliance(22.9800, 88.4000)
        self.assertEqual(res_safe["status"], "SAFE")
        self.assertIsNone(res_safe["is_legal"])
        self.assertGreater(res_safe["distance_to_nearest_sanctuary_km"], 0.0)

    def test_03_tools_river_metrics_and_alternatives(self):
        """Test telemetry retrieval and alternative species catalog."""
        from tools import get_river_metrics, get_alternative_species_info
        
        metrics = get_river_metrics("farakka")
        self.assertEqual(metrics["station_id"], "farakka")
        self.assertGreater(metrics["discharge_cumecs"], 1000)
        self.assertGreater(metrics["dissolved_oxygen_mg_l"], 5.0)

        alt_all = get_alternative_species_info("all")
        self.assertIn("Rohu / Rui", alt_all)
        self.assertIn("Catla", alt_all)
        self.assertIn("Tilapia", alt_all)

        alt_rui = get_alternative_species_info("rui")
        self.assertIn("Labeo rohita", alt_rui)

    def test_04_rag_engine_indexing_and_search(self):
        """Test RAG initialization and grounded similarity search."""
        from rag_engine import initialize_rag, search_regulations
        
        # Test index initialization
        col = initialize_rag()
        self.assertIsNotNone(col)

        # Test search query
        search_res = search_regulations("minimum legal catch size for Hilsa")
        self.assertIsInstance(search_res, str)
        self.assertIn("25", search_res)
        self.assertIn("CITATION", search_res)

    def test_05_agent_execution_loop(self):
        """Test agent reasoning and tool dispatch trace."""
        from agent import run_safehilsa_agent
        
        prompt = "Can I fish at my current coordinates? What is the legal size limit?"
        answer, tools_trace, metadata = run_safehilsa_agent(
            user_prompt=prompt,
            current_lat=24.8100,
            current_lon=87.9250
        )
        
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 50)
        self.assertIsInstance(tools_trace, list)
        self.assertGreaterEqual(len(tools_trace), 1)
        
        tool_names = [t["tool_name"] for t in tools_trace]
        self.assertIn("check_location_compliance", tool_names)
        self.assertIn("engine", metadata)

    def test_06_app_syntax_compilation(self):
        """Ensure app.py compiles without syntax errors."""
        app_path = os.path.join(APP_DIR, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, app_path, "exec")
        self.assertIsNotNone(compiled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
