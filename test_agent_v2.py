"""
SafeHilsa AI - Phase 4 Agent Logic & System Prompt Verification Suite
Validates:
1. Agent deterministic execution on Farakka coordinates (INCONCLUSIVE_AREA_BOUNDARY).
2. Agent deterministic execution on outside coordinates (OUTSIDE_CHECKED_SANCTUARIES as limited geographic finding, NOT legal clearance).
3. Agent deterministic execution on seasonal reach during active ban (ADVISORY_SANCTUARY_REACH).
4. Agent deterministic execution on seasonal reach during inactive window (PERMISSIBLE_SEASONAL_WINDOW).
5. Minimum catch size queries ground on Rule 46(4) 23 cm (never 25 cm).
6. State livelihood scheme queries trigger get_livelihood_support (Samudra Sathi & Matsyajeebi Bandhu).
7. River hydrology queries trigger get_river_metrics with historical observation disclaimers.
8. Ollama tools schema includes all 5 verified tool declarations.
"""

import os
import sys
import unittest

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from agent import run_safehilsa_agent, SYSTEM_PROMPT
from tools import OLLAMA_TOOLS_SCHEMA


class TestSafeHilsaAgentV2(unittest.TestCase):

    def test_01_farakka_inconclusive_boundary_agent(self):
        """Agent must report Farakka as inconclusive boundary with high enforcement risk."""
        prompt = "Can I fish near Farakka Barrage at coordinates 24.8100, 87.9250?"
        ans, tools, meta = run_safehilsa_agent(prompt, current_lat=24.8100, current_lon=87.9250)
        
        self.assertIsInstance(ans, str)
        self.assertIn("INCONCLUSIVE_AREA_BOUNDARY", ans)
        self.assertIn("Farakka", ans)
        self.assertIn("Rule 50(3)", ans)
        self.assertIn("HIGH ENFORCEMENT RISK", ans)
        tool_names = [t["tool_name"] for t in tools]
        self.assertIn("check_location_compliance", tool_names)
        print("\n[PASS] Agent Test 1: Farakka vicinity reported as INCONCLUSIVE_AREA_BOUNDARY with high enforcement warning")

    def test_02_outside_sanctuaries_limited_geographic_finding(self):
        """Agent must interpret outside sanctuaries as a limited geographic finding, NOT a definitive legal clearance."""
        prompt = "Is it legal to fish at Tribeni coordinates 22.9800, 88.4000?"
        ans, tools, meta = run_safehilsa_agent(prompt, current_lat=22.9800, current_lon=88.4000)
        
        self.assertIsInstance(ans, str)
        self.assertIn("OUTSIDE_CHECKED_SANCTUARIES", ans)
        # Verify contract: limited geographic finding, NOT a definitive legal clearance
        self.assertIn("NOT a definitive legal clearance", ans)
        self.assertIn("null", ans)
        self.assertIn("Rule 46(3)", ans)
        self.assertIn("Rule 46(4)", ans)
        print("[PASS] Agent Test 2: Outside coordinates strictly framed as limited geographic finding, NOT legal clearance")

    def test_03_active_seasonal_ban_reach_advisory(self):
        """Agent must report seasonal sanctuary reaches during ban months as ADVISORY_SANCTUARY_REACH."""
        prompt = "Can I fish near Katwa in July?"
        ans, tools, meta = run_safehilsa_agent(
            prompt, current_lat=23.5000, current_lon=88.2500, date_str="2026-07-15"
        )
        
        self.assertIsInstance(ans, str)
        self.assertIn("ADVISORY_SANCTUARY_REACH", ans)
        self.assertIn("Katwa", ans)
        self.assertIn("Rule 50(2)", ans)
        self.assertIn("Lateral Boundary Limitation", ans)
        print("[PASS] Agent Test 3: Seasonal ban reach corridor returns ADVISORY_SANCTUARY_REACH with lateral limitations")

    def test_04_inactive_seasonal_window_permissible(self):
        """Agent must report inactive seasonal windows as PERMISSIBLE_SEASONAL_WINDOW with Rule 46 reminders."""
        prompt = "Can I fish near Katwa in May?"
        ans, tools, meta = run_safehilsa_agent(
            prompt, current_lat=23.5000, current_lon=88.2500, date_str="2026-05-10"
        )
        
        self.assertIsInstance(ans, str)
        self.assertIn("PERMISSIBLE_SEASONAL_WINDOW", ans)
        self.assertIn("Rule 46(3)", ans)
        self.assertIn("Rule 46(4)", ans)
        print("[PASS] Agent Test 4: Inactive seasonal window correctly identified with gear and size conditions")

    def test_05_west_bengal_23cm_statutory_rule(self):
        """Agent system prompt and responses must cite West Bengal 23 cm limit under Rule 46(4), never 25 cm."""
        # 1. Verify system prompt text
        self.assertIn("23 cm", SYSTEM_PROMPT)
        self.assertIn("Rule 46(4)", SYSTEM_PROMPT)
        self.assertNotIn("smaller than 25 cm (Jatka)", SYSTEM_PROMPT)
        
        # 2. Verify deterministic query response citing primary gazette
        prompt = "What is the minimum legal catch size for Hilsa in West Bengal?"
        ans, tools, meta = run_safehilsa_agent(prompt, current_lat=None, current_lon=None)
        self.assertIn("23", ans)
        self.assertIn("SRC-REG-001", ans)
        print("[PASS] Agent Test 5: Minimum legal size confirmed as 23 cm under Gazette SRC-REG-001")

    def test_06_livelihood_support_schemes(self):
        """Agent must dispatch livelihood assistance queries to get_livelihood_support."""
        # Query Samudra Sathi in South 24 Parganas
        prompt = "What livelihood compensation or financial assistance is available during the ban in South 24 Parganas?"
        ans, tools, meta = run_safehilsa_agent(prompt, current_lat=None, current_lon=None)
        
        tool_names = [t["tool_name"] for t in tools]
        self.assertIn("get_livelihood_support", tool_names)
        self.assertIn("Samudra Sathi", ans)
        self.assertIn("5,000", ans)
        print("[PASS] Agent Test 6: Livelihood assistance tool dispatched with Samudra Sathi (Rs 5,000/mo DBT)")

    def test_07_river_hydrology_historical_disclaimer(self):
        """Agent must report river hydrology as historical observations, never live telemetry."""
        prompt = "What is the water discharge and dissolved oxygen flow at Farakka?"
        ans, tools, meta = run_safehilsa_agent(prompt, current_lat=None, current_lon=None)
        
        tool_names = [t["tool_name"] for t in tools]
        self.assertIn("get_river_metrics", tool_names)
        self.assertIn("HISTORICAL OBSERVATION", ans)
        self.assertIn("NOT live telemetry", ans)
        print("[PASS] Agent Test 7: River metrics returned with mandatory historical baseline disclaimer")

    def test_08_ollama_tools_schema_completeness(self):
        """Ollama function-calling tools schema must contain all 5 verified tools."""
        self.assertEqual(len(OLLAMA_TOOLS_SCHEMA), 5)
        schema_names = [t["function"]["name"] for t in OLLAMA_TOOLS_SCHEMA]
        expected_tools = [
            "check_location_compliance",
            "get_river_metrics",
            "get_livelihood_support",
            "search_regulations_tool",
            "get_alternative_species_info"
        ]
        for exp in expected_tools:
            self.assertIn(exp, schema_names)
        print("[PASS] Agent Test 8: Ollama function-calling schema includes all 5 verified tools")


if __name__ == "__main__":
    unittest.main(verbosity=2)
