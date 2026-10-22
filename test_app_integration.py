"""
SafeHilsa AI - Phase 5 Application & UI Integration Verification Suite
Validates:
1. app.py compilation and syntax integrity.
2. format_compliance_card_html for INCONCLUSIVE_AREA_BOUNDARY (Farakka).
3. format_compliance_card_html for ADVISORY_SANCTUARY_REACH (Katwa).
4. format_compliance_card_html for PERMISSIBLE_SEASONAL_WINDOW (Inactive ban window).
5. format_compliance_card_html for OUTSIDE_CHECKED_SANCTUARIES: strictly verifies
   that it is NEVER displayed as a definitive legal clearance (is_legal: null).
6. End-to-end query workflow: Location compliance check & structured statutory citations.
7. End-to-end query workflow: Livelihood assistance (Samudra Sathi Rs 5,000/mo DBT).
8. End-to-end query workflow: Historical river hydrology observations (with mandatory non-live disclaimer).
9. End-to-end query workflow: Graceful handling of missing coordinates without crash.
"""

import os
import sys
import unittest

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from app import format_compliance_card_html
from tools import check_location_compliance
from agent import run_safehilsa_agent


class TestSafeHilsaAppIntegration(unittest.TestCase):

    def test_01_app_compilation(self):
        """app.py must compile without syntax errors."""
        app_path = os.path.join(APP_DIR, "app.py")
        self.assertTrue(os.path.exists(app_path), "app.py does not exist")
        with open(app_path, "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, app_path, "exec")
        self.assertIsNotNone(compiled)
        print("\n[PASS] App Integration Test 1: app.py compiles cleanly without syntax errors")

    def test_02_format_compliance_card_html_inconclusive(self):
        """Card for Farakka must show INCONCLUSIVE_AREA_BOUNDARY and high enforcement risk."""
        compliance = check_location_compliance(24.8100, 87.9250)
        html = format_compliance_card_html(compliance, 24.8100, 87.9250)
        
        self.assertIn("INCONCLUSIVE BOUNDARY", html)
        self.assertIn("INCONCLUSIVE_AREA_BOUNDARY", html)
        self.assertIn("Farakka", html)
        self.assertIn("Rule 50(3)", html)
        self.assertIn("HIGH ENFORCEMENT RISK", html)
        print("[PASS] App Integration Test 2: Inconclusive boundary card properly rendered for Farakka")

    def test_03_format_compliance_card_html_advisory_reach(self):
        """Card for sanctuary reach in active ban must show ADVISORY_SANCTUARY_REACH."""
        compliance = check_location_compliance(23.5000, 88.2500, date_str="2026-07-15")
        html = format_compliance_card_html(compliance, 23.5000, 88.2500)
        
        self.assertIn("ADVISORY: ACTIVE SEASONAL SANCTUARY REACH", html)
        self.assertIn("ADVISORY_SANCTUARY_REACH", html)
        self.assertIn("Katwa", html)
        self.assertIn("June–August & October–December", html)
        self.assertIn("lateral_boundary_limitation", compliance)
        print("[PASS] App Integration Test 3: Advisory reach card rendered with seasonal ban details")

    def test_04_format_compliance_card_html_permissible_window(self):
        """Card for inactive window must show PERMISSIBLE_SEASONAL_WINDOW with Rule 46 reminders."""
        compliance = check_location_compliance(23.5000, 88.2500, date_str="2026-05-10")
        html = format_compliance_card_html(compliance, 23.5000, 88.2500)
        
        self.assertIn("PERMISSIBLE SEASONAL WINDOW", html)
        self.assertIn("PERMISSIBLE_WITH_GEAR_RESTRICTION", html)
        self.assertIn("Rule 46(3)", html)
        self.assertIn("Rule 46(4)", html)
        print("[PASS] App Integration Test 4: Permissible seasonal window card rendered with conditions")

    def test_05_format_compliance_card_html_outside_never_legal_clearance(self):
        """OUTSIDE_CHECKED_SANCTUARIES card must NEVER be displayed as a definitive legal clearance."""
        compliance = check_location_compliance(22.9800, 88.4000)
        self.assertIsNone(compliance["is_legal"])
        
        html = format_compliance_card_html(compliance, 22.9800, 88.4000)
        
        # Must explicitly state NOT a definitive legal clearance
        self.assertIn("NOT a definitive legal clearance", html)
        self.assertIn("is_legal: null", html)
        self.assertIn("Limited Geographic Finding", html)
        self.assertIn("Rule 46", html)
        
        # Must NOT claim safe or legal waters
        self.assertNotIn("PERMISSIBLE LEGAL FISHING WATERS", html)
        self.assertNotIn("status-card-safe", html)
        print("[PASS] App Integration Test 5: Outside sanctuaries card strictly verified - NEVER displayed as definitive legal clearance")

    def test_06_e2e_compliance_and_citation_workflow(self):
        """End-to-end query must return compliance analysis and structured citations citing primary statutes."""
        prompt = "Is my current location near Farakka legal to fish? What are the size restrictions?"
        ans, tools, meta = run_safehilsa_agent(prompt, current_lat=24.8100, current_lon=87.9250)
        
        self.assertIsInstance(ans, str)
        self.assertIn("Farakka", ans)
        self.assertIn("23", ans)
        tool_names = [t["tool_name"] for t in tools]
        self.assertIn("check_location_compliance", tool_names)
        self.assertIn("search_regulations_tool", tool_names)
        print("[PASS] App Integration Test 6: E2E location & regulatory citation query passed")

    def test_07_e2e_livelihood_support_workflow(self):
        """End-to-end query on ban compensation must return Samudra Sathi with Rs 5,000/mo DBT."""
        prompt = "What livelihood assistance is available during the fishing ban in South 24 Parganas?"
        ans, tools, meta = run_safehilsa_agent(prompt, current_lat=None, current_lon=None)
        
        self.assertIsInstance(ans, str)
        self.assertIn("Samudra Sathi", ans)
        self.assertIn("5,000", ans)
        self.assertIn("SRC-REG-027", ans)
        tool_names = [t["tool_name"] for t in tools]
        self.assertIn("get_livelihood_support", tool_names)
        print("[PASS] App Integration Test 7: E2E livelihood support query passed (Samudra Sathi Rs 5,000/mo)")

    def test_08_e2e_historical_hydrology_workflow(self):
        """End-to-end query on river hydrology must return metrics with historical baseline disclaimer."""
        prompt = "What is the water discharge and dissolved oxygen level at Farakka station?"
        ans, tools, meta = run_safehilsa_agent(prompt, current_lat=None, current_lon=None)
        
        self.assertIsInstance(ans, str)
        self.assertIn("HISTORICAL OBSERVATION", ans)
        self.assertIn("NOT live telemetry", ans)
        self.assertIn("cumecs", ans)
        self.assertIn("mg/L", ans)
        tool_names = [t["tool_name"] for t in tools]
        self.assertIn("get_river_metrics", tool_names)
        print("[PASS] App Integration Test 8: E2E historical river hydrology query passed with non-live disclaimer")

    def test_09_e2e_missing_coordinates_graceful_handling(self):
        """Missing or invalid coordinates must be handled gracefully without crashing or hallucination."""
        prompt = "Check if my location is safe"
        ans, tools, meta = run_safehilsa_agent(prompt, current_lat=None, current_lon=None)
        
        self.assertIsInstance(ans, str)
        self.assertGreater(len(ans), 30)
        # Should not crash and should not claim restricted or safe definitively
        self.assertNotIn("Traceback", ans)
        self.assertNotIn("Error", ans)
        print("[PASS] App Integration Test 9: Graceful handling of missing coordinates verified")


if __name__ == "__main__":
    unittest.main(verbosity=2)
