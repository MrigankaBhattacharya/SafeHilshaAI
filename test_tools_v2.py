"""
SafeHilsa AI - Phase 3 Tools & Observational Engine Verification Suite
Validates:
1. Geospatial coordinate compliance against statutory geometries (Rule 50).
2. Farakka 5.0 sq km area boundary (exact radius ~1.262 km, NOT 5 km radius).
3. Missing / invalid coordinates handling (inconclusive, no false determination).
4. Riparian unclassified margin handling (0.8 - 2.3 km).
5. Seasonal ban temporal evaluation (June-Aug & Oct-Dec vs inactive windows).
6. Observational river metrics lookup against genuine local CSV datasets.
7. Strict river basin isolation (Ganges-Bhagirathi-Hooghly vs Subernarekha).
8. Historical observation disclaimers (never labeled as live telemetry).
9. Livelihood assistance scheme eligibility (Samudra Sathi coastal vs inland, Matsyajeebi Bandhu).
10. RAG regulatory search tool integration with V2 hybrid engine.
"""

import os
import sys
import unittest

# Ensure app directory is in path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from tools import (
    check_location_compliance,
    get_river_metrics,
    get_livelihood_support,
    search_regulations_tool,
    get_alternative_species_info,
    FARAKKA_BARRAGE_LAT,
    FARAKKA_BARRAGE_LON,
    FARAKKA_STATUTORY_RADIUS_KM
)


class TestSafeHilsaToolsV2(unittest.TestCase):

    # ---------------------------------------------------------------------
    # 1. GEOSPATIAL COMPLIANCE & BOUNDARY TESTS (PHASE 3.1 REVISED SEMANTICS)
    # ---------------------------------------------------------------------

    def test_01_farakka_statutory_boundary(self):
        """Farakka 5 sq km area under Rule 50(3) has missing cadastral boundaries; returns INCONCLUSIVE_AREA_BOUNDARY."""
        # Inside vicinity of statutory 5 sq km area (~1.22 km from barrage)
        res_inside = check_location_compliance(24.8100, 87.9250)
        self.assertEqual(res_inside["status"], "INCONCLUSIVE_AREA_BOUNDARY")
        self.assertIsNone(res_inside["is_legal"])
        self.assertEqual(res_inside["determination"], "INCONCLUSIVE_AREA_BOUNDARY")
        self.assertIn("Farakka", res_inside["zone_name"])
        self.assertEqual(res_inside["statutory_area_sq_km"], 5.0)
        self.assertIn("Rule 50(3)", res_inside["legal_basis"])
        self.assertIn("missing_boundary_information", res_inside)
        self.assertIn("five (5) square kilometer area", res_inside["missing_boundary_information"])
        self.assertIn("HIGH ENFORCEMENT RISK", res_inside["recommended_action"])

        # Outside statutory 5 sq km area (~3.03 km east of barrage, outside reach corridor)
        res_outside_5sqkm = check_location_compliance(24.8000, 87.9500)
        self.assertEqual(res_outside_5sqkm["status"], "OUTSIDE_CHECKED_SANCTUARIES")
        self.assertIsNone(res_outside_5sqkm["is_legal"])
        self.assertEqual(res_outside_5sqkm["determination"], "OUTSIDE_CHECKED_SANCTUARIES")
        self.assertIn("scope_limitation", res_outside_5sqkm)
        self.assertGreater(res_outside_5sqkm.get("distance_to_nearest_sanctuary_km", 0.0), 2.0)
        print("\n[PASS] Test 1: Farakka boundary correctly evaluated as INCONCLUSIVE_AREA_BOUNDARY with missing cadastral data")

    def test_02_missing_and_invalid_coordinates(self):
        """Missing or invalid coordinates must NOT produce a definitive legal determination."""
        # None coordinates
        res_none = check_location_compliance(None, None)
        self.assertEqual(res_none["status"], "COORDINATES_MISSING_OR_INVALID")
        self.assertIsNone(res_none["is_legal"])
        self.assertEqual(res_none["determination"], "INCONCLUSIVE")
        self.assertIn("Missing coordinates", res_none["description"])

        # Out-of-bounds coordinates
        res_invalid = check_location_compliance(999.0, -999.0)
        self.assertEqual(res_invalid["status"], "COORDINATES_MISSING_OR_INVALID")
        self.assertIsNone(res_invalid["is_legal"])
        self.assertEqual(res_invalid["determination"], "INCONCLUSIVE")
        print("[PASS] Test 2: Missing and invalid coordinates handled inconclusive")

    def test_03_seasonal_ban_temporal_windows(self):
        """Sanctuary reach must evaluate seasonal ban windows (June-Aug & Oct-Dec) under Rule 50(2)."""
        # Katwa-Nabadwip point during active ban window (July 2026)
        res_july = check_location_compliance(23.5000, 88.2500, date_str="2026-07-15")
        self.assertEqual(res_july["status"], "ADVISORY_SANCTUARY_REACH")
        self.assertIsNone(res_july["is_legal"])
        self.assertEqual(res_july["determination"], "ADVISORY_SANCTUARY_REACH")
        self.assertIn("Katwa", res_july["zone_name"])
        self.assertIn("Rule 50(1)(b)", res_july["legal_basis"])
        self.assertIn("Rule 50(2)", res_july["legal_basis"])
        self.assertIn("lateral_boundary_limitation", res_july)

        # Katwa-Nabadwip point during inactive window (May 2026)
        res_may = check_location_compliance(23.5000, 88.2500, date_str="2026-05-10")
        self.assertEqual(res_may["status"], "PERMISSIBLE_SEASONAL_WINDOW")
        self.assertTrue(res_may["is_legal"])
        self.assertEqual(res_may["determination"], "PERMISSIBLE_WITH_GEAR_RESTRICTION")
        self.assertIn("inactive", res_may["description"].lower())
        self.assertIn("Rule 46(3)", res_may["description"])  # Monofilament gill net >= 90 mm
        self.assertIn("Rule 46(4)", res_may["description"])  # Size >= 23 cm transport/sale restriction
        print("[PASS] Test 3: Seasonal ban temporal windows accurately evaluated with statutory rule distinction")

    def test_04_reach_centrelines_advisory_semantics(self):
        """Points near river reach centrelines must return ADVISORY_SANCTUARY_REACH with lateral limitations."""
        # Reach 1: Lalbagh to Farakka point (~1.95 km from centerline)
        res_r1 = check_location_compliance(24.0950, 88.4830)
        self.assertEqual(res_r1["status"], "ADVISORY_SANCTUARY_REACH")
        self.assertEqual(res_r1["determination"], "ADVISORY_SANCTUARY_REACH")
        self.assertIn("lateral_boundary_limitation", res_r1)
        self.assertIn("Lalbagh to Farakka", res_r1["zone_name"])

        # Reach 2: Katwa to Hooghly Ghat point
        res_r2 = check_location_compliance(23.5000, 88.2500)
        self.assertEqual(res_r2["status"], "ADVISORY_SANCTUARY_REACH")
        self.assertEqual(res_r2["determination"], "ADVISORY_SANCTUARY_REACH")
        self.assertIn("lateral_boundary_limitation", res_r2)

        # Reach 3: Diamond Harbour to Nischintapur Godakhali point
        res_r3 = check_location_compliance(22.4047, 88.1425)
        self.assertEqual(res_r3["status"], "ADVISORY_SANCTUARY_REACH")
        self.assertEqual(res_r3["determination"], "ADVISORY_SANCTUARY_REACH")
        self.assertIn("lateral_boundary_limitation", res_r3)
        print("[PASS] Test 4: All 3 reach centrelines return ADVISORY_SANCTUARY_REACH with lateral limitations (arbitrary 1.8/2.3km removed)")

    def test_05_outside_checked_sanctuaries_scope_limitation(self):
        """Points outside sanctuaries must return OUTSIDE_CHECKED_SANCTUARIES with is_legal is None and scope limitation."""
        # Tribeni (latitude 22.98, longitude 88.40)
        res_safe = check_location_compliance(22.9800, 88.4000)
        self.assertEqual(res_safe["status"], "OUTSIDE_CHECKED_SANCTUARIES")
        self.assertIsNone(res_safe["is_legal"])
        self.assertTrue(res_safe["is_legal"] is None, "Regression assertion: is_legal must be None because tool only checks supplied geometries")
        self.assertEqual(res_safe["determination"], "OUTSIDE_CHECKED_SANCTUARIES")
        self.assertGreater(res_safe["distance_to_nearest_sanctuary_km"], 2.0)
        self.assertIn("scope_limitation", res_safe)
        self.assertIn("supplied", res_safe["scope_limitation"].lower())
        self.assertIn("EEZ", res_safe["scope_limitation"])
        self.assertIn("statutory_reminders", res_safe)
        print("[PASS] Test 5: Outside locations return OUTSIDE_CHECKED_SANCTUARIES with is_legal=None regression assertion")

    # ---------------------------------------------------------------------
    # 2. RIVER OBSERVATIONAL METRICS & BASIN ISOLATION TESTS
    # ---------------------------------------------------------------------

    def test_06_hooghly_observational_metrics(self):
        """Hooghly stations must return verified ICAR-CIFRI historical observations with disclaimers."""
        metrics = get_river_metrics("farakka")
        self.assertEqual(metrics["status"], "AVAILABLE")
        self.assertEqual(metrics["station_id"], "farakka")
        self.assertEqual(metrics["basin"], "Ganges-Bhagirathi-Hooghly")
        self.assertEqual(metrics["source_id"], "SRC-REG-019")
        self.assertTrue(metrics["is_historical_observation"])
        self.assertFalse(metrics["is_live_telemetry"])
        self.assertEqual(metrics["data_type"], "HISTORICAL_OBSERVATION")
        self.assertIn("NOT live", metrics["disclaimer"])
        self.assertGreater(metrics["dissolved_oxygen_mg_l"], 5.0)
        self.assertGreater(metrics["discharge_cumecs"], 1000.0)
        print("[PASS] Test 6: Hooghly observational metrics verified from CIFRI 179")

    def test_07_subarnarekha_basin_separation(self):
        """Subarnarekha stations must be isolated from Hooghly-Bhagirathi observations."""
        metrics_sub = get_river_metrics("fekoghat")
        self.assertEqual(metrics_sub["status"], "AVAILABLE")
        self.assertEqual(metrics_sub["station_id"], "fekoghat")
        self.assertEqual(metrics_sub["basin"], "Subernarekha")
        self.assertEqual(metrics_sub["source_id"], "SRC-REG-013")
        self.assertTrue(metrics_sub["is_historical_observation"])
        self.assertFalse(metrics_sub["is_live_telemetry"])

        # Cross-basin conflict: Requesting Subernarekha station under Hooghly basin filter
        cross_res = get_river_metrics("fekoghat", basin="Hooghly")
        self.assertEqual(cross_res["status"], "UNAVAILABLE")
        self.assertIn("Cross-basin restriction", cross_res["message"])

        # Cross-basin conflict: Requesting Hooghly station under Subernarekha basin filter
        cross_res2 = get_river_metrics("farakka", basin="Subernarekha")
        self.assertEqual(cross_res2["status"], "UNAVAILABLE")
        self.assertIn("Cross-basin restriction", cross_res2["message"])
        print("[PASS] Test 7: Strict basin isolation verified (Hooghly vs Subarnarekha)")

    def test_08_unsupported_station_lookup(self):
        """Unknown or unmonitored river station must return explicit unavailable status."""
        res_unknown = get_river_metrics("unsupported_river_station_xyz")
        self.assertEqual(res_unknown["status"], "UNAVAILABLE")
        self.assertIn("unavailable", res_unknown["message"].lower())
        self.assertIn("available_stations", res_unknown)
        print("[PASS] Test 8: Unsupported monitoring stations return explicit UNAVAILABLE")

    # ---------------------------------------------------------------------
    # 3. LIVELIHOOD SUPPORT & SCHEME ELIGIBILITY TESTS
    # ---------------------------------------------------------------------

    def test_09_samudra_sathi_eligibility(self):
        """Samudra Sathi must restrict eligibility to coastal districts under Notification 724-Fish."""
        # Eligible Coastal District (South 24 Parganas)
        res_coastal = get_livelihood_support("samudra_sathi", district="South 24 Parganas")
        self.assertEqual(res_coastal["status"], "SUCCESS")
        scheme = res_coastal["scheme"]
        self.assertEqual(scheme["source_id"], "SRC-REG-027")
        self.assertIn("5,000", scheme["benefit_amount"])
        self.assertIn("15th April to 14th June", scheme["ban_window"])
        self.assertTrue(scheme["district_evaluation"]["is_district_eligible"])

        # Ineligible Inland District (Murshidabad)
        res_inland = get_livelihood_support("samudra_sathi", district="Murshidabad")
        self.assertEqual(res_inland["status"], "SUCCESS")
        self.assertFalse(res_inland["scheme"]["district_evaluation"]["is_district_eligible"])
        self.assertIn("inland", res_inland["scheme"]["district_evaluation"]["evaluation_note"].lower())
        print("[PASS] Test 9: Samudra Sathi coastal eligibility and inland exclusion verified")

    def test_10_matsyajeebi_bandhu_scheme(self):
        """Matsyajeebi Bandhu death benefit must cover all districts and verify criteria."""
        res_mb = get_livelihood_support("matsyajeebi_bandhu", district="Murshidabad")
        self.assertEqual(res_mb["status"], "SUCCESS")
        scheme = res_mb["scheme"]
        self.assertEqual(scheme["source_id"], "SRC-REG-028 / SRC-REG-029")
        self.assertIn("2,00,000", scheme["benefit_amount"])
        self.assertIn("All 23 Districts", scheme["eligible_districts"])
        print("[PASS] Test 10: Matsyajeebi Bandhu (Rs 2 Lakh death benefit) verified")

    # ---------------------------------------------------------------------
    # 4. RAG REGULATORY TOOL CITATION INTEGRATION
    # ---------------------------------------------------------------------

    def test_11_search_regulations_tool_citations(self):
        """search_regulations_tool must return structured citations citing primary statutes."""
        output = search_regulations_tool("What is the minimum legal catch size for Hilsa in West Bengal?")
        self.assertIsInstance(output, str)
        self.assertIn("CITATION 1", output)
        self.assertIn("SRC-REG-001", output)
        self.assertIn("23", output)
        self.assertIn("Authority:", output)
        print("[PASS] Test 11: search_regulations_tool returns structured citations with SRC-REG-001")

    # ---------------------------------------------------------------------
    # 5. STATUTORY RULE DISTINCTION & PRIMARY PENALTY VERIFICATION
    # ---------------------------------------------------------------------

    def test_12_statutory_rule_distinction_and_penalties(self):
        """Rule 46(4) (transport/sale/possession < 23cm) must be distinct from Rule 46(3) (gear mesh < 90mm), with Sec 20 penalty."""
        # 1. Verify Primary Legislation Penalty (Section 20 & 22 of WB Inland Fisheries Act, 1984)
        res_farakka = check_location_compliance(24.8100, 87.9250)
        self.assertIn("Section 20", res_farakka["penalty"])
        self.assertIn("5,000", res_farakka["penalty"])
        self.assertIn("Section 22", res_farakka["penalty"])
        self.assertIn("DFO", res_farakka["penalty"])
        self.assertIn("25,000", res_farakka["penalty"])

        # 2. Verify Rule 46(4) transport/marketing/selling/possession vs Rule 46(3) catching gear distinction
        res_may = check_location_compliance(23.5000, 88.2500, date_str="2026-05-10")
        desc = res_may["description"]
        self.assertIn("Rule 46(3)", desc)
        self.assertIn("catching", desc.lower())
        self.assertIn("90 mm", desc)
        self.assertIn("Rule 46(4)", desc)
        self.assertIn("transporting, marketing, selling, or possessing", desc)
        self.assertIn("23 cm", desc)

        # 3. Verify Statutory reminders on outside waters
        res_out = check_location_compliance(22.9800, 88.4000)
        reminders = res_out["statutory_reminders"]
        self.assertIn("Rule 46(3)", reminders)
        self.assertIn("Rule 46(4)", reminders)
        self.assertIn("Section 20", reminders)
        print("[PASS] Test 12: Statutory penalties (Section 20) and Rule 46(3) vs 46(4) distinctions strictly verified")


if __name__ == "__main__":
    unittest.main(verbosity=2)
