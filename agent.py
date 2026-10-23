"""
SafeHilsa AI - Agentic Loop & Tool Execution Engine (Phase 4 Overhaul)
Connects to local Ollama (Llama 3, Qwen2.5, etc.) with native function calling.
Executes deterministic Python tools (geospatial checks, RAG search, river metrics,
livelihood schemes) and provides fallback reasoning when Ollama is offline.
Grounded in Kolkata Gazette Notification No. 719-Fish and verified local data sources.
"""

import json
import time
import re
from typing import List, Dict, Any, Tuple, Optional

from tools import (
    AVAILABLE_TOOLS,
    OLLAMA_TOOLS_SCHEMA,
    check_location_compliance,
    search_regulations_tool,
    get_river_metrics,
    get_livelihood_support,
    get_alternative_species_info
)

SYSTEM_PROMPT = """You are SafeHilsa AI, a Principal Riverine Compliance Guardian and Marine Fisheries Advisor for the Department of Fisheries, Government of West Bengal.
Your mission is to conserve Hilsa fish (Tenualosa ilisha) populations, protect critical riverine and estuarine spawning corridors, enforce statutory fisheries regulations, and guide fishers on legal compliance, sustainable alternative catches, and state livelihood assistance schemes.

STATUTORY LEGAL FRAMEWORK (Kolkata Gazette Notification No. 719-Fish & West Bengal Inland Fisheries Act, 1984):

1. MINIMUM LEGAL SIZE & SUPPLY CHAIN BAN (Rule 46(4)):
   - In West Bengal, it is strictly prohibited to TRANSPORT, MARKET, SELL, or POSSESS any Hilsa fish having length less than 23 cm (not 25 cm).
   - Any juvenile caught unintentionally must be released alive into the river immediately.
   - Note: Bangladesh enforces a 25 cm Jatka restriction under its own domestic law; in West Bengal the statutory limit is 23 cm.

2. FISHING GEAR RESTRICTIONS (Rule 46(3) & Rule 50(4)):
   - Rule 46(3): Strictly prohibited to CATCH or cause to be caught any Hilsa using monofilament gill net having mesh size less than 90 mm (approx 3.5 inches).
   - Rule 50(4): Strictly prohibited to operate small-meshed nets (mesh size less than 1 inch, such as 'Current Jal', 'Kharki Jal', 'Bhendi Jal') in inland waters from February to April.

3. STATUTORY SANCTUARIES & CLOSED SEASONS (Rule 50(1), 50(2), 50(3)):
   - River Reach Sanctuaries (Rule 50(1)):
     (a) Lalbagh (Murshidabad) to Farakka (Malda) along Bhagirathi-Ganga.
     (b) Katwa (Purba Bardhaman) to Hooghly Ghat (Hooghly) along Bhagirathi-Hooghly.
     (c) Diamond Harbour to Nischintapur Godakhali (South 24 Parganas) along Hooghly Estuary.
     (d) Demarcated sand bar areas in rivers Matla, Roymongal, and Thakuran (Sundarbans).
   - Seasonal Ban (Rule 50(2)): ALL types of fish catching are prohibited along these river reaches during JUNE TO AUGUST and OCTOBER TO DECEMBER.
   - Farakka Barrage Year-Round Prohibited Area (Rule 50(3)): All Hilsa fishing is prohibited within a five (5) square kilometer area of the Farakka Barrage round the year.
   - Marine EEZ Ban: 61-day uniform marine fishing ban on the East Coast from 15th April to 14th June annually under GoI order.

4. PRIMARY STATUTORY PENALTIES & ENFORCEMENT:
   - Primary Enactment: Section 20 of the West Bengal Inland Fisheries Act, 1984 (West Bengal Act XXV of 1984).
   - Statutory Penalty: Fine up to ₹5,000 and confiscation of prohibited gear/catch upon written report by a Fishery Officer (not below DFO) or Police Officer (not below Sub-Inspector) under Section 22.
   - Administrative compounding or executive orders may state compounding maximums up to ₹25,000.

5. STATE LIVELIHOOD ASSISTANCE SCHEMES:
   - Samudra Sathi Scheme (Notification No. 724-Fish, 2024): Direct Benefit Transfer of ₹5,000/month during the 61-day marine ban (15 April - 14 June, total ₹10,000) for registered marine fishers in coastal districts: Purba Medinipur, South 24 Parganas, and North 24 Parganas. Inland fishers are excluded.
   - Matsyajeebi Bandhu Scheme: Death benefit grant of ₹2,00,000 (₹2 Lakh) to legal heir/nominee upon death of registered fisher (aged 18-60) across all 23 districts of West Bengal.

6. OBSERVATIONAL RIVER METRICS & BASIN SEPARATION:
   - Hydrology and water quality data are HISTORICAL BASELINE OBSERVATIONS from ICAR-CIFRI Bulletin 179 and CWC. Never label them as live real-time telemetry.
   - Strict basin isolation: Ganges-Bhagirathi-Hooghly stations must never be conflated with Subarnarekha stations.

7. INTERPRETATION OF GEOSPATIAL COMPLIANCE:
   - If status is INCONCLUSIVE_AREA_BOUNDARY: Warn of high enforcement risk near Farakka Barrage (Rule 50(3)); explain that cadastral coordinates are missing in the Gazette.
   - If status is ADVISORY_SANCTUARY_REACH: River reach corridor seasonal ban is active (June-Aug, Oct-Dec) under Rule 50(2); advise cessation of fishing along the corridor.
   - If status is PERMISSIBLE_SEASONAL_WINDOW: Sanctuary ban is inactive; fishing is permissible subject to mandatory mesh (>= 90mm) and size (>= 23cm) rules.
   - If status is OUTSIDE_CHECKED_SANCTUARIES: CRITICAL: Treat this as a LIMITED GEOGRAPHIC FINDING, NOT a definitive legal clearance. Explicitly inform the user that while the coordinate is outside checked sanctuary reaches, statewide gear/size rules (Rule 46) and any maritime EEZ bans or local port orders remain applicable and unverified.
   - If status is COORDINATES_MISSING_OR_INVALID: Inconclusive; request valid decimal GPS coordinates.

TOOL USAGE PROTOCOL:
- When checking coordinates or location legality: ALWAYS call `check_location_compliance(lat, lon, date_str)`.
- When querying fisheries laws, ban dates, size limits, or penalties: ALWAYS call `search_regulations_tool(query)`.
- When querying river hydrology or water quality: ALWAYS call `get_river_metrics(station_id, basin)`.
- When querying ban financial support or compensation: ALWAYS call `get_livelihood_support(scheme_name, district)`.
- When querying alternative catches during bans: ALWAYS call `get_alternative_species_info(species_name)`.

RESPONSE STYLE:
- Authoritative, respectful to artisanal fishers, and grounded in official law.
- Always provide structured citations referencing official source IDs (e.g. SRC-REG-001) and specific rule numbers.
- Never give false legal clearances for outside locations; clearly state scope limitations.
"""


def check_ollama_status(host: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Checks if local Ollama daemon is reachable and lists available models.
    """
    try:
        import ollama
        client = ollama.Client(host=host)
        models_resp = client.list()
        
        # models_resp might be a list or dict depending on ollama-python version
        model_names = []
        if hasattr(models_resp, "models"):
            model_names = [m.model for m in models_resp.models]
        elif isinstance(models_resp, dict) and "models" in models_resp:
            model_names = [m.get("name", m.get("model", "")) for m in models_resp["models"]]
        
        return {
            "online": True,
            "host": host,
            "models": model_names,
            "error": None
        }
    except Exception as e:
        return {
            "online": False,
            "host": host,
            "models": [],
            "error": str(e)
        }


def _execute_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> Tuple[Any, float]:
    """Executes a registered tool function and records execution elapsed time."""
    start_time = time.time()
    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: '{tool_name}' requested by agent.")
    
    func = AVAILABLE_TOOLS[tool_name]
    result = func(**tool_args)
    duration = round(time.time() - start_time, 3)
    return result, duration


def _run_deterministic_agent(
    user_prompt: str,
    current_lat: Optional[float] = None,
    current_lon: Optional[float] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    date_str: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deterministic rule-based agentic fallback.
    Executes actual tools based on semantic intent and coordinates when Ollama is offline.
    """
    executed_tools = []
    prompt_lower = user_prompt.lower()

    # Extract coordinates from text if not explicitly provided
    if current_lat is None or current_lon is None:
        coord_match = re.search(r'(-?\d{1,2}\.\d+)[,\s]+(-?\d{1,3}\.\d+)', user_prompt)
        if coord_match:
            try:
                current_lat = float(coord_match.group(1))
                current_lon = float(coord_match.group(2))
            except Exception:
                pass
    
    # 1. Location compliance intent
    coords_in_prompt = (
        "location" in prompt_lower or "here" in prompt_lower or "safe" in prompt_lower 
        or "coordinate" in prompt_lower or "gps" in prompt_lower or "legal to fish" in prompt_lower
        or "can i fish" in prompt_lower or "sanctuary" in prompt_lower or "zone" in prompt_lower
        or "farakka" in prompt_lower or "katwa" in prompt_lower or "nabadwip" in prompt_lower
        or "lalbagh" in prompt_lower or "diamond harbour" in prompt_lower or "sundarbans" in prompt_lower
        or (current_lat is not None and current_lon is not None and any(w in prompt_lower for w in ["lat", "lon", "where", "fish"]))
    )
    
    loc_result = None
    if coords_in_prompt and current_lat is not None and current_lon is not None:
        tool_args = {"lat": current_lat, "lon": current_lon}
        if date_str:
            tool_args["date_str"] = date_str
        res, dur = _execute_tool_call("check_location_compliance", tool_args)
        loc_result = res
        executed_tools.append({
            "tool_name": "check_location_compliance",
            "args": tool_args,
            "result": res,
            "duration_s": dur,
            "status": "success"
        })

    # 2. Livelihood assistance / compensation scheme intent
    livelihood_result = None
    if any(k in prompt_lower for k in [
        "livelihood", "compensation", "samudra sathi", "matsyajeebi", "bandhu",
        "dbt", "allowance", "bata", "financial", "assistance", "death benefit"
    ]):
        scheme = "matsyajeebi_bandhu" if any(w in prompt_lower for w in ["matsyajeebi", "death", "2 lakh", "accident"]) else "samudra_sathi"
        
        # Check for district mention
        matched_dist = None
        districts = [
            "south 24 parganas", "north 24 parganas", "purba medinipur",
            "murshidabad", "malda", "hooghly", "howrah", "kolkata", "nadia", "bardhaman"
        ]
        for d in districts:
            if d in prompt_lower:
                matched_dist = d.title()
                break
        
        scheme_args = {"scheme_name": scheme}
        if matched_dist:
            scheme_args["district"] = matched_dist

        res, dur = _execute_tool_call("get_livelihood_support", scheme_args)
        livelihood_result = res
        executed_tools.append({
            "tool_name": "get_livelihood_support",
            "args": scheme_args,
            "result": res,
            "duration_s": dur,
            "status": "success"
        })

    # 3. Regulations / Size / Ban intent
    reg_result = None
    if any(k in prompt_lower for k in [
        "size", "23", "25", "ban", "jatka", "law", "rule", "fine", "penalty",
        "illegal", "current jal", "net", "spawning", "act", "mesh", "section",
        "gazette", "notification", "719", "monofilament"
    ]):
        res, dur = _execute_tool_call("search_regulations_tool", {"query": user_prompt})
        reg_result = res
        executed_tools.append({
            "tool_name": "search_regulations_tool",
            "args": {"query": user_prompt},
            "result": res,
            "duration_s": dur,
            "status": "success"
        })

    # 4. River metrics intent
    metrics_result = None
    if any(k in prompt_lower for k in [
        "discharge", "water", "turbidity", "oxygen", "do", "cumec", "temperature",
        "river", "flow", "metric", "cifri", "cwc", "hydrology", "quality", "salinity"
    ]):
        station = "farakka"
        basin = "Ganges-Bhagirathi-Hooghly"
        
        if "katwa" in prompt_lower:
            station = "katwa"
        elif "nabadwip" in prompt_lower:
            station = "nabadwip"
        elif "kalna" in prompt_lower:
            station = "kalna"
        elif "tribeni" in prompt_lower:
            station = "tribeni"
        elif "barrackpore" in prompt_lower:
            station = "barrackpore"
        elif "kolkata" in prompt_lower or "howrah" in prompt_lower:
            station = "kolkata_hooghly"
        elif "diamond" in prompt_lower:
            station = "diamond_harbour"
        elif "kakdwip" in prompt_lower:
            station = "kakdwip"
        elif "sagar" in prompt_lower or "bay" in prompt_lower:
            station = "sagar_island"
        elif "fekoghat" in prompt_lower or "subarnarekha" in prompt_lower or "subernarekha" in prompt_lower:
            station = "fekoghat"
            basin = "Subernarekha"
        
        res, dur = _execute_tool_call("get_river_metrics", {"station_id": station, "basin": basin})
        metrics_result = res
        executed_tools.append({
            "tool_name": "get_river_metrics",
            "args": {"station_id": station, "basin": basin},
            "result": res,
            "duration_s": dur,
            "status": "success"
        })

    # 5. Alternative species intent
    alt_result = None
    if any(k in prompt_lower for k in [
        "alternative", "other fish", "rui", "rohu", "catla", "tilapia", "pangas", "bata", "what else", "substitute"
    ]):
        species_target = "all"
        for s in ["rui", "catla", "tilapia", "pangas", "bata"]:
            if s in prompt_lower:
                species_target = s
                break
        res, dur = _execute_tool_call("get_alternative_species_info", {"species_name": species_target})
        alt_result = res
        executed_tools.append({
            "tool_name": "get_alternative_species_info",
            "args": {"species_name": species_target},
            "result": res,
            "duration_s": dur,
            "status": "success"
        })

    # If no specific tool triggered, query regulations as general context
    if not executed_tools:
        res, dur = _execute_tool_call("search_regulations_tool", {"query": user_prompt})
        reg_result = res
        executed_tools.append({
            "tool_name": "search_regulations_tool",
            "args": {"query": user_prompt},
            "result": res,
            "duration_s": dur,
            "status": "success"
        })

    # Synthesize comprehensive response
    response_parts = []
    
    if loc_result:
        status_val = loc_result.get("status")
        det_val = loc_result.get("determination")
        
        if status_val == "INCONCLUSIVE_AREA_BOUNDARY":
            response_parts.append(
                f"### ⚠️ INCONCLUSIVE BOUNDARY / HIGH ENFORCEMENT RISK (Farakka Barrage)\n"
                f"Your selected coordinates (**{current_lat:.4f}°N, {current_lon:.4f}°E**) are within proximity of the **{loc_result['zone_name']}** ({loc_result['river']}).\n\n"
                f"- **Determination:** `{det_val}` (Definitive legal inclusion is legally inconclusive)\n"
                f"- **Legal Basis:** {loc_result['legal_basis']}\n"
                f"- **Cadastral Boundary Limitation:** {loc_result.get('missing_boundary_information')}\n"
                f"- **Statutory Penalties:** {loc_result['penalty']}\n"
                f"- **Fisher Advisory:** {loc_result['recommended_action']}\n"
            )
        elif status_val == "ADVISORY_SANCTUARY_REACH":
            response_parts.append(
                f"### 🚨 ADVISORY: ACTIVE SEASONAL SANCTUARY REACH\n"
                f"Your selected coordinates (**{current_lat:.4f}°N, {current_lon:.4f}°E**) fall along the **{loc_result['zone_name']}** ({loc_result['river']}).\n\n"
                f"- **Determination:** `{det_val}` (Seasonal Ban Active: June–August & October–December)\n"
                f"- **Legal Basis:** {loc_result['legal_basis']}\n"
                f"- **Lateral Boundary Limitation:** {loc_result.get('lateral_boundary_limitation')}\n"
                f"- **Statutory Penalties:** {loc_result['penalty']}\n"
                f"- **Mandatory Action:** {loc_result['recommended_action']}\n"
            )
        elif status_val == "PERMISSIBLE_SEASONAL_WINDOW":
            response_parts.append(
                f"### ℹ️ PERMISSIBLE SEASONAL WINDOW (Sanctuary Ban Currently Inactive)\n"
                f"Your coordinates (**{current_lat:.4f}°N, {current_lon:.4f}°E**) fall along **{loc_result['zone_name']}**, where the seasonal sanctuary ban under Rule 50(2) is currently **INACTIVE**.\n\n"
                f"- **Status:** `PERMISSIBLE_SEASONAL_WINDOW`\n"
                f"- **Determination:** `{det_val}`\n"
                f"- **Statutory Rules:** {loc_result.get('description')}\n"
                f"- **Advisory:** {loc_result.get('recommended_action')}\n"
            )
        elif status_val == "OUTSIDE_CHECKED_SANCTUARIES":
            response_parts.append(
                f"### 📍 OUTSIDE CHECKED SANCTUARIES (Limited Geographic Finding)\n"
                f"Your coordinates (**{current_lat:.4f}°N, {current_lon:.4f}°E**) are **outside the checked West Bengal Hilsa sanctuary reach centrelines**.\n\n"
                f"- **Determination:** `{det_val}`\n"
                f"- **Legal Clearance Status:** **NOT a definitive legal clearance** (`is_legal: null`).\n"
                f"- **Nearest Checked Sanctuary:** {loc_result.get('nearest_sanctuary')} (~{loc_result.get('distance_to_nearest_sanctuary_km')} km away)\n"
                f"- **Scope Limitation:** {loc_result.get('scope_limitation')}\n"
                f"- **Statutory Reminders:** {loc_result.get('statutory_reminders')}\n"
                f"- **Advisory:** {loc_result.get('recommended_action')}\n"
            )
        elif status_val == "COORDINATES_MISSING_OR_INVALID":
            response_parts.append(
                f"### ❓ COORDINATES MISSING OR INVALID\n"
                f"{loc_result.get('description')}\n\n"
                f"- **Action Required:** {loc_result.get('recommended_action')}\n"
            )

    if livelihood_result:
        if livelihood_result.get("status") == "SUCCESS":
            sch = livelihood_result.get("scheme", {})
            dist_eval = sch.get("district_evaluation", {})
            eval_text = f"\n- **District Evaluation:** {dist_eval.get('evaluation_note')}" if dist_eval else ""
            response_parts.append(
                f"### 🤝 State Livelihood Assistance Scheme ({sch.get('scheme_name')})\n"
                f"- **Benefit Amount:** **{sch.get('benefit_amount')}**\n"
                f"- **Target Beneficiaries:** {sch.get('target_beneficiaries')}\n"
                f"- **Eligible Districts:** {sch.get('eligible_districts')}\n"
                f"- **Ban Window:** {sch.get('ban_window', 'N/A')}\n"
                f"- **Legal Basis & Authority:** {sch.get('legal_basis')} ({sch.get('source_id')}){eval_text}\n"
            )
        else:
            response_parts.append(
                f"### 🤝 Livelihood Scheme Information\n"
                f"{livelihood_result.get('message', 'Scheme information unavailable.')}\n"
            )

    if metrics_result:
        if metrics_result.get("status") == "AVAILABLE":
            response_parts.append(
                f"### 🌊 River Hydrology & Environmental Observations ({metrics_result['station_name']})\n"
                f"- **Basin:** {metrics_result['basin']} | **River Reach:** {metrics_result.get('river_reach', 'Main Channel')}\n"
                f"- **Discharge:** {metrics_result.get('discharge_cumecs')} cumecs\n"
                f"- **Dissolved Oxygen:** {metrics_result.get('dissolved_oxygen_mg_l')} mg/L (Min optimal migration: > 5.0 mg/L)\n"
                f"- **Turbidity:** {metrics_result.get('turbidity_ntu')} NTU | **Water Temp:** {metrics_result.get('surface_water_temp_c', metrics_result.get('water_temperature_c', 'N/A'))} °C\n"
                f"- **Source:** {metrics_result['source_id']} ({metrics_result.get('source_title', 'Hydrology Dataset')})\n"
                f"- **Observation Period:** {metrics_result.get('observation_period', 'Historical Baseline')}\n"
                f"- **⚠️ Mandatory Notice:** {metrics_result.get('disclaimer')} (HISTORICAL OBSERVATION — NOT live telemetry).\n"
            )
        else:
            response_parts.append(
                f"### 🌊 River Monitoring Station Notice\n"
                f"{metrics_result.get('message')}\n"
            )

    if reg_result:
        response_parts.append(
            f"### 📜 Applicable Fisheries Statutes & Gazette Notifications\n"
            f"{reg_result}\n"
        )

    if alt_result:
        response_parts.append(
            f"{alt_result}\n"
        )

    # Extract focus coordinates for map synchronization
    focus_coords = None
    if loc_result and isinstance(loc_result, dict) and "coordinates" in loc_result:
        coords = loc_result["coordinates"]
        if coords.get("latitude") is not None and coords.get("longitude") is not None:
            focus_coords = {"lat": float(coords["latitude"]), "lon": float(coords["longitude"])}
    elif metrics_result and isinstance(metrics_result, dict) and "coordinates" in metrics_result:
        coords = metrics_result["coordinates"]
        if coords.get("lat") is not None and coords.get("lon") is not None:
            focus_coords = {"lat": float(coords["lat"]), "lon": float(coords["lon"])}

    meta = {
        "engine": "SafeHilsa Deterministic Fallback Agent (Ollama Offline)",
        "tool_count": len(executed_tools),
        "execution_mode": "local_fast",
        "focus_coordinates": focus_coords
    }

    final_text = "\n\n".join(response_parts)
    return final_text, executed_tools, meta


def run_safehilsa_agent(
    user_prompt: str,
    current_lat: Optional[float] = None,
    current_lon: Optional[float] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_name: str = "llama3.2",
    host: str = "http://localhost:11434",
    date_str: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main Agentic execution function.
    Connects to Ollama, runs multi-step tool calls, and falls back gracefully
    to the deterministic agent when Ollama is offline or model is downloading.

    Args:
        user_prompt: The user's query or action request.
        current_lat: Active latitude from map or manual input.
        current_lon: Active longitude from map or manual input.
        chat_history: Previous conversational messages.
        model_name: Model identifier (e.g. 'llama3.2', 'llama3', 'qwen2.5').
        host: Ollama server endpoint.
        date_str: Optional ISO date string for seasonal ban evaluation.

    Returns:
        Tuple of (response_text, executed_tools_trace, metadata)
    """
    executed_tools = []

    # 1. Test Ollama connectivity
    status = check_ollama_status(host=host)
    if not status["online"] or not status["models"]:
        # Fallback to local deterministic agent
        return _run_deterministic_agent(user_prompt, current_lat, current_lon, chat_history, date_str=date_str)

    # 2. Check if requested model or a suitable alternative exists
    available = status["models"]
    selected_model = model_name
    if not any(selected_model in m for m in available):
        # Pick first available or fallback
        if available:
            selected_model = available[0]
        else:
            return _run_deterministic_agent(user_prompt, current_lat, current_lon, chat_history, date_str=date_str)

    try:
        import ollama
        client = ollama.Client(host=host)

        # Context-augmented system message
        lat_str = f"{current_lat:.4f}" if current_lat is not None else "Not provided"
        lon_str = f"{current_lon:.4f}" if current_lon is not None else "Not provided"
        date_info = f"- Evaluation Date: {date_str}\n" if date_str else ""
        
        contextual_system = (
            f"{SYSTEM_PROMPT}\n\n"
            f"CURRENT USER CONTEXT:\n"
            f"- GPS Latitude: {lat_str}\n"
            f"- GPS Longitude: {lon_str}\n"
            f"{date_info}"
            f"CRITICAL COMPLIANCE DIRECTIVES:\n"
            f"1. When check_location_compliance returns OUTSIDE_CHECKED_SANCTUARIES, treat this as a LIMITED GEOGRAPHIC FINDING "
            f"(the coordinate is not within proximity of the checked inland sanctuary centrelines) and NOT a definitive legal clearance. "
            f"Explicitly inform the user that statewide Rule 46(3) mesh >= 90mm and Rule 46(4) size >= 23cm restrictions, as well as "
            f"marine EEZ bans, remain applicable and unverified.\n"
            f"2. Minimum legal catch size in West Bengal is 23 cm under Notification 719-Fish Rule 46(4) (not 25 cm).\n"
            f"3. River metrics from get_river_metrics are HISTORICAL BASELINE OBSERVATIONS from ICAR-CIFRI Bulletin 179 / CWC, never live telemetry."
        )

        messages = [{"role": "system", "content": contextual_system}]

        # Append previous chat history
        if chat_history:
            for ch in chat_history[-6:]:  # Keep recent context window
                messages.append({"role": ch.get("role", "user"), "content": ch.get("content", "")})

        # Append current user prompt
        messages.append({"role": "user", "content": user_prompt})

        # 3. Agentic Loop (up to 4 tool iterations)
        max_turns = 4
        current_turn = 0

        while current_turn < max_turns:
            current_turn += 1

            response = client.chat(
                model=selected_model,
                messages=messages,
                tools=OLLAMA_TOOLS_SCHEMA
            )

            response_msg = response.message
            tool_calls = getattr(response_msg, "tool_calls", None)

            # Check dict-style if object attribute not present
            if not tool_calls and isinstance(response, dict):
                tool_calls = response.get("message", {}).get("tool_calls", [])

            if not tool_calls:
                # No more tools requested, model returned final text
                final_content = response_msg.content if hasattr(response_msg, "content") else response.get("message", {}).get("content", "")
                meta = {
                    "engine": f"Ollama ({selected_model})",
                    "tool_count": len(executed_tools),
                    "execution_mode": "neural_agentic"
                }
                return final_content, executed_tools, meta

            # Append the assistant's tool-call request to messages
            messages.append(response_msg)

            # Execute all requested tools
            for tc in tool_calls:
                func_data = tc.function if hasattr(tc, "function") else tc.get("function", {})
                tool_name = func_data.name if hasattr(func_data, "name") else func_data.get("name", "")
                tool_args = func_data.arguments if hasattr(func_data, "arguments") else func_data.get("arguments", {})

                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except Exception:
                        tool_args = {}

                try:
                    result, duration = _execute_tool_call(tool_name, tool_args)
                    executed_tools.append({
                        "tool_name": tool_name,
                        "args": tool_args,
                        "result": result,
                        "duration_s": duration,
                        "status": "success"
                    })
                    content_str = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
                except Exception as err:
                    executed_tools.append({
                        "tool_name": tool_name,
                        "args": tool_args,
                        "result": str(err),
                        "duration_s": 0.0,
                        "status": "error"
                    })
                    content_str = f"Tool execution error: {err}"

                # Append tool response
                messages.append({
                    "role": "tool",
                    "content": content_str
                })

        # Final turn completion
        final_response = client.chat(model=selected_model, messages=messages)
        content = final_response.message.content if hasattr(final_response, "message") else final_response.get("message", {}).get("content", "")
        # Extract focus coordinates from executed tools
        focus_coords = None
        for t in executed_tools:
            res = t.get("result", {})
            if isinstance(res, dict) and "coordinates" in res:
                c = res["coordinates"]
                if isinstance(c, dict):
                    lat = c.get("latitude", c.get("lat"))
                    lon = c.get("longitude", c.get("lon"))
                    if lat is not None and lon is not None:
                        focus_coords = {"lat": float(lat), "lon": float(lon)}
                        break

        meta = {
            "engine": f"Ollama ({selected_model})",
            "tool_count": len(executed_tools),
            "execution_mode": "neural_agentic",
            "focus_coordinates": focus_coords
        }
        return content, executed_tools, meta

    except Exception as exc:
        print(f"[SafeHilsa Agent Warning] Ollama invocation error: {exc}. Reverting to deterministic agent.")
        return _run_deterministic_agent(user_prompt, current_lat, current_lon, chat_history, date_str=date_str)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Testing agent.py...")
    prompt = "Is my current location safe for fishing? What is the legal size limit and penalty?"
    ans, tools, metadata = run_safehilsa_agent(prompt, current_lat=24.8100, current_lon=87.9250)
    print("Metadata:", metadata)
    print("Tools executed:", [t["tool_name"] for t in tools])
    print("\nAnswer preview:\n", ans[:400], "...")
