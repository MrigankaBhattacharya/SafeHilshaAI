"""
SafeHilsa AI - Agentic Loop & Tool Execution Engine
Connects to local Ollama (Llama 3, Qwen2.5, etc.) with native function calling.
Executes deterministic Python tools (geospatial checks, RAG search, river metrics)
and provides fallback reasoning when Ollama is offline.
"""

import json
import time
from typing import List, Dict, Any, Tuple, Optional

from tools import (
    AVAILABLE_TOOLS,
    OLLAMA_TOOLS_SCHEMA,
    check_location_compliance,
    search_regulations_tool,
    get_river_metrics,
    get_alternative_species_info
)

SYSTEM_PROMPT = """You are SafeHilsa AI, a Principal Riverine Compliance Guardian and Marine Ecologist for the Department of Fisheries (West Bengal & Bangladesh Riverine Systems).
Your mission is to protect Hilsa fish (Tenualosa ilisha) populations, prevent illegal harvesting in designated sanctuaries (such as juvenile 'Jatka' zones and spawning corridors), and guide artisanal fishermen toward legal waters and sustainable alternative catches (Rui, Catla, Tilapia, Pangas).

CORE COMPLIANCE RULES:
1. Minimum Legal Size: It is strictly ILLEGAL to catch, transport, or sell any Hilsa smaller than 25 cm (Jatka).
2. Spawning Ban: Strict 22-day blanket ban during peak October spawning (Ashvin-Kartik lunar window).
3. Jatka Ban: Active November to May, intensified zero-tolerance vigil during March and April.
4. Banned Gear: Monofilament nylon gill nets ('Current Jal') with mesh under 90mm are strictly banned under penalty of net confiscation, boat seizure, and fines up to ₹25,000.
5. Sanctuaries: Farakka Barrage (5km up/downstream), Katwa-Nabadwip, Diamond Harbour-Sagar Island, and Sundarbans Matla Delta are strictly restricted zones.

TOOL USAGE PROTOCOL:
- When the user asks about their current coordinates, location legality, or whether they can fish: ALWAYS call `check_location_compliance(lat, lon)`.
- When the user asks about fisheries laws, ban dates, catch size limits, mesh rules, or legal penalties: ALWAYS call `search_regulations_tool(query)`.
- When the user asks about water quality, discharge, dissolved oxygen, or river conditions: ALWAYS call `get_river_metrics(station_id)`.
- When the user asks what else they can catch during bans or in restricted areas: ALWAYS call `get_alternative_species_info(species_name)`.

RESPONSE STYLE:
- Authoritative, clear, respectful to artisanal fishermen, and grounded in official law.
- Always highlight whether a location is SAFE or RESTRICTED clearly.
- Provide practical guidance and mention sustainable alternatives whenever restrictions apply.
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
    current_lat: float,
    current_lon: float,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deterministic rule-based agentic fallback.
    Executes actual tools based on semantic intent and coordinates when Ollama is offline.
    """
    executed_tools = []
    prompt_lower = user_prompt.lower()
    
    # 1. Location compliance intent
    coords_in_prompt = ("location" in prompt_lower or "here" in prompt_lower or "safe" in prompt_lower 
                        or "coordinate" in prompt_lower or "gps" in prompt_lower or "legal to fish" in prompt_lower
                        or "can i fish" in prompt_lower or "sanctuary" in prompt_lower)
    
    loc_result = None
    if coords_in_prompt or (current_lat is not None and current_lon is not None and ("lat" in prompt_lower or "lon" in prompt_lower or "where" in prompt_lower)):
        res, dur = _execute_tool_call("check_location_compliance", {"lat": current_lat, "lon": current_lon})
        loc_result = res
        executed_tools.append({
            "tool_name": "check_location_compliance",
            "args": {"lat": current_lat, "lon": current_lon},
            "result": res,
            "duration_s": dur,
            "status": "success"
        })

    # 2. Regulations / Size / Ban intent
    reg_result = None
    if any(k in prompt_lower for k in ["size", "25", "ban", "jatka", "law", "fine", "penalty", "illegal", "current jal", "net", "spawning", "act"]):
        res, dur = _execute_tool_call("search_regulations_tool", {"query": user_prompt})
        reg_result = res
        executed_tools.append({
            "tool_name": "search_regulations_tool",
            "args": {"query": user_prompt},
            "result": res,
            "duration_s": dur,
            "status": "success"
        })

    # 3. River metrics intent
    metrics_result = None
    if any(k in prompt_lower for k in ["discharge", "water", "turbidity", "oxygen", "do", "cumec", "temperature", "river", "flow", "metric"]):
        station = "farakka"
        if "kolkata" in prompt_lower or "howrah" in prompt_lower:
            station = "kolkata_hooghly"
        elif "diamond" in prompt_lower:
            station = "diamond_harbour"
        elif "sagar" in prompt_lower or "bay" in prompt_lower:
            station = "sagar_island"
        
        res, dur = _execute_tool_call("get_river_metrics", {"station_id": station})
        metrics_result = res
        executed_tools.append({
            "tool_name": "get_river_metrics",
            "args": {"station_id": station},
            "result": res,
            "duration_s": dur,
            "status": "success"
        })

    # 4. Alternative species intent
    alt_result = None
    if any(k in prompt_lower for k in ["alternative", "other fish", "rui", "rohu", "catla", "tilapia", "pangas", "bata", "what else"]):
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
        if loc_result["status"] == "RESTRICTED":
            response_parts.append(
                f"### 🚨 RESTRICTED SANCTUARY DETECTED\n"
                f"Your selected coordinates (**{current_lat:.4f}°N, {current_lon:.4f}°E**) fall inside the **{loc_result['zone_name']}** ({loc_result['river']}).\n\n"
                f"- **Status:** Prohibited Waterway\n"
                f"- **Legal Basis:** {loc_result['legal_basis']}\n"
                f"- **Penalties:** {loc_result['penalty']}\n"
                f"- **Action Required:** {loc_result['recommended_action']}\n"
            )
        else:
            response_parts.append(
                f"### ✅ SAFE FISHING WATERS\n"
                f"Your coordinates (**{current_lat:.4f}°N, {current_lon:.4f}°E**) are outside restricted sanctuary boundaries.\n\n"
                f"- **Nearest Sanctuary:** {loc_result.get('nearest_sanctuary')} (~{loc_result.get('distance_to_nearest_sanctuary_km')} km away)\n"
                f"- **Advisory:** You are permitted to fish here, provided all Hilsa caught are **greater than 25 cm (9.84 in)** and no monofilament Current Jal is used.\n"
            )

    if reg_result:
        response_parts.append(
            f"### 📜 Applicable Fisheries Regulations\n"
            f"Based on grounded West Bengal Inland Fisheries legal statutes:\n\n"
            f"{reg_result}\n"
        )

    if metrics_result:
        response_parts.append(
            f"### 🌊 River Telemetry & Hydrology ({metrics_result['station_name']})\n"
            f"- **Discharge:** {metrics_result['discharge_cumecs']} cumecs\n"
            f"- **Dissolved Oxygen:** {metrics_result['dissolved_oxygen_mg_l']} mg/L (Min optimal: >5.0 mg/L)\n"
            f"- **Turbidity:** {metrics_result['turbidity_ntu']} NTU\n"
            f"- **Water Temp:** {metrics_result['water_temperature_c']} °C\n"
            f"- **Salinity:** {metrics_result['salinity_ppt']} ppt\n"
            f"- **Hilsa Migration Suitability:** **{metrics_result['hilsa_migration_suitability']}**\n"
            f"- **Ecological Note:** {metrics_result['ecological_notes']}\n"
        )

    if alt_result:
        response_parts.append(
            f"{alt_result}\n"
        )

    meta = {
        "engine": "SafeHilsa Deterministic Fallback Agent (Ollama Offline)",
        "tool_count": len(executed_tools),
        "execution_mode": "local_fast"
    }

    final_text = "\n\n".join(response_parts)
    return final_text, executed_tools, meta


def run_safehilsa_agent(
    user_prompt: str,
    current_lat: float,
    current_lon: float,
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_name: str = "llama3.2",
    host: str = "http://localhost:11434"
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

    Returns:
        Tuple of (response_text, executed_tools_trace, metadata)
    """
    executed_tools = []

    # 1. Test Ollama connectivity
    status = check_ollama_status(host=host)
    if not status["online"] or not status["models"]:
        # Fallback to local deterministic agent
        return _run_deterministic_agent(user_prompt, current_lat, current_lon, chat_history)

    # 2. Check if requested model or a suitable alternative exists
    available = status["models"]
    selected_model = model_name
    if not any(selected_model in m for m in available):
        # Pick first available or fallback
        if available:
            selected_model = available[0]
        else:
            return _run_deterministic_agent(user_prompt, current_lat, current_lon, chat_history)

    try:
        import ollama
        client = ollama.Client(host=host)

        # Context-augmented system message
        contextual_system = (
            f"{SYSTEM_PROMPT}\n\n"
            f"CURRENT USER CONTEXT:\n"
            f"- GPS Latitude: {current_lat:.4f}\n"
            f"- GPS Longitude: {current_lon:.4f}\n"
            f"If the user asks whether their location is safe, use these coordinates with check_location_compliance."
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
        meta = {
            "engine": f"Ollama ({selected_model})",
            "tool_count": len(executed_tools),
            "execution_mode": "neural_agentic"
        }
        return content, executed_tools, meta

    except Exception as exc:
        print(f"[SafeHilsa Agent Warning] Ollama invocation error: {exc}. Reverting to deterministic agent.")
        return _run_deterministic_agent(user_prompt, current_lat, current_lon, chat_history)


if __name__ == "__main__":
    print("Testing agent.py...")
    prompt = "Is my current location safe for fishing? What are the penalties if I catch small Jatka?"
    ans, tools, metadata = run_safehilsa_agent(prompt, current_lat=24.8100, current_lon=87.9250)
    print("Metadata:", metadata)
    print("Tools executed:", [t["tool_name"] for t in tools])
    print("\nAnswer preview:\n", ans[:300], "...")
