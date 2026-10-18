"""
SafeHilsa AI - Streamlit Dashboard & Agentic Compliance Hub
Interactive geospatial conservation interface with Folium map overlays,
real-time sanctuary checks, environmental telemetry, and local Ollama agent chat.
"""

import os
import json
import streamlit as st
import folium
from streamlit_folium import st_folium

# Import backend modules
from tools import (
    check_location_compliance,
    get_river_metrics,
    get_alternative_species_info,
    SANCTUARIES_GEOJSON_PATH
)
from agent import run_safehilsa_agent, check_ollama_status
from rag_engine import initialize_rag, search_regulations_structured

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SafeHilsa AI - Riverine Compliance & Conservation Guardian",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM MODERN RIVERINE THEME CSS ---
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #F8FAFC;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0284C7 100%);
        padding: 24px;
        border-radius: 14px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        max-width: 900px;
    }
    
    /* Compliance Alert Banners */
    .status-card-restricted {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 16px;
        color: #991B1B;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.15);
    }
    .status-card-safe {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 16px;
        color: #065F46;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.15);
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 12px 14px;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0284C7;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    /* Tool Execution Badges */
    .tool-badge {
        display: inline-flex;
        align-items: center;
        background-color: #EFF6FF;
        border: 1px solid #93C5FD;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Preset Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = 24.8100  # Default near Farakka
if "selected_lon" not in st.session_state:
    st.session_state.selected_lon = 87.9250
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to SafeHilsa AI Compliance Guardian!**\n\n"
                "I am your local AI compliance officer and river ecologist. "
                "I monitor protected sanctuaries, verify fishing coordinates, enforce the **25 cm minimum legal size**, "
                "track seasonal spawning bans, and suggest legal alternative catches (Rui, Catla, Tilapia).\n\n"
                "👉 Click anywhere on the river map to test your GPS coordinates, or ask any legal question below!"
            ),
            "tools": []
        }
    ]
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False


# Lazy RAG Indexing
@st.cache_resource
def get_rag_collection():
    return initialize_rag()

try:
    _ = get_rag_collection()
    st.session_state.rag_ready = True
except Exception as e:
    st.session_state.rag_ready = False

# --- PRESET HOTSPOTS ---
PRESETS = {
    "Farakka Barrage (Restricted Sanctuary)": (24.8100, 87.9250),
    "Katwa Spawning Grounds (Restricted)": (23.6400, 88.1400),
    "Diamond Harbour Estuary (Restricted)": (22.1850, 88.1900),
    "Tribeni Ghat (Permissible Legal Waters)": (22.9850, 88.4050),
    "Kolkata Hooghly (Permissible Legal Waters)": (22.5600, 88.3300)
}

# --- SIDEBAR: TELEMETRY & SYSTEM CONFIG ---
with st.sidebar:
    st.markdown("### 🌊 SafeHilsa Control Station")
    st.caption("Local-First Riverine AI & Environmental Telemetry")
    
    # 1. River Station Telemetry
    st.markdown("#### 📡 River Telemetry & Hydrology")
    station_selection = st.selectbox(
        "Monitoring Station",
        options=["farakka", "kolkata_hooghly", "diamond_harbour", "sagar_island"],
        format_func=lambda x: {
            "farakka": "Farakka Barrage (Ganges)",
            "kolkata_hooghly": "Kolkata Riverfront (Hooghly)",
            "diamond_harbour": "Diamond Harbour (Estuary)",
            "sagar_island": "Sagar Island (Bay Ingress)"
        }.get(x, x)
    )
    
    metrics = get_river_metrics(station_selection)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['discharge_cumecs']:,.0f}</div>
            <div class="metric-label">Discharge (cumecs)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['dissolved_oxygen_mg_l']}</div>
            <div class="metric-label">Dissolved O₂ (mg/L)</div>
        </div>
        """, unsafe_allow_html=True)
        
    col_m3, col_m4 = st.columns(2)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card" style="margin-top:8px;">
            <div class="metric-value">{metrics['turbidity_ntu']}</div>
            <div class="metric-label">Turbidity (NTU)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-card" style="margin-top:8px;">
            <div class="metric-value">{metrics['salinity_ppt']}</div>
            <div class="metric-label">Salinity (ppt)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"**Migration Suitability:** `{metrics['hilsa_migration_suitability']}`")
    st.caption(f"ℹ️ {metrics['ecological_notes']}")
    
    st.divider()
    
    # 2. LLM Engine & Ollama Configuration
    st.markdown("#### 🧠 Local LLM Engine")
    ollama_host = st.text_input("Ollama Host URL", value="http://localhost:11434")
    ollama_info = check_ollama_status(host=ollama_host)
    
    if ollama_info["online"]:
        st.success(f"🟢 Ollama Online ({len(ollama_info['models'])} models found)")
        model_options = ollama_info["models"] if ollama_info["models"] else ["llama3.2", "qwen2.5"]
        selected_model = st.selectbox("Active Neural Model", options=model_options)
    else:
        st.warning("🟡 Ollama Offline (Using Fast Local Fallback Agent)")
        st.caption("Start Ollama (`ollama run llama3.2`) for full neural generation.")
        selected_model = "llama3.2"

    st.divider()

    # 3. Knowledge Base Status
    st.markdown("#### 📚 RAG Regulatory Vector Store")
    if st.session_state.rag_ready:
        st.success("🟢 Regulations Indexed (Local ChromaDB)")
    else:
        st.info("🟡 In-Memory TF-IDF Fallback Vector Store Active")
        
    if st.button("🔄 Re-index Legal Regulations", use_container_width=True):
        with st.spinner("Re-indexing regulations.md into ChromaDB..."):
            initialize_rag(force_reindex=True)
            st.session_state.rag_ready = True
            st.success("Vector store refreshed successfully!")
            st.rerun()

    st.divider()
    
    # 4. Manual Coordinate Input
    st.markdown("#### 📍 Manual Coordinates")
    manual_lat = st.number_input("Latitude", value=float(st.session_state.selected_lat), format="%.4f", step=0.005)
    manual_lon = st.number_input("Longitude", value=float(st.session_state.selected_lon), format="%.4f", step=0.005)
    if (manual_lat != st.session_state.selected_lat) or (manual_lon != st.session_state.selected_lon):
        if st.button("Apply Coordinates", use_container_width=True):
            st.session_state.selected_lat = manual_lat
            st.session_state.selected_lon = manual_lon
            st.rerun()

    st.caption("Protected Species: *Tenualosa ilisha* | Legal Standard: W.B. Inland Fisheries Act, 1984")


# --- MAIN INTERFACE: HEADER ---
st.markdown("""
<div class="header-box">
    <div class="header-title">
        <span>🐟 SafeHilsa AI</span>
        <span style="font-size:1.1rem; background:rgba(255,255,255,0.2); padding:4px 12px; border-radius:20px;">Local Agentic Compliance Guardian</span>
    </div>
    <div class="header-subtitle">
        Protecting the queen of fish (Hilsa / Ilish) across the Ganges-Bhagirathi-Hooghly delta. 
        Autonomous sanctuary geo-fencing, legal size (>25cm) compliance, seasonal ban enforcement, and sustainable alternative catch navigation.
    </div>
</div>
""", unsafe_allow_html=True)


# --- TWO-COLUMN WORKSPACE ---
col_map, col_chat = st.columns([1.15, 1.0], gap="large")

# ==========================================
# LEFT COLUMN: INTERACTIVE FOLIUM MAP & GEO
# ==========================================
with col_map:
    st.markdown("### 🗺️ River Sanctuary Geospatial Radar")
    st.caption("Click anywhere on the river system to inspect compliance, or use quick presets below:")

    # Quick Jump Buttons
    preset_cols = st.columns(3)
    preset_keys = list(PRESETS.keys())
    with preset_cols[0]:
        if st.button("📍 Farakka Barrage", use_container_width=True):
            st.session_state.selected_lat, st.session_state.selected_lon = PRESETS["Farakka Barrage (Restricted Sanctuary)"]
            st.rerun()
    with preset_cols[1]:
        if st.button("📍 Katwa Sanctuary", use_container_width=True):
            st.session_state.selected_lat, st.session_state.selected_lon = PRESETS["Katwa Spawning Grounds (Restricted)"]
            st.rerun()
    with preset_cols[2]:
        if st.button("📍 Tribeni (Legal)", use_container_width=True):
            st.session_state.selected_lat, st.session_state.selected_lon = PRESETS["Tribeni Ghat (Permissible Legal Waters)"]
            st.rerun()

    # Instant Compliance Evaluation
    compliance = check_location_compliance(st.session_state.selected_lat, st.session_state.selected_lon)

    # Status Alert Widget
    if compliance["status"] == "RESTRICTED":
        st.markdown(f"""
        <div class="status-card-restricted">
            <div style="font-size:1.15rem; font-weight:800; display:flex; align-items:center; gap:8px;">
                <span>🚨 RESTRICTED SANCTUARY DETECTED</span>
            </div>
            <div style="margin-top:6px; font-size:0.95rem;">
                <b>Zone:</b> {compliance['zone_name']} ({compliance['river']})<br>
                <b>Status:</b> Strict No-Fishing Zone | <b>Legal Basis:</b> {compliance['legal_basis']}<br>
                <b>Penalties:</b> {compliance['penalty']}
            </div>
            <div style="margin-top:8px; font-size:0.85rem; background:rgba(239,68,68,0.15); padding:8px; border-radius:6px;">
                ⚠️ <b>Directive:</b> {compliance['recommended_action']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-card-safe">
            <div style="font-size:1.15rem; font-weight:800; display:flex; align-items:center; gap:8px;">
                <span>✅ PERMISSIBLE LEGAL FISHING WATERS</span>
            </div>
            <div style="margin-top:6px; font-size:0.95rem;">
                <b>Selected Coordinates:</b> {st.session_state.selected_lat:.4f}°N, {st.session_state.selected_lon:.4f}°E<br>
                <b>Nearest Sanctuary:</b> {compliance.get('nearest_sanctuary')} (~{compliance.get('distance_to_nearest_sanctuary_km')} km away)<br>
                <b>Permissible Catch:</b> Adult Hilsa (Length &gt; 25 cm only) & Freshwater Carps (Rui, Catla, Tilapia)
            </div>
            <div style="margin-top:8px; font-size:0.85rem; background:rgba(16,185,129,0.15); padding:8px; border-radius:6px;">
                ℹ️ <b>Reminder:</b> Catching Jatka (&lt; 25cm) or deploying monofilament Current Jal is strictly banned.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Create Folium Map
    center_lat = st.session_state.selected_lat
    center_lon = st.session_state.selected_lon
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Add CartoDB Positron as alternative tile layer
    folium.TileLayer("CartoDB positron", name="Clean Nautical Vector").add_to(m)

    # Add GeoJSON sanctuaries overlay
    if os.path.exists(SANCTUARIES_GEOJSON_PATH):
        with open(SANCTUARIES_GEOJSON_PATH, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        def style_function(feature):
            return {
                "fillColor": "#DC2626",
                "color": "#991B1B",
                "weight": 2.5,
                "fillOpacity": 0.35,
                "dashArray": "5, 5"
            }

        folium.GeoJson(
            geojson_data,
            name="Restricted Hilsa Sanctuaries",
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "status", "river", "fine"],
                aliases=["Sanctuary:", "Status:", "River:", "Penalties:"],
                style="background-color: white; color: #1E293B; font-weight: bold; border-radius: 4px; padding: 6px;"
            ),
            popup=folium.GeoJsonPopup(
                fields=["name", "status", "zone_type", "fine", "legal_basis", "description"],
                aliases=["Zone:", "Status:", "Type:", "Fines:", "Law:", "Details:"]
            )
        ).add_to(m)

    # Add Selected Point Marker
    marker_color = "red" if compliance["status"] == "RESTRICTED" else "green"
    marker_icon = "ban" if compliance["status"] == "RESTRICTED" else "ok"
    
    folium.Marker(
        [center_lat, center_lon],
        popup=f"Selected GPS: {center_lat:.4f}, {center_lon:.4f} ({compliance['status']})",
        tooltip=f"Active Location ({compliance['status']})",
        icon=folium.Icon(color=marker_color, icon=marker_icon, prefix="glyphicon")
    ).add_to(m)

    # Circle radius around selected point
    folium.Circle(
        location=[center_lat, center_lon],
        radius=5000,  # 5 km buffer
        color=marker_color,
        weight=1,
        fill=True,
        fill_opacity=0.1
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # Render Map and Capture Clicks
    map_output = st_folium(
        m,
        height=480,
        width="100%",
        returned_objects=["last_clicked"]
    )

    if map_output and map_output.get("last_clicked"):
        click_lat = round(map_output["last_clicked"]["lat"], 4)
        click_lon = round(map_output["last_clicked"]["lng"], 4)
        
        # Check if new point clicked
        if (click_lat != st.session_state.selected_lat) or (click_lon != st.session_state.selected_lon):
            st.session_state.selected_lat = click_lat
            st.session_state.selected_lon = click_lon
            st.rerun()


# ==========================================
# RIGHT COLUMN: AGENTIC CHAT INTERFACE
# ==========================================
with col_chat:
    st.markdown("### 💬 SafeHilsa AI Agentic Assistant")
    st.caption("Ask compliance questions, verify catch regulations, or plan sustainable alternative trips.")

    # Quick prompt buttons
    st.markdown("**Quick Inquiries:**")
    prompt_cols = st.columns(2)
    with prompt_cols[0]:
        if st.button("🧭 Is my GPS location legal to fish?", use_container_width=True):
            st.session_state.user_query_input = "Is it legal to fish at my currently selected GPS coordinates? What restrictions apply?"
    with prompt_cols[1]:
        if st.button("📏 Minimum legal catch size?", use_container_width=True):
            st.session_state.user_query_input = "What is the official minimum legal catch size for Hilsa and what happens if someone catches Jatka?"
    
    prompt_cols_2 = st.columns(2)
    with prompt_cols_2[0]:
        if st.button("🛑 When is the official ban period?", use_container_width=True):
            st.session_state.user_query_input = "When are the mandatory seasonal ban periods for Hilsa in West Bengal?"
    with prompt_cols_2[1]:
        if st.button("🐟 What alternative fish can I catch?", use_container_width=True):
            st.session_state.user_query_input = "What legal alternative fish species can I harvest during the ban period instead of Hilsa?"

    # Render Chat History
    chat_container = st.container(height=430)
    with chat_container:
        for msg in st.session_state.messages:
            avatar = "🐟" if msg["role"] == "assistant" else "🧑‍🌾"
            with st.chat_message(msg["role"], avatar=avatar):
                # Render tool execution badges if any
                if msg.get("tools"):
                    for t in msg["tools"]:
                        tool_name = t.get("tool_name", "tool")
                        dur = t.get("duration_s", 0.0)
                        st.markdown(f'<span class="tool-badge">⚡ Tool Executed: <b>{tool_name}</b> ({dur}s)</span>', unsafe_allow_html=True)
                        with st.expander(f"🔍 Trace: `{tool_name}` arguments & output", expanded=False):
                            st.json({
                                "arguments": t.get("args"),
                                "output": t.get("result")
                            })
                st.markdown(msg["content"])

    # Chat Input Box
    query_to_submit = st.chat_input("Ask SafeHilsa AI about compliance, ban dates, coordinates, or alternative species...")
    
    # Handle quick button inputs
    if "user_query_input" in st.session_state and st.session_state.user_query_input:
        query_to_submit = st.session_state.user_query_input
        st.session_state.user_query_input = None

    if query_to_submit:
        # 1. Add user message to state
        st.session_state.messages.append({"role": "user", "content": query_to_submit})
        
        # 2. Render user message in chat
        with chat_container:
            with st.chat_message("user", avatar="🧑‍🌾"):
                st.markdown(query_to_submit)

            # 3. Run Agentic Loop with Spinner
            with st.chat_message("assistant", avatar="🐟"):
                with st.spinner("SafeHilsa AI reasoning and inspecting geospatial registers..."):
                    history_payload = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    
                    response_text, tool_trace, meta = run_safehilsa_agent(
                        user_prompt=query_to_submit,
                        current_lat=st.session_state.selected_lat,
                        current_lon=st.session_state.selected_lon,
                        chat_history=history_payload,
                        model_name=selected_model,
                        host=ollama_host
                    )
                    
                    # Display tool execution badges
                    if tool_trace:
                        for t in tool_trace:
                            st.markdown(f'<span class="tool-badge">⚡ Tool Executed: <b>{t["tool_name"]}</b> ({t.get("duration_s", 0)}s)</span>', unsafe_allow_html=True)
                            with st.expander(f"🔍 Trace: `{t['tool_name']}` arguments & output", expanded=False):
                                st.json({
                                    "arguments": t.get("args"),
                                    "output": t.get("result")
                                })

                    st.markdown(response_text)
                    st.caption(f"⚙️ Engine: {meta.get('engine', 'SafeHilsa')} | Tools Run: {meta.get('tool_count', 0)}")

        # 4. Save assistant response to state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "tools": tool_trace
        })
        st.rerun()
