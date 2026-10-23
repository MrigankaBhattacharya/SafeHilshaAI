"""
SafeHilsa AI - Streamlit Dashboard & Riverine Compliance Hub (Phase 5 Overhaul)
Interactive geospatial conservation interface with Folium map overlays,
authoritative sanctuary checks, historical environmental observations, and local AI agent chat.
Grounded in Kolkata Gazette Notification No. 719-Fish & West Bengal Inland Fisheries Act, 1984.
"""

import os
import json
from typing import Dict, Any, Optional

try:
    import streamlit as st
    import folium
    from streamlit_folium import st_folium
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    folium = None
    st_folium = None
    STREAMLIT_AVAILABLE = False

# Import verified backend modules
from tools import (
    check_location_compliance,
    get_river_metrics,
    get_alternative_species_info,
    get_livelihood_support,
    SANCTUARIES_GEOJSON_PATH,
    FARAKKA_BARRAGE_LAT,
    FARAKKA_BARRAGE_LON,
    FARAKKA_STATUTORY_RADIUS_KM,
    NOTIFIED_REACHES,
    SUNDARBANS_SANDBARS,
    VERIFIED_HOOGHLY_BASELINE,
    fetch_live_river_discharge
)
from agent import run_safehilsa_agent, check_ollama_status, SYSTEM_PROMPT
from rag_engine import initialize_rag

# --- PRESET HOTSPOTS ---
PRESETS = {
    "Farakka Barrage (Rule 50(3) Statutory Area)": (24.8100, 87.9250),
    "Katwa Sanctuary Reach (Rule 50(1)(b))": (23.5000, 88.2500),
    "Diamond Harbour Estuary Reach (Rule 50(1)(c))": (22.2800, 88.1750),
    "Lalbagh Reach Corridor (Rule 50(1)(a))": (24.0950, 88.4830),
    "Tribeni Ghat (Outside Checked Sanctuaries)": (22.9800, 88.4000),
    "Kolkata Riverfront (Outside Checked Sanctuaries)": (22.5600, 88.3300)
}


# --- HELPER: FORMAT COMPLIANCE STATUS CARD ---
def format_compliance_card_html(compliance: Dict[str, Any], lat: float, lon: float) -> str:
    """
    Renders distinct HTML status cards for the 4 geographic outcomes:
    1. INCONCLUSIVE_AREA_BOUNDARY (Farakka vicinity)
    2. ADVISORY_SANCTUARY_REACH (River reach seasonal ban)
    3. PERMISSIBLE_SEASONAL_WINDOW (Inactive seasonal window)
    4. OUTSIDE_CHECKED_SANCTUARIES (Limited geographic finding, NOT legal clearance)
    """
    status_str = str(compliance.get("status", ""))
    det_str = str(compliance.get("determination", ""))

    if status_str == "INCONCLUSIVE_AREA_BOUNDARY" or det_str == "INCONCLUSIVE_AREA_BOUNDARY":
        return f"""
        <div class="status-card-inconclusive">
            <div style="font-size:1.1rem; font-weight:800; display:flex; align-items:center; gap:8px;">
                <span>⚠️ INCONCLUSIVE BOUNDARY / HIGH ENFORCEMENT RISK</span>
            </div>
            <div style="margin-top:6px; font-size:0.92rem;">
                <b>Location:</b> {lat:.4f}°N, {lon:.4f}°E | <b>Vicinity:</b> {compliance.get('zone_name')}<br>
                <b>Determination:</b> <code>INCONCLUSIVE_AREA_BOUNDARY</code> (Definitive legal inclusion is legally inconclusive)<br>
                <b>Statutory Basis:</b> {compliance.get('legal_basis')}<br>
                <b>Cadastral Limitation:</b> {compliance.get('missing_boundary_information')}
            </div>
            <div style="margin-top:8px; font-size:0.85rem; background:rgba(245,158,11,0.18); padding:8px; border-radius:6px;">
                <b>Directive:</b> {compliance.get('recommended_action')}
            </div>
            <div style="margin-top:6px; font-size:0.78rem; color:#78350F;">
                <b>Statutory Penalties:</b> {compliance.get('penalty')}
            </div>
        </div>
        """
    elif status_str in ["ADVISORY_SANCTUARY_REACH", "RESTRICTED"] and compliance.get("is_legal") is not True:
        return f"""
        <div class="status-card-restricted">
            <div style="font-size:1.1rem; font-weight:800; display:flex; align-items:center; gap:8px;">
                <span>🚨 ADVISORY: ACTIVE SEASONAL SANCTUARY REACH</span>
            </div>
            <div style="margin-top:6px; font-size:0.92rem;">
                <b>Zone:</b> {compliance.get('zone_name')} ({compliance.get('river', 'Bhagirathi-Hooghly')})<br>
                <b>Determination:</b> <code>ADVISORY_SANCTUARY_REACH</code> (Seasonal Ban: June–August & October–December)<br>
                <b>Statutory Basis:</b> {compliance.get('legal_basis')}<br>
                <b>Lateral Limitation:</b> {compliance.get('lateral_boundary_limitation')}
            </div>
            <div style="margin-top:8px; font-size:0.85rem; background:rgba(239,68,68,0.15); padding:8px; border-radius:6px;">
                <b>Directive:</b> {compliance.get('recommended_action')}
            </div>
            <div style="margin-top:6px; font-size:0.78rem; color:#7F1D1D;">
                <b>Statutory Penalties:</b> {compliance.get('penalty')}
            </div>
        </div>
        """
    elif status_str == "PERMISSIBLE_SEASONAL_WINDOW" or compliance.get("determination") == "PERMISSIBLE_WITH_GEAR_RESTRICTION":
        return f"""
        <div class="status-card-permissible">
            <div style="font-size:1.1rem; font-weight:800; display:flex; align-items:center; gap:8px;">
                <span>ℹ️ PERMISSIBLE SEASONAL WINDOW (Sanctuary Ban Currently Inactive)</span>
            </div>
            <div style="margin-top:6px; font-size:0.92rem;">
                <b>Sanctuary Reach:</b> {compliance.get('zone_name')}<br>
                <b>Determination:</b> <code>PERMISSIBLE_WITH_GEAR_RESTRICTION</code><br>
                <b>Statutory Conditions:</b> {compliance.get('description')}
            </div>
            <div style="margin-top:8px; font-size:0.85rem; background:rgba(22,163,74,0.15); padding:8px; border-radius:6px;">
                <b>Advisory:</b> {compliance.get('recommended_action')}
            </div>
        </div>
        """
    else:
        # OUTSIDE_CHECKED_SANCTUARIES: MUST NEVER BE DISPLAYED AS DEFINITIVE LEGAL CLEARANCE
        return f"""
        <div class="status-card-outside">
            <div style="font-size:1.1rem; font-weight:800; display:flex; align-items:center; gap:8px; color:#1E293B;">
                <span>📍 OUTSIDE CHECKED SANCTUARIES (Limited Geographic Finding)</span>
            </div>
            <div style="margin-top:6px; font-size:0.90rem;">
                <b>Selected Coordinates:</b> {lat:.4f}°N, {lon:.4f}°E<br>
                <b>Legal Clearance Status:</b> <span style="color:#DC2626; font-weight:bold;">NOT a definitive legal clearance</span> (<code>is_legal: null</code>)<br>
                <b>Nearest Checked Sanctuary:</b> {compliance.get('nearest_sanctuary')} (~{compliance.get('distance_to_nearest_sanctuary_km')} km away)<br>
                <b>Scope Limitation:</b> {compliance.get('scope_limitation')}
            </div>
            <div style="margin-top:8px; font-size:0.82rem; background:rgba(100,116,139,0.12); padding:8px; border-radius:6px; color:#0F172A;">
                <b>Statutory Requirements (Rule 46):</b> {compliance.get('statutory_reminders')}
            </div>
            <div style="margin-top:6px; font-size:0.80rem; color:#475569;">
                <b>Advisory:</b> {compliance.get('recommended_action')}
            </div>
        </div>
        """


def render_app():
    """Renders the Streamlit SafeHilsa AI Dashboard."""
    if not STREAMLIT_AVAILABLE or st is None:
        print("Streamlit is not available in this environment.")
        return

    # --- PAGE CONFIGURATION ---
    st.set_page_config(
        page_title="SafeHilsa AI - Riverine Conservation & Compliance Guardian",
        page_icon="🐟",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- CUSTOM MODERN RIVERINE THEME CSS ---
    st.markdown("""
    <style>
        .main { background-color: #F8FAFC; }
        .header-box {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0284C7 100%);
            padding: 24px;
            border-radius: 14px;
            color: white;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .header-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-subtitle {
            font-size: 0.98rem;
            opacity: 0.92;
            max-width: 950px;
            line-height: 1.5;
        }
        .status-card-restricted {
            background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
            border: 2px solid #EF4444;
            border-radius: 12px;
            padding: 16px;
            color: #991B1B;
            margin-bottom: 15px;
            box-shadow: 0 4px 10px rgba(239, 68, 68, 0.15);
        }
        .status-card-inconclusive {
            background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
            border: 2px solid #F59E0B;
            border-radius: 12px;
            padding: 16px;
            color: #92400E;
            margin-bottom: 15px;
            box-shadow: 0 4px 10px rgba(245, 158, 11, 0.15);
        }
        .status-card-permissible {
            background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
            border: 2px solid #16A34A;
            border-radius: 12px;
            padding: 16px;
            color: #166534;
            margin-bottom: 15px;
            box-shadow: 0 4px 10px rgba(22, 163, 74, 0.15);
        }
        .status-card-outside {
            background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
            border: 2px solid #64748B;
            border-radius: 12px;
            padding: 16px;
            color: #334155;
            margin-bottom: 15px;
            box-shadow: 0 4px 10px rgba(100, 116, 139, 0.15);
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 10px 12px;
            border: 1px solid #E2E8F0;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        }
        .metric-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #0284C7;
        }
        .metric-label {
            font-size: 0.72rem;
            color: #64748B;
            text-transform: uppercase;
            font-weight: 600;
        }
        .historical-tag {
            display: inline-block;
            background-color: #FEF3C7;
            border: 1px solid #F59E0B;
            color: #92400E;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.73rem;
            font-weight: 700;
            margin-top: 6px;
        }
        .tool-badge {
            display: inline-flex;
            align-items: center;
            background-color: #EFF6FF;
            border: 1px solid #93C5FD;
            color: #1E40AF;
            padding: 3px 9px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        .scheme-card {
            background: #F8FAFC;
            border-left: 4px solid #0284C7;
            border-radius: 6px;
            padding: 10px 14px;
            margin-bottom: 10px;
            font-size: 0.85rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- SESSION STATE INITIALIZATION ---
    if "selected_lat" not in st.session_state:
        st.session_state.selected_lat = 24.8100
    if "selected_lon" not in st.session_state:
        st.session_state.selected_lon = 87.9250
    if "selected_date" not in st.session_state:
        st.session_state.selected_date = "2026-07-15"
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Welcome to SafeHilsa AI Riverine Conservation Hub!**\n\n"
                    "I am your local AI compliance guardian and fisheries advisor for West Bengal. "
                    "I enforce the statutory **23 cm minimum legal catch size** under *Kolkata Gazette Notification No. 719-Fish (Rule 46(4))*, "
                    "evaluate seasonal sanctuary bans (*Rule 50(2)*: June–August & October–December), provide historical river hydrology observations "
                    "(ICAR-CIFRI Bulletin 179), navigate state livelihood assistance schemes (**Samudra Sathi** & **Matsyajeebi Bandhu**), "
                    "and guide fishers to legal alternative catches (Rui, Catla, Tilapia).\n\n"
                    "👉 Click anywhere on the river map or select presets to evaluate GPS compliance, or ask any question below!"
                ),
                "tools": []
            }
        ]
    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = False

    # Lazy RAG Initialization
    @st.cache_resource
    def get_rag_collection():
        return initialize_rag()

    try:
        _ = get_rag_collection()
        st.session_state.rag_ready = True
    except Exception:
        st.session_state.rag_ready = False

    # --- SIDEBAR: TELEMETRY & SYSTEM CONFIG ---
    with st.sidebar:
        st.markdown("### 🌊 SafeHilsa Control Station")
        st.caption("Grounded in Kolkata Gazette No. 719-Fish & Local Datasets")
        
        # 1. Historical River Hydrology & Observations
        st.markdown("#### 📡 River Hydrology & Observations")
        st.markdown("""<div class="historical-tag">⚠️ HISTORICAL BASELINE OBSERVATION — NOT LIVE TELEMETRY</div>""", unsafe_allow_html=True)
        
        station_selection = st.selectbox(
            "Monitoring Station",
            options=[
                "farakka", "katwa", "nabadwip", "tribeni",
                "kolkata_hooghly", "diamond_harbour", "sagar_island", "fekoghat"
            ],
            format_func=lambda x: {
                "farakka": "Farakka Barrage (Ganges-Bhagirathi)",
                "katwa": "Katwa (Bhagirathi Reach)",
                "nabadwip": "Nabadwip (Bhagirathi-Hooghly)",
                "tribeni": "Tribeni (Freshwater Tidal Zone)",
                "kolkata_hooghly": "Kolkata Riverfront (Tidal Hooghly)",
                "diamond_harbour": "Diamond Harbour (Hooghly Estuary)",
                "sagar_island": "Sagar Island (Marine Ingress)",
                "fekoghat": "Fekoghat (Subarnarekha Basin - Isolated)"
            }.get(x, x)
        )
        
        basin_filter = "Subernarekha" if station_selection == "fekoghat" else "Ganges-Bhagirathi-Hooghly"
        metrics = get_river_metrics(station_selection, basin=basin_filter)
        
        if metrics.get("status") == "AVAILABLE":
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                discharge_val = f"{metrics.get('discharge_cumecs', 0):,.0f}" if metrics.get('discharge_cumecs') is not None else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{discharge_val}</div>
                    <div class="metric-label">Discharge (cumecs)</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('dissolved_oxygen_mg_l', 'N/A')}</div>
                    <div class="metric-label">Dissolved O₂ (mg/L)</div>
                </div>
                """, unsafe_allow_html=True)
                
            col_m3, col_m4 = st.columns(2)
            with col_m3:
                st.markdown(f"""
                <div class="metric-card" style="margin-top:6px;">
                    <div class="metric-value">{metrics.get('turbidity_ntu', 'N/A')}</div>
                    <div class="metric-label">Turbidity (NTU)</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m4:
                sal_val = metrics.get('salinity_ppt', 'N/A')
                st.markdown(f"""
                <div class="metric-card" style="margin-top:6px;">
                    <div class="metric-value">{sal_val}</div>
                    <div class="metric-label">Salinity (ppt)</div>
                </div>
                """, unsafe_allow_html=True)

            st.caption(
                f"**Basin:** `{metrics.get('basin')}` | **Source:** `{metrics.get('source_id')}`\n\n"
                f"**Survey Period:** {metrics.get('observation_period')}\n\n"
                f"ℹ️ {metrics.get('disclaimer')}"
            )
        else:
            st.warning(metrics.get("message", "Monitoring station data unavailable."))

        st.divider()

        # 2. State Livelihood Assistance Overview
        st.markdown("#### 🤝 Livelihood Assistance Schemes")
        with st.expander("🌊 Samudra Sathi Scheme (Marine Ban)", expanded=False):
            st.markdown("""
            <div class="scheme-card">
                <b>Benefit:</b> ₹5,000 / month Direct Benefit Transfer (DBT)<br>
                <b>Ban Period:</b> 15th April to 14th June (61-day Marine EEZ Ban)<br>
                <b>Eligible Districts:</b> Purba Medinipur, South 24 Parganas, North 24 Parganas<br>
                <b>Authority:</b> Notification No. 724-Fish (2024) | <code>SRC-REG-027</code><br>
                <i>Note: Exclusively covers registered coastal marine fishers. Inland fishers are excluded.</i>
            </div>
            """, unsafe_allow_html=True)
            
        with st.expander("🛡️ Matsyajeebi Bandhu Scheme (All Districts)", expanded=False):
            st.markdown("""
            <div class="scheme-card">
                <b>Benefit:</b> ₹2,00,000 (₹2 Lakh) one-time grant to nominee<br>
                <b>Coverage:</b> Death (natural or accidental) of registered fisher (aged 18–60)<br>
                <b>Eligible Districts:</b> All 23 Districts of West Bengal<br>
                <b>Authority:</b> <code>SRC-REG-028 / SRC-REG-029</code>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # 3. LLM Engine & Ollama Configuration
        st.markdown("#### 🧠 Local AI Engine")
        ollama_host = st.text_input("Ollama Host URL", value="http://localhost:11434")
        ollama_info = check_ollama_status(host=ollama_host)
        
        if ollama_info["online"]:
            st.success(f"🟢 Ollama Online ({len(ollama_info['models'])} models found)")
            model_options = ollama_info["models"] if ollama_info["models"] else ["llama3.2", "qwen2.5"]
            selected_model = st.selectbox("Active Model", options=model_options)
        else:
            st.warning("🟡 Ollama Offline (Using Fast Local Fallback Agent)")
            st.caption("Start Ollama (`ollama run llama3.2`) for neural function-calling.")
            selected_model = "llama3.2"

        st.divider()

        # 4. Manual Coordinate & Date Controls
        st.markdown("#### 📍 Inspection Coordinates")
        manual_lat = st.number_input("Latitude", value=float(st.session_state.selected_lat), format="%.4f", step=0.005)
        manual_lon = st.number_input("Longitude", value=float(st.session_state.selected_lon), format="%.4f", step=0.005)
        eval_date = st.date_input("Evaluation Date", value=None)
        eval_date_str = str(eval_date) if eval_date else None
        
        if (manual_lat != st.session_state.selected_lat) or (manual_lon != st.session_state.selected_lon):
            if st.button("Apply Coordinates", use_container_width=True):
                st.session_state.selected_lat = manual_lat
                st.session_state.selected_lon = manual_lon
                st.rerun()

        st.caption("Protected Species: *Tenualosa ilisha* | Primary Law: W.B. Inland Fisheries Act, 1984")

    # --- MAIN INTERFACE: HEADER ---
    st.markdown("""
    <div class="header-box">
        <div class="header-title">
            <span>🐟 SafeHilsa AI</span>
            <span style="font-size:1.05rem; background:rgba(255,255,255,0.2); padding:4px 12px; border-radius:20px;">Riverine Conservation & Compliance Guardian</span>
        </div>
        <div class="header-subtitle">
            Protecting the queen of fish (Hilsa / Ilish) across the Ganges-Bhagirathi-Hooghly river system under the 
            <b>West Bengal Inland Fisheries Act, 1984</b> and <b>Kolkata Gazette Notification No. 719-Fish</b>. 
            Autonomous sanctuary geo-compliance, statutory 23 cm size enforcement, historical hydrological telemetry, 
            and state livelihood assistance navigation (Samudra Sathi & Matsyajeebi Bandhu).
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
        st.caption("Select preset navigation hotspots or click anywhere on the river system:")

        # Quick Jump Buttons
        preset_cols = st.columns(3)
        with preset_cols[0]:
            if st.button("📍 Farakka Barrage", use_container_width=True):
                st.session_state.selected_lat, st.session_state.selected_lon = PRESETS["Farakka Barrage (Rule 50(3) Statutory Area)"]
                st.rerun()
        with preset_cols[1]:
            if st.button("📍 Katwa Sanctuary", use_container_width=True):
                st.session_state.selected_lat, st.session_state.selected_lon = PRESETS["Katwa Sanctuary Reach (Rule 50(1)(b))"]
                st.rerun()
        with preset_cols[2]:
            if st.button("📍 Tribeni (Outside)", use_container_width=True):
                st.session_state.selected_lat, st.session_state.selected_lon = PRESETS["Tribeni Ghat (Outside Checked Sanctuaries)"]
                st.rerun()

        preset_cols_2 = st.columns(3)
        with preset_cols_2[0]:
            if st.button("📍 Lalbagh Reach", use_container_width=True):
                st.session_state.selected_lat, st.session_state.selected_lon = PRESETS["Lalbagh Reach Corridor (Rule 50(1)(a))"]
                st.rerun()
        with preset_cols_2[1]:
            if st.button("📍 Diamond Harbour", use_container_width=True):
                st.session_state.selected_lat, st.session_state.selected_lon = PRESETS["Diamond Harbour Estuary Reach (Rule 50(1)(c))"]
                st.rerun()
        with preset_cols_2[2]:
            if st.button("📍 Kolkata Riverfront", use_container_width=True):
                st.session_state.selected_lat, st.session_state.selected_lon = PRESETS["Kolkata Riverfront (Outside Checked Sanctuaries)"]
                st.rerun()

        # Instant Compliance Evaluation
        compliance = check_location_compliance(
            lat=st.session_state.selected_lat,
            lon=st.session_state.selected_lon,
            date_str=eval_date_str
        )

        # Render Geographic Outcome Card
        st.markdown(
            format_compliance_card_html(compliance, st.session_state.selected_lat, st.session_state.selected_lon),
            unsafe_allow_html=True
        )

        # Folium Map Setup
        center_lat = st.session_state.selected_lat
        center_lon = st.session_state.selected_lon
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles="OpenStreetMap",
            control_scale=True
        )

        # Base Tiles: OpenStreetMap + CartoDB Positron + Free ESRI Satellite Imagery
        folium.TileLayer("CartoDB positron", name="Clean Nautical Vector").add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite Imagery (ESRI Free)"
        ).add_to(m)

        # 1. Determine temporal active ban state under Rule 50(2) (June-Aug & Oct-Dec)
        if eval_date:
            ban_active = eval_date.month in [6, 7, 8, 10, 11, 12]
        else:
            import datetime
            ban_active = datetime.datetime.now().month in [6, 7, 8, 10, 11, 12]

        corridor_color = "#DC2626" if ban_active else "#16A34A"
        corridor_buffer_color = "#EF4444" if ban_active else "#22C55E"
        temporal_badge = "🚨 SEASONAL BAN ACTIVE (June–Aug & Oct–Dec)" if ban_active else "ℹ️ PERMISSIBLE WINDOW (Ban Inactive)"

        # 2. Render Notified Linear River Reaches (Rule 50(1)) with 2.5km Corridor Buffers
        fg_reaches = folium.FeatureGroup(name="Notified River Reaches (Rule 50(1))", show=True)
        for reach in NOTIFIED_REACHES:
            poly = reach["polyline"]
            # Polyline centerline
            folium.PolyLine(
                locations=poly,
                color=corridor_color,
                weight=5,
                opacity=0.9,
                dash_array="6, 4",
                tooltip=f"{reach['name']} ({temporal_badge})",
                popup=(
                    f"<b>{reach['name']}</b><br>"
                    f"<b>Statute:</b> {reach['legal_citation']}<br>"
                    f"<b>River:</b> {reach['river']}<br>"
                    f"<b>Districts:</b> {reach['districts']}<br>"
                    f"<b>Status:</b> {temporal_badge}<br>"
                    f"<b>Rule 50(2):</b> All fish catching prohibited during June–Aug and Oct–Dec."
                )
            ).add_to(fg_reaches)

            # 2.5 km corridor buffer circles along segment nodes
            for pt in poly:
                folium.Circle(
                    location=pt,
                    radius=2500,  # 2.5 km buffer
                    color=corridor_color,
                    weight=1,
                    fill=True,
                    fill_color=corridor_buffer_color,
                    fill_opacity=0.12,
                    tooltip=f"{reach['name']} 2.5km River Corridor Buffer"
                ).add_to(fg_reaches)
        fg_reaches.add_to(m)

        # 3. Render Farakka Barrage 5 sq km Statutory Prohibited Area (Rule 50(3)) & Sandbars
        fg_statutory_areas = folium.FeatureGroup(name="Statutory 5 sq km Areas (Rule 50(3) & 50(1)(d))", show=True)
        farakka_radius_m = FARAKKA_STATUTORY_RADIUS_KM * 1000  # ~1262m
        folium.Circle(
            location=[FARAKKA_BARRAGE_LAT, FARAKKA_BARRAGE_LON],
            radius=farakka_radius_m,
            color="#D97706",
            weight=3,
            fill=True,
            fill_color="#F59E0B",
            fill_opacity=0.35,
            tooltip="⚠️ Rule 50(3): Farakka Barrage 5.0 sq km Statutory Area",
            popup=(
                "<b>Farakka Barrage Statutory Area (Rule 50(3))</b><br>"
                "Status: Year-round prohibited area for Hilsa fishing.<br>"
                "Statutory Area: 5.0 sq km around Barrage.<br>"
                "Cadastral limitation: Coordinates missing in Gazette (High Enforcement Risk)."
            )
        ).add_to(fg_statutory_areas)

        # Sundarbans Sandbars (Rule 50(1)(d))
        for sb in SUNDARBANS_SANDBARS:
            folium.Circle(
                location=[sb["lat"], sb["lon"]],
                radius=1262,
                color="#B45309",
                weight=2,
                fill=True,
                fill_color="#D97706",
                fill_opacity=0.25,
                tooltip=f"Rule 50(1)(d): {sb['name']} (5 sq km Sandbar)",
                popup=(
                    f"<b>{sb['name']}</b><br>"
                    f"Statute: Rule 50(1)(d), W.B. Inland Fisheries Rules 1985<br>"
                    f"Restriction: 5 sq km demarcated sandbar area.<br>"
                    f"Ban window: June–August & October–December (Rule 50(2))."
                )
            ).add_to(fg_statutory_areas)
        fg_statutory_areas.add_to(m)

        # 4. Render River Hydrology Monitoring Stations Layer (ICAR-CIFRI 179 Buoys)
        fg_stations = folium.FeatureGroup(name="River Hydrology Stations (CIFRI 179)", show=True)
        for sid, st_data in VERIFIED_HOOGHLY_BASELINE.items():
            s_lat = st_data["latitude"]
            s_lon = st_data["longitude"]
            folium.Marker(
                location=[s_lat, s_lon],
                tooltip=f"📡 Station: {st_data['station_name']} (Reach: {st_data['river_reach']})",
                popup=(
                    f"<b>{st_data['station_name']} Monitoring Station</b><br>"
                    f"<b>Reach:</b> {st_data['river_reach']} ({st_data['basin']})<br>"
                    f"<b>Dissolved Oxygen:</b> {st_data['dissolved_oxygen_mg_l']} mg/L<br>"
                    f"<b>Discharge (baseline):</b> {st_data['discharge_cumecs']} cumecs<br>"
                    f"<b>Turbidity:</b> {st_data['turbidity_ntu']} NTU | <b>Water Temp:</b> {st_data['surface_water_temp_c']} °C<br>"
                    f"<b>Salinity:</b> {st_data['salinity_ppt']} ppt | <b>pH:</b> {st_data['ph']}<br>"
                    f"<i>Data Source: {st_data['source_id']} ({st_data['observation_period']})</i>"
                ),
                icon=folium.Icon(color="darkblue", icon="tint", prefix="glyphicon")
            ).add_to(fg_stations)
        fg_stations.add_to(m)

        # 5. Render GeoJSON Sanctuaries Overlay (if present)
        if os.path.exists(SANCTUARIES_GEOJSON_PATH):
            try:
                with open(SANCTUARIES_GEOJSON_PATH, "r", encoding="utf-8") as f:
                    geojson_data = json.load(f)

                def style_function(feature):
                    geom_type = feature.get("geometry", {}).get("type", "")
                    if geom_type in ["LineString", "MultiLineString"]:
                        return {
                            "color": "#DC2626",
                            "weight": 4,
                            "opacity": 0.85,
                            "dashArray": "6, 4"
                        }
                    return {
                        "fillColor": "#EA580C",
                        "color": "#C2410C",
                        "weight": 2.5,
                        "fillOpacity": 0.35
                    }

                folium.GeoJson(
                    geojson_data,
                    name="Notified Hilsa Sanctuaries (Rule 50)",
                    style_function=style_function,
                    tooltip=folium.GeoJsonTooltip(
                        fields=["name", "status"],
                        aliases=["Sanctuary:", "Status:"],
                        style="background-color: white; color: #1E293B; font-weight: bold; border-radius: 4px; padding: 6px;"
                    ),
                    popup=folium.GeoJsonPopup(
                        fields=["name", "status", "legal_citation", "restriction_period"],
                        aliases=["Sanctuary:", "Status:", "Legal Basis:", "Ban Window:"]
                    )
                ).add_to(m)
            except Exception:
                pass

        # Add Selected Inspection Point Marker
        status_key = str(compliance.get("status", ""))
        if status_key == "INCONCLUSIVE_AREA_BOUNDARY":
            marker_color = "orange"
            marker_icon = "warning-sign"
        elif status_key in ["ADVISORY_SANCTUARY_REACH", "RESTRICTED"]:
            marker_color = "red"
            marker_icon = "ban"
        elif status_key == "PERMISSIBLE_SEASONAL_WINDOW":
            marker_color = "blue"
            marker_icon = "info-sign"
        else:
            marker_color = "cadetblue"
            marker_icon = "map-marker"

        folium.Marker(
            [center_lat, center_lon],
            popup=f"GPS: {center_lat:.4f}, {center_lon:.4f} ({status_key})",
            tooltip=f"Inspection Point: {status_key}",
            icon=folium.Icon(color=marker_color, icon=marker_icon, prefix="glyphicon")
        ).add_to(m)

        folium.Circle(
            location=[center_lat, center_lon],
            radius=2000,
            color=marker_color,
            weight=1.5,
            fill=True,
            fill_opacity=0.08
        ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

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
            
            if (click_lat != st.session_state.selected_lat) or (click_lon != st.session_state.selected_lon):
                st.session_state.selected_lat = click_lat
                st.session_state.selected_lon = click_lon
                st.rerun()

    # ==========================================
    # RIGHT COLUMN: AGENTIC CHAT INTERFACE
    # ==========================================
    with col_chat:
        st.markdown("### 💬 SafeHilsa AI Compliance Agent")
        st.caption("Ask questions on fisheries law, size compliance, ban dates, or livelihood schemes:")

        # Quick prompt buttons
        st.markdown("**Quick Inquiries:**")
        prompt_cols_1 = st.columns(2)
        with prompt_cols_1[0]:
            if st.button("🧭 Is my GPS location legal to fish?", use_container_width=True):
                st.session_state.user_query_input = "Is it legal to fish at my currently selected GPS coordinates? What restrictions apply?"
        with prompt_cols_1[1]:
            if st.button("📏 Minimum catch size (23 cm)?", use_container_width=True):
                st.session_state.user_query_input = "What is the official minimum legal catch size for Hilsa in West Bengal under Rule 46(4)?"
        
        prompt_cols_2 = st.columns(2)
        with prompt_cols_2[0]:
            if st.button("🛑 Seasonal ban windows?", use_container_width=True):
                st.session_state.user_query_input = "When are the mandatory seasonal ban periods for Hilsa sanctuaries under Rule 50(2)?"
        with prompt_cols_2[1]:
            if st.button("🤝 Samudra Sathi financial aid?", use_container_width=True):
                st.session_state.user_query_input = "What financial assistance is available under the Samudra Sathi scheme during the marine ban?"

        prompt_cols_3 = st.columns(2)
        with prompt_cols_3[0]:
            if st.button("🌊 River hydrology at Farakka?", use_container_width=True):
                st.session_state.user_query_input = "What is the water discharge and dissolved oxygen level at Farakka station?"
        with prompt_cols_3[1]:
            if st.button("🐟 Alternative fish species?", use_container_width=True):
                st.session_state.user_query_input = "What alternative fish species can I catch legally during the Hilsa ban period?"

        # Render Chat History
        chat_container = st.container(height=430)
        with chat_container:
            for msg in st.session_state.messages:
                avatar = "🐟" if msg["role"] == "assistant" else "🧑‍🌾"
                with st.chat_message(msg["role"], avatar=avatar):
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
        
        if "user_query_input" in st.session_state and st.session_state.user_query_input:
            query_to_submit = st.session_state.user_query_input
            st.session_state.user_query_input = None

        if query_to_submit:
            st.session_state.messages.append({"role": "user", "content": query_to_submit})
            
            with chat_container:
                with st.chat_message("user", avatar="🧑‍🌾"):
                    st.markdown(query_to_submit)

                with st.chat_message("assistant", avatar="🐟"):
                    with st.spinner("SafeHilsa AI reasoning and inspecting statutory registers..."):
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
                            host=ollama_host,
                            date_str=eval_date_str
                        )
                        
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

                        # Bidirectional map sync: auto-center map when agent identifies location
                        if meta.get("focus_coordinates"):
                            fc = meta["focus_coordinates"]
                            if fc.get("lat") and fc.get("lon"):
                                new_lat = round(float(fc["lat"]), 4)
                                new_lon = round(float(fc["lon"]), 4)
                                if (new_lat != st.session_state.selected_lat) or (new_lon != st.session_state.selected_lon):
                                    st.session_state.selected_lat = new_lat
                                    st.session_state.selected_lon = new_lon

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "tools": tool_trace
            })
            st.rerun()


if STREAMLIT_AVAILABLE and st is not None:
    render_app()
elif __name__ == "__main__":
    print("Streamlit is not installed in this environment.")
    print("To launch the web interface, run: pip install streamlit streamlit-folium")
    print("Backend modules and test suites remain fully operational.")
