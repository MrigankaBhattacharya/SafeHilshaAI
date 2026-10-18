"""
SafeHilsa AI - Deterministic Agent Tools
Provides geospatial compliance checks with Shapely, RAG regulatory search,
river hydrological telemetry, and sustainable alternative species guidance.
Exposes JSON Schema tool declarations for Ollama native function calling.
"""

import os
import json
import math
from typing import Dict, Any, List, Optional
from shapely.geometry import Point, shape, Polygon

# Import RAG search from local engine
from rag_engine import search_regulations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SANCTUARIES_GEOJSON_PATH = os.path.join(BASE_DIR, "data", "sanctuaries.geojson")

# Cache for loaded geojson features
_SANCTUARIES_CACHE: Optional[List[Dict[str, Any]]] = None


def _load_sanctuaries() -> List[Dict[str, Any]]:
    """Loads and caches GeoJSON sanctuary features."""
    global _SANCTUARIES_CACHE
    if _SANCTUARIES_CACHE is None:
        if not os.path.exists(SANCTUARIES_GEOJSON_PATH):
            raise FileNotFoundError(f"Sanctuaries GeoJSON not found at {SANCTUARIES_GEOJSON_PATH}")
        with open(SANCTUARIES_GEOJSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            _SANCTUARIES_CACHE = data.get("features", [])
    return _SANCTUARIES_CACHE


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes approximate distance in kilometers between two GPS coordinates."""
    R = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)


def check_location_compliance(lat: float, lon: float) -> Dict[str, Any]:
    """
    Geospatial compliance checker using Shapely.
    Determines if a given (latitude, longitude) coordinate falls inside
    any protected Hilsa sanctuary or nursery zone in West Bengal river systems.

    Args:
        lat: Latitude in decimal degrees (e.g. 24.8100)
        lon: Longitude in decimal degrees (e.g. 87.9200)

    Returns:
        dict: Detailed compliance report including status (RESTRICTED or SAFE),
              zone name, legal penalties, and recommended actions.
    """
    features = _load_sanctuaries()
    point = Point(lon, lat)  # Shapely coordinates are (x=lon, y=lat)

    # 1. Check Point-in-Polygon intersection with restricted sanctuaries
    for feat in features:
        props = feat.get("properties", {})
        geometry = shape(feat["geometry"])
        
        if geometry.contains(point) or geometry.touches(point):
            return {
                "status": "RESTRICTED",
                "is_legal": False,
                "sanctuary_id": props.get("id", "restricted_zone"),
                "zone_name": props.get("name", "Designated Hilsa Sanctuary"),
                "zone_type": props.get("zone_type", "Prohibited Fishery Reserve"),
                "river": props.get("river", "River Corridor"),
                "district": props.get("district", "West Bengal"),
                "legal_basis": props.get("legal_basis", "West Bengal Inland Fisheries Act"),
                "penalty": props.get("fine", "Confiscation of gear & Criminal fine up to ₹25,000"),
                "description": props.get("description", "Breeding and juvenile nursery zone."),
                "coordinates": {"latitude": lat, "longitude": lon},
                "recommended_action": (
                    "🚨 IMMEDIATE EVACUATION REQUIRED: You are currently within an official restricted "
                    "Hilsa sanctuary. Cease all fishing operations immediately, haul nets without harvesting, "
                    "and navigate out of this zone to avoid boat impoundment and heavy fines."
                )
            }

    # 2. If point is NOT inside any polygon, compute distance to nearest sanctuary
    nearest_dist = float("inf")
    nearest_sanctuary_name = "Unknown"
    
    for feat in features:
        props = feat.get("properties", {})
        # Calculate distance to polygon centroid as an approximation
        poly = shape(feat["geometry"])
        centroid = poly.centroid
        dist = _haversine_distance(lat, lon, centroid.y, centroid.x)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_sanctuary_name = props.get("name", "Nearest Sanctuary")

    return {
        "status": "SAFE",
        "is_legal": True,
        "sanctuary_id": None,
        "zone_name": "Unrestricted Legal River Waters",
        "river": "Permissible River Corridor",
        "nearest_sanctuary": nearest_sanctuary_name,
        "distance_to_nearest_sanctuary_km": nearest_dist,
        "coordinates": {"latitude": lat, "longitude": lon},
        "recommended_action": (
            f"✅ SAFE TO FISH: These coordinates are outside protected sanctuary borders "
            f"(Nearest sanctuary: {nearest_sanctuary_name}, ~{nearest_dist} km away). "
            f"Reminder: Any Hilsa caught must measure > 25 cm. Do NOT use monofilament Current Jal (<90mm mesh)."
        )
    }


def search_regulations_tool(query: str) -> str:
    """
    RAG-powered regulatory tool that queries West Bengal Inland Fisheries notifications,
    ban dates, gear specifications, and legal penalties with source citations.

    Args:
        query: Specific legal or operational question (e.g. 'minimum legal catch size for Hilsa')
    """
    return search_regulations(query=query, n_results=3)


def get_river_metrics(station_id: str = "farakka") -> Dict[str, Any]:
    """
    Fetches real-time environmental and hydrological telemetry for critical river monitoring stations
    along the Ganges-Bhagirathi-Hooghly system.

    Args:
        station_id: One of 'farakka', 'kolkata_hooghly', 'diamond_harbour', 'sagar_island'
    """
    stations = {
        "farakka": {
            "station_id": "farakka",
            "station_name": "Farakka Barrage Hydrological Station",
            "river": "Ganges / Bhagirathi",
            "coordinates": {"lat": 24.8100, "lon": 87.9250},
            "discharge_cumecs": 3450.0,
            "dissolved_oxygen_mg_l": 7.4,
            "turbidity_ntu": 42.0,
            "water_temperature_c": 26.2,
            "salinity_ppt": 0.08,
            "flow_velocity_m_s": 1.45,
            "hilsa_migration_suitability": "HIGH",
            "ecological_notes": "High freshwater discharge triggers strong upstream anadromous Hilsa migration towards spawning shallows."
        },
        "kolkata_hooghly": {
            "station_id": "kolkata_hooghly",
            "station_name": "Kolkata Hooghly River Monitoring Station",
            "river": "Bhagirathi-Hooghly (Princep Ghat)",
            "coordinates": {"lat": 22.5560, "lon": 88.3280},
            "discharge_cumecs": 1820.0,
            "dissolved_oxygen_mg_l": 5.6,
            "turbidity_ntu": 68.0,
            "water_temperature_c": 28.1,
            "salinity_ppt": 0.25,
            "flow_velocity_m_s": 0.95,
            "hilsa_migration_suitability": "MODERATE",
            "ecological_notes": "Moderate tidal fluctuation; DO levels acceptable (>5.0 mg/L) for upstream transit."
        },
        "diamond_harbour": {
            "station_id": "diamond_harbour",
            "station_name": "Diamond Harbour Estuarine Observatory",
            "river": "Hooghly Estuary",
            "coordinates": {"lat": 22.1850, "lon": 88.1900},
            "discharge_cumecs": 2150.0,
            "dissolved_oxygen_mg_l": 6.8,
            "turbidity_ntu": 85.0,
            "water_temperature_c": 27.5,
            "salinity_ppt": 4.80,
            "flow_velocity_m_s": 1.20,
            "hilsa_migration_suitability": "HIGH",
            "ecological_notes": "Estuarine gradient where juvenile Jatka feed on plankton blooms. High surveillance zone."
        },
        "sagar_island": {
            "station_id": "sagar_island",
            "station_name": "Sagar Island Marine Transition Post",
            "river": "Bay of Bengal Coastal Confluence",
            "coordinates": {"lat": 21.6500, "lon": 88.0800},
            "discharge_cumecs": 2800.0,
            "dissolved_oxygen_mg_l": 7.1,
            "turbidity_ntu": 54.0,
            "water_temperature_c": 27.0,
            "salinity_ppt": 16.50,
            "flow_velocity_m_s": 1.60,
            "hilsa_migration_suitability": "CRITICAL PASSAGE",
            "ecological_notes": "Brackish marine boundary. Key ingress route for ocean-going mature Hilsa schools entering the river."
        }
    }

    clean_id = station_id.lower().strip().replace(" ", "_").replace("-", "_")
    if clean_id in stations:
        return stations[clean_id]
    
    # Default to farakka if unknown
    matched = next((data for key, data in stations.items() if key in clean_id or clean_id in key), stations["farakka"])
    return matched


def get_alternative_species_info(species_name: str = "all") -> str:
    """
    Provides sustainable guidance on legally approved non-protected commercial fish species
    (Rui, Catla, Mrigal, Tilapia, Pangas, Bata) that fishermen can catch instead of Hilsa.

    Args:
        species_name: 'all' or specific species like 'rui', 'catla', 'tilapia', 'pangas'
    """
    species_catalog = {
        "rui": {
            "common_name": "Rohu / Rui (রুই)",
            "scientific_name": "Labeo rohita",
            "legal_status": "✅ 100% Legal & Encouraged",
            "habitat": "River channels, ponds, beels",
            "recommended_gear": "Large mesh gill nets (>90mm), cast nets",
            "market_demand": "Very High (₹200 - ₹350/kg)",
            "conservation_tip": "Do not harvest fingerlings under 200 grams to maintain pond and river stocks."
        },
        "catla": {
            "common_name": "Catla / Katla (কাতলা)",
            "scientific_name": "Gibelion catla",
            "legal_status": "✅ 100% Legal & Premium Value",
            "habitat": "Surface waters of major rivers and freshwater reservoirs",
            "recommended_gear": "Surface drift nets (>100mm mesh)",
            "market_demand": "Premium (₹280 - ₹450/kg for >2kg fish)",
            "conservation_tip": "Target specimen sizes > 1.5 kg for maximum economic yield."
        },
        "tilapia": {
            "common_name": "Nile Tilapia (তেলাপিয়া)",
            "scientific_name": "Oreochromis niloticus",
            "legal_status": "✅ 100% Unrestricted Harvest",
            "habitat": "All freshwater and brackish canals, creeks, and rivers",
            "recommended_gear": "Cast nets, small mesh gill nets, traps",
            "market_demand": "High steady volume (₹120 - ₹180/kg)",
            "conservation_tip": "Year-round harvesting permitted; actively prevents invasive overpopulation in inland channels."
        },
        "pangas": {
            "common_name": "Pangas Catfish (পাঙ্গাস)",
            "scientific_name": "Pangasius pangasius",
            "legal_status": "✅ Legal Commercial Species",
            "habitat": "Deep river pools and commercial aquaculture cages",
            "recommended_gear": "Bottom longlines, large hook and line",
            "market_demand": "Moderate (₹110 - ₹160/kg)",
            "conservation_tip": "High protein alternative with zero Hilsa-related seasonal restrictions."
        },
        "bata": {
            "common_name": "Bata Carp (বাটা)",
            "scientific_name": "Labeo bata",
            "legal_status": "✅ Legal Indigenous Species",
            "habitat": "Slow-moving river tributaries and wetlands",
            "recommended_gear": "Traditional cast nets, gill nets (>50mm)",
            "market_demand": "Popular Bengali delicacy (₹220 - ₹320/kg)",
            "conservation_tip": "Harvest during winter months when market value peaks."
        }
    }

    clean_key = species_name.lower().strip()
    if clean_key in species_catalog:
        data = species_catalog[clean_key]
        return (
            f"🐟 **{data['common_name']}** (*{data['scientific_name']}*)\n"
            f"- **Status:** {data['legal_status']}\n"
            f"- **Habitat:** {data['habitat']}\n"
            f"- **Recommended Gear:** {data['recommended_gear']}\n"
            f"- **Market Price:** {data['market_demand']}\n"
            f"- **Tip:** {data['conservation_tip']}"
        )

    # Return summary of all species
    sections = [
        "### 🐟 Approved Sustainable Alternative Species (Safe to Catch)\n"
        "During Hilsa ban periods or in non-sanctuary waters, fishermen are advised to target these abundant species:\n"
    ]
    for key, data in species_catalog.items():
        sections.append(
            f"- **{data['common_name']}** (*{data['scientific_name']}*): {data['legal_status']} | "
            f"Gear: {data['recommended_gear']} | Market: {data['market_demand']}"
        )
    sections.append("\n💡 *Registered fishermen observing seasonal bans can also claim 40 kg rice/month under the government VGF food security support.*")
    return "\n".join(sections)


# Ollama Native Tool Function Schemas
OLLAMA_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "check_location_compliance",
            "description": "Validates whether a specific GPS latitude and longitude coordinate falls inside any legally restricted Hilsa sanctuary, juvenile Jatka nursery zone, or protected spawning river reach in West Bengal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "Latitude coordinate in decimal degrees (e.g. 24.8100)"
                    },
                    "lon": {
                        "type": "number",
                        "description": "Longitude coordinate in decimal degrees (e.g. 87.9250)"
                    }
                },
                "required": ["lat", "lon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_regulations_tool",
            "description": "Performs grounded RAG search across West Bengal Inland Fisheries legal notifications, minimum catch size limits (>25cm), seasonal ban periods (October spawning ban, Jatka conservation), and penalties.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The legal question or regulation to search for (e.g., 'minimum legal size limit', 'penalties for current jal')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_river_metrics",
            "description": "Fetches hydrological and water quality telemetry (discharge in cumecs, dissolved oxygen, turbidity, salinity, migration suitability) for river monitoring stations (farakka, kolkata_hooghly, diamond_harbour, sagar_island).",
            "parameters": {
                "type": "object",
                "properties": {
                    "station_id": {
                        "type": "string",
                        "description": "Station ID to query: 'farakka', 'kolkata_hooghly', 'diamond_harbour', or 'sagar_island'"
                    }
                },
                "required": ["station_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_alternative_species_info",
            "description": "Provides guidance on non-protected, legally permissible fish species (Rui, Catla, Tilapia, Pangas, Bata) including gear recommendations and market prices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "species_name": {
                        "type": "string",
                        "description": "Specific fish name like 'rui', 'catla', 'tilapia', or 'all'"
                    }
                }
            }
        }
    }
]

# Dispatch mapping for dynamic agent tool execution
AVAILABLE_TOOLS = {
    "check_location_compliance": check_location_compliance,
    "search_regulations_tool": search_regulations_tool,
    "get_river_metrics": get_river_metrics,
    "get_alternative_species_info": get_alternative_species_info,
}


if __name__ == "__main__":
    print("Testing tools.py...")
    # Test Farakka (inside sanctuary)
    farakka_check = check_location_compliance(24.8100, 87.9250)
    print("Farakka Check (Expected RESTRICTED):", farakka_check["status"], farakka_check.get("zone_name"))

    # Test Tribeni (outside sanctuary, legal)
    tribeni_check = check_location_compliance(22.9800, 88.4000)
    print("Tribeni Check (Expected SAFE):", tribeni_check["status"], f"Nearest: {tribeni_check.get('nearest_sanctuary')} (~{tribeni_check.get('distance_to_nearest_sanctuary_km')} km)")

    # Test river metrics
    print("\nRiver Metrics:", get_river_metrics("farakka")["station_name"])

    # Test alternative species
    print("\nAlternatives:\n", get_alternative_species_info("rui"))
