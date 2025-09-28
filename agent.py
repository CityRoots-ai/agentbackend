import json
import logging
from datetime import datetime
from fastapi import HTTPException
from google import genai

from models import AgentRequest, LocationQuery, IntentClassification, AnalyzeRequest, NDVIRequest
from database import (
    query_parks_by_location, query_park_area_by_id, query_park_stat_by_id,
    get_park_ndvi, get_park_information, get_park_air_quality,
    analyze_park_removal_impact
)
from utils import (
    geometry_from_geojson, compute_ndvi, compute_walkability, compute_pm25,
    compute_population, simulate_replacement_with_buildings
)

logger = logging.getLogger(__name__)

async def handle_agent_request(request: AgentRequest, client: genai.Client):
    """Handle agent requests and process user queries about parks"""
    try:
        message = request.message
        ui_context = request.uiContext or {}
        selected_park_id = ui_context.get("selectedParkId")
        session_id = request.sessionId or str(int(datetime.now().timestamp() * 1000000) % 1000000)

        # Enhanced prompt for structured output with more examples
        prompt = f"""Analyze this user query about parks and classify the intent:

User query: "{message}"

Examples:
- "show parks in Austin" -> show_parks intent, city location
- "show parks in zipcode 24060" -> show_parks intent, zip location
- "find parks in 90210" -> show_parks intent, zip location
- "parks in TX" -> show_parks intent, state location
- "how big is this park" -> ask_area intent
- "park area in square meters" -> ask_area intent, m2 unit
- "what's the NDVI of this park" -> park_ndvi_query intent
- "how green is this park" -> park_ndvi_query intent
- "how many people live here" -> park_stat_query intent, metric: "SUM_TOTPOP"
- "total population" -> park_stat_query intent, metric: "SUM_TOTPOP"
- "Asian population served" -> park_stat_query intent, metric: "SUM_ASIAN_"
- "how many kids are in this area" -> park_stat_query intent, metric: "SUM_KIDSVC"
- "seniors in the area" -> park_stat_query intent, metric: "SUM_SENIOR"
- "young adults population" -> park_stat_query intent, metric: "SUM_YOUNGP"
- "what happens if removed" -> park_removal_impact intent, landUseType: removed
- "tell me about this park" -> park_info_query intent
- "describe this park" -> park_info_query intent
- "park information" -> park_info_query intent
- "when was this park built" -> park_info_query intent
- "what's the air quality here" -> air_quality_query intent
- "pollution levels in this area" -> air_quality_query intent
- "is the air safe to breathe" -> air_quality_query intent
- "PM2.5 levels" -> air_quality_query intent
- "air pollution near this park" -> air_quality_query intent
- "how polluted is this area" -> air_quality_query intent
- "hello" -> greeting intent"""

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": IntentClassification,
                }
            )
            logger.info(f"Gemini structured response: {response.text}")

            # Parse the structured JSON response
            parsed = json.loads(response.text)

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            parsed = {"intent": "unknown"}

        logger.info(f"Parsed intent: {parsed.get('intent')}")

        if parsed.get("intent") == "show_parks":
            return await handle_show_parks_intent(parsed, session_id)
        elif parsed.get("intent") == "ask_area":
            return await handle_ask_area_intent(parsed, selected_park_id, session_id)
        elif parsed.get("intent") == "park_removal_impact":
            return await handle_park_removal_impact_intent(parsed, selected_park_id, session_id)
        elif parsed.get("intent") == "park_ndvi_query":
            return await handle_park_ndvi_query_intent(selected_park_id, session_id)
        elif parsed.get("intent") == "park_stat_query":
            return await handle_park_stat_query_intent(parsed, selected_park_id, session_id)
        elif parsed.get("intent") == "park_info_query":
            return await handle_park_info_query_intent(selected_park_id, session_id, client)
        elif parsed.get("intent") == "air_quality_query":
            return await handle_air_quality_query_intent(selected_park_id, session_id)
        elif parsed.get("intent") == "greeting":
            return handle_greeting_intent(session_id)

        fallback_reply = "I can show parks by zipcode/city/state, or tell you the area of a clicked park."
        return {
            "sessionId": session_id,
            "action": "answer",
            "reply": fallback_reply,
        }

    except Exception as e:
        logger.error(f"Error in agent endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Server error")

async def handle_show_parks_intent(parsed, session_id):
    """Handle show parks intent"""
    query = LocationQuery()
    if parsed.get("locationType") == "zip":
        query.zip = parsed.get("locationValue")
    elif parsed.get("locationType") == "city":
        query.city = parsed.get("locationValue")
    elif parsed.get("locationType") == "state":
        query.state = parsed.get("locationValue")

    fc = await query_parks_by_location(query)
    reply = f"Loaded {len(fc['features'])} park(s) for {parsed.get('locationType')}: {parsed.get('locationValue')}."

    return {
        "sessionId": session_id,
        "action": "render_parks",
        "reply": reply,
        "data": {"featureCollection": fc},
    }

async def handle_ask_area_intent(parsed, selected_park_id, session_id):
    """Handle ask area intent"""
    if not selected_park_id:
        return {
            "sessionId": session_id,
            "action": "need_selection",
            "reply": "Please click a park first.",
        }

    info = await query_park_area_by_id(selected_park_id)
    if not info:
        return {
            "sessionId": session_id,
            "action": "error",
            "reply": "Could not find that park.",
        }

    value = info["acres"]
    unit = parsed.get("unit", "acres")
    converted = value
    unit_label = "acres"

    if unit == "m2":
        converted = value * 4046.86
        unit_label = "m²"
    elif unit == "km2":
        converted = value * 0.00404686
        unit_label = "km²"
    elif unit == "hectares":
        converted = value * 0.404686
        unit_label = "hectares"

    formatted = f"{converted:,.2f}"
    reply = f"Area of \"{info['name']}\": {formatted} {unit_label}."

    return {
        "sessionId": session_id,
        "action": "answer",
        "reply": reply,
        "data": {
            "parkId": selected_park_id,
            "area": converted,
            "unit": unit_label,
        },
    }

async def handle_park_removal_impact_intent(parsed, selected_park_id, session_id):
    """Handle park removal impact intent"""
    if not selected_park_id:
        return {
            "sessionId": session_id,
            "action": "need_selection",
            "reply": "Please select a park to analyze its removal impact.",
        }

    impact = await analyze_park_removal_impact(
        selected_park_id, parsed.get("landUseType", "removed")
    )

    return {
        "sessionId": session_id,
        "action": "removal_impact",
        "reply": impact["message"],
        "data": impact,
    }

async def handle_park_ndvi_query_intent(selected_park_id, session_id):
    """Handle park NDVI query intent"""
    if not selected_park_id:
        return {
            "sessionId": session_id,
            "action": "need_selection",
            "reply": "Please select a park.",
        }

    ndvi = await get_park_ndvi(selected_park_id)
    reply = f"The NDVI of this park is approximately {ndvi:.3f}."

    return {
        "sessionId": session_id,
        "action": "answer",
        "reply": reply,
        "data": {"ndvi": ndvi},
    }

async def handle_park_stat_query_intent(parsed, selected_park_id, session_id):
    """Handle park statistics query intent"""
    if not selected_park_id:
        return {
            "sessionId": session_id,
            "action": "need_selection",
            "reply": "Please select a park.",
        }
    if not parsed.get("metric"):
        return {
            "sessionId": session_id,
            "action": "error",
            "reply": "Metric not specified.",
        }

    stat = await query_park_stat_by_id(selected_park_id, parsed["metric"])
    reply = f"The value for {parsed['metric']} is {stat['formatted']}."

    return {
        "sessionId": session_id,
        "action": "answer",
        "reply": reply,
        "data": {"metric": parsed["metric"], "value": stat["value"]},
    }

async def handle_park_info_query_intent(selected_park_id, session_id, client):
    """Handle park information query intent"""
    if not selected_park_id:
        return {
            "sessionId": session_id,
            "action": "need_selection",
            "reply": "Please select a park to get information about.",
        }

    park_info = await get_park_information(selected_park_id, client)

    return {
        "sessionId": session_id,
        "action": "park_information",
        "reply": park_info["description"],
        "data": park_info,
    }

async def handle_air_quality_query_intent(selected_park_id, session_id):
    """Handle air quality query intent"""
    if not selected_park_id:
        return {
            "sessionId": session_id,
            "action": "need_selection",
            "reply": "Please select a park to check air quality.",
        }

    air_quality_data = await get_park_air_quality(selected_park_id)

    return {
        "sessionId": session_id,
        "action": "air_quality_assessment",
        "reply": air_quality_data["message"],
        "data": air_quality_data,
    }

def handle_greeting_intent(session_id):
    """Handle greeting intent"""
    reply = "Hello! Try: \"show parks of zipcode 20008\" or \"show parks of city Austin\"."
    return {
        "sessionId": session_id,
        "action": "answer",
        "reply": reply,
    }

async def handle_analyze_request(request: AnalyzeRequest):
    """Handle analyze endpoint requests"""
    try:
        geometry = request.geometry
        land_use_type = request.landUseType

        try:
            park_geom = geometry_from_geojson(geometry)
            buffer_geom = park_geom.buffer(800)
        except ValueError as e:
            logger.error(f"Geometry error in analyze endpoint: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid geometry: {str(e)}")

        ndvi_before = compute_ndvi(buffer_geom)
        walkability_before = compute_walkability(buffer_geom)
        pm25_before = compute_pm25(buffer_geom)
        affected_population = compute_population(buffer_geom)

        if land_use_type == "removed":
            buffer_after = buffer_geom.difference(park_geom)
            ndvi_after = compute_ndvi(buffer_after)
        elif land_use_type == "replaced_by_building":
            ndvi_after = simulate_replacement_with_buildings(buffer_geom, park_geom)
        else:
            ndvi_after = ndvi_before

        walkability_after = compute_walkability(buffer_geom.difference(park_geom))
        pm25_after = compute_pm25(buffer_geom.difference(park_geom))

        return {
            "affectedPopulation10MinWalk": int(affected_population),
            "ndviBefore": round(ndvi_before, 4) if ndvi_before else None,
            "ndviAfter": round(ndvi_after, 4) if ndvi_after else None,
            "walkabilityBefore": walkability_before,
            "walkabilityAfter": walkability_after,
            "pm25Before": round(pm25_before, 2) if pm25_before else None,
            "pm25After": round(pm25_after, 2) if pm25_after else None
        }

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_ndvi_request(request: NDVIRequest):
    """Handle NDVI endpoint requests"""
    try:
        try:
            geometry = geometry_from_geojson(request.geometry)
        except ValueError as e:
            logger.error(f"Geometry error in NDVI endpoint: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid geometry: {str(e)}")

        ndvi_value = compute_ndvi(geometry)

        return {
            "ndvi": round(ndvi_value, 4) if ndvi_value is not None else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in NDVI endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))