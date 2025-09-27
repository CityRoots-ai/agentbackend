import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager
import asyncpg
import ee
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import logging
from dotenv import load_dotenv

load_dotenv()

gee_project_id = os.getenv('GEE_PROJECT_ID', 'ee-vvdev25')
ee.Initialize(project=gee_project_id)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_pool = None
async def init_db():
    global db_pool
    db_pool = await asyncpg.create_pool(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        database=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        min_size=1,
        max_size=10
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Database connection pool initialized")
    yield
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")

app = FastAPI(
    title="CityRoots API",
    description="Urban planning and GIS-aware API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

try:
    openai_client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise e

class AgentRequest(BaseModel):
    message: str
    uiContext: Optional[Dict[str, Any]] = None
    sessionId: Optional[str] = None

class LocationQuery(BaseModel):
    zip: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

class AnalyzeRequest(BaseModel):
    geometry: Dict[str, Any]
    landUseType: str = "removed"

class NDVIRequest(BaseModel):
    geometry: Dict[str, Any]

SYSTEM_PROMPT = """
You are a highly capable urban planning assistant embedded in a GIS-aware chatbot.
Your role is to interpret natural language queries and return STRICT JSON in the following structure:
{
  "intent": "show_parks" | "ask_area" | "greeting" | "unknown" | "park_removal_impact" | "park_ndvi_query" | "park_stat_query",
  "locationType": "zip" | "city" | "state" | null,
  "locationValue": string | null,
  "unit": "acres" | "m2" | "km2" | "hectares" | null,
  "landUseType": "removed" | "replaced_by_building" | null,
  "metric": string | null
}

Examples:
- "show parks in Austin" => show_parks, city: Austin
- "show parks of zipcode 20008" => show_parks, zip, 20008
- "give me area of this park" => ask_area
- "how big is this park in hectares" => ask_area, unit: hectares
- "what is the area in square meters?" => ask_area, unit: "m2"
- "What happens if this park is removed?" => park_removal_impact, landUseType: removed
- "Replace this park with commercial buildings" => park_removal_impact, landUseType: replaced_by_building
- "what's the NDVI of this park?" => intent: "park_ndvi_query"
- "how many Asian people are served by this park?" => intent: "park_stat_query", metric: "SUM_ASIAN_"
- "what's the total population in the park's service area?" => intent: "park_stat_query", metric: "SUM_TOTPOP"
- "How many seniors are in this area?" => intent: "park_stat_query", metric: "SUM_SENIOR"
- "How many kids are in this area?" => intent: "park_stat_query", metric: "SUM_KIDSVC"
- "What's the per acre density?" => intent: "park_stat_query", metric: "PERACRE"
- "what income group is most served here?" => intent: "park_stat_query", metric: "income_distribution"
- "how many Hispanic households are in the service area?" => intent: "park_stat_query", metric: "SUM_HISP_S"
- "hello" => greeting

Respond with ONLY the JSON, no explanation or comments.
"""
def geometry_from_geojson(geojson):
    return ee.Geometry(geojson)

def compute_ndvi(geometry):
    collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(geometry) \
        .filterDate("2022-06-01", "2022-09-01") \
        .sort("CLOUD_COVER") \
        .map(lambda img: img.multiply(0.0000275).add(-0.2))

    def add_ndvi(image):
        return image.addBands(image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI'))

    ndvi_img = collection.map(add_ndvi).select('NDVI').median()
    stats = ndvi_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30,
        maxPixels=1e9
    )
    return stats.getInfo().get('NDVI', None)

def compute_walkability(geometry):
    population = ee.ImageCollection("WorldPop/GP/100m/pop").first()
    stats = population.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=100,
        maxPixels=1e9
    )
    total_pop = stats.getInfo().get('population', 0)
    area_km2 = geometry.area().getInfo() / 1e6
    density = total_pop / area_km2 if area_km2 > 0 else 0
    score = 100 / (1 + pow(2.71828, -0.03 * (density - 100)))
    return round(score, 2)

def compute_pm25(geometry):
    pm25 = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_AER_AI") \
        .filterBounds(geometry) \
        .filterDate("2022-06-01", "2022-09-01") \
        .select("absorbing_aerosol_index") \
        .median()

    stats = pm25.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=1000,
        maxPixels=1e9
    )
    return stats.getInfo().get("absorbing_aerosol_index", None)

def compute_population(geometry):
    buffer = geometry.buffer(800)
    population = ee.ImageCollection("WorldPop/GP/100m/pop").first()
    stats = population.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=buffer,
        scale=100,
        maxPixels=1e9
    )
    return stats.getInfo().get('population', 0)

def simulate_replacement_with_buildings(buffer_geom, park_geom):
    built_ndvi = ee.Image.constant(0.1).rename('NDVI')
    ndvi_img = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(buffer_geom) \
        .filterDate("2022-06-01", "2022-09-01") \
        .sort("CLOUD_COVER") \
        .map(lambda img: img.multiply(0.0000275).add(-0.2)) \
        .map(lambda img: img.addBands(img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI'))) \
        .median() \
        .select('NDVI')

    modified_ndvi = ndvi_img.blend(built_ndvi.clip(park_geom))

    stats = modified_ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer_geom,
        scale=30,
        maxPixels=1e9
    )
    return stats.getInfo().get('NDVI', None)

def build_feature_collection(rows):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": row["geometry"],
                "properties": {
                    "gid": row["gid"],
                    "Park_id": row["park_id"],
                    "Park_Name": row["park_name"],
                    "Park_Addre": row["park_addre"],
                    "Park_Owner": row["park_owner"],
                    "Park_Zip": row["park_zip"],
                    "Park_Size_Acres": row["area_acres"],
                },
            }
            for row in rows
        ],
    }

async def query_parks_by_location(query: LocationQuery, simplify_tolerance: float = 0.0002):
    async with db_pool.acquire() as conn:
        where_clauses = []
        params = []

        if query.zip:
            params.append(query.zip)
            where_clauses.append(f"park_zip = ${len(params)}")
        if query.city:
            params.append(query.city)
            where_clauses.append(f"LOWER(park_place) = LOWER(${len(params)})")
        if query.state:
            params.append(query.state)
            where_clauses.append(f"LOWER(park_state) = LOWER(${len(params)})")

        if not where_clauses:
            where_clauses.append("1=0")

        params.append(simplify_tolerance)

        sql = f"""
            SELECT
                gid,
                park_id,
                park_name,
                park_addre,
                park_owner,
                park_zip,
                COALESCE(park_size_, NULLIF(shape_area,0) * 0.000247105,
                         ST_Area(geography(geom)) * 0.000247105) AS area_acres,
                ST_AsGeoJSON(ST_SimplifyPreserveTopology(ST_Transform(geom, 4326), ${len(params)}))::json AS geometry
            FROM parks
            WHERE {" OR ".join(where_clauses)}
            LIMIT 5000;
        """

        rows = await conn.fetch(sql, *params)
        return build_feature_collection([dict(row) for row in rows])

async def query_park_area_by_id(park_id: str):
    async with db_pool.acquire() as conn:
        sql = """
            SELECT park_name,
                   COALESCE(park_size_, NULLIF(shape_area,0) * 0.000247105,
                            ST_Area(geography(geom)) * 0.000247105) AS area_acres
            FROM parks
            WHERE park_id = $1
            LIMIT 1;
        """
        row = await conn.fetchrow(sql, park_id)
        if not row:
            return None
        return {
            "name": row["park_name"] or "Unnamed Park",
            "acres": row["area_acres"],
        }

async def get_park_statistics_by_id(park_id: str):
    async with db_pool.acquire() as conn:
        sql = """
            SELECT SUM_TOTPOP, SUM_KIDSVC, SUM_YOUNGP, SUM_SENIOR,
                   SUM_HHILOW, SUM_HHIMED, SUM_HHIHIG, SUM_TOTHHS,
                   SUM_WHITE_, SUM_BLACK_, SUM_ASIAN_, SUM_HISP_S,
                   PERACRE
            FROM parks_stats
            WHERE park_id = $1
        """
        row = await conn.fetchrow(sql, park_id)
        return dict(row) if row else None

async def query_park_stat_by_id(park_id: str, metric: str):
    async with db_pool.acquire() as conn:
        sql = f"SELECT {metric} FROM parks_stats WHERE park_id = $1 LIMIT 1"
        row = await conn.fetchrow(sql, park_id)
        if not row:
            return None

        value = row[metric.lower()]
        return {
            "value": value,
            "formatted": f"{value:,}" if value else "0",
        }

async def analyze_park_removal_impact(park_id: str, land_use_type: str = "removed"):
    async with db_pool.acquire() as conn:
        sql = """
            SELECT park_name, ST_AsGeoJSON(ST_Transform(geom, 4326))::json AS geometry
            FROM parks_stats
            WHERE park_id = $1
        """
        row = await conn.fetchrow(sql, park_id)
        if not row:
            return None

        park_name = row["park_name"]
        geometry = row["geometry"]

        stats = await get_park_statistics_by_id(park_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Park statistics not found")

        park_geom = geometry_from_geojson(geometry)
        buffer_geom = park_geom.buffer(800)

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
            "parkId": park_id,
            "parkName": park_name,
            "landUseType": land_use_type,
            "affectedPopulation10MinWalk": int(affected_population),
            "ndviBefore": round(ndvi_before, 4) if ndvi_before else None,
            "ndviAfter": round(ndvi_after, 4) if ndvi_after else None,
            "walkabilityBefore": walkability_before,
            "walkabilityAfter": walkability_after,
            "pm25Before": round(pm25_before, 2) if pm25_before else None,
            "pm25After": round(pm25_after, 2) if pm25_after else None,
            "demographics": {
                "total": stats.get("sum_totpop"),
                "kids": stats.get("sum_kidsvc"),
                "adults": stats.get("sum_youngp"),
                "seniors": stats.get("sum_senior"),
                "white": stats.get("sum_white_"),
                "black": stats.get("sum_black_"),
                "asian": stats.get("sum_asian_"),
                "hispanic": stats.get("sum_hisp_s"),
            },
            "income": {
                "low": stats.get("sum_hhilow"),
                "middle": stats.get("sum_hhimed"),
                "high": stats.get("sum_hhihig"),
            },
            "households": stats.get("sum_tothhs"),
            "perAcreDemand": stats.get("peracre"),
            "message": f"If {park_name} is {land_use_type.replace('_', ' ')}, {stats.get('sum_totpop', 0)} people lose access. NDVI drops from {ndvi_before} to {ndvi_after}, PM2.5 may increase, and walkability decreases.",
        }

async def get_park_ndvi(park_id: str):
    async with db_pool.acquire() as conn:
        sql = "SELECT ST_AsGeoJSON(ST_Transform(geom, 4326))::json AS geometry FROM parks WHERE park_id = $1"
        row = await conn.fetchrow(sql, park_id)
        if not row:
            raise HTTPException(status_code=404, detail="Park not found")

        geometry = geometry_from_geojson(row["geometry"])
        ndvi_value = compute_ndvi(geometry)
        return ndvi_value

@app.post("/api/agent")
async def agent_endpoint(request: AgentRequest):
    try:
        message = request.message
        ui_context = request.uiContext or {}
        selected_park_id = ui_context.get("selectedParkId")
        session_id = request.sessionId or str(int(datetime.now().timestamp() * 1000000) % 1000000)

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            temperature=0
        )

        try:
            parsed = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            parsed = {"intent": "unknown"}

        logger.info(f"Parsed intent: {parsed.get('intent')}")

        if parsed.get("intent") == "show_parks":
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

        elif parsed.get("intent") == "ask_area":
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

        elif parsed.get("intent") == "park_removal_impact":
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

        elif parsed.get("intent") == "park_ndvi_query":
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

        elif parsed.get("intent") == "park_stat_query":
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

        elif parsed.get("intent") == "greeting":
            reply = "Hello! Try: \"show parks of zipcode 20008\" or \"show parks of city Austin\"."
            return {
                "sessionId": session_id,
                "action": "answer",
                "reply": reply,
            }

        fallback_reply = "I can show parks by zipcode/city/state, or tell you the area of a clicked park."
        return {
            "sessionId": session_id,
            "action": "answer",
            "reply": fallback_reply,
        }

    except Exception as e:
        logger.error(f"Error in agent endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Server error")

@app.post("/api/analyze")
async def analyze_endpoint(request: AnalyzeRequest):
    try:
        geometry = request.geometry
        land_use_type = request.landUseType

        park_geom = geometry_from_geojson(geometry)
        buffer_geom = park_geom.buffer(800)

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

@app.post("/api/ndvi")
async def ndvi_endpoint(request: NDVIRequest):
    try:
        geometry = geometry_from_geojson(request.geometry)
        ndvi_value = compute_ndvi(geometry)

        return {
            "ndvi": round(ndvi_value, 4) if ndvi_value is not None else None
        }

    except Exception as e:
        logger.error(f"Error in NDVI endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 4000)))