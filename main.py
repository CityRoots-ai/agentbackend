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
from google import genai
from google.genai import types
import logging
import warnings
from dotenv import load_dotenv

load_dotenv()

# Suppress pydantic warnings from google-genai package
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

gee_project_id = os.getenv('GEE_PROJECT_ID')
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
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

try:
    client = genai.Client(api_key=gemini_api_key)
    logger.info("Gemini API client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API client: {e}")
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

from enum import Enum

class Intent(str, Enum):
    SHOW_PARKS = "show_parks"
    ASK_AREA = "ask_area"
    GREETING = "greeting"
    UNKNOWN = "unknown"
    PARK_REMOVAL_IMPACT = "park_removal_impact"
    PARK_NDVI_QUERY = "park_ndvi_query"
    PARK_STAT_QUERY = "park_stat_query"
    PARK_INFO_QUERY = "park_info_query"
    AIR_QUALITY_QUERY = "air_quality_query"

class LocationType(str, Enum):
    ZIP = "zip"
    CITY = "city"
    STATE = "state"

class Unit(str, Enum):
    ACRES = "acres"
    M2 = "m2"
    KM2 = "km2"
    HECTARES = "hectares"

class LandUseType(str, Enum):
    REMOVED = "removed"
    REPLACED_BY_BUILDING = "replaced_by_building"

class IntentClassification(BaseModel):
    intent: Intent
    locationType: Optional[LocationType] = None
    locationValue: Optional[str] = None
    unit: Optional[Unit] = None
    landUseType: Optional[LandUseType] = None
    metric: Optional[str] = None
def geometry_from_geojson(geojson):
    try:
        if not geojson:
            raise ValueError("GeoJSON is None or empty")

        if isinstance(geojson, str):
            geojson = json.loads(geojson)

        if not isinstance(geojson, dict):
            raise ValueError("GeoJSON must be a dictionary")

        if 'type' not in geojson:
            raise ValueError("GeoJSON missing 'type' property")

        if 'coordinates' not in geojson:
            raise ValueError("GeoJSON missing 'coordinates' property")

        return ee.Geometry(geojson)
    except Exception as e:
        logger.error(f"Invalid GeoJSON geometry: {e}")
        raise ValueError(f"Invalid GeoJSON geometry: {e}")

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
    """
    Compute PM2.5 concentration using GHAP (Global High Air Pollutants) dataset
    Returns PM2.5 concentration in μg/m³
    """
    try:
        # Use GHAP monthly PM2.5 dataset for recent data
        pm25_collection = ee.ImageCollection("projects/sat-io/open-datasets/GHAP/GHAP_M1K_PM25")

        # Filter for recent data (last available year)
        recent_pm25 = pm25_collection \
            .filterBounds(geometry) \
            .filterDate("2022-01-01", "2022-12-31") \
            .select("b1") \
            .mean()

        # Calculate mean PM2.5 concentration for the area
        stats = recent_pm25.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=1000,  # 1km resolution
            maxPixels=1e9
        )

        pm25_value = stats.getInfo().get("b1", None)
        return pm25_value

    except Exception as e:
        logger.error(f"Error computing PM2.5: {e}")
        # Fallback to Sentinel-5P data if GHAP fails
        try:
            sentinel5p = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_AER_AI") \
                .filterBounds(geometry) \
                .filterDate("2022-06-01", "2022-09-01") \
                .select("absorbing_aerosol_index") \
                .median()

            stats = sentinel5p.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=1000,
                maxPixels=1e9
            )
            # Convert aerosol index to approximate PM2.5 (rough estimate)
            aerosol_index = stats.getInfo().get("absorbing_aerosol_index", None)
            if aerosol_index:
                # Rough conversion: aerosol index to PM2.5 estimate
                return aerosol_index * 10  # Approximate conversion factor
            return None
        except:
            return None

def assess_air_quality_and_damage(geometry):
    """
    Comprehensive air quality assessment with environmental damage analysis
    Returns PM2.5 levels, health risk category, and environmental impact
    """
    try:
        pm25_value = compute_pm25(geometry)

        if pm25_value is None:
            return {
                "pm25_ugm3": None,
                "health_risk": "Unknown",
                "environmental_damage": "Cannot assess",
                "who_guideline_exceedance": None,
                "health_impact": "Data unavailable"
            }

        # WHO Air Quality Guidelines (2021): Annual PM2.5 guideline = 5 μg/m³
        who_annual_guideline = 5.0

        # EPA National Ambient Air Quality Standards: Annual PM2.5 = 12 μg/m³
        epa_annual_standard = 12.0

        # Health risk categorization
        if pm25_value <= 5:
            health_risk = "Low"
            damage_level = "Minimal environmental impact"
            health_impact = "Meets WHO guidelines - minimal health risk"
        elif pm25_value <= 12:
            health_risk = "Moderate"
            damage_level = "Low to moderate environmental stress"
            health_impact = "Exceeds WHO guidelines - increased respiratory risk"
        elif pm25_value <= 25:
            health_risk = "Unhealthy for Sensitive Groups"
            damage_level = "Moderate environmental degradation"
            health_impact = "Unhealthy for children, elderly, and people with heart/lung disease"
        elif pm25_value <= 35:
            health_risk = "Unhealthy"
            damage_level = "Significant environmental stress"
            health_impact = "Everyone may experience health effects"
        elif pm25_value <= 50:
            health_risk = "Very Unhealthy"
            damage_level = "Severe environmental degradation"
            health_impact = "Emergency conditions - everyone at risk"
        else:
            health_risk = "Hazardous"
            damage_level = "Critical environmental damage"
            health_impact = "Health warnings - everyone should avoid outdoor activities"

        # Calculate exceedance factors
        who_exceedance = pm25_value / who_annual_guideline
        epa_exceedance = pm25_value / epa_annual_standard

        return {
            "pm25_ugm3": round(pm25_value, 2),
            "health_risk": health_risk,
            "environmental_damage": damage_level,
            "who_guideline_exceedance": round(who_exceedance, 2),
            "epa_standard_exceedance": round(epa_exceedance, 2),
            "health_impact": health_impact,
            "recommendations": get_air_quality_recommendations(pm25_value)
        }

    except Exception as e:
        logger.error(f"Error in air quality assessment: {e}")
        return {
            "pm25_ugm3": None,
            "health_risk": "Unknown",
            "environmental_damage": "Assessment failed",
            "error": str(e)
        }

def get_air_quality_recommendations(pm25_value):
    """Generate specific recommendations based on PM2.5 levels"""
    if pm25_value <= 5:
        return "Air quality is good. Continue enjoying outdoor activities."
    elif pm25_value <= 12:
        return "Air quality is acceptable. Sensitive individuals should consider reducing prolonged outdoor exertion."
    elif pm25_value <= 25:
        return "Sensitive groups should reduce outdoor activities. Consider wearing masks during outdoor exercise."
    elif pm25_value <= 35:
        return "Everyone should reduce outdoor activities. Avoid outdoor exercise. Consider air purifiers indoors."
    elif pm25_value <= 50:
        return "Avoid outdoor activities. Stay indoors with windows closed. Use air purifiers if available."
    else:
        return "Emergency conditions. Avoid all outdoor activities. Seek medical attention if experiencing symptoms."

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

        # Handle case-sensitive column names - try exact match first, then lowercase
        try:
            value = row[metric]
        except KeyError:
            try:
                value = row[metric.lower()]
            except KeyError:
                logger.error(f"Column not found: {metric} (tried both original and lowercase)")
                return None
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

        try:
            park_geom = geometry_from_geojson(geometry)
            buffer_geom = park_geom.buffer(800)
        except ValueError as e:
            logger.error(f"Geometry error for park {park_id}: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid park geometry: {str(e)}")

        ndvi_before = compute_ndvi(buffer_geom)
        walkability_before = compute_walkability(buffer_geom)
        pm25_before = compute_pm25(buffer_geom)
        affected_population = compute_population(buffer_geom)

        if land_use_type == "removed":
            buffer_after = buffer_geom.difference(park_geom)
            ndvi_after = compute_ndvi(buffer_after)
            # Parks act as air filters - removing them increases PM2.5 by ~20%
            pm25_pollution_factor = 1.2
        elif land_use_type == "replaced_by_building":
            ndvi_after = simulate_replacement_with_buildings(buffer_geom, park_geom)
            # Buildings increase urban heat island and pollution by ~35%
            pm25_pollution_factor = 1.35
        else:
            ndvi_after = ndvi_before
            pm25_pollution_factor = 1.1

        walkability_after = compute_walkability(buffer_geom.difference(park_geom))

        # Calculate PM2.5 after removal with pollution impact
        if pm25_before:
            pm25_after = pm25_before * pm25_pollution_factor
            pm25_increase = pm25_after - pm25_before
            pm25_increase_percent = ((pm25_pollution_factor - 1) * 100)
        else:
            pm25_after = None
            pm25_increase = None
            pm25_increase_percent = None

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
            "pm25Increase": round(pm25_increase, 2) if pm25_increase else None,
            "pm25IncreasePercent": round(pm25_increase_percent, 1) if pm25_increase_percent else None,
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
            "summary": {
                "people_affected": f"{stats.get('sum_totpop', 0):,} people lose park access within 10-minute walk",
                "ndvi_impact": f"Vegetation health drops from {round(ndvi_before, 3) if ndvi_before else 'unknown'} to {round(ndvi_after, 3) if ndvi_after else 'unknown'} ({'-' if ndvi_before and ndvi_after else '?'}{round((ndvi_before - ndvi_after) * 100, 1) if ndvi_before and ndvi_after else 'unknown'}% loss)",
                "pm25_impact": f"Air pollution increases by {round(pm25_increase_percent, 1) if pm25_increase_percent else 'unknown'}% (from {round(pm25_before, 2) if pm25_before else 'unknown'} to {round(pm25_after, 2) if pm25_after else 'unknown'} μg/m³)"
            },
            "message": f"Environmental Impact Summary:\n\n🏞️ VEGETATION HEALTH (NDVI)\n   • Before: {round(ndvi_before, 3) if ndvi_before else 'Unknown'}\n   • After: {round(ndvi_after, 3) if ndvi_after else 'Unknown'}\n   • Loss: {round((ndvi_before - ndvi_after) * 100, 1) if ndvi_before and ndvi_after else 'Unknown'}% vegetation decline\n\n👥 PEOPLE AFFECTED\n   • Total population losing access: {stats.get('sum_totpop', 0):,} people\n   • Demographics: {stats.get('sum_kidsvc', 0):,} kids, {stats.get('sum_youngp', 0):,} adults, {stats.get('sum_senior', 0):,} seniors\n\n🏭 AIR QUALITY (PM2.5)\n   • Before removal: {round(pm25_before, 2) if pm25_before else 'Unknown'} μg/m³\n   • After removal: {round(pm25_after, 2) if pm25_after else 'Unknown'} μg/m³\n   • Pollution increase: +{round(pm25_increase_percent, 1) if pm25_increase_percent else 'Unknown'}% ({'+' if pm25_increase else ''}{round(pm25_increase, 2) if pm25_increase else 'Unknown'} μg/m³)\n\nRemoving {park_name} would significantly impact {stats.get('sum_totpop', 0):,} residents through reduced air quality, loss of green space, and decreased environmental health.",
        }

async def get_park_ndvi(park_id: str):
    async with db_pool.acquire() as conn:
        sql = "SELECT ST_AsGeoJSON(ST_Transform(geom, 4326))::json AS geometry FROM parks WHERE park_id = $1"
        row = await conn.fetchrow(sql, park_id)
        if not row:
            raise HTTPException(status_code=404, detail="Park not found")

        try:
            geometry = geometry_from_geojson(row["geometry"])
            ndvi_value = compute_ndvi(geometry)
            return ndvi_value
        except ValueError as e:
            logger.error(f"Geometry error for park {park_id}: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid park geometry: {str(e)}")
        except Exception as e:
            logger.error(f"NDVI computation error for park {park_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error computing NDVI: {str(e)}")

async def get_park_information(park_id: str):
    async with db_pool.acquire() as conn:
        # Get park basic information and statistics
        sql = """
            SELECT p.park_name, p.park_addre, p.park_owner, p.park_zip,
                   COALESCE(p.park_size_, NULLIF(p.shape_area,0) * 0.000247105,
                            ST_Area(geography(p.geom)) * 0.000247105) AS area_acres,
                   ps.SUM_TOTPOP, ps.SUM_KIDSVC, ps.SUM_SENIOR, ps.PERACRE
            FROM parks p
            LEFT JOIN parks_stats ps ON p.park_id = ps.park_id
            WHERE p.park_id = $1
        """
        row = await conn.fetchrow(sql, park_id)
        if not row:
            raise HTTPException(status_code=404, detail="Park not found")

        # Get NDVI data
        try:
            ndvi_value = await get_park_ndvi(park_id)
        except:
            ndvi_value = None

        # Get air quality data
        try:
            # Get park geometry for air quality assessment
            geom_sql = "SELECT ST_AsGeoJSON(ST_Transform(geom, 4326))::json AS geometry FROM parks WHERE park_id = $1"
            geom_row = await conn.fetchrow(geom_sql, park_id)
            if geom_row:
                park_geometry = geometry_from_geojson(geom_row["geometry"])
                air_quality = assess_air_quality_and_damage(park_geometry)
            else:
                air_quality = None
        except Exception as e:
            logger.error(f"Error getting air quality for park {park_id}: {e}")
            air_quality = None

        # Generate description using Gemini
        park_data = {
            "name": row["park_name"] or "Unnamed Park",
            "address": row["park_addre"] or "Address not available",
            "owner": row["park_owner"] or "Owner not specified",
            "zipcode": row["park_zip"] or "Unknown",
            "area_acres": round(row["area_acres"], 2) if row["area_acres"] else "Unknown",
            "population_served": row["sum_totpop"] if row["sum_totpop"] else "Unknown",
            "kids_served": row["sum_kidsvc"] if row["sum_kidsvc"] else "Unknown",
            "seniors_served": row["sum_senior"] if row["sum_senior"] else "Unknown",
            "per_acre_demand": row["peracre"] if row["peracre"] else "Unknown",
            "ndvi": round(ndvi_value, 3) if ndvi_value else "Unknown",
            "air_quality": air_quality
        }

        # Use Gemini to generate a descriptive summary
        air_quality_text = ""
        if air_quality:
            air_quality_text = f"""
Air Quality Assessment:
- PM2.5 Concentration: {air_quality.get('pm25_ugm3', 'Unknown')} μg/m³
- Health Risk Level: {air_quality.get('health_risk', 'Unknown')}
- Environmental Impact: {air_quality.get('environmental_damage', 'Unknown')}
- WHO Guideline Exceedance: {air_quality.get('who_guideline_exceedance', 'Unknown')}x
- Health Impact: {air_quality.get('health_impact', 'Unknown')}
- Recommendations: {air_quality.get('recommendations', 'Unknown')}"""

        prompt = f"""Generate a comprehensive description of this park based on the following data:

Park Name: {park_data['name']}
Address: {park_data['address']}
Owner/Manager: {park_data['owner']}
ZIP Code: {park_data['zipcode']}
Area: {park_data['area_acres']} acres
Population Served (10-min walk): {park_data['population_served']} people
Kids Served: {park_data['kids_served']}
Seniors Served: {park_data['seniors_served']}
Demand per Acre: {park_data['per_acre_demand']}
Vegetation Health (NDVI): {park_data['ndvi']}{air_quality_text}

Please provide:
1. A brief overview of the park
2. Key features and characteristics
3. Community impact and demographics served
4. Environmental health indicators (including air quality and pollution impact)
5. Health recommendations based on air quality data
6. Any interesting insights about when it might have been established or its significance

Write in a friendly, informative tone suitable for residents and visitors. Pay special attention to air quality concerns and environmental health."""

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt
            )

            description = response.text
            logger.info(f"Generated park description for {park_id}")

        except Exception as e:
            logger.error(f"Error generating park description: {e}")
            description = f"""**{park_data['name']}**

This {park_data['area_acres']}-acre park is located at {park_data['address']} in ZIP code {park_data['zipcode']}.
It's managed by {park_data['owner']} and serves approximately {park_data['population_served']} people within a 10-minute walk.

The park provides recreational opportunities for {park_data['kids_served']} children and {park_data['seniors_served']} seniors in the surrounding community.
With a vegetation health index (NDVI) of {park_data['ndvi']}, it contributes to the local environmental quality and urban green space."""

        return {
            "parkId": park_id,
            "parkName": park_data['name'],
            "description": description,
            "details": park_data
        }

async def get_park_air_quality(park_id: str):
    """Get comprehensive air quality data for a specific park"""
    async with db_pool.acquire() as conn:
        # Get park basic information
        sql = """
            SELECT park_name, ST_AsGeoJSON(ST_Transform(geom, 4326))::json AS geometry
            FROM parks
            WHERE park_id = $1
        """
        row = await conn.fetchrow(sql, park_id)
        if not row:
            raise HTTPException(status_code=404, detail="Park not found")

        park_name = row["park_name"] or "Unnamed Park"
        geometry = row["geometry"]

        try:
            park_geom = geometry_from_geojson(geometry)

            # Get current air quality
            current_air_quality = assess_air_quality_and_damage(park_geom)

            return {
                "parkId": park_id,
                "parkName": park_name,
                "currentAirQuality": current_air_quality,
                "message": f"Air quality assessment for {park_name}: PM2.5 level is {current_air_quality.get('pm25_ugm3', 'unknown')} μg/m³ ({current_air_quality.get('health_risk', 'unknown')} risk level)"
            }

        except ValueError as e:
            logger.error(f"Geometry error for park {park_id}: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid park geometry: {str(e)}")
        except Exception as e:
            logger.error(f"Air quality computation error for park {park_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error computing air quality: {str(e)}")

async def analyze_park_removal_pollution_impact(park_id: str, land_use_type: str = "removed"):
    """Analyze how park removal affects pollution levels"""
    async with db_pool.acquire() as conn:
        sql = """
            SELECT park_name, ST_AsGeoJSON(ST_Transform(geom, 4326))::json AS geometry
            FROM parks
            WHERE park_id = $1
        """
        row = await conn.fetchrow(sql, park_id)
        if not row:
            return None

        park_name = row["park_name"] or "Unnamed Park"
        geometry = row["geometry"]

        try:
            park_geom = geometry_from_geojson(geometry)
            buffer_geom = park_geom.buffer(1000)  # 1km buffer for pollution assessment

            # Current air quality
            current_pm25 = compute_pm25(buffer_geom)

            # Simulate pollution increase after park removal
            if land_use_type == "removed":
                # Parks act as air filters - removing them increases PM2.5 by ~15-25%
                pollution_increase_factor = 1.2  # 20% increase
                impact_description = "complete removal"
            elif land_use_type == "replaced_by_building":
                # Buildings and concrete increase urban heat island and pollution
                pollution_increase_factor = 1.35  # 35% increase
                impact_description = "replacement with buildings"
            else:
                pollution_increase_factor = 1.15  # 15% increase
                impact_description = "modification"

            # Calculate estimated pollution after removal
            if current_pm25:
                estimated_pm25_after = current_pm25 * pollution_increase_factor
                pollution_increase = estimated_pm25_after - current_pm25
            else:
                estimated_pm25_after = None
                pollution_increase = None

            # Assess new health risk level
            if estimated_pm25_after:
                # Create temporary geometry for assessment
                temp_geom = park_geom  # Use park geometry for assessment
                after_assessment = {
                    "pm25_ugm3": round(estimated_pm25_after, 2),
                    "health_risk": get_health_risk_category(estimated_pm25_after),
                    "environmental_damage": get_environmental_damage_level(estimated_pm25_after)
                }
            else:
                after_assessment = None

            return {
                "parkId": park_id,
                "parkName": park_name,
                "landUseType": land_use_type,
                "currentPM25": round(current_pm25, 2) if current_pm25 else None,
                "estimatedPM25After": round(estimated_pm25_after, 2) if estimated_pm25_after else None,
                "pollutionIncrease": round(pollution_increase, 2) if pollution_increase else None,
                "pollutionIncreasePercent": round((pollution_increase_factor - 1) * 100, 1),
                "afterAssessment": after_assessment,
                "impactDescription": impact_description,
                "message": f"If {park_name} is {impact_description}, PM2.5 levels could increase by {round((pollution_increase_factor - 1) * 100, 1)}% (from {round(current_pm25, 2) if current_pm25 else 'unknown'} to {round(estimated_pm25_after, 2) if estimated_pm25_after else 'unknown'} μg/m³), worsening air quality for the surrounding area."
            }

        except Exception as e:
            logger.error(f"Error in pollution impact analysis: {e}")
            return None

def get_health_risk_category(pm25_value):
    """Get health risk category for a given PM2.5 value"""
    if pm25_value <= 5:
        return "Low"
    elif pm25_value <= 12:
        return "Moderate"
    elif pm25_value <= 25:
        return "Unhealthy for Sensitive Groups"
    elif pm25_value <= 35:
        return "Unhealthy"
    elif pm25_value <= 50:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_environmental_damage_level(pm25_value):
    """Get environmental damage level for a given PM2.5 value"""
    if pm25_value <= 5:
        return "Minimal environmental impact"
    elif pm25_value <= 12:
        return "Low to moderate environmental stress"
    elif pm25_value <= 25:
        return "Moderate environmental degradation"
    elif pm25_value <= 35:
        return "Significant environmental stress"
    elif pm25_value <= 50:
        return "Severe environmental degradation"
    else:
        return "Critical environmental damage"

@app.post("/api/agent")
async def agent_endpoint(request: AgentRequest):
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

        elif parsed.get("intent") == "park_info_query":
            if not selected_park_id:
                return {
                    "sessionId": session_id,
                    "action": "need_selection",
                    "reply": "Please select a park to get information about.",
                }

            park_info = await get_park_information(selected_park_id)

            return {
                "sessionId": session_id,
                "action": "park_information",
                "reply": park_info["description"],
                "data": park_info,
            }

        elif parsed.get("intent") == "air_quality_query":
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

@app.post("/api/ndvi")
async def ndvi_endpoint(request: NDVIRequest):
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

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 4000)))