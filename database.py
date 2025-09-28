import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager
import asyncpg
import ee
from fastapi import HTTPException
from google import genai
import logging
from dotenv import load_dotenv

load_dotenv()

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

async def close_db():
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")

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

async def query_parks_by_location(query, simplify_tolerance: float = 0.0002):
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

async def get_park_ndvi(park_id: str):
    from utils import geometry_from_geojson, compute_ndvi
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

async def get_park_information(park_id: str, client):
    from utils import geometry_from_geojson, assess_air_quality_and_damage
    async with db_pool.acquire() as conn:
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

        try:
            ndvi_value = await get_park_ndvi(park_id)
        except:
            ndvi_value = None

        try:
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
    from utils import geometry_from_geojson, assess_air_quality_and_damage
    async with db_pool.acquire() as conn:
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

async def analyze_park_removal_impact(park_id: str, land_use_type: str = "removed"):
    from utils import (geometry_from_geojson, compute_ndvi, compute_walkability,
                      compute_pm25, compute_population, simulate_replacement_with_buildings)

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
            pm25_pollution_factor = 1.2
        elif land_use_type == "replaced_by_building":
            ndvi_after = simulate_replacement_with_buildings(buffer_geom, park_geom)
            pm25_pollution_factor = 1.35
        else:
            ndvi_after = ndvi_before
            pm25_pollution_factor = 1.1

        walkability_after = compute_walkability(buffer_geom.difference(park_geom))

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

async def analyze_park_removal_pollution_impact(park_id: str, land_use_type: str = "removed"):
    from utils import geometry_from_geojson, compute_pm25, get_health_risk_category, get_environmental_damage_level

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
            buffer_geom = park_geom.buffer(1000)

            current_pm25 = compute_pm25(buffer_geom)

            if land_use_type == "removed":
                pollution_increase_factor = 1.2
                impact_description = "complete removal"
            elif land_use_type == "replaced_by_building":
                pollution_increase_factor = 1.35
                impact_description = "replacement with buildings"
            else:
                pollution_increase_factor = 1.15
                impact_description = "modification"

            if current_pm25:
                estimated_pm25_after = current_pm25 * pollution_increase_factor
                pollution_increase = estimated_pm25_after - current_pm25
            else:
                estimated_pm25_after = None
                pollution_increase = None

            if estimated_pm25_after:
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