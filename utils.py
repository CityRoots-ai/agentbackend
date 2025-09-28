import os
import json
import ee
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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