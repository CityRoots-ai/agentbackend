from typing import Optional, Dict, Any
from pydantic import BaseModel
from enum import Enum

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
    CREATE_PROPOSAL = "create_proposal"

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