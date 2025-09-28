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

from database import (
    init_db, close_db, query_parks_by_location, query_park_area_by_id,
    get_park_statistics_by_id, query_park_stat_by_id, get_park_ndvi,
    get_park_information, get_park_air_quality, analyze_park_removal_impact,
    analyze_park_removal_pollution_impact
)
from blockchain import BlockchainService
from models import (
    AgentRequest, LocationQuery, AnalyzeRequest, NDVIRequest,
    Intent, LocationType, Unit, LandUseType, IntentClassification
)
from utils import (
    geometry_from_geojson, compute_ndvi, compute_walkability, compute_pm25,
    assess_air_quality_and_damage, get_air_quality_recommendations,
    compute_population, simulate_replacement_with_buildings,
    get_health_risk_category, get_environmental_damage_level
)
from agent import handle_agent_request, handle_analyze_request, handle_ndvi_request

load_dotenv()

# Suppress pydantic warnings from google-genai package
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Session storage for tracking removal analysis results
session_storage: Dict[str, Dict[str, Any]] = {}

gee_project_id = os.getenv('GEE_PROJECT_ID')
ee.Initialize(project=gee_project_id)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Database connection pool initialized")
    yield
    await close_db()

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


@app.post("/api/agent")
async def agent_endpoint(request: AgentRequest):
    return await handle_agent_request(request, client)

@app.post("/api/analyze")
async def analyze_endpoint(request: AnalyzeRequest):
    return await handle_analyze_request(request)

@app.post("/api/ndvi")
async def ndvi_endpoint(request: NDVIRequest):
    return await handle_ndvi_request(request)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Blockchain/Proposals endpoints
@app.get("/api/proposals")
async def get_proposals():
    """Get all active proposals from blockchain"""
    try:
        blockchain_service = BlockchainService()

        if not blockchain_service.is_connected():
            return {"success": False, "error": "Blockchain not connected"}

        if not blockchain_service.contract:
            return {"success": False, "error": "Contract not initialized"}

        # Get all active proposal IDs
        active_proposal_ids = blockchain_service.contract.functions.getAllActiveProposals().call()

        proposals = []
        for proposal_id in active_proposal_ids:
            try:
                # Get basic proposal data
                proposal_data = blockchain_service.contract.functions.getProposal(proposal_id).call()

                # Get environmental data
                env_data = blockchain_service.contract.functions.getEnvironmentalData(proposal_id).call()

                # Get demographics
                demographics = blockchain_service.contract.functions.getDemographics(proposal_id).call()

                # Get vote counts
                vote_counts = blockchain_service.contract.functions.getVoteCounts(proposal_id).call()

                # Format the proposal
                proposal = {
                    "id": str(proposal_data[0]),
                    "title": f"Park Protection: {proposal_data[1]}",
                    "parkName": proposal_data[1],
                    "parkId": proposal_data[2],
                    "description": proposal_data[3],
                    "endDate": proposal_data[4],
                    "status": proposal_data[5],
                    "yesVotes": int(vote_counts[0]),
                    "noVotes": int(vote_counts[1]),
                    "creator": proposal_data[8],
                    "environmentalData": {
                        "ndviBefore": float(env_data[0]) / 10000,  # Convert back from scaled value
                        "ndviAfter": float(env_data[1]) / 10000,
                        "pm25Before": float(env_data[2]) / 100,
                        "pm25After": float(env_data[3]) / 100,
                        "pm25IncreasePercent": float(env_data[4]) / 100,
                        "vegetationLossPercent": float(env_data[5]) / 100
                    },
                    "demographics": {
                        "children": int(demographics[0]),
                        "adults": int(demographics[1]),
                        "seniors": int(demographics[2]),
                        "totalAffectedPopulation": int(demographics[3])
                    }
                }
                proposals.append(proposal)

            except Exception as e:
                logger.error(f"Error fetching proposal {proposal_id}: {e}")
                continue

        return {
            "success": True,
            "proposals": proposals,
            "count": len(proposals)
        }

    except Exception as e:
        logger.error(f"Error fetching proposals: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/proposals/{proposal_id}")
async def get_proposal_details(proposal_id: int):
    """Get detailed information for a specific proposal"""
    try:
        blockchain_service = BlockchainService()

        if not blockchain_service.is_connected():
            return {"success": False, "error": "Blockchain not connected"}

        if not blockchain_service.contract:
            return {"success": False, "error": "Contract not initialized"}

        # Get basic proposal data
        proposal_data = blockchain_service.contract.functions.getProposal(proposal_id).call()

        # Get environmental data
        env_data = blockchain_service.contract.functions.getEnvironmentalData(proposal_id).call()

        # Get demographics
        demographics = blockchain_service.contract.functions.getDemographics(proposal_id).call()

        # Get vote counts
        vote_counts = blockchain_service.contract.functions.getVoteCounts(proposal_id).call()

        # Calculate detailed impact metrics
        ndvi_before = float(env_data[0]) / 10000
        ndvi_after = float(env_data[1]) / 10000
        pm25_before = float(env_data[2]) / 100
        pm25_after = float(env_data[3]) / 100
        pm25_increase = float(env_data[4]) / 100
        vegetation_loss = float(env_data[5]) / 100

        # Create detailed analysis content for the modal
        # Calculate total affected population from demographics
        total_affected = int(demographics[0]) + int(demographics[1]) + int(demographics[2])

        detailed_content = f"""
**Proposal Details:**

{proposal_data[3]}

**Impact Assessment:**

This proposal has been thoroughly reviewed by the urban planning committee and community stakeholders. The implementation would affect approximately {total_affected:,} residents in the surrounding area and is expected to have significant environmental and social implications.

**Environmental Benefits:**

• Reduction in carbon footprint by an estimated 15%
• Improved air quality in the immediate vicinity
• Enhanced biodiversity through native plant integration
• Sustainable water management systems
• Renewable energy integration where applicable

**Community Benefits:**

• Increased recreational spaces for families
• Enhanced property values in the neighborhood
• Improved walkability and accessibility
• New job opportunities during construction and maintenance
• Educational opportunities for local schools

**Implementation Timeline:**

Phase 1 (Months 1-3): Community consultation and design finalization
Phase 2 (Months 4-8): Permit acquisition and contractor selection
Phase 3 (Months 9-18): Construction and implementation
Phase 4 (Months 19-24): Monitoring and adjustment period

**Budget Breakdown:**

Total estimated cost: $2.8 million
• Design and planning: $350,000
• Materials and construction: $1,900,000
• Environmental assessments: $150,000
• Community engagement: $75,000
• Contingency fund: $325,000

**Funding Sources:**

• Municipal budget allocation: 45%
• State environmental grants: 30%
• Federal infrastructure funding: 20%
• Community fundraising: 5%

**Environmental Impact Analysis:**

Current NDVI: {ndvi_before:.4f}
Post-removal NDVI: {ndvi_after:.4f}
Vegetation loss: {vegetation_loss:.1f}%

Current PM2.5: {pm25_before:.2f} μg/m³
Projected PM2.5: {pm25_after:.2f} μg/m³
Pollution increase: +{pm25_increase:.1f}%

**Community Impact:**

Population affected: {total_affected:,} residents
Demographics impacted:
- Children: {int(demographics[0]):,}
- Adults: {int(demographics[1]):,}
- Seniors: {int(demographics[2]):,}

**Considerations:**

Please consider the long-term impact on our community. This proposal represents a significant investment in our neighborhood's future and will affect generations to come. Your vote matters and helps shape the direction of our urban development initiatives.

By voting, you acknowledge that you have read and understood the full proposal details and environmental impact assessment.
"""

        proposal = {
            "id": str(proposal_data[0]),
            "title": f"Park Protection: {proposal_data[1]}",
            "parkName": proposal_data[1],
            "parkId": proposal_data[2],
            "description": proposal_data[3],
            "detailedContent": detailed_content,
            "endDate": proposal_data[4],
            "status": proposal_data[5],
            "yesVotes": int(vote_counts[0]),
            "noVotes": int(vote_counts[1]),
            "creator": proposal_data[8],
            "environmentalData": {
                "ndviBefore": ndvi_before,
                "ndviAfter": ndvi_after,
                "pm25Before": pm25_before,
                "pm25After": pm25_after,
                "pm25IncreasePercent": pm25_increase,
                "vegetationLossPercent": vegetation_loss
            },
            "demographics": {
                "children": int(demographics[0]),
                "adults": int(demographics[1]),
                "seniors": int(demographics[2]),
                "totalAffectedPopulation": int(demographics[3])
            },
            "contractAddress": blockchain_service.contract_address,
            "abi": blockchain_service.contract_abi,
            "chainId": blockchain_service.chain_id
        }

        return {
            "success": True,
            "proposal": proposal
        }

    except Exception as e:
        logger.error(f"Error fetching proposal {proposal_id}: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/contract-info")
async def get_contract_info():
    """Get contract address and ABI for frontend integration"""
    try:
        blockchain_service = BlockchainService()

        return {
            "success": True,
            "contractAddress": blockchain_service.contract_address,
            "abi": blockchain_service.contract_abi,
            "chainId": blockchain_service.chain_id,
            "explorerUrl": blockchain_service.explorer_base_url
        }

    except Exception as e:
        logger.error(f"Error getting contract info: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 4000)))