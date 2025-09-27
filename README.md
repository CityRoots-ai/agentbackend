# CityRoots API Backend

A FastAPI backend for urban planning and GIS-aware analysis, combining PostgreSQL database operations with Google Earth Engine integration for park analysis and environmental impact assessment.

## Features

- Natural language query processing with OpenAI GPT-4o-mini
- PostgreSQL database integration for parks data
- Google Earth Engine integration for NDVI, walkability, and PM2.5 analysis
- Park search by location (zip, city, state)
- Environmental impact analysis for park removal scenarios
- Demographic and statistical analysis for park service areas

## Prerequisites

- Python 3.8+
- PostgreSQL database with parks data
- Google Earth Engine account
- OpenAI API key

## Setup Instructions

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd cityrootsbe
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip3 install -r requirements.txt
```

### 4. Environment Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```env
PGHOST=localhost
PGPORT=5432
PGDATABASE=cityroots
PGUSER=your_username
PGPASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
PORT=4000
GEE_PROJECT_ID=your-gee-project-id
```

### 5. Database Setup

Ensure your PostgreSQL database contains the required tables:
- `parks` - Park geometries and metadata
- `parks_stats` - Demographic and statistical data for parks

### 6. Google Earth Engine Authentication

Authenticate with Google Earth Engine:

```bash
earthengine authenticate
```

### 7. Run the Application

```bash
source venv/bin/activate
python main.py
```

The API will be available at `http://localhost:4000`

## API Documentation

Once running, visit `http://localhost:4000/docs` for interactive API documentation.

## Main Endpoints

- `POST /api/agent` - Natural language agent for park queries
- `POST /api/analyze` - Environmental impact analysis
- `POST /api/ndvi` - NDVI calculation for geometries
- `GET /health` - Health check

## Usage Examples

### Natural Language Queries

```bash
curl -X POST "http://localhost:4000/api/agent" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "show parks in Austin",
    "uiContext": {},
    "sessionId": "test123"
  }'
```

### Park Area Query

```bash
curl -X POST "http://localhost:4000/api/agent" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "what is the area of this park in hectares?",
    "uiContext": {
      "selectedParkId": "PARK_001"
    },
    "sessionId": "test123"
  }'
```

### Environmental Impact Analysis

```bash
curl -X POST "http://localhost:4000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "geometry": {"type": "Point", "coordinates": [-97.7431, 30.2672]},
    "landUseType": "removed"
  }'
```

## Development

### Running in Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 4000
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PGHOST` | PostgreSQL host | Yes |
| `PGPORT` | PostgreSQL port | Yes |
| `PGDATABASE` | Database name | Yes |
| `PGUSER` | Database username | Yes |
| `PGPASSWORD` | Database password | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `GEE_PROJECT_ID` | Google Earth Engine project ID | Yes |
| `PORT` | Server port | No (default: 4000) |