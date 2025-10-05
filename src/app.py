from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json

#
# --- Data Structure for Request Body ---
class MetricsRequest(BaseModel):
    regions: list[str]
    threshold_ms: int

# --- FastAPI App Initialization ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"] # important
)


# --- Sample Telemetry Data (Loaded from JSON) ---
# NOTE: In a real deployment, this would be loaded from a database or storage.
TELEMETRY_DATA_JSON = """
[
  {"region": "apac", "service": "payments", "latency_ms": 206.53, "uptime_pct": 97.942, "timestamp": 20250301},
  {"region": "apac", "service": "payments", "latency_ms": 216, "uptime_pct": 99.289, "timestamp": 20250302},
  {"region": "apac", "service": "recommendations", "latency_ms": 136.47, "uptime_pct": 97.192, "timestamp": 20250303},
  {"region": "apac", "service": "checkout", "latency_ms": 224.4, "uptime_pct": 97.589, "timestamp": 20250304},
  {"region": "apac", "service": "analytics", "latency_ms": 215.35, "uptime_pct": 98.567, "timestamp": 20250305},
  {"region": "apac", "service": "analytics", "latency_ms": 230.57, "uptime_pct": 98.256, "timestamp": 20250306},
  {"region": "apac", "service": "support", "latency_ms": 198.59, "uptime_pct": 98.063, "timestamp": 20250307},
  {"region": "apac", "service": "recommendations", "latency_ms": 181.23, "uptime_pct": 99.136, "timestamp": 20250308},
  {"region": "apac", "service": "payments", "latency_ms": 220.27, "uptime_pct": 98.762, "timestamp": 20250309},
  {"region": "apac", "service": "support", "latency_ms": 115.38, "uptime_pct": 97.644, "timestamp": 20250310},
  {"region": "apac", "service": "catalog", "latency_ms": 210.07, "uptime_pct": 98.969, "timestamp": 20250311},
  {"region": "apac", "service": "payments", "latency_ms": 240.39, "uptime_pct": 97.53, "timestamp": 20250312},
  {"region": "emea", "service": "checkout", "latency_ms": 178.48, "uptime_pct": 98.46, "timestamp": 20250301},
  {"region": "emea", "service": "support", "latency_ms": 169.27, "uptime_pct": 99.079, "timestamp": 20250302},
  {"region": "emea", "service": "analytics", "latency_ms": 199.58, "uptime_pct": 98.038, "timestamp": 20250303},
  {"region": "emea", "service": "checkout", "latency_ms": 212.44, "uptime_pct": 98.16, "timestamp": 20250304},
  {"region": "emea", "service": "support", "latency_ms": 215.56, "uptime_pct": 97.279, "timestamp": 20250305},
  {"region": "emea", "service": "support", "latency_ms": 207.96, "uptime_pct": 97.765, "timestamp": 20250306},
  {"region": "emea", "service": "support", "latency_ms": 146.69, "uptime_pct": 98.851, "timestamp": 20250307},
  {"region": "emea", "service": "catalog", "latency_ms": 133.16, "uptime_pct": 98.024, "timestamp": 20250308},
  {"region": "emea", "service": "payments", "latency_ms": 171.45, "uptime_pct": 97.933, "timestamp": 20250309},
  {"region": "emea", "service": "analytics", "latency_ms": 136.21, "uptime_pct": 99.248, "timestamp": 20250310},
  {"region": "emea", "service": "checkout", "latency_ms": 178.61, "uptime_pct": 98.066, "timestamp": 20250311},
  {"region": "emea", "service": "analytics", "latency_ms": 123.7, "uptime_pct": 99.195, "timestamp": 20250312},
  {"region": "amer", "service": "payments", "latency_ms": 145.8, "uptime_pct": 98.422, "timestamp": 20250301},
  {"region": "amer", "service": "payments", "latency_ms": 107.55, "uptime_pct": 97.53, "timestamp": 20250302},
  {"region": "amer", "service": "payments", "latency_ms": 213.97, "uptime_pct": 97.271, "timestamp": 20250303},
  {"region": "amer", "service": "analytics", "latency_ms": 191.78, "uptime_pct": 98.776, "timestamp": 20250304},
  {"region": "amer", "service": "payments", "latency_ms": 161.77, "uptime_pct": 98.71, "timestamp": 20250305},
  {"region": "amer", "service": "payments", "latency_ms": 185.96, "uptime_pct": 97.521, "timestamp": 20250306},
  {"region": "amer", "service": "analytics", "latency_ms": 206.21, "uptime_pct": 97.118, "timestamp": 20250307},
  {"region": "amer", "service": "checkout", "latency_ms": 184.43, "uptime_pct": 99.256, "timestamp": 20250308},
  {"region": "amer", "service": "analytics", "latency_ms": 169.43, "uptime_pct": 97.982, "timestamp": 20250309},
  {"region": "amer", "service": "checkout", "latency_ms": 146.48, "uptime_pct": 97.928, "timestamp": 20250310},
  {"region": "amer", "service": "catalog", "latency_ms": 169.78, "uptime_pct": 99.313, "timestamp": 20250311},
  {"region": "amer", "service": "catalog", "latency_ms": 125.87, "uptime_pct": 99.352, "timestamp": 20250312}
]
"""
df = pd.DataFrame(json.loads(TELEMETRY_DATA_JSON))

# --- Metrics Calculation Function ---
def calculate_metrics(df_filtered, threshold):
    """Calculates metrics for a given filtered DataFrame."""
    if df_filtered.empty:
        return {
            "avg_latency": 0.0,
            "p95_latency": 0.0,
            "avg_uptime": 0.0,
            "breaches": 0,
        }

    latencies = df_filtered['latency_ms']
    uptimes = df_filtered['uptime_pct']
    
    # Calculate metrics
    avg_latency = latencies.mean()
    p95_latency = np.percentile(latencies, 95)
    avg_uptime = uptimes.mean()
    breaches = (latencies > threshold).sum()

    return {
        "avg_latency": round(avg_latency, 2),
        "p95_latency": round(p95_latency, 2),
        "avg_uptime": round(avg_uptime, 2),
        "breaches": int(breaches),
    }

# --- POST Endpoint ---
@app.post("/api/latency")
async def get_telemetry_metrics(data: MetricsRequest):
    """
    Calculates and returns per-region performance metrics based on the input regions and latency threshold.
    """
    results = {}
    threshold = data.threshold_ms

    for region in data.regions:
        # Filter by region
        df_region = df[df['region'] == region.lower()]
        
        # Calculate metrics and add to results dictionary
        metrics = calculate_metrics(df_region, threshold)
        results[region.lower()] = metrics # Ensure region keys are lowercase for consistency
    
    return {"regions": results}

# --- Test Endpoint (Optional, for Vercel health check) ---
@app.get("/")
def read_root():
    return {"status": "Telemetry service is running."}
