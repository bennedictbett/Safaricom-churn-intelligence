"""
Safaricom Churn Intelligence API
FastAPI-powered real-time churn prediction service
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Literal
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import joblib
import io
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model loading
MODEL = None
FEATURE_NAMES = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, FEATURE_NAMES
    model_path = os.getenv("MODEL_PATH", "models/churn_model.pkl")
    try:
        with open(model_path, "rb") as f:
            artifact = joblib.load(f)
        if isinstance(artifact, dict):
            MODEL = artifact["model"]
            FEATURE_NAMES = artifact.get("feature_names")
        else:
            MODEL = artifact
        logger.info("✅ Model loaded from %s", model_path)
    except FileNotFoundError:
        logger.warning("⚠️  Model not found at %s — /predict will return 503", model_path)
    except Exception as e:
        logger.warning("⚠️  Could not load model: %s — /predict will return 503", e)
    yield



# App setup  (only ONE app definition)

app = FastAPI(
    title="Safaricom Churn Intelligence API",
    lifespan=lifespan,
    description="""
## ML-Powered Customer Churn Prediction

Predict churn risk for Safaricom customers using Kenya-specific features.

### Endpoints
- **POST /predict** — Single customer churn prediction
- **POST /predict/batch** — Batch predictions via CSV upload
- **GET /feature-importance** — Top churn drivers from the model
- **GET /health** — Service health check
    """,
    version="1.0.0",
    contact={
        "name": "Benedict Bett",
        "email": "benedictbett08@gmail.com",
        "url": "https://github.com/bennedictbett/Safaricom-churn-intelligence",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landing_page.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()

@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
def prediction_app():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()
# Schemas  (only ONE CustomerFeatures definition)


class CustomerFeatures(BaseModel):
    # Standard telco features
    tenure: int = Field(..., ge=0, le=72, description="Months as a customer")
    monthly_charges: float = Field(..., ge=0, description="Monthly bill in KES")
    total_charges: float = Field(..., ge=0, description="Cumulative charges in KES")
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract type")
    payment_method: str = Field(..., description="'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'")
    paperless_billing: bool = Field(default=True)
    phone_service: bool = Field(default=True)
    multiple_lines: str = Field(default="No", description="'Yes', 'No', 'No phone service'")
    internet_service: str = Field(default="Fiber optic", description="'DSL', 'Fiber optic', 'No'")
    online_security: str = Field(default="No", description="'Yes', 'No', 'No internet service'")
    online_backup: str = Field(default="No")
    device_protection: str = Field(default="No")
    tech_support: str = Field(default="No")
    streaming_tv: str = Field(default="No")
    streaming_movies: str = Field(default="No")
    senior_citizen: bool = Field(default=False)
    partner: bool = Field(default=False)
    dependents: bool = Field(default=False)
    # Kenya-specific features
    mpesa_usage_score: float = Field(default=5.0, ge=0, le=10, description="M-Pesa engagement score (0-10)")
    bonga_points_active: bool = Field(default=False, description="Actively redeeming Bonga points")
    safaricom_home: bool = Field(default=False, description="Subscribed to Safaricom Home fiber")
    county: str = Field(default="Nairobi", description="Customer's county in Kenya")
    network_quality_score: float = Field(default=7.0, ge=0, le=10, description="Perceived network quality (0-10)")
    competitor_exposure: float = Field(default=3.0, ge=0, le=10, description="Competitor marketing exposure (0-10)")
    rural: bool = Field(default=False, description="Rural customer flag")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "tenure": 8,
            "monthly_charges": 3200.0,
            "total_charges": 25600.0,
            "contract": "Month-to-month",
            "payment_method": "Electronic check",
            "paperless_billing": True,
            "phone_service": True,
            "multiple_lines": "No",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "Yes",
            "streaming_movies": "Yes",
            "senior_citizen": False,
            "partner": False,
            "dependents": False,
            "mpesa_usage_score": 3.5,
            "bonga_points_active": False,
            "safaricom_home": False,
            "county": "Nakuru",
            "network_quality_score": 4.0,
            "competitor_exposure": 7.0,
            "rural": True,
        }
    })


class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    risk_level: str
    risk_score: int
    top_churn_drivers: List[str]
    recommended_actions: List[str]
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

# Constants

FEATURE_IMPORTANCE_LABELS = {
    "contract_Month-to-month": "Month-to-month contract (no commitment)",
    "tenure": "Short customer tenure",
    "monthly_charges": "High monthly charges",
    "mpesa_usage_score": "Low M-Pesa engagement",
    "network_quality_score": "Poor network quality perception",
    "competitor_exposure": "High competitor exposure",
    "internet_service_Fiber optic": "Fiber optic internet",
    "online_security_No": "No online security add-on",
    "tech_support_No": "No tech support add-on",
    "bonga_points_active": "Inactive Bonga loyalty points",
    "safaricom_home": "Not subscribed to Safaricom Home",
    "rural": "Rural location",
}

RETENTION_ACTIONS = {
    "high": [
        "🚨 Personal outreach within 24 hours",
        "Offer 2x Bonga points for 3 months",
        "Propose annual contract with 2 months free",
        "Escalate to retention specialist team",
        "Provide dedicated network quality report for their area",
    ],
    "medium": [
        "📞 Proactive call within 1 week",
        "Offer M-Pesa bundle discount (20% off for 6 months)",
        "Activate Bonga loyalty programme",
        "Send personalised network upgrade notification",
        "Offer device upgrade tied to contract renewal",
    ],
    "low": [
        "✅ Maintain regular engagement",
        "Include in quarterly satisfaction survey",
        "Send Bonga points rewards reminder",
        "Offer referral bonus programme",
    ],
}

# Helpers

def get_risk_level(prob: float):
    if prob >= 0.31:      # Top 25% → High
        return "High", 3
    elif prob >= 0.26:    # Middle 50% → Medium  
        return "Medium", 2
    return "Low", 1       # Bottom 25% → Low


def preprocess(data: dict) -> pd.DataFrame:
    """Map API input fields to the 40 features the model expects."""

    tenure      = data.get('tenure', 0)
    monthly     = data.get('monthly_charges', 0)
    mpesa       = data.get('mpesa_usage_score', 5.0)
    network     = data.get('network_quality_score', 7.0)
    competitor  = data.get('competitor_exposure', 3.0)
    is_rural    = int(data.get('rural', False))
    bonga_on    = int(data.get('bonga_points_active', False))

    # Categorical encodings — matched exactly to label encoder order from training
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    internet_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    payment_map  = {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1,
                    'Electronic check': 2, 'Mailed check': 3}
    lines_map    = {'No': 0, 'No phone service': 1, 'Yes': 2}
    county_map   = {'Kakamega': 0, 'Kisumu': 1, 'Machakos': 2, 'Malindi': 3,
                    'Mombasa': 4, 'Nairobi': 5, 'Nakuru': 6, 'Nyeri': 7,
                    'Thika': 8, 'Uasin Gishu': 9}
    yesno_map    = {'No': 0, 'No internet service': 1, 'Yes': 2}


    # ── Engineered features with correct ranges ──
    mpesa_eng    = 1 if mpesa < 4 else (2 if mpesa < 7 else 0)   # high=0, low=1, medium=2
    net_sat      = 0 if network < 4 else (1 if network < 7 else 2)
    bundle_tier  = 0 if monthly <= 2000 else (2 if monthly <= 4000 else (1 if monthly <= 6000 else 3))
    location     = 0 if is_rural else 1  # rural=0, urban=1

    bonga_pts    = mpesa * 432.4  

    days_bonga   = int(364 - (mpesa * 36.4)) 
    if bonga_on:
        days_bonga = min(days_bonga, 90)

    avg_data     = mpesa * 4.5 if data.get('internet_service') == 'Fiber optic' else (mpesa * 2.0 if data.get('internet_service') == 'DSL' else mpesa * 0.5)

    dig_loyalty  = (mpesa * 3.5) + (network * 2.5) + ((10 - competitor) * 1.5) + (bonga_on * 10) + (int(data.get('safaricom_home', False)) * 8)

    engagement   = (mpesa * 2.5) + (network * 2.0) + ((10 - competitor) * 1.5) + (bonga_on * 8) + (tenure * 0.3)
    engagement   = min(round(engagement, 2), 90)


    mpesa_trans  = int(mpesa * 12) 

    row = {
        'gender':                       0,
        'SeniorCitizen':                int(data.get('senior_citizen', False)),
        'Partner':                      int(data.get('partner', False)),
        'Dependents':                   int(data.get('dependents', False)),
        'tenure':                       tenure,
        'PhoneService':                 int(data.get('phone_service', True)),
        'MultipleLines':                lines_map.get(data.get('multiple_lines', 'No'), 0),
        'InternetService':              internet_map.get(data.get('internet_service', 'Fiber optic'), 1),
        'OnlineSecurity':               yesno_map.get(data.get('online_security', 'No'), 0),
        'OnlineBackup':                 yesno_map.get(data.get('online_backup', 'No'), 0),
        'DeviceProtection':             yesno_map.get(data.get('device_protection', 'No'), 0),
        'TechSupport':                  yesno_map.get(data.get('tech_support', 'No'), 0),
        'StreamingTV':                  yesno_map.get(data.get('streaming_tv', 'No'), 0),
        'StreamingMovies':              yesno_map.get(data.get('streaming_movies', 'No'), 0),
        'Contract':                     contract_map.get(data.get('contract', 'Month-to-month'), 0),
        'PaperlessBilling':             int(data.get('paperless_billing', True)),
        'PaymentMethod':                payment_map.get(data.get('payment_method', 'Electronic check'), 2),
        'MonthlyCharges':               monthly,
        'TotalCharges':                 data.get('total_charges', 0),
        'county':                       county_map.get(data.get('county', 'Nairobi'), 5),
        'is_rural':                     is_rural,
        'location_type':                location,
        'mpesa_usage_score':            mpesa,
        'mpesa_engagement':             mpesa_eng,
        'mpesa_monthly_transactions':   mpesa_trans,
        'bonga_points':                 bonga_pts,
        'days_since_bonga_redemption':  days_bonga,
        'bonga_active':                 bonga_on,
        'has_safaricom_home':           int(data.get('safaricom_home', False)),
        'competitor_exposure':          int(competitor),
        'high_competitor_risk':         int(competitor >= 7),
        'network_quality_score':        int(network),
        'network_satisfaction':         net_sat,
        'uses_data_rollover':           int(data.get('internet_service', 'Fiber optic') != 'No'),
        'data_bundle_tier':             bundle_tier,
        'avg_monthly_data_gb':          avg_data,
        'PaperlessBilling':             1 if data.get('paperless_billing', True) else 0,
        'digital_loyalty_score':        dig_loyalty,
        'rural_network_risk':           int(is_rural and network < 5),
        'price_sensitive_risk':         int(monthly > 4000 and data.get('contract') == 'Month-to-month'),
        'customer_engagement_score':    engagement,
    }

    df = pd.DataFrame([row])

    if FEATURE_NAMES is not None:
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0
        df = df[FEATURE_NAMES]

    return df


def get_top_drivers(data: dict, n: int = 4) -> List[str]:
    drivers = []
    if data.get("contract") == "Month-to-month":
        drivers.append("Month-to-month contract (no long-term commitment)")
    if data.get("tenure", 99) < 12:
        drivers.append(f"Short tenure ({data.get('tenure')} months — high early churn risk)")
    if data.get("mpesa_usage_score", 10) < 4:
        drivers.append(f"Low M-Pesa usage score ({data.get('mpesa_usage_score')}/10)")
    if data.get("network_quality_score", 10) < 5:
        drivers.append(f"Poor network quality perception ({data.get('network_quality_score')}/10)")
    if data.get("competitor_exposure", 0) > 6:
        drivers.append(f"High competitor exposure ({data.get('competitor_exposure')}/10)")
    if not data.get("bonga_points_active"):
        drivers.append("Inactive in Bonga loyalty programme")
    if not data.get("safaricom_home"):
        drivers.append("Not on Safaricom Home (lower ecosystem lock-in)")
    if data.get("rural"):
        drivers.append("Rural location — network quality challenges")
    if data.get("monthly_charges", 0) > 5000:
        drivers.append(f"High monthly charges (KES {data.get('monthly_charges'):,.0f})")
    return (drivers or ["Insufficient signal — monitor closely"])[:n]


# Endpoints

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the API and model are running correctly."""
    return HealthResponse(
        status="ok" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        version="1.0.0",
    )


@app.get("/feature-importance", tags=["Model"])
def feature_importance():
    """Top churn drivers learned by the Random Forest model."""
    if MODEL is None:
        raise HTTPException(503, "Model not loaded.")

    if hasattr(MODEL, "feature_importances_") and FEATURE_NAMES:
        ranked = sorted(
            zip(FEATURE_NAMES, MODEL.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:15]
        return {
            "top_features": [
                {
                    "rank": i + 1,
                    "feature": feat,
                    "importance": round(float(imp), 4),
                    "label": FEATURE_IMPORTANCE_LABELS.get(feat, feat.replace("_", " ").title()),
                }
                for i, (feat, imp) in enumerate(ranked)
            ]
        }

    return {
        "note": "Using estimated importances from model documentation",
        "top_features": [
            {"rank": 1, "feature": "contract_Month-to-month", "importance": 0.18, "label": "Month-to-month contract"},
            {"rank": 2, "feature": "tenure",                  "importance": 0.15, "label": "Customer tenure"},
            {"rank": 3, "feature": "monthly_charges",         "importance": 0.13, "label": "Monthly charges"},
            {"rank": 4, "feature": "mpesa_usage_score",       "importance": 0.11, "label": "M-Pesa engagement"},
            {"rank": 5, "feature": "network_quality_score",   "importance": 0.10, "label": "Network quality perception"},
            {"rank": 6, "feature": "competitor_exposure",     "importance": 0.08, "label": "Competitor exposure"},
            {"rank": 7, "feature": "bonga_points_active",     "importance": 0.06, "label": "Bonga loyalty active"},
            {"rank": 8, "feature": "safaricom_home",          "importance": 0.05, "label": "Safaricom Home subscriber"},
        ],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(
    customer: CustomerFeatures,
    customer_id: Optional[str] = Query(default=None, description="Optional customer ID for tracking"),
):
    """
    Predict churn probability for a single customer.
    Returns churn probability, risk level (Low/Medium/High), top churn drivers,
    and tailored retention recommendations.
    """
    if MODEL is None:
        raise HTTPException(503, "Model not loaded. Ensure models/churn_model.pkl exists.")

    data = customer.model_dump()
    try:
        df = preprocess(data)
        prob = float(MODEL.predict_proba(df)[0][1])
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(500, f"Prediction error: {e}")

    risk_level, risk_score = get_risk_level(prob)

    return PredictionResponse(
        customer_id=customer_id,
        churn_probability=round(prob, 4),
        risk_level=risk_level,
        risk_score=risk_score,
        top_churn_drivers=get_top_drivers(data),
        recommended_actions=RETENTION_ACTIONS[risk_level.lower()],
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(..., description="CSV file with customer data")):
    """
    Score multiple customers at once by uploading a CSV.
    The CSV columns should match the CustomerFeatures schema.
    Returns a summary and per-row predictions.
    """
    if MODEL is None:
        raise HTTPException(503, "Model not loaded.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted.")

    contents = await file.read()
    try:
        df_input = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    results, errors = [], []

    for idx, row in df_input.iterrows():
        try:
            data = row.to_dict()
            for b in ["paperless_billing", "phone_service", "senior_citizen",
                      "partner", "dependents", "bonga_points_active", "safaricom_home", "rural"]:
                if b in data:
                    data[b] = bool(data[b])

            df_row = preprocess(data)
            prob = float(MODEL.predict_proba(df_row)[0][1])
            risk_level, risk_score = get_risk_level(prob)

            results.append({
                "row": int(idx),
                "customer_id": str(data.get("customer_id", idx)),
                "churn_probability": round(prob, 4),
                "risk_level": risk_level,
                "risk_score": risk_score,
                "top_churn_drivers": get_top_drivers(data),
                "recommended_actions": RETENTION_ACTIONS[risk_level.lower()],
            })
        except Exception as e:
            errors.append({"row": int(idx), "error": str(e)})

    return {
        "summary": {
            "total_customers": len(df_input),
            "scored": len(results),
            "errors": len(errors),
            "high_risk_count": sum(1 for r in results if r["risk_level"] == "High"),
            "medium_risk_count": sum(1 for r in results if r["risk_level"] == "Medium"),
            "low_risk_count": sum(1 for r in results if r["risk_level"] == "Low"),
            "avg_churn_probability": round(
                sum(r["churn_probability"] for r in results) / len(results), 4
            ) if results else 0,
        },
        "predictions": results,
        "parse_errors": errors,
    }
