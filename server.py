"""
endpoints:
    GET  /health                 — model metadata and status
    GET  /features/schema        — valid categorical values per column
    POST /predict                — classify one or more incidents
    POST /predict/explain        — classify + waterfall SHAP plot
    POST /predict/explain/text   — classify + waterfall + text reasoning for the prediction
"""

import base64
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional
import joblib
import matplotlib
matplotlib.use("Agg") # to render to memory, not a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from aggregation import INCIDENT_KEY, aggregate_incidents

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# artifact paths
MODEL_PATH = "artifacts/best_model.pth"
SCALER_PATH = "artifacts/scaler.pkl"
CONFIG_PATH = "artifacts/model_config.json"
CAT_MAPPINGS_PATH = "artifacts/cat_mappings.json"
ORG_STATS_PATH = "artifacts/org_stats.parquet"
FEAT_GROUPS_PATH = "artifacts/feature_groups.json"
SHAP_BG_PATH = "artifacts/shap_background.npy"

# build the model architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        self.shourtcut = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shourtcut(x)

class ResidualModel(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.input = nn.Linear(in_dim, 256)
        self.blocks = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128)
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.head(x)

class SoftmaxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)

# create global state container
class State:
    model = None
    scaler = None
    config = None
    feature_cols = None
    class_names = None
    cat_mappings = None
    feature_groups = None
    explainer = None
    device = None
    groq_client = None
    org_stats = None
    expected_values = None

state = State()

# define the lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Loading artifacts...")

    # load config
    # contains: input_dim, feature_cols, num_classes, classes, best_val_f1 test_f1
    with open(CONFIG_PATH) as f:
        state.config = json.load(f)
    state.feature_cols = state.config["feature_cols"]
    state.class_names  = state.config["classes"]

    # load cat_mappings
    # containes: dict of column : valid values
    with open(CAT_MAPPINGS_PATH) as f:
        state.cat_mappings = json.load(f)

    # load org_stats
    # containes historical data about organizations
    state.org_stats = pd.read_parquet(ORG_STATS_PATH)

    # load feature_groups
    # containes: dict of original col : aggregated feature names
    with open(FEAT_GROUPS_PATH) as f:
        state.feature_groups = json.load(f)

    # load data scaler
    state.scaler = joblib.load(SCALER_PATH)

    logger.info("Loading the ai model...")

    # detect and choose device
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the model and load weights
    state.model = ResidualModel(state.config["input_dim"], state.config["num_classes"]).to(state.device)
    state.model.load_state_dict(
        torch.load(MODEL_PATH, map_location=state.device, weights_only=True)
    )
    state.model.eval()

    logger.info("preparing the explainer...")

    background_data = np.load(SHAP_BG_PATH).astype(np.float32)
    background_data_tensor = torch.tensor(background_data).to(state.device)
    softmax_model = SoftmaxWrapper(state.model).to(state.device)
    state.explainer = shap.GradientExplainer(softmax_model, background_data_tensor)

    with torch.no_grad():
        bg_logits = state.model(background_data_tensor)
        bg_probs  = torch.softmax(bg_logits, dim=1).cpu().numpy()
        state.expected_values = bg_probs.mean(axis=0)

    logger.info("explainer ready")

    # prepare llm api
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        state.groq_client = Groq(api_key=groq_key)
        logger.info("Groq client ready.")
    else:
        logger.warning("Groq client failed — /predict/explain/text will be unavailable.")

    logger.info(f"Server ready | device={state.device} | features={len(state.feature_cols)} | classes={state.class_names}")

    yield

    # SHUTDOWN

    logger.info("Server shutting down.")

# itiate FastAPI app
app = FastAPI(
    title="GUIDE Incident Classifier",
    description=(
        "Residual network trained on the Microsoft GUIDE dataset."
        "Classifies cybersecurity incidents as BenignPositive, FalsePositive, or TruePositive from raw evidence rows."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# create request schema using Pydantic
class EvidenceRow(BaseModel):
    IncidentId : int
    OrgId : int
    Timestamp : Optional[str] = None
    DetectorId : Optional[int] = None
    AlertTitle : Optional[int] = None
    Category : Optional[str] = None
    MitreTechniques : Optional[str] = None
    EntityType : Optional[str] = None
    EvidenceRole : Optional[str] = None
    DeviceId : Optional[int] = None
    Sha256 : Optional[int] = None
    IpAddress : Optional[int] = None
    Url : Optional[int] = None
    AccountSid : Optional[int] = None
    AccountUpn : Optional[int] = None
    AccountObjectId : Optional[int] = None
    AccountName : Optional[int] = None
    DeviceName : Optional[int] = None
    NetworkMessageId : Optional[int] = None
    EmailClusterId : Optional[float] = None
    RegistryKey : Optional[int] = None
    RegistryValueName : Optional[int] = None
    RegistryValueData : Optional[int] = None
    ApplicationId : Optional[int] = None
    ApplicationName : Optional[int] = None
    OAuthApplicationId : Optional[int] = None
    ThreatFamily : Optional[str] = None
    FileName : Optional[int] = None
    FolderPath : Optional[int] = None
    ResourceIdName : Optional[int] = None
    ResourceType : Optional[str] = None
    Roles : Optional[str] = None
    OSFamily : Optional[int] = None
    OSVersion : Optional[int] = None
    AntispamDirection : Optional[str] = None
    SuspicionLevel : Optional[str] = None
    LastVerdict : Optional[str] = None
    CountryCode : Optional[int] = None
    State : Optional[int] = None
    City : Optional[int] = None
    AlertId : Optional[int] = None
    ActionGrouped : Optional[str] = None
    ActionGranular : Optional[str] = None

# function to preprocess the data for prediction.
# returns (one row per incident):
#   agg_df : the data aggregated (ready for shap explainer)
#   X_scaled : the data aggregated and scaled (ready for the model)
def rows_to_features(rows: list[EvidenceRow]):

    # converts a Pydantic object to a plain Python dict
    df = pd.DataFrame([r.model_dump() for r in rows])

    agg_df = aggregate_incidents(df, include_target=False, mappings=state.cat_mappings, org_stats=state.org_stats)

    # align to exact feature order the model was trained on
    for col in state.feature_cols:
        if col not in agg_df.columns:
            agg_df[col] = 0.0
    X = agg_df[state.feature_cols].values.astype(np.float32)

    X_scaled = state.scaler.transform(X)

    return agg_df, X_scaled

# function to predict the incidents grade using the sent data
# returns:
#   predictions : list of predicted class names
#   probabilities : list of dicts {class_name: probability}
def run_model(X_scaled):

    tensor = torch.tensor(X_scaled, dtype=torch.float32).to(state.device)

    with torch.no_grad():
        logits = state.model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

    predictions   = list(probs.argmax(axis=1))
    probabilities = [
        {name: round(float(p), 4) for name, p in zip(state.class_names, row)}
        for row in probs
    ]
    return predictions, probabilities

# Compute SHAP values for sent data.
# returns (n_samples, n_features, n_classes).
def compute_shap(X_scaled):

    tensor = torch.tensor(X_scaled, dtype=torch.float32).to(state.device)
    shap_vals = state.explainer.shap_values(tensor)

    # shap output version independance
    if isinstance(shap_vals, list):
        # turn list of [(n_samples, n_features), (n_samples, n_features),...] stack to (n_samples, n_features, n_classes)
        shap_array = np.stack(shap_vals, axis=-1)
    else:
        shap_array = shap_vals

    # detect and fix axis order if needed
    if shap_array.ndim == 3 and shap_array.shape[0] == len(state.feature_cols) and shap_array.shape[2] != len(state.feature_cols):
        shap_array = shap_array.transpose(2, 0, 1)

    return shap_array

# sum shap values to shows original column names instead of aggregated feature names.
# returns: a list of dicts sorted by absolute SHAP value descending:[{"feature": "OrgId", "shap": 0.67}, ...]
def deaggregate_shap(shap_vec: np.ndarray):
    feature_cols = state.feature_cols
    result = []

    for original_col, agg_features in state.feature_groups.items():
        indices = [i for i, f in enumerate(feature_cols) if f in agg_features]
        if not indices:
            continue

        group_shap = float(shap_vec[indices].sum())
        result.append({"feature": original_col, "shap" : round(group_shap, 4)})

    result.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return result

# plot waterfall plot from de-aggregated SHAP values
# returns: the plot as a base64-encoded PNG string.
def make_waterfall_png(deagg: list[dict], base_value: float, predicted_class: str):
    TOP_N = 8
    top = list(reversed(deagg[:TOP_N]))
    n = len(top) # in case the sent deagg is less than 8
    features  = [d["feature"] for d in top]
    shap_vals = [d["shap"]    for d in top]

    # compute the start point of each bar
    lefts   = []
    running = base_value
    for val in shap_vals:
        lefts.append(running)
        running += val
    final_value = running

    colors = ["red" if s > 0 else "blue" for s in shap_vals]

    all_x = lefts + [l + v for l, v in zip(lefts, shap_vals)] + [base_value, final_value]

    xmin, xmax = min(all_x), max(all_x)
    xrange = xmax - xmin if xmax != xmin else 1e-6

    pad = xrange * 0.15

    

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.7 + 2)))

    ax.set_xlim(xmin - pad, xmax + pad)

    y_pos = list(range(n))
    ax.barh(
        y_pos,
        shap_vals,
        left=lefts,
        color=colors,
        height=0.6
    )

    # baseline vertical line
    ax.axvline(base_value, color="grey", linewidth=1.2, linestyle="--")
    ax.text(
        base_value,
        -1,
        f"E[f(x)] = {base_value:.3f}",
        ha="center", va="top",
        fontsize=9, color="black"
    )

    # final output vertical line
    ax.axvline(final_value, color="grey", linewidth=1.2, linestyle="--")
    ax.text(
        final_value,
        n - 0.05,
        f"f(x) = {final_value:.3f}",
        ha="center", va="bottom",
        fontsize=9, color="black"
    )
    
    # shap value of each bar
    for i, (val, left) in enumerate(zip(shap_vals, lefts)):
        bar_end = left + val
        ax.text(
            bar_end + (xrange * 0.02 if val >= 0 else -(xrange * 0.02)),
            i,
            f"{val:+.3f}",
            ha="left" if val >= 0 else "right",
            va="center",
            fontsize=8,
            color="black",
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("Model output value")
    ax.set_title(f"Prediction: {predicted_class}", fontsize=13, fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def call_groq(predicted_class : str, probabilities : dict, top10_shap : list[dict], incident_id : int):
    if state.groq_client is None:
        return "LLM unavailable — GROQ_API_KEY not set."
    
    shap_lines = "\n".join([f"{d['feature']} = {d['value']:.4f}  (SHAP: {d['shap']:+.4f})"for d in top10_shap])
    conf = probabilities[predicted_class] * 100

    prompt = f"""
    You are a cybersecurity analyst assistant helping a Security Operations Center (SOC).
    A machine learning model has analysed a cybersecurity incident and made a prediction.

    FEATURE CONTEXT (important — read before explaining):
    - org_rate_TruePositive/FalsePositive/BenignPositive: historical grade rates for this organisation
    - org_incident_count: total past incidents from this organisation
    - evidence_count: number of raw evidence items in this incident
    - *_nunique: count of unique values for that entity type (e.g. IpAddress_nunique = number of unique IPs)
    - Category_*, EntityType_*, MitreTechniques_*: counts of evidence rows matching that category/technique
    - duration_seconds: time from first to last evidence item
    - hour_of_day / day_of_week: when the incident started
    - LastVerdict_*, SuspicionLevel_*, ThreatFamily_*: verdict/threat signals from detectors
    - If a feature value is 0, that entity type or category was NOT present in this incident.
    - A positive SHAP value for a zero-value feature means its ABSENCE pushed toward this prediction.

    PREDICTION:
    Incident ID : {incident_id}
    Predicted   : {predicted_class}
    Confidence  : {conf:.1f}%
    All classes : BenignPositive={probabilities.get('BenignPositive',0)*100:.1f}%  FalsePositive={probabilities.get('FalsePositive',0)*100:.1f}%  TruePositive={probabilities.get('TruePositive',0)*100:.1f}%

    TOP 10 FEATURES DRIVING THIS PREDICTION (feature value + SHAP contribution):
    {shap_lines}

    Write a concise 3-5 sentence explanation for a SOC analyst. Explain:
    1. What the prediction means in plain English
    2. The 4-5 strongest reasons the model made this prediction (from the SHAP values)
    3. Any caveats the analyst should be aware of

    Be specific — use the actual feature values. Do not use generic filler sentences.
    """

    response = state.groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.3,   # low temperature = more factual, less creative
    )
    return response.choices[0].message.content


# endpoints

# server status and model metadata
@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "device"      : str(state.device),
        "classes"     : state.class_names,
        "num_features": len(state.feature_cols),
        "best_val_f1" : state.config.get("best_val_f1"),
        "test_f1"     : state.config.get("test_f1"),
        "llm_available": state.groq_client is not None,
    }

# categorical columns valid values
@app.get("/features/schema")
def features_schema():
    return {
        "categorical_fields": state.cat_mappings,
        "note": (
            "Fields not listed here accept any numeric value."
            "Values outside the listed options will be treated as 'Other'."
        )
    }

# classify one or more incidents from raw evidence rows.
@app.post("/predict")
def predict(rows: list[EvidenceRow]):
    if not rows:
        raise HTTPException(status_code=400, detail="No evidence rows provided.")
    
    try:
        agg_df, X_scaled = rows_to_features(rows)
        predictions, probabilities = run_model(X_scaled)

        incident_ids = agg_df[INCIDENT_KEY].tolist()

        return {
            "results": [
                {
                    "incident_id" : iid,
                    "prediction"  : state.class_names[pred],
                    "probabilities": probs,
                }
                for iid, pred, probs in zip(incident_ids, predictions, probabilities)
            ]
        }
    except Exception as e:
        logger.error(f"/predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# classify incidents and return a waterfall SHAP plot for each.
@app.post("/predict/explain")
def predict_explain(rows: list[EvidenceRow]):
    if not rows:
        raise HTTPException(status_code=400, detail="No evidence rows provided.")

    try:
        agg_df, X_scaled = rows_to_features(rows)
        predictions, probabilities = run_model(X_scaled)

        incident_ids = agg_df[INCIDENT_KEY].tolist()

        shap_array = compute_shap(X_scaled)
        
        results = []
        for i, (iid, pred, probs) in enumerate(zip(incident_ids, predictions, probabilities)):
            shap_vec   = shap_array[i, :, pred]
            base_value = float(state.expected_values[pred])
            deagg = deaggregate_shap(shap_vec)
            predicted_class_name = state.class_names[pred]

            png_b64 = make_waterfall_png(deagg, base_value, predicted_class_name)

            results.append({
                "incident_id" : iid,
                "prediction" : predicted_class_name,
                "probabilities" : probs,
                "waterfall_png_b64" : png_b64
            })

        return {"results": results}

    except Exception as e:
        logger.error(f"/predict/explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Classify incidents, return waterfall plot and LLM text explaning the reason behing the prediction.
@app.post("/predict/explain/text")
def predict_explain_text(rows: list[EvidenceRow]):
    if not rows:
        raise HTTPException(status_code=400, detail="No evidence rows provided.")

    if state.groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM unavailable — set GROQ_API_KEY environment variable.",
        )

    try:
        agg_df, X_scaled = rows_to_features(rows)
        predictions, probabilities = run_model(X_scaled)
        
        incident_ids = agg_df[INCIDENT_KEY].tolist()

        shap_array = compute_shap(X_scaled)

        results = []
        for i, (iid, pred, probs) in enumerate(zip(incident_ids, predictions, probabilities)):
            shap_vec = shap_array[i, :, pred]
            base_value = float(state.expected_values[pred])
            deagg    = deaggregate_shap(shap_vec)
            predicted_class_name = state.class_names[pred]

            png_b64  = make_waterfall_png(deagg, base_value, predicted_class_name)

            agg_row  = agg_df.iloc[i]
            top10_idx = np.argsort(np.abs(shap_vec))[-10:][::-1]
            top10 = [
                {
                    "feature": state.feature_cols[j],
                    "value"  : float(agg_row.get(state.feature_cols[j], 0)),
                    "shap"   : float(shap_vec[j]),
                }
                for j in top10_idx
            ]

            # call Groq LLM
            narrative = call_groq(predicted_class_name, probs, top10, iid)

            results.append({
                "incident_id"       : iid,
                "prediction"        : predicted_class_name,
                "probabilities"     : probs,
                "waterfall_png_b64" : png_b64,
                "explanation"       : narrative
            })

        return {"results": results}

    except Exception as e:
        logger.error(f"/predict/explain/text error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
