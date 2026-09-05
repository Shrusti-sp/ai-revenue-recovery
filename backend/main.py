"""FastAPI service for PulseRecover, a merchant payment-recovery copilot."""
from pathlib import Path
import json
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.generator import generate_entities
from src.features.build_features import build_customer_features
from src.models.train import train_models, load_models, predict
from src.risk.engine import score
from src.recovery.actions import apply_actions
from src.database.repository import write_database, query, database_path
from src.llm.service import recommendation, analytics_answer
from src.explainability.explain import top_factors

app = FastAPI(title="PulseRecover API", version="1.0.0", description="Merchant payment recovery decision intelligence.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
_data: pd.DataFrame | None = None

def data() -> pd.DataFrame:
    global _data
    if _data is not None: return _data
    model_dir=ROOT/"models"
    if database_path(ROOT).exists() and (model_dir/"metrics.json").exists():
        _data=query(ROOT,"SELECT * FROM predictions")
    else:
        frames=generate_entities(); features=build_customer_features(frames); models,_=train_models(features,model_dir)
        loss,payment,churn=predict(models,features); _data=apply_actions(score(features,loss,payment,churn)); _data["churn_probability"]=churn
        write_database(frames,_data.drop(columns=["renewal_date"],errors="ignore"),ROOT)
    return _data

def money_metrics(d: pd.DataFrame):
    risk=float(d.revenue_at_risk.sum()); recover=float(d.estimated_recoverable_revenue.sum())
    return {"merchant_gmv":float(d.monthly_revenue.sum()),"revenue_at_risk":risk,"recoverable_revenue":recover,"recovery_rate":recover/risk if risk else 0,"priority_cases":int((d.priority_score>=d.priority_score.quantile(.85)).sum()),"critical_cases":int((d.risk_level.astype(str)=="CRITICAL").sum())}

@app.get("/health")
def health(): return {"status":"ok","service":"PulseRecover API"}

@app.get("/api/dashboard")
def dashboard():
    d=data(); return {"metrics":money_metrics(d),"risk_by_region":d.groupby("customer_region").revenue_at_risk.sum().reset_index().to_dict("records"),"recovery_by_action":d.groupby("recommended_action").estimated_recoverable_revenue.sum().reset_index().to_dict("records")}

@app.get("/api/cases")
def cases(region: str | None=None, risk_level: str | None=None, limit: int=Query(50,ge=1,le=200)):
    d=data()
    if region: d=d[d.customer_region==region]
    if risk_level: d=d[d.risk_level.astype(str)==risk_level.upper()]
    cols=["customer_id","customer_segment","customer_region","industry","risk_probability","risk_level","revenue_at_risk","recovery_probability","estimated_recoverable_revenue","priority_score","recommended_action","payment_failure_rate"]
    return d.nlargest(limit,"priority_score")[cols].to_dict("records")

@app.get("/api/customers/{customer_id}")
def customer(customer_id: str):
    selected=data()[data().customer_id==customer_id]
    if selected.empty: raise HTTPException(404,"Merchant customer not found")
    r=selected.iloc[0]; payload={k:(v.item() if hasattr(v,"item") else str(v) if pd.isna(v) else v) for k,v in r.to_dict().items()}
    payload["contributors"]=[{"feature":a,"description":b} for a,b in top_factors(r)]
    payload["copilot_recommendation"]=recommendation(r)
    return payload

@app.get("/api/analytics")
def analytics(question: str):
    result=analytics_answer(question,data())
    return {"result":result.to_dict("records") if isinstance(result,pd.DataFrame) else result,"source":"SQLite analytics and model outputs"}

@app.get("/api/model-metrics")
def model_metrics():
    path=ROOT/"models"/"metrics.json"
    if not path.exists(): data()
    return json.loads(path.read_text())
