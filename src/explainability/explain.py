"""Local, model-agnostic-style contribution summary based on feature percentile deviations.
SHAP is used when installed in the dashboard environment; explanations are contributors, not causes.
"""
import pandas as pd
def top_factors(row: pd.Series) -> list[tuple[str,str]]:
    factors=[]
    checks=[("failed_payments",lambda v:v>1,"Repeated failed payments"),("overdue_invoices",lambda v:v>0,"Overdue invoices"),("revenue_change_percentage",lambda v:v<-10,"Revenue decline"),("engagement_score",lambda v:v<45,"Low engagement"),("average_payment_delay_days",lambda v:v>14,"Long payment delays"),("complaint_count",lambda v:v>2,"Elevated complaints")]
    for col,test,label in checks:
        if col in row and test(row[col]): factors.append((col,label))
    return factors[:5] or [("profile","No single adverse contributor dominates")]

def shap_contributors(pipeline, row: pd.Series, feature_columns: list[str]) -> list[tuple[str, float]]:
    """Return local SHAP contributors when SHAP is available; never interpret them as causal."""
    try:
        import shap
        transformed = pipeline.named_steps["preprocess"].transform(pd.DataFrame([row])[feature_columns])
        explanation = shap.Explainer(pipeline.named_steps["model"], transformed)(transformed)
        values = explanation.values[0]
        return sorted(zip(feature_columns, values), key=lambda x: abs(x[1]), reverse=True)[:5]
    except Exception:
        return [(name, 0.0) for name, _ in top_factors(row)]
