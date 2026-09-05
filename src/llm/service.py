"""LLM enhancement with safe deterministic fallback. ML probabilities are never altered."""
import os
from src.explainability.explain import top_factors
def recommendation(row, tone="Professional"):
    factors=", ".join(label for _,label in top_factors(row))
    base={"risk_explanation":f"The model flags this account as {row.risk_level} risk. Contributors: {factors}.","recommended_action":row.recommended_action,"reasoning":"The action balances payment friction, outstanding value, engagement, and recovery likelihood.","expected_objective":f"Recover up to ₹{row.estimated_recoverable_revenue:,.0f} of modeled at-risk revenue.","next_step":"Review account context and approve the draft; do not send automatically."}
    greeting="Hello" if tone!="Urgent" else "Action required"
    base["draft"]=f"{greeting} {row.customer_id},\n\nWe’re reaching out regarding your account. Please review the outstanding balance and let us know if we can help complete payment.\n\nRegards,\nRevenue Recovery Team"
    return base
def analytics_answer(question, df):
    q=question.lower()
    if "highest" in q and "risk" in q: return df.nlargest(10,"revenue_at_risk")[["customer_id","revenue_at_risk","risk_level"]]
    if "region" in q and ("leakage" in q or "risk" in q): return df.groupby("customer_region",as_index=False).revenue_at_risk.sum().sort_values("revenue_at_risk",ascending=False)
    if "recover" in q: return f"Modeled recoverable revenue: ₹{df.estimated_recoverable_revenue.sum():,.0f}."
    if "action" in q: return df.groupby("recommended_action",as_index=False).estimated_recoverable_revenue.sum().sort_values("estimated_recoverable_revenue",ascending=False)
    return "Try asking about highest revenue at risk, regional leakage, recovery opportunities, or actions."

def optional_llm_narrative(row, structured: dict) -> str | None:
    """Optional API enhancement. It receives fixed numbers but can never change them in storage."""
    key=os.getenv("OPENAI_API_KEY")
    if not key: return None
    from openai import OpenAI
    prompt=("Write a concise recovery explanation using only these supplied facts. Do not create or alter "
            "probabilities, currency amounts, or actions. Return narrative prose only. Facts: " + str(structured))
    return OpenAI(api_key=key).responses.create(model=os.getenv("OPENAI_MODEL","gpt-4.1-mini"),input=prompt).output_text
