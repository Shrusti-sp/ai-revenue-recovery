import pandas as pd
def recommend(row) -> str:
    if row.risk_level == "CRITICAL" and row.outstanding_revenue > 20000: return "Human Escalation"
    if row.failed_payments >= 2: return "Alternative Payment Method"
    if row.churn_probability >= .60: return "Retention Offer"
    if row.overdue_invoices > 0 and row.number_of_reminders >= 2: return "Customer Support Follow-up"
    if row.overdue_invoices > 0: return "Payment Reminder"
    return "Payment Retry"
def apply_actions(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy(); df["recommended_action"]=[recommend(r) for r in df.itertuples()]; return df
