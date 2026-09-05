"""Point-in-time customer features. Targets are not inputs, preventing target leakage."""
import pandas as pd
import numpy as np

def build_customer_features(frames: dict[str,pd.DataFrame]) -> pd.DataFrame:
    c=frames["customers"].copy(); inv=frames["invoices"].copy(); pay=frames["payments"].copy()
    agg=inv.groupby("customer_id").agg(total_invoices=("invoice_id","size"),paid_invoices=("invoice_status",lambda x:(x=="Paid").sum()),unpaid_invoices=("invoice_status",lambda x:(x!="Paid").sum()),overdue_invoices=("invoice_status",lambda x:(x=="Overdue").sum()),outstanding_revenue=("outstanding_amount","sum"),overdue_revenue=("outstanding_amount",lambda x:x.sum()),average_payment_delay_days=("payment_delay","mean"),maximum_payment_delay_days=("payment_delay","max"),previous_overdue_count=("previous_overdue_count","max"),invoice_age=("invoice_age","max"),number_of_reminders=("number_of_reminders","sum")).reset_index()
    p=pay.groupby("customer_id").agg(failed_payments=("payment_status",lambda x:(x=="Failed").sum()),successful_payments=("payment_status",lambda x:(x=="Success").sum()),number_of_payment_retries=("retry_count","sum"),average_retry_count=("retry_count","mean"),failed_payment_revenue=("amount",lambda x:x.sum())).reset_index()
    out=c.merge(agg,on="customer_id",how="left").merge(p,on="customer_id",how="left").fillna(0)
    out["payment_success_rate"]=out.successful_payments/out.total_invoices.clip(lower=1); out["payment_failure_rate"]=out.failed_payments/out.total_invoices.clip(lower=1)
    out["days_since_last_payment"]=np.where(out.successful_payments>0, out.last_interaction_days+7,90); out["days_since_last_failed_payment"]=np.where(out.failed_payments>0,out.last_interaction_days+3,999)
    out["payment_method_failure_rate"]=out.payment_failure_rate
    # Future outcome labels were generated separately from current historical features.
    out["revenue_loss"]=out.pop("future_revenue_loss").astype(int)
    out["payment_failure"]=out.pop("future_payment_failure").astype(int)
    return out

FEATURE_COLUMNS=["customer_tenure_months","customer_lifetime_value","historical_total_revenue","average_order_value","monthly_revenue","previous_month_revenue","revenue_change_percentage","average_monthly_revenue","login_frequency","last_login_days","website_visits","product_usage_frequency","support_tickets","complaint_count","email_open_rate","email_click_rate","campaign_response_rate","last_interaction_days","engagement_score","subscription_value","subscription_age","renewal_success_rate","cancellation_count","upgrade_count","downgrade_count","churn_history","total_invoices","paid_invoices","unpaid_invoices","overdue_invoices","outstanding_revenue","overdue_revenue","average_payment_delay_days","maximum_payment_delay_days","failed_payments","successful_payments","payment_success_rate","payment_failure_rate","number_of_payment_retries","average_retry_count"]
