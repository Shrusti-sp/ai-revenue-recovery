"""Correlated synthetic data for a revenue-recovery demonstration."""
from __future__ import annotations
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
AS_OF = pd.Timestamp("2026-09-01")

def _sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -25, 25)))

def generate_entities(n_customers: int = 5000, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate linked entities; latent health drives payment, churn, and recovery outcomes."""
    rng = np.random.default_rng(seed)
    ids = [f"C{i:05d}" for i in range(1, n_customers + 1)]
    segment = rng.choice(["SMB", "Mid-Market", "Enterprise"], n_customers, p=[.55,.32,.13])
    region = rng.choice(["North", "South", "East", "West"], n_customers)
    industry = rng.choice(["Retail", "SaaS", "Healthcare", "Manufacturing", "Finance"], n_customers)
    account = np.where(segment == "Enterprise", "Strategic", rng.choice(["Standard", "Premium"], n_customers, p=[.7,.3]))
    tenure = rng.integers(1, 84, n_customers)
    health = rng.normal(0, 1, n_customers) + np.where(segment == "Enterprise", .25, 0)
    engagement = np.clip(65 + 16*health + rng.normal(0, 10, n_customers), 1, 100)
    monthly = np.clip(rng.lognormal(8.6 + np.where(segment == "Enterprise", .9, np.where(segment == "Mid-Market", .35, 0)), .55), 1800, 250000)
    previous = monthly * np.clip(1 + rng.normal(.02 + .10*health, .16, n_customers), .35, 1.8)
    revenue_change = 100 * (monthly - previous) / previous
    failure_rate = _sigmoid(-1.5 - 1.05*health - .012*(engagement-50) - .012*revenue_change + rng.normal(0,.35,n_customers))
    churn_prob = _sigmoid(-1.25 - 1.1*health - .025*(engagement-50) - .018*revenue_change + rng.normal(0,.35,n_customers))
    ltv = monthly * (tenure + 8) * rng.uniform(.65, 1.3, n_customers)
    customers = pd.DataFrame({
        "customer_id": ids, "customer_segment": segment, "customer_region": region, "industry": industry,
        "acquisition_channel": rng.choice(["Organic", "Partner", "Paid", "Referral"], n_customers), "account_type": account,
        "customer_tenure_months": tenure, "customer_lifetime_value": ltv.round(2), "historical_total_revenue": (monthly*tenure*rng.uniform(.7,1.25,n_customers)).round(2),
        "average_order_value": (monthly/rng.uniform(1.5,5,n_customers)).round(2), "monthly_revenue": monthly.round(2), "previous_month_revenue": previous.round(2),
        "revenue_change_percentage": revenue_change.round(2), "average_monthly_revenue": (monthly*rng.uniform(.82,1.2,n_customers)).round(2),
        "revenue_last_30_days": monthly.round(2), "revenue_last_90_days": (monthly*3*rng.uniform(.8,1.15,n_customers)).round(2), "revenue_last_12_months": (monthly*12*rng.uniform(.75,1.2,n_customers)).round(2),
        "login_frequency": np.clip((10+health*4+rng.normal(0,3,n_customers)).round(),0,None), "last_login_days": np.clip((18-health*8+rng.normal(0,7,n_customers)).round(),0,90),
        "website_visits": np.clip((22+health*9+rng.normal(0,7,n_customers)).round(),0,None), "product_usage_frequency": np.clip((15+health*6+rng.normal(0,4,n_customers)).round(),0,None),
        "support_tickets": np.clip((3-health*1.2+rng.normal(0,1.5,n_customers)).round(),0,None), "complaint_count": np.clip((1-health*.65+rng.normal(0,.7,n_customers)).round(),0,None),
        "email_open_rate": np.clip(.48+health*.12+rng.normal(0,.07,n_customers),.02,.98), "email_click_rate": np.clip(.15+health*.06+rng.normal(0,.04,n_customers),.01,.65),
        "campaign_response_rate": np.clip(.25+health*.1+rng.normal(0,.07,n_customers),.01,.9), "last_interaction_days": np.clip((22-health*8+rng.normal(0,8,n_customers)).round(),0,120),
        "engagement_score": engagement.round(1), "subscription_status": np.where(rng.random(n_customers)<churn_prob*.45,"At Risk","Active"), "subscription_type": rng.choice(["Monthly","Annual","Enterprise"],n_customers,p=[.45,.4,.15]),
        "subscription_value": (monthly*rng.uniform(10,13,n_customers)).round(2), "subscription_age": tenure, "renewal_date": AS_OF + pd.to_timedelta(rng.integers(1,180,n_customers),unit="D"),
        "previous_renewals": np.maximum(0,(tenure/12).astype(int)-1), "renewal_success_rate": np.clip(.75+health*.1+rng.normal(0,.08,n_customers),.05,.99),
        "cancellation_count": (rng.random(n_customers)<churn_prob*.5).astype(int), "upgrade_count": (rng.random(n_customers)<_sigmoid(health)).astype(int), "downgrade_count": (rng.random(n_customers)<_sigmoid(-health-.5)).astype(int), "churn_history": (rng.random(n_customers)<churn_prob*.3).astype(int),
        # Forward-looking labels are sampled from latent propensity; they are never used as inputs.
        "future_revenue_loss": (rng.random(n_customers) < _sigmoid(-1.0-1.0*health-.012*(engagement-50)-.018*revenue_change)).astype(int),
        "future_payment_failure": (rng.random(n_customers)<failure_rate).astype(int),
        "churn": (rng.random(n_customers)<churn_prob).astype(int), "latent_failure_rate": failure_rate
    })
    customers["days_to_renewal"]=(pd.to_datetime(customers.renewal_date)-AS_OF).dt.days
    inv_rows=[]; pay_rows=[]; txn_rows=[]; interaction_rows=[]; action_rows=[]
    for r in customers.itertuples(index=False):
        count=int(rng.integers(2,6)); previous_overdue=0
        for j in range(count):
            amount=float(np.clip(r.monthly_revenue*rng.uniform(.35,1.25),200,300000)); date=AS_OF-pd.Timedelta(days=int(rng.integers(5,240))); due=date+pd.Timedelta(days=30)
            failed=rng.random()<r.latent_failure_rate; delay=max(0,int(rng.normal(5+22*r.latent_failure_rate,8)))
            paid=(not failed) or rng.random()<.35; payment_date=(due+pd.Timedelta(days=delay)) if paid else pd.NaT
            overdue=int(max(0,(AS_OF-due).days if not paid else delay)); status="Paid" if paid else ("Overdue" if overdue>0 else "Open")
            outstanding=0 if paid else amount; retries=int(failed*rng.integers(1,4)); reminders=int((overdue>0)*rng.integers(1,5))
            iid=f"I{r.customer_id[1:]}-{j+1}"; inv_rows.append([iid,r.customer_id,amount,date,due,payment_date,status,overdue,delay,previous_overdue, int((AS_OF-date).days),reminders, "Responded" if rng.random()<r.email_open_rate else "No response", int(rng.random()<.15),outstanding]); previous_overdue += overdue>0
            if paid: txn_rows.append([f"T{iid[1:]}",r.customer_id,date,amount,"Completed"])
            pay_rows.append([f"P{iid[1:]}",iid,r.customer_id, "Card" if rng.random()<.65 else "Bank Transfer", "Success" if paid else "Failed", payment_date, retries, amount])
        interaction_rows.append([f"X{r.customer_id[1:]}",r.customer_id,AS_OF-pd.Timedelta(days=int(r.last_interaction_days)),"Support" if r.support_tickets else "Product",float(r.engagement_score)])
    invoices=pd.DataFrame(inv_rows,columns=["invoice_id","customer_id","invoice_amount","invoice_date","due_date","payment_date","invoice_status","days_overdue","payment_delay","previous_overdue_count","invoice_age","number_of_reminders","reminder_response","discount_applied","outstanding_amount"])
    payments=pd.DataFrame(pay_rows,columns=["payment_id","invoice_id","customer_id","payment_method","payment_status","payment_date","retry_count","amount"])
    transactions=pd.DataFrame(txn_rows,columns=["transaction_id","customer_id","transaction_date","amount","status"])
    interactions=pd.DataFrame(interaction_rows,columns=["interaction_id","customer_id","interaction_date","interaction_type","engagement_score"])
    subs=customers[["customer_id","subscription_status","subscription_type","subscription_value","subscription_age","renewal_date","previous_renewals","renewal_success_rate","cancellation_count","upgrade_count","downgrade_count","churn_history"]].copy(); subs.insert(0,"subscription_id",[f"S{i:05d}" for i in range(1,n_customers+1)])
    return {"customers":customers.drop(columns="latent_failure_rate"),"invoices":invoices,"payments":payments,"transactions":transactions,"customer_interactions":interactions,"subscriptions":subs,"recovery_actions":pd.DataFrame(columns=["action_id","customer_id","action_type","action_date","action_result","recovered_amount"])}
