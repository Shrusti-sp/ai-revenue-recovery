import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS={"low":.25,"medium":.50,"high":.75}
def score(features: pd.DataFrame, loss_prob, payment_prob, churn_prob, thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    d=features.copy(); d["revenue_loss_probability"]=np.asarray(loss_prob); d["payment_failure_probability"]=np.asarray(payment_prob); d["churn_probability"]=np.asarray(churn_prob); d["risk_probability"]=(.50*d.revenue_loss_probability+.30*d.payment_failure_probability+.20*d.churn_probability).clip(0,1)
    d["probability_of_default"]=d.risk_probability
    d["revenue_at_risk"]=d.outstanding_revenue*d.probability_of_default
    d["recovery_probability"]=(.80-.55*d.payment_failure_rate+.004*d.engagement_score-.004*d.average_payment_delay_days+.10*d.renewal_success_rate).clip(.08,.92)
    d["estimated_recoverable_revenue"]=d.revenue_at_risk*d.recovery_probability
    raw=d.risk_probability*d.revenue_at_risk*d.recovery_probability*(.5+.5*(d.customer_lifetime_value/d.customer_lifetime_value.max()))*(1+np.minimum(d.invoice_age,90)/180)
    d["priority_score"]=(100*raw/raw.max()).fillna(0)
    d["risk_level"]=pd.cut(d.risk_probability,[-.01,thresholds["low"],thresholds["medium"],thresholds["high"],1],labels=["LOW","MEDIUM","HIGH","CRITICAL"])
    return d
