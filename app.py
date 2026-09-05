from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data.generator import generate_entities
from src.data.quality import quality_report
from src.features.build_features import build_customer_features
from src.models.train import train_models, load_models, predict
from src.risk.engine import score
from src.recovery.actions import apply_actions
from src.database.repository import write_database, query, database_path
from src.explainability.explain import top_factors, shap_contributors
from src.llm.service import recommendation, analytics_answer

ROOT=Path(__file__).parent
st.set_page_config(page_title="Revenue Recovery AI",page_icon="₹",layout="wide")
@st.cache_data(show_spinner="Preparing analytics data and models…")
def bootstrap():
    if database_path(ROOT).exists() and (ROOT/"models"/"metrics.json").exists():
        return query(ROOT,"SELECT * FROM predictions"), json.loads((ROOT/"models"/"metrics.json").read_text())
    frames=generate_entities(); features=build_customer_features(frames); models,metrics=train_models(features,ROOT/"models")
    lp,pp,cp=predict(models,features); scored=apply_actions(score(features,lp,pp,cp)); scored["churn_probability"]=cp
    keep=[c for c in scored.columns if c not in ["renewal_date"]]; write_database(frames,scored[keep],ROOT)
    quality_report(frames).to_csv(ROOT/"data"/"processed"/"data_quality_report.csv",index=False)
    return scored,metrics
df,metrics=bootstrap()
st.title("AI-Powered Revenue Recovery & Decision Intelligence")
page=st.sidebar.radio("Navigate",["Executive Overview","AI Risk Monitor","Revenue Recovery","Customer 360","AI Decision Center","AI Analytics Assistant","Model Evaluation"])
def money(x): return f"₹{x:,.0f}"
def filtered(data):
    with st.sidebar:
        regions=st.multiselect("Region",sorted(data.customer_region.unique()),default=sorted(data.customer_region.unique())); levels=st.multiselect("Risk level",["LOW","MEDIUM","HIGH","CRITICAL"],default=["LOW","MEDIUM","HIGH","CRITICAL"])
    return data[data.customer_region.isin(regions)&data.risk_level.astype(str).isin(levels)]
if page=="Executive Overview":
    d=filtered(df); cols=st.columns(6)
    values=[d.monthly_revenue.sum(),d.revenue_at_risk.sum(),d.estimated_recoverable_revenue.sum(),0,(d.estimated_recoverable_revenue.sum()/d.revenue_at_risk.sum() if d.revenue_at_risk.sum() else 0),len(d[d.risk_level.isin(["HIGH","CRITICAL"])])]
    for c,label,v in zip(cols,["Total Revenue","Revenue at Risk","Recoverable Revenue","Recovered Revenue","Recovery Rate","High-risk customers"],values): c.metric(label,money(v) if label not in ["Recovery Rate","High-risk customers"] else (f"{v:.1%}" if label=="Recovery Rate" else f"{v:,}"))
    a,b=st.columns(2); a.plotly_chart(px.bar(d.groupby("customer_segment",as_index=False).revenue_at_risk.sum(),x="customer_segment",y="revenue_at_risk",title="Revenue at risk by segment"),use_container_width=True); b.plotly_chart(px.histogram(d,x="risk_probability",color="risk_level",nbins=30,title="Revenue risk distribution"),use_container_width=True)
    st.plotly_chart(px.funnel(x=[len(d),len(d[d.outstanding_revenue>0]),len(d[d.risk_level.isin(["HIGH","CRITICAL"])])],y=["Customers","Open balance","Priority cases"],title="Recovery funnel"),use_container_width=True)
elif page=="AI Risk Monitor":
    d=filtered(df); st.caption("Scores are ML probabilities plus transparent revenue/recovery calculations. Sort the table to triage cases.")
    cols=["customer_id","customer_region","industry","customer_segment","risk_probability","risk_level","revenue_at_risk","recovery_probability","estimated_recoverable_revenue","priority_score","recommended_action"]
    st.dataframe(d[cols].sort_values("priority_score",ascending=False),use_container_width=True,hide_index=True)
    st.plotly_chart(px.density_heatmap(d,x="customer_region",y="risk_level",z="revenue_at_risk",histfunc="sum",title="Regional risk heatmap"),use_container_width=True)
elif page=="Revenue Recovery":
    d=filtered(df); c=st.columns(5)
    for col,label,val in zip(c,["Total Revenue","Revenue Lost (modeled)","Revenue at Risk","Potentially Recoverable","Recovered Revenue"],[d.monthly_revenue.sum(),d.loc[d.revenue_loss==1,"monthly_revenue"].sum(),d.revenue_at_risk.sum(),d.estimated_recoverable_revenue.sum(),0]): col.metric(label,money(val))
    a,b=st.columns(2); a.plotly_chart(px.bar(d.groupby("recommended_action",as_index=False).estimated_recoverable_revenue.sum().sort_values("estimated_recoverable_revenue"),x="estimated_recoverable_revenue",y="recommended_action",orientation="h",title="Recovery opportunity by action"),use_container_width=True); b.plotly_chart(px.bar(d.groupby("customer_region",as_index=False).revenue_at_risk.sum(),x="customer_region",y="revenue_at_risk",title="Regional exposure"),use_container_width=True)
elif page=="Customer 360":
    cid=st.selectbox("Customer",df.sort_values("priority_score",ascending=False).customer_id); r=df[df.customer_id==cid].iloc[0]
    c=st.columns(5)
    for col,label,val in zip(c,["Lifetime Value","Risk","Revenue at Risk","Recoverable","Churn Probability"],[money(r.customer_lifetime_value),f"{r.risk_probability:.1%}",money(r.revenue_at_risk),money(r.estimated_recoverable_revenue),f"{r.churn_probability:.1%}"]): col.metric(label,val)
    st.subheader(f"{r.risk_level} risk - {r.recommended_action}"); st.write("Model contributors (not causal proof):", "; ".join(x[1] for x in top_factors(r)))
    if st.checkbox("Calculate local SHAP explanation", key=cid):
        try:
            model=load_models(ROOT/"models")["revenue_loss"]
            shap_items=shap_contributors(model,r,__import__("src.features.build_features",fromlist=["FEATURE_COLUMNS"]).FEATURE_COLUMNS)
            st.dataframe(pd.DataFrame(shap_items,columns=["Feature","SHAP contribution"]).round(4),hide_index=True)
        except Exception as exc: st.warning(f"SHAP explanation unavailable: {exc}")
    st.dataframe(pd.DataFrame({"Metric":["Invoices","Overdue invoices","Failed payments","Payment failure rate","Engagement","Revenue change"],"Value":[r.total_invoices,r.overdue_invoices,r.failed_payments,f"{r.payment_failure_rate:.1%}",r.engagement_score,f"{r.revenue_change_percentage:.1f}%"]}),hide_index=True)
    tone=st.selectbox("Communication tone",["Professional","Friendly","Concise","Urgent"]); rec=recommendation(r,tone); st.info(rec["risk_explanation"]); st.write("**Next step:**",rec["next_step"]); st.text_area("Draft (review before sending)",rec["draft"],height=150)
elif page=="AI Decision Center":
    d=filtered(df).nlargest(25,"priority_score"); st.dataframe(d[["customer_id","risk_level","revenue_at_risk","estimated_recoverable_revenue","recommended_action","priority_score"]],hide_index=True,use_container_width=True)
    pick=st.selectbox("Open case",d.customer_id); r=df[df.customer_id==pick].iloc[0]; st.json(recommendation(r))
elif page=="AI Analytics Assistant":
    st.caption("Answers are constrained to the analytics dataset; it does not invent financial values.")
    q=st.text_input("Ask a question",placeholder="Which region has the largest revenue leakage?")
    if q:
        ans=analytics_answer(q,df)
        st.dataframe(ans,use_container_width=True,hide_index=True) if isinstance(ans,pd.DataFrame) else st.success(ans)
elif page=="Model Evaluation":
    st.caption("Holdout metrics and stratified cross-validation; recall and PR-AUC are emphasized for risk detection.")
    st.dataframe(pd.DataFrame(metrics).T,use_container_width=True)
    st.write("Data-quality report:"); st.dataframe(pd.read_csv(ROOT/"data"/"processed"/"data_quality_report.csv"),hide_index=True)
