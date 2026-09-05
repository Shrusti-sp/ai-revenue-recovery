# PulseRecover — AI Payment Recovery & Merchant Revenue Copilot

A production-style internship project that identifies revenue at risk, predicts future payment failure, revenue loss, and churn, estimates recoverability, and turns results into transparent recovery actions.

## Architecture

Synthetic linked source data → validation and point-in-time feature engineering → SQLite analytics store → ML prediction layer → risk and recoverability engine → action rules and explanations → Streamlit dashboard.

## Included capabilities

- 5,000 correlated synthetic customers plus invoices, payments, transactions, subscriptions, interactions, and recovery-action schema.
- Customer-level point-in-time features, three ML models, stratified train/test splits, cross-validation, and persisted metrics.
- Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, configurable risk bands, transparent priority scoring, and recovery recommendations.
- Executive Overview, AI Risk Monitor, Revenue Recovery, Customer 360, AI Decision Center, analytics assistant, and model evaluation pages.

## Risk methodology

Combined risk = 0.50 × revenue-loss probability + 0.30 × payment-failure probability + 0.20 × churn probability. The weights are an explicit business policy to calibrate with stakeholders.

Revenue at risk = outstanding revenue × risk probability. Estimated recoverable revenue = revenue at risk × recovery probability. The normalized priority score combines risk, revenue at risk, recovery probability, account value, and invoice-age urgency; it is a ranking aid, not an automated collection decision.

## Data and leakage controls

The generator creates behavioral correlation through an unexposed latent account-health signal. Future revenue-loss and future payment-failure labels are sampled separately and are never feature columns. Models use historical/current payment and engagement information only. In production, enforce an as-of timestamp and a future outcome window. Never use post-event payment or recovery outcomes as predictors.

Data-quality reporting covers rows, duplicate rows, missing cells, and negative amount anomalies.

## LLM design

Numerical probabilities, risk bands, revenue figures, and actions originate in structured data and ML/rules. The guarded LLM service only enhances explanations, analytics presentation, and unsent communication drafts. It cannot override an ML score or invent financial values; a local fallback works without an API key.

## Run locally — Streamlit demo

    pip install -r requirements.txt
    streamlit run app.py

On its first launch, the app generates source data, trains models, writes `data/database/revenue_recovery.db`, and saves models/metrics. Later launches reuse those artifacts.

## Run locally — full-stack application

The API serves the same ML and SQLite decision layer to a separate merchant-operations frontend.

    pip install -r requirements.txt
    uvicorn backend.main:app --reload --port 8000

In a second terminal:

    cd frontend
    npm install
    npm run dev

Open the Vite URL shown in the terminal (normally `http://localhost:5173`). API documentation is available at `http://localhost:8000/docs`.

## Razorpay-style payment context

The product is designed as a merchant recovery copilot: it analyzes failed and delayed payments, ranks recovery opportunities by modeled value, recommends safe next steps, and keeps human approval in the loop before any communication. For a final demo, show payment-method fields such as UPI, cards, netbanking, wallet, recurring mandate, gateway response, and webhook event status flowing through the same ingestion interface. Do not use live credentials in the project; `.env.example` is the only configuration template.

## Layout

- `app.py` — Streamlit entry point for the fast demo.
- `backend/main.py` — FastAPI service and OpenAPI documentation.
- `frontend` — React/Vite merchant operations frontend.
- `src/data`, `src/features`, `src/models`, `src/risk`, `src/recovery`, `src/explainability`, `src/llm`, `src/database` — application layers.
- `data/database` and `data/processed` — generated database and quality report.
- `models` — persisted sklearn models and evaluation metrics.
- `tests` — smoke tests.

## Limitations and future work

Synthetic relationships do not replace production calibration. Recoverability is an estimate, not a promise. Validate fairness, monitor drift, implement access controls/auditing, integrate real CRM/payment systems, and validate actions through controlled experiments before operational automation.
