from src.data.generator import generate_entities
from src.features.build_features import build_customer_features
from src.risk.engine import score
from src.recovery.actions import apply_actions

def test_feature_and_risk_pipeline():
    frames=generate_entities(100,seed=7); f=build_customer_features(frames)
    assert len(f)==100
    assert {"revenue_loss","payment_failure","churn"}.issubset(f.columns)
    result=apply_actions(score(f,[.2]*100,[.3]*100,[.1]*100))
    assert result.revenue_at_risk.ge(0).all()
    assert result.estimated_recoverable_revenue.le(result.revenue_at_risk).all()
    assert result.recommended_action.notna().all()
