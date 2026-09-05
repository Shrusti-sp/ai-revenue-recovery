from __future__ import annotations
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from src.features.build_features import FEATURE_COLUMNS

def _pipeline():
    pre=ColumnTransformer([("numeric",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),FEATURE_COLUMNS)])
    return Pipeline([("preprocess",pre),("model",HistGradientBoostingClassifier(max_iter=180,learning_rate=.07,max_leaf_nodes=18,random_state=42))])
def train_models(data: pd.DataFrame, models_dir: Path):
    models_dir.mkdir(parents=True,exist_ok=True); metrics={}; fitted={}
    for target,label in [("revenue_loss","revenue_loss"),("payment_failure","payment_failure"),("churn","churn")]:
        X=data[FEATURE_COLUMNS]; y=data[target].astype(int); xtr,xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
        pipe=_pipeline(); pipe.fit(xtr,ytr); proba=pipe.predict_proba(xte)[:,1]; pred=(proba>=.5).astype(int)
        cv=cross_val_score(_pipeline(),X,y,cv=StratifiedKFold(4,shuffle=True,random_state=42),scoring="average_precision")
        metrics[label]={"accuracy":round(accuracy_score(yte,pred),3),"precision":round(precision_score(yte,pred,zero_division=0),3),"recall":round(recall_score(yte,pred,zero_division=0),3),"f1":round(f1_score(yte,pred,zero_division=0),3),"roc_auc":round(roc_auc_score(yte,proba),3),"pr_auc":round(average_precision_score(yte,proba),3),"cv_pr_auc_mean":round(cv.mean(),3),"confusion_matrix":confusion_matrix(yte,pred).tolist()}
        joblib.dump(pipe,models_dir/f"{label}.joblib"); fitted[label]=pipe
    (models_dir/"metrics.json").write_text(json.dumps(metrics,indent=2)); return fitted,metrics
def load_models(models_dir: Path): return {name:joblib.load(models_dir/f"{name}.joblib") for name in ["revenue_loss","payment_failure","churn"]}
def predict(models, data):
    return tuple(models[name].predict_proba(data[FEATURE_COLUMNS])[:,1] for name in ["revenue_loss","payment_failure","churn"])
