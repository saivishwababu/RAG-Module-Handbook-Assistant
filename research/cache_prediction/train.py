"""Train a small cache reuse classifier on a temporally isolated synthetic trace."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score
from research.cache_prediction.dataset import generate_zipfian_trace
from research.cache_prediction.features import extract_causal_features_and_labels

def train_cache_predictor(num_requests=1000,seed=42):
    trace=generate_zipfian_trace(num_requests=num_requests,seed=seed)
    X,y,_=extract_causal_features_and_labels(trace)
    split=max(1,int(len(y)*.7)); train_end=max(1,split-30)  # purge labels whose future window crosses the holdout boundary
    model=RandomForestClassifier(n_estimators=40,random_state=seed,class_weight="balanced")
    model.fit(X[:train_end],y[:train_end]); pred=model.predict(X[split:]) if split<len(y) else model.predict(X[:train_end]); actual=y[split:] if split<len(y) else y[:train_end]
    prob=model.predict_proba(X[split:] if split<len(y) else X[:train_end])[:,1]
    metrics={"precision":float(precision_score(actual,pred,zero_division=0)),"recall":float(recall_score(actual,pred,zero_division=0)),"f1":float(f1_score(actual,pred,zero_division=0)),"roc_auc":float(roc_auc_score(actual,prob)) if len(set(actual))>1 else 0.5}
    return model,metrics
