import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, auc
import joblib
import os

def load_data(data_dir='data/processed'):
    train = pd.read_csv(f'{data_dir}/train.csv')
    val = pd.read_csv(f'{data_dir}/val.csv')
    test = pd.read_csv(f'{data_dir}/test.csv')
    
    X_train, y_train = train.drop(columns=['flag']), train['flag']
    X_val, y_val = val.drop(columns=['flag']), val['flag']
    X_test, y_test = test.drop(columns=['flag']), test['flag']
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    print("Classification Report:\n", classification_report(y_val, y_pred))
    
    roc_auc = roc_auc_score(y_val, y_prob)
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recall, precision)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"F1-Score: {f1_score(y_val, y_pred):.4f}")
    return model

def save_model(model, filename='../models/baseline_rf.pkl'):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, filename)
    print(f"Модель сохранена в {filename}")