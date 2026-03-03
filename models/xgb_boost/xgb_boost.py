import os
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score, mean_squared_error
import pyarrow.parquet as pq
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import torch.nn as nn
from collections import defaultdict
import json

def xg_boost(X, y, cols, f_dropped: str):
    # with open('models/xgb_boost/xgb_boost.json', 'r') as fp:
    #     xgb_dict = json.load(fp)
    # fp.close()
    #xgb_dict = defaultdict(dict)

    y = y_pd.to_numpy()
    X = np.hstack([
        np.vstack(X_pd[c].values)
        for c in cols
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    selector = SelectKBest(f_classif, k=50)
    X_train_k = selector.fit_transform(X_train, y_train)
    X_test_k  = selector.transform(X_test)

    clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc"
    )
    clf.fit(X_train_k, y_train, eval_set=[(X_test_k, y_test)], verbose=False)
    #y_pred = clf.predict_proba(X_test_k)[:, 1]
    y_pred = clf.predict(X_test_k)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    # xgb_dict[f_dropped] = {
    #     'accuracy': acc,
    #     'precision': precision,
    #     'recall': recall,
    #     'f1': f1,
    #     'test_mse': test_mse,
    #     'roc': auc
    # }

    print(50)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"ROC: {auc:.4f}")

    # with open('models/xgb_boost/xgb_boost.json', 'w') as f:
    #     json.dump(xgb_dict, f, indent=2)
    # f.close()
    pass


if __name__ == '__main__':
    table = pq.read_table('data/cleaned/data.parquet')
    df = table.to_pandas()
    df = df.drop(columns=['App ID', 'PUID', 'Enrolled (Binary)', 'Decision History', 'Continent'])
    

    y_pd = df['Admitted (Binary)']
    X_pd = df.drop(columns=['Admitted (Binary)'])
    X_pd = X_pd.fillna(0)
    X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("School 6")])
    X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("School 5")])
    X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("Country")])
    X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("School 5 Country_")])
    X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("School 6_Country")])
    X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("Job 5_Country")])
    X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("Job 6_Country")])
    X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("School_")])
    


    embedding_cols = ['Job ' + str(i) + ' Title Enc' for i in range(1,7)]
    embedding_cols.extend(['Job ' + str(i) + ' Description (embed)' for i in range(1,7)])
    embedding_cols.extend(['Job ' + str(i) + ' Organization (embed)' for i in range(1,7)])

    #X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("Job")])
    xg_boost(X_pd, y_pd, embedding_cols, "Country")