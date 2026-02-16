from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from typing import List
from collections import defaultdict

def random_regression(X_pd: pd.DataFrame, y_pd: pd.DataFrame, embedding_cols: List[str], f_dropped: str) -> None:
    with open('models/random_forest/random_forest_drop.json', 'r') as fp:
        random_dict = json.load(fp)
    fp.close()
    #random_dict = defaultdict(dict)

    y = y_pd.to_numpy()
    # X = np.hstack([
    #     np.vstack(X_pd[c].values)
    #     for c in embedding_cols
    # ])
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pd, y, test_size=0.2, random_state=42
    )

    # https://www.geeksforgeeks.org/machine-learning/performing-feature-selection-with-gridsearchcv-in-sklearn/

    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=8, random_state=42)
    clf.fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    random_dict[str((f_dropped))] = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_mse': test_mse,
        'roc': auc
    }
    
    with open('models/random_forest/random_forest_drop.json', 'w') as f:
        json.dump(random_dict, f, indent=2)
    f.close()
    pass

if __name__ == '__main__':
    table = pq.read_table('data/cleaned/data.parquet')
    df = table.to_pandas()
    df = df.drop(columns=['App ID', 'PUID', 'Enrolled (Binary)', 'Decision History', 'Continent'])

    y_pd = df['Admitted (Binary)']
    X_pd = df.drop(columns=['Admitted (Binary)'])
    X_pd = X_pd.fillna(0)
    feature_cols = X_pd.columns.tolist()

    embedding_cols = ['Job ' + str(i) + ' Title Enc' for i in range(1,7)]
    embedding_cols.extend(['Job ' + str(i) + ' Description (embed)' for i in range(1,7)])
    embedding_cols.extend(['Job ' + str(i) + ' Organization (embed)' for i in range(1,7)])

    #random_regression(X_pd, y_pd, embedding_cols, "None")

    #X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("School")])
    X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("Job")])
    # for f in feature_cols:
    #     new_embeddings = [c for c in embedding_cols if c!=f]
    #     X_pd_new = X_pd.drop(columns=[f])
    random_regression(X_pd, y_pd, embedding_cols, "Job")