import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from typing import List
from collections import defaultdict

def ridge_regression(X, y, cols, f_dropped: str):
    with open('models/regression/ridge_regression/ridge_regression.json', 'r') as fp:
        alpha_dict = json.load(fp)
    fp.close()
    #alpha_dict = defaultdict(dict)

    alphas = [0.0001, 0.001, 0.01, 0.1]

    y = y_pd.to_numpy()
    X = np.hstack([
        np.vstack(X_pd[c].values)
        for c in embedding_cols
    ])
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for a in alphas:
        clf = Ridge(alpha=a)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)

        alpha_dict[str((a, f_dropped))] = {
            'r2':r2,
            'test_mse': test_mse
        }
    with open('models/regression/ridge_regression/ridge_regression.json', 'w') as f:
        json.dump(alpha_dict, f, indent=2)
    f.close()
    pass

if __name__ == '__main__':
    table = pq.read_table('data/cleaned/data.parquet')
    df = table.to_pandas()
    df = df.drop(columns=['App ID', 'PUID', 'Enrolled (Binary)'])

    y_pd = df['Admitted (Binary)']
    X_pd = df.drop(columns=['Admitted (Binary)'])
    X_pd = X_pd.fillna(0)
    feature_cols = X_pd.columns.tolist()

    embedding_cols = ['Job ' + str(i) + ' Title Enc' for i in range(1,7)]
    embedding_cols.extend(['Job ' + str(i) + ' Description (embed)' for i in range(1,7)])
    embedding_cols.extend(['Job ' + str(i) + ' Organization (embed)' for i in range(1,7)])

    #ridge_regression(X_pd, y_pd, embedding_cols, "None")

    for f in feature_cols:
        new_embeddings = [c for c in embedding_cols if c!=f]
        X_pd_new = X_pd.drop(columns=[f])
        ridge_regression(X_pd_new, y_pd, new_embeddings, f)