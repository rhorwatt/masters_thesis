import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from typing import List
from collections import defaultdict

def knn(X, y, cols, f_dropped: str):
    with open('models/knn/knn_drop.json', 'r') as fp:
        knn_dict = json.load(fp)
    fp.close()
    #knn_dict = defaultdict(dict)

    #k_values = range(1, 21)

    y = y_pd.to_numpy()
    X = np.hstack([
        np.vstack(X_pd[c].values)
        for c in cols
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=12)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    knn_dict[str((12, f_dropped))] = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_mse': test_mse,
        'roc': auc
    }

    with open('models/knn/knn_drop.json', 'w') as f:
        json.dump(knn_dict, f, indent=2)
    f.close()
    pass

if __name__ == '__main__':
    embedding_cols = ['Job ' + str(i) + ' Title Enc' for i in range(1,7)]
    embedding_cols.extend(['Job ' + str(i) + ' Description (embed)' for i in range(1,7)])
    embedding_cols.extend(['Job ' + str(i) + ' Organization (embed)' for i in range(1,7)])

    table = pq.read_table('grouped_data.parquet')
    df = table.to_pandas()
    df = df.drop(columns=['App ID', 'PUID', 'Enrolled (Binary)'])
    df = df.fillna(0)

    table2 = pq.read_table('models/transformer/data.parquet')
    df2 = table2.to_pandas()
    df2 = df2.fillna(0)

    application_terms = {
            'Fall 2020'  : 1,
            'Fall 2021'  : 2,
            'Fall 2022'  : 3,
            'Fall 2023'  : 4,
            'Fall 2024'  : 5,
            'Summer 2024': 6
        }
    
    df['App Term'] = df2['App Term'].map(application_terms).astype(int)
    y_pd = df['Admitted (Binary)']
    X_pd = df.drop(columns=['Admitted (Binary)'])
    X_pd = X_pd.fillna(0)

    knn(X_pd, y_pd, embedding_cols, "None")