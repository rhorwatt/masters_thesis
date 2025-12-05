import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pyarrow.parquet as pq
from typing import List, Tuple
import seaborn as sns
import csv
import json

def create_heatmap(df: pd.DataFrame) -> None:
    feature_cols = df.select_dtypes(include=["int64", "float64", "float32", "int32"])
    plt.figure(figsize=(40, 40))
    plt.imshow(feature_cols.corr(), cmap='viridis')
    plt.colorbar()
    plt.savefig('cleaned_var_heatmap.pdf')
    plt.close()
    pass

def kernel_density_plot(df: pd.DataFrame) -> None:
    """
    Plotting variance in each feature.
    Used this resource: https://www.geeksforgeeks.org/data-analysis/exploratory-data-analysis-in-python/
    """
    plots_per_page = 6
    rows, cols = 3, 2
    pdf_filename = "cleaned_feature_plots.pdf"
    feature_cols = df.select_dtypes(include=["int64", "float64", "float32", "int32"]).columns
    x_data = np.arange(1276)

    with PdfPages(pdf_filename) as pdf:
        for i, c in enumerate(feature_cols):
            plot_num = (i % plots_per_page) + 1

            if (plot_num == 1):
                fig = plt.figure(figsize=(12, 16))
                plt.tight_layout(pad=3.0)

            ax = fig.add_subplot(rows, cols, plot_num)
            ax.plot(x_data, df[feature_cols[i]])
            ax.set_title(f'Feature {i+1, c}')
            ax.set_xlabel('Student Num')
            ax.set_ylabel('{c}')

            if (plot_num == plots_per_page) or (i == (len(feature_cols) - 1)):
                pdf.savefig(fig)
                plt.close(fig)
    pass

def linear_regression(X_pd: pd.DataFrame, y_pd: pd.DataFrame, embedding_cols: List[str]) -> Tuple:
    """
    This function performs linear regression on cleaned admissions data with 80/20 train test split.
    It returns the coefficients and intercept from the training fit, R**2 score, and test MSE as a tuple.
    """
    
    y = y_pd.to_numpy()
    X = np.hstack([
        np.vstack(X_pd[c].values)
        for c in embedding_cols
    ])
    y = y.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reg = LinearRegression().fit(X_train, y_train)
    coef = reg.coef_
    intercept = reg.intercept_
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    return coef, intercept, r2, test_mse

def leave_one_out_cv() -> None:
    """
    This function fits a linear regression model to the entire cleaned admissions dataset, as well as computing the model
    when leaving out each feature.
    """
    table = pq.read_table('data/cleaned/data.parquet')
    df = table.to_pandas()
    df = df.drop(columns=['App ID', 'PUID', 'Enrolled (Binary)'])

    y_pd = df['Admitted (Binary)']
    X_pd = df.drop(columns=['Admitted (Binary)'])
    X_pd = X_pd.fillna(0)
    features = X_pd.columns.to_list()

    embedding_cols = ['Job ' + str(i) + ' Title Enc' for i in range(1,7)]
    embedding_cols.extend(['Job ' + str(i) + ' Description (embed)' for i in range(1,7)])
    embedding_cols.extend(['Job ' + str(i) + ' Organization (embed)' for i in range(1,7)])

    with open("models/linear_regression.json", "a") as fp:
        coef, intercept, r2, test_mse = linear_regression(X_pd, y_pd, embedding_cols)

        record = {
            "feature_dropped": "None",
            "r2": float(r2),
            "test_mse": float(test_mse),
            "coefficients": coef.tolist(),
            "intercept": intercept.tolist()
        }

        fp.write(json.dumps(record) + "\n")

        for f in features:
            X_pd_dropped = X_pd.drop(columns=[f])
            new_embedding_cols = [c for c in embedding_cols if c!=f]
            coef, intercept, r2, test_mse = linear_regression(X_pd_dropped, y_pd, new_embedding_cols)

            record = {
                "feature_dropped": f,
                "r2": float(r2),
                "test_mse": float(test_mse),
                "coefficients": coef.tolist(),
                "intercept": intercept.tolist()
            }
            fp.write(json.dumps(record) + "\n")
    fp.close()
    pass

def print_important_features(file_path: str) -> None:
    with open(file_path, 'r') as f:
        for line in f:
            x = json.loads(line)
            if x["r2"] > -0.8854497766703286:
                print(x["feature_dropped"])
    f.close()
    pass

def lr_on_important_features():
    table = pq.read_table('data/cleaned/data.parquet')
    df = table.to_pandas()
    df = df.drop(columns=['App ID', 'PUID', 'Enrolled (Binary)'])

    y_pd = df['Admitted (Binary)']
    X_pd = df.drop(columns=['Admitted (Binary)'])
    X_pd = X_pd.fillna(0)

    drop_cols = ["Job 1 Title Enc", "Job 6 Title Enc", "Job 5 Description (embed)", "Job 6 Description (embed)", "Job 3 Organization (embed)", 
                 "Job 4 Organization (embed)", "Job 6 Organization (embed)"]
    embedding_cols = ["Job " + str(i) + ' Title Enc' for i in range(2,6)]
    embedding_cols.extend(['Job ' + str(i) + ' Description (embed)' for i in range(1,5)])
    embedding_cols.extend(['Job ' + str(i) + ' Organization (embed)' for i in range(1,5) if ((i != 4) and (i != 3))])

    X_pd = X_pd.drop(columns=drop_cols)
    coef, intercept, r2, test_mse = linear_regression(X_pd, y_pd, embedding_cols)
    print(coef, intercept, r2, test_mse)
    # r2 of -0.6675936077563671 with these features:
    # embedding_cols = ["Job 2 Title Enc", "Job 3 Title Enc", "Job 4 Title Enc", "Job 5 Title Enc", "Job 1 Description (embed)", "Job 2 Description (embed)", 
                      #"Job 3 Description (embed)", "Job 4 Description (embed)", "Job 1 Organization (embed)", "Job 2 Organization (embed)", "Job 5 Organization (embed)"]

    # r2 of -0.7158007371395885 with these features dropped:
    # drop_cols = ["Job 1 Title Enc", "Job 6 Title Enc", "Job 5 Description (embed)", "Job 6 Description (embed)", "Job 3 Organization (embed)", 
    #              "Job 4 Organization (embed)", "Job 6 Organization (embed)"]
    
if __name__ == '__main__':
    #print_important_features("models/linear_regression.json")
    lr_on_important_features()