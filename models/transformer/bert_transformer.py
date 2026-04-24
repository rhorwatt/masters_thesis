import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
from transformers import AutoModelForSequenceClassification
from torch.utils.data import random_split
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer
from transformers import BertConfig, BertModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

scaler = StandardScaler()

class TabularBertTransformer(nn.Module):
    def __init__(self,
                 float_dim,
                 job_desc_dim,
                 job_org_dim,
                 job_title_dim,
                 d_model=768,    # must match BERT hidden size
                 num_layers=6):
        super().__init__()

        # Project each group → d_model
        self.float_proj     = nn.Linear(float_dim,     d_model)
        self.job_desc_proj  = nn.Linear(job_desc_dim,  d_model)
        self.job_org_proj   = nn.Linear(job_org_dim,   d_model)
        self.job_title_proj = nn.Linear(job_title_dim, d_model)

        # Use BERT's transformer encoder
        config = BertConfig(
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=12,   # must divide d_model evenly
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        bert = BertModel(config)
        self.encoder = bert.encoder

        self.classifier = nn.Linear(d_model, 1)

    def forward(self, floats, job_desc, job_org, job_title):
        # Project each group to d_model and treat as token
        t1 = self.float_proj(floats).unsqueeze(1)      # (batch, 1, d_model)
        t2 = self.job_desc_proj(job_desc).unsqueeze(1)
        t3 = self.job_org_proj(job_org).unsqueeze(1)
        t4 = self.job_title_proj(job_title).unsqueeze(1)

        # (batch, num_tokens, d_model)
        x = torch.cat([t1, t2, t3, t4], dim=1)

        # BertEncoder expects attention mask of shape (batch, 1, 1, seq_len)
        attention_mask = torch.ones(x.shape[0], 1, 1, x.shape[1]).to(x.device)

        x = self.encoder(x, attention_mask=attention_mask).last_hidden_state
        x = x.mean(dim=1)               # pool over tokens
        return self.classifier(x)       # (batch, 1)

class AdmissionsDataset(Dataset):
    def __init__(self, floats, job_desc, job_org, job_title, y):
        self.floats     = floats
        self.job_desc   = job_desc
        self.job_org    = job_org
        self.job_title  = job_title
        self.y          = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.floats[idx], self.job_desc[idx], self.job_org[idx], self.job_title[idx], self.y[idx]
    

def set_seed(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seeds = [0, 1, 2, 3, 4]
    results = []

    def run_seeds(df_input, df2, seeds):
        for seed in seeds:
            set_seed(seed)
            df = df_input.copy()

            application_terms = {
                    'Fall 2020'  : 1,
                    'Fall 2021'  : 2,
                    'Fall 2022'  : 3,
                    'Fall 2023'  : 4,
                    'Fall 2024'  : 5,
                    'Summer 2024': 6
                }
            
            df['App Term'] = df2['App Term'].map(application_terms).astype(int)


            job_desc_cols = ["Job " + str(i) + " Description (embed)" for i in range(1,7)]
            job_org_cols = ["Job " + str(i) + " Organization (embed)" for i in range(1,7)]
            job_title_cols = ["Job " + str(i) + " Title Enc" for i in range(1,7)]
            tensor_cols = job_desc_cols + job_org_cols + job_title_cols

            def make_tensor(frame, cols):
                return torch.stack([
                    torch.cat([x if isinstance(x, torch.Tensor) else torch.tensor(x)
                               for x in row])
                    for _, row in frame[cols].iterrows()
                ])

            test = df[df['App Term'] == 5]
            y_test = torch.tensor(test['Admitted (Binary)'].values)
            test = test.drop(columns=['Admitted (Binary)'])

            train = df[df['App Term'] != 5]
            y_train = torch.tensor(train['Admitted (Binary)'].values)
            train = train.drop(columns=['Admitted (Binary)'])
            
            job_desc_tensor_test = make_tensor(test, job_desc_cols)
            job_desc_tensor_train = make_tensor(train, job_desc_cols)
            job_org_tensor_test = make_tensor(test, job_org_cols)
            job_org_tensor_train = make_tensor(train, job_org_cols)
            job_title_tensor_test = make_tensor(test, job_title_cols)
            job_title_tensor_train = make_tensor(train, job_title_cols)
            
            float_cols = list(set(train.columns) - set(tensor_cols))
            df = df[float_cols].astype(np.float64)

            float_tensor_test = torch.tensor(scaler.fit_transform(test[float_cols].values), dtype=torch.float32)
            float_tensor_train = torch.tensor(scaler.fit_transform(train[float_cols].values), dtype=torch.float32)

            X_test = torch.cat([float_tensor_test, job_desc_tensor_test, job_org_tensor_test, job_title_tensor_test], dim=1)
            X_train = torch.cat([float_tensor_train, job_desc_tensor_train, job_org_tensor_train, job_title_tensor_train], dim=1)

            # Instantiate
            model = TabularBertTransformer(
                float_dim     = float_tensor_train.shape[1],
                job_desc_dim  = job_desc_tensor_train.shape[1],
                job_org_dim   = job_org_tensor_train.shape[1],
                job_title_dim = job_title_tensor_train.shape[1],
                d_model=768,
                num_layers=6
            )

            train_set = AdmissionsDataset(float_tensor_train, job_desc_tensor_train, job_org_tensor_train, job_title_tensor_train, y_train)
            test_set = AdmissionsDataset(float_tensor_test, job_desc_tensor_test, job_org_tensor_test, job_title_tensor_test, y_test)
            train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            criterion = torch.nn.BCEWithLogitsLoss()
            model.to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) # FIXME: learning rate
            model.train()

            for floats, job_desc, job_org, job_title, y_batch in train_loader:
                floats    = floats.float().to(device)
                job_desc  = job_desc.float().to(device)
                job_org   = job_org.float().to(device)
                job_title = job_title.float().to(device)
                y_batch   = y_batch.float().to(device)

                optimizer.zero_grad()
                outputs = model(floats, job_desc, job_org, job_title).squeeze(1)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for floats, job_desc, job_org, job_title, y_batch in test_loader:
                    floats    = floats.float().to(device)
                    job_desc  = job_desc.float().to(device)
                    job_org   = job_org.float().to(device)
                    job_title = job_title.float().to(device)
                    y_batch   = y_batch.float().to(device)
                    outputs = model(floats, job_desc, job_org, job_title).squeeze()
                    probs = torch.sigmoid(outputs.squeeze())  # prob of positive class
                    
                    all_preds.append(probs.cpu().numpy())
                    all_labels.append(y_batch.numpy())

            all_preds  = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            all_preds_binary = (all_preds > 0.5).astype(int)
            auc       = roc_auc_score(all_labels, all_preds)
            accuracy  = accuracy_score(all_labels, all_preds_binary)
            f1        = f1_score(all_labels, all_preds_binary)
            precision = precision_score(all_labels, all_preds_binary)
            recall    = recall_score(all_labels, all_preds_binary)
            cm        = confusion_matrix(all_labels, all_preds_binary)

            # Filter to admitted students only
            admitted_mask = all_labels == 0  # admitted students in test set

            admitted_preds   = all_preds[admitted_mask]


            # Load raw data for dropout labels
            df_raw = pd.read_excel('data/cleaned/processed_admissions.xlsx')

            # --- Get dropout labels aligned to test and train splits ---
            train_raw = df_raw[df_raw['App Term'] != 'Fall 2024']
            dropout_labels_train = train_raw['is_dropout'].astype(int).values

            # --- Masks for admitted students ---
            admitted_mask_train = y_train.numpy() == 1                     # from training data

            all_train_preds = np.array(all_preds)

            # --- Subset to admitted only ---

            train_admitted_preds   = all_train_preds[admitted_mask_train]
            train_admitted_dropout = dropout_labels_train[admitted_mask_train]

            # --- Plot: 2 rows (test / train), 2 cols (scatter / histogram) ---
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            dropped_patch = mpatches.Patch(color='red',   label='Dropped Out')
            stayed_patch  = mpatches.Patch(color='green', label='Did Not Drop Out')

            for row, (preds, dropout, split_label) in enumerate([
                (train_admitted_preds, train_admitted_dropout, "Train — All Other Semesters"),
            ]):
                sort_idx = np.argsort(preds)
                colors   = ['red' if d == 1 else 'green' for d in dropout[sort_idx]]

                # Scatter
                axes[row, 0].scatter(range(len(preds)), preds[sort_idx],
                                    c=colors, alpha=0.5, s=15)
                axes[row, 0].axhline(0.5, color='black', linestyle='--')
                axes[row, 0].set_xlabel("Admitted Students (sorted by predicted prob)")
                axes[row, 0].set_ylabel("Predicted Admission Probability")
                axes[row, 0].set_title(f"Admission Prob vs Dropout — {split_label}")
                axes[row, 0].legend(handles=[dropped_patch, stayed_patch])

                # Histogram
                axes[row, 1].hist(preds[dropout == 0], bins=20, alpha=0.6, color='green', label='Did Not Drop Out')
                axes[row, 1].hist(preds[dropout == 1], bins=20, alpha=0.6, color='red',   label='Dropped Out')
                axes[row, 1].axvline(0.5, color='black', linestyle='--')
                axes[row, 1].set_xlabel("Predicted Admission Probability")
                axes[row, 1].set_ylabel("Count")
                axes[row, 1].set_title(f"Probability Distribution by Dropout — {split_label}")
                axes[row, 1].legend()

            plt.tight_layout()
            plt.savefig("admitted_vs_dropout_train_and_test.png", dpi=150)
            plt.show()

            print(f"\n--- Train (All Other Semesters) ---")
            print(f"  Admitted: {admitted_mask_train.sum()}, Dropped: {train_admitted_dropout.sum()}, Retained: {(train_admitted_dropout==0).sum()}")

            # After splitting test from df
            df = pd.read_excel('data/cleaned/processed_admissions.xlsx')
            test = df[df['App Term'] == 'Fall 2024']
            test_index = test.index  # integer positions 0..1275

            # Pull dropout labels aligned to the test set rows
            dropout_labels = df.loc[test_index, 'is_dropout'].astype(int).values
            #dropout_labels = torch.tensor(test['is_dropout'].astype(int).values)  # your actual column name
            admitted_dropout = dropout_labels[admitted_mask]  # dropout status for admitted only

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Scatter
            sort_idx = np.argsort(admitted_preds)
            colors = ['red' if d == 1 else 'green' for d in admitted_dropout[sort_idx]]
            axes[0].scatter(range(len(admitted_preds)), admitted_preds[sort_idx],
                            c=colors, alpha=0.5, s=15)
            axes[0].axhline(0.5, color='black', linestyle='--')
            axes[0].set_xlabel("Admitted Students (sorted by predicted prob)")
            axes[0].set_ylabel("Predicted Admission Probability")
            axes[0].set_title("Admission Probability vs Dropout (Admitted Students Only)")
            dropped_patch  = mpatches.Patch(color='red',   label='Dropped Out')
            stayed_patch   = mpatches.Patch(color='green', label='Did Not Drop Out')
            axes[0].legend(handles=[dropped_patch, stayed_patch])

            # Histogram
            axes[1].hist(admitted_preds[admitted_dropout == 0], bins=20, alpha=0.6, color='green', label='Did Not Drop Out')
            axes[1].hist(admitted_preds[admitted_dropout == 1], bins=20, alpha=0.6, color='red',   label='Dropped Out')
            axes[1].axvline(0.5, color='black', linestyle='--')
            axes[1].set_xlabel("Predicted Admission Probability")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Probability Distribution by Dropout (Admitted Only)")
            axes[1].legend()

            plt.tight_layout()
            plt.savefig("admitted_only_vs_dropout.png", dpi=150)
            plt.show()

            print(f"Admitted students: {admitted_mask.sum()}")
            print(f"  Dropped out:     {admitted_dropout.sum()}")
            print(f"  Retained:        {(admitted_dropout == 0).sum()}")

            all_seeds_probs  = []
            all_seeds_labels = []

            # Inside run_seeds, after computing all_preds / all_labels:
            all_seeds_probs.append(all_preds)
            all_seeds_labels.append(all_labels)

            # After the loop, average probabilities across seeds:
            mean_probs  = np.mean(all_seeds_probs, axis=0)
            mean_labels = all_seeds_labels[0]  # labels are the same across seeds

            print(f"AUC:       {auc:.4f}")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"F1:        {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")

            results.append({"auc": auc, "accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall})

        df = pd.DataFrame(results)
        print("\n=== Summary ===")
        print(df.to_string(index=False))
        print(f"\nROC AUC:   mean={df.auc.mean():.4f}, std={df.auc.std():.4f}")
        print(f"Accuracy: mean={df.accuracy.mean():.4f}, std={df.accuracy.std():.4f}")
        print(f"F1: mean={df.f1.mean():.4f}, std={df.f1.std():.4f}")
        print(f"Precision: mean={df.precision.mean():.4f}, std={df.precision.std():.4f}")
        print(f"Recall: mean={df.recall.mean():.4f}, std={df.recall.std():.4f}")
        return df.auc.mean()

    table = pq.read_table('grouped_data.parquet')
    df = table.to_pandas()
    df = df.drop(columns=['App ID', 'PUID', 'Enrolled (Binary)', 'Decision History', 'School 1 Missing',
                        'School 3 Recency', 'School 4 Recency', 'School 5 Recency', 'School 6 Region',
                        'School 6 Class Rank (Numeric)', 'School 6 Class Size (Numeric)', 'School 6 Recency',
                        'Job 1 Missing', 'Job 2 Missing', 'Job 3 Missing', 'Job 4 Missing', 'Job 5 Missing',
                        'Job 6 Missing', 'is_fall', 'Purdue (Binary)', 'Job 6 Recency'])
    df = df.fillna(0)

    table2 = pq.read_table('models/transformer/data.parquet')
    df2 = table2.to_pandas()
    df2 = df2.fillna(0)
    
    job_desc_cols  = ["Job " + str(i) + " Description (embed)" for i in range(1, 7)]
    job_org_cols   = ["Job " + str(i) + " Organization (embed)" for i in range(1, 7)]
    job_title_cols = ["Job " + str(i) + " Title Enc"            for i in range(1, 7)]
    tensor_cols    = job_desc_cols + job_org_cols + job_title_cols

    skip_cols = {'Admitted (Binary)', 'App Term'} | set(tensor_cols)
    ablation_cols = [c for c in df.columns if c not in skip_cols]

    # ── baseline ─────────────────────────────────────────────────────────────
    print("Running baseline...")

    # After loading df
    baseline_auc = run_seeds(df.copy(), df2, seeds)
    print(f"Baseline AUC: {baseline_auc:.4f}")

    # ── ablation loop ────────────────────────────────────────────────────────
    ablation_results = []

    for col in ablation_cols:
        print(f"Dropping '{col}'...")
        df_ablated = df.drop(columns=[col])
        auc  = run_seeds(df_ablated, df2, seeds)
        delta = auc - baseline_auc          # negative = column was helpful
        ablation_results.append({"column": col, "auc": auc, "auc_change": delta})
        print(f"  AUC={auc:.4f}  Δ={delta:.4f}")

    # ── save results ─────────────────────────────────────────────────────────
    ablation_df = pd.DataFrame(ablation_results).sort_values("auc_change")
    ablation_df.to_csv("ablation_results.csv", index=False)
    print("\n=== Ablation Results (sorted by AUC change) ===")
    print(ablation_df.to_string(index=False))


    # Drop all job description embeddings at once
    # Drop all job org embeddings at once  
    # Drop all job title embeddings at once
    # Drop all job embeddings at once
    # Drop all school features at once

    group_ablations = {
        "all_job_desc_embeds"  : job_desc_cols,
        "all_job_org_embeds"   : job_org_cols,
        "all_job_title_embeds" : job_title_cols,
        "all_job_features"     : job_desc_cols + job_org_cols + job_title_cols,
        "all_school_features"  : [c for c in df.columns if "School" in c],
        "all_job_scalar_features" : [c for c in df.columns if "Job" in c and c not in tensor_cols],
    }

    for group_name, cols_to_drop in group_ablations.items():
        df_ablated = df.drop(columns=cols_to_drop, errors='ignore')
        auc = run_seeds(df_ablated, df2, seeds)
        delta = auc - baseline_auc
        print(f"{group_name}: AUC={auc:.4f}  Δ={delta:.4f}")

