import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pyarrow.parquet as pq
import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
from transformers import AutoModelForSequenceClassification
from torch.utils.data import random_split
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer

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
    
class TabularTransformer(nn.Module):
    def __init__(self, 
                 float_dim,       # number of scalar float features
                 job_desc_dim,    # embedding size per job desc (e.g. 768)
                 job_org_dim,
                 job_title_dim,
                 d_model=64, 
                 nhead=8, 
                 num_layers=3):
        super().__init__()
        
        # Project each group → d_model (one token per group)
        self.float_proj     = nn.Linear(float_dim,     d_model)
        self.job_desc_proj  = nn.Linear(job_desc_dim,  d_model)
        self.job_org_proj   = nn.Linear(job_org_dim,   d_model)
        self.job_title_proj = nn.Linear(job_title_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, floats, job_desc, job_org, job_title):
        # Each projection: (batch, dim) → (batch, d_model) → (batch, 1, d_model)
        t1 = self.float_proj(floats).unsqueeze(1)
        t2 = self.job_desc_proj(job_desc).unsqueeze(1)
        t3 = self.job_org_proj(job_org).unsqueeze(1)
        t4 = self.job_title_proj(job_title).unsqueeze(1)

        # Stack as sequence of tokens: (batch, num_tokens, d_model)
        x = torch.cat([t1, t2, t3, t4], dim=1)  # (batch, 4, d_model)

        x = self.transformer(x)          # (batch, 4, d_model)
        x = x.mean(dim=1)                # (batch, d_model) — pool tokens
        return self.classifier(x)        # (batch, 1)

def set_seed(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    table = pq.read_table('grouped_data.parquet')
    df = table.to_pandas()
    y = torch.tensor(df['Admitted (Binary)'].values)
    df = df.drop(columns=['App ID', 'PUID', 'Enrolled (Binary)', 'Admitted (Binary)'])
    df = df.fillna(0)

    job_desc_cols = ["Job " + str(i) + " Description (embed)" for i in range(1,7)]
    job_org_cols = ["Job " + str(i) + " Organization (embed)" for i in range(1,7)]
    job_title_cols = ["Job " + str(i) + " Title Enc" for i in range(1,7)]
    tensor_cols = job_desc_cols + job_org_cols + job_title_cols

    job_desc_tensor = torch.stack([
    torch.cat([x if isinstance(x, torch.Tensor) else torch.tensor(x) 
               for x in row])
    for _, row in df[job_desc_cols].iterrows()
    ])

    job_org_tensor = torch.stack([
    torch.cat([x if isinstance(x, torch.Tensor) else torch.tensor(x) 
               for x in row])
    for _, row in df[job_org_cols].iterrows()
    ])

    job_title_tensor = torch.stack([
    torch.cat([x if isinstance(x, torch.Tensor) else torch.tensor(x) 
               for x in row])
    for _, row in df[job_title_cols].iterrows()
    ])
    
    float_cols = list(set(df.columns) - set(tensor_cols))
    df = df[float_cols].astype(np.float64)

    # arr = df[float_cols].applymap(
    # lambda x: x.item() if isinstance(x, torch.Tensor) else float(x)
    # ).values.astype(np.float32)

    # float_tensor = torch.from_numpy(arr)

    # Check all columns for problematic types

    float_tensor = torch.tensor(df[float_cols].values.astype(np.float32))
    # print(float_tensor.shape)
    # print(job_desc_tensor.shape)
    # print(job_org_tensor.shape)
    # print(job_title_tensor.shape)
    X = torch.cat([float_tensor, job_desc_tensor, job_org_tensor, job_title_tensor], dim=1)

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "roberta-base",
    #     num_labels=2
    # )

    model = TabularTransformer(
        float_dim     = float_tensor.shape[1],
        job_desc_dim  = job_desc_tensor.shape[1],
        job_org_dim   = job_org_tensor.shape[1],
        job_title_dim = job_title_tensor.shape[1],
        d_model=64,
        nhead=8,
        num_layers=3
    )

    dataset = AdmissionsDataset(float_tensor, job_desc_tensor, job_org_tensor, job_title_tensor, y)
    train_set, test_set = random_split(dataset, [0.8, 0.2])
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
        outputs = model(floats, job_desc, job_org, job_title).squeeze()
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

    auc = roc_auc_score(all_labels, all_preds)
    print(f"AUC: {auc:.4f}")