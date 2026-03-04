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
from transformers import BertConfig, BertModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, 
                              f1_score, precision_score, 
                              recall_score, confusion_matrix)

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

    float_tensor = torch.tensor(scaler.fit_transform(df[float_cols].values), dtype=torch.float32)
    X = torch.cat([float_tensor, job_desc_tensor, job_org_tensor, job_title_tensor], dim=1)

    # Instantiate
    model = TabularBertTransformer(
        float_dim     = float_tensor.shape[1],
        job_desc_dim  = job_desc_tensor.shape[1],
        job_org_dim   = job_org_tensor.shape[1],
        job_title_dim = job_title_tensor.shape[1],
        d_model=768,
        num_layers=6
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
    all_preds_binary = (all_preds > 0.5).astype(int)
    auc       = roc_auc_score(all_labels, all_preds)
    accuracy  = accuracy_score(all_labels, all_preds_binary)
    f1        = f1_score(all_labels, all_preds_binary)
    precision = precision_score(all_labels, all_preds_binary)
    recall    = recall_score(all_labels, all_preds_binary)
    cm        = confusion_matrix(all_labels, all_preds_binary)

    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Confusion Matrix:\n{cm}")