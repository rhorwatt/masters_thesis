import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pyarrow.parquet as pq
import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random

class EducationDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

# Example starting point: https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# https://medium.com/@sahin.samia/train-a-neural-network-in-pytorch-a-complete-beginners-walkthrough-3897d18d6078
class NeuralNetwork(torch.nn.Module):
    def __init__(self, feature_size, embed_dim, num_classes, dropout):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(feature_size, embed_dim),
            torch.nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            torch.nn.Linear(embed_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def set_seed(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_auc(model, dataloader, device):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())

    y_score = torch.cat(all_probs).view(-1).cpu().numpy()
    y_true = torch.cat(all_targets).view(-1).cpu().numpy()

    return roc_auc_score(y_true, y_score)


if __name__ == '__main__':
    #n_comps = [300, 400, 500, 600]
    #lrs = [1e-4, 3e-4, 1e-3]
    # dropout = [0.2, 0.3, 0.4, 0.5] 
    # weight_decay = [0, 1e-4, 1e-3, 1e-2]
    #batch_size = [32, 64, 128, 256]
    # first_layer = [512, 1024, 2048]

    n = 300
    lr = 3e-4
    d = 0.4
    w = 0
    f = 2048
    b = 32

    for seed in [0, 1, 2, 3, 4]:
        set_seed(seed)
        #table = pq.read_table('data/cleaned/data.parquet')
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
        # X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("School 6")])
        # X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("School 5")])
        # X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("Country")])
        # X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("School 5 Country_")])
        # X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("School 6_Country")])
        # X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("Job 5_Country")])
        # X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("Job 6_Country")])
        # X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("School_")])
        


        embedding_cols = ['Job ' + str(i) + ' Title Enc' for i in range(1,7)]
        embedding_cols.extend(['Job ' + str(i) + ' Description (embed)' for i in range(1,7)])
        embedding_cols.extend(['Job ' + str(i) + ' Organization (embed)' for i in range(1,7)])

        #X_pd = X_pd.drop(columns=[x for x in X_pd.columns.tolist() if x.startswith("Job")])
        X_pd = np.hstack([
            np.vstack(X_pd[c].values)
            for c in embedding_cols
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X_pd, y_pd, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        aucs = []
        
        pca = PCA(n_components=n)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

        X_train = X_train.to(torch.float32)
        X_test = X_test.to(torch.float32)
        y_train = y_train.to(torch.long)
        y_test = y_test.to(torch.long)

        train_data = EducationDataset(X_train, y_train)
        test_data = EducationDataset(X_test, y_test)

        train_dataloader = DataLoader(train_data, batch_size=b, shuffle=True, worker_init_fn=lambda _: np.random.seed(seed))
        test_dataloader = DataLoader(test_data, batch_size=b, shuffle=False, worker_init_fn=lambda _: np.random.seed(seed))

        model = NeuralNetwork(n, f, 2, d)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=w)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="max", 
            factor=0.3, 
            patience=10
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        num_epochs = 150
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for input, labels in train_dataloader:
                outputs = model(input)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            val_auc = compute_auc(model, test_dataloader, device)
            # if epoch % 20 == 0:
            #     print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")
            scheduler.step(val_auc)

        
        model.eval()
        correct, total = 0, 0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for input, labels in test_dataloader:
                outputs = model(input)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.append(probs)
                all_targets.append(labels)
        y_score = torch.cat(all_probs).view(-1).cpu().numpy()
        y_true = torch.cat(all_targets).view(-1).cpu().numpy()

        preds = (y_score >= 0.5).astype(int)
        auc = roc_auc_score(y_true, y_score)
        accuracy = accuracy_score(y_true, preds)
        # print(f"Test Accuracy: {accuracy:.4f}")
        # print(f"ROC Score: {auc:.4f}")
        aucs.append(auc)
    print(np.mean(aucs), np.std(aucs))