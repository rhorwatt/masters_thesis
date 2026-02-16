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
    def __init__(self, feature_size, embed_dim, num_classes):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(feature_size, embed_dim),
            torch.nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(0.3),
            torch.nn.Linear(embed_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


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
    X_pd = np.hstack([
        np.vstack(X_pd[c].values)
        for c in embedding_cols
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_pd, y_pd, test_size=0.2, random_state=42
    )

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

    train_dataloader = DataLoader(train_data, batch_size=32)
    test_dataloader = DataLoader(test_data, batch_size=32)

    model = NeuralNetwork(11016, 512, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 100
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

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

    
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
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"ROC Score: {auc:.4f}")