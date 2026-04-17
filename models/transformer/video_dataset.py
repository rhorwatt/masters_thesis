from datasets import load_dataset
import torch

ds = load_dataset("GrassData/grass-clickstream-dataset")

print(ds)
dataset = ds["test"]
embeddings = torch.tensor(dataset["embedding"])
labels = dataset["type"]
print(embeddings.shape)
print(embeddings)