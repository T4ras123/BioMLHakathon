import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('../Data/train.csv', 'r') as f:
    reader = csv.reader(f)
    train_data = list(reader)[1:]
    

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.tokenizer import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
tokenizer.load_vocab('vocab.json')
tokenizer.vocab[512] = '<none>'


encoded_data = []
for row in train_data:
    text = row[0]  # SMILES string
    label = int(row[1])  # Label
    tokens = tokenizer.encode(text)
    encoded_data.append((torch.tensor(tokens), label))
    
# pad all sequences to the same length
padded_data = []
for tokens, label in encoded_data:
  padded_tokens = F.pad(tokens, (0, 64 - len(tokens)), value=512)
  padded_data.append((padded_tokens, torch.tensor(label)))


class ClassificationTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, num_heads=8, num_layers=6, max_len=64, num_classes=2, dropout=0.1):
        super(ClassificationTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=512)
        self.pos_embedding = nn.Embedding(max_len, emb_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.fc = nn.Linear(emb_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_length = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)
        return logits

class TokenDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # List of token tensors
        self.labels = labels  # List of labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_batches(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in loader:
        tokens, labels = batch
        yield tokens.to(device), labels.to(device)

def train_model(model, train_dataset, val_dataset, epochs=1000, batch_size=32, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        val_accuracy = evaluate(model, val_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_classification_transformer.pth')
        
        scheduler.step()

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for tokens, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            outputs = model(tokens)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab_size = len(tokenizer.vocab)

    tokens = [item[0] for item in padded_data]
    labels = [item[1] for item in padded_data]
    
    # Create the dataset
    dataset = TokenDataset(tokens, labels)
    
    # Split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Initialize the model
    model = ClassificationTransformer(
      vocab_size=vocab_size,
      emb_dim=64,
      num_heads=4,
      num_layers=3,
      max_len=64,
      num_classes=2,
      dropout=0.1
    ).to(device)
    
    train_model(model, train_dataset, val_dataset, epochs=1000, batch_size=32, lr=0.005)