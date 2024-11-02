import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.utils import class_weight
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import sys

# Define the device
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

# Encode data
encoded_data = []
for row in train_data:
    text = row[0]  # SMILES string
    label = int(row[1])  # Label
    tokens = tokenizer.encode(text)
    encoded_data.append((tokens, label))

# Pad all sequences to the same length (64) with padding value 0
padded_data = []
for tokens, label in encoded_data:
    if len(tokens) < 64:
        padded_tokens = F.pad(tokens, (0, 64 - len(tokens)), value=tokenizer.pad_idx)
    else:
        padded_tokens = tokens[:64]
    padded_data.append((padded_tokens, torch.tensor(label, dtype=torch.long)))

# Check label distribution
labels = [label.item() for _, label in padded_data]
label_counts = Counter(labels)
print(f'Label Distribution: {label_counts}')

# If imbalanced, compute class weights
if len(label_counts) > 1:
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
else:
    class_weights = None

# Define the Dataset
class TokenDataset(Dataset):
    def __init__(self, data):
        self.data = data  # List of (padded_tokens, label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

# Split into training and validation sets
train_size = int(0.8 * len(padded_data))
val_size = len(padded_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(padded_data, [train_size, val_size])

# Define the Model
class ClassificationTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, num_heads=8, num_layers=6, max_len=64, num_classes=2, dropout=0.1):
        super(ClassificationTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, emb_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.fc = nn.Linear(emb_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_dim)
    
    def forward(self, x):
        batch_size, seq_length = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, emb_dim)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_length)
        x = self.embedding(x) + self.pos_embedding(positions)  # (batch_size, seq_length, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, seq_length + 1, emb_dim)
        x = self.transformer(x)  # (batch_size, seq_length + 1, emb_dim)
        cls_output = x[:, 0]  # (batch_size, emb_dim)
        cls_output = self.norm(cls_output)
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)  # (batch_size, num_classes)
        return logits

# Define the Evaluation Function
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for tokens, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            outputs = model(tokens)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.4f}')
    return accuracy

# Define the Training Function
def train_model(model, train_dataset, val_dataset, epochs=1000, batch_size=32, lr=0.005, patience=10):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    best_val_accuracy = 0.0
    trigger_times = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        val_accuracy = evaluate(model, val_loader)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(val_accuracy)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            trigger_times = 0
            torch.save(model.state_dict(), 'best_classification_transformer.pth')
            print(f'Best model saved with accuracy: {best_val_accuracy:.4f}')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping triggered')
                break
        
        # Optional: Plot training progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss over Epochs')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy over Epochs')
            plt.legend()

            plt.show()

# Example Usage
if __name__ == "__main__":
    # Define vocabulary size
    vocab_size = len(vocab)  # Ensure vocab is properly loaded and defined
    
    # Initialize the model
    model = ClassificationTransformer(
        vocab_size=vocab_size,
        emb_dim=128,     # Increased embedding dimension
        num_heads=8,     # Increased number of heads
        num_layers=6,    # Increased number of layers
        max_len=64,
        num_classes=2,
        dropout=0.1
    ).to(device)
    
    # Train the model
    train_model(model, train_dataset, val_dataset, epochs=1000, batch_size=32, lr=1e-3, patience=10)
    
    # Plot final training progress
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()

    plt.show()