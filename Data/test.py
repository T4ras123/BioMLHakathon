# Python

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel, AdamW
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# ==============================
# 1. Data Preprocessing
# ==============================

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return np.zeros((n_bits,), dtype=int)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = np.array([smiles_to_fingerprint(s) for s in train_df['smiles']])
y = train_df['activity'].values
X_test = np.array([smiles_to_fingerprint(s) for s in test_df['smiles']])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# 2. Dataset Definition
# ==============================

class FingerprintDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = FingerprintDataset(X_train, y_train)
val_dataset = FingerprintDataset(X_val, y_val)
test_dataset = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# ==============================
# 3. Model Definition
# ==============================

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads, num_layers, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

model = TransformerClassifier(input_dim=2048, hidden_dim=512, num_classes=2, num_heads=8, num_layers=6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ==============================
# 4. Training Setup
# ==============================

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-4)

# ==============================
# 5. Training Loop
# ==============================

for epoch in range(20):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f'Epoch {epoch+1}, Validation Accuracy: {acc:.4f}')


# ==============================
# 6. Prediction on Test Data
# ==============================

model.eval()
predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

test_df['predicted_class'] = predictions
test_df[['id', 'predicted_class']].to_csv('predictions.csv', index=False)