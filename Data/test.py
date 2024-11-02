import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ==============================
# 1. Data Preprocessing
# ==============================

# Load training data
train_df = pd.read_csv('train.csv')

# Function to compute Morgan fingerprints
def get_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    else:
        return None

# Convert SMILES to fingerprints
train_df['fingerprint'] = train_df['smiles'].apply(get_fingerprint)
train_df = train_df.dropna(subset=['fingerprint'])

# Convert fingerprints to numpy arrays
X = np.array([list(fp) for fp in train_df['fingerprint']])
y = train_df['activity'].values

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ==============================
# 2. Dataset and DataLoader
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# ==============================
# 3. Model Definition (GNN)
# ==============================

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

input_dim = X_train.shape[1]
hidden_dim = 512
output_dim = 2

model = SimpleNN(input_dim, hidden_dim, output_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ==============================
# 4. Training Setup
# ==============================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ==============================
# 5. Training and Evaluation Functions
# ==============================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    for X_batch, y_batch in tqdm(loader, desc="Training"):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == y_batch)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct.double() / len(loader.dataset)
    return epoch_loss, epoch_acc.item()

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Evaluating"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == y_batch)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct.double() / len(loader.dataset)
    return epoch_loss, epoch_acc.item()

# ==============================
# 6. Training Loop with Early Stopping
# ==============================

best_val_acc = 0.0
patience = 5
counter = 0
num_epochs = 50

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = eval_model(model, val_loader, criterion, device)
    
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    scheduler.step()
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# ==============================
# 7. Inference on Test Data
# ==============================

# Load test data
test_df = pd.read_csv('test.csv')

# Convert SMILES to fingerprints
test_df['fingerprint'] = test_df['smiles'].apply(get_fingerprint)
test_df = test_df.dropna(subset=['fingerprint'])

# Convert to numpy array
X_test = np.array([list(fp) for fp in test_df['fingerprint']])

class TestDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

test_dataset = TestDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=64)

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

predictions = []
with torch.no_grad():
    for X_batch in tqdm(test_loader, desc="Predicting"):
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

# Save Predictions
test_df['predicted_class'] = predictions
test_df[['id', 'predicted_class']].to_csv('predictions.csv', index=False)