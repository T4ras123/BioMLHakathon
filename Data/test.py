import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================
# 1. Device Configuration
# ==============================

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ==============================
# 2. Data Preprocessing
# ==============================

# Load training data
train_df = pd.read_csv('train.csv')

# Function to compute Morgan fingerprints using AllChem.GetMorganFingerprintAsBitVect
def get_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Generate Morgan Fingerprint as Bit Vector
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        # Convert ExplicitBitVect to a NumPy array
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return None

# Alternative: Using a generator if you prefer incremental fingerprint generation
# However, using GetMorganFingerprintAsBitVect is more straightforward for this use case

# Import DataStructs for conversion
from rdkit import DataStructs

# Convert SMILES to fingerprints
print("Generating fingerprints...")
train_df['fingerprint'] = train_df['smiles'].apply(get_fingerprint)
initial_count = len(train_df)
train_df = train_df.dropna(subset=['fingerprint'])
final_count = len(train_df)
print(f'Dropped {initial_count - final_count} samples due to invalid SMILES.')

# Convert fingerprints to numpy arrays
X = np.array([fp for fp in train_df['fingerprint']])
y = train_df['activity'].values

# Handle class imbalance using SMOTE
print("Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f'Resampled training samples: {len(X_res)}')

# Feature Scaling
print("Scaling features...")
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)

# Split data into training and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print(f'Training samples: {len(X_train)}, Validation samples: {len(X_val)}')

# ==============================
# 3. Dataset and DataLoader
# ==============================

class FingerprintDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create Dataset instances
train_dataset = FingerprintDataset(X_train, y_train)
val_dataset = FingerprintDataset(X_val, y_val)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# ==============================
# 4. Model Definition
# ==============================

class EnhancedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.5):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        return out

# Define model parameters
input_dim = X_train.shape[1]      # Number of fingerprint bits (e.g., 2048)
hidden_dim1 = 512
hidden_dim2 = 256
output_dim = 2                    # Number of classes
dropout = 0.5

# Initialize the model
model = EnhancedNN(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout)
model = model.to(device)
print(model)

# ==============================
# 5. Training Setup
# ==============================

# Define loss function with class weights (optional if class imbalance handled by SMOTE)
# If you haven't applied SMOTE, compute class weights as follows:
# from sklearn.utils import class_weight
# class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

# ==============================
# 6. Training and Evaluation Functions
# ==============================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
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
        total += y_batch.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Evaluating"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == y_batch)
            total += y_batch.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()

# ==============================
# 7. Training Loop with Early Stopping
# ==============================

best_val_acc = 0.0
patience = 10
counter = 0
num_epochs = 50

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 30)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = eval_model(model, val_loader, criterion, device)
    
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    # Step the scheduler based on validation accuracy
    scheduler.step(val_acc)
    
    # Check for improvement
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Best model saved with Val Acc: {best_val_acc:.4f}')
        counter = 0
    else:
        counter += 1
        print(f'No improvement for {counter} epochs')
        if counter >= patience:
            print("Early stopping triggered")
            break

# ==============================
# 8. Inference on Test Data
# ==============================

# Load test data
test_df = pd.read_csv('test.csv')

# Function to compute fingerprints for test data
def get_fingerprint_test(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return None

# Convert SMILES to fingerprints for test data
print("\nGenerating fingerprints for test data...")
test_df['fingerprint'] = test_df['smiles'].apply(get_fingerprint_test)
initial_test_count = len(test_df)
test_df = test_df.dropna(subset=['fingerprint'])
final_test_count = len(test_df)
print(f'Dropped {initial_test_count - final_test_count} test samples due to invalid SMILES.')

# Convert to numpy array
X_test = np.array([fp for fp in test_df['fingerprint']])

# Apply the same scaling
print("Scaling test features...")
X_test = scaler.transform(X_test)

class TestDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

# Create Test Dataset and DataLoader
test_dataset = TestDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Load the best model
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
print("\nLoaded the best model for inference.")

# Make predictions
predictions = []
with torch.no_grad():
    for X_batch in tqdm(test_loader, desc="Predicting"):
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

# Save Predictions
test_df['predicted_class'] = predictions
output_df = test_df[['id', 'predicted_class']]
output_df.to_csv('predictions.csv', index=False)
print("\nPredictions saved to 'predictions.csv'")