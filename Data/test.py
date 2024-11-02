# Python

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import numpy as np
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

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Generate Morgan Fingerprint as Bit Vector
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        # Convert ExplicitBitVect to a NumPy array
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return None

# Load training and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Convert SMILES to fingerprints
print("Generating fingerprints for training data...")
train_df['fingerprint'] = train_df['smiles'].apply(smiles_to_fingerprint)
initial_train_count = len(train_df)
train_df = train_df.dropna(subset=['fingerprint'])
final_train_count = len(train_df)
print(f'Dropped {initial_train_count - final_train_count} training samples due to invalid SMILES.')

print("Generating fingerprints for test data...")
test_df['fingerprint'] = test_df['smiles'].apply(smiles_to_fingerprint)
initial_test_count = len(test_df)
test_df = test_df.dropna(subset=['fingerprint'])
final_test_count = len(test_df)
print(f'Dropped {initial_test_count - final_test_count} test samples due to invalid SMILES.')

# Prepare feature matrices and labels
X = np.array(train_df['fingerprint'].tolist())
y = train_df['activity'].values
X_test = np.array(test_df['fingerprint'].tolist())

# Handle class imbalance using SMOTE
print("Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f'Resampled training samples: {len(X_res)}')

# Feature Scaling
print("Scaling features...")
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)
X_test = scaler.transform(X_test)

# Split data into training and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print(f'Training samples: {len(X_train)}, Validation samples: {len(X_val)}')

# ==============================
# 3. Dataset Definition
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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# ==============================
# 4. Model Definition
# ==============================

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True  # Enables batch as the first dimension
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, input_dim) -> (batch_size, hidden_dim)
        x = x.unsqueeze(1)     # Add sequence dimension: (batch_size, 1, hidden_dim)
        x = self.transformer(x)  # Shape: (batch_size, sequence_length, hidden_dim)
        x = x.mean(dim=1)         # Global average pooling: (batch_size, hidden_dim)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)            # Shape: (batch_size, num_classes)
        return x

# Define model parameters
input_dim = X_train.shape[1]      # Number of fingerprint bits (e.g., 2048)
hidden_dim = 512
num_heads = 8
num_layers = 3
num_classes = 2                    # Number of classes (binary classification)
dropout = 0.3

# Initialize the model
model = TransformerClassifier(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout=dropout
)
model.to(device)
print(model)

# ==============================
# 5. Training Setup
# ==============================

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=5, verbose=True
)

# ==============================
# 6. Training and Evaluation Functions
# ==============================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in tqdm(loader, desc="Training", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Evaluating", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

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
        torch.save(model.state_dict(), 'best_transformer_model.pth')
        print(f'Best model saved with Val Acc: {best_val_acc:.4f}')
        counter = 0
    else:
        counter += 1
        print(f'No improvement for {counter} epochs')
        if counter >= patience:
            print("Early stopping triggered")
            break
    
    # Early exit if desired accuracy is achieved
    if best_val_acc >= 0.7:
        print(f"Desired validation accuracy of {best_val_acc:.4f} achieved. Stopping training.")
        break

# ==============================
# 8. Prediction on Test Data
# ==============================

# Prepare test data
class TestDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

test_dataset = TestDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Load the best model
model.load_state_dict(torch.load('best_transformer_model.pth', map_location=device))
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

# Assign predictions to test dataframe
test_df['predicted_class'] = predictions

# Save predictions to CSV
output_df = test_df[['id', 'predicted_class']]
output_df.to_csv('transformer_predictions.csv', index=False)
print("\nPredictions saved to 'transformer_predictions.csv'")