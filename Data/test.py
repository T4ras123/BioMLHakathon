import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Custom Dataset
class SMILESDataset(Dataset):
    def __init__(self, smiles, labels, tokenizer, max_length=100):
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smile = self.smiles[idx]
        label = self.labels[idx] if self.labels is not None else -1
        encoding = self.tokenizer.encode_plus(
            smile,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long) if label != -1 else torch.tensor(label, dtype=torch.long)
        }

# Transformer Classification Model
class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

# Parameters
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 100
LEARNING_RATE = 2e-5

# Load Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Split training data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['smiles'], train_df['activity'], test_size=0.2, random_state=42
)

# Initialize Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create Datasets
train_dataset = SMILESDataset(
    smiles=train_texts.to_numpy(),
    labels=train_labels.to_numpy(),
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

val_dataset = SMILESDataset(
    smiles=val_texts.to_numpy(),
    labels=val_labels.to_numpy(),
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize Model
model = BertClassifier(n_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define Optimizer and Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training Function
def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    losses = []
    correct_predictions = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)
        
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses)

# Evaluation Function
def eval_model(model, data_loader, criterion, device):
    model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses)

# Training Loop
for epoch in range(EPOCHS):
    print(f'Epoch {epoch +1}/{EPOCHS}')
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f'Train loss {train_loss} accuracy {train_acc}')
    
    val_acc, val_loss = eval_model(model, val_loader, criterion, device)
    print(f'Validation loss {val_loss} accuracy {val_acc}')

# Inference on Test Data
test_dataset = SMILESDataset(
    smiles=test_df['smiles'].to_numpy(),
    labels=None,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model.eval()
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())

# Save Predictions
test_df['predicted_class'] = predictions
test_df[['id', 'predicted_class']].to_csv('predictions.csv', index=False)