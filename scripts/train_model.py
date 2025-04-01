import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Load and preprocess dataset
df = pd.read_csv("data/cleaned_iphonexreview.csv")

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'UserName'])

# Print columns to ensure correct data is being used
print("Columns in dataset:", df.columns)

# Simple sentiment labeling based on review text
def label_sentiment(review):
    positive_keywords = ['good', 'great', 'love', 'awesome', 'excellent']
    negative_keywords = ['bad', 'poor', 'hate', 'worst']

    # Check for positive and negative words in the review text
    if any(keyword in review.lower() for keyword in positive_keywords):
        return 1  # Positive sentiment
    elif any(keyword in review.lower() for keyword in negative_keywords):
        return 0  # Negative sentiment
    else:
        return 1  # Default to positive if no strong sentiment is detected

# Apply sentiment labeling
df['sentiment'] = df['Review'].apply(label_sentiment)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx], truncation=True, padding="max_length", max_length=128, return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Prepare dataset
dataset = ReviewDataset(df['Review'].tolist(), df['sentiment'].tolist())
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model setup
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(3):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed.")

# Save model
model.save_pretrained("models/saved_model")
tokenizer.save_pretrained("models/saved_model")
print("Model training completed and saved.")
