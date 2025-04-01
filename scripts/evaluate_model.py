from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.metrics import accuracy_score

# Load trained model
model = BertForSequenceClassification.from_pretrained("models/saved_model")
model.to("cpu")

# Sample test data
test_texts = ["This phone is amazing!", "Worst phone I've ever used."]
test_labels = [1, 0]

# Tokenization
tokenizer = BertTokenizer.from_pretrained("models/saved_model")
inputs = tokenizer(test_texts, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

# Prediction
with torch.no_grad():
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).tolist()

# Accuracy
print(f"Accuracy: {accuracy_score(test_labels, preds):.2f}")
