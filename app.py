from flask import Flask, request, render_template, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("models/saved_model")
tokenizer = BertTokenizer.from_pretrained("models/saved_model")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the POST request
    data = request.json
    text = data['text']
    
    # Tokenization and prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({"sentiment": sentiment})

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    review = "I absolutely love my new iPhone X!"
    # Tokenization and prediction
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({"review": review, "sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
