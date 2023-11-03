from flask import Flask, request, jsonify
import torch
from transformers import AutoModel, AutoTokenizer

app = Flask(__name)

# Load the pre-trained model and tokenizer
model_name = "jcblaise/bert-tagalog-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SpamClassifier(bert_model, num_classes)  # Initialize your trained model here

# Define a function to make predictions
def predict_spam(input_text):
    encoded_text = tokenizer(input_text, truncation=True, padding=True, return_tensors="pt")
    input_ids = encoded_text.input_ids
    attention_mask = encoded_text.attention_mask

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    _, predicted = torch.max(outputs, 1)

    if predicted.item() == 0:
        return "Ham"
    else:
        return "Spam"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']
    prediction = predict_spam(input_text)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
