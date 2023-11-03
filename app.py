import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='.')

# Load the pre-trained Tagalog BERT model and tokenizer
model_name = "jcblaise/bert-tagalog-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

# Define a custom classifier on top of BERT
class SpamClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(SpamClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
num_classes = 2
# Load the model and tokenizer
model = SpamClassifier(bert_model, num_classes)
model.load_state_dict(torch.load("spam_classifier_weights.pth"))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("tokenizer_directory")

def predict_spam(input_text):
    # Tokenize and encode the input text
    encoded_text = tokenizer(input_text, truncation=True, padding=True, return_tensors="pt")
    input_ids = encoded_text.input_ids
    attention_mask = encoded_text.attention_mask

    # Make predictions using the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    _, predicted = torch.max(outputs, 1)

    # Decode the predicted label (0 for not spam, 1 for spam)
    if predicted.item() == 0:
        prediction = "Ham"
    else:
        prediction = "Spam"

    return prediction

@app.route("/")
def index():
    return render_template("index.html", prediction=None)



@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    prediction = predict_spam(text)
    return render_template("index.html", prediction=prediction, text=text)

@app.route("/upload", methods=["POST"])
def upload_csv():
    if "csv_file" not in request.files:
        return "No file part"

    file = request.files["csv_file"]
    
    if file.filename == "":
        return "No selected file"

    if file:
        # Read the CSV file and determine if each text in each row is spam or not
        # You can use Python's CSV library to parse the file and process the data
        import csv

        # Create a list to store the results
        results = []

        # Read the CSV file
        csv_text = file.read().decode("utf-8")
        csv_data = csv.reader(csv_text.splitlines())

        for row in csv_data:
            text = row[0]  # Assuming the text is in the first column
            prediction = predict_spam(text)
            results.append((text, prediction))

        return render_template("results.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
