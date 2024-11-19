#!/usr/bin/env python3
import torch
from flask  import Flask, render_template, request
from dasher import preprocess_text, make_prediction, Dasher

app = Flask(__name__)

# Initalise model once
input_dimensions = 47824
embedding_dimensions = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Dasher(input_dimensions, embedding_dimensions)
model.to(device)
model.load_state_dict(torch.load('dasher.pth', weights_only=True ,map_location=torch.device(device)))
model.eval()

def process_email(subject:str, body:str) -> torch.tensor:
    tokens = preprocess_text(subject, body)

    with torch.no_grad():
        output = model(tokens)
        prediction = torch.sigmoid(output)
    return prediction.item()


@app.route("/", methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        subject = request.form['subject']
        body = request.form['subject']

        result = process_email(subject, body)
        prediction = "Phishing" if result > 0.8 else "Safe"
        
        return render_template('index.html', prediction = result)
    else:
        return render_template('index.html', prediction = None)

if __name__ == '__main__':
    app.run(debug = True)
