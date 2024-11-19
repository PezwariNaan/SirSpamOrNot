# Phishing Detection System

## Overview
This project is a **Phishing/Spam Detection System** built using **PyTorch** and **Flask**. It processes email data, analyzes the content for potential phishing threats, and predicts whether the email is safe or malicious. The project combines natural language processing (NLP) and machine learning to detect phishing attempts based on features extracted from the email subject, body, URLs, and domains.

---

## Features
- **NLP Tokenization and Encoding**: Processes email content, tokenizes text, and encodes it into numerical representations.
- **Feature Extraction**: Extracts and analyzes features from URLs and domains present in the email content.
- **Neural Network Model**: A PyTorch-based model, `Dasher`, predicts phishing likelihood.
- **Flask Web Interface**: User-friendly interface to input email subject and body, and view phishing predictions.

---

## Technology Stack
- **Backend Framework**: Flask
- **Machine Learning**: PyTorch
- **NLP Libraries**: NLTK
- **Frontend**: HTML (via Flask Templates)

---

## Prerequisites
Ensure the following dependencies are installed:
- **Python 3.7+**
- Required Python libraries:
  ```bash
  pip install torch flask nltk numpy
  pip install -r requirements.txt
  ```
  
- Python:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```

- **Run the Application**:
  ```bash
  python app.py
  ```

## Usage

1. Open your browser and navigate to `http://127.0.0.1:5000`.
2. Input the email subject and body in the provided form.
3. Submit the form to see the prediction:
   - **Phishing**: If the model determines a high likelihood of phishing.
   - **Safe**: If the model finds the email unlikely to be phishing.

## Future Enhancements

- Expand dataset to include more diverse phishing examples.
- Add support for email attachments and embedded links.
- Optimize prediction accuracy with more sophisticated models (e.g., transformers).
- Implement real-time email monitoring and alerts.

## Data
- The dataset used to train this model:
    
```https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset```

## License
- This project is open-source and distributed under GPL3.0.

