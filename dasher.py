#!/usr/bin/env python3

import re
import nltk 
import torch 
import numpy        as np

from torch          import nn
from urllib.parse   import urlparse
from nltk.corpus    import stopwords
from nltk.tokenize  import word_tokenize
from nltk.stem      import WordNetLemmatizer 

#nltk.download('punkt')
#nltk.download('punkt_tab')
#nltk.download('stopwords')
#nltk.download('wordnet')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoding_table = np.load('encoding_table.npy', allow_pickle = True).item()
max_len = 1024

class Dasher(nn.Module):
  def __init__(self, input_dimensions, embedding_dimensions):
    super(Dasher, self).__init__()
    self.embedding = nn.Embedding(input_dimensions, embedding_dimensions)
    self.linear = nn.Linear(embedding_dimensions, 1)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
      x = self.embedding(x)                      # x shape: (batch_size, seq_len, embedding_dim)
      x = self.dropout(x)
      bs, seq_len, embedding_dim = x.shape
      x = x.permute(0, 2, 1)                     # Change to (batch_size, embedding_dim, seq_len)
      x = nn.functional.adaptive_avg_pool1d(x, 1).reshape(bs, -1)  # Now (batch_size, embedding_dim)
      out = self.linear(x)                       # Linear output (batch_size, 1)
      return out                                 # Return the linear output

def get_url_features(url:str) -> list:
    features = []

    parsed_url = urlparse(url)
    subdomain_count = len(parsed_url.netloc.split('.'))

    features.append(len(url))
    features.append(1 if "redirect" in url.lower() else 0)
    features.append(url.count('?'))
    features.append(subdomain_count)

    return features

def get_domain_features(domain:str)-> list:
    features = []
    domain_length = len(domain)
    subdomain_count  = len(domain.split('.'))

    features.append(domain_length)

    if subdomain_count < 10:
        features.append(subdomain_count)
    else:
        features.append(0)

    return features

def extract_urls_and_domains(text:str) -> tuple:
    url_regex = r'https?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' # Regex to capture URLs
    urls = re.findall(url_regex, text)
    body_without_urls = re.sub(url_regex, 'url', text)

    domain_regex = r'(?:(?<!-)[a-zA-Z0-9-]+(?<!-)\.)+[a-zA-Z]{2,}' # Regex to capture domains
    domains = re.findall(domain_regex, text)
    body_without_domains_and_urls = re.sub(domain_regex, 'domain', body_without_urls)

    return body_without_domains_and_urls, urls, domains

def extract_features_from_urls(urls:list) -> list:
    all_features = [get_url_features(url) for url in urls]

    if not all_features:
        return [0] * 4

    aggregated_features = []

    for i in range(len(all_features[0])):
        if isinstance(all_features[0][i], (int, float)):
            aggregated_features.append(sum(f[i] for f in all_features) / len(all_features))
        else:
            aggregated_features.append(None)

    return aggregated_features

def extract_features_from_domains(domains:list) -> list:
    all_features = [get_domain_features(domain) for domain in domains]

    if not all_features:
        return [0] * 2

    aggregated_features = []

    for i in range(len(all_features[0])):
        if isinstance(all_features[0][i], (int, float)):
            aggregated_features.append(sum(f[i] for f in all_features) / len(all_features))
        else:
            aggregated_features.append(None)

    return aggregated_features

def get_features(urls:list, domains:list) -> tuple:
    url_features = extract_features_from_urls(urls)
    domain_features = extract_features_from_domains(domains)
    return url_features, domain_features

def tokenize(text:str) -> list:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    return [lemmatizer.lemmatize(token) for token in tokens]

def encode(tokens:list) -> list:
    return [encoding_table.get(token, 0) for token in tokens]

def pad_text(encoded_tokens:list) -> list:
    if len(encoded_tokens) > max_len:
        padded_token_list = encoded_tokens[:max_len]  # Slice if longer than max_len
    else:
        padded_token_list = encoded_tokens + [0] * (max_len - len(encoded_tokens))  # Pad if shorter than max_len

    return padded_token_list

def preprocess_text(subject:str, body:str) -> torch.tensor:
    text = subject + " " + body # Concat subject + body
    text, urls, domains = extract_urls_and_domains(text)
    url_features, domain_features = get_features(urls, domains)
    tokens = tokenize(text) # tokenize text
    encoded_tokens = encode(tokens) # encode tokens
    padded_token_list = pad_text(encoded_tokens) # truncate or pad list

    values = padded_token_list + domain_features + url_features + [len(urls)] + [len(domains)]
    value_tensor = torch.tensor(values, dtype = torch.long)
    return value_tensor.unsqueeze(0)

def make_prediction(tokens:torch.tensor, model:nn.Module) -> torch.tensor:
    with torch.no_grad():
        output = model(tokens)
        return torch.sigmoid(output)

def main():
    input_dimensions = 46789
    embedding_dimensions = 100

    test_text = "Hello, this is not a phishing email but here is a link anyway https:facebook.com"
    test_subject = "Super safe email"
    tokens = preprocess_text(test_subject, test_text).to(device)

    model = Dasher(input_dimensions, embedding_dimensions)
    model.load_state_dict(torch.load('dasher.pth', weights_only = True, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    pred = make_prediction(tokens, model)
    if pred > 0.5:
        print("Phishing")
    else:
        print("Safe")
    return 0

if __name__ == '__main__':
    main()
