import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from hazm import word_tokenize, stopwords_list
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

with open('data/persian_texts.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]

stopwords = set(stopwords_list())

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stopwords]
    return tokens

tokenized_texts = [preprocess(t) for t in texts]

vectorizer = TfidfVectorizer(tokenizer=preprocess, max_features=100)
tfidf_matrix = vectorizer.fit_transform(texts).toarray()

w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

def text_to_w2v(tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

w2v_matrix = np.array([text_to_w2v(t) for t in tokenized_texts])

model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_gemma_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return emb

sample_size = min(5, len(texts))
gemma_embeddings = []
for i in range(sample_size):
    emb = get_gemma_embedding(texts[i])
    gemma_embeddings.append(emb)
