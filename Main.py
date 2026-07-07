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