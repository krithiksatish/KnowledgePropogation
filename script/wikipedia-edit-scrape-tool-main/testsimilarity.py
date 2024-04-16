from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np
import torch

print ("Hello World")

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded")

# Define two excerpts
excerpt1 = "Charles Tappert writes that [[Frank Rosenblatt]] developed and explored all of the basic ingredients of the deep learning systems of today, referring to Rosenblatt's 1962 book which introduced a [[multilayer perceptron]] (MLP) with 3 layers: an input layer, a hidden layer with randomized weights that did not learn, and an output layer. However, since only the output layer had learning connections, this was not yet deep learning. It was what later was called an [[extreme learning machine]].."
excerpt2 = "Charles Tappert writes that [[Frank Rosenblatt]] developed and explored all of the basic ingredients of the deep learning systems of today, referring to Rosenblatt's 1962 book which introduced [[multilayer perceptron]] (MLP) with 3 layers: an input layer, a hidden layer with randomized weights that did not learn, and an output layer. It also introduced variants, including a version with four-layer perceptrons where the last two layers have learned weights (and thus a proper multilayer perceptron)."
print("Excerpts defined")

# Split excerpts into sentences
sentences1 = sent_tokenize(excerpt1)
sentences2 = sent_tokenize(excerpt2)
print("Sentences split")

# Compute embeddings for all sentences
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)
print("Embeddings computed")

# Compute cosine similarity between each pair of sentences and find the maximum similarity for each sentence in excerpt1
max_similarities = []
for emb1 in embeddings1:
    similarities = [cosine_similarity([emb1], [emb2])[0][0] for emb2 in embeddings2]
    max_similarities.append(max(similarities))
print("Similarities computed")

# Calculate the average or minimum of the maximum similarities
print (max_similarities)
avg_similarity = np.mean(max_similarities)
min_similarity = np.min(max_similarities)
max_similarity = np.max(max_similarities)
print("Statistics computed")

print(f'Average similarity: {avg_similarity}')
print(f'Minimum similarity: {min_similarity}')
print(f'Maximum similarity: {max_similarity}')