import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import time

# Initialize Contriver model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
model = AutoModel.from_pretrained('facebook/contriever-msmarco')

# Function to split content into documents every 10 lines
def split_into_documents(content, lines_per_document=10):
    lines = content.split('\n')
    return [' '.join(lines[i:i + lines_per_document]) for i in range(0, len(lines), lines_per_document)]

# Read content from the text file
with open("pile_val_pubmedabstract.txt", "r") as file:
    content = file.read()

# Split content into documents every 10 lines
lines_per_document = 20
documents = split_into_documents(content, lines_per_document)

# Example query
query = "Where was Marie Curie born?"

# Tokenize query and documents
inputs_query = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
inputs_documents = tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512, return_attention_mask=True)

# Get embeddings for query and documents
with torch.no_grad():
    outputs_query = model(**inputs_query)
    outputs_documents = model(**inputs_documents)

# Pooling function
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

start_time = time.time()
# Compute embeddings using mean_pooling
query_embedding = mean_pooling(outputs_query.last_hidden_state, inputs_query['attention_mask'])
document_embeddings = mean_pooling(outputs_documents.last_hidden_state, inputs_documents['attention_mask'])

# Calculate cosine similarity between query and documents [one method of several types of similarity calculation]
similarity_scores = cosine_similarity(query_embedding, document_embeddings)

# Print similarity scores
for i, score in enumerate(similarity_scores.squeeze()):
    print(f"Document {i+1}: Similarity Score = {score:.4f}")

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")