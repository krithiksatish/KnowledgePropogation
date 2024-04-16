from transformers import pipeline

# Load a pre-trained NLI model
nli_pipeline = pipeline("zero-shot-classification", model="roberta-large-mnli")

# Define two sentences
sentence1 = "Pluto is a planet"
sentence2 = "Pluto was a planet"

# Use the NLI model to predict the relationship
result = nli_pipeline(sentence1, candidate_labels=[sentence2], hypothesis_template="{}")

print(f"Label: {result['labels'][0]}, Score: {result['scores'][0]}")
