from transformers import BertTokenizer, BertModel
import torch

class SentenceSimilarityCalculator:
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def calculate_similarity(self, sentence1, sentence2):
        # Tokenize and encode sentences
        inputs1 = self.tokenizer(sentence1, return_tensors="pt", truncation=True, padding=True).to(self.device)
        inputs2 = self.tokenizer(sentence2, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # Get BERT embeddings for sentences
        with torch.no_grad():
            outputs1 = self.model(**inputs1)
            outputs2 = self.model(**inputs2)

        # Extract the embeddings (CLS token embeddings)
        sentence_embedding1 = outputs1.last_hidden_state.mean(dim=1).squeeze()
        sentence_embedding2 = outputs2.last_hidden_state.mean(dim=1).squeeze()

        # Calculate cosine similarity
        similarity_score = torch.nn.functional.cosine_similarity(sentence_embedding1, sentence_embedding2, dim=0)
        return similarity_score.item()
    
    def is_signifcant_edit(self, sentence1, sentence2, threshold):
        return self.calculate_similarity(sentence1, sentence2) < threshold