from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import SentenceTransformer

class SentenceSimilarityCalculator:
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        '''
        Notes:
        - bert-large seems to do worst?
        - SBERT doesn't seem to do great either

        ClinicalBERT may be better for our use case (since search space is medical related)
        - haven't tested this yet
        '''
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        #self.model = SentenceTransformer('all-MiniLM-L6-v2')

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

        # sentence_embedding1 = self.model.encode(sentence1, convert_to_tensor=True)
        # sentence_embedding2 = self.model.encode(sentence2, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity_score = torch.nn.functional.cosine_similarity(sentence_embedding1, sentence_embedding2, dim=0)

        # Adjust similarity score based on length difference
        length_difference = abs(len(sentence1.split()) - len(sentence2.split()))
        if length_difference > 0:
            similarity_score -= length_difference * 0.01  # Can definitely adjust this weight

        # Could also adjust score based on edit difference?

        return similarity_score.item()
    
    def is_signifcant_edit(self, sentence1, sentence2, threshold):
        # With current classifier, threshold ~0.80 seems reasonable?
        return self.calculate_similarity(sentence1, sentence2) < threshold