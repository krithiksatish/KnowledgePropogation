from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import SentenceTransformer
import difflib

# Example use: similarity_calculator.is_signifcant_edit("OpenAI is most known for ChatGPT", "Open AI is most known for ChatGPT and Sora")
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
        #self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    def cosine_similarity(self, sentence1, sentence2):
        sentence1 = sentence1.lower()
        sentence2 = sentence2.lower()
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

        return similarity_score.item()
    
    def edit_distance(self, sentence1, sentence2):
        # See https://www.geeksforgeeks.org/edit-distance-dp-5/ for documentation
        words1 = sentence1.split()
        words2 = sentence2.split()

        m = len(words1)
        n = len(words2)
        
        curr = [0] * (n + 1)

        for j in range(n + 1):
            curr[j] = j
        
        previous = 0
        
        for i in range(1, m + 1):
            previous = curr[0]
            curr[0] = i
        
            for j in range(1, n + 1):
                temp = curr[j]
                
                if words1[i - 1] == words2[j - 1]:
                    curr[j] = previous
                else:
                    curr[j] = 1 + min(previous, curr[j - 1], curr[j])
                previous = temp

        return curr[n]
    
    def is_signifcant_edit(self, sent1, sent2):
        # With multi-qa-mpnet-base-dot-v1, threshold ~0.75/0.60 seems reasonable
        # With bert-base-uncased, threshold ~0.80 (seems better than the above)
        a = 0.8
        b = 0.10
        c = 0.10

        words1 = sent1.split()
        words2 = sent2.split()

        cosine_similarity_score = self.cosine_similarity(sent1, sent2)
        # Normalize edit distance and length_difference into probability value
        edit_distance_score = 1 - self.edit_distance(sent1, sent2) / max(len(words1), len(words2))

        length_difference = 1 - abs(len(words1) - len(words2)) / max(len(words1), len(words2))

        simlarity_score = a * cosine_similarity_score + b * edit_distance_score + c * length_difference

        return simlarity_score