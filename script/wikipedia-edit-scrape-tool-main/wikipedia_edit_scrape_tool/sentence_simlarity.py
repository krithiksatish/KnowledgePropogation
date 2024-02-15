from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import SentenceTransformer
import difflib


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
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.model = BertModel.from_pretrained('bert-base-uncased')
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        # self.model.eval()
        self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    def cosine_similarity(self, sentence1, sentence2):
        # sentence1 = sentence1.lower()
        # sentence2 = sentence2.lower()
        # # Tokenize and encode sentences
        # inputs1 = self.tokenizer(sentence1, return_tensors="pt", truncation=True, padding=True).to(self.device)
        # inputs2 = self.tokenizer(sentence2, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # # Get BERT embeddings for sentences
        # with torch.no_grad():
        #     outputs1 = self.model(**inputs1)
        #     outputs2 = self.model(**inputs2)

        # # Extract the embeddings (CLS token embeddings)
        # sentence_embedding1 = outputs1.last_hidden_state.mean(dim=1).squeeze()
        # sentence_embedding2 = outputs2.last_hidden_state.mean(dim=1).squeeze()

        sentence_embedding1 = self.model.encode(sentence1, convert_to_tensor=True)
        sentence_embedding2 = self.model.encode(sentence2, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity_score = torch.nn.functional.cosine_similarity(sentence_embedding1, sentence_embedding2, dim=0)

        return similarity_score.item()
    
    def edit_distance(self, sentence1, sentence2):
        # Split the sentences into words
        words1 = sentence1.split()
        words2 = sentence2.split()

        # Get the lengths of the input strings
        m = len(words1)
        n = len(words2)
        
        # Initialize a list to store the current row
        curr = [0] * (n + 1)
        
        # Initialize the first row with values from 0 to n
        for j in range(n + 1):
            curr[j] = j
        
        # Initialize a variable to store the previous value
        previous = 0
        
        # Loop through the rows of the dynamic programming matrix
        for i in range(1, m + 1):
            # Store the current value at the beginning of the row
            previous = curr[0]
            curr[0] = i
            
            # Loop through the columns of the dynamic programming matrix
            for j in range(1, n + 1):
                # Store the current value in a temporary variable
                temp = curr[j]
                
                # Check if the characters at the current positions in str1 and str2 are the same
                if words1[i - 1] == words2[j - 1]:
                    curr[j] = previous
                else:
                    # Update the current cell with the minimum of the three adjacent cells
                    curr[j] = 1 + min(previous, curr[j - 1], curr[j])
                
                # Update the previous variable with the temporary value
                previous = temp
     
        # The value in the last cell represents the minimum number of operations
        return curr[n]
    
    def is_signifcant_edit(self, sent1, sent2):
        # With current classifier, threshold ~0.85 seems reasonable with a = 0.8, b = 0.2?
        a = 0.7
        b = 0.20
        c = 0.10

        cosine_similarity_score = self.cosine_similarity(sent1, sent2)
        # Normalize edit distance into probability value
        edit_distance_score = 1 - self.edit_distance(sent1, sent2) / max(len(sent1), len(sent2))

        length_difference = 1 - abs(len(sent1) - len(sent2)) / max(len(sent1), len(sent2))

        simlarity_score = a * cosine_similarity_score + b * edit_distance_score + c * length_difference

        return simlarity_score