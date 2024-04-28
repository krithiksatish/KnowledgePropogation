from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel,\
                            AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
import difflib
from nltk.tokenize import sent_tokenize

class SentenceSimilarityCalculator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize BERT
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.to(self.device).eval()
        
        # Initialize RoBERTa
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.roberta_model.to(self.device).eval()
        
        # Initialize RoBERTa-MNLI for contradiction detection
        self.mnli_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
        self.mnli_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
        self.mnli_model.to(self.device).eval()

        # Initialize spaCy (for subject/verb detection)
        self.nlp_sm = spacy.load("en_core_web_sm")  # Load spaCy's English model
        self.nlp_lg = spacy.load("en_core_web_lg")  # Load spaCy's large English model

    def cosine_similarity(self, model, tokenizer, sentence1, sentence2):
        """Computes the cosine similarity between two sentences using specified model and tokenizer."""
        inputs1 = tokenizer(sentence1, return_tensors="pt", truncation=True, padding=True).to(self.device)
        inputs2 = tokenizer(sentence2, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)

        emb1 = outputs1.last_hidden_state.mean(dim=1).squeeze()
        emb2 = outputs2.last_hidden_state.mean(dim=1).squeeze()

        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()

    def bert_cosine_similarity(self, sentence1, sentence2):
        """Computes cosine similarity using BERT."""
        return self.cosine_similarity(self.bert_model, self.bert_tokenizer, sentence1, sentence2)

    def roberta_cosine_similarity(self, sentence1, sentence2):
        """Computes cosine similarity using RoBERTa."""
        return self.cosine_similarity(self.roberta_model, self.roberta_tokenizer, sentence1, sentence2)

    def check_contradiction(self, sentence1, sentence2):
        """Checks if two sentences contradict each other using RoBERTa trained on MNLI."""
        """Returns the predicted class and the confidence score of the predicted class."""
        inputs = self.mnli_tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.mnli_model(**inputs).logits
            results = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(results).item()

        labels = ['contradiction', 'neutral', 'entailment']
        
        return labels[predicted_class], results[:, predicted_class].item()
    
    def has_subject_and_verb(self, sentence, model='sm'):
        """Check if the sentence contains at least one verb and one subject."""
        if model == 'sm':
            doc = self.nlp_sm(sentence)
        elif model == 'lg':
            doc = self.nlp_lg(sentence)
        else:
            raise ValueError("Invalid model name. Choose 'sm' or 'lg'.")
        
        has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in doc)  # Including passive subjects
        has_verb = any(token.pos_ == 'VERB' for token in doc)
        return has_subject and has_verb

    def edit_distance(self, sentence1, sentence2):
        words1 = sentence1.split()
        words2 = sentence2.split()
        dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]

        for i in range(len(words1) + 1):
            for j in range(len(words2) + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif words1[i-1] == words2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[len(words1)][len(words2)]

    def is_significant_edit(self, sent1, sent2):
        # # With multi-qa-mpnet-base-dot-v1, threshold ~0.75/0.60 seems reasonable
        # # With bert-base-uncased, threshold ~0.80 (seems better than the above)
        # a = 0.8
        # b = 0.10
        # c = 0.10

        # words1 = sent1.split()
        # words2 = sent2.split()

        # cosine_similarity_score = self.bert_cosine_similarity(sent1, sent2)
        # # Normalize edit distance and length_difference into probability value

        # max_len = max(len(words1), len(words2))

        # edit_distance_score = 0 if max_len == 0 else self.edit_distance(sent1, sent2) / max_len
        # length_difference = 0 if max_len == 0 else 1 - abs(len(words1) - len(words2)) / max_len
        # simlarity_score = a * cosine_similarity_score + b * edit_distance_score + c * length_difference

        # return simlarity_score

        bert_cosine_similarity = self.bert_cosine_similarity(sent1, sent2)
        has_subject_and_verb = self.has_subject_and_verb(sent2)
        # 0.80 threshold value for cosine similarity
        return has_subject_and_verb and (bert_cosine_similarity <= 0.80)