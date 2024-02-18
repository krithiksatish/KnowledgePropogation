# go through pile_val_pubmedabstract and treat every 10 lines as a new document for ranking purposes
# in the wikipedia file, ignore the headings and extract atomic facts from every line 

# use atomic fact response from chatGPT to do retrieval lookup in pile_val_abstract on diff retrieval types 
# in table list overall scoring 

from math import log
import string
import time
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load the English stopwords
stop_words = set(stopwords.words('english'))

# Function to remove stopwords from a list of tokens
def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

def simple_tokenizer(document):
    if isinstance(document, str):
        translator = str.maketrans("", "", string.punctuation)
        tokens = document.lower().translate(translator).split(None)
        return remove_stopwords(tokens)
    elif isinstance(document, list):
        # Assuming each element of the list is a line in the document
        return [remove_stopwords(line.lower().translate(str.maketrans("", "", string.punctuation)).split(None)) for line in document]
    else:
        raise ValueError("Invalid input type for simple_tokenizer")

# Returns integer Term Count for a document
# tc = count number of term occurrences in document
def term_count(term, document_tokens):
    # Flatten the list of lists into a single list
    flat_tokens = [token for sublist in document_tokens for token in sublist]
    count = flat_tokens.count(term)
    return count

# Returns integer with total number of tokens in a document
# toc = count number of tokens in a document
def token_count(document_tokens):
    return len(document_tokens)

# Returns float term frequency (TF),
# normalized for document size
# tf = term count / token count
def term_frequency(term, document_tokens):
    return term_count(term, document_tokens) / float(token_count(document_tokens))

# Returns the number of documents containing the term
# from a list of document tokens
def nr_docs_with_term(term, document_tokens_list):
    nr = 0
    for document_tokens in document_tokens_list:
        if term_count(term, document_tokens) > 0:
            nr += 1
    return nr

# Returns the float Inverse Document Frequency (IDF)
# normalized to reduce non-unique/common words that appear in many documents
def inverse_document_frequency(term, total_docs, term_doc_counts, idf_values):
    if term in idf_values:
        return idf_values[term]
    else:
        nr_docs_with_term_value = term_doc_counts.get(term, 0)
        document_frequency = nr_docs_with_term_value / total_docs
        idf_value = log(total_docs / (1 + document_frequency)) 
        idf_values[term] = idf_value
        return idf_value


# Returns the float Term Frequency - Inverse Document Frequency or tf-idf
def tf_idf(term, document_tokens, document_tokens_list, idf_values):
    tf = term_frequency(term, document_tokens)
    idf = inverse_document_frequency(term, len(document_tokens_list), term_doc_counts, idf_values)
    return tf * idf
# Function to split content into documents every 10 lines

def split_into_documents(content, lines_per_document=10):
    lines = content.split('\n')
    return [lines[i:i + lines_per_document] for i in range(0, len(lines), lines_per_document)]

def calculate_document_scores(query, document_tokens_list, idf_values, total_docs, term_doc_counts):
    document_scores = []
    query_terms = set(query.split())
    for i, document_tokens in enumerate(document_tokens_list, start=1):
        common_terms = query_terms.intersection(set(token for sublist in document_tokens for token in sublist))
        document_score = sum(term_frequency(term, document_tokens) * inverse_document_frequency(term, total_docs, term_doc_counts, idf_values) for term in common_terms)
        document_scores.append((i, document_score))
    return document_scores


# Print the top 5 documents with their TF-IDF scores
def print_top_documents(document_scores, top_n=5):
    sorted_documents = sorted(document_scores, key=lambda x: x[1], reverse=True)[:top_n]
    print("\nTop Documents:")
    for rank, (document_idx, score) in enumerate(sorted_documents, start=1):
        print(f"Rank {rank}: Document {document_idx} - Total TF-IDF Score: {score}")
def tf_idf_report(query, document_tokens_list, idf_values, total_docs, term_doc_counts):
    print("Query:", query)
    print("Number of documents:", len(document_tokens_list))

    # Tokenize the query
    query_tokens = simple_tokenizer(query)

    document_scores = calculate_document_scores(" ".join(query_tokens), document_tokens_list, idf_values, total_docs, term_doc_counts)

    sorted_documents = sorted(document_scores, key=lambda x: x[1], reverse=True)[:5]

    for i, (document_idx, score) in enumerate(sorted_documents, start=1):
        document_tokens = document_tokens_list[document_idx - 1]  # Adjust index to start from 0

        print("\nDocument", document_idx, "Report:")
        print("First 5 document tokens:", document_tokens[:5])
        print("Token count in document:", token_count(document_tokens))

        document_score = 0  # Initialize document score
        for term in query_tokens:
            print("\nTerm:", term)
            print("Term count in document:", term_count(term, document_tokens))
            print("TF:\t\t", term_frequency(term, document_tokens))
            print("IDF:\t\t", inverse_document_frequency(term, total_docs, term_doc_counts, idf_values))
            tfidf_score = tf_idf(term, document_tokens, document_tokens_list, idf_values)
            print("TF--IDF:\t", tfidf_score)
            document_score += tfidf_score  # Accumulate the TF-IDF score for this term

        print("Total TF-IDF Score for Document", document_idx, ":", document_score)  # Print total TF-IDF score for the document

    print_top_documents(document_scores)




start_time = time.time()
# Read content from the text file
with open("pile_val_pubmedabstract.txt", "r") as file:
    #content = ''.join(file.readlines()[:20000])
    content = file.read()

# Split content into documents every 10 lines
lines_per_document = 20
document_tokens_list = [simple_tokenizer(document) for document in split_into_documents(content, lines_per_document)]

# Simple sample usage with a query string
#query = "No compensation is announced for other victims"
query =  "Effect of sleep quality on memory, executive function, and language performance in patients"
# for term in query.split():
#     term_counts = [term_count(term, document_tokens) for document_tokens in document_tokens_list]
#     print(f"Term: {term}, Counts: {term_counts}")
# Count term occurrences in documents
query = query.lower()

term_doc_counts = {}
for document_tokens in document_tokens_list:
    unique_tokens = set(token for sublist in document_tokens for token in sublist)
    for token in unique_tokens:
        term_doc_counts[token] = term_doc_counts.get(token, 0) + 1

total_docs = len(document_tokens_list)
idf_values = {}  # Initialize IDF values dictionary
tf_idf_report(query, document_tokens_list, idf_values, total_docs, term_doc_counts)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")