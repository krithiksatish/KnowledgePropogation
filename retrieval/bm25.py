# go through pile_val_pubmedabstract and treat every 10 lines as a new document for ranking purposes
# in the wikipedia file, ignore the headings and extract atomic facts from every line 

# use atomic fact response from chatGPT to do retrieval lookup in pile_val_abstract on diff retrieval types 
# in table list overall scoring 

# go through pile_val_pubmedabstract and treat every 10 lines as a new document for ranking purposes
# in the wikipedia file, ignore the headings and extract atomic facts from every line 

# use atomic fact response from chatGPT to do retrieval lookup in pile_val_abstract on diff retrieval types 
# in table list overall scoring 

from math import log
import string
import time

def simple_tokenizer(document):
    if isinstance(document, str):
        translator = str.maketrans("", "", string.punctuation)
        return document.lower().translate(translator).split()  # removed None from split
    elif isinstance(document, list):
        # Assuming each element of the list is a line in the document
        return [line.lower().translate(str.maketrans("", "", string.punctuation)).split() for line in document]
    else:
        raise ValueError("Invalid input type for simple_tokenizer")

def term_count(term, document_tokens):
    flat_tokens = [token for sublist in document_tokens for token in sublist]
    count = flat_tokens.count(term)
    return count

def token_count(document_tokens):
    return len(document_tokens)

def term_frequency(term, document_tokens):
    return term_count(term, document_tokens) / float(token_count(document_tokens))

def nr_docs_with_term(term, document_tokens_list):
    nr = 0
    for document_tokens in document_tokens_list:
        if term_count(term, document_tokens) > 0:
            nr += 1
    return nr

def inverse_document_frequency(term, total_docs, term_doc_counts, idf_values):
    if term in idf_values:
        return idf_values[term]
    else:
        nr_docs_with_term_value = term_doc_counts.get(term, 0)
        document_frequency = nr_docs_with_term_value / total_docs
        idf_value = log((total_docs - document_frequency + 0.5) / (document_frequency + 0.5))  # BM25 IDF formula
        idf_values[term] = idf_value
        return idf_value

def tf_idf(term, document_tokens, document_tokens_list, idf_values):
    tf = term_frequency(term, document_tokens)
    idf = inverse_document_frequency(term, len(document_tokens_list), term_doc_counts, idf_values)
    return tf * idf

def split_into_documents(content, lines_per_document=20):
    lines = content.split('\n')
    return [lines[i:i + lines_per_document] for i in range(0, len(lines), lines_per_document)]

def build_regular_index(document_tokens_list):
    regular_index = {}
    for i, document_tokens in enumerate(document_tokens_list, start=1):
        for term_list in document_tokens:  # Iterate over each list of terms in the document
            for term in term_list:
                if term not in regular_index:
                    regular_index[term] = {}
                if i not in regular_index[term]:
                    regular_index[term][i] = 0
                regular_index[term][i] += 1  # Increment term frequency for the document
    return regular_index

def build_inverted_index(regular_index):
    inverted_index = {}
    for term, postings in regular_index.items():
        for doc_id, term_freq in postings.items():
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term].append((doc_id, term_freq))
    return inverted_index


def bm25_term_weight(term, document_tokens, document_tokens_list, term_doc_counts, avg_doc_length, k1=1.5, b=0.75):
    tf = term_frequency(term, document_tokens)
    document_length = len(document_tokens)
    idf = inverse_document_frequency(term, len(document_tokens_list), term_doc_counts, {})  # Use an empty IDF dictionary here
    
    # Document length normalization
    # Normalize term frequency based on document length relative to average document length
    # Adjust the term frequency using the BM25 document length normalization formula
    doc_length_normalization = (1 - b) + b * (document_length / avg_doc_length)
    tf *= doc_length_normalization
    
    return idf * ((tf * (k1 + 1)) / (tf + k1))

def bm25_score(query_terms, document_tokens, inverted_index, idf_values, total_docs, avg_doc_length, term_doc_counts, k1=1.5, b=0.75):
    score = 0
    doc_length = len(document_tokens)
    
    for term in query_terms:
        if term in inverted_index:
            tf = term_frequency(term, document_tokens)
            idf = inverse_document_frequency(term, total_docs, term_doc_counts, idf_values)
            
            # Check if the term exists in the document's inverted index
            if term in inverted_index:
                # Get the term frequency from the inverted index
                postings = inverted_index[term]
                doc_freq = len(postings)  # Number of documents containing the term
                term_doc_freq = term_doc_counts[term]  # Total frequency of the term in all documents
                
                # Calculate the BM25 term weight for the term in the current document
                term_weight = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length)))
                
                # Update the score with the term weight multiplied by the document frequency factor
                score += term_weight * doc_freq / term_doc_freq
    
    return score


def bm25_report(query, document_tokens_list, inverted_index, idf_values, term_doc_counts, avg_doc_length):
    print("Query:", query)
    print("Number of documents:", len(document_tokens_list))

    document_scores = []
    query_terms = query.lower().split()
    for i, document_tokens in enumerate(document_tokens_list, start=1):
        score = bm25_score(query_terms, document_tokens, inverted_index, idf_values, len(document_tokens_list), avg_doc_length, term_doc_counts)
        document_scores.append((i, score))

    sorted_documents = sorted(document_scores, key=lambda x: x[1], reverse=True)[:5]

    for i, (document_idx, score) in enumerate(sorted_documents, start=1):
        document_tokens = document_tokens_list[document_idx - 1]

        print("\nDocument", document_idx, "Report:")
        print("First 5 document tokens:", document_tokens[:5])
        print("Token count in document:", len(document_tokens))
        print("BM25 Score:", score)

start_time = time.time()
with open("pile_val_pubmedabstract.txt", "r") as file:
    content = file.read()

lines_per_document = 20
document_tokens_list = [simple_tokenizer(document) for document in split_into_documents(content, lines_per_document)]

query = "and are of certain changes in the"

term_doc_counts = {}
for document_tokens in document_tokens_list:
    unique_tokens = set(token for sublist in document_tokens for token in sublist)
    for token in unique_tokens:
        term_doc_counts[token] = term_doc_counts.get(token, 0) + 1

regular_index = build_regular_index(document_tokens_list)
inverted_index = build_inverted_index(regular_index)

total_docs = len(document_tokens_list)
idf_values = {}
total_doc_length = sum(len(doc) for doc in document_tokens_list)
avg_doc_length = total_doc_length / len(document_tokens_list)

bm25_report(query, document_tokens_list, inverted_index, idf_values, term_doc_counts, avg_doc_length)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")