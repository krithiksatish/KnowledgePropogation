import wikipedia_edit_scrape_tool
from wikipedia_edit_scrape_tool.sentence_simlarity import SentenceSimilarityCalculator

similarity_calculator = SentenceSimilarityCalculator()

sentences = [
    ("Wikipedia is a free online encyclopedia, created and edited by volunteers around the world.",
     "Wikipedia is a freely accessible online encyclopedia, created and edited collaboratively by volunteers worldwide."),
    ("The capital of France is Paris.",
     "Paris is the capital city of France."),
    ("The Earth revolves around the Sun.",
     "The Sun is orbited by the Earth."),
    ("Albert Einstein was a theoretical physicist.",
     "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."),
    ("The Mona Lisa is a famous painting by Leonardo da Vinci.",
     "The Mona Lisa, a masterpiece of Renaissance art, is one of the most famous paintings in the world."),
    ("The Industrial Revolution marked a significant turning point in history.",
     "The Industrial Revolution was a period of major changes in the economy, technology, and society."),
    ("The Great Wall of China is a UNESCO World Heritage Site.",
     "The Great Wall of China, one of the most impressive architectural feats in history, is a UNESCO World Heritage Site."),
    ("The human brain is composed of billions of neurons.",
     "The brain, a complex organ, consists of billions of specialized cells called neurons."),
    ("Mount Everest is the highest mountain in the world.",
     "Mount Everest, the tallest peak on Earth, is located in the Himalayas."),
    ("Rome is the capital city of Italy.",
     "The capital of Italy is Rome.")
]

additional_sentences = [
    ("The cat sat on the mat.",
     "A feline was positioned on the straw mat."),
    ("Pizza is a popular dish.",
     "Pizza is widely enjoyed, often topped with ingredients like pepperoni and extra cheese."),
    ("The weather is pleasant today.",
     "Today's weather is characterized by clear skies and a gentle breeze."),
    ("She walked to the store.",
     "She proceeded on foot to the nearby store."),
    ("The meeting was highly productive.",
     "The meeting resulted in significant progress, culminating in the development of a comprehensive plan."),
    ("Grocery shopping is necessary.",
     "Procuring groceries is imperative due to an empty refrigerator."),
    ("He is recognized as a talented musician.",
     "He is acknowledged for his musical talent and skillful performances."),
    ("The book was engaging.",
     "The book held readers' attention with its intriguing plot twists."),
    ("They embarked on a hiking trip.",
     "They undertook a hiking expedition, equipped with essential gear such as sturdy boots and backpacks."),
    ("She wore an elegant dress.",
     "She donned an elegant gown, embellished with intricate lace and shimmering beads.")
]

for edit_pair in sentences:
    print("Original: " + edit_pair[0])
    print("Edited: " + edit_pair[1])
    print("Similarity: " + str(similarity_calculator.is_significant_edit(edit_pair[0], edit_pair[1])))

    print()

for edit_pair in additional_sentences:
    print("Original: " + edit_pair[0])
    print("Edited: " + edit_pair[1])
    print("Similarity: " + str(similarity_calculator.is_significant_edit(edit_pair[0], edit_pair[1])))

    print()


comparison_sentences = [
    ("Deep learning is a subset of machine learning methods based on artificial neural networks with representation learning.",
    "Deep learning is a type of machine learning methods based on artificial neural networks with representation learning.",
    "Deep learning, which involves artificial neural networks, is a subset of machine learning methods with representation learning.",
    "In the landscape of machine learning, deep learning emerges as a unique methodology, characterized by its heavy reliance on artificial neural networks and representation learning, which enables the extraction of intricate patterns from vast datasets.",
    "Artificial neural networks, pivotal to the field of deep learning, undergo representation learning, an essential process that distinguishes this subset within machine learning, revolutionizing various fields like healthcare, finance, and autonomous systems."),
    (
    "Mental illness is thought to be highly prevalent among homeless populations, though access to proper diagnoses is limited.",
    "It's widely believed that mental illness is disproportionately common among individuals experiencing homelessness, yet obtaining accurate diagnoses remains challenging.",
    "Despite widespread assumptions about the high prevalence of mental illness among the homeless, the actual accessibility of accurate diagnoses remains a considerable challenge",
    "The perception of a significant presence of mental illness among the homeless is prevalent, yet the reality of accessing precise diagnostic measures remains substantially restricted.",
    "While mental health issues are commonly associated with homelessness, the limited availability of accurate diagnoses exacerbates the situation, leading to a lack of appropriate support.")
]


for comparison in comparison_sentences:
    original_sentence = comparison[0]

    for i in range(1, len(comparison)):
        print("Original: " + original_sentence)
        print("Edited: " + comparison[i])
        print(str(similarity_calculator.is_signifcant_edit(original_sentence, comparison[i])))

    print()


# Testing edit distance
# edit_distance_sentences = [
#     ("the quick brown fox", "the smart quick fox")
# ]

# for edit_pair in edit_distance_sentences:
#     print(edit_pair[0])
#     print(str(similarity_calculator.edit_distance(edit_pair[0], edit_pair[1])))



# Testing multi-sentence similarity

# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from nltk.tokenize import sent_tokenize
# import numpy as np
# import torch

# print ("Hello World")

# # Load a pre-trained sentence transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# print("Model loaded")

# # Define two excerpts
# excerpt1 = "Charles Tappert writes that [[Frank Rosenblatt]] developed and explored all of the basic ingredients of the deep learning systems of today, referring to Rosenblatt's 1962 book which introduced a [[multilayer perceptron]] (MLP) with 3 layers: an input layer, a hidden layer with randomized weights that did not learn, and an output layer. However, since only the output layer had learning connections, this was not yet deep learning. It was what later was called an [[extreme learning machine]].."
# excerpt2 = "Charles Tappert writes that [[Frank Rosenblatt]] developed and explored all of the basic ingredients of the deep learning systems of today, referring to Rosenblatt's 1962 book which introduced [[multilayer perceptron]] (MLP) with 3 layers: an input layer, a hidden layer with randomized weights that did not learn, and an output layer. It also introduced variants, including a version with four-layer perceptrons where the last two layers have learned weights (and thus a proper multilayer perceptron)."
# print("Excerpts defined")

# # Split excerpts into sentences
# sentences1 = sent_tokenize(excerpt1)
# sentences2 = sent_tokenize(excerpt2)
# print("Sentences split")

# # Compute embeddings for all sentences
# embeddings1 = model.encode(sentences1)
# embeddings2 = model.encode(sentences2)
# print("Embeddings computed")

# # Compute cosine similarity between each pair of sentences and find the maximum similarity for each sentence in excerpt1
# max_similarities = []
# for emb1 in embeddings1:
#     similarities = [cosine_similarity([emb1], [emb2])[0][0] for emb2 in embeddings2]
#     max_similarities.append(max(similarities))
# print("Similarities computed")

# # Calculate the average or minimum of the maximum similarities
# print (max_similarities)
# avg_similarity = np.mean(max_similarities)
# min_similarity = np.min(max_similarities)
# max_similarity = np.max(max_similarities)
# print("Statistics computed")

# print(f'Average similarity: {avg_similarity}')
# print(f'Minimum similarity: {min_similarity}')
# print(f'Maximum similarity: {max_similarity}')