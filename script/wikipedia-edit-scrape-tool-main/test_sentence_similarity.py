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

for edit_pair in sentences:
    print("Original: " + edit_pair[0])
    print("Edited: " + edit_pair[1])
    print("Similarity: " + str(similarity_calculator.calculate_similarity(edit_pair[0], edit_pair[1])))

    print()

