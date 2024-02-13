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
    print("Similarity: " + str(similarity_calculator.calculate_similarity(edit_pair[0], edit_pair[1])))

    print()

print()

for edit_pair in additional_sentences:
    print("Original: " + edit_pair[0])
    print("Edited: " + edit_pair[1])
    print("Similarity: " + str(similarity_calculator.calculate_similarity(edit_pair[0], edit_pair[1])))

    print()
