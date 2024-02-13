def extract_atomic_facts_from_file(filename, ignore_keywords=[]):
    atomic_facts = []
    with open(filename, 'r') as file:
        for line in file:
            # Check if any of the ignore keywords are present in the line
            if any(keyword.lower() in line.lower() for keyword in ignore_keywords):
                continue  # Skip this line if any ignore keyword is found
            # Assuming each line contains one atomic fact
            atomic_facts.append(line.strip())
    return atomic_facts

# Example usage:
filename = "pile_val_wikipedia.txt"  # Replace "your_file.txt" with the path to your file
ignore_keywords = ["References", "External links", "Category", "Sources"]  # Add keywords to ignore
atomic_facts = extract_atomic_facts_from_file(filename, ignore_keywords)

# Print atomic facts in a more readable way
print("Atomic Facts:")
for fact in atomic_facts:
    print("-", fact)
