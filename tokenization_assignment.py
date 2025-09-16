import nltk
from nltk.tokenize import word_tokenize

# Download resources if not already available
nltk.download('punkt')

# ---------------------------
# 1. Tokenize a Paragraph
# ---------------------------
paragraph = (
    "Artificial intelligence is transforming the world. "
    "It's used in healthcare, finance, and education. "
    "However, many people worry about its ethical impact."
)

# Naïve space-based tokenization
naive_tokens = paragraph.split(" ")

# Manual correction: handling punctuation, clitics (e.g., It's → It + 's)
manual_tokens = [
    "Artificial", "intelligence", "is", "transforming", "the", "world", ".",
    "It", "'s", "used", "in", "healthcare", ",", "finance", ",", "and", "education", ".",
    "However", ",", "many", "people", "worry", "about", "its", "ethical", "impact", "."
]

print("--- 1. Naïve Tokenization ---")
print(naive_tokens)
print("\n--- 1. Manual Corrected Tokens ---")
print(manual_tokens)

# ---------------------------
# 2. Compare with a Tool (NLTK)
# ---------------------------
nltk_tokens = word_tokenize(paragraph)

print("\n--- 2. NLTK Tokens ---")
print(nltk_tokens)

# Differences (manual vs. NLTK)
diff_nltk = set(manual_tokens).symmetric_difference(set(nltk_tokens))

print("\n--- Differences (Manual vs. NLTK) ---")
print(diff_nltk)

# ---------------------------
# 3. Multiword Expressions (MWEs)
# ---------------------------
MWEs = [
    ("New York City", "A fixed place name that functions as a single unit."),
    ("kick the bucket", "An idiom meaning 'to die' – cannot be understood word by word."),
    ("machine learning", "A technical term that represents a concept, not individual words."),
]

print("\n--- 3. Multiword Expressions ---")
for mwe, explanation in MWEs:
    print(f"MWE: {mwe} -> {explanation}")

# ---------------------------
# 4. Reflection
# ---------------------------
reflection = (
    "The hardest part of tokenization in English is handling contractions (like It's → It + 's) and punctuation. "
    "Compared to English, some languages with richer morphology (like Turkish or Finnish) are even more complex because suffixes carry grammatical meaning. "
    "Punctuation often attaches directly to words, which makes naïve space-based tokenization inaccurate. "
    "Multiword expressions add another challenge since they are semantically single units but consist of multiple words. "
    "Tools like NLTK handle most cases well, but they may still split or misinterpret MWEs. "
    "Overall, tokenization shows how language structure deeply affects NLP preprocessing."
)

print("\n--- 4. Reflection ---")
print(reflection)
