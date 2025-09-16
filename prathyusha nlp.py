
"""
Tokenization Demo for Q2
This version uses only Python's built-in modules.
"""

import re

# -----------------------------------------------------------
# Sample paragraph (3–4 sentences)
# -----------------------------------------------------------
text = (
    "New York is a vibrant city. "
    "It's packed with culture, music, and endless opportunities! "
    "People often call it the city that never sleeps."
)

print("=== Original Paragraph ===")
print(text)

# -----------------------------------------------------------
# 1A. Naïve space-based tokenization
# -----------------------------------------------------------
naive_tokens = text.split()
print("\n--- 1A. Naïve Tokens ---")
print(naive_tokens)

# -----------------------------------------------------------
# 1B. Manual correction
#    - Detach punctuation
#    - Split contractions (It's -> [It, 's])
# -----------------------------------------------------------
manual_tokens = []
for tok in naive_tokens:
    # Regex: words, apostrophe groups, or punctuation
    pieces = re.findall(r"[A-Za-z0-9]+|'\w+|[.,!]", tok)
    manual_tokens.extend(pieces)

print("\n--- 1B. Manually Corrected Tokens ---")
print(manual_tokens)

# Show differences
extra_naive  = set(naive_tokens) - set(manual_tokens)
extra_manual = set(manual_tokens) - set(naive_tokens)
print("\nDifferences:")
print("Only in naive :", extra_naive)
print("Only in manual:", extra_manual)

# -----------------------------------------------------------
# 3. Multiword Expressions (MWEs)
# -----------------------------------------------------------
mwes = [
    "New York",                 # proper noun
    "city that never sleeps",   # idiom
    "endless opportunities"     # collocation
]
print("\n--- 3. Multiword Expressions ---")
for phrase in mwes:
    print(f"- {phrase}: represents one concept despite multiple words.")

# -----------------------------------------------------------
# 4. Reflection
# -----------------------------------------------------------
reflection = (
    "Tokenizing English is usually simple, but punctuation and contractions "
    "like It's require careful handling. Separating apostrophes while "
    "preserving meaning can be tricky. Languages with richer morphology would "
    "be even harder. Multiword expressions remain challenging because they look "
    "like several tokens but represent one concept. Automatic tools help, yet "
    "human judgment is still valuable."
)
print("\n--- 4. Reflection ---")
print(reflection)
