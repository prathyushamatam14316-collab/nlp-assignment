# Tokenization Assignment

**Name:** Prathyusha Matam  
**ID:** 700776555  

## Project Description

This project demonstrates different approaches to **tokenization** using Python and the **NLTK library**.  
The goal is to understand how naïve, manual, and tool-based tokenization differ, and how **multiword expressions (MWEs)** and language features affect tokenization.

The project is implemented in the file `tokenization_assignment.py`.

---

## Code Explanation

### 1. Naïve Tokenization
- The code first tokenizes a paragraph using simple space-based splitting with `split(" ")`.
- This approach separates words but fails with punctuation and contractions.  
  - Example: `"It's"` remains one token instead of `"It"` and `"'s"`.
  - Punctuation like `"world."` remains attached to the word.

### 2. Manual Tokenization
- A corrected token list is manually created to handle punctuation and contractions properly.
- For example:

  - `"It's"` → `"It", "'s"`

  - `"world."` → `"world", "."`

- This acts as a **gold standard** for comparison with NLTK.

### 3. Comparison with NLTK
- The code uses **NLTK’s `word_tokenize()`** to tokenize the paragraph.

- Then, it compares the tool output with the manually corrected tokens using **set differences**.

- This shows which tokens differ between the tool’s result and the manual correction.

- NLTK usually handles punctuation and contractions well, but the comparison highlights any mismatches.

### 4. Multiword Expressions (MWEs)
- The script lists three examples of MWEs:

  - `New York City` → A place name acting as a single unit.

  - `kick the bucket` → Idiom meaning *to die*, cannot be taken literally.

  - `machine learning` → A technical concept, not just two independent words.

- These should ideally be treated as single tokens in NLP tasks.

### 5. Reflection
- The script prints a reflection paragraph covering:

  - Challenges in English tokenization (contractions, punctuation).

  - Comparison with morphologically rich languages.

  - The complexity added by MWEs.

  - The usefulness and limitations of NLTK for tokenization.


---

## Requirements

- Python 3.x  
- NLTK library (`pip install nltk`)  

The script automatically downloads the required NLTK resource (`punkt`).

---

## How to Run

1. Save the code as `tokenization_assignment.py`  
2. Run the script in your terminal or IDE with:

```bash
python tokenization_assignment.py
```

---

## Expected Output

When you run the script, you will see:

1. **Naïve tokens** (space-based splitting)  
2. **Manual corrected tokens**  
3. **NLTK tokens**  
4. **Differences between manual and NLTK tokens**  
5. **List of MWEs with explanations**  
6. **Reflection paragraph**  

---

This project highlights the importance of accurate tokenization and shows how NLP tools like NLTK compare with manual linguistic analysis.
