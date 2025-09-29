# Bigram Language Model

## Author
- **Name:** Prathyusha Matam  
- **ID:** 700776555  

## Project Description
This project implements a simple **Bigram Language Model** using Maximum Likelihood Estimation (MLE).  
It calculates unigram and bigram counts from a training corpus and estimates sentence probabilities.

## Files
- `bigram_model.py`: Python program that builds the bigram model, computes probabilities, and tests sentences.

## Training Corpus
The model is trained on the following small dataset:

```
<s> I love NLP </s>
<s> I love deep learning </s>
<s> deep learning is fun </s>
```

## How It Works
1. Reads the training corpus (hardcoded in the script).  
2. Computes unigram and bigram counts.  
3. Estimates bigram probabilities using MLE:  
   \( P(w_i | w_{i-1}) = C(w_{i-1}, w_i) / C(w_{i-1}) \)  
4. Implements a function to calculate the probability of any given sentence.  
5. Tests on two example sentences and prints which one the model prefers.

## How to Run
1. Make sure you have Python 3 installed.  
2. Run the program:

```bash
python bigram_model.py
```

## Expected Output
The program prints probabilities for:

- `<s> I love NLP </s>`  
- `<s> I love deep learning </s>`  

And tells you which sentence is more probable under the model.
