# Homework 2 - Q5 Evaluation Metrics

**Student Name:** Prathyusha Matam  
**Student ID:** 700776555  

## Description
This project solves Question 5 from Homework 2 (CS5760 Natural Language Processing).  
The task is to compute evaluation metrics from a multi-class confusion matrix for the classes **Cat, Dog, Rabbit**.

The system classified 90 animals, and the confusion matrix is:

| System \ Gold | Cat | Dog | Rabbit |
|----------------|-----|-----|--------|
| **Cat**        |  5  | 10  |   5    |
| **Dog**        | 15  | 20  |  10    |
| **Rabbit**     |  0  | 15  |  10    |

The Python script (`q5_metrics.py`) performs the following:

1. **Per-Class Metrics**  
   - Computes precision and recall for each class (Cat, Dog, Rabbit).

2. **Macro vs. Micro Averaging**  
   - Computes macro-averaged precision and recall.  
   - Computes micro-averaged precision and recall.  
   - Prints an explanation of the difference between macro and micro averaging.

3. **Output**  
   - Displays results in a clear and formatted way.  
   - Provides interpretation notes for macro and micro metrics.

## Files
- `q5_metrics.py` → Python implementation of evaluation metrics.  
- `README.md` → Documentation about the code, project, and execution details.

## How to Run
Make sure you have Python 3 installed. Run the script using:

```bash
python q5_metrics.py
```

## Expected Output
The script prints:
- Precision & Recall for each class (Cat, Dog, Rabbit)
- Macro-averaged Precision & Recall
- Micro-averaged Precision & Recall
- Explanation of macro vs micro interpretation

Example snippet of output:

```
Per-Class Precision and Recall:
  Cat: Precision=0.xxx, Recall=0.xxx
  Dog: Precision=0.xxx, Recall=0.xxx
  Rabbit: Precision=0.xxx, Recall=0.xxx

Macro-Averaged Metrics:
  Precision=0.xxx, Recall=0.xxx

Micro-Averaged Metrics:
  Precision=0.xxx, Recall=0.xxx

Note:
  - Macro averaging gives equal weight to each class, regardless of size.
  - Micro averaging aggregates over all decisions, so larger classes dominate.
```

This completes the implementation for Homework 2 - Question 5.
