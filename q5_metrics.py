import numpy as np

# Confusion matrix (rows = predicted, cols = actual)
# System \\ Gold   Cat   Dog   Rabbit
confusion_matrix = np.array([
    [5, 10, 5],    # Predicted Cat
    [15, 20, 10],  # Predicted Dog
    [0, 15, 10]    # Predicted Rabbit
])

classes = ["Cat", "Dog", "Rabbit"]

# ---- 1. Per-Class Precision & Recall ----
def compute_per_class_metrics(cm, class_names):
    precisions, recalls = {}, {}
    for i, cls in enumerate(class_names):
        tp = cm[i, i]                           # True positives
        fp = cm[i, :].sum() - tp                # False positives
        fn = cm[:, i].sum() - tp                # False negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions[cls] = precision
        recalls[cls] = recall
    return precisions, recalls

# ---- 2. Macro and Micro Averaging ----
def compute_macro_metrics(precisions, recalls):
    macro_precision = np.mean(list(precisions.values()))
    macro_recall = np.mean(list(recalls.values()))
    return macro_precision, macro_recall

def compute_micro_metrics(cm):
    tp = np.trace(cm)               # sum of diagonal
    total = cm.sum()
    micro_precision = tp / total if total > 0 else 0.0
    micro_recall = micro_precision  # same in multi-class setting
    return micro_precision, micro_recall

# ---- Run calculations ----
precisions, recalls = compute_per_class_metrics(confusion_matrix, classes)
macro_p, macro_r = compute_macro_metrics(precisions, recalls)
micro_p, micro_r = compute_micro_metrics(confusion_matrix)

# ---- Print results ----
print("Per-Class Precision and Recall:")
for cls in classes:
    print(f"  {cls}: Precision={precisions[cls]:.3f}, Recall={recalls[cls]:.3f}")

print("\nMacro-Averaged Metrics:")
print(f"  Precision={macro_p:.3f}, Recall={macro_r:.3f}")

print("\nMicro-Averaged Metrics:")
print(f"  Precision={micro_p:.3f}, Recall={micro_r:.3f}")

print("\nNote:")
print("  - Macro averaging gives equal weight to each class, regardless of size.")
print("  - Micro averaging aggregates over all decisions, so larger classes dominate.")
