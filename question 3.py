
"""
q3_attention.py
Scaled dot-product attention implementation and stability checks using TensorFlow
Requirements:
    pip install tensorflow numpy
Usage:
    python q3_attention.py
Outputs printed to console and saved to /mnt/data/q3_attention_outputs.txt
"""

import numpy as np
import tensorflow as tf
import os

out_dir = '/mnt/data'
os.makedirs(out_dir, exist_ok=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scores = tf.matmul(Q, K, transpose_b=True)  # (..., seq_q, seq_k)
    scores_scaled = scores / tf.sqrt(dk)
    if mask is not None:
        scores_scaled += (mask * -1e9)
    weights = tf.nn.softmax(scores_scaled, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights, scores, scores_scaled

def main():
    # random test tensors (batch=1, heads=1, seq=4, d_k=8)
    np.random.seed(0)
    Q = tf.constant(np.random.randn(1,4,8).astype(np.float32))
    K = tf.constant(np.random.randn(1,4,8).astype(np.float32))
    V = tf.constant(np.random.randn(1,4,8).astype(np.float32))

    out, weights, raw_scores, scaled_scores = scaled_dot_product_attention(Q, K, V)

    # convert to numpy for printing
    raw = raw_scores.numpy()[0]
    scaled = scaled_scores.numpy()[0]
    w = weights.numpy()[0]
    out_np = out.numpy()[0]

    report = []
    report.append('Raw scores (QK^T) before scaling:') 
    report.append(str(np.round(raw, 4)))
    report.append('\\nScaled scores (divided by sqrt(dk)):')
    report.append(str(np.round(scaled, 4)))
    report.append('\\nAttention weights (softmax of scaled scores):')
    report.append(str(np.round(w, 4)))
    report.append('\\nOutput vectors:')
    report.append(str(np.round(out_np, 4)))

    # softmax stability check: compare softmax of raw vs scaled
    def softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    soft_raw = softmax(raw)
    soft_scaled = softmax(scaled)
    report.append('\\nSoftmax on raw scores (unstable if values large):')
    report.append(str(np.round(soft_raw, 4)))
    report.append('\\nSoftmax on scaled scores:')
    report.append(str(np.round(soft_scaled, 4)))

    out_file = os.path.join(out_dir, 'q3_attention_outputs.txt')
    with open(out_file, 'w') as f:
        f.write('\\n'.join(report))

    print('Saved attention report to', out_file)
    print('\\n'.join(report))

if __name__ == '__main__':
    main()
