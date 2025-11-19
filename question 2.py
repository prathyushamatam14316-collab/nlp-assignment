
"""
q1_char_rnn.py
Character-level RNN language model (TensorFlow / Keras)
- Embedding -> LSTM/GRU -> Dense -> Softmax
- Teacher forcing is implemented by training on input sequences and predicting next character at each timestep
- Saves loss curve, sample generations at temperatures 0.7, 1.0, 1.2
Usage:
    python q1_char_rnn.py --text_file optional_text.txt --rnn_type lstm --epochs 10
Requirements:
    pip install tensorflow matplotlib numpy
Outputs:
    ./q1_outputs/loss_curve.png
    ./q1_outputs/sample_tau0.7.txt (and other temps)
"""

import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def build_dataset(text, seq_len=80, step=1):
    chars = sorted(list(set(text)))
    char2idx = {c:i for i,c in enumerate(chars)}
    idx2char = {i:c for c,i in char2idx.items()}
    data = [char2idx[c] for c in text]
    inputs, targets = [], []
    for i in range(0, len(data)-seq_len, step):
        inputs.append(data[i:i+seq_len])
        targets.append(data[i+1:i+seq_len+1])
    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32), char2idx, idx2char

def build_model(vocab_size, embed_dim=64, hidden=128, rnn_type='lstm', seq_len=80):
    inp = layers.Input(shape=(seq_len,), dtype='int32')
    x = layers.Embedding(vocab_size, embed_dim, input_length=seq_len)(inp)
    if rnn_type=='lstm':
        x = layers.LSTM(hidden, return_sequences=True)(x)
    elif rnn_type=='gru':
        x = layers.GRU(hidden, return_sequences=True)(x)
    else:
        x = layers.SimpleRNN(hidden, activation='tanh', return_sequences=True)(x)
    out = layers.Dense(vocab_size)(x)   # logits per timestep
    model = models.Model(inp, out)
    return model

def sample_text(model, start_str, dataset_chars, idx2char, char2idx, length=300, temperature=1.0):
    # start_str can be multiple chars to prime the model
    seq = [char2idx.get(c, 0) for c in start_str]
    seq = seq[-model.input_shape[1]:]  # ensure length <= seq_len
    generated = start_str
    for _ in range(length):
        x = np.array([seq], dtype=np.int32)
        logits = model.predict(x, verbose=0)[0, -1]  # last timestep logits
        logits = logits.astype('float64') / max(1e-8, temperature)
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        next_idx = np.random.choice(len(probs), p=probs)
        generated += idx2char[next_idx]
        seq = seq[1:] + [next_idx] if len(seq) >= model.input_shape[1] else seq + [next_idx]
    return generated

def train(text, seq_len=80, hidden=128, embed=64, rnn_type='lstm', batch=64, epochs=8, lr=1e-3, out='q1_outputs'):
    ensure_dir(out)
    X, Y, char2idx, idx2char = build_dataset(text, seq_len=seq_len)
    vocab_size = len(char2idx)
    print(f'vocab_size={vocab_size}, sequences={len(X)}')

    # build model
    model = build_model(vocab_size, embed_dim=embed, hidden=hidden, rnn_type=rnn_type, seq_len=seq_len)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True))
    # split
    split = int(0.9 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:] if split < len(X) else (X[:1], Y[:1])

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=batch, epochs=epochs)

    # save loss curve
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out, 'loss_curve.png'))

    # samples
    for tau in [0.7, 1.0, 1.2]:
        sample = sample_text(model, start_str='h', dataset_chars=char2idx, idx2char=idx2char, char2idx=char2idx, length=300, temperature=tau)
        with open(os.path.join(out, f'sample_tau{tau}.txt'), 'w', encoding='utf-8') as f:
            f.write(sample)

    # brief reflection (saved)
    reflection = (
        "Reflection:\\n"
        "Increasing sequence length usually helps the model capture longer context but increases memory and makes training slower.\\n"
        "Larger hidden size improves capacity and often reduces training loss, but can overfit and is slower to train.\\n"
        "Higher temperature increases randomness in sampling (more diverse but less coherent), lower temperature makes outputs conservative and repetitive.\\n"
    )
    with open(os.path.join(out, 'reflection.txt'), 'w', encoding='utf-8') as f:
        f.write(reflection)

    # save model
    model.save(os.path.join(out, 'char_rnn_model'))

    print('Training complete. Outputs in', out)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=80)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--embed', type=int, default=64)
    parser.add_argument('--rnn_type', type=str, default='lstm')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', type=str, default='q1_outputs')
    args = parser.parse_args()

    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # toy corpus
        text = '\\n'.join([
            'hello world',
            'hello help',
            'hello helios',
            'help me please',
            'hero here',
            'helicopter hover',
            'heap of words',
            'hello hello hello'
        ] * 200)  # repeated to create some data

    train(text, seq_len=args.seq_len, hidden=args.hidden, embed=args.embed, rnn_type=args.rnn_type,
          batch=args.batch, epochs=args.epochs, lr=args.lr, out=args.out)
