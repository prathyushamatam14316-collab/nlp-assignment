from collections import defaultdict

# Training corpus (list of tokenized sentences)
corpus = [
    ["<s>", "I", "love", "NLP", "</s>"],
    ["<s>", "I", "love", "deep", "learning", "</s>"],
    ["<s>", "deep", "learning", "is", "fun", "</s>"]
]

# 1. Compute unigram and bigram counts
unigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)

for sentence in corpus:
    for i in range(len(sentence)):
        unigram_counts[sentence[i]] += 1
        if i < len(sentence) - 1:
            bigram = (sentence[i], sentence[i+1])
            bigram_counts[bigram] += 1

# 2. Function: bigram probability with MLE
def bigram_prob(w1, w2):
    if unigram_counts[w1] > 0:
        return bigram_counts[(w1, w2)] / unigram_counts[w1]
    else:
        return 0

# 3. Function: sentence probability
def sentence_prob(sentence):
    prob = 1.0
    for i in range(len(sentence) - 1):
        prob *= bigram_prob(sentence[i], sentence[i+1])
    return prob

# 4. Test sentences
s1 = ["<s>", "I", "love", "NLP", "</s>"]
s2 = ["<s>", "I", "love", "deep", "learning", "</s>"]

p1 = sentence_prob(s1)
p2 = sentence_prob(s2)

# 5. Print results
print("P(<s> I love NLP </s>) =", p1)
print("P(<s> I love deep learning </s>) =", p2)

if p1 > p2:
    print("Model prefers: <s> I love NLP </s> because it has higher probability.")
else:
    print("Model prefers: <s> I love deep learning </s> because it has higher probability.")
