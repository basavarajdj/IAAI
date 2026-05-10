# NLP Exam Study Guide — CS F429
**Based on:** Notes, past papers (Sem 1 & 2, 2021–2026), cheatsheets, and Jurafsky textbook  
**Professor:** Abhisek Chakrabarty  
**Exam format:** Open book (one A4 cheatsheet allowed)

---

## 🔴 PRIORITY MAP (based on every past paper)

| Topic | Frequency in Past Papers | Typical Marks |
|---|---|---|
| HMM (Forward/Backward/Viterbi) | **Every single exam** | 15–30 marks |
| N-gram LM (counting + smoothing) | **Every exam** | 10–20 marks |
| Naive Bayes (multinomial + binary) | **Every exam** | 10–20 marks |
| Neural Networks (backprop, computation graph) | **Every exam** | 15–25 marks |
| Transformers (attention, architecture) | Recent exams | 10–15 marks |
| BERT / MLM | Recent exams | 10–15 marks |
| LLMs / Post-training | Recent exams | 10 marks |
| IR / RAG | Recent exams | 10–15 marks |
| IBM Translation Model | Sem 1 24-25 | 20 marks |
| LSTM / RNN | Multiple exams | 5–10 marks |

---

## TOPIC 1: N-GRAM LANGUAGE MODELS

### Core Idea
A language model assigns probability to sequences of words. An n-gram model approximates P(w₁...wₙ) using the Markov assumption: each word depends only on the previous n−1 words.

### Key Formulas

**Unigram:**  P(w) = C(w) / N

**Bigram:**  P(wᵢ | wᵢ₋₁) = C(wᵢ₋₁ wᵢ) / C(wᵢ₋₁)

**Trigram:**  P(wᵢ | wᵢ₋₂ wᵢ₋₁) = C(wᵢ₋₂ wᵢ₋₁ wᵢ) / C(wᵢ₋₂ wᵢ₋₁)

**Full sentence probability:**
P(w₁...wₙ) = P(w₁|START) × P(w₂|START, w₁) × ... × P(STOP|wₙ₋₁, wₙ)   ← trigram

### Counting N-grams with START/STOP (⚠️ EXAM FAVOURITE)

For a vocabulary of m words + START + STOP, counting valid n-grams:

**Rules for valid n-gram ⟨W₁, W₂, ..., Wₙ⟩:**
- Wₙ ∈ {STOP ∪ V}  (last word: vocab or STOP)
- W₁ ∈ {START ∪ V}  (first word: vocab or START)
- For i=2 to n−1: Wᵢ can be START only if all Wⱼ (j<i) = START

**Worked Example (from Sem 2 25-26 Midsem & Sem 1 24-25 Midsem):**

Vocabulary V = m words, plus START and STOP. Find distinct valid n-grams.

**Case (a): No START, No STOP**
All n positions: any of m words → mⁿ n-grams

**Case (b): Both START and STOP present**
- Positions where START can appear: START must be a contiguous prefix. If k STARTs at the beginning (k = 0 to n−1), remaining n−k positions fill as follows:
  - Position k+1 (first non-START): cannot be START, cannot be STOP unless it's the last position → (m or STOP)
  - Positions k+2 to n−1: any of m words
  - Position n: STOP or any of m words
- Count by summing over k=0 to n−1 the arrangements:
  - k STARTs at start, then position k+1 ∈ V (m choices), positions k+2 to n−1 ∈ V (mⁿ⁻ᵏ⁻² choices), position n ∈ {V ∪ STOP} (m+1 choices)
  - Plus the case with n STARTs (not valid since last must be STOP/V)

**Shortcut formula (from solutions):**
Total = Σₖ₌₀ⁿ⁻¹ [m × mⁿ⁻ᵏ⁻² × (m+1)] ... work out case by case for small n

### Smoothing

**Add-1 (Laplace) Smoothing:**
- Bigram: P(wᵢ | wᵢ₋₁) = [C(wᵢ₋₁ wᵢ) + 1] / [C(wᵢ₋₁) + |V|]
- Trigram: P(wᵢ | wᵢ₋₂ wᵢ₋₁) = [C(wᵢ₋₂ wᵢ₋₁ wᵢ) + 1] / [C(wᵢ₋₂ wᵢ₋₁) + |V|]

**Add-k:** Replace 1 with k, replace |V| with k|V|

**Linear Interpolation:**
P_smooth(w|uv) = λ₁·P(w|uv) + λ₂·P(w|v) + λ₃·P(w)
- λ₁ + λ₂ + λ₃ = 1, all λ > 0
- λ₁ = C(uv)/[C(uv)+γ], λ₂ = (1−λ₁)·C(v)/[C(v)+γ], λ₃ = 1−λ₁−λ₂

### Perplexity
PP(W) = 2^(−l), where l = (1/M) × log₂ Π P(xᵢ)  
M = total number of tokens. Lower perplexity = better model.

---

## TOPIC 2: NAIVE BAYES CLASSIFIER

### Core Formula
c* = argmax P(c) × Π P(wᵢ|c)

In log space: c* = argmax [log P(c) + Σ log P(wᵢ|c)]

### Multinomial Naive Bayes
**Training:**
- P(c) = (number of docs in class c) / (total docs)
- P(w|c) = [count(w in class c) + 1] / [total words in class c + |V|]  ← Add-1 smoothing
- |V| = union of all words across all classes

**Test:** words NOT in training vocabulary are completely ignored

**Worked Example (Sem 2 25-26 Midsem):**
```
d1: "the movie is not bad" → +
d2: "the food is bad" → −
d3: "the movie is not good" → −
d4: "the food is good" → +

Key words only: "good", "bad"

P(+) = 2/4 = 0.5, P(−) = 2/4 = 0.5
Positive class has: "bad"(d1), "good"(d4) → bad:1, good:1
Negative class has: "bad"(d2), "good"(d3) → bad:1, good:1

|V| = 2 (just "good" and "bad")
P(good|+) = (1+1)/(2+2) = 0.5, P(bad|+) = (1+1)/(2+2) = 0.5
P(good|−) = (1+1)/(2+2) = 0.5, P(bad|−) = (1+1)/(2+2) = 0.5

Test: "The food is not bad"
Key words: "bad"
P(+|doc) ∝ 0.5 × 0.5 = 0.25
P(−|doc) ∝ 0.5 × 0.5 = 0.25 → TIE!

Fix: Negation handling — replace "not bad" with "NOT_bad"
Then "not bad" → positive signal. Re-classify accordingly.
```

### Binary (Binarized) Naive Bayes
- Clip all word counts to 1 (remove duplicates within each document before training)
- During training: for each document, keep only unique words, then pool
- During testing: remove duplicates from test doc too
- Better for sentiment tasks

**Key difference:** In multinomial, a word appearing 5 times counts 5 times. In binary, it counts once per document.

### Precision, Recall, F1 (from 22-23 paper)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × P × R / (P + R)
- For imbalanced classes → use macro-averaged F1

---

## TOPIC 3: HIDDEN MARKOV MODELS (HMM) — ⚠️ HIGHEST PRIORITY

### HMM Components
- **Q** = {q₁...qₙ}: set of states (e.g., POS tags)
- **A** = aᵢⱼ: transition probability P(qⱼ|qᵢ)
- **B** = bⱼ(oₜ): emission probability P(oₜ|qⱼ)
- **π**: initial state distribution (or START state)

### Bigram HMM
P(O, Q) = P(q₁|START) × Π P(qᵢ|qᵢ₋₁) × Π P(oᵢ|qᵢ) × P(STOP|qₙ)

### Trigram HMM
P(y₁...yₙ₊₁, x₁...xₙ) = Π P(yᵢ|yᵢ₋₂ yᵢ₋₁) × Π P(xᵢ|yᵢ)
where y₋₁ = y₀ = START, yₙ₊₁ = STOP

---

### Problem 1: FORWARD ALGORITHM (Likelihood)

**Goal:** Compute P(O|λ) — probability of observation sequence given HMM

**αₜ(j) = P(o₁, o₂, ..., oₜ, qₜ = j | λ)**

**Initialization:** α₁(j) = πⱼ × bⱼ(o₁)  [or: P(j|START) × P(o₁|j)]

**Recursion:** αₜ(j) = [Σᵢ αₜ₋₁(i) × aᵢⱼ] × bⱼ(oₜ)

**Termination:** P(O|λ) = Σⱼ αT(j) × P(STOP|j)

**Worked Example (from EVERY exam — Cricket/Football/Hello HMM):**
```
States: N, V  (START, STOP don't emit)
Vocabulary: Cricket, Football, Hello
Transitions (from the FSM diagram):
  P(N|START)=0.5(?), P(V|START)=0.5(?) — read from diagram
  P(N|N), P(V|N), P(N|V), P(V|V) — read from diagram
  P(STOP|N), P(STOP|V) — read from diagram

Emissions:
  P(Cricket|N)=0.2, P(Football|N)=0.4, P(Hello|N)=0.4
  P(Cricket|V)=0.5, P(Football|V)=0.4, P(Hello|V)=0.1

Observation: Hello Cricket Hello

Step 1 (t=1, o₁=Hello):
  α₁(N) = P(N|START) × P(Hello|N)
  α₁(V) = P(V|START) × P(Hello|V)

Step 2 (t=2, o₂=Cricket):
  α₂(N) = [α₁(N)×P(N|N) + α₁(V)×P(N|V)] × P(Cricket|N)
  α₂(V) = [α₁(N)×P(V|N) + α₁(V)×P(V|V)] × P(Cricket|V)

Step 3 (t=3, o₃=Hello):
  α₃(N) = [α₂(N)×P(N|N) + α₂(V)×P(N|V)] × P(Hello|N)
  α₃(V) = [α₂(N)×P(V|N) + α₂(V)×P(V|V)] × P(Hello|V)

Termination:
  P(O|λ) = α₃(N)×P(STOP|N) + α₃(V)×P(STOP|V) = 0.002193
```

---

### Problem 2: BACKWARD ALGORITHM

**βₜ(i) = P(oₜ₊₁, ..., oT | qₜ = i, λ)**

**Initialization:** βT(i) = P(STOP|i) for all i

**Recursion:** βₜ(i) = Σⱼ aᵢⱼ × bⱼ(oₜ₊₁) × βₜ₊₁(j)

**Termination:** P(O|λ) = Σᵢ π(i) × bᵢ(o₁) × β₁(i)

Answer should equal P(O|λ) from forward algorithm = **0.002193**

---

### Problem 3: VITERBI (Decoding)

**Goal:** Find most probable tag sequence

**vₜ(j) = max over sequences P(q₁...qₜ₋₁, o₁...oₜ, qₜ=j | λ)**

**Initialization:** v₁(j) = P(j|START) × P(o₁|j), backpointer bt₁(j) = 0

**Recursion:** vₜ(j) = max_i [vₜ₋₁(i) × P(j|i)] × P(oₜ|j)
backpointer: btₜ(j) = argmax_i [vₜ₋₁(i) × P(j|i)]

**Termination:** best sequence ends at argmax_j vT(j) × P(STOP|j)

**Traceback:** follow backpointers from end to start

**Example answer: NNN** (for Hello Cricket Hello with the given HMM)

---

### Problem 4: ξ (Xi) — Joint Probability at Two Time Steps

**P(qₜ=i, qₜ₊₁=j | O, λ) = ξₜ(i,j)**

**Formula:** ξₜ(i,j) = αₜ(i) × aᵢⱼ × bⱼ(oₜ₊₁) × βₜ₊₁(j) / P(O|λ)

**Example from HMM Problems file:**
P(N at t=2 and goes to V at t=3) = ξ₂(N,V)
= α₂(N) × P(V|N) × P(Hello|V) × β₃(V) / P(O|λ)
= **0.0547** (≈ 0.00012 / 0.002193)

---

### Trigram HMM — Parameter Estimation (Sem 1 24-25 Midsem)

**Training sentences:**
```
The dog saw the cat : D N V D N
The cat saw the saw : D N V D N
```

**Transition probabilities (trigram):** P(yᵢ|yᵢ₋₂ yᵢ₋₁) = C(yᵢ₋₂ yᵢ₋₁ yᵢ) / C(yᵢ₋₂ yᵢ₋₁)

Count trigrams from training data including START START at beginning.

**Emission probabilities:** P(word|tag) = C(word, tag) / C(tag)

---

## TOPIC 4: NEURAL NETWORKS & BACKPROPAGATION

### Architecture
- Input layer → Hidden layer(s) → Output layer
- z[l] = W[l]a[l-1] + b[l]
- a[l] = activation(z[l])

### Activation Functions
| Function | Formula | Derivative | Range |
|---|---|---|---|
| Sigmoid | σ(z) = 1/(1+e⁻ᶻ) | σ(z)(1−σ(z)) | (0,1) |
| Tanh | (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ) | 1−tanh²(z) | (−1,1) |
| ReLU | max(0,z) | 0 if z<0, 1 if z≥0 | [0,∞) |

### Computation Graph & Backprop (⚠️ EXAM FAVOURITE)

**Forward pass:** compute values left to right  
**Backward pass:** compute gradients right to left using chain rule

**Chain rule:** ∂L/∂x = (∂L/∂y) × (∂y/∂x)

**Worked Example (Sem 2 25-26 Midsem — exact question):**
```
Network: 3 inputs (x1=0.5, x2=0.7, x3=−0.9)
2 hidden units h1, h2 with ReLU
Output f with no activation
Loss: E = (y − f)²,  y = 1, lr = 0.01

Weights: W1=0.1, W2=0.1, W3=0.2, W4=0.2, W5=0.22, W6=0.22, U1=0.1, U2=0.1

Step 1 — Forward pass:
  h1_pre = W1×x1 + W2×x2 + W3×x3 = 0.1×0.5 + 0.1×0.7 + 0.2×(−0.9) = 0.05+0.07−0.18 = −0.06
  h1 = ReLU(−0.06) = 0
  h2_pre = W4×x1 + W5×x2 + W6×x3 = 0.2×0.5 + 0.22×0.7 + 0.22×(−0.9) = 0.1+0.154−0.198 = 0.056
  h2 = ReLU(0.056) = 0.056
  f = U1×h1 + U2×h2 = 0.1×0 + 0.1×0.056 = 0.0056
  E = (1 − 0.0056)² = (0.9944)² ≈ 0.9888

Step 2 — Backward pass:
  ∂E/∂f = −2(y−f) = −2(0.9944) = −1.9888
  ∂f/∂h1 = U1 = 0.1
  ∂h1/∂h1_pre = dReLU = 0 (since h1_pre < 0)
  ∂h1_pre/∂W1 = x1 = 0.5
  
  ∂E/∂W1 = ∂E/∂f × ∂f/∂h1 × ∂h1/∂h1_pre × ∂h1_pre/∂W1
           = −1.9888 × 0.1 × 0 × 0.5 = 0

  W1_new = W1 − lr × ∂E/∂W1 = 0.1 − 0.01 × 0 = 0.1
```

### Loss Functions
- **MSE:** E = (y − f)²
- **Cross-entropy (binary):** L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
- **Cross-entropy (multi-class):** L = −Σ yₖ·log(ŷₖ)

### Gradient for Sigmoid + Cross-entropy
∂L/∂z = ŷ − y  (this is a clean result — memorize it!)

### Softmax
P(class k) = exp(zₖ) / Σⱼ exp(zⱼ)

For multinomial logistic regression: gradient w.r.t. weight = (ŷₖ − yₖ) × xᵢ

---

## TOPIC 5: RNN & LSTM

### RNN
hₜ = g(W·xₜ + U·hₜ₋₁)  [+ bias]
yₜ = softmax(V·hₜ)

**For sentiment classification:** use last hidden state hₙ (or average of all hᵢ)

**Encoder-Decoder (for MT):**
- Encoder: reads source words, produces hidden states h₁ᵉ...h₄ᵉ
- Decoder: hₜᵈ = f(W·yₜ₋₁ + U·hₜ₋₁ᵈ + Z·hₜᵉ)
- With attention: c = Σ wᵢ·hᵢᵉ, where wᵢ = score(hₜ₋₁ᵈ, hᵢᵉ) / Σ scores

**Teacher forcing:** during training, feed the gold (correct) output at each step, not the model's prediction

### LSTM (⚠️ EXAM QUESTION)

**Purpose:** Solve vanishing gradient; preserve long-range dependencies

**3 gates (all use sigmoid → output 0 to 1 = "mask"):**
- **Forget gate:** fₜ = σ(Uf·hₜ₋₁ + Wf·xₜ)  → how much of old memory to keep
- **Input gate:** iₜ = σ(Uᵢ·hₜ₋₁ + Wᵢ·xₜ)  → how much new info to add
- **Output gate:** oₜ = σ(Uo·hₜ₋₁ + Wo·xₜ)  → what to expose as hidden state

**Cell state update:**
- gₜ = tanh(Ug·hₜ₋₁ + Wg·xₜ)  [new candidate memory, range −1 to 1]
- kₜ = cₜ₋₁ ⊙ fₜ  [refined old memory — redundant info discarded by forget gate]
- jₜ = gₜ ⊙ iₜ  [useful new info]
- cₜ = kₜ + jₜ  [new cell memory]
- hₜ = oₜ ⊙ tanh(cₜ)  [new hidden state]

**How redundant info is discarded:** The forget gate fₜ produces values near 0 for dimensions that should be forgotten. When kₜ = cₜ₋₁ ⊙ fₜ, those near-0 values zero out the corresponding memory components.

**GRU:** Alternative to LSTM with fewer parameters (2 gates instead of 3)

---

## TOPIC 6: WORD EMBEDDINGS (Word2Vec)

### CBOW (Continuous Bag of Words)
- **Input:** surrounding/context words → predict center word
- **Two weight matrices:**
  - W (input-hidden): shape [embedding_dim × vocab_size] — shared for all context words
  - W' (hidden-output): shape [vocab_size × embedding_dim]
- **Which to use for word vectors?** The input-hidden matrix W (or average of both) — W' is used for prediction, W captures semantic similarity

### Skip-gram
- **Input:** center word → predict surrounding words
- **Final word vector:** from the input embedding matrix

### Embedding Matrices for Multi-component Tokens (from Sem 1 24-25 Compre)
Token has 3 components: word, POS, lemma
- E_word: [|V| × d_word]
- E_pos: [|P| × d_pos]
- E_lemma: [|C| × d_lemma]
- Token embedding = concat of 3 vectors → dimension d_word + d_pos + d_lemma

---

## TOPIC 7: TRANSFORMERS

### Self-Attention
**Score:** score(xᵢ, xⱼ) = xᵢ · xⱼ  (simplified)
**Alpha:** αᵢⱼ = softmax(score(xᵢ, xⱼ))
**Output:** aᵢ = Σⱼ αᵢⱼ · xⱼ

### Scaled Dot-Product Attention
Q, K, V matrices:
- qᵢ = xᵢ · W^Q  (query)
- kⱼ = xⱼ · W^K  (key)
- vⱼ = xⱼ · W^V  (value)

Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ) · V

**Why √dₖ scaling?** (⚠️ EXAM QUESTION — from Sem 2 25-26 Midsem)
As the dimensionality dₖ grows, the dot products Q·Kᵀ can become very large in magnitude. Large values push the softmax into saturation regions where gradients become extremely small (vanishing gradients). Dividing by √dₖ keeps the dot products in a reasonable range and stabilizes training.

### Multi-Head Attention
For each head c: qᵢᶜ = xᵢ·W^Qc, kⱼᶜ = xⱼ·W^Kc, vⱼᶜ = xⱼ·W^Vc
headᵢᶜ = softmax(mask(QᵢKᵢᵀ / √dₖ))·Vᵢ
aᵢ = (head¹ ⊕ head² ⊕ ... ⊕ headᴬ) · W^O

### Feed-Forward Network (FFN)
FFN(xᵢ) = ReLU(xᵢ·W₁ + b₁)·W₂ + b₂

- Weights same for each token position, different across layers
- d_ff > d (expanded dimension) — acts as memory for complex features

### Layer Normalization
LayerNorm(x) = γ·(x−μ)/σ + β, where σ = √[(1/d)·Σ(xᵢ−μ)²]
Applied to embedding vector of single token (not entire layer)

### Full Transformer Block (per layer)
```
t1 = LayerNorm(x)
t2 = MultiHeadAttention(t1, [t1₁,...,t1ₙ])
t3 = t2 + x                 ← residual connection
t4 = LayerNorm(t3)
t5 = FFN(t4)
h  = t5 + t3               ← residual connection
```

### Positional Encodings
- **Learned:** lookup table, trained by backprop. Problem: fewer training examples for large positions.
- **Sinusoidal:** helps capture relationships between positions
- **Relative (RPE):** score = Query·(Key + Distance_Vector). Generalizes to unseen lengths. Solves the length limit problem.

### Encoder-Decoder Transformer (Sem 2 25-26 Midsem Q5)
- **Encoder:** processes source sentence, each token gets a contextualized representation
- **Decoder:** generates target tokens autoregressively, attends to encoder output via cross-attention
- Cross-attention: Queries from decoder, Keys and Values from encoder

---

## TOPIC 8: BERT & MASKED LANGUAGE MODELS

### Pre-training Objectives
1. **Masked Language Modeling (MLM):** 15% tokens manipulated → 80% replaced with [MASK], 10% random word, 10% unchanged. Model predicts original token. Bidirectional (sees all tokens).
2. **Next Sentence Prediction (NSP):** Given two sentences, predict if B follows A. Adds [CLS] at start, segment embeddings to distinguish sentences.

### Special Tokens
- **[CLS]:** Contains aggregate sequence representation → used for classification tasks
- **[SEP]:** Separates two sentences
- **[MASK]:** Replaced token during MLM
- **[PAD]:** Padding to same length

### Input Representation
Token embedding + Positional embedding + Segment embedding

**Segment embedding:** distinguishes which sentence each token belongs to (Sentence A vs. Sentence B). Used in NSP task.

### Fine-tuning BERT
- **Sentiment classification:** Feed [CLS] output to linear classifier → cross-entropy loss
- **NER:** Feed each token's output to a classifier → sequence labeling (BIO tagging)
- **NLI:** Feed [CLS] of [sentence A; SEP; sentence B] to classifier
- May update last few layers of BERT during fine-tuning

**Changing downstream task (⚠️ EXAM QUESTION):**  
If BERT fine-tuned for 2-class sentiment (positive/negative) and you want to do NER with 3 entity types:
- Remove the 2-class classification head
- Add a new per-token classification head with output size = 2×3+1 = 7 (BIO tags for 3 entities + O)
- Fine-tune on NER data

### Which tasks can MLM-only BERT handle?
- **NER:** Yes — it's a token-level classification task (sequence labeling). MLM gives rich contextual embeddings for each token.
- **Sentiment classification:** Yes — the [CLS] token gives sequence-level representation.
- But NSP task is needed if you want sentence-pair reasoning.

### Curse of Multilinguality
When training multilingual BERT on many languages, adding more languages hurts per-language performance because parameters are shared. High-resource language patterns "bleed" into low-resource languages.

Sampling: qᵢ = pᵢᵅ / Σpⱼᵅ, where pᵢ = nᵢ/Σnᵢ, α ≈ 0.3 (upsamples rare languages)

### Contextual Embeddings
- Better than static (Word2Vec) — same word gets different embedding in different contexts
- **Anisotropy problem:** Embeddings clustered in small cone → cosine similarity unreliable. Solution: standardize z = (x−μ)/σ
- Common approach: average last 4 layers' output for word representation

### Static vs. Contextual Embeddings (⚠️ EXAM QUESTION)
- **Static (Word2Vec):** one fixed vector per word, regardless of context
- **Contextual (BERT):** each occurrence of a word gets a different vector based on surrounding context. "Bank" in "river bank" vs. "bank account" get different vectors.

---

## TOPIC 9: LARGE LANGUAGE MODELS (LLMs)

### Three Architectures
| Type | Model Examples | Use Case | Direction |
|---|---|---|---|
| Decoder | GPT, Claude, Llama | Text generation | Left-to-right |
| Encoder | BERT, RoBERTa | Classification | Bidirectional |
| Encoder-Decoder | T5, BART | Translation, Summarization | Seq-to-seq |

### Pre-training
- **Decoder (causal LM):** predict next token from previous tokens. Loss = cross-entropy.
- **Encoder (MLM):** predict masked tokens using both left and right context.

### Decoding Strategies
| Strategy | Description | Tradeoff |
|---|---|---|
| Greedy | Pick highest prob token | Repetitive, generic |
| Sampling | Sample from full distribution | Can produce strange text |
| Top-K | Sample from top K tokens | K is fixed regardless of distribution |
| Top-P (Nucleus) | Sample from top P% probability mass | Adapts to distribution shape |
| Temperature | y = softmax(u/τ). τ<1: sharper; τ>1: flatter | Adjusts randomness |

### In-Context Learning (ICL)
- Provide examples (demonstrations) in the prompt — model learns task without gradient updates
- **Zero-shot:** no examples; **Few-shot:** k examples in prompt
- Demonstrations help even with wrong labels (format matters more than correctness)
- Learning happens through induction heads (not weight updates)

**ICL vs. Instruction Tuning (⚠️ EXAM QUESTION):**
- **ICL:** No weight updates. Examples in the prompt at inference time. Weights frozen.
- **Instruction Tuning:** Supervised fine-tuning on instruction-response pairs. Weights ARE updated.

---

## TOPIC 10: POST-TRAINING

### Stage 1: Instruction Tuning (SFT)
- Fine-tune pretrained LM on corpus of instructions + responses
- Loss: cross-entropy (same as pretraining)
- Teaches model to follow instructions and generalize to new tasks
- **Evaluation:** leave-one-out at cluster level (tasks clustered by similarity, evaluate on held-out cluster)

### Stage 2: Alignment
- **Goal:** Make model helpful, honest, harmless
- **RLHF:** Human provides preferences between model outputs → train reward model → optimize with RL (PPO)
- **DPO (Direct Preference Optimization):** More stable alternative to RLHF; directly optimizes on preference data without separate reward model

### Parameter-Efficient Fine-Tuning (PEFT)
- **LoRA:** Freeze W, learn low-rank approximation ΔW = A·B where A∈ℝᴺˣʳ, B∈ℝʳˣᵈ, r ≪ min(N,d)
- During fine-tuning, only A and B are updated
- After training, merge: W_new = W + AB (no inference overhead)

---

## TOPIC 11: INFORMATION RETRIEVAL & RAG

### TF-IDF
- **TF:** tf(t,d) = 1 + log₁₀(count(t,d))  if count > 0, else 0
- **IDF:** idf(t) = log₁₀(N/df_t)  where df_t = number of docs containing term t
- **TF-IDF:** tf-idf(t,d) = tf(t,d) × idf(t)
- **Why log?** (⚠️ EXAM QUESTION) Raw term frequency overweights frequent terms. Log compresses the scale so that a word appearing 100× is not 100× more important than one appearing 1×. Same for IDF — log smooths the large range of document frequencies.

### Cosine Similarity
score(q,d) = (q·d) / (|q|·|d|) = Σ(qₜ·dₜ) / (√Σqₜ² · √Σdₜ²)
where qₜ = tf-idf(t,q) and dₜ = tf-idf(t,d)

### BM25
BM25(t,d) = idf(t) × tf(t,d)×(k+1) / [tf(t,d) + k×(1−b+b×|d|/|d_avg|)]
- k: controls TF saturation (k=0: binary; large k: raw TF)
- b: controls length normalization (b=1: full normalization; b=0: none)

### Evaluation
- **Precision@k:** fraction of top-k results that are relevant
- **Recall:** fraction of all relevant docs that are retrieved
- **Average Precision (AP):** average precision at each relevant document rank
- **MAP:** mean AP across all queries
- **Interpolated Precision:** P(r) = max_{i≥r} P(i)

### Dense Retrieval
- **Single encoder:** BERT sees query + document together → expensive, can't precompute
- **Bi-encoder:** Separate encoders for query and document. Score = z_q · z_d. Documents can be pre-indexed → fast retrieval. Less accurate (no query-doc interaction).
- **ColBERT:** Token-level bi-encoder. Score = Σ_q max_d (q_token · d_token)

### RAG (Retrieval-Augmented Generation)
Before RAG: P(x₁...xₙ) = Π P(xᵢ | prompt, q, x<ᵢ)
After RAG: P(x₁...xₙ) = Π P(xᵢ | R(q), prompt, q, x<ᵢ)
where R(q) = retrieved documents relevant to query q

**Why RAG?** LLMs hallucinate — they generate plausible-sounding but factually wrong answers. RAG grounds answers in retrieved documents and can handle proprietary/up-to-date information.

---

## TOPIC 12: IBM TRANSLATION MODEL (if relevant)

### IBM Model 1 Basic
p(f, a | e, m) = Π [p(aᵢ|l,m) × p(fᵢ|e_{aᵢ})]
- f: source (foreign) sentence
- e: target (English) sentence  
- a: alignment variables (each source word aligns to one target word)
- m: source length, l: target length

**Uniform alignment:** p(aᵢ|l,m) = 1/(l+1) [each source word equally likely to align to any target word]

**Why O(l+1)m instead of O((l+1)^m)?** (from Sem 1 24-25 Midsem)
Each source position aᵢ can independently choose its alignment. So instead of considering all (l+1)^m combinations together, we optimize each aᵢ independently → (l+1)^m factored into m independent choices of (l+1) = O(m·(l+1)) = O((l+1)m).

---

## QUICK-REFERENCE FORMULAS SHEET

### HMM Summary
```
Forward:   α₁(j) = π_j × b_j(o₁)
           αₜ(j) = [Σᵢ αₜ₋₁(i) × aᵢⱼ] × bⱼ(oₜ)
           P(O|λ) = Σⱼ αT(j) × P(STOP|j)

Backward:  βT(i) = P(STOP|i)
           βₜ(i) = Σⱼ aᵢⱼ × bⱼ(oₜ₊₁) × βₜ₊₁(j)
           P(O|λ) = Σᵢ α₁(i) × β₁(i) / b₁(i)... [use forward check]

Viterbi:   v₁(j) = P(j|START) × P(o₁|j)
           vₜ(j) = max_i[vₜ₋₁(i) × aᵢⱼ] × bⱼ(oₜ)
           bt(j) = argmax_i[vₜ₋₁(i) × aᵢⱼ]

Xi:        ξₜ(i,j) = αₜ(i)×aᵢⱼ×bⱼ(oₜ₊₁)×βₜ₊₁(j) / P(O|λ)
Gamma:     γₜ(j) = αₜ(j)×βₜ(j) / P(O|λ)
```

### Naive Bayes Summary
```
P(w|c)_multinomial = [C(w,c) + 1] / [Σ C(w',c) + |V|]
P(w|c)_binary     = [binary_C(w,c) + 1] / [Σ binary_C(w',c) + |V|]
P(c) = docs_in_c / total_docs
c* = argmax_c [log P(c) + Σ log P(wᵢ|c)]
```

### Transformer Attention Summary
```
Q = X·W^Q,  K = X·W^K,  V = X·W^V
Attention = softmax(Q·Kᵀ / √dₖ) · V
FFN(x) = ReLU(x·W₁+b₁)·W₂+b₂
LayerNorm(x) = γ·(x−μ)/σ + β
```

### LSTM Summary
```
fₜ = σ(Uf·hₜ₋₁ + Wf·xₜ)    ← forget gate
iₜ = σ(Uᵢ·hₜ₋₁ + Wᵢ·xₜ)    ← input gate
oₜ = σ(Uo·hₜ₋₁ + Wo·xₜ)    ← output gate
gₜ = tanh(Ug·hₜ₋₁ + Wg·xₜ) ← candidate memory
cₜ = cₜ₋₁⊙fₜ + gₜ⊙iₜ       ← new cell state
hₜ = oₜ ⊙ tanh(cₜ)          ← new hidden state
```

---

## 📝 2-DAY STUDY PLAN

### Day 1 — Master the Calculation Topics
**Morning (3h):** HMM — forward, backward, Viterbi, ξ  
→ Do the HMM Example Problems file fully with pen and paper  
→ Redo Sem 1 24-25 Midsem Q2 (trigram HMM) and Sem 2 25-26 Midsem Q4

**Afternoon (2h):** N-gram counting + Naive Bayes  
→ Redo the n-gram counting question from Sem 2 25-26 Midsem Q1  
→ Redo Sem 2 25-26 Midsem Q2 (Naive Bayes with negation)

**Evening (2h):** Neural networks — computation graphs, backprop  
→ Redo Sem 2 25-26 Midsem Q4 (ReLU network, find w₁ after update)  
→ Redo Sem 1 24-25 Compre Q2 (tied weights, sigmoid)

### Day 2 — Modern NLP + Review
**Morning (2h):** Transformers, BERT, LLMs  
→ Read Cheatsheet 2 very carefully — it covers all the key equations  
→ Review: attention scaling, BERT objectives, fine-tuning, ICL vs instruction tuning

**Afternoon (2h):** IR, RAG, Post-training + LSTM  
→ Review TF-IDF formula and cosine similarity  
→ Review bi-encoder vs cross-encoder  
→ Review LSTM gates and how forget gate discards info

**Evening (2h):** Full past paper practice  
→ Do Sem 1 25-26 Compre exam fully under exam conditions  
→ Check answers against the solutions PDF (NLP_Midsem_Solutions.PDF)

---

## ⚡ KEY THINGS TO MEMORIZE

1. **HMM forward recursion:** αₜ(j) = [Σᵢ αₜ₋₁(i)·aᵢⱼ] × bⱼ(oₜ) — and termination includes P(STOP|j)
2. **Viterbi difference from forward:** use MAX not SUM, and keep backpointers
3. **Naive Bayes add-1 smoothing denominator:** total_words_in_class + |V| (not +1)
4. **Transformer scaling:** divide by √dₖ to avoid vanishing gradients from large dot products
5. **LSTM forget gate discards redundancy:** fₜ near 0 → old memory zeroed out
6. **BERT [CLS] for classification, per-token for NER**
7. **In-context learning ≠ fine-tuning:** no weight updates in ICL
8. **Bi-encoder precomputes document embeddings; cross-encoder cannot**
9. **Binary Naive Bayes: binarize per-document, not per-class**
10. **ξₜ(i,j) formula:** α × transition × emission × β / P(O)
