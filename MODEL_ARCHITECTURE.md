# Neural Machine Translation - Model Architecture Flow

## 🏗️ **Complete System Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NEURAL MACHINE TRANSLATION SYSTEM                   │
│                            (English → French)                                │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: "hello world"
   ↓

┌──────────────────────────────────────────────────────────────────────────────┐
│                          📥 DATA PREPROCESSING PIPELINE                       │
└──────────────────────────────────────────────────────────────────────────────┘

1. TOKENIZATION
   ├─ Input: "hello world"
   ├─ Tokenizer.texts_to_sequences()
   └─ Output: [45, 231]  (word indices)

2. PADDING
   ├─ Input: [45, 231]
   ├─ pad_sequences(maxlen=20)
   └─ Output: [45, 231, 0, 0, 0, ..., 0]  (shape: [20])

3. TENSOR CONVERSION
   └─ PyTorch Tensor: shape [1, 20]  (batch_size=1, seq_len=20)

   ↓

┌──────────────────────────────────────────────────────────────────────────────┐
│                          🧠 ENCODER (BIDIRECTIONAL LSTM)                      │
└──────────────────────────────────────────────────────────────────────────────┘

INPUT TENSOR: [batch_size, src_seq_len] = [1, 20]
   ↓
   
┌─────────────────────────┐
│  EMBEDDING LAYER        │
│  ─────────────────      │
│  Input: [1, 20]         │
│  Lookup in embedding    │
│  matrix [vocab, 256]    │
│  Output: [1, 20, 256]   │  ← Each word → 256-dim vector
└─────────────────────────┘
   ↓

┌─────────────────────────────────────────────────────────────────┐
│  MULTI-LAYER BIDIRECTIONAL LSTM                                 │
│  ───────────────────────────────────────                        │
│                                                                  │
│  Layer 1: Bidirectional LSTM                                    │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │  FORWARD →   │         │  ← BACKWARD  │                     │
│  │  LSTM Cell   │         │  LSTM Cell   │                     │
│  │  [1,20,256]  │         │  [1,20,256]  │                     │
│  └──────────────┘         └──────────────┘                     │
│         ↓                         ↓                             │
│         └─────────── CONCAT ──────┘                             │
│                 ↓                                                │
│         [1, 20, 512]  (forward + backward)                      │
│                 ↓                                                │
│            DROPOUT (0.1)                                         │
│                 ↓                                                │
│  Layer 2: Bidirectional LSTM (if num_layers > 1)               │
│         [1, 20, 512] → ... → [1, 20, 512]                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
   ↓

ENCODER OUTPUTS:
├─ encoder_outputs: [1, 20, 512]  ← All hidden states (for attention)
├─ state_h: [1, 256]              ← Final forward hidden state
└─ state_c: [1, 256]              ← Final forward cell state

   ↓

┌──────────────────────────────────────────────────────────────────────────────┐
│                     🎯 DECODER (UNIDIRECTIONAL LSTM + ATTENTION)              │
└──────────────────────────────────────────────────────────────────────────────┘

INITIALIZATION:
├─ decoder_input: [1, 1] = [SOS_TOKEN_ID]  ← Start token
└─ decoder_state: (state_h, state_c) from encoder

┌──────────────────────────────────────────────────────────────────┐
│                    STEP-BY-STEP GENERATION LOOP                   │
│                  (Repeat for each output word)                    │
└──────────────────────────────────────────────────────────────────┘

For t = 0, 1, 2, ..., max_length:

   ┌─────────────────────────┐
   │  EMBEDDING LAYER        │
   │  ─────────────────      │
   │  Input: [1, 1]          │  ← Current decoder input token
   │  Output: [1, 1, 256]    │
   └─────────────────────────┘
      ↓
      
   ┌─────────────────────────────────────────┐
   │  MULTI-LAYER LSTM                       │
   │  ────────────────                       │
   │  Input: [1, 1, 256]                     │
   │  State: (h_t, c_t)                      │
   │  Output: [1, 1, 256]                    │
   │  New State: (h_{t+1}, c_{t+1})          │
   └─────────────────────────────────────────┘
      ↓
   lstm_output: [1, 1, 256]
      ↓
      
   ┌──────────────────────────────────────────────────────────────┐
   │  ATTENTION MECHANISM (Bahdanau/Additive)                     │
   │  ────────────────────────────────────────                    │
   │                                                               │
   │  Query (from decoder):  [1, 1, 256]                          │
   │  Keys (from encoder):   [1, 20, 512]                         │
   │  Values (from encoder): [1, 20, 512]                         │
   │                                                               │
   │  Step 1: SCORE COMPUTATION                                   │
   │  ─────────────────────                                       │
   │  query_projection = W_q @ query      → [1, 1, attention_dim]│
   │  key_projection = W_k @ keys         → [1, 20, attention_dim]│
   │  combined = tanh(query_proj + key_proj)                      │
   │  scores = V^T @ combined             → [1, 20]               │
   │                                                               │
   │  Step 2: ATTENTION WEIGHTS (Softmax)                         │
   │  ────────────────────────────────                            │
   │  attention_weights = softmax(scores) → [1, 20]               │
   │  ┌────────────────────────────────────────────────┐          │
   │  │ [0.05, 0.32, 0.18, 0.08, 0.01, ..., 0.00]     │          │
   │  │   ↑     ↑                                       │          │
   │  │  word1 word2 ← Focuses here!                   │          │
   │  └────────────────────────────────────────────────┘          │
   │                                                               │
   │  Step 3: CONTEXT VECTOR (Weighted Sum)                       │
   │  ──────────────────────────────────                          │
   │  context = Σ(attention_weights * values) → [1, 1, 512]      │
   │                                                               │
   └──────────────────────────────────────────────────────────────┘
      ↓
   context_vector: [1, 1, 512]
      ↓
      
   ┌─────────────────────────────────────────┐
   │  CONCATENATION                          │
   │  ─────────────                          │
   │  [context, lstm_output]                 │
   │  [1, 1, 512] + [1, 1, 256]             │
   │  = [1, 1, 768]                          │
   └─────────────────────────────────────────┘
      ↓
      
   ┌─────────────────────────────────────────┐
   │  OUTPUT PROJECTION (Linear Layer)       │
   │  ────────────────────────────            │
   │  Input: [1, 768]                        │
   │  Weight: [768, french_vocab_size]       │
   │  Output: [1, french_vocab_size]         │
   │  = [1, 10000] (logits)                  │
   └─────────────────────────────────────────┘
      ↓
      
   ┌─────────────────────────────────────────┐
   │  SOFTMAX (Probability Distribution)     │
   │  ──────────────────────────────          │
   │  P(word) for all French vocab           │
   │  [0.001, 0.003, 0.856, ..., 0.002]     │
   │                    ↑                     │
   │                 highest probability      │
   └─────────────────────────────────────────┘
      ↓
      
   ┌─────────────────────────────────────────┐
   │  TOKEN SELECTION                        │
   │  ───────────────                        │
   │  • Greedy: argmax(logits)               │
   │  • Sampling: sample with temperature    │
   │                                          │
   │  Selected token ID: 7834 → "bonjour"   │
   └─────────────────────────────────────────┘
      ↓
      
   ┌─────────────────────────────────────────┐
   │  NEXT INPUT DECISION                    │
   │  ───────────────────                    │
   │  • Training (Teacher Forcing):          │
   │    Use ground truth next token          │
   │                                          │
   │  • Inference:                            │
   │    Use predicted token (7834)           │
   │                                          │
   │  • Teacher Forcing Ratio:               │
   │    Mix of both with probability         │
   └─────────────────────────────────────────┘
      ↓
      
   decoder_input = selected_token
   
   Stop if:
   ├─ Token == EOS_TOKEN_ID  (End of sequence)
   ├─ Length >= max_length   (Maximum length reached)
   └─ All sequences in batch finished

LOOP BACK TO TOP ↑

┌──────────────────────────────────────────────────────────────────────────────┐
│                          📤 OUTPUT GENERATION                                 │
└──────────────────────────────────────────────────────────────────────────────┘

GENERATED TOKEN IDs: [1, 7834, 4521, 2]
                     [SOS, "bonjour", "monde", EOS]
   ↓

DETOKENIZATION:
   ├─ Remove SOS and EOS tokens
   ├─ Map IDs back to words
   └─ Join words with spaces

OUTPUT: "bonjour monde"

```

---

## 🔄 **Training vs Inference Flow**

### **TRAINING MODE** (with Teacher Forcing)

```
┌──────────────────────────────────────────────────────────────┐
│                      TRAINING STEP                           │
└──────────────────────────────────────────────────────────────┘

Input Batch:
├─ encoder_inputs: [batch_size, src_seq_len]
├─ decoder_inputs: [batch_size, tgt_seq_len]  ← Ground truth (shifted)
└─ targets: [batch_size, tgt_seq_len]         ← Ground truth

   ↓

┌────────────────────┐
│  ENCODER           │
│  (Bidirectional)   │
└────────────────────┘
   ↓
encoder_outputs, state_h, state_c

   ↓

┌────────────────────────────────────────────┐
│  DECODER (Step-by-Step)                    │
│                                             │
│  For each time step t:                     │
│    ┌─────────────────────────────────┐    │
│    │  Current Input Decision:        │    │
│    │                                  │    │
│    │  if random() < teacher_forcing: │    │
│    │    use decoder_inputs[:, t]     │    │  ← Ground truth
│    │  else:                           │    │
│    │    use predicted_token          │    │  ← Model prediction
│    └─────────────────────────────────┘    │
│                                             │
│  → Embedding → LSTM → Attention →          │
│    Concatenate → Output Projection         │
│                                             │
└────────────────────────────────────────────┘
   ↓
predictions: [batch_size, tgt_seq_len, vocab_size]

   ↓

┌────────────────────────────────────────────┐
│  LOSS COMPUTATION                          │
│  ────────────────                          │
│  loss = CrossEntropy(predictions, targets) │
│  mask out padding tokens (token_id = 0)    │
└────────────────────────────────────────────┘
   ↓
   
┌────────────────────────────────────────────┐
│  BACKPROPAGATION                           │
│  ───────────────                           │
│  loss.backward()                           │
│  ↓                                          │
│  Compute gradients for all parameters      │
│  ↓                                          │
│  Gradient Clipping (prevent exploding)     │
│  ↓                                          │
│  optimizer.step()  ← Update weights        │
└────────────────────────────────────────────┘

```

### **INFERENCE MODE** (Generation)

```
┌──────────────────────────────────────────────────────────────┐
│                      INFERENCE/TRANSLATION                   │
└──────────────────────────────────────────────────────────────┘

Input: "hello world"

   ↓

┌────────────────────┐
│  ENCODER           │
│  (Bidirectional)   │
└────────────────────┘
   ↓
encoder_outputs, state_h, state_c

   ↓

┌────────────────────────────────────────────┐
│  DECODER (Autoregressive Generation)       │
│                                             │
│  Initialize:                                │
│  ├─ decoder_input = [SOS]                  │
│  └─ decoder_state = (state_h, state_c)     │
│                                             │
│  Loop until EOS or max_length:             │
│    ┌─────────────────────────────────┐    │
│    │  1. Embed current token         │    │
│    │  2. Pass through LSTM           │    │
│    │  3. Apply attention             │    │
│    │  4. Concatenate & project       │    │
│    │  5. Sample/Select next token    │    │
│    │  6. Append to output sequence   │    │
│    │  7. Use as next input           │    │
│    └─────────────────────────────────┘    │
│                                             │
│  NO teacher forcing - always use previous  │
│  predicted token as next input             │
│                                             │
└────────────────────────────────────────────┘
   ↓
   
Generated Sequence: [SOS, token1, token2, ..., EOS]

   ↓

Detokenize → "bonjour monde"
```

---

## 📊 **Detailed Component Dimensions**

### **Configuration Example**
```python
batch_size = 64
src_vocab_size = 8000      # English vocabulary
tgt_vocab_size = 10000     # French vocabulary
embedding_dim = 256
lstm_units = 256
encoder_num_layers = 2
decoder_num_layers = 2
dropout_rate = 0.1
bidirectional = True
max_src_length = 20
max_tgt_length = 20
```

### **Tensor Shapes Through the Network**

```
ENCODER:
─────────
Input:              [64, 20]           # batch_size × src_seq_len
  ↓ Embedding
Embedded:           [64, 20, 256]      # batch_size × src_seq_len × embedding_dim
  ↓ Bidirectional LSTM Layer 1
Forward Hidden:     [64, 20, 256]
Backward Hidden:    [64, 20, 256]
Concatenated:       [64, 20, 512]      # doubled due to bidirectional
  ↓ Dropout
  ↓ Bidirectional LSTM Layer 2
Encoder Outputs:    [64, 20, 512]      # batch_size × src_seq_len × (lstm_units*2)
Final state_h:      [64, 256]          # batch_size × lstm_units (forward only)
Final state_c:      [64, 256]          # batch_size × lstm_units (forward only)

DECODER (per time step):
────────────────────────
Current Input:      [64, 1]            # batch_size × 1
  ↓ Embedding
Embedded:           [64, 1, 256]       # batch_size × 1 × embedding_dim
  ↓ LSTM Layer 1
LSTM Output:        [64, 1, 256]       # batch_size × 1 × lstm_units
  ↓ LSTM Layer 2
LSTM Output:        [64, 1, 256]       # batch_size × 1 × lstm_units

ATTENTION:
──────────
Query (decoder):    [64, 1, 256]       # batch_size × 1 × lstm_units
Keys (encoder):     [64, 20, 512]      # batch_size × src_seq_len × encoder_output_size
Values (encoder):   [64, 20, 512]      # batch_size × src_seq_len × encoder_output_size
  ↓ Compute scores
Attention Weights:  [64, 1, 20]        # batch_size × 1 × src_seq_len
  ↓ Weighted sum
Context Vector:     [64, 1, 512]       # batch_size × 1 × encoder_output_size

OUTPUT:
───────
Concatenated:       [64, 1, 768]       # [context(512) + lstm_output(256)]
  ↓ Flatten to [64, 768]
  ↓ Linear Projection
Logits:             [64, 10000]        # batch_size × tgt_vocab_size
  ↓ Softmax
Probabilities:      [64, 10000]        # batch_size × tgt_vocab_size
```

---

## 🧮 **Mathematical Operations**

### **1. LSTM Cell Forward Pass**
```
Gates Computation (for each time step):

Input Gate:     i_t = σ(W_ii·x_t + W_hi·h_{t-1} + b_i)
Forget Gate:    f_t = σ(W_if·x_t + W_hf·h_{t-1} + b_f)
Cell Gate:      g_t = tanh(W_ig·x_t + W_hg·h_{t-1} + b_g)
Output Gate:    o_t = σ(W_io·x_t + W_ho·h_{t-1} + b_o)

Cell State:     c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
Hidden State:   h_t = o_t ⊙ tanh(c_t)

Where:
  σ = sigmoid activation
  ⊙ = element-wise multiplication
  W = weight matrices
  b = bias vectors
```

### **2. Attention Mechanism**
```
Additive (Bahdanau) Attention:

1. Project query and keys:
   query_proj = W_q @ query         # [batch, 1, attn_dim]
   key_proj = W_k @ keys            # [batch, src_len, attn_dim]

2. Compute alignment scores:
   combined = query_proj + key_proj # Broadcasting
   activated = tanh(combined)
   scores = V^T @ activated         # [batch, 1, src_len]

3. Attention weights (probabilities):
   α = softmax(scores / √d_k)       # [batch, 1, src_len]
   
4. Context vector (weighted sum):
   context = Σ(α_i · value_i)       # [batch, 1, value_dim]
```

### **3. Loss Function**
```
Sparse Categorical Cross-Entropy:

For each position (i, j):
  loss_ij = -log(P(y_ij | x, y_{<j}))
  
Where:
  y_ij = true token at position j in sequence i
  P = softmax(logits)

Total Loss:
  L = (1/N) Σ Σ loss_ij  (masked for padding)
      
Accuracy:
  acc = (correct_predictions / non_padding_tokens) × 100%
```

---

## 🔀 **Bidirectional Processing**

```
UNIDIRECTIONAL (Standard):
───────────────────────────
Input:  [w1, w2, w3, w4, w5]
        ↓   ↓   ↓   ↓   ↓
       h1→ h2→ h3→ h4→ h5→

Each h_t only sees past: w1...w_t


BIDIRECTIONAL:
──────────────
Input:      [w1, w2, w3, w4, w5]

Forward:    h1→ h2→ h3→ h4→ h5→
Backward:  ←h1 ←h2 ←h3 ←h4 ←h5

Combined:   [h1→;←h1]  [h2→;←h2]  [h3→;←h3]  [h4→;←h4]  [h5→;←h5]
            └────┬────┘
                 Concatenate
                 
Each output sees both past AND future context!
→ Better understanding of sentence structure
→ Improved translation quality
```

---

## 🎯 **Parameter Count Calculation**

```
ENCODER:
────────
Embedding:              8,000 × 256 = 2,048,000

Bidirectional LSTM Layer 1:
  Forward cell:
    W_ii, W_if, W_ig, W_io:  (256 + 256) × 256 × 4 = 524,288
    Biases:                   256 × 4 = 1,024
  Backward cell:             525,312
  Total Layer 1:             1,050,624

Bidirectional LSTM Layer 2:
  Forward cell (input now 512):  (512 + 256) × 256 × 4 = 786,432 + 1,024
  Backward cell:                 787,456
  Total Layer 2:                 1,574,912

Total Encoder:                   ~4.7M parameters

DECODER:
────────
Embedding:              10,000 × 256 = 2,560,000

LSTM Layer 1:           (256 + 256) × 256 × 4 + 1,024 = 525,312
LSTM Layer 2:           (256 + 256) × 256 × 4 + 1,024 = 525,312

Attention:
  W_q:                  256 × 256 = 65,536
  W_k:                  512 × 256 = 131,072
  V:                    256 × 1 = 256
  Total:                196,864

Output Dense:           768 × 10,000 = 7,680,000

Total Decoder:          ~11.5M parameters

TOTAL MODEL:            ~16.2M parameters
```

---

## 🚦 **Training Process Flow**

```
┌─────────────────────────────────────────────────────────────┐
│                    EPOCH LOOP (1 to N)                      │
└─────────────────────────────────────────────────────────────┘
    │
    ├── TRAINING PHASE
    │   │
    │   ├── For each batch in train_loader:
    │   │   ├─ 1. Get batch (encoder_inputs, decoder_inputs, targets)
    │   │   ├─ 2. Forward pass through model
    │   │   ├─ 3. Compute loss and accuracy
    │   │   ├─ 4. Backward pass (compute gradients)
    │   │   ├─ 5. Clip gradients
    │   │   ├─ 6. Optimizer step (update weights)
    │   │   └─ 7. Accumulate metrics
    │   │
    │   └── Calculate average train loss and accuracy
    │
    ├── VALIDATION PHASE
    │   │
    │   ├── For each batch in val_loader:
    │   │   ├─ 1. Forward pass (no gradients)
    │   │   ├─ 2. Compute loss and accuracy
    │   │   └─ 3. Accumulate metrics
    │   │
    │   └── Calculate average val loss and accuracy
    │
    ├── CHECKPOINT SAVING
    │   │
    │   └── If val_accuracy > best_val_accuracy:
    │       ├─ Save model state
    │       ├─ Save data_dict
    │       ├─ Save training history
    │       └─ Update best_val_accuracy
    │
    ├── LEARNING RATE SCHEDULING
    │   │
    │   └── Adjust learning rate based on plateau
    │
    └── TEACHER FORCING ADJUSTMENT
        │
        └── Decrease teacher_forcing_ratio over epochs
            (e.g., 1.0 → 0.9 → 0.8 → ... → 0.5)
```

---

## 🎬 **Complete Example: Translating "hello world"**

```
INPUT: "hello world"

STEP 1: PREPROCESSING
─────────────────────
Tokenize:        ['hello', 'world']
To indices:      [45, 231]
Pad:             [45, 231, 0, 0, ..., 0]  (length 20)
To tensor:       torch.tensor([[45, 231, 0, ..., 0]])  # shape [1, 20]

STEP 2: ENCODER
───────────────
Embed:           [1, 20, 256]
BiLSTM Layer 1:  [1, 20, 512]  (forward + backward concatenated)
BiLSTM Layer 2:  [1, 20, 512]
Final states:    h=[1, 256], c=[1, 256]

STEP 3: DECODER (Iterative)
────────────────────────────
t=0:  Input=[SOS] → LSTM → Attention → Output → Predict: "bonjour" (ID=7834)
t=1:  Input=7834  → LSTM → Attention → Output → Predict: "monde"   (ID=4521)
t=2:  Input=4521  → LSTM → Attention → Output → Predict: [EOS]     (ID=2)
Stop!

STEP 4: POST-PROCESSING
───────────────────────
Token IDs:       [7834, 4521]
Detokenize:      ["bonjour", "monde"]
Join:            "bonjour monde"

OUTPUT: "bonjour monde"
```

---

## 📈 **Attention Visualization**

```
Source (English):  "hello"   "world"   <PAD>  <PAD>  ...
                      ↑         ↑         ↑      ↑
Attention Weights:  [0.48]   [0.50]   [0.01]  [0.01] ...
                      │         │         │      │
                      └─────┬───┘         │      │
                            ↓             ↓      ↓
Target Word:            "bonjour"
                      (focuses on "hello" and "world")


Source (English):  "hello"   "world"   <PAD>  <PAD>  ...
                      ↑         ↑         ↑      ↑
Attention Weights:  [0.12]   [0.85]   [0.02]  [0.01] ...
                      │         │         │      │
                      └─────────┼─────────┘      │
                                ↓                ↓
Target Word:              "monde"
                      (focuses mainly on "world")
```

---

## 🔧 **Key Design Decisions**

| Component | Decision | Rationale |
|-----------|----------|-----------|
| **Encoder** | Bidirectional LSTM | Captures context from both directions |
| **Decoder** | Unidirectional LSTM | Can only see past during generation |
| **Attention** | Additive (Bahdanau) | Effective for seq2seq, interpretable |
| **Layers** | Multi-layer (1-3) | Deeper representations, better performance |
| **Dropout** | 0.1-0.2 | Prevents overfitting without hurting performance |
| **Teacher Forcing** | Scheduled (1.0→0.5) | Stable training, gradual independence |
| **Optimizer** | Adam | Adaptive learning rates, fast convergence |
| **Loss** | CrossEntropy | Standard for classification tasks |
| **Embedding** | 256-512 dim | Balance between capacity and computation |

---

This architecture represents a **complete, modern seq2seq system** with all the components needed for high-quality neural machine translation! 🚀
