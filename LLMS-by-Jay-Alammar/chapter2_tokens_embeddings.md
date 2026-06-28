# Chapter 2: Tokens and Embeddings — Comprehensive Study Guide

> **Source:** *Hands-On Large Language Models* by Alammar & Grootendorst  
> **Level:** AI Engineering / Data Science / ML Interviews  
> **Coverage:** Every concept, figure, code example, note, and technical term in the chapter

---

## Executive Summary

This chapter establishes the two foundational numerical pillars of modern LLMs: **tokens** (how text is discretized into processable units) and **embeddings** (how those units are mapped into meaningful vector spaces). Without understanding both, you cannot reason clearly about how LLMs work, why they succeed or fail on certain inputs, or how to build applications on top of them.

The chapter covers:
1. How tokenizers break raw text into token IDs that a model can consume
2. The major tokenization methods (BPE, WordPiece, SentencePiece, character, byte)
3. A side-by-side comparison of seven real tokenizers across model generations
4. The three design axes that govern tokenizer behavior
5. How LLMs store and generate **static token embeddings**
6. How Transformer-based models produce **contextualized word embeddings**
7. How **text (sentence) embeddings** power downstream applications
8. How the word2vec algorithm works (skip-gram + negative sampling)
9. How embeddings power recommendation systems beyond NLP

---

## Key Concepts Map

```
RAW TEXT
    │
    ▼
┌─────────────┐
│  TOKENIZER  │  ← Method (BPE / WordPiece / SentencePiece / char / byte)
│             │  ← Parameters (vocab size, special tokens, capitalization)
│             │  ← Training data domain (English / code / multilingual)
└──────┬──────┘
       │ token IDs  [1, 14350, 385, ...]
       ▼
┌─────────────────────┐
│  EMBEDDING MATRIX   │  ← one vector per token in vocabulary
│  (inside the LLM)   │  ← randomly init'd, learned during training
└──────┬──────────────┘
       │ static token embeddings
       ▼
┌──────────────────────────────┐
│  TRANSFORMER LAYERS          │  ← produce CONTEXTUALIZED token embeddings
│  (attention + feed-forward)  │
└──────┬───────────────────────┘
       │
  ┌────┴─────────────────────────────────┐
  │                                      │
  ▼                                      ▼
Token Embeddings                  Text Embeddings
(per-token context vectors)       (single vector for whole sentence/doc)
Used for: NER, classification,    Used for: semantic search, RAG,
extractive summarization,         topic modeling, recommendation
image-generation conditioning
```

---

## Detailed Q&A Guide

### Section 1 — LLM Tokenization

---

**Q1. What is a token, and why does it matter that LLMs generate one token at a time?**

**A:**  
A **token** is the smallest unit of text that an LLM processes. It can be a full word, a fragment of a word (subword), a single character, or a single byte depending on the tokenization scheme.

LLMs are *autoregressive*: they generate output **one token at a time**, appending each newly generated token to the context before predicting the next. This matters for several reasons:
- **Latency**: perceived speed of a model depends on tokens-per-second throughput.
- **Context window**: limits are measured in tokens, not characters or words.
- **Cost**: API pricing is tokens-in + tokens-out.
- **Debugging**: understanding the token breakdown explains surprising model behaviors (e.g., why "870" and "871" may be treated very differently).

Tokens are also the input representation: the prompt is tokenized before the model ever sees it.

---

**Q2. What exactly does a tokenizer return when it processes an input prompt?**

**A:**  
A tokenizer converts raw text into a list of **integer token IDs**. Each integer is an index into the tokenizer's vocabulary table (a lookup table mapping IDs ↔ text strings).

Example from the chapter:
```python
prompt = "Write an email apologizing to Sarah..."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# Returns: tensor([[ 1, 14350, 385, 4876, 27746, ...]])
```
Each integer corresponds to one token. The model never sees the raw string — only these integers, which are then used to look up embedding vectors from the model's embedding matrix.

**Why it matters:** The model's computation is entirely numerical. Text→tokens→IDs→embeddings is the pipeline that bridges human-readable language and matrix operations.

---

**Q3. Walk through the full code pipeline for tokenizing an input and generating text with an LLM.**

**A:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load model and tokenizer (they must be paired)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Step 2: Define prompt with special chat token
prompt = "Write an email apologizing to Sarah...<|assistant|>"

# Step 3: Tokenize → get integer IDs on GPU
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Step 4: Generate — model produces token IDs autoregressively
generation_output = model.generate(input_ids=input_ids, max_new_tokens=20)

# Step 5: Decode — IDs back to human-readable text
print(tokenizer.decode(generation_output[0]))
```

Key observations:
- `return_tensors="pt"` gives PyTorch tensors.
- `max_new_tokens=20` caps generation at 20 additional tokens.
- The model output contains *both* the input token IDs *and* the newly generated ones.
- `tokenizer.decode()` is the inverse of `tokenizer()` — it translates IDs back to text.

---

**Q4. How can you inspect individual tokens in a tokenized prompt?**

**A:**
```python
for id in input_ids[0]:
    print(tokenizer.decode(id))
```
Example output for the gardening email prompt:
```
<s>       ← special beginning-of-text token
Write
an
email
apolog    ← partial word
izing     ← continuation of "apologizing"
to
Sarah
...
<|assistant|>  ← special chat role token
```

Key observations:
- Some tokens are **complete words** (`Write`, `Sarah`).
- Some are **subwords** (`apolog`, `izing`, `trag`, `ic`).
- **Punctuation** gets its own token.
- There is **no separate space token**. Instead, a hidden special character at the start of a subword signals that it's a *continuation* (connected to the previous token). Tokens without this character are assumed to be preceded by a space.

---

**Q5. What are the three major factors that determine how a tokenizer breaks down text?**

**A:**

| Factor | Description |
|--------|-------------|
| **Tokenization Method** | The algorithm used (BPE, WordPiece, SentencePiece, char, byte). Chosen at model design time. |
| **Initialization Parameters** | Vocabulary size, special tokens, capitalization handling. |
| **Training Dataset** | Even with identical method and parameters, a tokenizer trained on English text differs from one trained on code or multilingual corpora. |

These three together fully determine the tokenizer's vocabulary and behavior.

---

**Q6. What are the four major tokenization granularities, and what are the trade-offs of each?**

**A:**

| Scheme | Description | Pros | Cons |
|--------|-------------|------|------|
| **Word tokens** | Each full word = one token | Simple, interpretable | Cannot handle new/OOV words; bloated vocab (apology ≠ apologize ≠ apologetic) |
| **Subword tokens** | Mix of full and partial words | Handles OOV by falling back to subparts; expressive; fits more text in context window (~3× vs character) | Counterintuitive splits |
| **Character tokens** | Each character = one token | Handles any new word | Very long sequences; harder to model; wastes context window |
| **Byte tokens** | Each unicode byte = one token | Truly tokenization-free; handles any script | Even longer sequences; complex |

**Subword tokenization** is the dominant approach in modern LLMs. It achieves a balance: a token for `apolog` plus suffix tokens (`-y`, `-ize`, `-etic`, `-ist`) covers the whole word family without bloating the vocabulary.

> **Note:** Some subword tokenizers (GPT-2, RoBERTa) include bytes *as a fallback* for characters not otherwise representable. This does **not** make them byte-level tokenizers; they only use bytes for a small subset of inputs.

---

**Q7. What is Byte Pair Encoding (BPE) and where is it used?**

**A:**  
**Byte Pair Encoding (BPE)** is a data compression algorithm adapted for tokenization. It was introduced to NLP in *"Neural machine translation of rare words with subword units"*.

**How it works (high-level):**
1. Start with a character-level vocabulary.
2. Count all adjacent character pairs in the training corpus.
3. Merge the most frequent pair into a new token.
4. Repeat until the vocabulary reaches the desired size.

**Used by:** GPT-2, GPT-4, StarCoder2, Phi-3/Llama 2, Galactica.

**Why it matters:** BPE gives a vocabulary that efficiently represents the training data. Common words become single tokens; rare words are split into familiar subparts.

---

**Q8. What is WordPiece and how does it differ from BPE?**

**A:**  
**WordPiece** is a subword tokenization method introduced in *"Japanese and Korean voice search"* and used by **BERT** models.

Like BPE, WordPiece starts with a character-level vocabulary and iteratively merges pairs. The difference is in the **merge criterion**:
- BPE merges the *most frequent* pair.
- WordPiece merges the pair that *maximizes the likelihood* of the training data under a language model.

WordPiece uses `##` to mark continuation tokens: `capital ##ization` means "capitalization" was split into two tokens, and `##ization` is a suffix.

---

**Q9. What is SentencePiece and which model uses it?**

**A:**  
**SentencePiece** (introduced in *"SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing"*) is a *tokenizer implementation* (a library) rather than a specific algorithm. It supports both BPE and the **unigram language model** method (from *"Subword regularization: Improving neural network translation models with multiple subword candidates"*).

Key feature: SentencePiece treats whitespace like any other character and does not require pre-tokenization into words. This makes it truly language-independent.

**Used by:** Flan-T5.

**Weakness seen in the chapter:** The Flan-T5 SentencePiece tokenizer has no newline or whitespace tokens, making it poor for code generation. It also replaces emojis and Chinese characters with `<unk>`.

---

### Section 2 — Comparing Real Tokenizers

---

**Q10. What test string is used to compare tokenizers in the chapter, and what properties does it probe?**

**A:**
```python
text = """
English and CAPITALIZATION
🎵鸟
show_tokens False None elif == >= else: two tabs:" " Three tabs: "   "
12.0*50=600
"""
```

Properties probed:

| Property | What it reveals |
|----------|----------------|
| Capitalization (`CAPITALIZATION`) | Whether the tokenizer is case-sensitive |
| Non-English (`🎵鸟`) | Multilingual/emoji support |
| Code keywords (`elif`, `==`, `>=`) | Code-awareness of vocabulary |
| Whitespace/indentation | Python/code indentation handling |
| Numbers (`12.0*50=600`) | Numeric tokenization strategy |
| Special tokens | Model-specific markers |

---

**Q11. Describe BERT (uncased, 2018) tokenizer behavior and its key limitations.**

**A:**
- **Method:** WordPiece | **Vocab size:** 30,522
- **Special tokens:** `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `[MASK]`

Output:
```
[CLS] english and capital ##ization [UNK] [UNK] show _ token ##s false none eli ##f = = > = else : ...
```

**Key observations:**
- All text is lowercased → loses capitalization information
- Newlines are removed → model is blind to line-break structure (e.g., chat logs)
- Emoji and Chinese characters → `[UNK]` (invisible to model)
- `##` marks continuation subwords
- Input is wrapped in `[CLS]` ... `[SEP]`

**`[CLS]` purpose:** Classification token — its output embedding is used for sentence-level classification tasks.  
**`[SEP]` purpose:** Separator — separates two input sequences (e.g., query and passage in reranking, Chapter 8).

---

**Q12. How does BERT (cased) differ from BERT (uncased)?**

**A:**
- **Vocab size:** 28,996 (slightly smaller)
- Preserves capitalization.
- `CAPITALIZATION` → 8 tokens: `CA ##PI ##TA ##L ##I ##Z ##AT ##ION`
- This is highly inefficient for all-caps words.

**When to use cased vs uncased:** Use cased when capitalization carries meaning (e.g., Named Entity Recognition: "Apple" the company vs "apple" the fruit). Use uncased when case is irrelevant noise.

---

**Q13. How does the GPT-2 (2019) tokenizer improve over BERT?**

**A:**
- **Method:** BPE | **Vocab size:** 50,257 | **Special token:** `<|endoftext|>`

Key improvements:
- **Newlines are preserved** as tokens → model can understand structure
- **Capitalization preserved** → `CAPITALIZATION` in 4 tokens (vs 8 for BERT cased)
- **Emoji and CJK characters** are tokenized into byte sequences (not `[UNK]`), reconstructible via decode
- No `[CLS]`/`[SEP]` wrapping (GPT is a decoder-only model, not encoder)

Limitations:
- Tabs → 2 tokens; 4 spaces → 3 tokens (suboptimal for Python indentation)

---

**Q14. What are the notable properties of the GPT-4 tokenizer and how does it improve over GPT-2?**

**A:**
- **Method:** BPE | **Vocab size:** ~100,000+
- **Special tokens:** `<|endoftext|>`, fill-in-the-middle tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`)

Improvements over GPT-2:
1. **Whitespace:** 4 spaces → **1 token** (GPT-4 has a specific token for every whitespace sequence up to 83 characters). Crucial for Python indentation.
2. **Code keywords:** `elif` gets its own dedicated token.
3. **Vocabulary efficiency:** Most words use fewer tokens (`CAPITALIZATION` → 2 tokens vs 4; `tokens` → 1 token vs 3).

**Fill-in-the-middle (FIM):** These special tokens enable the model to generate a completion given *both* the text before *and* after the cursor position — important for code editors (e.g., GitHub Copilot).

---

**Q15. What is unique about the Flan-T5 tokenizer?**

**A:**
- **Method:** SentencePiece (BPE + unigram LM) | **Vocab size:** 32,100
- **Special tokens:** `<unk>`, `<pad>`, `</s>`

Unique weaknesses:
- **No newline tokens** → model cannot understand text structure based on line breaks
- **No whitespace/indentation tokens** → bad for code
- **Emoji + Chinese** → `<unk>` (invisible to model)

This reflects that Flan-T5 was optimized for structured NLP tasks (question answering, summarization) on English text, not code or multilingual tasks.

---

**Q16. What makes StarCoder2 (2024) tokenizer distinctive?**

**A:**
- **Method:** BPE | **Vocab size:** 49,152
- **Special tokens:** FIM tokens (`<fim_prefix>`, `<fim_middle>`, `<fim_suffix>`, `<fim_pad>`), plus file/repo management tokens: `<filename>`, `<reponame>`, `<gh_stars>`

Distinctive features:
1. **Each digit gets its own token:** `600` → `6 0 0`. Hypothesis: better mathematical representation. In GPT-2, `870` is one token but `871` is two (`8` + `71`) — inconsistent and confusing for arithmetic.
2. **Whitespace sequences → single tokens** (like GPT-4): good for Python indentation.
3. **Repository/file tokens:** A function in one file may call code defined in another. These tokens help the model track cross-file references.

---

**Q17. What is special about the Galactica tokenizer?**

**A:**
- **Method:** BPE | **Vocab size:** 50,000
- Focused on **scientific knowledge** (papers, reference material, knowledge bases)

Special tokens:
- `[START_REF]` / `[END_REF]` — wrap citations: `...long short-term memory [START_REF]Long Short-Term Memory, Hochreiter[END_REF]`
- `<work>` — used for **chain-of-thought reasoning** (the model can "think" inside `<work>` tags before producing a final answer)

Numeric tokenization: Each digit is its own token (like StarCoder2).  
Whitespace: Single token per whitespace sequence — the only tokenizer to also do this for **tabs**.

---

**Q18. What is notable about Phi-3 (and Llama 2) tokenizer?**

**A:**
- **Method:** BPE | **Vocab size:** 32,000
- Reuses Llama 2's tokenizer and adds special tokens

Special tokens:
- `<|endoftext|>`
- **Chat tokens:** `<|user|>`, `<|assistant|>`, `<|system|>` — these mark speaker turns in a conversation, reflecting that chat LLMs are a dominant use case post-2023.

Each digit gets its own token (consistent with code/math-focused models).

---

**Q19. Summarize the complete tokenizer comparison table.**

**A:**

| Model | Method | Vocab Size | Capitalization | Emoji/CJK | Newlines | Indentation | Digits | Special Tokens |
|-------|--------|-----------|----------------|-----------|----------|-------------|--------|----------------|
| BERT (uncased) | WordPiece | 30,522 | Lowercased | [UNK] | Removed | Poor | Combined | CLS, SEP, PAD, UNK, MASK |
| BERT (cased) | WordPiece | 28,996 | Preserved (verbose) | [UNK] | Removed | Poor | Combined | Same as above |
| GPT-2 | BPE | 50,257 | Preserved | Byte fallback | Preserved | Poor (multi-token) | Combined | endoftext |
| Flan-T5 | SentencePiece | 32,100 | Preserved | [UNK] | Removed | Poor | Combined | unk, pad, /s |
| GPT-4 | BPE | ~100,000 | Preserved | Byte fallback | Preserved | Single token | Combined | endoftext, FIM |
| StarCoder2 | BPE | 49,152 | Preserved | Byte fallback | Preserved | Single token | Individual | FIM, repo/file |
| Galactica | BPE | 50,000 | Preserved | Byte fallback | Preserved | Single token (incl. tabs) | Individual | citations, work |
| Phi-3/Llama 2 | BPE | 32,000 | Preserved | Byte fallback | Preserved | Single token | Individual | chat role tokens |

---

### Section 3 — Tokenizer Design Decisions

---

**Q20. Explain the three major groups of tokenizer design choices in depth.**

**A:**

**Group 1: Tokenization Method**
The algorithm that decides which text fragments become tokens. BPE is currently dominant because it efficiently balances vocabulary coverage with sequence length.

**Group 2: Tokenizer Parameters**

| Parameter | Description | Example values |
|-----------|-------------|----------------|
| Vocabulary size | How many distinct tokens | 30K, 50K, 100K |
| Special tokens | Task/model-specific markers | `[CLS]`, `<|user|>`, `[START_REF]` |
| Capitalization | Lowercase all? Preserve? | BERT-uncased lowers; GPT-4 preserves |

**Group 3: Training Data Domain**
The tokenizer optimization procedure finds the vocabulary that best compresses the target dataset. Consequences:
- A code-trained tokenizer gives `elif` its own token (common in Python); a text-only tokenizer splits it as `el` + `if`.
- A multilingual tokenizer allocates vocabulary space to CJK/Arabic characters; an English-only tokenizer replaces them with `[UNK]`.
- A math-focused tokenizer tokenizes digits individually for consistent numeric representation.

**Code example of how indentation tokenization matters:**

*Text-focused tokenizer (suboptimal):*
```
def   add_  numbers  (a  ,   b  ):
   ....  # 4 spaces = multiple tokens
```

*Code-focused tokenizer (optimal):*
```
def  add_numbers  (a, b):
....  # 4 spaces = single token
```
The single-token indentation makes it easier for the model to track Python scope.

---

**Q21. What are special tokens and why are they important?**

**A:**  
Special tokens are reserved tokens that serve a *structural or functional role* rather than representing natural language text.

| Token | Purpose | Used by |
|-------|---------|---------|
| `[CLS]` | Classification marker; output embedding used for sentence classification | BERT |
| `[SEP]` | Separator between two input sequences | BERT |
| `[PAD]` | Padding to fill unused positions in fixed-length input | BERT, Flan-T5 |
| `[UNK]` | Unknown token (character/word not in vocabulary) | BERT, Flan-T5 |
| `[MASK]` | Hides tokens during masked language model training | BERT |
| `<s>` | Beginning of text | Galactica, Phi-3 |
| `</s>` | End of text | Flan-T5, Galactica |
| `<|endoftext|>` | End of text / generation stop signal | GPT-2/4, StarCoder2 |
| `<|fim_prefix|>` etc. | Fill-in-the-middle: generate given prefix + suffix context | GPT-4, StarCoder2 |
| `<|user|>`, `<|assistant|>`, `<|system|>` | Conversation turns | Phi-3/Llama 2 |
| `[START_REF]`, `[END_REF]` | Citation demarcation | Galactica |
| `<work>` | Chain-of-thought reasoning block | Galactica |
| `<filename>`, `<reponame>` | Code file/repo identification | StarCoder2 |

---

### Section 4 — Token Embeddings

---

**Q22. What is a token embedding and where is it stored inside an LLM?**

**A:**  
A **token embedding** is a dense vector of floating-point numbers that represents a token. Inside an LLM, these are stored in an **embedding matrix** of shape `(vocab_size, embedding_dim)`.

- Each row corresponds to one token in the vocabulary.
- The embedding matrix is a **learned parameter** of the model — randomly initialized before training and updated through backpropagation.
- When a token ID arrives as input, the model performs a simple **table lookup** (index into the matrix) to retrieve that token's embedding vector.

When you download a pretrained LLM, the embedding matrix is included as part of the model weights file.

**Why this matters:** The embedding matrix is the model's "dictionary" of meaning. After training on vast corpora, each vector encodes rich semantic and syntactic information about its token.

---

**Q23. Why is a pretrained language model inseparably linked to its tokenizer?**

**A:**  
Because the model's embedding matrix has exactly `vocab_size` rows — one for each token ID in *that specific tokenizer's vocabulary*. The IDs are meaningless without knowing which token they correspond to.

If you swapped the tokenizer:
- Token ID 3822 would refer to a *different* token.
- The wrong embedding vector would be retrieved.
- Model outputs would be nonsense.

Therefore: **tokenizer + model are a matched pair**. Reusing a model with a different tokenizer requires re-training the embedding matrix from scratch (at minimum).

---

**Q24. What are contextualized word embeddings and how do they differ from static embeddings?**

**A:**

| Property | Static Embeddings (word2vec, GloVe) | Contextualized Embeddings (LLMs) |
|----------|-------------------------------------|----------------------------------|
| One vector per word? | Yes — same vector always | No — vector depends on surrounding context |
| "Bank" in "river bank" vs "bank account" | Same vector | Different vectors |
| Source | Pre-trained; frozen | Output of a live forward pass |
| Applications | Word similarity, analogies | NER, extractive summarization, classification, image generation |

**How LLMs produce contextualized embeddings:**
After the token embeddings are looked up from the embedding matrix, they pass through multiple Transformer layers (attention + feed-forward). Each layer mixes information from all tokens in the sequence, so the final output vector for each token encodes *both* the token's identity *and* its context.

---

**Q25. Show the code for generating contextualized word embeddings with DeBERTa v3.**

**A:**
```python
from transformers import AutoModel, AutoTokenizer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

# Tokenize
tokens = tokenizer('Hello world', return_tensors='pt')

# Forward pass → contextualized embeddings
output = model(**tokens)[0]

print(output.shape)  # torch.Size([1, 4, 384])
```

**Decoding the shape `[1, 4, 384]`:**
- `1` = batch size (1 sentence)
- `4` = number of tokens: `[CLS]`, `Hello`, `world`, `[SEP]`
- `384` = embedding dimensionality (number of values per token vector)

**DeBERTa v3** is described in the paper *"DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training gradient-disentangled embedding sharing"* — at the time of writing, one of the best-performing small models for token embeddings.

---

**Q26. What are text (sentence) embeddings and how are they produced?**

**A:**  
A **text embedding** (also called sentence embedding or document embedding) is a **single vector** representing an entire piece of text — a sentence, paragraph, or document.

**Common production methods:**
1. **Average pooling:** Average the token-level output vectors from the LLM.
2. **[CLS] token pooling:** Use only the `[CLS]` token's output (common with BERT-style models).
3. **Specialized training:** Train a model explicitly to produce high-quality sentence-level vectors (e.g., Sentence-BERT / `sentence-transformers`).

**Code example:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
vector = model.encode("Best movie ever!")
print(vector.shape)  # (768,)
```

A 768-dimensional vector now represents the entire sentence's meaning.

**Applications:** semantic search, RAG (retrieval-augmented generation), topic modeling, clustering, classification.

---

### Section 5 — Word2Vec and Contrastive Training

---

**Q27. What is the word2vec algorithm and what are its two core ideas?**

**A:**  
**Word2vec** (introduced in *"Efficient estimation of word representations in vector space"*) is a neural network-based algorithm that trains **word embeddings** using a classification task over text.

**Core idea 1: Skip-gram**  
A sliding window moves over the training corpus. For each central word, neighboring words within the window are treated as *positive pairs* (they co-occur in context).

**Core idea 2: Negative Sampling**  
Since all training examples would have label `1` (positive pairs), random words from the vocabulary are sampled and labeled `0` (negative examples). The model must distinguish true neighbors from random words.

**Training objective:** Train a neural network that takes two word embeddings and outputs `1` if the words tend to appear in the same context, `0` otherwise. Through training, the embeddings are updated to capture semantic relationships.

---

**Q28. Walk through the skip-gram training example used in the chapter.**

**A:**  
Text: `"Thou shalt not make a machine in the likeness of a human mind"` (Dune, Frank Herbert)

With window size = 2:

**Step 1:** Slide the window. For central word `"not"`:
- Positive pairs: (`not`, `Thou`), (`not`, `shalt`), (`not`, `make`), (`not`, `a`)

**Step 2:** Add negative examples:
- (`not`, `pizza`) → label 0
- (`not`, `Jupiter`) → label 0
- etc.

**Step 3:** Train a neural network on this dataset. The network:
1. Looks up the embedding for word 1 (the central word)
2. Looks up the embedding for word 2 (neighbor or random)
3. Computes their interaction
4. Predicts 0 or 1

**Step 4:** Backpropagate: if wrong, update both embeddings. Over millions of examples, semantically similar words (those that appear in similar contexts) drift toward each other in the embedding space.

**Result:** `king` is near `queen`, `prince`, `emperor` because they share contexts like "the ___ ruled" or "the ___ ordered".

---

**Q29. What is noise-contrastive estimation and how does it relate to negative sampling?**

**A:**  
**Noise-contrastive estimation (NCE)** is a statistical method (from *"Noise-contrastive estimation: A new estimation principle for unnormalized statistical models"*) that replaces computing a full softmax over the vocabulary (expensive) with a binary classification between real data and sampled noise.

Word2vec's **negative sampling** is inspired by NCE: instead of computing probabilities over all ~50,000 vocabulary words for each training step, you sample a small number (e.g., 5–50) of random "noise" words and only update those embeddings.

**Why it matters:** This makes training tractable on large corpora and is one of the reasons word2vec was so influential. The same idea — contrastive learning with positive and negative pairs — reappears in Chapter 10 for fine-tuning LLMs for retrieval.

---

**Q30. How are word2vec embeddings used for music recommendation?**

**A:**  
The chapter presents an elegant analogy:
- **Word → Song**: treat each song ID as a "word"
- **Sentence → Playlist**: treat each playlist as a "sentence"

**Training:**
```python
from gensim.models import Word2Vec

# playlists: list of lists of song ID strings
model = Word2Vec(
    playlists, vector_size=32, window=20, negative=50, min_count=1, workers=4
)
```

**Querying:**
```python
model.wv.most_similar(positive=str(song_id))
```

**Results:**
- Nearest neighbors to Michael Jackson's "Billie Jean" → Madonna, Prince, other MJ songs
- Nearest neighbors to 2Pac's "California Love" → Nas, Notorious B.I.G., Snoop Dogg
- Nearest neighbors to Metallica's "Fade To Black" → Van Halen, Dio, Guns N' Roses, Judas Priest

Songs that co-appear in playlists end up near each other in the embedding space — just like words that appear in similar contexts.

**Key insight:** Embeddings are a general-purpose tool for **any domain where sequence co-occurrence captures similarity**. Other applications include: user session data, e-commerce product sequences, protein sequences, network graphs.

---

**Q31. How does the word2vec embedding update process work step by step?**

**A:**

1. **Initialize:** Create an embedding matrix of shape `(vocab_size, embedding_dim)`. Fill with random values.
2. **For each training example** (word pair + label):
   a. Look up embedding for word 1 → vector **v₁**
   b. Look up embedding for word 2 → vector **v₂**
   c. Feed through a neural network (often just a dot product + sigmoid)
   d. Compute prediction: `ŷ = sigmoid(v₁ · v₂)`
   e. Compare to true label `y` (0 or 1)
   f. Compute loss (binary cross-entropy)
   g. Backpropagate: update both **v₁** and **v₂** to reduce the loss
3. **After millions of examples:** Embeddings of semantically related words cluster together.

---

**Q32. How can you use pretrained GloVe embeddings with Gensim?**

**A:**
```python
import gensim.downloader as api

# Download 50-dimensional GloVe trained on Wikipedia + Gigaword (66MB)
model = api.load("glove-wiki-gigaword-50")

# Find nearest neighbors of "king"
model.most_similar([model['king']], topn=11)
```

Output (cosine similarity):
```
[('king', 1.00), ('prince', 0.82), ('queen', 0.78), ('emperor', 0.77),
 ('son', 0.77), ('uncle', 0.76), ('kingdom', 0.75), ('throne', 0.75),
 ('brother', 0.75), ('ruler', 0.74)]
```

Other available models include `"word2vec-google-news-300"`.

---

**Q33. What is the significance of the "king - man + woman = queen" analogy in word embeddings?**

**A:**  
*(Chapter mentions the broader concept; this is an expansion.)*  
This famous analogy emerges because word2vec embeddings capture **relational structure**:
- `king` vector − `man` vector + `woman` vector ≈ `queen` vector

This works because word2vec encodes gender relationships, royalty relationships, etc. as consistent *directions* in the vector space. The embedding space has linear structure that reflects real-world semantic relationships.

This was one of the most surprising and celebrated properties of word embeddings when discovered, validating the idea that dense vectors can capture meaning.

---

### Section 6 — The Full Picture

---

**Q34. What is the complete pipeline from raw text to LLM output?**

**A:**

```
Raw Text (string)
     │
     ▼  [Tokenizer]
Token IDs (list of integers)  e.g., [1, 14350, 385, ...]
     │
     ▼  [Embedding Matrix lookup]
Static Token Embeddings (matrix: n_tokens × embed_dim)
     │
     ▼  [Transformer Layers: Attention + FFN]
Contextualized Token Embeddings (matrix: n_tokens × embed_dim)
     │
     ├──► [Pool / average] → Text Embedding (1 × embed_dim)
     │                       Used for: semantic search, RAG, classification
     │
     └──► [Language Model Head] → Logits over vocabulary → Next Token ID
                                  Used for: text generation
```

---

**Q35. How is the output of a contextualized embedding model used for downstream tasks?**

**A:**

| Task | How embeddings are used |
|------|------------------------|
| Named Entity Recognition (NER) | Per-token classification head on top of contextualized token embeddings |
| Extractive summarization | Identify tokens/sentences with highest relevance scores |
| Text classification | Pool sentence embedding → linear classifier |
| Semantic search | Embed query + documents → cosine similarity |
| RAG | Embed chunks → retrieve relevant chunks → feed to generation model |
| Image generation (DALL·E, Midjourney, Stable Diffusion) | Use contextualized text embeddings to condition image synthesis |
| Reranking | Cross-encoder with `[SEP]` separating query and passage |

---

## Technical Terms Glossary

*(Alphabetical order)*

---

**Autoregressive**  
- **Definition:** A generation paradigm where each token is conditioned on all previously generated tokens.  
- **Intuition:** The model reads everything it has written so far before writing the next word.  
- **Example:** GPT models.  
- **Interview Q:** "Why do LLMs generate text slowly?" Because they're autoregressive — each token requires a full forward pass.

---

**BPE (Byte Pair Encoding)**  
- **Definition:** A subword tokenization algorithm that iteratively merges the most frequent pair of adjacent tokens.  
- **Intuition:** Start with characters; keep combining the most common pairs until you reach your desired vocabulary size.  
- **Used by:** GPT-2, GPT-4, Llama 2, Phi-3, StarCoder2, Galactica.  
- **Misconception:** BPE is not the same as WordPiece. WordPiece uses a language-model likelihood criterion; BPE uses raw frequency.

---

**Byte Tokens**  
- **Definition:** Tokenization at the level of individual Unicode bytes.  
- **Intuition:** Ultimate fallback — can represent any character in any language.  
- **Example:** ByT5, CANINE models.  
- **Trade-off:** Very long sequences; harder to model long-range dependencies.

---

**Character Tokens**  
- **Definition:** Each individual character is one token.  
- **Trade-off:** Handles all new words; but sequences are ~3× longer than subword, wasting context window.

---

**[CLS] Token**  
- **Definition:** A special BERT token prepended to input; its output embedding is used for sentence-level classification.  
- **Intuition:** A "summary slot" — the model aggregates global sentence information into this position.

---

**Context Window**  
- **Definition:** Maximum number of tokens an LLM can process in a single forward pass (both input + output).  
- **Measured in:** Tokens (not words or characters).  
- **Why it matters:** Limits how much text can be provided to the model at once; subword tokenization fits ~3× more text than character tokenization for the same window.

---

**Contrastive Training**  
- **Definition:** Training paradigm where a model learns to pull together positive pairs and push apart negative pairs in embedding space.  
- **Used in:** word2vec (skip-gram + negative sampling), CLIP (image-text pairs), sentence embedding fine-tuning (Chapter 10).

---

**Contextualized Word Embeddings**  
- **Definition:** Per-token embeddings that vary based on surrounding context, produced by running a full Transformer forward pass.  
- **Contrast with:** Static embeddings (word2vec, GloVe), which give the same vector regardless of context.  
- **Example:** "bank" in "river bank" has a different contextualized vector than "bank" in "bank account."

---

**DeBERTa v3**  
- **Definition:** A highly efficient Transformer encoder model described in *"DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training gradient-disentangled embedding sharing."*  
- **Strength:** Among the best per-performance-per-parameter models for token embeddings.

---

**Embedding**  
- **Definition:** A dense vector (array of floats) that represents a discrete entity (token, sentence, image, user, song, etc.) in a continuous vector space.  
- **Intuition:** Semantically similar objects have nearby vectors.  
- **Mathematical meaning:** A point in ℝᵈ (d-dimensional real space). Similarity measured by cosine similarity or dot product.  
- **Example:** `"Hello world"` → vector of 768 floats.

---

**Embedding Matrix**  
- **Definition:** A learnable matrix of shape `(vocab_size, embedding_dim)` stored inside the LLM. Row i = embedding for token i.  
- **How it's used:** Token ID → row lookup → embedding vector.

---

**Fill-in-the-Middle (FIM)**  
- **Definition:** Special tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`) that enable an LLM to generate a completion given both the text before *and* after the cursor.  
- **Use case:** Code completion in editors where context exists on both sides.

---

**GloVe (Global Vectors)**  
- **Definition:** A word embedding method that uses global co-occurrence matrix statistics rather than local window predictions.  
- **Relation:** Similar in spirit to word2vec; both produce static word vectors.

---

**Negative Sampling**  
- **Definition:** In word2vec training, adding randomly sampled words as negative (label=0) examples to prevent the model from always outputting 1.  
- **Inspiration:** Noise-contrastive estimation.

---

**RAG (Retrieval-Augmented Generation)**  
- **Definition:** Combining a retrieval system (semantic search over a knowledge base) with a generative LLM. The retrieved context is appended to the prompt.  
- **Why it exists:** LLMs hallucinate facts; RAG grounds generation in retrieved real documents.

---

**[SEP] Token**  
- **Definition:** BERT's separator token, used to divide two input sequences.  
- **Example:** In reranking, `[CLS] query [SEP] passage [SEP]` is the input structure.

---

**SentencePiece**  
- **Definition:** A tokenizer *library* (not an algorithm per se) that supports BPE and unigram LM. Treats whitespace as a regular character.  
- **Used by:** Flan-T5.

---

**sentence-transformers**  
- **Definition:** A Python library wrapping pre-trained embedding models optimized for sentence-level representations.  
- **Key model:** `all-mpnet-base-v2` → 768-dimensional sentence embeddings.

---

**Skip-gram**  
- **Definition:** In word2vec, the method of treating a center word and its context neighbors as positive training pairs.  
- **Window size:** Controls how many neighbors to consider on each side.

---

**Static Embeddings**  
- **Definition:** Word embeddings (e.g., word2vec, GloVe) where each word always maps to the same vector regardless of context.  
- **Limitation:** Cannot disambiguate polysemous words.

---

**Subword Tokens**  
- **Definition:** Tokens that can be full words or fragments of words. The dominant scheme in modern LLMs.  
- **Advantages:** Handles OOV words; more expressive vocabulary; fits ~3× more text than character tokens.

---

**Token ID**  
- **Definition:** An integer index that uniquely identifies a token in a tokenizer's vocabulary.  
- **Range:** 0 to vocab_size − 1.

---

**Tokenizer**  
- **Definition:** A module that converts raw text to token IDs (encoding) and token IDs back to text (decoding).  
- **Two roles:** Processes LLM inputs *and* converts LLM output IDs back to readable text.

---

**Vocabulary**  
- **Definition:** The complete set of tokens a tokenizer knows, stored as a lookup table mapping token IDs to token strings.  
- **Size range:** 30K (BERT) → 32K (Phi-3) → 50K (GPT-2) → 100K+ (GPT-4).

---

**WordPiece**  
- **Definition:** Subword tokenization method used by BERT. Merges token pairs based on language model likelihood (vs. BPE's frequency criterion).  
- **Marker:** `##` prefix marks continuation subwords.

---

**word2vec**  
- **Definition:** Neural network algorithm for training static word embeddings via skip-gram and negative sampling.  
- **Paper:** *"Efficient estimation of word representations in vector space."*  
- **Successor:** Contextualized embeddings from Transformer LLMs have largely replaced word2vec for NLP tasks, but the same ideas are used in recommendation systems, graph embeddings, etc.

---

## Important Tables

### Table 1: Tokenization Method Comparison

| Method | Algorithm | Key Criterion | Uses `##`? | Models |
|--------|-----------|--------------|-----------|--------|
| BPE | Merge most frequent pair | Frequency | No (uses hidden space char) | GPT-2/4, Llama, StarCoder2 |
| WordPiece | Merge highest LM likelihood pair | Likelihood | Yes | BERT |
| SentencePiece | Library supporting BPE or Unigram LM | Varies | No | Flan-T5 |
| Character | No merging | N/A | N/A | Some older models |
| Byte | Unicode byte-level | N/A | N/A | ByT5, CANINE |

---

### Table 2: Vocabulary Size Evolution

| Year | Model | Vocab Size |
|------|-------|-----------|
| 2018 | BERT | 30,522 / 28,996 |
| 2019 | GPT-2 | 50,257 |
| 2022 | Flan-T5 | 32,100 |
| 2023 | GPT-4 | ~100,000+ |
| 2024 | StarCoder2 | 49,152 |
| — | Galactica | 50,000 |
| — | Phi-3/Llama 2 | 32,000 |

---

### Table 3: Embedding Types at a Glance

| Type | Scope | Context-Aware | Output Shape | Applications |
|------|-------|--------------|-------------|--------------|
| Static word embedding | One word | No | (embed_dim,) | Word similarity, analogies, reco systems |
| Static token embedding | One token | No | (embed_dim,) | LLM input layer |
| Contextualized token embedding | One token in context | Yes | (n_tokens, embed_dim) | NER, extractive summarization |
| Text/sentence embedding | Full sentence/doc | Yes | (embed_dim,) | Semantic search, RAG, classification |

---

## Exam Questions

### Beginner Questions

1. What is a token? Give three examples of different types of tokens.
2. What does a tokenizer return when it processes the string `"Hello world"`?
3. What is the difference between `tokenizer.encode()` and `tokenizer.decode()`?
4. What does `max_new_tokens=20` control in the `model.generate()` call?
5. What is a vocabulary in the context of tokenization?
6. Name two special tokens used by BERT and explain their purpose.
7. What does the `<|endoftext|>` token signal to a GPT model?
8. What does it mean for an LLM to generate text "one token at a time"?
9. What is the shape of the output of `SentenceTransformer.encode("Best movie ever!")`?
10. What is an embedding matrix and where is it stored?

### Intermediate Questions

1. Explain the three major factors that determine tokenizer behavior.
2. Compare word tokenization and subword tokenization. Why did NLP move to subword?
3. Why is a model permanently linked to its tokenizer and cannot easily switch?
4. What is Byte Pair Encoding (BPE)? How does it build its vocabulary?
5. What are fill-in-the-middle (FIM) tokens and what problem do they solve?
6. How does the GPT-4 tokenizer handle whitespace differently from GPT-2? Why does this matter?
7. What is the difference between static and contextualized word embeddings?
8. Why does StarCoder2 assign each digit its own token?
9. Explain skip-gram in word2vec. How are training examples generated?
10. How does the song recommendation system use word2vec? What is the analogy?
11. How many dimensions does the `all-mpnet-base-v2` sentence embedding model produce?
12. What is the `[CLS]` token and how is it used for classification in BERT?
13. What does `output.shape = torch.Size([1, 4, 384])` mean in the DeBERTa example?
14. What is the difference between BPE and WordPiece?
15. Why does the Flan-T5 tokenizer perform poorly on code tasks?

### Advanced Questions

1. Trace the full pipeline from a raw string to a contextualized embedding vector, including all intermediate representations and which component produces each.
2. You need to build an LLM that handles mathematical reasoning with multi-digit numbers. What tokenization choices would you make, and why?
3. Explain how contrastive training in word2vec relates to the broader idea of contrastive learning in modern ML (CLIP, SimCLR, etc.).
4. A user reports that your LLM handles English well but treats emojis as `[UNK]`. What is the likely cause and how would you fix it at the tokenizer level?
5. Why would a code-generation model benefit from having a single token for sequences of whitespace characters (e.g., 4 spaces)? Explain in terms of what the model needs to learn.
6. Describe noise-contrastive estimation and explain why it makes word2vec training tractable.
7. How would you design a tokenizer for a scientific LLM (like Galactica)? What special tokens would you add and why?
8. Explain why the context window limit is measured in tokens, not characters, and what the practical implications are for users.
9. Compare the embedding strategies for recommendation systems (word2vec on playlists) with those used for semantic search (sentence-transformers). When would you use each?
10. How would adding chat role tokens (`<|user|>`, `<|assistant|>`) to a tokenizer change model behavior compared to encoding role information in plain text?

### Interview Questions

1. **"Explain tokens and tokenization to a non-technical product manager."**  
   *(Expected: Clear analogy, explanation of why it matters for cost/speed, no jargon.)*

2. **"You're seeing weird behavior where your LLM treats '870' and '871' very differently. What might be the cause?"**  
   *(Expected: Tokenization — '870' might be a single token while '871' is two tokens '8'+'71'.)*

3. **"When would you choose a character-level tokenizer over subword?"**  
   *(Expected: Very small vocabulary domains, highly morphologically complex languages, when OOV handling is critical.)*

4. **"Why can't you just use the same tokenizer across all LLMs?"**  
   *(Expected: Each model's embedding matrix is trained to match its specific tokenizer's vocabulary; mixing them produces nonsense.)*

5. **"How does word2vec relate to modern LLM embeddings?"**  
   *(Expected: Static vs. contextualized; word2vec preceded LLMs; same skip-gram/negative sampling ideas used in contrastive fine-tuning today.)*

6. **"How would you build a recommendation system using embeddings?"**  
   *(Expected: Treat items as words, sequences (playlists, sessions) as sentences; train word2vec; use cosine similarity for nearest neighbors.)*

7. **"What is the [CLS] token and why does BERT need it for classification?"**  
   *(Expected: Placeholder token that aggregates global sequence information; its output embedding is fed to a classification head.)*

8. **"A user complains your LLM costs too much to run. You have no GPU budget to change the model. What tokenizer-level levers do you have?"**  
   *(Expected: Check if tokenizer is efficient; use a model with larger vocabulary (fewer tokens per text = shorter sequences = cheaper); optimize prompt wording.)*

---

## Common Mistakes and Misconceptions

| Misconception | Reality |
|--------------|---------|
| "Tokens are the same as words." | Tokens are often *parts* of words (subwords), and punctuation gets its own token. One word can be multiple tokens. |
| "All LLMs use the same tokenizer." | Each model family uses a different tokenizer with different vocabulary, method, and special tokens. |
| "You can swap tokenizers between compatible models." | No — the embedding matrix row indices are meaningless without the specific tokenizer they were trained with. |
| "A bigger vocabulary is always better." | Larger vocabulary → fewer tokens per text (good for context window), but increases model size and memory. Trade-offs exist. |
| "BERT and GPT use the same tokenization approach." | BERT uses WordPiece with `[CLS]`/`[SEP]` wrapping; GPT uses BPE with no such wrapping. |
| "Byte-level tokenizers are the same as tokenizers that include bytes as a fallback." | Byte-level tokenizers (ByT5) represent *everything* as bytes. GPT-2/RoBERTa include bytes only for unrepresentable characters. |
| "Word2vec embeddings are contextualized." | No — word2vec produces static embeddings. "Bank" always gets the same vector regardless of meaning. |
| "The embedding matrix is fixed after download." | It is fixed for inference, but it is a *trained* parameter — it was learned during pretraining and is stored in the weights file. |
| "Lowercasing is always neutral for tokenizers." | BERT uncased loses named entity information (Apple vs apple). It's a design choice with real semantic consequences. |
| "The context window is measured in characters." | No — it's measured in tokens. 1 token ≈ 4 characters on average for English subword models. |
| "Text embeddings and token embeddings are the same thing." | Token embeddings are per-token vectors; text embeddings are a *single* vector representing an entire sentence/document, often via pooling. |

---

## Chapter Summary

**Tokenization** is the first and essential preprocessing step for any LLM. Raw text → tokens → integer IDs → embedding vectors. Tokenizers are trained components (not simple rules) and must be paired with their associated model.

The major tokenization schemes — **BPE**, **WordPiece**, **SentencePiece**, **character**, and **byte** — each make different trade-offs. **Subword tokenization (especially BPE)** dominates modern LLMs because it balances vocabulary expressivity, OOV handling, and sequence length efficiency.

Three design axes govern tokenizer behavior: the **algorithm**, the **initialization parameters** (vocab size, special tokens, capitalization), and the **training data domain**. Domain specialization explains why code-focused tokenizers handle whitespace and numbers differently than general-purpose ones.

**Token embeddings** are dense vectors stored in the model's embedding matrix — one per vocabulary token — learned during pretraining. After the tokenizer produces IDs, the model performs a table lookup to retrieve these vectors.

**Contextualized word embeddings**, produced by Transformer layers operating on all token embeddings jointly, are the richer, context-dependent representations used for tasks like NER and extractive summarization.

**Text (sentence) embeddings** — single vectors covering an entire sentence — power semantic search, RAG, classification, and clustering.

**Word2vec** (skip-gram + negative sampling) showed that meaningful geometry could emerge from a simple classification task over text. The same contrastive learning principle is foundational to modern embedding fine-tuning and cross-modal models (CLIP). Word2vec-style training extends naturally to recommendation systems by treating item sequences (playlists, browsing sessions) as "sentences."

---

## Knowledge Checklist

- [ ] I can explain what a token is and give examples of word, subword, character, and byte tokens.
- [ ] I understand the full pipeline: raw text → tokenizer → token IDs → embedding matrix → LLM.
- [ ] I can run the code to tokenize a prompt and decode individual token IDs with HuggingFace.
- [ ] I can explain BPE, WordPiece, and SentencePiece and their differences.
- [ ] I know why a model and its tokenizer must remain paired.
- [ ] I can describe the three design axes of tokenizer behavior (method, parameters, domain).
- [ ] I can explain at least 5 special tokens and their purposes.
- [ ] I can compare the tokenization strategies of BERT, GPT-2, GPT-4, Flan-T5, StarCoder2, Galactica, and Phi-3.
- [ ] I can explain why code-focused tokenizers handle whitespace and digits differently.
- [ ] I understand the embedding matrix: what it contains, how it's used, and how it's trained.
- [ ] I can distinguish static embeddings from contextualized embeddings with an example.
- [ ] I can run code to generate contextualized token embeddings with DeBERTa v3 and interpret the output shape.
- [ ] I can explain text (sentence) embeddings and use `sentence-transformers` to produce one.
- [ ] I can explain skip-gram and negative sampling in word2vec with a concrete example.
- [ ] I understand how noise-contrastive estimation motivates negative sampling.
- [ ] I can explain how the word2vec song recommendation system works and why it produces meaningful results.
- [ ] I can use Gensim to load pretrained GloVe embeddings and find nearest neighbors.
- [ ] I know the major applications of each embedding type: static, contextualized token, and text.
- [ ] I can identify and correct the most common misconceptions about tokens and embeddings.
- [ ] I can answer LLM-interview questions about tokenization and embeddings fluently.

---

*Study guide generated from Chapter 2 of Hands-On Large Language Models by Alammar & Grootendorst. Covers every concept, code example, figure description, note, and technical term in the chapter.*
