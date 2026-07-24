# Vector Databases and Embeddings: A Complete Reference

## From Neural Representations to Retrieval-Augmented Generation

---

# Part I — Foundations: Why Vector Databases Exist

## 1.1 The Knowledge Gap in Large Language Models

Large language models (LLMs) have unlocked a wide range of new applications, but they share a fundamental structural limitation: **a trained model's knowledge is frozen at the moment its training data was collected.** Two consequences follow directly from this:

1. **No knowledge of recent events.** Anything that happened after the training cutoff is simply absent from the model's parameters.
2. **No knowledge of proprietary or private documents.** Information that was never part of the public training corpus (internal company wikis, private codebases, personal notes, unpublished research, etc.) cannot be recalled by the model, no matter how the question is phrased.

Naively, one might try to solve this by periodically retraining or fine-tuning the model on new data. In practice this is expensive, slow, and impractical for keeping pace with constantly changing or highly private information. This motivates a different architectural solution.

## 1.2 Retrieval-Augmented Generation (RAG)

**Retrieval-Augmented Generation (RAG)** is a technique that lets an LLM answer questions using external, up-to-date, or proprietary information without retraining the model itself. The central idea: instead of relying purely on the model's internal (parametric) memory, the system **retrieves** relevant text from an external store and **inserts it into the prompt** before generation, giving the model the context it needs to answer accurately.

### 1.2.1 The RAG Pipeline

A RAG system works in three broad steps:

1. **Ingestion (offline / preprocessing step).** Proprietary or recent documents are stored in a **vector database** ahead of time.
2. **Retrieval (query time).** When a user submits a query that concerns information covered by those documents, the query is sent to the vector database, which retrieves the most relevant text passages ("related text data").
3. **Augmented generation.** The retrieved passages are inserted into the prompt sent to the LLM, giving the model grounded context with which to construct its answer.

> **Analogy.** Retrieval-augmented generation can be compared to visiting a library before answering a question. Without access to any reference material, a person asked a question outside their personal knowledge might simply guess or fabricate a plausible-sounding answer. If that same person can first walk into a library, find a relevant book, and read the pertinent passage, they can then produce an answer grounded in real information. RAG gives the LLM this "library visit" capability at query time.

### 1.2.2 Key Benefits of RAG

- **Reduces hallucination** — because the model's output is grounded in retrieved, real source text rather than purely generated from parametric memory.
- **Enables source citation** — since the specific retrieved passages are known, the system can point back to where an answer came from.
- **Solves knowledge-intensive tasks** — particularly useful for information that is rare or highly specific, the kind of long-tail knowledge a general-purpose LLM is unlikely to have memorized well.
- **No retraining or fine-tuning required** — new or private information can be made available to the LLM simply by adding it to the vector database, which is far cheaper and faster than retraining.

## 1.3 The Central Role of the Vector Database

The component that makes retrieval possible is the **vector database**: a data store specialized for holding and searching numerical vector representations ("embeddings") of data such as text or images.

It is important to note that vector databases are **not a new invention created for LLM applications** — they significantly predate the recent generative AI boom. Historically, they have been a core part of:

- **Semantic search applications** — search systems that match on the *meaning* of a query rather than on exact keyword overlap (contrast this with traditional keyword search, which only finds literal string matches).
- **Recommender systems** — systems that find items similar to a reference item (e.g., "customers who liked this also liked...") by comparing vector representations of items.

### 1.3.1 Why an AI Developer Should Understand the Internals

Understanding what happens *under the hood* of a vector database — rather than treating it as an opaque black box — has concrete practical payoffs for building better AI applications:

- **Choosing a search strategy.** Knowing the mechanics of sparse (keyword-based) search, dense (embedding-based) search, and hybrid search allows a developer to decide which is appropriate for a given use case, rather than defaulting blindly to one approach.
- **Choosing a distance/similarity metric.** Understanding how different similarity calculations work (Euclidean, Manhattan, dot product, cosine) helps in selecting the most appropriate distance algorithm for the data and model in use.
- **Understanding scaling trade-offs.** Understanding the computational challenge of scaling search across many vectors motivates the choice between exact search algorithms and approximate search algorithms, and clarifies *why* approximate methods are used in production systems.

---

# Part II — Embeddings: Representing Data as Vectors

## 2.1 What Is a Vector Embedding?

A **vector embedding** is a numerical representation of a piece of data (an image, a sentence, an audio clip, etc.) expressed as a dense array of real numbers. Its defining property is that it **captures the meaning** of the underlying data in a machine-understandable format — semantically similar data objects are mapped to vectors that are close together in the embedding space, while dissimilar data objects are mapped to vectors that are far apart.

**Key intuition:** you can think of a vector embedding as translating any kind of raw data — pixels, characters, sounds — into a common numerical "language" that captures underlying meaning, allowing quantitative comparison ("how similar are these two things?") through vector math rather than through manual rules.

Embeddings are the atomic building block on top of which everything else in this document rests: similarity search, approximate nearest neighbor algorithms, vector databases, hybrid search, and RAG are all, ultimately, operations performed on embeddings.

## 2.2 Generating Embeddings with an Autoencoder

### 2.2.1 Motivation and Architecture

One straightforward way to *illustrate* how neural networks can learn to compress data into meaningful vectors is the **autoencoder** architecture. An autoencoder consists of two connected sub-networks:

- **Encoder** — compresses the input into a smaller, dense vector representation.
- **Decoder** — reconstructs the original input from that compressed vector.

```
Input (784-dim) → Encoder → Embedding (2-dim, "bottleneck") → Decoder → Output (784-dim)
```

The network is trained so that the reconstructed output matches the original input as closely as possible. Critically, the reconstruction is generated using **only** the small bottleneck vector in the middle — meaning that vector must contain enough information to reconstruct the entire input. This bottleneck vector is precisely the **embedding**: it is a compressed numerical summary that captures the meaning (or at least the reconstructable structure) of the original data object.

### 2.2.2 Worked Example: MNIST Handwritten Digits

The MNIST dataset of handwritten digit images was used to make the autoencoder concept concrete.

- Each image is **28 × 28 pixels**, which flattens to **784 dimensions** when represented as a single vector.
- Passed through the encoder, the image is compressed step-by-step through dense layers of decreasing size: **784 → 256 → 128 → 2** dimensions.
- The decoder mirrors this in reverse: **2 → 128 → 256 → 784** dimensions, reconstructing an output image from the 2-dimensional embedding.

At the start of training, the reconstructed output does **not** closely match the input — the two images are visibly different. Training proceeds over multiple passes (**epochs**), and at each pass the network's internal weights are adjusted (via backpropagation, implicit in the described gradient-based training) to reduce this reconstruction error. As training progresses, the match between input and reconstructed output steadily improves until the model converges to an acceptable quality.

> **Note on dimensionality.** The embedding dimension was deliberately set to **2** purely for pedagogical convenience — a 2-dimensional embedding can be plotted directly on an X–Y scatter plot for visualization. In real-world use, embeddings are usually **far higher-dimensional**, frequently reaching **hundreds or over a thousand dimensions**, since more dimensions generally allow the model to capture more nuanced distinctions in meaning. The 2D case here is a simplification for teaching, not a realistic production configuration.

### 2.2.3 Visualizing the Embedding Space

After training, feeding the full dataset through only the encoder half of the network and plotting the resulting 2D embeddings produces a scatter plot in which:

- Images of the same digit **cluster together** (e.g., all embeddings of the digit "0" fall close to one another; all embeddings of "9" form their own separate cluster).
- Different digit clusters are spatially separated from each other.

This is direct visual confirmation of the core embedding property described in §2.1: **semantic similarity in the original data corresponds to spatial proximity in the embedding space.**

### 2.2.4 Code Walkthrough: Building and Training the Autoencoder

The implementation proceeds through the following stages (using TensorFlow):

1. **Load libraries.** TensorFlow is loaded to build and train the network.
2. **Load the dataset.** The MNIST dataset is loaded, producing a training set and a test set of images.
3. **Normalize and flatten the data.** Each 28×28 image is normalized (pixel values rescaled) and flattened into a 784-dimensional vector. Printing the shapes before and after this step confirms the transformation: 60,000 training objects, originally shaped 28×28, become 60,000 vectors of 784 dimensions each (and correspondingly for the test set).
4. **Set hyperparameters.**
   - **Batch size:** 100 objects per training batch.
   - **Epochs:** 50 full passes over the training data.
   - **Hidden layer size:** starts at 256 dimensions.
   - **Target embedding size:** 2 dimensions (chosen for visualization, as noted above).
5. **Inspect a sample input.** An example digit image is displayed (visually resembling a "0") to sanity-check that the data loaded correctly before training.
6. **Construct a sampling function**, used during training to draw batches of images.
7. **Build the encoder** — two dense layers, the first with 256 dimensions, the second with 128 dimensions, feeding down toward the 2-dimensional bottleneck. A normalization step is included as part of encoding.
8. **Build the decoder** — the mirror image of the encoder, starting from the 2-dimensional embedding and expanding through 128 and then 256 dimensions back up to the original 784-dimensional output space.
9. **Define the loss function** used to train the network — this model is identified in the source material as a **variational autoencoder (VAE)**, and the loss function is constructed so that training pushes the reconstructed outputs to match the original inputs as closely as possible.

   > *Terminology note (author's addition for clarity):* A standard (vanilla) autoencoder learns a deterministic mapping from input to a single bottleneck vector. A **variational autoencoder (VAE)** instead learns a *distribution* over the latent space, typically regularized to be close to a standard normal distribution, and samples from that distribution during encoding. This regularization tends to produce a smoother, more well-structured latent space, which is beneficial for the kind of clustering/visualization demonstrated in §2.2.3. The transcript identifies the model as a VAE but demonstrates its use exactly like a plain autoencoder — extracting the bottleneck vector as an embedding — so the practical embedding-extraction workflow described here applies to both variants.

10. **Train the model** for 50 epochs, with each epoch iterating across batches of 100 objects. This is computationally intensive enough that, in the original demonstration, the training step was time-lapsed/sped up for the video.
11. **Visualize the results.** After training, a "flat encoder" (i.e., the encoder half of the trained network, used standalone) is built. Passing the dataset through it and plotting the resulting 2D vectors produces the clustered visualization described in §2.2.3.

#### 2.2.4.1 Full Reference Implementation

The complete, runnable implementation corresponding to the eleven stages above is reproduced below. This is a **variational autoencoder (VAE)**: instead of mapping each input directly to a single point in the latent space, the encoder produces a mean (`mu`) and a log-variance (`log_var`) for a Gaussian distribution, and a sample is then drawn from that distribution using the *reparameterization trick* (the `sampling` function) — this is what lets gradients flow back through an otherwise-random sampling step during training.

```python
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from scipy.stats import norm

# Load data – training and test
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()

# Normalize and Reshape images (flatten)
x_tr, x_te = x_tr.astype('float32')/255., x_te.astype('float32')/255.
x_tr_flat, x_te_flat = x_tr.reshape(x_tr.shape[0], -1), x_te.reshape(x_te.shape[0], -1)

print(x_tr.shape, x_te.shape)
print(x_tr_flat.shape, x_te_flat.shape)

# Neural Network Parameters
batch_size, n_epoch = 100, 50
n_hidden, z_dim = 256, 2

# Example of a training image
plt.imshow(x_tr[1]);

# sampling function
def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_var) * eps

# Encoder - from 784->256->128->2
inputs_flat = Input(shape=(x_tr_flat.shape[1:]))
x_flat = Dense(n_hidden, activation='relu')(inputs_flat)      # first hidden layer
x_flat = Dense(n_hidden//2, activation='relu')(x_flat)        # second hidden layer

# hidden state, which we will pass into the Model to get the Encoder.
mu_flat = Dense(z_dim)(x_flat)
log_var_flat = Dense(z_dim)(x_flat)
z_flat = Lambda(sampling, output_shape=(z_dim,))([mu_flat, log_var_flat])

# Decoder - from 2->128->256->784
latent_inputs = Input(shape=(z_dim,))
z_decoder1 = Dense(n_hidden//2, activation='relu')
z_decoder2 = Dense(n_hidden, activation='relu')
y_decoder = Dense(x_tr_flat.shape[1], activation='sigmoid')
z_decoded = z_decoder1(latent_inputs)
z_decoded = z_decoder2(z_decoded)
y_decoded = y_decoder(z_decoded)
decoder_flat = Model(latent_inputs, y_decoded, name="decoder_conv")

outputs_flat = decoder_flat(z_flat)

# variational autoencoder (VAE) - to reconstruction input
reconstruction_loss = losses.binary_crossentropy(inputs_flat,
                                                 outputs_flat) * x_tr_flat.shape[1]
kl_loss = 0.5 * K.sum(K.square(mu_flat) + K.exp(log_var_flat) - log_var_flat - 1, axis=-1)
vae_flat_loss = reconstruction_loss + kl_loss

# Build model
# Ensure that the reconstructed outputs are as close to the inputs
vae_flat = Model(inputs_flat, outputs_flat)
vae_flat.add_loss(vae_flat_loss)
vae_flat.compile(optimizer='adam')

# train
vae_flat.fit(
    x_tr_flat,
    shuffle=True,
    epochs=n_epoch,
    batch_size=batch_size,
    validation_data=(x_te_flat, None),
    verbose=1
)
```

**Line-by-line notes:**

- `x_tr.astype('float32')/255.` — pixel values, originally integers in `[0, 255]`, are rescaled to floats in `[0, 1]`. Neural networks generally train more stably on small, normalized input ranges than on raw pixel intensities.
- `x_tr.reshape(x_tr.shape[0], -1)` — flattens each 28×28 image into a single 784-length vector (`-1` tells NumPy/TensorFlow to infer that dimension automatically from the total element count), since the dense (fully-connected) layers used here expect 1-D input per example, not a 2-D image grid.
- `mu_flat` / `log_var_flat` — two separate dense output heads from the shared encoder trunk, producing the mean and log-variance of the latent Gaussian distribution for each input, respectively. Predicting the *log* of the variance (rather than the variance directly) is a standard numerical-stability trick, since it avoids ever needing to constrain a raw output to be non-negative.
- `Lambda(sampling, ...)` — wraps the reparameterization-trick sampling function as a Keras layer so it can sit inside the computational graph; this is the actual bottleneck embedding-producing step.
- `reconstruction_loss` — binary cross-entropy between the original flattened input and the reconstructed output, scaled by the number of pixels (784), measuring how well the decoder reconstructs the original image from the sampled latent vector.
- `kl_loss` — the Kullback–Leibler divergence term, which regularizes the learned latent distribution to stay close to a standard normal distribution. This is the term that specifically distinguishes a *variational* autoencoder's loss from a plain autoencoder's (which would use reconstruction loss alone).
- `vae_flat.add_loss(...)` — attaches the combined custom loss (reconstruction + KL) to the model, since this loss depends on intermediate tensors (`mu_flat`, `log_var_flat`) rather than only on the final model output, so it cannot be expressed as a normal `loss=` argument to `compile()`.
- **Edge case / practical note:** the `sampling` function hard-codes `batch_size` into the shape of the random noise tensor `eps`. This means the final training batch of an epoch must divide evenly into `batch_size`, or the shapes will mismatch — a common source of subtle bugs in from-scratch VAE implementations if the dataset size isn't an exact multiple of the batch size.

**Visualizing the trained embedding space:**

```python
# Build encoders
encoder_f = Model(inputs_flat, z_flat)  # flat encoder

# Plot of the digit classes in the latent space
x_te_latent = encoder_f.predict(x_te_flat, batch_size=batch_size, verbose=0)
plt.figure(figsize=(8, 6))
plt.scatter(x_te_latent[:, 0], x_te_latent[:, 1], c=y_te, alpha=0.75)
plt.title('MNIST 2D Embeddings')
plt.colorbar()
plt.show()
```

This constructs a standalone **encoder-only** model (`encoder_f`) by reusing the already-trained encoder layers, discarding the decoder. Running the test set through it produces `x_te_latent`, an array of 2D embeddings, one row per test image. Plotting these as a scatter plot, colored by the true digit label `y_te`, produces the clustering visualization discussed in §2.2.3 — each digit class should form its own visually distinct cluster if training was successful.

### 2.2.5 Comparing Individual Embeddings Numerically

To make the notion of "similarity" concrete beyond just visual clustering, three specific images were selected and embedded:

- **Zero A** — an image of digit "0"
- **Zero B** — a second, different image of digit "0"
- **One** — an image of digit "1"

Generating embeddings for all three and printing the raw vector values shows, numerically, that **Zero A and Zero B have similar vector values to each other**, while **the embedding for "One" is noticeably different** from both. This sets up the motivation for the next topic: how do we *quantify* "similar" versus "different" precisely, using a formal distance or similarity metric, rather than just eyeballing the numbers?

**Reference code:**

```python
# Visualize the three chosen digit images before embedding them
plt.imshow(x_te_flat[10].reshape(28, 28));
plt.imshow(x_te_flat[13].reshape(28, 28));
plt.imshow(x_te_flat[2].reshape(28, 28));

# calculate vectors for each digit
zero_A = x_te_latent[10]
zero_B = x_te_latent[13]
one = x_te_latent[2]

print(f"Embedding for the first ZERO is  {zero_A}")
print(f"Embedding for the second ZERO is {zero_B}")
print(f"Embedding for the ONE is         {one}")
```

Here, `x_te_latent` is the array of 2D embeddings produced by the standalone encoder in §2.2.4.1; indices `10`, `13`, and `2` were simply chosen (by inspecting the corresponding images with `imshow`) as, respectively, two different images of digit "0" and one image of digit "1." Because `z_dim = 2`, each of `zero_A`, `zero_B`, and `one` is a plain 2-element NumPy array — small enough to print and compare by eye before formal distance metrics are introduced in Part III.

## 2.3 Text Embeddings

The same underlying principle — compressing raw data into a meaning-preserving vector — applies to text, not just images. Using a **sentence transformer** model, several example sentences were converted into embeddings.

- The resulting embedding shape was **384 dimensions** per sentence (a typical output size for compact sentence-transformer models).
- To visually compare sentence embeddings, each vector was rendered as a **"barcode"** — a horizontal strip in which each dimension's value is represented as a shade or bar. Visually, this showed that the first two example sentences produced barcodes that looked broadly similar to each other, while the third sentence's barcode looked visibly different — a qualitative preview of the quantitative distance calculations that follow in the next section.

**Reference code:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sentences we want to encode. Example:
sentence = ['The team enjoyed the hike through the meadow',
            'The national park had great views',
            'Olive oil drizzled over pizza tastes delicious']

# Sentences are encoded by calling model.encode()
embedding = model.encode(sentence)

# Preview the embeddings
print(embedding)
embedding.shape   # -> (3, 384)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(embedding[0].reshape(-1, 384), cmap="Greys", center=0, square=False)
plt.gcf().set_size_inches(10, 1)
plt.axis('off')
plt.show()

sns.heatmap(embedding[1].reshape(-1, 384), cmap="Greys", center=0, square=False)
plt.gcf().set_size_inches(10, 1)
plt.axis('off')
plt.show()

sns.heatmap(embedding[2].reshape(-1, 384), cmap="Greys", center=0, square=False)
plt.gcf().set_size_inches(10, 1)
plt.axis('off')
plt.show()
```

**Notes:**

- `SentenceTransformer('paraphrase-MiniLM-L6-v2')` loads a pretrained transformer-based sentence-embedding model specifically fine-tuned for paraphrase/semantic-similarity tasks — a compact ("MiniLM," 6-layer) model chosen for speed in a teaching context; larger sentence-transformer models exist that trade speed for higher embedding quality.
- `model.encode(sentence)` runs the full list of sentences through the model in a single batched call, returning a NumPy array of shape `(3, 384)` — one 384-dimensional row per input sentence.
- The three example sentences were deliberately chosen so that the first two ("hike through the meadow," "national park had great views") share an outdoor/nature theme, while the third (pizza and olive oil) is thematically unrelated — setting up the expectation, later confirmed numerically in §3.5, that the first two sentences should be measurably more similar to each other than either is to the third.
- `sns.heatmap(..., cmap="Greys", center=0, square=False)` renders each 384-dimensional embedding vector as a single-row heatmap ("barcode"): each of the 384 dimensions becomes one narrow vertical strip, shaded by its numeric value, with `center=0` ensuring the color scale is centered at zero so that positive and negative values are visually distinguishable.

### 2.3.1 Key Takeaway: Embeddings Are Modality-Agnostic

The same conceptual pipeline — raw data → neural network → dense vector capturing meaning — applies whether the raw data is:
- An image (as in the MNIST autoencoder example), or
- A piece of text (as in the sentence transformer example).

This generality is precisely why vector databases are useful across so many modalities and applications: **any data type that can be embedded into a vector space can be indexed, compared, and searched using the same underlying machinery.**

---

# Part III — Quantifying Similarity: Distance and Similarity Metrics

Once data has been embedded into vectors, the next question is: **how do we mathematically measure how "close" or "similar" two vectors are?** Four common metrics were covered: **Euclidean distance, Manhattan distance, dot product,** and **cosine distance**. Of these, **dot product and cosine distance are the most commonly used in natural language processing (NLP) applications.**

## 3.1 Euclidean Distance

**Definition / intuition.** Euclidean distance calculates the length of the **straight-line ("as the crow flies") shortest path** between two points in vector space. For two vectors $\mathbf{a}$ and $\mathbf{b}$ of dimension $n$:

$$d_{\text{Euclidean}}(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

**Interpretation.** Lower values indicate greater similarity (the points are close together); higher values indicate the points are far apart.

**Worked example.** Calculating the Euclidean distance between the "Zero A" and "Zero B" MNIST embeddings gave a result of approximately **0.6** — a small value, consistent with the two zeros being visually and semantically similar. NumPy provides built-in vector operations that can compute this distance directly without writing the summation manually. When distances were computed across all three example vectors (Zero A, Zero B, One), the two zeros had the smallest distance between them, while both zeros were comparatively far from the "One" embedding — a quantitative confirmation of the earlier qualitative/visual observation.

**Reference code:**

```python
# Euclidean Distance — manual computation
L2 = [(zero_A[i] - zero_B[i])**2 for i in range(len(zero_A))]
L2 = np.sqrt(np.array(L2).sum())
print(L2)

# An alternative way of doing this
np.linalg.norm((zero_A - zero_B), ord=2)

# Calculate L2 distances across all three pairs
print("Distance zeroA-zeroB:", np.linalg.norm((zero_A - zero_B), ord=2))
print("Distance zeroA-one:  ", np.linalg.norm((zero_A - one), ord=2))
print("Distance zeroB-one:  ", np.linalg.norm((zero_B - one), ord=2))
```

`np.linalg.norm(vector, ord=2)` computes the **L2 norm** (Euclidean length) of a vector; applying it to the *difference* vector `zero_A - zero_B` is mathematically equivalent to the manual sum-of-squares-then-square-root computed in the list comprehension above, but is shorter, less error-prone, and runs faster since it is implemented in optimized C code under the hood rather than a Python loop.

## 3.2 Manhattan Distance

**Definition / intuition.** Manhattan distance (also known as "taxicab distance" or "city block distance") measures distance as if movement were restricted to travelling along **one axis at a time** — like navigating a grid of city blocks where you cannot cut diagonally across a block.

$$d_{\text{Manhattan}}(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} |a_i - b_i|$$

**Interpretation.** As with Euclidean distance, lower values indicate greater similarity.

**Worked example.** Applying Manhattan distance to the same MNIST embeddings again showed that Zero A and Zero B were very close to each other, while both were comparatively far from the "One" embedding — the same qualitative conclusion as Euclidean distance, computed via a different formula. NumPy again provides a direct method for this computation, avoiding a manual loop.

**Reference code:**

```python
# Manhattan Distance — manual computation
L1 = [zero_A[i] - zero_B[i] for i in range(len(zero_A))]
L1 = np.abs(L1).sum()
print(L1)

# An alternative way of doing this
np.linalg.norm((zero_A - zero_B), ord=1)

# Calculate L1 distances across all three pairs
print("Distance zeroA-zeroB:", np.linalg.norm((zero_A - zero_B), ord=1))
print("Distance zeroA-one:  ", np.linalg.norm((zero_A - one), ord=1))
print("Distance zeroB-one:  ", np.linalg.norm((zero_B - one), ord=1))
```

Note the close parallel with the Euclidean distance code in §3.1: the only change needed to switch from L2 (Euclidean) to L1 (Manhattan) norm is the `ord` argument passed to `np.linalg.norm` — `ord=2` for Euclidean, `ord=1` for Manhattan. This is a convenient illustration of how both metrics are special cases of the general $L_p$ norm family.

## 3.3 Dot Product

**Definition / intuition.** The dot product measures **the magnitude of the projection of one vector onto another**. Formally:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

**Interpretation — important distinction from the previous two metrics.** Unlike Euclidean and Manhattan distance, where a *smaller* value means a *better* (closer) match, for dot product the relationship is **reversed**: **higher values indicate a better match**, while **negative values typically indicate the vectors are far apart / dissimilar**.

**Worked example.** The dot product between Zero A and Zero B was **3.6**, a relatively high (positive) value indicating strong similarity. In contrast, the dot products comparing the zero embeddings against the "One" embedding were **negative**, indicating dissimilarity — consistent with all prior metrics but expressed on an inverted scale.

**Reference code:**

```python
# Dot Product
np.dot(zero_A, zero_B)

# Calculate dot products across all three pairs
print("Distance zeroA-zeroB:", np.dot(zero_A, zero_B))
print("Distance zeroA-one:  ", np.dot(zero_A, one))
print("Distance zeroB-one:  ", np.dot(zero_B, one))
```

`np.dot(a, b)` computes $\sum_i a_i b_i$ directly — no intermediate difference vector is needed here (contrast with Euclidean/Manhattan distance in §3.1–3.2, which both operate on `a - b`), since the dot product is a function of the two original vectors' relative orientation and magnitude, not their pointwise difference.

> **Practical implication:** Because the interpretation direction of dot product is opposite to Euclidean/Manhattan distance, care must be taken when writing ranking or filtering code — sorting must be done in *descending* order for dot product (to put the best matches first) versus *ascending* order for Euclidean/Manhattan distance.

## 3.4 Cosine Distance (and Cosine Similarity)

**Definition / intuition.** Cosine distance is based on the **angle** between two vectors rather than their magnitude or straight-line separation: **similar vectors point in a similar direction, producing a very small angle between them.**

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}$$

Cosine *similarity* is this cosine-of-the-angle value; cosine *distance* is typically derived from it (e.g., as $1 - \cos(\theta)$) so that, like Euclidean/Manhattan distance, **lower values indicate better matches**.

**Worked example.** For the two zero embeddings, the computed cosine value was very close to **zero**, indicating a very small angle between the vectors and therefore a strong match. As an additional sanity check, dividing "Zero A" by "Zero B" element-wise produced a result with magnitude close to a constant — visually confirming that the two vectors point in very similar directions (i.e., one is close to being a scaled copy of the other). Encapsulating this into a reusable cosine-distance function and running it across all three example vectors again showed a small angle (high similarity) between the two zeros, and a comparatively high cosine distance value between the zeros and the "One" embedding — proving the same underlying similarity relationship as the other three metrics, but through the geometric lens of angle rather than straight-line or projection-based distance.

**Reference code:**

```python
# Cosine Distance — single manual computation (Zero A vs Zero B)
cosine = 1 - np.dot(zero_A, zero_B) / (np.linalg.norm(zero_A) * np.linalg.norm(zero_B))
print(f"{cosine:.6f}")

# Sanity check: element-wise ratio of the two vectors
zero_A / zero_B

# Reusable Cosine Distance function
def cosine_distance(vec1, vec2):
    cosine = 1 - (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return cosine

# Cosine Distance across all three pairs
print(f"Distance zeroA-zeroB: {cosine_distance(zero_A, zero_B): .6f}")
print(f"Distance zeroA-one:   {cosine_distance(zero_A, one): .6f}")
print(f"Distance zeroB-one:   {cosine_distance(zero_B, one): .6f}")
```

This implementation directly follows the formula in §3.4: `np.dot(vec1, vec2)` computes the numerator ($\mathbf{a} \cdot \mathbf{b}$), and `np.linalg.norm(vec1) * np.linalg.norm(vec2)` computes the denominator ($\|\mathbf{a}\| \, \|\mathbf{b}\|$, the product of the two vectors' lengths). Subtracting the resulting cosine-similarity value from 1 converts it into a cosine *distance*, so that — consistent with Euclidean and Manhattan distance — **smaller values indicate closer matches.**

## 3.5 Applying Distance Metrics to Sentence Embeddings

Returning to the earlier text-embedding example (§2.3), dot product and cosine distance — the two metrics most commonly used in NLP — were applied to the sentence embeddings:

- **Dot product** between the first two example sentences was **high**, indicating strong similarity, while dot products involving the remaining sentences were noticeably lower, indicating less similarity.
- **Cosine distance** told the same qualitative story: the angle between the first two sentences was comparatively small (high similarity), while the other sentence pairs showed larger angles (lower similarity). It was noted, however, that even the "most similar" pair of sentences was **not a perfect match** — the similarity was strong but not identical.

**Reference code** (reusing the `embedding` array and `cosine_distance` function defined above; recall from §2.3 that `embedding[0]`, `embedding[1]`, and `embedding[2]` correspond respectively to the hike, national-park, and pizza sentences):

```python
# Dot Product
print("Distance 0-1:", np.dot(embedding[0], embedding[1]))
print("Distance 0-2:", np.dot(embedding[0], embedding[2]))
print("Distance 1-2:", np.dot(embedding[1], embedding[2]))

# Cosine Distance
print("Distance 0-1: ", cosine_distance(embedding[0], embedding[1]))
print("Distance 0-2: ", cosine_distance(embedding[0], embedding[2]))
print("Distance 1-2: ", cosine_distance(embedding[1], embedding[2]))
```

Notice that the exact same `cosine_distance` helper function defined in §3.4 for the (2-dimensional) MNIST embeddings is reused here, unmodified, for the (384-dimensional) sentence embeddings — a direct illustration of the point made in §2.3.1: the same distance-metric machinery applies uniformly regardless of the embedding's dimensionality or source modality (image vs. text).

## 3.6 Summary Table: Distance and Similarity Metrics

| Metric | Formula (conceptual) | "Better match" direction | Typical use case / notes |
|---|---|---|---|
| Euclidean distance | Straight-line distance between points | Lower is better | General-purpose; sensitive to vector magnitude |
| Manhattan distance | Sum of absolute per-axis differences | Lower is better | Grid-like/axis-aligned distance notion |
| Dot product | Sum of elementwise products (projection magnitude) | **Higher is better**; negative ⇒ dissimilar | Very common in NLP; scale-sensitive (affected by vector magnitude) |
| Cosine distance | Based on the angle between vectors | Lower is better (smaller angle ⇒ closer) | Very common in NLP; **scale-invariant** — only direction matters, not magnitude |

> **Author's clarifying note (not stated explicitly in the source, but standard background knowledge that fills a natural gap here):** The reason dot product and cosine similarity are especially popular in NLP is that many embedding models are trained (or explicitly normalized) so that vector **direction** carries the semantic meaning, while magnitude may be less informative or even normalized away. Cosine similarity ignores magnitude entirely, so it is a natural fit when only direction matters. Dot product on **unit-normalized** vectors is mathematically equivalent to cosine similarity — which is part of why many production systems compute one or the other depending on whether the underlying vectors are pre-normalized.

---

# Part IV — Searching Across Vectors: From Brute Force to Approximate Methods

## 4.1 Semantic (Vector) Search — Definition

**Semantic search** (also called **vector search**) is search that retrieves results based on the *meaning* of the query rather than exact string/keyword overlap. Because vectors capture the meaning of the underlying data (Part II), semantic search can be performed by finding the data points whose vectors are **closest** to a query vector, using one of the distance metrics from Part III, and returning those closest points.

## 4.2 Brute-Force Search: The K-Nearest Neighbors (KNN) Algorithm

The simplest possible approach to semantic search is **brute-force search**, which in classical machine learning is known as the **K-Nearest Neighbors (KNN)** algorithm. It proceeds in three steps:

1. **Given a query vector, compute the distance between the query and every single vector in the dataset.**
2. **Sort all of the computed distances.**
3. **Return the top K objects with the best (smallest, or for dot product, largest) distance.**

### 4.2.1 The Computational Cost Problem

Brute-force KNN is conceptually simple and guarantees finding the *exact* nearest neighbors (it is an **exact** search algorithm), but it comes with a **significant computational cost**: the total query time grows directly with the number of objects stored in the database. If the size of the dataset doubles or triples, the time required to answer a single query also roughly doubles or triples. This is a **linear-time, $O(n)$**, algorithm with respect to the number of stored vectors, per query (before even accounting for the added cost of higher vector dimensionality).

### 4.2.2 Code Demonstration and Empirical Scaling Results

The brute-force algorithm was demonstrated and benchmarked using scikit-learn's `NearestNeighbors` utility (configured explicitly to use the `'brute'` algorithm).

**Basic demonstration (2 dimensions, small scale):**
1. 20 random 2-dimensional points were generated and plotted to visualize their spatial distribution.
2. These points were indexed using `NearestNeighbors` with the brute-force algorithm, producing a queryable index object.
3. A query was run asking for the **4 nearest neighbors** to a given query vector. The result returned object indices **10, 4, 19, and 15** (in the illustrative run), each with an associated distance.
4. Timing this query (by capturing timestamps before and after execution) showed that, for only 20 objects, the query completed almost instantly — as expected for such a small dataset.

**Scaling test — a `speedtest` helper function:** A reusable benchmarking function was built to measure brute-force query time as the number of objects (`count`) grows. It works in three steps:
1. Randomly generate `count` objects.
2. Build a `NearestNeighbors` brute-force index over them.
3. Measure and return the time taken to execute a single query against that index.

**Empirical results — scaling by number of objects (2 dimensions):**

| Number of objects | Approximate observed query behavior |
|---|---|
| 20,000 | Fast |
| 200,000 | Noticeably slower, but still fast |
| 2,000,000 | Slower still |
| 20,000,000 | ~10× increase over the 2 million case |
| 200,000,000 | ~12 seconds per query |

This clearly demonstrates that a naive linear-time algorithm becomes impractical well before reaching billion-scale datasets, which are common in real-world production systems.

**Scaling test — the effect of higher dimensionality:** The above tests were run using only 2-dimensional vectors, which understates the real-world cost, since production embeddings are typically much higher-dimensional (e.g., 384 or 768 dimensions, as seen earlier with sentence-transformer embeddings). To test this:

1. 1,000 documents of **768 dimensions** each were randomly generated and normalized, along with a query vector.
2. A query was timed manually: a timer was started, dot product distances were computed between the query and **all** stored vectors, the results were sorted, and the top 5 nearest matches were returned. This took about **half a millisecond** for 1,000 vectors of 768 dimensions.
3. The same procedure was scaled up across **1,000; 10,000; 100,000;** and **500,000** objects (all at 768 dimensions):

| Number of 768-dim objects | Approximate observed query time |
|---|---|
| 1,000 | ~0.5 ms |
| 10,000 | Fast |
| 100,000 | Still returns fairly fast |
| 500,000 | ~2 seconds per query |

At 500,000 objects, a single query already took **nearly 2 seconds**. Extrapolating, running **1,000 queries** against a 500,000-object, 768-dimensional dataset would take on the order of **half an hour in total** — clearly impractical for any interactive application (e.g., a chatbot or search UI where users expect sub-second responses).

### 4.2.3 Conclusion: Why Brute Force Doesn't Scale

Two separate scaling dimensions compound the cost of brute-force search:

1. **More stored vectors** → linearly more distance computations per query.
2. **Higher-dimensional vectors** (more realistic for production embedding models, e.g., 768+ dimensions) → each individual distance computation itself becomes more expensive.

In realistic production scenarios — tens or hundreds of millions of objects, with embeddings of several hundred dimensions — brute-force KNN search is **not viable** for latency-sensitive applications. This motivates the need for **approximate nearest neighbor (ANN)** algorithms, which trade a small amount of accuracy for a large improvement in query speed.

## 4.3 Approximate Nearest Neighbors (ANN)

**Approximate Nearest Neighbor (ANN)** algorithms sacrifice a small amount of accuracy (or **recall** — the fraction of true nearest neighbors actually returned) in exchange for a large improvement in query performance. Rather than exhaustively comparing the query to every single stored vector, ANN algorithms use pre-built index structures that let the search skip large portions of the dataset while still finding results that are *usually* the true nearest neighbors, or very close to them.

The algorithm examined in depth is **Hierarchical Navigable Small World (HNSW)**, which underlies many of the most widely used production vector databases.

### 4.3.1 Conceptual Foundation: The Small World Phenomenon

HNSW is inspired by the **small-world phenomenon** observed in human social networks — the informal notion of **"six degrees of separation,"** the idea that any two people in the world are connected, on average, through a chain of roughly six mutual-acquaintance links. Intuitively: you likely know someone who is very well-connected, who in turn knows someone else who is very well-connected, and by hopping through a small number of such well-connected intermediaries, you can reach almost anyone.

This same principle can be applied to vectors: if vectors are connected to their nearby neighbors in a graph structure (with a few "well-connected hub" vectors providing long-range shortcuts), then a search can efficiently "hop" through this graph toward the region of the query vector, without needing to compare against every single node.

### 4.3.2 Navigable Small World (NSW) — The Building Block

**Navigable Small World (NSW)** is the (single-layer) graph-construction algorithm that HNSW builds upon.

**Construction process.** Nodes (vectors) are added to the graph one at a time, and each new node is connected to its nearest existing neighbors already in the graph (in the illustrative example, each node was connected to its **2** nearest neighbors, though real systems typically use more connections — e.g., **8, 32, or more**, depending on dataset size).

**Worked construction example (8 vectors, 2 connections each):**

| Step | Node added | Connections formed |
|---|---|---|
| 1 | Vector 0 | None (first node) |
| 2 | Vector 1 | → 0 |
| 3 | Vector 2 | → 1, 0 |
| 4 | Vector 3 | → 2, 0 |
| 5 | Vector 4 | → 2, 0 |
| 6 | Vector 5 | → 2, 0 |
| 7 | Vector 6 | → 2, 4 |
| 8 | Vector 7 | → 5, 3 |

This produces a connected graph in which nearby vectors are directly linked, forming the "navigable small world."

**Querying an NSW graph.** To search for the nearest neighbor to a query vector:
1. Start at a **random entry node**.
2. Examine the neighbors of the current node; move to whichever connected neighbor is closer to the query than the current node.
3. Repeat until no connected neighbor is closer than the current node — at that point, the search **concludes**, and the current node is returned as the (approximate) nearest neighbor.

**Worked search example.** Starting from node 7 (connected to nodes 3 and 5): node 5 is closer to the query than node 7, so the search moves to node 5. From node 5 (connected to 0 and 2), node 2 is much closer, so the search moves to node 2. From node 2, several candidates are available, with node 6 being the best; the search moves to node 6. From node 6, no connected neighbor is closer to the query than node 6 itself, so the search terminates at node 6 — and in this particular example, node 6 does indeed turn out to be the true nearest neighbor.

**A key limitation: NSW does not always find the true best match.** This same graph, queried from a *different* starting point, illustrates the core trade-off of approximate search. Starting from node 0, the best available candidate from that position is node 1; from node 1, no better candidate is available, so the search halts at node 1. In this case, node 1 is **not** the true globally-closest vector to the query — the search found only an **approximately** nearest neighbor. This is still a reasonably good result, but it is not guaranteed to be optimal. This illustrates precisely the accuracy/recall trade-off inherent to all ANN methods: depending on the random starting point and the greedy, local nature of the graph traversal, the search can occasionally converge on a local optimum rather than the true global nearest neighbor.

### 4.3.3 Hierarchical Navigable Small World (HNSW)

HNSW extends NSW by stacking **multiple layers** of navigable small world graphs on top of one another, forming a hierarchy that lets search move efficiently from coarse, long-range jumps down to fine-grained, local refinement.

> **Analogy — traveling to a destination.** Reaching a specific address somewhere in the world typically proceeds hierarchically: first take a **plane** to the nearest major airport (a coarse, long-distance jump); then take a **train** to the specific town (a medium-range refinement); and finally **walk or take a taxi** for the last, short distance to the exact destination (fine-grained, local movement). HNSW's layered structure mirrors this: higher layers allow big jumps across the vector space, while the bottom layer allows precise local refinement.

**Layer construction.** The construction process for each individual layer is the same NSW construction process described in §4.3.2 — the novelty in HNSW is not a new construction rule, but the way **nodes are probabilistically assigned to multiple layers.**

**Layer assignment.** Each node, when inserted, is randomly assigned a maximum layer number:
- If the random draw yields **0**, the node exists **only** on the bottom layer (layer 0).
- If the random draw yields **2**, the node exists on layers **0, 1, and 2** (i.e., a node present at a given layer is also present at every layer below it).

The probability of a node being assigned to progressively higher layers decreases **logarithmically**. The practical consequence: **far fewer nodes exist at the higher layers than at the lower layers**, with the bottom layer (layer 0) containing *all* nodes, and each successive layer above it containing a sparser subset. This produces exactly the "airport → train station → walking distance" hierarchy from the analogy above: the top layer has only a few widely-spaced "hub" nodes suitable for large jumps, while the bottom layer has full local resolution.

**Querying HNSW.** Search proceeds top-down through the layers:
1. Start at a **random entry node** available at the **highest** layer.
2. Within that layer, greedily move toward the nearest neighboring node (same local-search rule as NSW).
3. Once no further improvement is available at the current layer, **drop down** to the next layer below, carrying over the current best node as the new entry point for that layer.
4. Repeat this process, refining the current-best node at each successively lower layer.
5. Once the **bottom layer** (layer 0, containing all nodes) is reached, the final local search there produces the returned nearest-neighbor result — this is described as covering the **"last mile"** of the search.

### 4.3.4 Why HNSW Scales So Well: Logarithmic Query Time

Two structural properties give HNSW its scalability advantage:

1. **Sparse upper layers.** Because the probability of landing in a higher layer decreases logarithmically, there are dramatically fewer nodes to compare against at the top layers, where the search spends most of its "coarse navigation" effort.
2. **Logarithmic growth in comparisons.** As the total number of data points increases, the number of comparisons needed to perform a vector search increases only **logarithmically**, not linearly. In computer science notation, this is described as **$O(\log n)$** runtime complexity.

**Practical implication.** As the dataset grows — for example, going from half a million to a full million vectors — the increase in query time is **minimal**, in stark contrast to brute-force KNN's linear ($O(n)$) growth (§4.2.1–4.2.3). This logarithmic scaling is precisely *why* HNSW (and ANN algorithms generally) can support real-time search over datasets containing hundreds of millions of vectors, where brute-force search would be computationally infeasible.

### 4.3.5 Code Demonstration: Building and Querying HNSW

A hands-on demonstration built an HNSW index from scratch to make the abstract layer-hopping process concrete:

1. **Setup.** 40 random 2-dimensional vectors were generated, with the number of nearest-neighbor connections per node set to **2** (mirroring the earlier NSW example).
2. **Query vector.** A query point was placed at coordinates **(0.5, 0.5)**. A node list including this query point was created, and the `NetworkX` graph library was used purely for **visualization purposes** (drawing the graph and its connections).
3. **Ground-truth comparison.** A brute-force search was first run to establish the actual best-matching vector to the query, for later comparison against the HNSW result — plotted so the query point and its true best match could be seen directly on the graph.
4. **Constructing the HNSW layers.** The multi-layer structure was built and then iterated layer-by-layer to print/visualize the nodes and connections present at each layer:
   - **Top layer:** a small handful of nodes (illustratively: nodes 20, 34, 28, and 39) with connections among themselves.
   - **Layer 2:** more nodes present, with more connections.
   - **Layer 1:** most nodes already present and reconnected.
   - **Layer 0 (bottom):** all nodes present, each connected to its nearest neighbors.

   This directly demonstrates the "sparse at the top, dense at the bottom" property described in §4.3.3.

5. **Running an HNSW query.** The query returns:
   - A **search path graph array**, capturing the full traversal path across all layers (used for visualization of how the search moved).
   - An **entry graph array**, identifying the starting entry point of the search.

   The traversal, layer by layer, in the illustrative run:
   - **Top layer:** start at node 39 → move to node 20 (closer to the query); no further improving neighbor at this layer → drop down.
   - **Layer 2:** from node 20 → move to node 16; no further improving neighbor → drop down.
   - **Layer 1:** from node 16 → move to node 2; no further improving neighbor → drop down.
   - **Layer 0 (bottom):** from node 2 → move to node 25, which is the true best match to the query.

   The HNSW search thus successfully located the correct nearest neighbor (matching the brute-force ground truth established in step 3) while only ever examining a small subset of the total nodes at each layer — illustrating the core efficiency benefit of the hierarchical structure in practice.

### 4.3.6 Summary: Brute-Force KNN vs. HNSW/ANN

| Property | Brute-Force KNN | HNSW (ANN) |
|---|---|---|
| Accuracy | Exact — guaranteed true nearest neighbors | Approximate — usually correct, occasionally suboptimal |
| Query time complexity | $O(n)$ — linear in dataset size | $O(\log n)$ — logarithmic in dataset size |
| Scalability to large datasets | Poor — becomes impractical at hundreds of millions of objects | Good — scales to very large datasets with minimal query-time growth |
| Underlying structure | None — exhaustive comparison against every stored vector | Multi-layer graph of nearest-neighbor connections |
| Typical use case | Small datasets, or scenarios that demand perfect accuracy | Production-scale vector databases and semantic search systems |

---

# Part V — Vector Databases in Practice: Weaviate

Having covered the theoretical and algorithmic foundations (embeddings, distance metrics, brute-force and approximate search), the material turns to hands-on use of a production vector database: **Weaviate**, an open-source vector database.

## 5.1 The Embedded Weaviate Mode

Weaviate offers an **embedded mode**, which allows the vector database to run **directly inside a notebook environment**, without needing to separately stand up and manage a server — convenient for learning, prototyping, and local experimentation.

## 5.2 Working with Raw Vectors (Manual Vectorization)

### 5.2.1 Creating a Collection (Schema)

The first step in using Weaviate is to define a **collection** (referred to in the material as a "data schema" or "data collection") — the equivalent of a table in a traditional database, describing what kind of objects will be stored.

Key configuration choices when creating a collection:
- **Name** — e.g., `myCollection`.
- **Vectorizer** — set to **`null`** when vectors will be supplied manually by the developer (as opposed to having Weaviate generate them automatically; contrast with §5.3).
- **Distance metric** — set to **cosine** distance in the demonstrated example (recall from Part III that cosine distance/similarity is one of the two metrics most favored in NLP applications).

Running the collection-creation call produces a new, empty collection ready to receive data.

> **Practical tip — idempotent re-creation.** For repeatable experimentation (e.g., re-running a notebook from scratch), it is useful to have a helper snippet that first **deletes** the collection if it already exists, and then **recreates** it fresh. This avoids errors from trying to create a collection that already exists, though it is not something that needs to be re-run on every single execution — only when a genuinely fresh start is desired.

### 5.2.2 Importing Data (Manual Vectors)

Five example data objects were used, each consisting of a `title`, a `foo` value, and a manually-specified vector embedding (`itemVector`).

**Best practice — batch loading.** Even though the demonstrated example used only five objects (where batching is not strictly necessary), the batch-loading pattern shown is described as **best practice** for realistic workloads. When importing **tens of thousands or millions** of objects, using a proper batch-loading process is important for performance and reliability — batching amortizes overhead across many objects instead of issuing one write per object.

**Import procedure:**
1. Loop through each item in the source dataset.
2. Construct a `properties` object holding the object's non-vector fields.
3. Call the batch **`add_data_object`** method, passing:
   - The target **collection name** (`myCollection`).
   - The **`properties`** object.
   - The **vector**, taken from the `itemVector` field.

### 5.2.3 Verifying the Import

Running an **aggregate/count query** against the collection confirms the number of objects stored — in the example, this correctly returned **5** objects after the import.

### 5.2.4 Querying with a Raw Vector

A query can be issued by supplying:
- The target collection (`myCollection`).
- The properties to return (e.g., `title`).
- A **raw query vector** (matching the dimensionality of the stored vectors — 6 dimensions in the illustrative example).
- A **limit** on the number of results to return (e.g., the top 2 best matches).

Running this returned the objects determined to be closest to the query vector (illustratively, the "second" and "fourth" objects in the dataset).

**Retrieving additional metadata.** By adding extra fields to the query, it is possible to also retrieve, for each returned result:
- The computed **distance** to the query vector.
- The object's own **vector**.
- The object's **ID**.

In the illustrative run, the top result had a computed distance of **0.65** to the query vector.

### 5.2.5 Filtered Vector Search

Because the data is stored in a full-featured database (not just a raw vector index), it is possible to combine vector similarity search with **traditional property-based filtering**. In the example, a filter was added requiring the `foo` property to be **greater than 44**, so that the vector search only considered (pre-filtered) objects meeting that condition. The results correctly included only objects satisfying both the vector similarity criterion and the `foo > 44` filter.

### 5.2.6 "More Like This" Search (Search by Object ID)

Rather than supplying a raw vector, it is also possible to search for objects similar to an **existing object**, referenced by its ID. In the example, the first result from a previous query was used as the reference object, and a search was run for the **3 most similar objects** to it. The results correctly included the reference object itself, along with two other similar objects (the "fourth" and "first" objects in the running example) — demonstrating that "find similar items to this one" style search (as used in recommender systems, per §1.3) works directly on top of the same underlying vector index.

## 5.3 Automatic Vectorization: Text2Vec Modules and a Real Dataset

The second, more realistic hands-on project used a **Jeopardy questions dataset**, where each record has a `category`, `question`, and `answer`. Rather than manually computing embeddings ahead of time, this project demonstrates Weaviate's ability to **automatically generate embeddings** using an external embedding model provider.

### 5.3.1 Weaviate's Modular Vectorization System

Weaviate offers a **modular system** of pluggable vectorizers, including (among the options mentioned):
- **Generative search with OpenAI**
- **Text vectorization with Cohere**
- **Text vectorization with Hugging Face**
- **Text vectorization with OpenAI**

The major practical benefit of this modular system is that it **eliminates the need for manual vectorization** — the developer does not need to separately call an embedding API and manage the resulting vectors; the database handles it internally, both at import time and at query time.

**Setup requirement.** To use an OpenAI-backed vectorizer, an **OpenAI API key** must be supplied to the embedded Weaviate instance. (When running such a project independently, a developer would need to substitute in their own personal API key.) It was also noted that certain informational warning messages may appear during setup that are harmless and do not indicate a problem — "everything works just fine."

### 5.3.2 Creating a Collection with Automatic Vectorization

A new collection named `question` was created, this time configured with the **`Text2Vec OpenAI`** vectorizer module (rather than `null`, contrast with §5.2.1). This module automatically:
- Generates a vector embedding for each object **as it is imported**.
- Generates a vector embedding for the **query text itself** at query time, so that queries can be issued as plain text rather than as raw vectors.

### 5.3.3 Importing Data with Automatic Vectorization

The import procedure mirrors the manual case (§5.2.2) — batching in groups (illustratively, groups of five), constructing a `properties` object per record (containing `question`, `answer`, and `category`), and calling the batch-add method against the `question` collection.

**Key difference from the manual-vector workflow:** no vector is passed in manually. The `Text2Vec OpenAI` module automatically computes and attaches an embedding for every object during import.

After import, an aggregate/count query confirmed **10 objects** were successfully stored (matching the size of the sample dataset).

### 5.3.4 Inspecting a Generated Embedding

Retrieving a single stored object and requesting its vector shows the auto-generated embedding — a long numeric vector, reported as approximately **1536 dimensions** (the material states "about 1500-dimensional," consistent with typical OpenAI text-embedding model output sizes at the time of the recording).

### 5.3.5 Semantic Search with `nearText`

Rather than supplying a raw vector, text queries can be issued directly using Weaviate's **`nearText`** operator, which internally uses the collection's configured vectorizer to embed the query text on the fly. In the example, a query with the concept **"biology"** returned two matching questions, with distances of **0.19** and **0.2** respectively.

**Interpreting the distance values.** Because the collection uses **cosine distance**, smaller distance values indicate stronger matches — so 0.19 and 0.2 represent fairly strong matches to the "biology" query concept.

### 5.3.6 Sorting and Distance Distributions

Running a query that retrieves **all** stored objects and inspecting their distances shows the full ranked ordering: as one scans down the returned list, distances **increase** (since cosine distance is used and lower values mean better matches) — meaning the *best* matches appear at the **top** of the results, and the *worst* matches appear at the **bottom**.

### 5.3.7 Distance-Threshold Filtering

Rather than requesting a fixed count of top-K results (which requires guessing an appropriate K), it is possible to instead specify a **maximum acceptable distance threshold** and let the number of returned results vary naturally based on that quality bar. In the example, a threshold of **0.24** was used — meaning the system returns *any* object whose distance to the query is at or below that value, and rejects anything worse. In the illustrative run, results beyond a distance of about **0.23** were excluded.

**Practical implication.** This threshold-based approach is well-suited to situations where the application has an explicit **quality requirement** for results (e.g., "only show me matches I'm fairly confident are relevant") rather than a fixed desired result count.

## 5.4 CRUD Operations

Because a vector database is a full-featured database (not merely a similarity index), it supports the standard **CRUD** operations: **Create, Read, Update, Delete.**

### 5.4.1 Create

A single object is created via a `data_object.create` call, supplying the object's data and the target collection name. As with batch import, the `Text2Vec OpenAI` module automatically generates the vector embedding for the newly created object. The call returns the new object's **UUID**, which can be used to reference it in subsequent operations.

### 5.4.2 Read

An object is retrieved by its ID. By default, this returns the object's stored properties. Adding a `with_vector=True`-style flag (referred to generically as "with vector true") additionally returns the object's full vector embedding alongside its properties.

### 5.4.3 Update

An existing object (referenced by ID) can be updated — in the example, an object's `answer` field was changed from **"Italy"** to **"Florence, Italy."** Re-reading the object by its ID after the update confirms the change was applied.

### 5.4.4 Delete

An object is deleted by its ID. Before deletion, the object count was checked (**11** objects, reflecting the 10 originally imported plus the 1 created in the Create step above); after deleting the created object, re-checking the aggregate count confirmed the collection returned to **10** objects.

---

# Part VI — Sparse Search, Dense Search, and Hybrid Search

## 6.1 Dense Search (Vector/Semantic Search) — Strengths and Limitations

**Dense search** is the vector-embedding-based semantic search covered throughout Parts II–V: it relies on the *meaning* of the data to find matches. For example, a query for "baby dogs" can correctly surface content about "puppies" even without any literal keyword overlap, because the embeddings capture semantic similarity.

**Limitations of dense search:**

1. **Domain mismatch degrades accuracy.** If the embedding model was trained on data from a different domain than the one it's being applied to, query accuracy suffers substantially.

   > **Analogy.** This is compared to asking a medical doctor how to fix a car engine — the doctor may be highly capable within their own domain, but has no particular expertise in car mechanics, so the "answer" they provide would likely be poor. Similarly, an embedding model trained on one kind of data will not necessarily produce meaningful, well-separated embeddings for data from an unrelated domain.

2. **Poor performance on data with little inherent "meaning," such as serial numbers or random codes.** A semantically random string like `BB43300` doesn't carry conceptual meaning the way natural language does, so a semantic search engine has no strong basis to distinguish or match it meaningfully against similar codes.

## 6.2 Sparse Search (Keyword Search)

For cases where dense/semantic search struggles — particularly exact matching on codes, identifiers, or specific rare terms — **sparse search**, also known as **keyword search**, is a better fit.

### 6.2.1 Bag of Words

One foundational technique for sparse search is the **bag-of-words** representation. For every passage of text in the dataset, a running vocabulary table is built by recording every distinct word encountered, and counting how often each word appears in each passage.

**Worked example.** In an example sentence, the words "extremely" and "cute" each appeared **once**, while the word "eat" appeared **twice** — these counts become the entries of that passage's bag-of-words vector.

### 6.2.2 Why It's Called "Sparse"

Across an entire dataset, the total vocabulary (the set of all distinct words appearing anywhere) can be very large. Any single passage of text, however, will only use a small fraction of that vocabulary — illustratively, perhaps only about **1%** of all available words. As a result, the vector representing any one passage is mostly filled with **zeros** (for all the vocabulary words that don't appear in that passage), with only a small number of non-zero entries for words that actually occur. This is precisely why the representation is called **sparse** — as opposed to **dense** embeddings, where essentially every dimension carries some (typically non-zero) value contributing to the overall meaning.

### 6.2.3 BM25 (Best Matching 25)

A widely used and strong-performing keyword-based ranking algorithm is **BM25 (Best Matching 25)**. It is particularly effective when searching across many keywords.

**How it works, conceptually:** BM25 counts how often each word in the query phrase appears within the target text, and then weights matches by word rarity:
- **Common words** (that appear very frequently across the corpus) are weighted as **less important** when a match occurs on them.
- **Rare words** are weighted much more heavily — a match on a rare, distinctive term contributes a much higher score than a match on a common term.

Like the bag-of-words representation, a BM25-based sparse vector for a given piece of text is mostly filled with zeros, since only a small subset of the total vocabulary is present in any given passage — hence the "sparse vector search" terminology applies here as well.

## 6.3 Hybrid Search: Combining the Best of Both

Sparse and dense search are not mutually exclusive choices — **hybrid search** runs **both** a sparse (keyword) query and a dense (vector) query for the same request, obtains a separate score from each, and then **combines** those scores into a single unified score used to re-rank and return the final result list.

### 6.3.1 The `alpha` Parameter

Weaviate's hybrid search exposes a single tunable parameter, **`alpha`**, that controls the relative weighting between the two search modes:

- **`alpha` close to 1** → favors the **dense (vector) search** scores more heavily.
- **`alpha` close to 0** → favors the **sparse (keyword) search** scores more heavily.
- **`alpha = 1`** → equivalent to pure dense vector search (no keyword contribution).
- **`alpha = 0`** → equivalent to pure sparse/keyword search (no vector contribution).

### 6.3.2 Worked Example: Querying for "animal"

Using the same Jeopardy dataset as in Part V, three variants of the same query text ("animal") were run:

1. **Pure `nearText` (dense) search for "animal":** returned semantically related results, including things like mammals and crocodile-related content, as well as an exact textual match on the word "animal" itself.
2. **Pure BM25 (sparse/keyword) search for "animal":** returned only **one** result — the object that contained an exact keyword match on "animal." No semantically-related-but-lexically-different objects were returned, since sparse search only matches on literal term overlap.
3. **Hybrid search for "animal" (varying `alpha`):**
   - With a **default/mid-range `alpha`**, the results looked broadly similar to the pure dense search, **but** the object containing the literal keyword "animal" was now pushed to the **top rank** — combining the recall benefits of semantic matching with the precision benefit of rewarding the exact keyword hit.
   - Setting **`alpha = 0`** reproduced the pure keyword-search behavior — none of the purely-semantic-only matches were returned, only the literal keyword match.
   - Setting **`alpha = 1`** reproduced the pure dense-vector-search behavior.

### 6.3.3 Summary Table: Sparse vs. Dense vs. Hybrid Search

| Property | Sparse (Keyword) Search | Dense (Vector) Search | Hybrid Search |
|---|---|---|---|
| Basis of matching | Literal term/keyword overlap | Semantic meaning (embeddings) | Combination of both, weighted |
| Strong for | Exact codes, serial numbers, rare precise terms, out-of-domain data | Conceptual/semantic similarity, synonyms, paraphrase matching | General-purpose; best of both |
| Weak for | No understanding of meaning/synonyms | Domain mismatch; meaningless/random strings | N/A — tunable to compensate for either weakness |
| Representative algorithm | BM25, bag-of-words | Embedding models + distance metrics (Part III) | Weaviate `alpha`-weighted fusion of both scores |
| Weaviate control | `bm25` query mode | `nearText` / `nearVector` query mode | `hybrid` query mode with `alpha` parameter |

---

# Part VII — Multilingual Search and Retrieval-Augmented Generation (RAG)

## 7.1 Multilingual Search

### 7.1.1 Concept

**Multilingual search** extends the core idea of semantic search (comparing "dog" to "puppy" and still finding a match, per Part IV) to work **across languages**. If an embedding model was trained on multilingual data, then the **same underlying concept expressed in different languages** will map to very similar — or even nearly identical — vector embeddings. Because the search operates purely on these vector representations, the exact same search methods used for single-language semantic search work automatically across content and queries written in **any language** the underlying model supports, with no special-cased logic required.

### 7.1.2 Worked Demonstration

This section used a **cloud-hosted (non-embedded) instance of Weaviate**, backed by **Cohere's multilingual embedding models** for vectorization. Two separate API keys were required:
- A **Cohere API key**, used for multilingual vector search.
- An **OpenAI API key**, used for generative search (i.e., the RAG generation step, covered in §7.2).

The demonstration dataset was large-scale: approximately **4.3 million Wikipedia articles**.

**Query 1 — English query, mixed-language results.** A query for "vacation spots in California" returned five results: most in English, but also **one in French** and **two in Spanish** — despite the query itself being written in English. Notably, this query executed near-instantly ("in a blink of an eye") even though it searched across the full 4.3-million-object dataset — demonstrating the practical scalability benefit of approximate nearest-neighbor search (Part IV) at real production scale.

**Restricting the output language.** A language filter was added so that returned results were constrained to **English only**, with the result count also reduced to three objects. Even with this filter applied, the underlying **query mechanism itself remained fully multilingual** — the filter only constrains which language the *results* must be in, not which languages can be *searched over* or *queried in*.

**Query 2 — non-English queries.** The same underlying question ("vacation spots in California") was issued:
- **In Polish** — and returned essentially the same relevant results as the English-language query.
- **In Arabic** (a language using an entirely different alphabet/script from English or Polish) — and this, too, correctly returned relevant results about vacation spots in California.

This progression — English → Polish (same alphabet as English) → Arabic (a fundamentally different alphabet) — was specifically chosen to demonstrate that the multilingual matching is **not** simply relying on superficial character/alphabet overlap, but is genuinely operating on cross-lingual semantic meaning captured by the embedding model.

### 7.1.3 Reference Code: Client Setup and Multilingual Queries

**Helper function and client connection:**

```python
def json_print(data):
    print(json.dumps(data, indent=2))

import weaviate, os, json
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

auth_config = weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))

client = weaviate.Client(
    url=os.getenv("WEAVIATE_API_URL"),
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.getenv("COHERE_API_KEY"),
        "X-Cohere-BaseURL": os.getenv("CO_API_URL")
    }
)

client.is_ready()  # check if True
```

**Notes:**
- `json_print` is a small convenience wrapper around `json.dumps(..., indent=2)`, used throughout this section to pretty-print Weaviate's (often deeply nested) JSON query responses in a human-readable format, rather than the compact single-line default.
- `load_dotenv(find_dotenv())` loads API keys and connection settings (`WEAVIATE_API_KEY`, `WEAVIATE_API_URL`, `COHERE_API_KEY`, `CO_API_URL`) from a local `.env` file rather than hardcoding secrets directly in the notebook — a standard credential-management best practice.
- `weaviate.auth.AuthApiKey(...)` constructs the authentication object needed to connect to a **cloud-hosted** Weaviate instance (contrast with the embedded-mode examples in Part V, which required no external authentication since everything ran locally).
- The `additional_headers` dictionary passes the Cohere API key and base URL through to Weaviate so that Weaviate itself can call out to Cohere's multilingual embedding model on the developer's behalf, both at import time (for stored documents) and at query time (for `nearText` queries).
- `client.is_ready()` is a simple connectivity/health check that returns `True` if the client can successfully reach and authenticate against the Weaviate instance — a useful first call to make when debugging connection issues.

**Checking the dataset size:**

```python
print(json.dumps(client.query.aggregate("Wikipedia").with_meta_count().do(), indent=2))
```

This issues an aggregate query against the `Wikipedia` collection requesting only the object count metadata (`with_meta_count()`), confirming the approximately 4.3 million stored articles referenced in §7.1.2 above, without needing to actually retrieve any of the article content itself.

**Query 1 — English query, unfiltered:**

```python
response = (client.query
            .get("Wikipedia", ['text', 'title', 'url', 'views', 'lang'])
            .with_near_text({"concepts": "Vacation spots in california"})
            .with_limit(5)
            .do()
           )

json_print(response)
```

**Query 1b — English query, filtered to English-language results only:**

```python
response = (client.query
            .get("Wikipedia", ['text', 'title', 'url', 'views', 'lang'])
            .with_near_text({"concepts": "Vacation spots in california"})
            .with_where({
                "path": ['lang'],
                "operator": "Equal",
                "valueString": 'en'
            })
            .with_limit(3)
            .do()
           )

json_print(response)
```

**Query 2 — Polish-language query, same English-only filter:**

```python
response = (client.query
            .get("Wikipedia", ['text', 'title', 'url', 'views', 'lang'])
            .with_near_text({"concepts": "Miejsca na wakacje w Kalifornii"})
            .with_where({
                "path": ['lang'],
                "operator": "Equal",
                "valueString": 'en'
            })
            .with_limit(3)
            .do()
           )

json_print(response)
```

**Query 3 — Arabic-language query, same English-only filter:**

```python
response = (client.query
            .get("Wikipedia", ['text', 'title', 'url', 'views', 'lang'])
            .with_near_text({"concepts": "أماكن العطلات في كاليفورنيا"})
            .with_where({
                "path": ['lang'],
                "operator": "Equal",
                "valueString": 'en'
            })
            .with_limit(3)
            .do()
           )

json_print(response)
```

**Structural notes on the query pattern:**
- `.get("Wikipedia", [...])` specifies the collection to query and which properties to retrieve for each returned object — here, `text`, `title`, `url`, `views`, and `lang` (the article's language code).
- `.with_near_text({"concepts": "..."})` is the same `nearText` semantic-query operator introduced in §5.3.5, here supplied with the query phrase under the `"concepts"` key. Note that the query text itself changes across the three queries above (English → Polish → Arabic), while the `.with_where(...)` filter and `.with_limit(...)` remain identical — isolating the language of the *query* as the only variable, which is what makes this a clean demonstration of true cross-lingual semantic matching rather than any other confound.
- `.with_where({"path": ['lang'], "operator": "Equal", "valueString": 'en'})` applies a structured property filter — directly paralleling the `foo > 44` filter pattern introduced in §5.2.5 — constraining results to only those objects whose `lang` property equals `'en'`, regardless of what language the incoming query itself was written in.
- `.with_limit(n)` caps the number of returned results, exactly as in earlier examples.
- `.do()` executes the constructed query chain against the Weaviate instance and returns the JSON response, which is then pretty-printed via `json_print`.

## 7.2 Retrieval-Augmented Generation (RAG) — Implementation

Building on the conceptual overview given in §1.2, this section demonstrates RAG concretely using the same Weaviate + Cohere + OpenAI setup as the multilingual search example.

### 7.2.1 The Two Core RAG Query Types Demonstrated

**A. Single-prompt ("per-object") generation.** A prompt template is defined, incorporating a placeholder for each retrieved object's content — illustratively, something like:

```
"Write me a Facebook ad about {title}, using information inside {text}"
```

Here, `{title}` and the retrieved article text are dynamically substituted in from **each individual object returned by the underlying semantic query**. The query is then extended with a **`generate`** clause (referred to as "with generate"), which passes this same constructed prompt template to the LLM **separately for every single retrieved object**.

**Worked example.** A semantic query first retrieves three relevant objects, and then a prompt like "write me a Facebook ad about..., using information inside..." is applied independently to each of the three retrieved objects — producing **three separate, independently generated pieces of content**, one per source object (e.g., three distinct ad copy variations, each grounded in the text of a different retrieved Wikipedia article).

**Reference code — Single Prompt:**

```python
prompt = "Write me a facebook ad about {title} using information inside {text}"

result = (
  client.query
  .get("Wikipedia", ["title", "text"])
  .with_generate(single_prompt=prompt)
  .with_near_text({
    "concepts": ["Vacation spots in california"]
  })
  .with_limit(3)
).do()

json_print(result)
```

**Notes:**
- The `prompt` string uses `{title}` and `{text}` as placeholders that Weaviate automatically fills in, **per retrieved object**, from that object's own `title` and `text` properties — the developer does not need to manually loop over results and perform string substitution themselves.
- `.with_generate(single_prompt=prompt)` is the extension to the standard query chain (compare directly against the plain `nearText` queries in §7.1.3) that activates the generative module and specifies **single-prompt mode**: the LLM is called once *per retrieved object*, using that object's own data to fill the template.
- The overall chain otherwise looks identical to an ordinary semantic search (`.get(...)`, `.with_near_text(...)`, `.with_limit(...)`) — reinforcing the point made in §7.2.2 below that RAG queries are best understood as an ordinary semantic search with one additional `.with_generate(...)` clause layered on top.
- Because `.with_limit(3)` retrieves three objects, and single-prompt mode generates once per object, this call results in **three separate LLM generations**, one embedded in the response corresponding to each of the three retrieved Wikipedia articles.

**B. Group-task ("combined") generation.** Rather than generating one output per retrieved object, a **group task** takes a single prompt and combines **all** of the retrieved results together into **one single call** to the LLM, producing **one unified generation** as output.

**Worked example.** A group-task prompt requesting a summary of the retrieved posts "into paragraphs" was applied to the same three retrieved objects, producing a **single combined summary** synthesizing information from all three source articles together — rather than three independent outputs.

**Reference code — Group Task:**

```python
generate_prompt = "Summarize what these posts are about in two paragraphs."

result = (
  client.query
  .get("Wikipedia", ["title", "text"])
  .with_generate(grouped_task=generate_prompt)  # Pass in all objects at once
  .with_near_text({
    "concepts": ["Vacation spots in california"]
  })
  .with_limit(3)
).do()

json_print(result)
```

**Notes:**
- The only structural difference from the single-prompt example is swapping `.with_generate(single_prompt=prompt)` for `.with_generate(grouped_task=generate_prompt)` — a small API-level change that produces a fundamentally different generation behavior (per-object vs. combined-across-objects), which is a useful detail to remember when choosing between the two modes in application code.
- Because `grouped_task` does not reference per-object placeholders like `{title}` or `{text}`, the prompt itself is written as a general instruction ("summarize what these posts are about") — Weaviate handles bundling the full text of all three retrieved objects into the single underlying LLM call automatically, without needing manual concatenation logic in the client code.
- The explanatory inline comment `# Pass in all objects at once` in the source code directly describes this behavior: rather than three separate LLM calls, this results in exactly **one** LLM call that has been given the combined content of all `.with_limit(3)`-worth of retrieved objects as its context.

### 7.2.2 RAG Query Structure — Summary

The overall pattern for a RAG query in this framework closely mirrors an ordinary semantic search query (as covered in Parts IV–VI), with one addition: a `generate` clause specifying either:
- A **single-prompt** generation (per-object), or
- A **group-task** generation (combined across all retrieved objects).

This directly implements the three-step RAG pipeline introduced in §1.2.1 — **query the vector database → combine retrieved context with a prompt → send to the LLM for generation** — and demonstrates both of the natural variations on how retrieved context can be packaged for the generation step (individually vs. combined).

---

# Part VIII — Consolidated Reference Material

## 8.1 Key Terminology Glossary

| Term | Definition |
|---|---|
| **Embedding** | A dense numerical vector representation of a data object (image, text, etc.) that captures its meaning; similar data yields similar (spatially close) vectors. |
| **Vector database** | A database specialized for storing and efficiently searching vector embeddings, typically also supporting standard database features (CRUD, filtering). |
| **RAG (Retrieval-Augmented Generation)** | A technique for grounding an LLM's output in externally retrieved information (often from a vector database) instead of relying solely on the model's trained parametric knowledge. |
| **Semantic search** | Search based on the meaning of a query, as opposed to exact keyword matching. |
| **Sparse search / keyword search** | Search based on exact term/keyword overlap; underlying vectors are mostly zero-valued (e.g., bag-of-words, BM25). |
| **Dense search** | Semantic/vector-embedding-based search; underlying vectors have (largely) non-zero values across most dimensions. |
| **Hybrid search** | Search combining sparse and dense search scores into a single ranked result set, typically via a tunable weighting parameter. |
| **KNN (K-Nearest Neighbors)** | The classical, exact algorithm for finding the K closest vectors to a query by computing and sorting distances to all stored vectors ("brute-force search"). |
| **ANN (Approximate Nearest Neighbors)** | A family of algorithms that trade some accuracy/recall for large gains in query speed, avoiding exhaustive brute-force comparison. |
| **NSW (Navigable Small World)** | A graph-based ANN construction where each node connects to its nearest neighbors, enabling greedy graph traversal toward a query. |
| **HNSW (Hierarchical Navigable Small World)** | A multi-layer extension of NSW, with sparser graphs at higher layers enabling fast coarse navigation and denser graphs at lower layers enabling fine-grained refinement; achieves $O(\log n)$ query time. |
| **Recall** | The fraction of true nearest neighbors that a search algorithm actually succeeds in returning; the key accuracy metric that ANN algorithms trade off against speed. |
| **Autoencoder** | A neural network architecture with an encoder (compresses input to a bottleneck vector) and decoder (reconstructs input from that vector); used here to illustrate how embeddings can be learned. |
| **Variational autoencoder (VAE)** | An autoencoder variant that learns a distribution over the latent (embedding) space rather than a single deterministic vector, typically producing a smoother, better-structured embedding space. |
| **BM25 (Best Matching 25)** | A widely used keyword-ranking algorithm that scores matches by term frequency, down-weighting common words and up-weighting rare, distinctive term matches. |
| **CRUD** | Create, Read, Update, Delete — the four basic operations supported by a full-featured database, including vector databases like Weaviate. |
| **`alpha` parameter (Weaviate hybrid search)** | Tunable weighting between dense and sparse search scores in a hybrid query; near 1 favors dense/vector search, near 0 favors sparse/keyword search. |
| **`nearText` (Weaviate)** | Query operator for issuing a plain-text semantic query, which Weaviate automatically vectorizes using the collection's configured vectorizer module before searching. |
| **`nearVector` (Weaviate, implicit)** | Query mode for issuing a raw, pre-computed vector directly as the query. |
| **Vectorizer module (Weaviate)** | A pluggable component (e.g., `Text2Vec OpenAI`, Cohere, Hugging Face) that automatically generates embeddings for objects at import time and for queries at search time. |
| **Embedded mode (Weaviate)** | A deployment mode allowing Weaviate to run directly inside a local notebook/process, without a separately managed server — convenient for prototyping. |

## 8.2 Distance Metric Cheat Sheet

| Metric | "Better match" direction | Sensitive to vector magnitude? | Common domain |
|---|---|---|---|
| Euclidean | Lower is better | Yes | General purpose |
| Manhattan | Lower is better | Yes | Grid-like distance notions |
| Dot product | **Higher** is better; negative ⇒ dissimilar | Yes | NLP (common) |
| Cosine distance | Lower is better | **No** (direction only) | NLP (common) |

## 8.3 Search Strategy Decision Guide

| If your data / query is... | Prefer... | Why |
|---|---|---|
| Natural language with conceptual/semantic meaning (e.g., product descriptions, articles, general Q&A) | Dense (vector) search | Captures synonyms, paraphrase, conceptual similarity |
| Exact codes, serial numbers, IDs, or highly specific rare terms | Sparse (keyword) search | Dense embeddings struggle with "meaningless" strings; exact term match is what matters |
| Query domain differs substantially from the embedding model's training domain | Sparse (keyword) search, or a domain-appropriate embedding model | Dense search accuracy degrades sharply under domain mismatch (the "doctor fixing a car" problem) |
| Mixed needs — some queries benefit from semantics, others need exact term precision | Hybrid search, tuning `alpha` | Combines strengths of both approaches in a single query |
| Need guaranteed exact nearest neighbors, small dataset | Brute-force KNN | Exact search is computationally feasible only at small scale |
| Large-scale production dataset (millions+ objects), latency-sensitive | ANN (e.g., HNSW) | $O(\log n)$ scaling makes large-scale, low-latency search feasible |
| Content and/or queries span multiple languages | Multilingual embedding model (e.g., Cohere multilingual) + dense/hybrid search | Semantically equivalent content in different languages maps to similar embeddings |
| Need to ground LLM output in current/proprietary information | RAG (vector database + prompt augmentation) | Avoids costly retraining/fine-tuning; reduces hallucination; enables citation |

## 8.4 End-to-End Pipeline Checklist (Building a RAG Application)

1. **Choose an embedding strategy.**
   - Select an embedding model appropriate to your data's domain and language requirements (e.g., a multilingual model if serving multiple languages).
   - Decide whether to compute vectors manually or use a vector database's built-in vectorizer module (e.g., Weaviate's `Text2Vec OpenAI`).
2. **Choose a distance metric** appropriate to the embedding model (cosine and dot product are the standard defaults for NLP embeddings; confirm which convention — magnitude-sensitive vs. direction-only — matches your model's training assumptions).
3. **Set up the vector database collection/schema**, specifying the vectorizer (or `null` for manual vectors) and the distance metric.
4. **Ingest data using batch loading**, especially at scale (tens of thousands of objects or more), rather than one-object-at-a-time inserts.
5. **Decide on a search strategy**: pure dense, pure sparse, or hybrid — using the decision guide in §8.3 — and tune the hybrid `alpha` parameter if applicable.
6. **Decide on a result-selection strategy**: fixed top-K count, or a distance/quality threshold, depending on whether a consistent result count or a consistent quality bar is more important for your application.
7. **Add filtering** on non-vector properties where useful, to combine structured filtering with semantic relevance.
8. **Implement the RAG generation step**, choosing between:
   - **Per-object (single-prompt) generation** — when you want independent outputs grounded in each individual retrieved source.
   - **Group-task generation** — when you want a single unified output synthesized across all retrieved sources.
9. **Maintain the data over time** using standard CRUD operations (create, read, update, delete) as your underlying source data changes.
10. **Monitor for scaling concerns** — if the deployment grows toward production scale, ensure the vector database is using an ANN index (like HNSW) rather than brute-force search, to keep query latency acceptable.

## 8.5 Frequently Asked Questions

**Q: Why not just fine-tune the LLM on my proprietary data instead of using RAG?**
A: Fine-tuning/retraining is expensive and slow, and does not scale well to frequently changing information. RAG lets new or private information be incorporated simply by adding it to the vector database, with no retraining required.

**Q: Why does dot product behave differently from the other distance metrics?**
A: Dot product is not a "distance" in the same sense as Euclidean or Manhattan distance — it measures the magnitude of one vector's projection onto another. Because of this, **higher** values indicate greater similarity (with negative values indicating dissimilarity), the opposite convention from Euclidean/Manhattan/cosine distance, where **lower** values indicate greater similarity. This must be accounted for explicitly when sorting or ranking results.

**Q: When should I use cosine distance versus dot product?**
A: Both are common in NLP. Cosine distance considers only the *direction* of vectors (ignoring magnitude), while dot product is influenced by both direction and magnitude. If the embedding vectors are unit-normalized, dot product and cosine similarity become mathematically equivalent — so the choice sometimes comes down to whether your vectors are pre-normalized and which convention your surrounding tooling expects.

**Q: Why does brute-force KNN fail at scale, when it's the most accurate method?**
A: Its query time grows **linearly** with the number of stored vectors ($O(n)$), and also increases with vector dimensionality. At production scales (hundreds of millions of high-dimensional vectors), this becomes far too slow for interactive applications — a single query was shown taking around 2 seconds at just 500,000 objects with 768 dimensions, which would compound to unacceptable total latency across many queries.

**Q: Does approximate nearest neighbor search always find the "wrong" answer sometimes?**
A: Not "wrong" so much as occasionally suboptimal — depending on the random entry point and the greedy nature of graph traversal, an ANN search (like NSW/HNSW) can sometimes converge on a **locally** optimal node that is not the **globally** closest vector. This is an accepted trade-off: in exchange for occasionally returning a near-miss instead of the perfect match, query time improves from linear to logarithmic scaling, which is essential at production scale.

**Q: Should I always use hybrid search instead of choosing one method?**
A: Not necessarily — hybrid search is a strong general-purpose default because it lets you tune between semantic and keyword matching using `alpha`, but pure dense or pure sparse search each still has cases where they are clearly the better fit (see the decision guide in §8.3) — for example, pure sparse search is well-suited to precise identifier/code lookups, while pure dense search is well-suited to purely conceptual/paraphrase-style queries.

**Q: How does multilingual search actually work — is it just translating the query first?**
A: No explicit translation step is involved. A multilingual embedding model is trained such that semantically equivalent text in different languages is mapped to similar vector representations directly. Because search operates purely on vector similarity, queries and documents in different languages can be matched directly through their embeddings, without any separate machine-translation step.

**Q: What's the difference between the "single-prompt" and "group task" RAG generation modes?**
A: Single-prompt generation applies a prompt template to **each retrieved object individually**, producing one generation per object (useful when you want distinct, independently-grounded outputs). Group task generation applies a single prompt to **all retrieved objects combined**, producing one unified generation that synthesizes information across all of them (useful for summarization or synthesis tasks).

---

# Appendix A — Full Empirical Scaling Data (Brute-Force KNN)

For quick reference, the empirical brute-force query timing results discussed in §4.2.2 are consolidated here.

**Scaling by object count, 2-dimensional vectors:**

| Object count | Relative query time behavior |
|---|---|
| 20 | Near-instant |
| 20,000 | Fast |
| 200,000 | Slower, but still fast |
| 2,000,000 | Slower |
| 20,000,000 | ~10× increase versus the 2,000,000 case |
| 200,000,000 | ~12 seconds |

**Scaling by object count, 768-dimensional vectors (more realistic embedding size):**

| Object count | Relative query time behavior |
|---|---|
| 1,000 | ~0.5 ms |
| 10,000 | Fast |
| 100,000 | Still fairly fast |
| 500,000 | ~2 seconds; ~30 minutes total for 1,000 sequential queries |

**Interpretation.** These two tables together demonstrate that both **dataset size** and **vector dimensionality** independently drive up brute-force query cost, and that realistic production embedding dimensionality (hundreds of dimensions) makes the scaling problem considerably worse than the simplified 2-dimensional illustrative examples suggest.

---

# Appendix B — Course and Contributor Context

This material is based on a short course titled **"Vector Databases: Embeddings to Applications,"** developed in partnership with **Weaviate**. The instructor was **Sebastian Witalec**, Head of Developer Relations at Weaviate. The course was introduced by Andrew Ng. Additional contributors acknowledged include **Zain Hassan** (Weaviate) and **Jeff Ludwig** and **Ismail Gaggari** from DeepLearning.AI.

The course's stated learning objectives, as an overview:
- Understand how neural networks can represent and embed data as numeric vectors (via an autoencoder architecture, applied to MNIST digit images).
- Understand what it means for data objects to be similar or dissimilar, and how to quantify this using distance metrics (Euclidean, Manhattan, dot product, cosine).
- Understand different vector search paradigms: linear (brute-force/exact) search and approximate search.
- Understand different search strategies: sparse, dense, and hybrid search.
- Build real-world applications of vector databases, including a RAG system with hybrid and multilingual search functionality.

> **Note on names:** Contributor names are rendered here using standard spellings based on context; minor phonetic transcription artifacts in the original spoken material (e.g., alternate renderings of "Witalec," "Gaggari") have been normalized for clarity, without altering their substantive role or contribution as described.
