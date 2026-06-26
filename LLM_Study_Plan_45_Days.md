# 🧠 LLM Mastery: 45-Day Intensive Study Plan
### Based on: *LLM Open University (GitHub)* + *Hands-On LLMs* by Jay Alammar

---

> **Format:** 9 hours/day · 3 slots of 3 hours each
> **Slot A — Morning (3h):** Theory & Reading
> **Slot B — Afternoon (3h):** Videos, Articles & Deep Dives
> **Slot C — Evening (3h):** Hands-On Coding & Practice

---

## 📋 Overview & Phases

| Phase | Days | Focus |
|---|---|---|
| **Phase 1: Foundations** | Days 1–10 | LLM Basics, Transformers, Architecture |
| **Phase 2: Build & Train** | Days 11–25 | Datasets, Fine-Tuning, Evaluation, Quantization, RLHF |
| **Phase 3: Production Apps** | Days 26–38 | Prompt Engineering, RAG, Agents, Deployment, LLMOps |
| **Phase 4: Portfolio & Capstone** | Days 39–45 | Portfolio Projects & Integration |

---

## 🔵 PHASE 1: LLM Foundations & Architecture (Days 1–10)

---

### Day 1 — What Are LLMs? Big Picture & Landscape

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* Ch. 1: Introduction — What Are Large Language Models?
- Focus on: history of language models, what makes LLMs different from classical NLP, the pre-training/fine-tuning paradigm

**Slot B — Afternoon (Videos & Articles)**
- Watch: [Intro to Large Language Models by Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g) (1h)
- Read: Repo Section 1 overview — skim the roadmap structure to understand what's ahead
- Explore: What is GPT? What is BERT? A survey of model families (blog articles of your choice)

**Slot C — Evening (Practice)**
- Set up your Python environment: install `transformers`, `torch`, `datasets`, `openai`
- Run your first inference with a Hugging Face model (e.g., GPT-2 text generation, BERT fill-mask)
- Journal: Write 1 page summarizing "What is an LLM?" in your own words

---

### Day 2 — Tokenization Deep Dive

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* Ch. 2 (or relevant tokenization sections): How text becomes tokens
- Study: BPE (Byte Pair Encoding), WordPiece, SentencePiece — concepts and trade-offs

**Slot B — Afternoon (Videos & Articles)**
- Watch: [Let's Build the GPT Tokenizer by Andrej Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE) (2.5h — core video for this day)

**Slot C — Evening (Practice)**
- Code along with Karpathy's tokenizer from scratch in a Jupyter notebook
- Experiment with the `tiktoken` library (OpenAI's tokenizer) and Hugging Face tokenizers
- Compare token counts across different models for the same text; observe vocabulary differences

---

### Day 3 — Attention Mechanisms & Self-Attention

**Slot A — Morning (Theory)**
- Read: [Attention? Attention! by Lilian Weng](https://lilianweng.github.io/posts/2018-06-24-attention/) — full post
- Read *Hands-On LLMs* Ch. 3 (Attention & embedding sections as applicable)
- Focus on: dot-product attention, scaled attention, multi-head attention concepts

**Slot B — Afternoon (Videos & Articles)**
- Watch: [Visual Intro to Transformers by 3Blue1Brown](https://www.youtube.com/watch?v=wjZofJX0v4M) (45 min)
- Read: [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/) — read slowly, draw diagrams

**Slot C — Evening (Practice)**
- Implement a simple scaled dot-product attention function in PyTorch from scratch
- Visualize attention weights on a sample sentence using a pre-trained BERT model
- Test: Feed different sentences, inspect how attention patterns shift

---

### Day 4 — The Transformer Architecture (Encoder & Decoder)

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* relevant architecture chapters (encoder-decoder, decoder-only)
- Study: Layer normalization, Feed-Forward layers, residual connections, positional encoding
- Compare: BERT (encoder-only) vs. GPT (decoder-only) vs. T5 (encoder-decoder)

**Slot B — Afternoon (Videos & Articles)**
- Re-visit [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — second careful pass
- Read: [The Illustrated GPT-2 by Jay Alammar](https://jalammar.github.io/illustrated-gpt2/) — full post

**Slot C — Evening (Practice)**
- Implement a minimal Transformer block in PyTorch (embedding → attention → FFN → layer norm)
- Load a GPT-2 model, inspect its architecture with `print(model)` and `model.config`
- Draw the full architecture from memory, then verify against the paper/diagrams

---

### Day 5 — Building nanoGPT (Part 1)

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* — pre-training objectives: Causal Language Modeling (CLM) vs. Masked Language Modeling (MLM)
- Study: Cross-entropy loss for language modeling, next-token prediction

**Slot B — Afternoon (Videos & Articles)**
- Watch: [nanoGPT by Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) — first 1.5 hours (character-level bigram model through attention)

**Slot C — Evening (Practice)**
- Follow along and code the bigram language model from Karpathy's nanoGPT video
- Train on the tiny Shakespeare dataset; watch the loss decrease
- Experiment with hyperparameters: learning rate, batch size, context length

---

### Day 6 — Building nanoGPT (Part 2) & Decoding Strategies

**Slot A — Morning (Theory)**
- Study decoding strategies: Greedy search, Beam search, Top-k sampling, Top-p (nucleus) sampling, Temperature scaling
- Read: [Decoding Strategies in LLMs by Maxime Lebonne](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html)

**Slot B — Afternoon (Videos & Articles)**
- Watch: [nanoGPT by Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) — remaining portion (full transformer implementation)

**Slot C — Evening (Practice)**
- Complete nanoGPT implementation end-to-end — working GPT from scratch
- Implement and compare different decoding strategies on your nanoGPT model
- Run text generation with greedy vs. top-p=0.9 vs. temperature=1.5 — compare outputs

---

### Day 7 — Embeddings & Semantic Representations

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* chapters on embeddings: word embeddings, contextual embeddings, sentence embeddings
- Study: Word2Vec intuition, ELMo context dependency, how BERT embeddings differ from static embeddings

**Slot B — Afternoon (Videos & Articles)**
- Read: Jay Alammar's blog posts on word2vec and BERT visual guides (available at jalammar.github.io)
- Study: How sentence transformers work (SBERT), cosine similarity for semantic search

**Slot C — Evening (Practice)**
- Use `sentence-transformers` library to embed sentences; compute similarity between pairs
- Build a tiny semantic search: embed 20 sentences, find the most similar to a query
- Visualize embeddings in 2D using t-SNE/PCA with `matplotlib`

---

### Day 8 — Generative AI with LLMs Course (Part 1)

**Slot A — Morning (Theory)**
- Start: [Generative AI with Large Language Models (Coursera / DeepLearning.AI)](https://www.coursera.org/learn/generative-ai-with-llms) — Week 1 content
- Focus on: LLM use case framing, the prompt → completion paradigm, in-context learning

**Slot B — Afternoon (Videos & Articles)**
- Continue Week 1 course lectures
- Supplementary: Watch the H2O.ai LLM Learning Path playlist (first 2 episodes)

**Slot C — Evening (Practice)**
- Complete all Week 1 course labs
- Experiment with few-shot prompting on a public LLM API
- Document: 3 examples where few-shot prompting improved output quality

---

### Day 9 — Generative AI with LLMs Course (Part 2) + Scaling Laws

**Slot A — Morning (Theory)**
- Continue course: Week 2 — Fine-tuning concepts, instruction tuning, PEFT overview
- Study: Scaling laws — Chinchilla paper intuition (model size vs. data size trade-offs)

**Slot B — Afternoon (Videos & Articles)**
- Continue course lectures (Week 2)
- Read: Chinchilla scaling law summary blog post / article of your choice

**Slot C — Evening (Practice)**
- Complete Week 2 course labs
- Explore: Load different-sized GPT-2 variants (small/medium/large/xl) and compare generation quality

---

### Day 10 — Phase 1 Review & Consolidation

**Slot A — Morning (Theory)**
- Review all key concepts from Days 1–9: tokenization, attention, transformer, embeddings, decoding, scaling
- Re-read any sections of *Hands-On LLMs* that felt unclear

**Slot B — Afternoon (Videos & Articles)**
- Watch: [State of GPT by Andrej Karpathy](https://www.youtube.com/watch?v=bZQun8Y4L2A) — training pipeline from tokenization to RLHF

**Slot C — Evening (Practice)**
- **Mini-project:** Build a complete text generation pipeline — tokenize input → GPT-2 inference → decode with multiple strategies → display outputs
- Write a 2-page technical summary of "How LLMs Work" — your Phase 1 knowledge checkpoint
- Set up your experiment tracking with Weights & Biases (W&B) for upcoming fine-tuning work

---

## 🟠 PHASE 2: Building & Training LLMs (Days 11–25)

---

### Day 11 — LLM Datasets: Foundations & Hubs

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* relevant sections on training data
- Study: What makes a good LLM training dataset? Data quality, diversity, filtering, deduplication

**Slot B — Afternoon (Videos & Articles)**
- Watch: [How to Use HuggingFace's Datasets](https://www.youtube.com/watch?v=GhGUZrcB-WM)
- Explore: [LLMDataHub on GitHub](https://github.com/Zjh-819/LLMDataHub) — browse available datasets

**Slot C — Evening (Practice)**
- Use the `datasets` library: load 3 different public datasets (e.g., The Pile, OpenWebText, Alpaca)
- Write a data exploration script: count tokens, analyze text length distribution, check for duplicates
- Hands-on: Filter and clean a small dataset of 10,000 examples

---

### Day 12 — Building Instruction Datasets

**Slot A — Morning (Theory)**
- Read: [How to Fine-Tune an LLM Part 1: Preparing a Dataset for Instruction Tuning (W&B)](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2)
- Study: Instruction format (system prompt, user, assistant), chat templates, alpaca vs. ShareGPT format

**Slot B — Afternoon (Videos & Articles)**
- Read: [How to Generate Instruction Datasets from Any Documents for LLM Fine-Tuning](https://towardsdatascience.com/how-to-generate-instruction-datasets-from-any-documents-for-llm-fine-tuning-abb319a05d91/)
- Read: Medium article on creating datasets with GPT-3.5 for Llama 2 fine-tuning (from repo resources)

**Slot C — Evening (Practice)**
- Build a small instruction dataset from a topic of your choice using an LLM API (GPT/Claude)
- Format it correctly in alpaca JSON format (instruction / input / output)
- Save it to HuggingFace Hub (create a free account)

---

### Day 13 — Introduction to Fine-Tuning

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* fine-tuning chapters
- Study: Full fine-tuning vs. PEFT; when to fine-tune vs. prompt engineer; catastrophic forgetting

**Slot B — Afternoon (Videos & Articles)**
- Take: [Finetuning Large Language Models (DeepLearning.AI short course)](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/) — Lessons 1–4
- Read: [Fine-Tuning LLMs: Overview, Methods, and Best Practices (Turing)](https://www.turing.com/resources/finetuning-large-language-models)

**Slot C — Evening (Practice)**
- Run Notebook 1 from the repo's fine-tuning collection: [LoRA + HuggingFace Fine-Tuning](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/1.Efficiently_train_Large_Language_Models_with_LoRA_and_Hugging_Face.ipynb)
- Document every step, change one hyperparameter, observe the difference

---

### Day 14 — PEFT: LoRA & QLoRA (Theory)

**Slot A — Morning (Theory)**
- Study LoRA in depth: low-rank decomposition, rank r, alpha parameter, target modules
- Read: [LoRA Fine-tuning & Hyperparameters Explained in Plain English](https://www.entrypointai.com/blog/lora-fine-tuning/)
- Read: [Optimizing Pre-trained Models: A Guide to PEFT (LeewayHertz)](https://www.leewayhertz.com/parameter-efficient-fine-tuning/)

**Slot B — Afternoon (Videos & Articles)**
- Watch: [QLoRA paper explained (Efficient Finetuning of Quantized LLMs)](https://www.youtube.com/watch?v=6l8GZDPbFn8)
- Read: [Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments](https://lightning.ai/pages/community/lora-insights/)

**Slot C — Evening (Practice)**
- Run Notebook 7: [Fine-Tune LLaMA 2 with QLoRA](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/7.FineTune_LLAMA2_with_QLORA.ipynb)
- Experiment with different LoRA ranks (r=4, r=8, r=16) — compare training loss and GPU memory

---

### Day 15 — PEFT: Practical Fine-Tuning Lab Day

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* on instruction tuning and PEFT in production
- Study: Prompt formatting for instruction-following models (Mistral, Llama 2 chat templates)

**Slot B — Afternoon (Videos & Articles)**
- Watch: [QLoRA — How to Fine-tune an LLM on a Single GPU](https://www.youtube.com/watch?v=XpoKB3usmKc)
- Watch: [Fine-Tuning Mistral-7B with LoRA](https://www.youtube.com/watch?v=kV8yXIUC5_4)

**Slot C — Evening (Practice)**
- Run Notebook 2: [Fine-Tune Llama 2 in Colab](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/2.Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.ipynb)
- Run Notebook 14: [Finetuning Mistral-7b using Autotrain](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/14.Finetuning_Mistral_7b_Using_AutoTrain.ipynb)
- Log results to W&B; compare both experiments

---

### Day 16 — Fine-Tuning Diverse Architectures

**Slot A — Morning (Theory)**
- Study: Model-specific fine-tuning considerations — Falcon, Bloom, GPT-NeoX architectures
- Read about: Instruction tuning (FLAN), chat fine-tuning, domain adaptation

**Slot B — Afternoon (Videos & Articles)**
- Take: [Finetuning Large Language Models (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/) — complete remaining lessons

**Slot C — Evening (Practice)**
- Run Notebook 4: [PEFT Finetune Bloom-560m-tagger](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/4.PEFT%20Finetune-Bloom-560m-tagger.ipynb)
- Run Notebook 6: [Fine-Tuning Falcon-7b with Self-Supervised Training](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/6.Finetune%20Falcon-7b%20with%20BNB%20Self%20Supervised%20Training.ipynb)
- Compare architectures: What changed in the code? What stayed the same?

---

### Day 17 — LLM Evaluation: Overview & Metrics

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* evaluation chapters
- Read: [Understanding LLM Evaluation and Benchmarks: A Complete Guide (Turing)](https://www.turing.com/resources/understanding-llm-evaluation-and-benchmarks)
- Study: BLEU, ROUGE, Perplexity — their formulas, limitations, and use cases

**Slot B — Afternoon (Videos & Articles)**
- Read: [BLEU at Your Own Risk by Rachael Tatman](https://medium.com/data-science/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213)
- Read: [Perplexity of Fixed-Length Models (HuggingFace docs)](https://huggingface.co/docs/transformers/en/perplexity)
- Read: [HumanEval: Decoding the LLM Benchmark for Code Generation](https://deepgram.com/learn/humaneval-llm-benchmark)

**Slot C — Evening (Practice)**
- Implement BLEU score calculation from scratch in Python
- Use `evaluate` library to compute BLEU, ROUGE on sample text-summary pairs
- Calculate perplexity of GPT-2 on a test set using the HuggingFace formula

---

### Day 18 — LLM Benchmarking & Chatbot Evaluation

**Slot A — Morning (Theory)**
- Read: [The Definitive Guide to LLM Benchmarking (ConfidentAI)](https://www.confident-ai.com/blog/the-current-state-of-benchmarking-llms)
- Study: MMLU, HellaSwag, ARC, TruthfulQA benchmarks — what they measure and why

**Slot B — Afternoon (Videos & Articles)**
- Take: [Evaluating and Debugging Generative AI (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/) — full short course
- Read: [Chatbot Arena: Benchmarking LLMs with Elo Ratings](https://lmsys.org/blog/2023-05-03-arena/)

**Slot C — Evening (Practice)**
- Take: [Automated Testing for LLMOps (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/automated-testing-llmops/) — first half
- Set up an evaluation pipeline for a text summarization task using `evaluate`
- Run your fine-tuned model from Day 15 through ROUGE scoring vs. a baseline GPT-2

---

### Day 19 — RAG Evaluation & Automated Testing

**Slot A — Morning (Theory)**
- Study: RAG evaluation triad — Context Relevance, Groundedness, Answer Relevance
- Read: [Decoding LLM Performance: A Guide to Evaluating LLM Applications](https://amagastya.medium.com/decoding-llm-performance-a-guide-to-evaluating-llm-applications-e8609665a213)

**Slot B — Afternoon (Videos & Articles)**
- Take: [Building and Evaluating Advanced RAG Applications (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)
- Complete: [Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/) — remaining lessons

**Slot C — Evening (Practice)**
- Build a simple LLM-as-judge evaluation pipeline: use one LLM to score another's outputs
- Implement a G-Eval style evaluator for coherence on generated summaries
- Document: evaluation results table comparing 3 models on the same task

---

### Day 20 — Quantization: Theory & Overview

**Slot A — Morning (Theory)**
- Read: [A Guide to Quantization in LLMs (Symbl.ai)](https://symbl.ai/developers/blog/a-guide-to-quantization-in-llms/)
- Read: [Introduction to Weight Quantization (MLAbonne)](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html)
- Study: FP32 → FP16 → INT8 → INT4; symmetric vs. asymmetric quantization

**Slot B — Afternoon (Videos & Articles)**
- Watch: [Which Quantization Method is Right for You? GPTQ vs GGUF vs AWQ](https://www.youtube.com/watch?v=mNE_d-C82lI)
- Read: [LLM Quantization: GPTQ, QAT, AWQ, GGUF, GGML, PTQ (Medium)](https://medium.com/@siddharth.vij10/llm-quantization-gptq-qat-awq-gguf-ggml-ptq-2e172cd1b3b5)

**Slot C — Evening (Practice)**
- Load a GPT-2 model in FP32 and FP16; compare memory usage with `torch.cuda.memory_allocated()`
- Apply dynamic INT8 quantization using PyTorch `torch.quantization`
- Benchmark inference speed: FP32 vs. FP16 vs. INT8 on the same input, 100 runs

---

### Day 21 — Quantization: GGUF, GPTQ & AWQ (Hands-On)

**Slot A — Morning (Theory)**
- Study GPTQ: second-order optimization, layer-by-layer quantization, calibration dataset
- Study GGUF: llama.cpp format, CPU offloading, layer offloading to GPU

**Slot B — Afternoon (Videos & Articles)**
- Watch: [AWQ for LLM Quantization](https://www.youtube.com/watch?v=3dYLj9vjfA0)
- Watch: [How to Quantize an LLM with GGUF or AWQ](https://www.youtube.com/watch?v=XM8pllpBVA0)
- Read: [Quantize Llama models with GGUF and llama.cpp (MLAbonne)](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html)

**Slot C — Evening (Practice)**
- Install `llama.cpp` and run a GGUF quantized model locally (e.g., Mistral-7B-Q4_K_M)
- Use `auto-gptq` to quantize a small model (e.g., GPT-2) to INT4
- Compare: model size on disk, loading time, and generation quality across quantization levels

---

### Day 22 — RLHF & LLM Alignment (Part 1)

**Slot A — Morning (Theory)**
- Read: [How RLHF Actually Works by Nathan Lambert](https://www.interconnects.ai/p/how-rlhf-works)
- Read: [RLHF: Reinforcement Learning from Human Feedback by Chip Huyen](https://huyenchip.com/2023/05/02/rlhf.html)
- Study: Reward model training, policy optimization, PPO algorithm basics

**Slot B — Afternoon (Videos & Articles)**
- Watch: [RLHF: From Zero to ChatGPT (talk)](https://www.youtube.com/watch?v=2MBJOuVq380) — full talk
- Read: [StackLLaMA: A hands-on guide to train LLaMA with RLHF (HuggingFace)](https://huggingface.co/blog/stackllama)

**Slot C — Evening (Practice)**
- Read and understand the RLHF notebook: Notebook 11 — [RLHF Training for Custom Dataset](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/11_RLHF_Training_for_CustomDataset_for_AnyModel.ipynb)
- Study `trl` (Transformer Reinforcement Learning) library: PPOTrainer, RewardTrainer classes
- Sketch a diagram of the full RLHF pipeline from pre-training to aligned chat model

---

### Day 23 — RLHF & LLM Alignment (Part 2: DPO & Alternatives)

**Slot A — Morning (Theory)**
- Study: DPO (Direct Preference Optimization) — how it simplifies RLHF by removing the reward model
- Study: Constitutional AI (Anthropic), RLAIF, Rejection Sampling Fine-Tuning
- Read the InstructGPT paper abstract and key findings (from repo resources)

**Slot B — Afternoon (Videos & Articles)**
- Watch: [State of GPT by Andrej Karpathy](https://www.youtube.com/watch?v=bZQun8Y4L2A) (second watch — now with RLHF context)
- Explore: DPO tutorial blog post or HuggingFace `trl` DPO documentation

**Slot C — Evening (Practice)**
- Run a DPO training experiment using `trl`'s `DPOTrainer` on a small preference dataset
- Compare: outputs from supervised fine-tuned model vs. DPO-aligned model
- Run Notebook 13: [Fine-Tuning OpenAI GPT-3.5 Turbo](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/13.Fine_tuning_OpenAI_GPT_3_5_turbo.ipynb)

---

### Day 24 — Vision Language Models (VLMs)

**Slot A — Morning (Theory)**
- Read: [Multimodality and Large Multimodal Models by Chip Huyen](https://huyenchip.com/2023/10/10/multimodal.html) — full post
- Study: CLIP architecture, contrastive learning, image-text alignment, SigLIP

**Slot B — Afternoon (Videos & Articles)**
- Watch: [Coding a Multimodal Vision Language Model from Scratch in PyTorch](https://www.youtube.com/watch?v=vAmKB7iPkWw) — first 2.5 hours
- Explore: [smol-vision course on GitHub](https://github.com/merveenoyan/smol-vision)

**Slot C — Evening (Practice)**
- Continue watching the VLM from scratch video (remaining 2.5h)
- Use a Hugging Face VLM (e.g., LLaVA or PaliGemma) for image captioning
- Build: a simple image Q&A pipeline with a vision-language model

---

### Day 25 — Phase 2 Review & Integration

**Slot A — Morning (Theory)**
- Review: Dataset pipelines → Fine-tuning (LoRA/QLoRA) → Evaluation → Quantization → RLHF → VLMs
- Re-read *Hands-On LLMs* summary chapters or any chapters that need revisiting

**Slot B — Afternoon (Videos & Articles)**
- Watch 2 episodes from the H2O.ai LLM Learning Path playlist (advanced topics)
- Read: [Training & Fine-Tuning LLMs for Production (ActiveLoop course overview)](https://learn.activeloop.ai/courses/llms)

**Slot C — Evening (Practice)**
- **Integration mini-project:** Fine-tune a small model on your custom dataset (Day 12) → quantize it → evaluate it with BLEU/ROUGE
- Write a 2-page technical report: "My Fine-Tuning Experiment" — what you did, results, learnings
- Set up your "staying updated" system: subscribe to 3 newsletters from the repo's list

---

## 🟢 PHASE 3: Building LLM Production Applications (Days 26–38)

---

### Day 26 — Prompt Engineering Fundamentals

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* prompt engineering chapters
- Study: Zero-shot, few-shot, chain-of-thought, self-consistency, ReAct, role prompting

**Slot B — Afternoon (Videos & Articles)**
- Take: [Prompt Engineering for LLMs (Maven — DAIR.AI)](https://maven.com/dair-ai/prompt-engineering-llms) — first 2 sessions
- Read: [Prompt Engineering Guide](https://www.promptingguide.ai/) — core techniques

**Slot C — Evening (Practice)**
- Run a systematic prompt comparison: same task, 5 different prompt strategies, score outputs
- Build a prompt template library for: summarization, classification, extraction, generation
- Implement chain-of-thought prompting for a math word problem; measure accuracy improvement

---

### Day 27 — Advanced Prompt Engineering & Guardrails

**Slot A — Morning (Theory)**
- Study: Prompt injection attacks, jailbreaking, adversarial prompts — how they work and defenses
- Study: Structured output prompting (JSON mode, function calling, tool use)

**Slot B — Afternoon (Videos & Articles)**
- Continue: [Prompt Engineering for LLMs (Maven)](https://maven.com/dair-ai/prompt-engineering-llms) — remaining sessions
- Read: [Maximizing the Potential of LLMs: A Guide to Prompt Engineering (DZone)](https://dzone.com/articles/maximizing-the-potential-of-llms-a-guide-to-prompt)

**Slot C — Evening (Practice)**
- Implement a prompt injection detector using an LLM
- Build a function-calling pipeline with OpenAI API for weather/search tools
- Experiment with structured JSON output constraints using Pydantic + LLM

---

### Day 28 — Vector Databases & Embeddings for Search

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* vector database sections
- Study: FAISS, Pinecone, Chroma, Weaviate, Qdrant — architectures and trade-offs
- Study: HNSW algorithm, approximate nearest neighbor search, indexing strategies

**Slot B — Afternoon (Videos & Articles)**
- Explore: Repo Section 9 — Top Resources to Master Vector Databases
- Read documentation: Chroma getting started, FAISS tutorial

**Slot C — Evening (Practice)**
- Build a semantic search engine using Chroma + sentence-transformers on 1,000 Wikipedia chunks
- Add, query, and delete documents; test with 10 different queries
- Compare search quality: keyword (BM25) vs. semantic embedding search

---

### Day 29 — RAG Fundamentals (Naive RAG)

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* RAG chapters (core architecture)
- Study: Document loading → chunking → embedding → storing → retrieval → generation pipeline
- Study: Chunking strategies: fixed-size, sentence, recursive, semantic chunking

**Slot B — Afternoon (Videos & Articles)**
- Take: [Building and Evaluating Advanced RAG Applications (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — first half
- Explore Repo Section 10: Master RAG resources

**Slot C — Evening (Practice)**
- Build a naive RAG system from scratch using LangChain + Chroma + OpenAI:
  1. Load a PDF document
  2. Chunk it (RecursiveCharacterTextSplitter)
  3. Embed + store in Chroma
  4. Retrieve top-k chunks per query
  5. Generate an answer with context
- Test with 10 questions; identify failure modes

---

### Day 30 — Advanced RAG Techniques

**Slot A — Morning (Theory)**
- Study: Query expansion, HyDE (Hypothetical Document Embeddings), multi-query retrieval
- Study: Reranking with cross-encoders, contextual compression, RAPTOR
- Study: Hybrid search (BM25 + dense retrieval), parent-child chunking

**Slot B — Afternoon (Videos & Articles)**
- Complete: [Building and Evaluating Advanced RAG Applications (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)
- Read: Advanced RAG blog posts from the repo's resource list

**Slot C — Evening (Practice)**
- Upgrade your Day 29 RAG system with:
  - HyDE query transformation
  - Cohere/HuggingFace reranker
  - Hybrid BM25 + vector search
- Evaluate: use your Day 19 RAG evaluation framework to compare naive vs. advanced RAG

---

### Day 31 — Multimodal RAG

**Slot A — Morning (Theory)**
- Study: Multimodal embeddings (CLIP-based), storing and retrieving image+text pairs
- Read course materials: [Multimodal RAG GitHub Course](https://github.com/To-Data-Beyond/Multimodal-RAG)

**Slot B — Afternoon (Videos & Articles)**
- Work through the Multimodal RAG course (first 4 chapters): intro, architecture, embeddings, video processing

**Slot C — Evening (Practice)**
- Build a multimodal RAG pipeline that can retrieve both text and images given a query
- Test: Query a dataset with text+image documents; verify both modalities are retrieved
- Optional: Add a VLM as the reader for image-heavy documents

---

### Day 32 — LLM Agents (Part 1: Foundations)

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* agent chapters (if available) and Repo Section 11
- Study: ReAct framework, tool use, function calling, planning loops
- Study: Agent memory types: in-context, external (vector DB), episodic

**Slot B — Afternoon (Videos & Articles)**
- Explore all resources listed in Repo Section 11: Learning Resources to Master LLM Agents
- Study: LangGraph or AutoGen for multi-agent orchestration

**Slot C — Evening (Practice)**
- Build a ReAct-style agent with 3 tools: web search, calculator, code interpreter
- Test on 10 multi-step reasoning tasks
- Log: How many tasks required tool use? How many failed and why?

---

### Day 33 — LLM Agents (Part 2: Advanced Patterns)

**Slot A — Morning (Theory)**
- Study: Multi-agent frameworks — supervisor pattern, debate pattern, collaborative agents
- Study: Memory management in agents — summarization, entity tracking, episodic recall

**Slot B — Afternoon (Videos & Articles)**
- Take: [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) — agent-related sessions

**Slot C — Evening (Practice)**
- Build a multi-agent system with a planner agent + executor agent + critic agent
- Build a research agent that: searches the web → summarizes → cites sources
- Test on 5 research tasks; evaluate answer quality manually

---

### Day 34 — MCP (Model Context Protocol)

**Slot A — Morning (Theory)**
- Read: Repo Section 17 — Master MCP: The Best Free Learning Resources
- Study: What is MCP? Client-server architecture, tool registration, resource management

**Slot B — Afternoon (Videos & Articles)**
- Work through the free MCP learning resources from the repo

**Slot C — Evening (Practice)**
- Set up a local MCP server with 2 custom tools (e.g., file reader + database query)
- Connect it to Claude Desktop or another MCP-compatible client
- Test: Build an agent workflow using MCP tools

---

### Day 35 — Running LLMs Locally & Inference Optimization

**Slot A — Morning (Theory)**
- Read: Repo Section 16 — Five Free Tools to Run LLMs Locally
- Study: Ollama, LM Studio, Jan.ai, GPT4All — setup and use cases
- Study: Inference optimization: batching, KV-cache, continuous batching, PagedAttention

**Slot B — Afternoon (Videos & Articles)**
- Read: [Getting Started with LLM Inference Optimization (Repo Section 12)](https://github.com/youssefHosni/LLM-Open-University-From-Begineer-to-Advanced)
- Watch: Videos from the inference optimization resources in the repo

**Slot C — Evening (Practice)**
- Set up Ollama locally; run Mistral-7B and Llama 3
- Benchmark: tokens/second at different batch sizes (1, 4, 8, 16)
- Compare: vLLM vs. standard HuggingFace inference speed on the same model

---

### Day 36 — LLM Deployment

**Slot A — Morning (Theory)**
- Read: Repo Section 15 — Deploying LLMs: Top Learning & Educational Resources
- Study: REST API serving, streaming responses, Docker containerization for LLM APIs
- Study: Modal, Replicate, HuggingFace Inference Endpoints — managed deployment options

**Slot B — Afternoon (Videos & Articles)**
- Work through deployment tutorials from the repo resource list
- Study: Load balancing, autoscaling, cost optimization for LLM APIs

**Slot C — Evening (Practice)**
- Deploy your fine-tuned model from Day 15 as a REST API using FastAPI + HuggingFace
- Containerize it with Docker
- Test the endpoint with `curl` and a simple Python client

---

### Day 37 — LLMOps

**Slot A — Morning (Theory)**
- Read: Repo Section 13 — What is LLMOps and How to Get Started
- Study: Experiment tracking, model versioning, prompt versioning, monitoring in production
- Study: Data flywheel — how production data improves future model versions

**Slot B — Afternoon (Videos & Articles)**
- Explore: LangSmith, Weights & Biases LLM tracking, MLflow for LLMs
- Read blog posts on LLMOps best practices from the repo resources

**Slot C — Evening (Practice)**
- Set up LangSmith (free tier) to trace LLM calls from your RAG system (Day 29/30)
- Create a monitoring dashboard: track latency, token count, cost per query
- Implement a simple A/B testing framework for comparing two prompt versions

---

### Day 38 — LLM Security

**Slot A — Morning (Theory)**
- Read: Repo Section 14 — Securing LLMs: Best Learning & Educational Resources
- Study: Prompt injection, jailbreaking, data extraction, PII leakage, model inversion
- Study: Defenses: input validation, output filtering, system prompt hardening, rate limiting

**Slot B — Afternoon (Videos & Articles)**
- Work through the security resources listed in the repo
- Read: OWASP Top 10 for LLMs (2024 edition)

**Slot C — Evening (Practice)**
- Run adversarial prompts against your deployed API (Day 36) — document vulnerabilities
- Implement an input/output guardrail using `guardrails-ai` or `llm-guard`
- Write a security checklist for your LLM application

---

## 🔴 PHASE 4: Portfolio Projects & Capstone (Days 39–45)

---

### Day 39 — Portfolio Project Planning

**Slot A — Morning (Theory)**
- Read: Repo Section IV — Building Your LLM Portfolio Projects
- Study: What makes a strong LLM portfolio? Technical depth + real-world applicability
- Plan: Select 2 projects from the ideas below (or your own):
  - RAG-powered Q&A over a custom document corpus
  - Fine-tuned domain expert chatbot
  - Multimodal product image search + description generator
  - Automated research agent with citations
  - LLM-powered data extraction pipeline

**Slot B — Afternoon (Videos & Articles)**
- Review: Full Stack LLM Bootcamp — project structure and best practices sessions
- Study: How to write a great technical README and project report

**Slot C — Evening (Practice)**
- Set up your project repositories on GitHub (2 repos)
- Write detailed README drafts for both projects
- Define: success metrics, system design diagram, tech stack, and timeline

---

### Day 40 — Project 1: Build (Part 1)

**Slot A — Morning (Theory)**
- Deep dive into the specific concepts needed for Project 1
- Read any remaining *Hands-On LLMs* sections relevant to your project

**Slot B — Afternoon (Videos & Articles)**
- Research: 2 state-of-the-art approaches to your project's core problem

**Slot C — Evening (Practice)**
- Build the data pipeline and core retrieval/inference logic for Project 1
- Commit your working v0.1 to GitHub

---

### Day 41 — Project 1: Build (Part 2) & Evaluation

**Slot A — Morning (Theory)**
- Design your evaluation framework for Project 1
- Review evaluation metrics appropriate to your project type

**Slot B — Afternoon (Videos & Articles)**
- Research and integrate 1 advanced technique into your project (e.g., reranking, hybrid search, DPO)

**Slot C — Evening (Practice)**
- Build the frontend/API interface for Project 1 (Gradio or FastAPI)
- Run evaluation; document results in a results table
- Commit v1.0 to GitHub with full documentation

---

### Day 42 — Project 2: Build (Part 1)

**Slot A — Morning (Theory)**
- Study the specific architecture for Project 2
- Plan the fine-tuning or RAG or agent pipeline required

**Slot B — Afternoon (Videos & Articles)**
- Research: existing similar open-source projects for inspiration

**Slot C — Evening (Practice)**
- Build the core of Project 2
- Commit v0.1 to GitHub with a clear commit message

---

### Day 43 — Project 2: Build (Part 2) & Polish

**Slot A — Morning (Theory)**
- Review: LLMOps checklist for your project — observability, error handling, rate limiting

**Slot B — Afternoon (Videos & Articles)**
- Study: How to write a great technical blog post about your project (for Medium/Substack)

**Slot C — Evening (Practice)**
- Complete Project 2 with evaluation results
- Add monitoring with LangSmith or W&B
- Commit v1.0; write a detailed project post-mortem in your README

---

### Day 44 — Write-Ups, Blog Posts & GitHub Portfolio Polish

**Slot A — Morning (Theory)**
- Outline: 2 technical blog posts (one per project)
- Study: How top AI practitioners present their projects (LinkedIn, X, Medium, Substack)

**Slot B — Afternoon (Videos & Articles)**
- Review: Stay-updated resources from Repo Section 7 — pick your top 3 newsletters, follow key researchers

**Slot C — Evening (Practice)**
- Write blog post drafts for both projects (aim for 800–1,200 words each)
- Polish GitHub READMEs: add architecture diagrams, demo GIFs, installation instructions
- Update your LinkedIn/resume with your two new portfolio projects

---

### Day 45 — Final Review, Gaps & Learning Roadmap Forward

**Slot A — Morning (Theory)**
- Read *Hands-On LLMs* — any remaining unread sections; revisit any concept that still feels fuzzy
- Review your notes and journals from all 45 days — what changed in your understanding?

**Slot B — Afternoon (Videos & Articles)**
- Watch: Any 2 talks or papers you bookmarked but didn't get to
- Subscribe to remaining newsletters: [To Data & Beyond](https://todatabeyond.substack.com/), [Ahead of AI](https://magazine.sebastianraschka.com/), [The Rundown](https://www.therundown.ai/subscribe)

**Slot C — Evening (Reflection & Planning)**
- Write a 3-page "What I Know About LLMs Now" document — your personal LLM reference guide
- Create your 90-day forward learning plan:
  - What to deepen (e.g., training from scratch, specific agent frameworks)
  - Papers to read (e.g., Llama 3, Gemma, Mistral papers)
  - Community to join (HuggingFace Discord, NeurIPS, local AI meetups)
- Publish your blog posts and share your portfolio!

---

## 📚 Key Resources Quick Reference

### Books
| Resource | Used In |
|---|---|
| *Hands-On LLMs* by Jay Alammar | Throughout all phases |

### Foundational Articles & Blogs
| Resource | Days |
|---|---|
| The Illustrated Transformer (Jay Alammar) | Days 3, 4 |
| The Illustrated GPT-2 (Jay Alammar) | Day 4 |
| Attention? Attention! (Lilian Weng) | Day 3 |
| Decoding Strategies (MLAbonne) | Day 6 |
| RLHF by Chip Huyen | Day 22 |
| Multimodality by Chip Huyen | Day 24 |

### Key Video Lectures
| Resource | Days |
|---|---|
| Andrej Karpathy — Intro to LLMs | Day 1 |
| Andrej Karpathy — GPT Tokenizer | Day 2 |
| Andrej Karpathy — nanoGPT | Days 5, 6 |
| Andrej Karpathy — State of GPT | Days 10, 23 |
| 3Blue1Brown — Transformers | Day 3 |
| RLHF: Zero to ChatGPT | Day 22 |
| VLM from Scratch | Day 24 |

### Courses
| Course | Days |
|---|---|
| Generative AI with LLMs (Coursera) | Days 8, 9 |
| Finetuning LLMs (DeepLearning.AI) | Days 13, 16 |
| Evaluating & Debugging GenAI | Day 18 |
| Advanced RAG (DeepLearning.AI) | Days 29, 30 |
| Automated Testing for LLMOps | Days 18, 19 |
| Full Stack LLM Bootcamp | Days 33, 39 |

### Fine-Tuning Notebooks Covered
| Notebook | Day |
|---|---|
| #1 — LoRA + HuggingFace | Day 13 |
| #2 — Llama 2 Fine-Tune | Day 15 |
| #4 — PEFT Bloom-560m | Day 16 |
| #6 — Falcon-7b Self-Supervised | Day 16 |
| #7 — LLaMA 2 + QLoRA | Day 14 |
| #11 — RLHF Custom Dataset | Day 22 |
| #13 — GPT-3.5 Turbo Fine-Tune | Day 23 |
| #14 — Mistral-7b Autotrain | Day 15 |

---

## ✅ Daily Habit Checklist

For every day, check off the following:

- [ ] Completed Slot A (morning theory)
- [ ] Completed Slot B (videos/articles)
- [ ] Completed Slot C (hands-on coding)
- [ ] Committed code to GitHub
- [ ] Wrote a brief journal entry (3–5 bullet points: what I learned, what confused me, what I want to revisit)
- [ ] Reviewed yesterday's journal to consolidate learning

---

## 🏁 End-of-Plan Milestones

By Day 45 you will have:

- Built a GPT from scratch (nanoGPT)
- Fine-tuned at least 3 different LLMs with LoRA/QLoRA
- Implemented quantization at multiple levels (GGUF, GPTQ, AWQ, INT8)
- Evaluated LLMs using BLEU, ROUGE, perplexity, and LLM-as-judge
- Trained an RLHF/DPO-aligned model
- Built both a naive and advanced RAG pipeline
- Created a multimodal RAG application
- Built and tested LLM agents with tool use
- Deployed an LLM as a production REST API with monitoring
- Completed 2 portfolio-ready GitHub projects with blog posts
- Established a daily LLM news feed via newsletters and key researchers to follow

---

*Study plan synthesized from: [LLM Open University GitHub Repo](https://github.com/youssefHosni/LLM-Open-University-From-Begineer-to-Advanced) by Youssef Hosni + "Hands-On LLMs" by Jay Alammar*
