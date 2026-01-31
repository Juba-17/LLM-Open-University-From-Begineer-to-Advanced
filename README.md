# LLM-Roadmap-From-Beginner-to-Advanced
A complete roadmap to master LLMs for absolute beginners to advanced 

Large Language Models (LLMs) have transformed the landscape of AI, driving advancements in natural language processing and generating human-like text with unprecedented accuracy. It is now becoming an essential skill if you are joining the AI market, and most of the open positions will require LLM-related skills. Whether you're a beginner eager to dip your toes into the fascinating world of LLMs or an advanced practitioner aiming to refine your skills, this comprehensive roadmap will guide you through every stage of mastering these powerful models.

This roadmap is divided into four comprehensive sections. In the first section, you will grasp the foundational concepts of Large Language Models (LLMs) and delve into their core architectures, learning about key components like transformers, attention mechanisms, and tokenization.
The second section guides you through the complete process of training and fine-tuning an LLM from scratch, starting with data preparation and moving on to training, fine-tuning, evaluating, and aligning the model for optimal performance.

In the third section, you'll focus on developing real-world applications using LLMs, beginning with prompt engineering and progressing to building Retrieval-Augmented Generation (RAG) applications, as well as learning best practices for deploying and optimizing LLMs in production environments.
The final section is dedicated to helping you build your portfolio, offering ten project ideas and ten guided projects to provide a solid starting point for showcasing your expertise to potential employers or clients.

Each section builds upon the previous one, ensuring a smooth and logical progression from basic concepts to advanced applications, ultimately equipping you with the essential knowledge and practical skills needed to master LLMs and leverage their capabilities effectively.


## Part I: LLM Basics & Architecture ##

<img width="1429" height="771" alt="image" src="https://github.com/user-attachments/assets/9e8cd3bc-86d6-439d-97eb-2d904c09fc7c" />



In the first section, you'll grasp the foundational concepts of Large Language Models (LLMs) and their core architectures, focusing on transformers, attention mechanisms, and tokenization.
Key resources include Andrej Karpathy's "Let's Build the GPT Tokenizer," Jay Alammar's "The Illustrated Transformer" and "The Illustrated GPT-2," and 3Blue1Brown's "Visual Intro to Transformers." You'll also explore "nanoGPT" by Karpathy, "Attention? Attention!" by Lilian Weng, various decoding strategies, Karpathy's "Intro to Large Language Models," and top practical and theoretical courses on LLMs. This section provides a blend of theoretical and useful insights, preparing you for the next sections.

### 1. Articles: ###

1. [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
2. [The Illustrated GPT-2 by Jay Alammar](https://jalammar.github.io/illustrated-gpt2/)
3. [Attention? Attention! by Lilian Weng](https://lilianweng.github.io/posts/2018-06-24-attention/)
4. [Decoding Strategies in LLMs by Maxime Lebonne](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html)

### 2. YouTube Videos: ###

1. [Let's build the GPT Tokenizer by Andrej Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)
2. [nanoGPT by Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
3. [Intro to Large Language Models by Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g)
4. [Visual Intro to Transformers by 3Blue1Brown](https://www.youtube.com/watch?v=wjZofJX0v4M&t=187s)

### 3. Courses: ###

1. [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms?utm_campaign=WebsiteCoursesGAIA&utm_medium=institutions&utm_source=deeplearning-ai#modules)
3. [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/)
4. [Training & Fine-Tuning LLMs for Production](https://learn.activeloop.ai/courses/llms)
5. [H2O.ai LLM Learning Path](https://www.youtube.com/playlist?list=PLNtMya54qvOHQHDpUDtZytwEV2Miali9l)

--------------------------------------------------
## Part II: Building & Training LLM From Scratch ##
<img width="1429" height="771" alt="image" src="https://github.com/user-attachments/assets/5f408c67-cefd-4fac-b395-20ceb3db72da" />


The second section guides you through the complete process of training and fine-tuning a Large Language Model (LLM) from scratch, covering every crucial step from data preparation to ensuring optimal model performance. This section begins with the best resources for building datasets to train LLMs, providing comprehensive guidance on collecting, cleaning, and organizing data for effective model training.

Next, you will delve into mastering the fine-tuning process with top learning resources, exploring techniques and strategies to adapt pre-trained models to specific tasks or domains. The section also includes 14 free LLM fine-tuning notebooks, offering practical, hands-on experience with fine-tuning processes.
Evaluating LLMs is a critical aspect covered in this section, with the best resources to learn and understand various evaluation metrics and methodologies, ensuring your model's performance meets the desired standards. Additionally, you will gain an overview of LLM quantization techniques, which help in optimizing models for efficiency and speed, along with resources for mastering each technique.

Understanding Reinforcement Learning from Human Feedback (RLHF) and LLM alignment is another key component, with top resources provided to deepen your knowledge in aligning models with human values and preferences. Finally, this section offers insights on how to stay updated with the latest LLM research and industry news, ensuring you remain at the forefront of advancements in the field.

By the end of this section, you will have a comprehensive understanding of how to build, train, fine-tune, evaluate, and optimize LLMs, equipped with the knowledge and practical skills to develop high-performing models tailored to specific needs.


### [1. Best Resources on Building Datasets to Train LLMs](https://open.substack.com/pub/youssefh/p/best-resources-on-building-datasets?utm_campaign=post-expanded-share&utm_medium=web) ###

Large language models (LLMs), such as OpenAI’s GPT series and Google’s Bard, are driving profound technological changes. Recently, with the emergence of open-source large model frameworks like LlaMa and ChatGLM, training an LLM is no longer the exclusive domain of resource-rich companies.
Training LLMs by small organizations or individuals has become an important interest in the open-source community, with some notable works including Alpaca, Vicuna, and Luotuo.

In addition to large model frameworks, large-scale and high-quality training corpora are also essential for training large language models. Currently, relevant open-source corpora in the community are still scattered. Therefore, this section aims to introduce and collect high-quality resources to learn how to build training datasets for LLM applications.

#### 1.1. Dataset Hubs #### 

<img width="875" height="438" alt="image" src="https://github.com/user-attachments/assets/fcefc91c-aedc-426c-be1f-787d2694ec7f" />


LLM datasets are extensive sets of text used to train large language models. These datasets typically contain texts in multiple languages, topics, and styles, used to train models to predict and generate text related to given input text. They are commonly employed for various natural language processing tasks like machine translation, summarization, question-answering systems, and more. The dataset hubs contain open-source datasets that are pivotal in training or fine-tuning many LLMs that ML engineers use today.

**Resources:** 

1. [LLMDataHub: Awesome Datasets for LLM Training](https://github.com/Zjh-819/LLMDataHub)
2. [How to Use HuggingFace’s Datasets](https://www.youtube.com/watch?v=GhGUZrcB-WM)
3. [Open-Sourced Training Datasets for Large Language Models (LLMs)](https://kili-technology.com/blog/9-open-sourced-datasets-for-training-large-language-models)


#### 1.2. Building an Instruction Dataset #####

<img width="875" height="438" alt="image" src="https://github.com/user-attachments/assets/9c0902ae-c071-4765-8f3a-8a1090a982df" />


Training a chatbot LLM that can follow human instructions effectively requires access to high-quality datasets that cover a range of conversation domains and styles. In this section, we provide a curated collection of resource datasets specifically designed for building an instruction dataset for instruction-tuning LLM.

**Resources:**

1. [How to Fine-Tune an LLM Part 1: Preparing a Dataset for Instruction Tuning](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2)
2. [How I created an instruction dataset using GPT 3.5 to fine-tune Llama 2 for news classification](https://medium.com/@kshitiz.sahay26/how-i-created-an-instruction-dataset-using-gpt-3-5-to-fine-tune-llama-2-for-news-classification-ed02fe41c81f)
3. [Dataset creation for fine-tuning LLM](https://colab.research.google.com/drive/1GH8PW9-zAe4cXEZyOIE-T9uHXblIldAg?usp=sharing)
4. [How to Generate Instruction Datasets from Any Documents for LLM Fine-Tuning](https://towardsdatascience.com/how-to-generate-instruction-datasets-from-any-documents-for-llm-fine-tuning-abb319a05d91/)

#### 1.3. Datasets Preparing & Processing ####

<img width="850" height="776" alt="image" src="https://github.com/user-attachments/assets/e340fdd2-c2e6-40b5-9208-5307de1bcd21" />

Enhancing the performance of the LLM and RAG systems depends on efficiently processing diverse unstructured data sources. In this section, you’ll learn techniques for representing all sorts of unstructured data, like text, images, and tables, from many different sources and implement them to extend your LLM RAG pipeline to include Excel, Word, PowerPoint, PDF, and EPUB files.

**Resources:**

1. [How to Build an Effective Data Collection and Processing Strategy for LLM Training](https://www.turing.com/resources/building-effective-data-strategy-for-llm-training)
2. [Preprocessing Unstructured Data for LLM Applications](https://www.deeplearning.ai/short-courses/preprocessing-unstructured-data-for-llm-applications/)

--------------------------------------------------
### [2. Practical Guide to LLM Fine-Tuning ](https://open.substack.com/pub/youssefh/p/mastering-large-language-model-llm?utm_campaign=post-expanded-share&utm_medium=web) ###

Large language models (LLMs) have transformed the field of natural language processing with their advanced capabilities and highly sophisticated solutions.
These models, trained on massive datasets of text, perform a wide range of tasks, including text generation, translation, summarization, and question-answering. But while LLMs are powerful tools, they’re often incompatible with specific tasks or domains.
Fine-tuning allows users to adapt pre-trained LLMs to more specialized tasks. By fine-tuning a model on a small dataset of task-specific data, you can improve its performance on that task while preserving its general language knowledge.
In this section, we will provide the best learning resource to learn what fine-tuning is, how it works, and how fine-tuning LLMs can significantly improve model performance, reduce training costs, and enable more accurate and context-specific results.
Also, these resources will cover different fine-tuning techniques and applications to show how fine-tuning has become a critical component of LLM-powered solutions.

#### 2.1. Introduction to LLM Fine-Tuning ####

<img width="875" height="492" alt="image" src="https://github.com/user-attachments/assets/c81deeee-bd19-46b0-b31d-622630bed6bb" />

LLMs are trained on massive datasets of text and can perform a wide range of tasks, including text generation, translation, summarization, and question-answering. But while LLMs are powerful tools, they’re often incompatible with specific tasks or domains.
Fine-tuning allows users to adapt pre-trained LLMs to more specialized tasks. By fine-tuning a model on a small dataset of task-specific data, you can improve its performance on that task while preserving its general language knowledge. For example, a Google study found that fine-tuning a pre-trained LLM for sentiment analysis improved its accuracy by 10 percent.

In this section, you will explore how fine-tuning LLMs can significantly improve model performance, reduce training costs, and enable more accurate and context-specific results.
You will also learn the different fine-tuning techniques and applications to show how fine-tuning has become a critical component of LLM-powered solutions.

**Learning Resources:**
1. [The Novice’s LLM Training Guide](https://rentry.org/llm-training)
2. [Fine-Tuning LLMs: Overview, Methods, and Best Practices](https://www.turing.com/resources/finetuning-large-language-models)
3. [Finetuning Large Language Models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)


#### 2.2. Parameter Efficient Fine-Tuning (PEFT) ####

<img width="875" height="583" alt="image" src="https://github.com/user-attachments/assets/581c50e7-a909-4672-b343-86efc14f14c7" />

Transfer learning plays a crucial role in the development of large language models such as GPT-3 and BERT. It is an ML technique in which a model trained on a certain task is used as a starting point for a distinct but similar task.
The idea behind transfer learning is that the knowledge gained by a model from solving one problem can be leveraged to help solve another problem. However, with the parameter count of large language models reaching trillions, fine-tuning the entire model has become computationally expensive and often impractical.

In response, the focus has shifted towards in-context learning, where the model is provided with prompts for a given task and returns in-context updates. However, inefficiencies like processing the prompt each time the model makes a prediction and its poor performance at times make it a less favorable choice.

This is where Parameter-efficient Fine-tuning (PEFT) comes in as an alternative paradigm to prompting. PEFT aims to fine-tune only a small subset of the model’s parameters, achieving comparable performance to full fine-tuning while significantly reducing computational requirements.
These learning resources will introduce you to the PEFT method in detail, exploring its benefits and how it has become an efficient way to fine-tune LLMs on downstream tasks. Also, it will discuss different methods of the PEFT with practical examples of each.

**Learning Resources:**

1. [Optimizing Pre-trained Models: A Guide to Parameter-Efficient Fine-Tuning (PEFT)](https://www.leewayhertz.com/parameter-efficient-fine-tuning/)
2. [QLoRA paper explained (Efficient Finetuning of Quantized LLMs)](https://www.youtube.com/watch?v=6l8GZDPbFn8)
3. [QLoRA — How to Fine-tune an LLM on a Single GPU (w/ Python Code)](https://www.youtube.com/watch?v=XpoKB3usmKc&t=2s)
4. [LoRA Fine-tuning & Hyperparameters Explained (in Plain English)](https://www.entrypointai.com/blog/lora-fine-tuning/)
5. [Fine-Tuning Mistral-7B with LoRA (Low-Rank Adaptation)](https://www.youtube.com/watch?v=kV8yXIUC5_4)
6. [Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments](https://lightning.ai/pages/community/lora-insights/)

#### 2.3. Practical Fine Tuning ####

Fine-tuning large language models (LLMs) has become a crucial skill for NLP practitioners, enabling customization and improved performance across various tasks. This article introduces 14 free Colab notebooks that provide hands-on experience in fine-tuning LLMs.
From efficient training methodologies like LoRA and Hugging Face to specialized models such as Llama, Guanaco, and Falcon, each notebook explores unique aspects of the fine-tuning process. Advanced techniques like PEFT Finetune, Bloom-560m-tagger, and Meta_OPT-6–1b_Model offer insights into state-of-the-art approaches.

Whether you’re interested in GPT-Neo-X, MPT-Instruct-30B, or Microsoft Phi 15B, these notebooks cover a diverse range of LLMs, making them suitable for both beginners and experienced practitioners. Delve into custom dataset training, self-supervised methods, and RLHF techniques, gaining a comprehensive understanding of fine-tuning.

This section provides a roadmap to navigate these notebooks, making it an essential read for anyone keen on mastering the art of fine-tuning large language models.

1. [Fine-Tuning Large Language Models with LoRA and Hugging Face](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/1.Efficiently_train_Large_Language_Models_with_LoRA_and_Hugging_Face.ipynb)
2. [Fine-Tuning Llama 2 Model in a Colab Notebook](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/2.Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.ipynb)
3. [Guanaco Chatbot Demo with LLaMA-7B Model](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/3.Guanaco%20Chatbot%20Demo%20with%20LLaMA-7B%20Model.ipynb#scrollTo=cJMiQrtRuNqH)
4. [PEFT Finetune-Bloom-560m-tagger](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/4.PEFT%20Finetune-Bloom-560m-tagger.ipynb#scrollTo=hsD1VKqeA62Z)
5. [Fine-Tuning Meta_OPT-6–1b Model with bnb_peft](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/5.Finetune_Meta_OPT-6-1b_Model_bnb_peft.ipynb#scrollTo=WE5GJ6s7y0Xo)
6. [Fine-Tuning Falcon-7b with BNB Self-Supervised Training](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/6.Finetune%20Falcon-7b%20with%20BNB%20Self%20Supervised%20Training.ipynb#scrollTo=C2EgqEPDQ8v6)
7. [Fine-Tuning LLaMa2 with QLoRa](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/7.FineTune_LLAMA2_with_QLORA.ipynb#scrollTo=imgHssi6BF0T)
8. [Stable Vicuna 1 3B-8bit in Google Colab](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/8.Stable_Vicuna13B_8bit_in_Colab.ipynb#scrollTo=EUGsc-8IpfEb)
9. [GPT-Neo-X 20B bnb2bit Training](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/9.GPT-neo-x-20B-bnb_4bit_training.ipynb#scrollTo=XIyP_0r6zuVc)
10. [MPT-Instruct-30B Model Training](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/10.MPT_Instruct_30B.ipynb#scrollTo=SsCR0FZHL1pg)
11. [RLHF Training for Custom Dataset for Any Model](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/11_RLHF_Training_for_CustomDataset_for_AnyModel.ipynb#scrollTo=hgj2hw_WQecA)
12. [Fine-Tuning Microsoft Phi 1.5 On Custom Dataset](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/12_Fine_tuning_Microsoft_Phi_1_5b_on_custom_dataset(dialogstudio).ipynb#scrollTo=rpf1Z0k4RJM6)
13. [Fine-Tuning OpenAI GPT3.5 Turbo](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/13.Fine_tuning_OpenAI_GPT_3_5_turbo.ipynb#scrollTo=AJagDsTsIn9z)
14. [Finetuning Mistral-7b using Autotrain Advanced](https://colab.research.google.com/github/ashishpatel26/LLM-Finetuning/blob/main/14.Finetuning_Mistral_7b_Using_AutoTrain.ipynb)

--------------------------------------------------
### [3. Best Resources to Learn & Understand Evaluating LLMs](https://open.substack.com/pub/youssefh/p/best-resources-to-learn-and-understand?utm_campaign=post-expanded-share&utm_medium=web)  ###
As LLMs continue to play a vital role in both research and daily use, their evaluation becomes increasingly critical, not only at the task level but also at the societal level for a better understanding of their potential risks. Over the past years, significant efforts have been made to examine LLMs from various perspectives.
This section presents a comprehensive set of resources that will help you understand LLM evaluation, starting from what to evaluate, where to evaluate, and how to evaluate.

#### 3.1. Overview of LLM Evaluation Methods ####

With the rapid advancement and integration of large language models (LLMs) in business workflows, ensuring these models are reliable and efficient has become critical. This need underscores the significance of understanding and deploying robust evaluation and benchmarking techniques for successful model implementation.

LLMs are evaluated and benchmarked on various tasks such as language generation, translation, reasoning, summarization, question-answering, and relevance. A representative set of evaluations helps build well-rounded, robust, and secure models across different dimensions and detects any regressions over a period of time.

In this section, we explore the nuances of evaluation metrics, the significance of LLM benchmarks in quantifying model performance, and the challenges associated with building standardized metrics. We also touch upon the latest trends in benchmarking and provide a comprehensive guide on building effective evaluation protocols.

**Resources:**
1. [Understanding LLM Evaluation and Benchmarks: A Complete Guide](https://www.turing.com/resources/understanding-llm-evaluation-and-benchmarks)
2. [Decoding LLM Performance: A Guide to Evaluating LLM Applications](https://amagastya.medium.com/decoding-llm-performance-a-guide-to-evaluating-llm-applications-e8d7939cafce)
3. [A Survey on Evaluation of LLMs](https://arxiv.org/abs/2307.03109)
4. [Evaluating and Debugging Generative AI](https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/)


#### 3.2. LLM Benchmarking ####

It is evident that merely training LLMs is not sufficient. Thus, the question arises: How can we confidently assert that LLM ‘A’ (with ’n’ number of parameters) is superior to LLM ‘B’ (with ‘m’ parameters)? Or is LLM ‘A’ more reliable than LLM ‘B’ based on quantifiable, reasonable observations? There needs to be a standard to benchmark LLMs, ensuring they are ethically reliable and factually performant.

In this section, you will learn about the current evaluation paradigm and understand the terminologies of LLM benchmarking/evaluation. You will also learn about some prominent research on evaluating benchmarking and comparing LLMs on various tasks or scenarios.

**Resources:**
1. [The Definitive Guide to LLM Benchmarking](https://www.confident-ai.com/blog/the-current-state-of-benchmarking-llms)

#### 3.3. LLM Evaluation Methods ####

In the era of artificial intelligence and machine learning, evaluating the performance of models is crucial for their development and improvement. Large Language Models (LLMs) have shown incredible capabilities in generating human-like text, and their application has been extended to code generation.
In this section, you will explore different LLM evaluation methods for different use cases. Starting with traditional ones, such as BLEU, to code generation evaluation metrics, such as HumanEval.

**Resources:**
1. [BLEU at your own risk by Rachael Tatman](https://medium.com/data-science/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213)
2. [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity)
3. [HumanEval: Decoding the LLM Benchmark for Code Generation](https://deepgram.com/learn/humaneval-llm-benchmark)


####  3.4. Evaluating Chatbots ####

Following the great success of ChatGPT, there has been a proliferation of open-source large language models that are finetuned to follow instructions. These models are capable of providing valuable assistance in response to users’ questions/prompts. Notable examples include Alpaca and Vicuna, based on LLaMA, and OpenAssistant and Dolly, based on Pythia.

Despite the constant release of new models every week, the community faces a challenge in benchmarking these models effectively. Benchmarking LLM assistants is extremely challenging because the problems can be open-ended, and it is very difficult to write a program to automatically evaluate the response quality. In this case, we typically have to resort to human evaluation based on pairwise comparison.
In this section, you will learn about the Elo rating system, which is a widely used rating system in chess and other competitive games. The Elo rating system is promising to provide the desired property mentioned above.

**Resources:**
1. [Chatbot Arena: Benchmarking LLMs in the Wild with Elo Ratings](https://lmsys.org/blog/2023-05-03-arena/)

#### 3.5. Evaluating RAG Applications ####

Retrieval Augmented Generation (RAG) stands out as one of the most popular use cases of large language models (LLMs). This method facilitates the integration of an LLM with an organization’s proprietary data. Therefore it is important to evaluate and track experimenting to improve your RAG pipeline’s performance. Also, to understand the RAG triad: Context Relevance, Groundedness, and Answer Relevance, which are methods to evaluate the relevance and truthfulness of your LLM’s response.

**Resources:**
1. [Building and Evaluating Advanced RAG Applications](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)

#### 3.6. Automated Testing for LLMs ####

When building applications with generative AI, model behavior is less predictable than traditional software. That’s why systematic testing can make an even bigger difference in saving you development time and cost.
Continuous integration, a key part of LLMOps, is the practice of making small changes to software in development and thoroughly testing them to catch issues early when they are easier to fix.
With a robust automated testing pipeline, you’ll be able to isolate bugs before they accumulate — when they’re easier and less costly to fix. Automated testing lets your team focus on building new features so that you can iterate and ship products faster.

**Resources:**
1. [Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)


--------------------------------------------------
### [4. Overview of LLM Quantization Techniques & Where to Learn Each of Them?](https://open.substack.com/pub/youssefh/p/overview-of-llm-quantization-techniques?utm_campaign=post-expanded-share&utm_medium=web) ###

Model Quantization enhances the efficiency of large language models (LLMs) by representing their parameters in low-precision data types. This article presents an overview of LLM quantization techniques and resources for learning each of them.
This section covers different quantization methods, including GGUF, AWQ, PTQ, GPTQ, and QAT, elucidating their mechanisms and applications in LLM optimization. Each sub-section provides learning resources, including tutorials, specifications, and practical guides, facilitating a deeper understanding of the quantization techniques. This section serves as a comprehensive guide for individuals interested in exploring LLM quantization, offering insights into various techniques and resources for continued learning and professional development.

#### 4.1. Introduction to Quantization ####
<img width="875" height="369" alt="image" src="https://github.com/user-attachments/assets/cc087bc0-f429-4176-bade-504dd395d3c3" />

Model Quantization is a topic that has been gaining popularity recently. The concept of quantization in AI or specifically neural networks, is a technique to represent the weights, biases, and activations in low-precision data types like 8-bit integer (int8) instead of the usual 32-bit floating point (float32). The two most common quantization cases are float32 -> float16 and float32 -> int8. In this section, you will be introduced to model quantization and what are the main techniques for it:

**Learning Resources:**

1. [A Guide to Quantization in LLMs](https://symbl.ai/developers/blog/a-guide-to-quantization-in-llms/)
2. [Introduction to Weight Quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html)
3. [LLM Quantization | GPTQ | QAT | AWQ | GGUF | GGML | PTQ](https://medium.com/@siddharth.vij10/llm-quantization-gptq-qat-awq-gguf-ggml-ptq-2e172cd1b3b5)
4. [Which Quantization Method is Right for You? (GPTQ vs. GGUF vs. AWQ)](https://www.youtube.com/watch?v=mNE_d-C82lI)
5. [Model Quantization Methods In TensorFlow Lite](https://studymachinelearning.com/model-quantization-methods-in-tensorflow-lite/)
6. [ExLlamaV2: The Fastest Library to Run LLMs](https://mlabonne.github.io/blog/posts/ExLlamaV2_The_Fastest_Library_to_Run%C2%A0LLMs.html)
7. [Democratizing LLMs: 4-bit Quantization for Optimal LLM Inference](https://towardsdatascience.com/democratizing-llms-4-bit-quantization-for-optimal-llm-inference-be30cf4e0e34/)


#### 4.2. GGUF ####

<img width="875" height="492" alt="image" src="https://github.com/user-attachments/assets/9daa02ed-2b22-49b7-9084-c608dd53760d" />

GGML is a C library focused on machine learning. It was created by Georgi Gerganov, which is what the initials “GG” stand for. This library not only provides foundational elements for machine learning, such as tensors but also a unique binary format to distribute LLMs.
This format recently changed to GGUF. This new format is designed to be extensible so that new features don’t break compatibility with existing models. It also centralizes all the metadata in one file, such as special tokens, RoPE scaling parameters, etc.
In short, it answers a few historical pain points and should be future-proof. For more information, you can read the specification at this address. GGML was designed to be used in conjunction with the llama.cpp library, also created by Georgi Gerganov.
The library is written in C/C++ for efficient inference of Llama models. It can load GGML models and run them on a CPU. Originally, this was the main difference with GPTQ models, which are loaded and run on a GPU. However, you can now offload some layers of your LLM to the GPU with llama.cpp. To give you an example, there are 35 layers for a 7b parameter model. This drastically speeds up inference and allows you to run LLMs that don’t fit in your VRAM.
Learning Resources:

**Learning Resources:**

1. [Quantize Llama models with GGUF and llama.cpp](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html)
2. [How to Quantize an LLM with GGUF or AWQ](https://www.youtube.com/watch?v=XM8pllpBVA0)

#### 4.3. Activation-aware Weight Quantization (AWQ) ####
<img width="875" height="353" alt="image" src="https://github.com/user-attachments/assets/29c38048-ee24-4b50-ad91-85cf11eb8ba4" />

AWQ takes the concept of weight quantization to the next level by considering the activations of the model during the quantization process. In traditional weight quantization, the weights are quantized independently of the data they process. In AWQ, the quantization process takes into account the actual data distribution in the activations produced by the model during inference.
Here’s how AWQ works:
- Collect Activation Statistics: During a calibration phase, a subset of the data is used to collect statistics on the activations produced by the model. This involves running the model on this data and recording the range of values and the distribution of activations.
- Searching Weight Quantization Parameters: Weights are quantized by taking the activation statistics into account. Concretely, we perform a space search for quantization parameters (e.g., scales and zero points), to minimize the distortions incurred by quantization on output activations. As a result, the quantized weights can be accurately represented with fewer bits.
- Quantizing: With the quantization parameters in place, the model weights are quantized using a reduced number of bits.

**Learning Resources:**

1. [AWQ for LLM Quantization](https://www.youtube.com/watch?v=3dYLj9vjfA0&t=360s)
2. [Understanding Activation-Aware Weight Quantization (AWQ): Boosting Inference Serving Efficiency in LLMs](https://medium.com/friendliai/understanding-activation-aware-weight-quantization-awq-boosting-inference-serving-efficiency-in-10bb0faf63a8)
3. [How to Quantize an LLM with GGUF or AWQ](https://www.youtube.com/watch?v=XM8pllpBVA0)


#### 4.4. Post-Training Model Quantization (PTQ) ####
<img width="875" height="556" alt="image" src="https://github.com/user-attachments/assets/5749194c-c186-4c92-b54f-0d0cd2f183c1" />


Post-training quantization computes the scale after the network has been trained. A representative dataset is used to capture the distribution of activations for each activation tensor, then this distribution data is used to compute the scale value for each tensor. Each weight distribution is used to compute the weight scale.

<img width="875" height="67" alt="image" src="https://github.com/user-attachments/assets/d665c06e-5b56-44e3-b7e6-07b85fabe6a0" />

**Learning Resources:**

1. [Post-training Quantization](https://www.youtube.com/watch?v=n8GT_XflSLA)
2. [Diving deeper into Quantization Realm: Post-Training Magic (PTQ)](https://www.youtube.com/watch?v=n8GT_XflSLA)
3. [Post Training Quantization with OpenVINO Toolkit](https://learnopencv.com/post-training-quantization-with-openvino-toolkit/#overview-of-deep-learning-model-quantization)

#### 4.5. Accurate Post-Training Quantization for Generative Pre-trained Transformers (GPTQ) ####
<img width="875" height="493" alt="image" src="https://github.com/user-attachments/assets/673b6823-12bf-4e2a-8dc5-8e36ff0ec7a1" />

GPTQ is a post-training quantization ( PTQ) method to make the model smaller with a calibration dataset. The idea behind GPTQ is very simple: it quantizes each weight by finding a compressed version of that weight, that will yield a minimum mean squared error. The GPTQ algorithm requires calibrating the quantized weights of the model by making inferences on the quantized model.

The effectiveness of quantization greatly depends on the samples for evaluating and refining their quality. These samples serve as a basis for comparing the outputs of the original and quantized models. By using a higher number of samples, the potential for precise and impactful comparisons increases, subsequently enhancing the quality of quantization.

**Learning Resources:**

1. [WTH is LLM quantization? 4-bit GPTQ?](https://dharanichowdary25.medium.com/wth-is-llm-quantization-4bit-gptq-6e635748178d)
2. [LLM Quantization w/ QLoRA, GPTQ and Llamacpp, LLama 2](https://www.youtube.com/watch?v=YEVyupJxt1Q&t=704s)
3. [Optimize open LLMs using GPTQ and Hugging Face Optimum](https://www.philschmid.de/gptq-llama)
4. [4-bit Quantization with GPTQ](https://towardsdatascience.com/4-bit-quantization-with-gptq-36b0f4f02c34/)

#### 4.6. Quantization-Aware Training (QAT) ####

<img width="721" height="377" alt="image" src="https://github.com/user-attachments/assets/5c6c26c7-08cd-4ede-ae67-0c616036eb23" />

Quantization Aware Training (QAT) aims at computing scale factors during training. Once the network is fully trained, Quantize (Q) and Dequantize (DQ) nodes are inserted into the graph following a specific set of rules. The network is then further trained for a few epochs in a process called Fine-Tuning. Q/DQ nodes simulate quantization loss and add it to the training loss during fine-tuning, making the network more resilient to quantization. In other words, QAT can better preserve accuracy when compared to PTQ.

<img width="875" height="77" alt="image" src="https://github.com/user-attachments/assets/f306f5e9-9c4a-4820-a60b-fc02177ba39a" />

**Learning Resources:**
1. [Quantization Aware Training — Concepts](https://www.youtube.com/watch?v=hGkTFa7FSE0)
2. [Quantization-aware training comprehensive guide](https://notebook.community/tensorflow/model-optimization/tensorflow_model_optimization/g3doc/guide/quantization/training_comprehensive_guide)


--------------------------------------------------
### [5. Top Resources to Learn & Understand RLHF & LLM Alignment](https://open.substack.com/pub/youssefh/p/top-resources-to-learn-and-understand?utm_campaign=post-expanded-share&utm_medium=web)

Reinforcement Learning from Human Feedback (RLHF) has become one of the main building blocks of building chatbots and large language applications. This section aims to provide you with a curated list of top resources that will help you learn and understand RLHF in depth. Whether you are a researcher, developer, or simply curious about this exciting field, this compilation of blogs, videos, talks, and research papers will serve as a comprehensive guide on RLHF.

#### What is RLHF? ####
Reinforcement Learning from Human Feedback (RLHF) is a branch of machine learning that combines reinforcement learning (RL) algorithms with human guidance or feedback to improve the learning process. In RLHF, instead of relying solely on an environmental reward signal, the learning agent interacts with human experts who provide feedback or demonstrations to guide the learning process.
The primary motivation behind RLHF is to enable machines to learn complex tasks more efficiently and effectively by leveraging human expertise. While RL algorithms can learn from trial and error, they can require a large number of interactions with the environment to achieve desirable performance. By incorporating human feedback, RLHF aims to reduce the number of interactions needed and accelerate the learning process.
RLHF is used in the development of chatbots to enhance their performance and improve their ability to interact with users. Here are some reasons why RLHF is used in developing chatbots:

1. **User Satisfaction:** Chatbots aim to provide helpful and engaging conversations with users. By incorporating RLHF, chatbots can learn from human feedback and adapt their responses based on user preferences, leading to more satisfying interactions. Human feedback helps the chatbot understand what kind of responses are desirable and how to improve over time.
2. **Rapid Learning:** Training chatbots solely through traditional methods, such as rule-based systems or supervised learning, can be time-consuming and limited in their ability to handle diverse user inputs. RLHF enables chatbots to learn directly from interactions with human experts, reducing the reliance on large amounts of pre-existing data. This accelerates the learning process and allows chatbots to quickly adapt to new situations.
3. **Handling Uncertainty:** Chatbot conversations can often involve ambiguity and uncertainty. RLHF provides a mechanism for chatbots to seek clarification and guidance from human experts when faced with challenging or unfamiliar user inputs. This helps the chatbot make more informed decisions and provide accurate responses, even in uncertain situations.
4. **Personalization:** Chatbots that can understand and adapt to individual user preferences can deliver a more personalized experience. RLHF allows chatbots to learn from user feedback and tailor their responses based on individual preferences, improving user satisfaction and engagement. This personalization enhances the chatbot’s ability to understand and fulfill user needs.
5. **Continuous Improvement:** Chatbots can benefit from continuous learning and improvement based on ongoing user interactions. RLHF enables chatbots to receive feedback from users in real-time, helping them refine their responses and behavior. This iterative learning process allows chatbots to continually enhance their performance and adapt to evolving user requirements.
6. **Ethical Considerations:** Chatbots that rely solely on pre-existing data may inherit biases or produce inappropriate responses. RLHF provides an opportunity to incorporate human guidance and ensure that the chatbot’s behavior aligns with ethical standards. Human experts can help shape the chatbot’s responses, ensuring they are fair, unbiased, and respectful.

#### Learning Resources: ####

**1. Important Blogs** 

1. [How RLHF actually works by Nathan Lambert:](https://www.interconnects.ai/p/how-rlhf-works) This article will build the intuition behind RLHF and how and why it works simply.
2. [RLHF: Reinforcement Learning from Human Feedback by Chip Huyen:](https://huyenchip.com/2023/05/02/rlhf.html) How exactly does RLHF work? Why does it work? This post will discuss the answers to those questions with a focus on theoretical details.
3. [StackLLaMA: A hands-on guide to train LLaMA with RLHF by HuggingFace:](https://huggingface.co/blog/stackllama) In this blog post, you will be shown all the steps involved in training a LlaMa model to answer

**2. Important Videos & Talks**
1. [Reinforcement Learning from Human Feedback: From Zero to ChatGPT:](https://www.youtube.com/watch?v=2MBJOuVq380) In this talk, speakers will cover the basics of Reinforcement Learning from Human Feedback (RLHF) and how this technology is being used to enable state-of-the-art ML tools like ChatGPT. Most of the talk will be an overview of the interconnected ML models and cover the basics of Natural Language Processing and RL that one needs to understand how RLHF is used on large language models. It will conclude with an open question in RLHF.
2. [State of GPT by Andrej Karpathy:](https://www.youtube.com/watch?v=bZQun8Y4L2A) In this talk, you will learn about the training pipeline of GPT assistants like ChatGPT, from tokenization to pretraining, supervised finetuning, and Reinforcement Learning from Human Feedback (RLHF). In addition to that, you will dive deeper into practical techniques and mental models for the effective use of these models, including prompting strategies, finetuning, the rapidly growing ecosystem of tools, and their future extensions.

**3. Important Research Papers**
1. [Deep reinforcement learning from human preferences by OpenAI:](https://arxiv.org/abs/1706.03741) In this work, the authors explore goals defined in terms of (non-expert) human preferences between pairs of trajectory segments. They show that this approach can effectively solve complex RL tasks without access to the reward function, including Atari games and simulated robot locomotion, while providing feedback on less than one percent of our agent’s interactions with the environment. This reduces the cost of human oversight far enough that it can be practically applied to state-of-the-art RL systems. To demonstrate the flexibility of this approach, they show that we can successfully train complex novel behaviors with about an hour of human time. These behaviors and environments are considerably more complex than any that have been previously learned from human feedback.
2. [Learning to summarize from human feedback by OpenAI:](https://arxiv.org/abs/2009.01325) In this work, the authors show that it is possible to significantly improve summary quality by training a model to optimize for human preferences. They collect a large, high-quality dataset of human comparisons between summaries, train a model to predict the human-preferred summary, and use that model as a reward function to fine-tune a summarization policy using reinforcement learning. They apply the method to a version of the TL;DR dataset of Reddit posts and find that our models significantly outperform both human reference summaries and much larger models fine-tuned with supervised learning alone. The models also transfer to CNN/DM news articles, producing summaries nearly as good as those of a human
3. [Training language models to follow instructions with human feedback by OpenAI:](https://arxiv.org/abs/2203.02155) In this paper, the authors show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning them with human feedback. Starting with labeler-written prompts and prompts submitted through the OpenAI API, they collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. They then collect a dataset of rankings of model outputs, which they use to fine-tune this supervised model further using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on the prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, the results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.


### [6. Best Resources to Build & Understand Vision Language Models](https://open.substack.com/pub/youssefh/p/best-resources-to-build-and-understand?utm_campaign=post-expanded-share&utm_medium=web) ###

Vision-Language Models (VLMs) lie at the intersection of computer vision and natural language processing, enabling systems to understand and generate language grounded in visual context.
These models power a wide range of applications — from image captioning and visual question answering to multimodal search and AI assistants. This article offers a curated guide to learning and building VLMs, exploring key concepts in multimodality, foundational architectures, hands-on coding resources, and advanced topics like retrieval-augmented generation for multimodal inputs.
Whether you’re a beginner trying to grasp the basics or a practitioner looking to deepen your technical understanding, this guide brings together practical and conceptual resources to support your journey into the world of vision-language modeling.


#### [6.1. Multimodality and Large Multimodal Models (LMMs)](https://huyenchip.com/2023/10/10/multimodal.html) ####

<img width="885" height="382" alt="image" src="https://github.com/user-attachments/assets/6d517bd0-2333-4088-a1df-8d7d7768237b" />

Chip Huyen, in this [article](https://huyenchip.com/2023/10/10/multimodal.html), explores the rising field of multimodal AI, where models handle and integrate different types of data like text, images, and audio. She explains why multimodality matters — real-world applications often involve multiple input types, from medical imaging paired with patient records to robotic systems interpreting both vision and commands.
The piece walks through how models like CLIP and Flamingo work, focusing on how they align modalities (like vision and language) using contrastive learning or combination architectures.
It also outlines key research directions, such as expanding to more data types, building instruction-following systems, efficiently training with adapters, and generating outputs across modalities. In short, the article gives a clear, technical yet accessible roadmap of how multimodal models are evolving and why they’re essential for building more capable AI systems.

The article covers the following topics:
- Part 1. Understanding Multimodal
- Part 2. Fundamentals of Multimodal Training
- Part 3. Research Directions for LMMs

#### [6.2. Smol Vision Course](https://github.com/merveenoyan/smol-vision) ####

<img width="969" height="459" alt="image" src="https://github.com/user-attachments/assets/809fb919-067f-4dc9-82c5-64795af943fc" />

The [smol vision course](https://github.com/merveenoyan/smol-vision) is maintained by Merve Noyan and others from Hugging Face. It contains tutorials for shrinking, optimizing, and customizing cutting-edge vision models.

#### [6.3. Coding a Multimodal (Vision) Language Model from scratch in PyTorch](https://www.youtube.com/watch?v=vAmKB7iPkWw) ####

In this [5-hour video](https://www.youtube.com/watch?v=vAmKB7iPkWw), Umar Jamil will be coding the PaliGemma Vision Language Model from scratch while explaining all the concepts behind it:
- Transformer model (Embeddings, Positional Encoding, Multi-Head Attention, Feed Forward Layer, Logits, Softmax)
- Vision Transformer model
- Contrastive learning (CLIP, SigLip)
- Numerical stability of the Softmax and the Cross-Entropy Loss
- Rotary Positional Embedding
- Multi-Head Attention
- Grouped Query Attention
- Normalization layers (Batch, Layer, and RMS)
- KV-Cache (prefilling and token generation)
- Attention masks (causal and non-causal)
- Weight tying
- Top-P Sampling and Temperature

#### [6.4. Awesome Vision-Language Models](https://github.com/jingyi0000/VLM_survey) ####

<img width="1400" height="517" alt="image" src="https://github.com/user-attachments/assets/540809b2-16da-4aaa-9666-165688818152" />

[Awesome Vision-Language Models](https://github.com/jingyi0000/VLM_survey) is a repository of Vision Language Models for Vision Tasks: a Survey, a systematic survey of VLM papers in various visual recognition tasks, including image classification, object detection, semantic segmentation, etc. LLM


#### [6.5. Multimodal RAG](https://github.com/To-Data-Beyond/Multimodal-RAG) ####

<img width="831" height="942" alt="image" src="https://github.com/user-attachments/assets/acb65ddb-c269-4609-b814-5499b96920c3" />

Finally, one of the important applications of vision language models is multimodal RAGs. In this [hands-on course](https://github.com/jingyi0000/VLM_survey), you will learn how to build a multimodal RAG system by breaking down each step and building it up.
This course covers:

- Introduction to Multimodal RAG Applications
- Multimodal RAG Application Architecture
- Multimodal Embeddings
- Processing Videos for Multimodal RAG
- Multimodal Retrieval from Vector Stores
- Large Vision Language Models (LVLMs)
- Multimodal RAG with Multimodal LangChain
- Putting it All Together! Building Multimodal RAG Application

------------------------------


* [How to Stay Updated with LLM Research & Industry News?](https://medium.com/gitconnected/how-to-stay-updated-with-llm-research-industry-news-c1d60e341bad?sk=99998b76402b2555c2bf998dd186ba0c)

# Part III: LLMs In Production # 
* [Best Resoruces to Learn Prompt Engineering](https://levelup.gitconnected.com/5-best-resoruces-to-learn-prompt-engineering-7a0ffb459324?sk=cf149a0c227e5ece367ab1405827498b)
* [Top Resources to Master Vector Databases & Building a Vector Storage](https://levelup.gitconnected.com/6-resources-to-master-vector-databases-building-a-vector-storage-8d94ca1e3897?sk=1447d3e09380b72ded2edd380e354f22)
* [Top Resources to Master RAG: From Basic Level to Advanced](https://pub.towardsai.net/top-resources-to-master-rag-from-basic-level-to-advanced-755a814d9348?sk=0e75d3790fbb62b5592b3858b4376cf8)
* [Top Free Learning Resources to Master LLM Agents](https://levelup.gitconnected.com/top-free-learning-resources-to-master-llm-agents-477fad6fcf9c?sk=b345796e178fa5624c72ad22b83bb2f8)
* [5 Free Tools to Run Large Language Models (LLM) Locally on Your Laptop](https://levelup.gitconnected.com/5-free-tools-to-run-large-language-models-llm-locally-on-your-laptop-9a359f1df506?sk=300f0008936ac81e2220fa4dbdb633bf)
* [Deploying LLMs: Top Learning & Educational Resources to Get Started](https://levelup.gitconnected.com/deploying-llms-top-learning-educational-resources-to-get-started-4db8c6034dc5?sk=5482bff8287673630144390bda4721d4)
* [Getting Started with LLM Inference Optimization: Best Resources](https://youssefh.substack.com/p/getting-started-with-llm-inference)
* [What is LLMOps and How to Get Started With It](https://levelup.gitconnected.com/what-is-llmops-and-how-to-get-started-with-it-04002c25e081?sk=f127154bd466cca6b5aae3ce88774f6e)
* [Securing LLMs: Best Learning & Educational Resources](https://levelup.gitconnected.com/securing-llms-best-learning-educational-resources-b9638c063c92?sk=0ba4ddeb4998f310509e99ec3d3d93b9)

# Part IV: Building Your LLM Portfolio #
* [10 Large Language Models Projects To Build Your Portfolio](https://levelup.gitconnected.com/10-large-language-models-projects-to-build-your-portfolio-d7974569aad4?sk=d59e963806e3ccf5fdcd3b5c0f715f48)
* [10 Guided Large Language Models Projects to Build Your Portfolio](https://levelup.gitconnected.com/10-guided-large-language-models-projects-to-build-your-portfolio-dc9bd79f09c?sk=fa1867433c0285c6f41470fba0d2198f)
