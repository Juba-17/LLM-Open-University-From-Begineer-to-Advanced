# LLM-Roadmap-From-Beginner-to-Advanced
A complete roadmap to master LLMs for absolute beginners to advanced 

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




* [Best Resources On Building Datasets to Trian LLMs](https://levelup.gitconnected.com/best-resources-on-building-datasets-to-trian-llms-f6c6e02fc375?sk=342b0b4f9587b2db3af5fe86c90e519e)
* [Mastering Large Language Model (LLM) Fine-Tuning: Top Learning Resources](https://pub.towardsai.net/mastering-large-language-model-llm-fine-tuning-top-learning-resources-dcef012256fd?sk=54ca7be29591c08bd12b6161b534f859)
* [14 Free Large Language Models Fine-Tuning Notebooks](https://levelup.gitconnected.com/14-free-large-language-models-fine-tuning-notebooks-532055717cb7?sk=ef3212821235db70871d72c86e179b07)
* [Best Resources to Learn & Understand Evaluating LLMs](https://pub.towardsai.net/best-resources-to-learn-understand-evaluating-llms-4610ee5dc5c1?sk=2e53e253dbf2890f0cc94c0cfe7c64c0)
* [Overview of LLM Quantization Techniques & Where to Learn Each of Them?](https://yousefhosni.medium.com/overview-of-llm-quantization-techniques-where-to-learn-each-of-them-0d8599acfec8?sk=594f81f338f15bb211d9356a6537e476)
* [Top Resources to Learn & Understand RLHF & LLM Alignment](https://levelup.gitconnected.com/top-resources-to-learn-understand-rlhf-69f7984f1e58?sk=79d44cc8a12394a958545096643bc583)
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
