# Understanding Multi-Agent Systems: From Language Models to Collaborative AI

## Executive Overview

This chapter answers a deceptively simple question: *when does a single AI model stop being enough, and why?* The answer runs through four ideas that this chapter builds carefully, one on top of the other: (1) generative models are fundamentally next-token predictors, and this fact both explains their power and predicts their failure modes; (2) an **agent** is what you get when you wrap a model in a perception–action loop and give it tools, memory, and the ability to communicate; (3) a **multi-agent system** is what you get when several such agents are orchestrated together, either through a fixed workflow or through emergent, AI-driven coordination; and (4) the decision to move from model → agent → multi-agent system should be driven by specific, identifiable properties of the task — planning depth, diversity of required expertise, context volume, and the need for adaptive, exploratory problem-solving — not by hype or fashion.

Along the way we will build the vocabulary that the rest of a multi-agent systems course depends on: reasoning vs. acting, tools vs. skills, short-term vs. long-term memory, application-managed vs. agent-managed memory, workflow orchestration vs. autonomous orchestration, and the economic argument (time arbitrage) for why any of this is worth doing at all. We close with a fully worked, runnable example — a two-agent "poet and critic" system — that makes every abstract concept concrete in code.

## Learning Objectives

By the end of this chapter you will be able to:

1. Explain, from first principles, why large language models (LLMs) are next-token predictors, and derive from that fact why they hallucinate on queries requiring current information.
2. State the formal definition of an **agent** used throughout the source material (reason, act, communicate, adapt) and map each capability onto a concrete example.
3. Decompose an agent into its three architectural components — **model**, **tools**, **memory** — and explain the role each plays in the agent's perception–action loop.
4. Distinguish **short-term** from **long-term** memory, and **application-managed** from **agent-managed** memory, with a worked example of each.
5. Define a **multi-agent system** and distinguish the two orchestration paradigms — **workflows** (explicit, predetermined control flow) and **autonomous orchestration** (emergent, AI-driven control flow) — including the trade-offs each makes between predictability and adaptability.
6. Identify the four characteristics of a task (planning, diverse expertise, extensive context, adaptive solutions) that motivate a multi-agent architecture, and apply a decision framework to choose between direct model calls, single agents, multi-agent workflows, and autonomous multi-agent systems.
7. Articulate the economic and technical arguments ("why now?") for multi-agent systems: reasoning capability advances, time arbitrage, self-improving systems, tacit-knowledge automation, platform economics, and the accompanying need for reliability and ethics.
8. Read and explain a working Python example that constructs two agents and orchestrates them using round-robin turn-taking with composable termination conditions.

## Prerequisites

Before proceeding, you should be comfortable with the following ideas. Where a concept is unfamiliar, a short primer is given inline so the chapter remains self-contained.

- **Basic Python** (functions, classes, `async`/`await`). If `async`/`await` is new to you: Python's `asyncio` library lets a program pause a function at an `await` point (typically while it is waiting on network I/O, such as a call to an LLM API) and let other code run in the meantime, rather than blocking the entire program. Agent frameworks use this heavily because agents spend most of their wall-clock time waiting on network calls to model providers or external APIs, and an async architecture lets many such waits overlap instead of queuing sequentially.
- **What an API is.** An Application Programming Interface is a contract that lets one piece of software (your code) request a service from another (an LLM provider's servers) over the network, typically by sending structured data (usually JSON) to a URL and receiving structured data back.
- **What a neural network training objective is**, at least informally: a model is shown enormous numbers of examples and its internal parameters are nudged, via gradient-based optimization, to make its predictions match the examples more closely over time.

No prior exposure to agent frameworks, orchestration patterns, or the AutoGen/Microsoft Agent Framework family is assumed — this chapter builds all of that vocabulary from the ground up.

---

## Main Content

### 1. Why This Book Exists: The Practitioner's Vantage Point

The chapter opens not with a definition but with a *lineage* — a personal history that quietly does a lot of argumentative work. The author (Victor Dibia) traces a path from early GitHub Copilot evaluation work at Microsoft Research (2022), through **LIDA**, a four-stage pipeline system for automatic data visualization, to **AutoGen**, an open-source conversational multi-agent framework released in May 2023, and finally to the **Microsoft Agent Framework**, described as AutoGen's production-ready successor.

Why does this matter pedagogically, rather than just biographically? Because it encodes an argument about *why pipelines were not enough*. LIDA was built as a rigid four-stage pipeline: a data summarizer, a visualization-goal generator, a visualization-code generator (with execution, error correction, and retries), and an optional infographics generator. Pipelines like this have a real virtue: **each stage can be tested independently**, which is a direct analogue of unit testing in traditional software engineering — you can verify the data summarizer produces correct summaries without having to also verify that the code generator produces correct charts, because the stages are decoupled. This is why pipelines "encourage reliability."

But this virtue comes at a cost, and the chapter is explicit about what that cost is: a pipeline **assumes the correct task decomposition is already known**, that **the expertise to implement each stage is available**, and that **the task is static and predictable**. Think about what each of these assumptions actually requires of the system's designer:

- *"The correct decomposition is known"* means a human engineer has already solved the hard planning problem in advance, at design time, once, for all future inputs. This works when the task shape doesn't vary — visualization generation always needs a summary, then a goal, then code — but breaks the moment the right sequence of steps depends on the specific input.
- *"The expertise to implement each step is available"* means each pipeline stage can be hand-engineered by someone who understands that stage deeply. This scales linearly with the number of distinct skills the task requires. A task needing many different kinds of expertise (as we'll see in Section 3 with the "build a stock/tax app" example) makes this assumption expensive to satisfy.
- *"The task is static and predictable"* means the environment does not change in ways that invalidate the pre-planned sequence — no failed API calls requiring a different fallback path, no need to backtrack.

This is the conceptual seed of the entire chapter: **multi-agent systems exist because these three assumptions fail for a meaningfully large class of real tasks**, and the natural response to their failure is to build a system that can, at runtime, dynamically decide the decomposition, delegate to specialized entities, and adapt when things go wrong — which is exactly the definition of an agent (and by extension, a multi-agent system) that the chapter builds toward.

> **Connection forward:** This pipeline-vs-agent tension reappears formally in Section 6 (the Decision Framework, Figure 1.8) as the question "is your solution approach well-known and can it be expressed as a workflow?" A workflow *is* a generalized, more flexible pipeline; an autonomous multi-agent system is the response to what happens when even a workflow's fixed structure is too rigid.

**A note on hype vs. reality.** The author includes a short reflective aside worth dwelling on because it establishes the epistemic stance of the whole book: two camps of AI practitioners exist — those who dismiss current limitations as proof AI "doesn't work," and those who treat current limitations as engineering problems to be iterated on. The illustrative anecdote is quantitative and falsifiable: an early LIDA prototype had roughly a 20% visualization-generation error rate; after prompt improvements and a new underlying OpenAI model release, this dropped to roughly 3%. The methodological lesson generalized from this anecdote is: **treat model capability as a moving parameter you can measure, not a fixed verdict.** This matters for how you should read every failure case discussed later in this chapter (e.g., LLMs neglecting the middle of long contexts) — these are *current, measurable* limitations that inform architecture choices today, not permanent laws of computation.

### 2. Book Roadmap and the picoagents Library

The book is organized into four parts following a **theory → build → optimize → apply** progression:

| Part | Focus | Chapters |
|---|---|---|
| I: Foundations | Purely conceptual — agents, multi-agent systems, orchestration patterns, UX principles | 1–3 |
| II: Building from Scratch | Framework-agnostic hands-on implementation | 4–9 |
| III: Evaluation, Optimization, Responsible AI | Testing, failure-mode analysis, protocols, ethics | 10–13 |
| IV: Real-World Applications | End-to-end case studies | 14–15 |

The pedagogical vehicle for Part II is a library the reader builds themselves, called **picoagents**. This is a deliberate teaching choice worth understanding: rather than teaching a specific commercial or open-source framework (which would tie the book's value to that framework's API surface, and would obscure *why* the framework is built the way it is), the book has the reader implement the core abstractions — the agent execution loop, tool calling, memory, workflow graphs, orchestration loops — from first principles. The explicit claim is that this mirrors the architectural patterns found in production frameworks like AutoGen, Microsoft Agent Framework, Google's Agent Development Kit (ADK), and Pydantic AI, so that understanding picoagents transfers to understanding (and evaluating) any of those frameworks. This is analogous to the pedagogical argument for implementing a hash table or a small neural network from scratch before using a production library: the abstraction stops being magic once you've built a minimal version of it yourself.

The book also reports its own scale as a piece of quantified metadata: 15 chapters, 54,940 words, 154 code snippets, 46 figures, 27 tables, 76 callout boxes, 73 references. This is useful context for calibrating how much implementation detail to expect later — this is not a purely conceptual text; roughly 60% of its stated value proposition is systems engineering (architecture, code, deployment, scaling) against 40% theory (models, evaluation, translating business problems into architectures).

### 3. Generative AI: The Substrate Agents Are Built On

Before defining "agent," the chapter pauses to make sure the reader understands *what kind of thing* sits inside an agent's "model" component. This is a deliberate sequencing choice — you cannot understand why agents need tools and memory until you understand what generative models can and cannot do on their own.

#### 3.1 Definition and Intuition

**Generative models** are a class of deep neural networks trained to model the underlying probability distribution of a dataset well enough that they can produce *new* samples resembling that dataset. This stands in contrast to **discriminative models**, whose training objective is to separate or classify existing data points into categories (e.g., "is this email spam or not spam?") rather than to produce new data points.

*Intuition:* imagine you've read every novel published in the English language. A discriminative model, in this analogy, has learned to sort new manuscripts into genres. A generative model has learned the *statistical texture* of English prose well enough that it can write a new, plausible-sounding paragraph of its own — not by retrieving a memorized paragraph, but by sampling from a learned distribution over what word plausibly follows what other words, given context.

**Mental model:** think of a generative language model as an extremely well-calibrated autocomplete system. At every position in a sequence, it outputs a probability distribution over "what token comes next," conditioned on everything that came before. Generating text is the repeated act of sampling from this distribution, appending the sampled token to the sequence, and repeating.

Generative models are further categorized by the *modality* of data they're trained on:

| Model Type | Training Modality | Example Model Families |
|---|---|---|
| Large Language Model (LLM) | Written text | GPT series |
| Image Generation Model (IGM) | Images | DALL-E series |
| Large Multimodal Model (LMM) | Text and images jointly | GPT-4V |

This book restricts its scope to models that *generate text output* — this includes both pure LLMs and multimodal models whose output modality is text, because the agentic reasoning, tool-calling, and orchestration patterns discussed throughout the book are built on top of text (or structured-text, e.g., JSON) generation.

#### 3.2 The Training Objective: Sequence Prediction

**Definition.** The fundamental training objective for an LLM is **sequence prediction**: given a sequence of tokens, predict the next token (or, in masked-language-model variants, fill in a deliberately hidden span within the sequence).

**Why this works — the intuition and the research finding.** This objective sounds almost too simple to produce the sophisticated behaviors observed in modern LLMs, and understanding *why* it works is essential background. If a model is trained on a genuinely enormous and diverse corpus of human-written text, then predicting the next word accurately requires the model to implicitly learn an enormous amount about the world the text describes — grammar, facts, reasoning patterns, social conventions, code syntax, and so on — because all of these regularities are what make some next-token predictions more probable than others in real text. The chapter states this precisely: applying the sequence-prediction objective to a sufficiently large dataset causes the model to **learn representations of the world as depicted through text**, and the model can then leverage these representations to generate coherent, relevant, contextually appropriate continuations.

**How tasks get reframed as sequence prediction.** Early efforts to make LLMs useful for concrete tasks worked by *reformulating* those tasks as next-token-prediction problems, using a template-based prompt structure. Two worked examples from the text:

- **Classification:** construct a prompt of the form `<problem description> <list of labels> Which of the provided labels is the correct class?` — the model's job reduces to predicting which label token(s) come next.
- **Summarization:** construct a prompt of the form `<passage> The summary of the passage above is ..` — the model's job reduces to predicting a continuation that happens to be a summary.

This reframing trick is the reason a *single* next-token-prediction objective can, in one model, perform translation, sentiment analysis, question answering, dialogue generation, named entity recognition, syntax parsing, code synthesis, paraphrasing, and grammar correction — none of these was the literal training objective; all of them are downstream tasks expressible as "predict what comes next" given the right framing. This capability — solving tasks the model was never explicitly trained on, purely via how the input is phrased — is one root of what the field calls **zero-shot** and **few-shot** generalization.

**Why this matters for agent design (connection forward):** this same reframing principle is *exactly* what agent frameworks rely on when they ask a model to output a *tool call* instead of a paragraph of prose, or a *plan* instead of an answer. An "agent" doesn't require a fundamentally different kind of model — it requires a model whose next-token-prediction objective has been pointed, via prompting or fine-tuning, at producing structured actions and reasoning traces rather than freeform prose. This is the conceptual bridge between Section 3 (generative models) and Section 5 (what is an agent) later in this chapter.

**A critical, named limitation.** The chapter is careful to state a specific, important failure mode directly tied to the sequence-prediction origin of these models: their reasoning capabilities (illustrated by the trivial example "2 apples + 10 apples = 12 apples") tend to work well on patterns well-represented in training data, and to fail to generalize to rare or unseen scenarios — the given example is solving linear equations in base 3. **Why does this follow from the training objective?** Because the model's competence at any given kind of reasoning is a function of how much (and how clearly) that reasoning pattern is represented in the statistical structure of its training corpus, not because the model has built a general-purpose symbolic reasoning engine independent of its training distribution. This is not a moral judgment about LLMs — it is a direct, mechanistic consequence of what "trained via next-token prediction over a text corpus" means, and it is the single most important limitation to keep in mind when later sections argue that agents need *tools* (e.g., a calculator, a code executor) to reliably handle tasks outside the well-represented core of the training distribution.

#### 3.3 Prompt Engineering and Alignment

**Prompt engineering** is defined as the practice of constructing input sequences that increase the likelihood a model successfully completes a task. Three named techniques are called out, each worth unpacking because they recur throughout applied LLM work:

- **Few-shot prompting** (Brown et al. 2020): include worked examples of the task-and-solution pattern directly in the prompt, so the model can infer the pattern by analogy before attempting the real query. This works because of the reframing principle above — showing examples effectively re-specifies, in-context, what "coming next" should look like for this task.
- **Chain-of-thought prompting** (J. Wei et al. 2022): include examples where the *solution steps*, not just the final answer, are shown. This nudges the model to generate its own intermediate reasoning steps before committing to a final answer, which empirically improves accuracy on multi-step reasoning tasks — plausibly because it gives the model more relevant tokens to condition on before the hardest prediction (the final answer) is made.
- **ReAct prompting** (Yao et al. 2022): interleaves *reasoning* (thinking in natural language about what to do) with *acting* (taking a concrete tool-using action, observing a result, and reasoning again). ReAct is the direct conceptual ancestor of the agent execution loop this chapter later formalizes as the perception–action loop (Figure 1.6) — you can read ReAct as "the first explicit statement, in the prompting literature, of what an agent's inner loop should look like."

Beyond prompting (which changes model *behavior* without changing model *parameters*), the chapter names two techniques that change the parameters themselves:

- **Fine-tuning on task-specific instruction data**
- **Reinforcement learning from human feedback (RLHF)-style alignment** (Ouyang et al. 2022), where explicitly collected human feedback data is used to further adjust the model

Both are grouped under the umbrella of **alignment** — efforts to make generated outputs match human intentions, preferences, and values, as distinct from merely making outputs statistically fluent.

#### 3.4 The Context Window

**Definition.** The **context window** is the maximum number of tokens (words or sub-word pieces called "tokens" — note token ≠ word; common words may be a single token while rarer words split into multiple sub-word tokens) that a model can process in a single input-plus-output sequence.

**Why it exists (motivation/mechanism).** This limit is not arbitrary policy; it is dictated by the model's architecture, specifically its computational and memory constraints. Transformer-based LLMs (the dominant architecture referenced via the Vaswani et al. 2017 citation in this chapter) compute self-attention over every pair of tokens in the input, which has computational and memory cost that grows with the *square* of the sequence length. This quadratic scaling is why context windows have historically been a hard engineering constraint rather than something arbitrarily extendable — doubling the context length roughly quadruples the compute and memory needed for the attention computation (before accounting for optimizations like FlashAttention, which reduce memory overhead but do not eliminate this fundamental scaling relationship).

**Practical implication.** Because the context window is finite, developers must actively manage what goes into it. The chapter names five mitigation strategies: **truncation** (drop the least relevant content), **sliding window techniques** (keep only the most recent span of content), **summarization** (compress older content into a shorter representation), **chunking** (split large documents into pieces processed separately), and **optimized prompt engineering** (be economical and structured about what is included). Balancing these strategies against computational resources is described as essential to maximizing a model's usefulness in real applications.

> **Connection forward:** this section is the direct technical justification for **memory** as a core agent component (Section 5.4.2.3) and for **selective context provisioning** as a mitigating strategy for the "extensive context" property of complex tasks (Section 6, item 3). It is also the direct justification for *why* a single, overloaded agent instruction set becomes brittle as a task's context requirements grow — eventually you run out of window, or you dilute the model's attention across too much material (see the "lost in the middle" finding, Section 6.3 below).

#### 3.5 Getting Access to Generative AI Models

State-of-the-art models such as GPT-4 or Claude are, practically speaking, accessed as cloud services rather than run locally, because of their computational requirements (these models typically have parameter counts and inference costs that are impractical for individual developers to host) and because their weights are proprietary. Access follows a now-standard developer workflow: create a developer account with a provider (OpenAI, Anthropic, Google, Microsoft Azure among others), configure billing (most providers use usage-based pricing with a free tier), and generate an API key used to authenticate requests.

**A standardization choice with downstream consequences.** The book states that it standardizes on the **OpenAI API specification** as a pragmatic choice, because this specification has become a *de facto* industry standard that multiple providers have converged on supporting, including **Azure OpenAI** (Microsoft's managed enterprise deployment of OpenAI models, offering additional security, compliance, and regional-availability guarantees) as well as third-party services like Together AI and Anyscale. The practical payoff of this convergence is portability: code written against the OpenAI API's request/response shape can typically run against any API-compatible provider with minimal changes — which is exactly the kind of decoupling that lets the multi-agent systems built later in the book remain **provider-agnostic**.

For local development or privacy-sensitive deployments, the chapter names **vLLM** as a tool that enables self-hosting open-source models while preserving OpenAI API compatibility — meaning the same agent code written against a cloud model can, in principle, be pointed at a local model server without modification, because the interface contract (the API shape) is held constant even though the implementation (cloud vs. local) changes. This is an instance of the broader software-engineering principle of **programming to an interface, not an implementation** — a principle multi-agent frameworks depend on heavily, since it lets you swap models, tools, or even entire agents without rewriting the orchestration logic around them.

### 4. Three Levels of Task Complexity

The chapter now grounds all of the abstract material above in a concrete, three-row table that becomes the organizing device for the rest of the chapter.

| Complexity Level | Task Example | What It Requires |
|---|---|---|
| Model-Level | "What is the height of the Eiffel Tower?" | Direct information retrieval from training data |
| Agent-Level | "Tell me the stock price of NVIDIA today." | Current data, planning, and tool use |
| Multi-Agent | "Build a mobile application that helps users view stock prices, buy stock, and file taxes." | Multiple domains of expertise, iterative development |

Let's walk through *why* each row requires what it requires, because the reasoning is not obvious on first read and the chapter builds significant weight on this table later (it is referenced again in Sections 6 and 8).

**Row 1 (Model-Level).** The Eiffel Tower's height is a static fact almost certainly represented, likely many times, in any large text-training corpus. Answering this question requires no planning, no tool use, and no adaptation — it is a case where the sequence-prediction mechanism from Section 3 can directly retrieve a memorized association between "Eiffel Tower height" and "330 meters" (or similar). This is model-level competence: text in, text out, no loop.

**Row 2 (Agent-Level).** "NVIDIA's stock price today" is fundamentally different in kind, not just degree: it is information the model's training data provably cannot contain, because stock prices change every trading day and the model's training data has a fixed cutoff date. If you ask a raw model this question, it faces a forced choice: refuse, or **hallucinate** — generate a plausible-sounding but almost certainly wrong number, because the sequence-prediction mechanism will still produce *some* statistically plausible-looking continuation even when it has no grounding for it. Solving this task correctly requires four things in sequence: **understanding** the query (recognizing "today" implies current, not historical, data), **planning** (recognizing the task decomposes into "fetch current data" then "answer using it"), **action** (actually calling a tool — a web search or a financial-data API — to retrieve the data), and **presentation** (returning the retrieved value formatted as an answer). This four-step requirement is *precisely* the definition of "agent" the chapter is about to state formally in Section 5 — notice the parallel wording (reason, act, ...).

**Row 3 (Multi-Agent).** Building a full mobile application spanning stock viewing, purchasing, and tax filing requires an amount and diversity of expertise (interface design, Android development, financial API integration, tax-filing logic, integration testing) that the chapter argues cannot be reasonably encoded into a single agent's instruction set, *and* requires an **iterative** solution process where the outcome of one action (e.g., "does this SDK support this call?") determines the next action taken — this is qualitatively different from Row 2's single fetch-and-answer loop. This is where **multiple, specialized, collaborating agents** become the natural unit of design.

**An important nuance: the single-vs-multi-agent boundary is blurry.** The chapter explicitly flags that in practice, the line between "one agent" and "a multi-agent system" is not sharp. An agent may itself use *other agents as tools* (a pattern formalized later, in Section 4.11 of Chapter 4 of the book, "Agents as Tools") — meaning what looks like a single agent from the outside may internally be a small multi-agent system. The book's choice to teach *explicit* multi-agent architectures, where the coordination between agents is visible and intentional, is a pedagogical decision made because explicit architectures make the underlying design principles clearest to a learner — but the author is careful to note that the orchestration patterns, evaluation methods, and ethical considerations developed throughout the book apply broadly, regardless of whether a system's architecture is packaged as "one agent" or "many" from a user's perspective.

**Formal definitions introduced at this point**, which will be expanded below:

- **Agent:** An AI system that can reason, act with tools, communicate, and adapt.
- **Multi-Agent System:** Multiple agents working together, each with specialized capabilities.
- **Tools:** External capabilities — APIs, code execution, web search — that extend what an agent can do beyond generating text.

The broader stakes of getting this right are made concrete: when agents understand a user's context (current activity, environment, recent interactions) and preferences (explicit settings and observed behavioral patterns), the same underlying architecture can extend to highly personalized assistance — scheduling meetings, drafting emails, booking flights, making purchases, filing taxes — which is why the rest of the chapter invests so heavily in getting the underlying definitions precise.

### 5. What Is an Agent?

#### 5.1 From Classical AI to Generative AI Agents

The chapter grounds its definition in the classical AI textbook formulation from Russell and Norvig (2020): an agent is **anything that perceives its environment through sensors and acts upon that environment through actuators**, via an **agent function** that maps perceptions to actions.

**Worked classical example: a robotic vacuum cleaner.**

- *Perceives* via sensors: obstacle detectors, dirt-level sensors, battery-level sensors.
- *Reasons* via hardcoded logical rules: `if obstacle detected → turn`; `if dirt detected → increase suction`; `if battery low → return to dock`.
- *Acts* via actuators: drive motors, suction mechanism, brush rotation.

This creates a continuous **perception–action loop**: sense → apply fixed rule → act → sense again. Crucially, in this classical model, the "reasoning" is a small, fixed, deterministic lookup from perceived state to prescribed action — there is no generative model here, no learning-in-context, no natural-language communication.

**What generative AI adds.** The book's move is to keep the same perception–action loop structure but replace (or augment) the fixed rule-based reasoning core with an LLM/LMM, which adds: **complex reasoning** (handling novel situations not explicitly enumerated by a rule table), **dynamic tool use** (choosing *which* action/tool to invoke based on context, rather than following a fixed if-then table), **natural language communication** (understanding and producing free-form language, not just discrete sensor/actuator signals), and **adaptive behavior based on outcomes** (adjusting strategy based on what happened, not just re-running the same fixed rule).

#### 5.2 The Formal Definition Used Throughout the Book

> **Agent:** an entity that can **reason**, **act**, **communicate**, and **adapt** to solve problems.

Each of the four verbs is given a precise, non-overlapping meaning:

| Capability | Definition | NVIDIA-stock-price illustration |
|---|---|---|
| **Reason** | Synthesize new information by applying rules or logic to available context — deductively, inductively, or abductively; may be driven by a generative model, custom code, or both | Understanding that "NVIDIA today" implies a need for *current* market data rather than training-time knowledge |
| **Act** | Take concrete actions that affect the environment or retrieve information — beyond generating text: executing code, calling APIs, searching the web, interacting with external systems | Actually calling a financial-data API or performing a web search |
| **Communicate** | Effectively exchange information with users, other agents, and external systems — understand natural-language input, format appropriate output, know when to ask for clarification | Presenting the retrieved price back to the user in natural language |
| **Adapt** | Modify approach based on feedback, changing conditions, or new information | If the first API call is rate-limited, retry after a delay, switch to a different data source, or adjust the request based on the error message |

**Why these four and not some other list?** Notice that each capability closes a different potential failure mode of a raw model: *reason* addresses the failure of blindly answering without recognizing a knowledge gap; *act* addresses the failure of being unable to do anything beyond emit text; *communicate* addresses the failure of being unable to interface naturally with users or other systems; *adapt* addresses the failure of being brittle to the first error encountered. Together they form a minimal, jointly sufficient set for autonomously completing tasks like the NVIDIA stock-price example that a raw model cannot complete alone.

#### 5.3 The Agent's Perception–Action Loop

Building on the classical robotic-vacuum loop, the chapter formalizes the generative-AI-era version: **take action → perceive the result → adapt based on the outcome**, repeated iteratively until the task resolves. Unlike a raw model, which produces one response and stops, an agent's defining behavioral signature is this *iteration*. On success, the agent produces a natural-language response to the user; on error, it may retry with adjusted parameters or explicitly communicate its limitation rather than silently failing or hallucinating a result.

**Why this matters architecturally (connection forward):** multi-agent systems, introduced formally in Section 7, are explicitly described as *building on* this same principle — multiple agents run **coordinated** perception–action loops. This means everything you learn about a single agent's loop (how it decides when to stop, how it handles failure, how memory feeds into its next decision) is a prerequisite for understanding how *multiple* such loops can be synchronized, which is the subject of Chapter 7 of the book (orchestrator loops, termination conditions).

#### 5.4 The Three Architectural Components

The agent's four capabilities are realized through three components working together, shown conceptually as: **Model → (reasons, selects) → Tools → (act, produce observations) → feeds into → Memory → (informs next) → Model...**

##### 5.4.1 Model — the Reasoning Engine

The model is the component that enables decision-making and planning: it is typically an LLM or LMM that processes context, generates plans, and determines what to do next — described evocatively as the agent's "brain." Critically, the chapter notes that decision-making does not *have* to be driven purely by a generative model — it can also be driven by predefined logic (e.g., "always call this API when a message of this shape arrives") or by explicit human input. The strongest systems, per the text, **intelligently combine multiple drivers**: using a generative model to reason over task state generally, while requesting just-in-time human input at specific decision points to improve quality and adaptability. This foreshadows the "Humans in the Loop" material later in Chapter 4 of the book (Section 4.13) and is an important corrective to the common misconception that "agentic" necessarily means "fully autonomous, no human involvement."

##### 5.4.2 Tools — the Action Mechanism

**Definition.** Tools (also called skills or plugins) are specific pieces of logic that carry out particular tasks; they are the primary mechanism by which an agent *acts* rather than merely *describes*.

Two categories are distinguished:

- **General-purpose tools** provide broad capability. The two named examples are illustrative of just how broad "general-purpose" can be: a **code executor** effectively lets an agent complete *any* task that can be expressed as code (which, given the breadth of what code can express, is an enormous capability surface from a single tool), and a **UI interface driver** lets an agent complete any task expressible as a sequence of UI interactions (this is the direct precursor to the "Computer Use Agents" material in Chapter 5 of the book).
- **Domain-specific tools** are narrower and purpose-built — the example given is calling a weather API with particular parameters.

**Why this taxonomy matters:** the range (diversity) and complexity of tasks an agent can address is directly, causally shaped by which tools it has access to — this is the practical, engineering-facing restatement of the "Act" capability from Section 5.2, and it is the reason tool design is treated as a first-class engineering discipline later in the book (Chapter 15 devotes significant space to "Designing Tools for a Software Engineering Agent").

##### 5.4.3 Memory — Recall and Reuse

**Motivation.** For an agent to perform effectively *and* improve over time, it needs the ability to recall and reuse information from past interactions — this is what memory provides, and it is what separates an agent that merely reacts from one that can be said to "learn from experience," at least at the level of accumulating and reusing successful strategies within or across sessions.

**Temporal distinction:**

- **Short-Term Memory** functions as working memory scoped to the *current* task: recent actions, conversation history, and temporary information needed to complete the immediate objective. In multi-agent systems specifically, short-term memory often includes **shared context**, allowing multiple agents to coordinate effectively — i.e., short-term memory is not necessarily private to one agent; it can be a shared substrate multiple agents read from and write to.
- **Long-Term Memory** stores accumulated knowledge, successful strategies, and learned patterns that persist *across* different tasks and sessions, letting an agent build expertise over time and transfer past experience to new, related situations.

**Control distinction (a second, orthogonal axis):**

- **Application-managed memory:** the developer controls storage and retrieval — information is stored and fetched automatically by application code according to predefined strategies the agent itself has no say over.
- **Agent-managed memory:** the agent itself has direct control — it explicitly decides what to store, when to retrieve it, and how to organize its own knowledge base, typically by using memory-management operations exposed to it as tools.

**Why this distinction is called out as important:** the text explicitly flags that this application-vs-agent-controlled distinction matters most when you are building agents that need to *actively curate their own knowledge* — i.e., cases where a fixed, developer-authored retrieval strategy would be too rigid, and the agent's own judgment about what is worth remembering is itself part of the task competence you want the system to exhibit. This tension — and its trade-offs — is deferred to a dedicated treatment in Section 4.8 of Chapter 4 ("Adding Memory") and Section 4.8 specifically on agent-managed memory.

##### 5.4.4 The Full Cycle

Putting the three components together: the model **reasons** about the current state and decides what action to take; it **selects and uses** an appropriate tool to execute that action; it **observes** the result; it **updates memory** with the new information; and it **adapts** its next step accordingly. This is a more mechanistic restatement of the perception–action loop from Section 5.3, now with each of the three components assigned an explicit role in one iteration of the loop.

**Conversational programming — an implementation pattern worth naming explicitly.** The chapter poses a genuinely practical question at this point: how do you actually *implement* this iterative reason-act-observe cycle in code? The answer given is **conversational programming**: represent each step of the cycle — reasoning, acting, and processing results — as a *message* in a conversation. This is a natural fit because modern LLMs are typically **chat-fine-tuned**, meaning their training specifically optimizes them to process a list of prior messages (a conversation history) and produce a contextually appropriate next message. Framing the agent loop as a growing message list lets an agent seamlessly interleave natural-language reasoning with concrete actions (tool calls, code execution results) within a single, uniform data structure — a list of messages — and this pattern is stated to be the foundation of multi-agent frameworks like AutoGen, where agents coordinate through **message passing**. This is an important implementation detail to internalize now, because Chapter 4 of the book builds the entire `Agent` class around exactly this message-list abstraction.

#### 5.5 Agent vs. Model: The Precise Distinction

It's worth stating the boundary explicitly, because the terms are often used loosely in casual discussion. Both agents and models can process and generate natural language. The distinguishing capability is that **agents can take actions, use tools, maintain context across interactions, and adapt their behavior based on results**, while models, on their own, are limited to text processing, analysis, and generation grounded only in their training data. Returning one final time to the NVIDIA example crystallizes this cleanly: a raw model, asked for today's NVIDIA stock price, will produce a plausible-sounding number that is very likely wrong, because it has no mechanism to recognize the query falls outside its training-time knowledge and no mechanism to act on that recognition; an agent recognizes the gap, invokes a tool to close it, and returns a grounded, accurate answer.

### 6. Why Multiple Agents? The Four Characteristics of Complex Tasks

Even a fully capable single agent — one with excellent reasoning, a rich tool library, and well-designed memory — hits real limits on sufficiently complex tasks. The chapter uses the Task 3 example from Section 4 ("build a mobile app for stocks and taxes") as a running illustration and identifies **four characteristics**, shown together in the book's Figure 1.4, that make a task a strong candidate for a multi-agent (rather than single-agent) architecture. The chapter is explicit that while *any one* of these can raise a task's complexity, it is typically the **combination** of several that creates the strongest case for going multi-agent.

#### 6.1 Planning

Complex tasks often require decomposing the overall objective into a sequence of steps that must each succeed for the overall task to succeed. Formally, given some context, a **plan** prescribes a set of actions to execute in order to reach a target success state. The chapter notes this is a long-studied concept borrowed from the robotics planning literature, signaling that multi-agent system design inherits real, decades-old research lineage rather than being an entirely novel problem invented by the LLM era.

#### 6.2 Diverse Expertise

Decomposing a complex task frequently produces sub-steps that each benefit from *different, specialized* expertise. In the stock/tax app example: requirements analysis, UI design, Android implementation, API integration, and testing/deployment are each meaningfully different skill domains, and each maps naturally onto a dedicated, specialized agent (a UX agent, an Android-development agent, and so on).

**Why this is architecturally valuable, not just conceptually tidy:** specialization enables a **separation-of-concerns** design — an idea directly borrowed from software engineering, where a single monolithic module that tries to do everything is harder to maintain, test, and reason about than a set of narrowly scoped modules with clear responsibilities. A generic, single-agent instruction set trying to be simultaneously an expert UX designer, Android engineer, and API integrator tends to dilute its effectiveness at each individual role — much as a single overly-general function in code tends to become harder to reason about than several small, focused functions. Building agents with narrow, specific directives gives you a clean abstraction for mapping responsibilities to entities, enabling what the text calls **domain-driven design** of agentic applications — a direct borrowing of the software-architecture term "domain-driven design," where system structure mirrors the structure of the problem domain itself.

#### 6.3 Extensive Context

Complex tasks often require assembling and processing large amounts of context — in the app-building example, agents might need to run multiple web searches and solicit human feedback across several turns just to assemble the *initial* requirements, before any implementation work begins. Long instructions or large context volumes are a genuine challenge for single-agent systems for a reason grounded directly in Section 3.4's discussion of the context window: **existing research (Liu et al. 2024) has found that LLMs, even when technically capable of processing long text, tend to attend most strongly to instructions at the very beginning and very end of a long context, while neglecting content in the middle** — a phenomenon often referred to informally as "lost in the middle." This is not merely a capacity limit (the text physically fits in the window) but an *attention-allocation* limit (the model doesn't weigh all of the text it technically processed equally).

The chapter grounds this further in **cognitive load theory** (Sweller 1988) from cognitive science, which posits that human working memory has limited capacity, and that overwhelming it with lengthy or underspecified instructions degrades comprehension and performance. This is offered as an *analogy*, not a claim that LLMs have literal human-like working memory — but it's a useful cross-disciplinary lens: both humans and (empirically) LLMs seem to degrade in performance when asked to track too much simultaneous context, and in both cases the practical fix is the same in spirit — reduce and structure what must be attended to at once.

**The multi-agent mitigation: selective context provisioning.** Rather than handing one agent the entire accumulated context of a complex task, a multi-agent architecture can give each agent *only* the subset of context relevant to its specific sub-task (e.g., only the relevant slice of action history), directly reducing the cognitive-load-analogous burden on any single agent and — per the cited research — likely improving performance because the model's attention is not diluted across irrelevant material.

#### 6.4 Adaptive Solutions

Complex tasks frequently live in **dynamic environments** where the correct solution path is not knowable in advance and can only be discovered by taking actions and observing outcomes. Data sources may be unavailable at a given moment; actions may have unexpected side effects; an API call may fail due to a network issue, a malformed argument, or an unannounced change to the API itself. In each such case, the system's only recourse is to **adapt**: retry with different parameters, seek an alternative resource, or explicitly request human assistance.

The chapter connects this to **metacognition** research — the idea that an agent's awareness and regulation of its own cognitive/reasoning process (noticing when a plan isn't working, and revising it) can significantly improve problem-solving. This capacity is described as *challenging* for single-agent systems today, and multi-agent collaboration is framed as one route toward giving a system this kind of reflective, self-correcting capability at the system level even where any individual agent's metacognitive ability is limited — for instance, a "critic" agent (as in the worked poet/critic example in Section 8) can serve as an external metacognitive check on a "generator" agent's output, effectively distributing metacognition across two agents rather than requiring one agent to reliably self-critique.

**Engineering consequence:** because these adaptive requirements make it hard to write fully deterministic, pre-scripted pipelines (fixed prompts, fixed tool sequences), the natural architectural response is agents capable of **self-orchestration** — deciding, with some autonomy, how to proceed — which is precisely the "autonomous orchestration" paradigm formalized in Section 7.

#### 6.5 A Cross-Domain Comparison Table

The book's Table 1.2 maps four example task domains against all four characteristics, and reading it carefully rewards attention to *how consistently* each domain triggers all four properties:

| Task | Planning | Diverse Expertise | Extensive Context | Adaptive Solutions |
|---|---|---|---|---|
| Web Development | App requirements, design, compliance, APIs | Developers, designers, security experts | Legacy systems, docs, workflows, standards | Security updates, feedback, changes |
| Financial Reporting | Data collection, analysis, reporting | Analysts, statisticians, writers, domain experts | Market data, tools, standards, trends | New sources, feedback, market shifts |
| Tax Filing | Data gathering, analysis, filing, compliance | Tax advisors, analysts, legal, jurisdiction experts | Records, tax codes, regulations, structures | Law changes, queries, corrections |
| Presentation Design | Content review, design, interactivity | Designers, editors, presentation experts | Content, audience, goals, guidelines | Feedback, content updates, format changes |

Notice that *every single row* has a non-trivial entry in *every* column — this is precisely the point the table is making: real-world, economically significant tasks tend to combine all four properties simultaneously, which is why multi-agent architectures are not a niche solution but a broadly applicable one across very different industries.

**Quantitative grounding.** The chapter reinforces this with a real empirical data point from its own later analysis (Chapter 14): of Y Combinator startups building AI agents in 2025 (47.7% of all YC startups, up from 6.1% in 2020 — a 7.8× increase in five years), the top domains are productivity (25.5%), software (18.4%), finance (15.6%), health (11.6%), and e-commerce (6.8%) — domains chosen specifically because they are rich in exactly the planning-and-diverse-expertise properties this section describes.

**Supporting research findings.** Two additional citations reinforce the empirical case for multi-agent collaboration specifically (not just "more compute" or "a bigger single model"):

- Du et al. (2023) find that a **"society of mind"** approach — a term borrowed from Minsky's (1986) theory that human intelligence emerges from the interaction of many simple agents — improves reasoning and factual accuracy when multiple agents *debate* an outcome across several conversational rounds. The mechanism here is plausible: independent agents (or independently-sampled reasoning traces) are less likely to share the *same* error, so cross-checking across them can catch mistakes a single reasoning pass would miss.
- Liang et al. (2023) find that assigning agents genuinely **separate roles** — for instance, some agents perform the task while other, distinct agents adjudicate the quality of that task's output — improves the diversity and quality of generated outcomes, compared to a single agent doing both generation and self-evaluation.

Both findings matter because they suggest the benefit of multiple agents is not merely "more inference-time compute" in an undifferentiated sense, but specifically the benefit of **structured, differentiated roles and independent perspectives** — which is exactly what the poet/critic worked example in Section 8 demonstrates in miniature.

### 7. What Is a Multi-Agent System?

**Formal definition.** A multi-agent system (or multi-agent application) is a collection of agents that collaborate to solve tasks, where each agent maintains its own reasoning, acting, and communicating capabilities and can adapt to changes in the task or environment. The chapter is explicit that the *defining* feature distinguishing one multi-agent system from another is not the agents themselves but their **orchestration mechanism** — the pattern governing how agents communicate, when each one acts, and how they share data and control flow during execution.

**Defining "orchestration" precisely.** Orchestration is the set of mechanisms and patterns enabling multiple agents to work together effectively toward shared goals, and it has two component questions: *how do agents share information* (the communication dimension), and *who controls the flow of execution* — i.e., in what order do agents act (the control dimension). The book deliberately standardizes on "orchestration" over the sometimes-interchangeable term "coordination," for consistency with prevailing industry and AI/ML framework usage.

#### 7.1 Two Orchestration Paradigms

The control-flow question above splits multi-agent systems into two fundamentally different families, visualized in the book's Figure 1.7 as two contrasting diagrams — a predefined workflow (left) versus AI-driven autonomous orchestration (right):

| Dimension | Multi-Agent Workflows (Defined Orchestration) | Autonomous Multi-Agent Orchestration (AI-Driven Orchestration) |
|---|---|---|
| Control flow | Predetermined at design time; explicitly programmed | Determined dynamically at runtime by the AI system itself |
| Agent roles/handoffs | Clearly specified in advance | Negotiated and adapted dynamically based on task requirements and intermediate results |
| Predictability | High — processes are predictable and repeatable | Lower — outcomes can be unpredictable and harder to reproduce across model versions |
| Best suited for | Tasks where the collaboration pattern is well-understood in advance | Tasks where the optimal strategy cannot be predetermined and must be discovered through exploration |
| Example | A document-processing pipeline with agents specialized for extraction, analysis, and formatting, run in a fixed sequence | A system where agents dynamically decide, at runtime, which of them should act next and how |
| Costs | Lower error surface, but less adaptable to novelty | Greater adaptability and potential for innovation, but higher error potential, and added cost/latency from more autonomous back-and-forth |

**Why both approaches use the same underlying "agent anatomy."** It's worth being precise here: the chapter states both paradigms use agents built from the same three components (model, memory, tools) introduced in Section 5.4 — the difference between the two paradigms is *entirely* about how orchestration decisions get made, not about what an individual agent looks like internally. This is an important conceptual economy: you do not need a different kind of agent to build a workflow versus an autonomous system; you need a different *orchestrator* sitting above a set of otherwise similarly-constructed agents.

The chapter explicitly defers the detailed catalog of specific patterns within each family to Chapter 2 — workflows subsuming **sequential** and **supervisor** patterns, and autonomous orchestration subsuming **group chat** and **handoff** patterns — but the conceptual split (predetermined vs. emergent control) established here is the organizing axis all of those specific patterns will be sorted along.

**A worked illustration: booking flights on an airline with no API.** The chapter provides an extended example designed to make the *necessity* of adaptive, multi-agent behavior concrete rather than abstract: booking flights on `specialairlines.com`, which offers only a web/mobile interface, no programmatic API. This single scenario is shown to require three distinct challenges simultaneously:

1. **Interface navigation** — understanding interface content (HTML structure, visual layout), deciding appropriate actions (click, fill a form field), and verifying success (detecting a confirmation message) — and critically, each action *changes* the interface state, which constrains what actions are subsequently possible. This is a state-machine-like problem, not a single-shot query.
2. **Dynamic adaptation** — handling interface changes (a button that moved), and recovering from errors (a failed booking, a network drop) through iterative problem-solving rather than a single fixed script.
3. **Multi-agent orchestration** — the natural decomposition here is a navigation agent (drives the interface), a payment agent (handles the transaction), and a monitoring agent (verifies success), potentially also communicating with a human user when it needs help.

This scenario is deliberately chosen because *none* of its three challenges can be resolved by a fixed, pre-scripted pipeline — the correct solution genuinely emerges only through a sequence of adaptive actions, which is exactly what motivates treating this as a multi-agent, autonomous-orchestration problem rather than a workflow.

#### 7.2 A Note on Distributed Agents

The book distinguishes two different mental models people bring to the phrase "multi-agent systems," depending on their background: some picture software entities collaborating within a single application/process/thread, sharing a common permission structure; others picture a distributed, internet-scale "society" of agents that are largely unaware of each other, requiring a discovery protocol, operating under markedly different security/permission boundaries, and needing to support both synchronous and asynchronous communication at scale.

The book's stated focus is primarily the **first** scenario (single process/thread, shared permissions), on the pragmatic grounds that an early focus on distributed agents is frequently **overengineering** — introducing complexity that most use cases do not yet need. The author draws on direct engineering experience here: in AutoGen's version 0.4 rewrite, the team introduced a *runtime* abstraction that could operate either as single-process or fully distributed, letting developers write agent logic once and deploy it in either mode without changing that logic — and found empirically that the vast majority of real use cases were well-served by the simpler single-process/thread mode. This is a specific, hard-won engineering lesson worth internalizing: **default to the simplest orchestration substrate that solves your problem, and only reach for distributed infrastructure when a concrete requirement demands it** — the general "choose the simplest architecture that works" principle recurs again, more formally, in Section 8 below. The book defers the deeper treatment of distributed agent-to-agent protocols to Chapter 12.

### 8. Choosing the Right AI Agent Architecture: A Decision Framework

Having established four architecture tiers — **direct model calls**, **single agents**, **multi-agent workflows**, and **autonomous multi-agent systems** — the chapter closes the conceptual portion with an explicit decision procedure (visualized as a flowchart in the book's Figure 1.8), intended to prevent over-engineering by defaulting toward the simplest sufficient architecture.

**The decision sequence, step by step:**

1. **Does the task require actions beyond text generation?** If the task is purely text processing/analysis/generation — answering from training-data knowledge, summarizing a provided document, generating code directly from a specification — a **direct model call** is sufficient. No agent machinery is needed.
2. **If action-taking is required: is the solution approach well-known and expressible as a workflow?** If yes, you're choosing between a single agent and a multi-agent *workflow*.
   - **If the workflow benefits from genuinely specialized expertise per step** (i.e., different steps require categorically different knowledge, as in the financial-analysis example of separate market-data, risk-assessment, and portfolio-optimization agents run in a defined sequence) → **multi-agent workflow**.
   - **If one type of expertise can handle the whole task** → a **single agent** with the necessary tools is more efficient; adding multiple agents here would be needless overhead.
3. **If the solution approach is not well-defined:** you're choosing between a single agent and an **autonomous multi-agent system**, based on how much exploration the task demands.
   - **Known action sequences with relatively simple perception-action loops** (e.g., an API call plus result validation) → a **single agent** suffices.
   - **Solutions that must emerge through exploration and dynamic interaction**, where agents must be independently improvable, must adapt collaboration based on intermediate results, or must improve iteratively from successes and failures (the example given is application development where requirements and implementation approaches evolve through experimentation) → **autonomous multi-agent system**.

**The guiding principle, stated explicitly:** choose the *simplest* architecture that effectively addresses your requirements, because each step up this ladder introduces additional complexity, development time, and potential failure points, in exchange for greater capability and adaptability. When genuinely uncertain, the recommended default is to start simple and evolve the architecture only as the task's true requirements become clearer through experience — an explicit endorsement of iterative, empirically-driven architecture decisions over up-front over-engineering.

**This isn't just a design heuristic — it's backed by the book's own later evaluation data** (previewed here, detailed in Section 10.6 of the full book): comparing direct model calls, tool-enabled single agents, and multi-agent systems head-to-head across diverse task types, direct model calls score 9.7/10 on simple reasoning tasks while being roughly **24× more token-efficient** than a multi-agent approach on the same task — meaning applying multi-agent machinery to a task that didn't need it is not merely unnecessary, it is measurably wasteful. Conversely, on tasks genuinely requiring tool coordination — the example given is research tasks requiring web search — direct models score only 3.2/10 versus 9.0/10 for tool-enabled agents, showing exactly where the added architectural complexity earns its cost. The lesson generalizes cleanly: **architectural sophistication should be matched to task requirements, verified empirically, not assumed by default.**

### 9. Why Now? The Confluence of Enabling Factors

The chapter closes its argument for why multi-agent systems merit serious attention *at this moment specifically* (rather than being a purely timeless architectural pattern) by naming seven concurrent forces:

1. **Advances in generative AI reasoning capabilities.** Multi-agent system *concepts* are old — the chapter notes decades of prior research on both human collective intelligence/crowdsourcing and artificial multi-agent coordination (robot planning, robot navigation, human-robot collaboration, swarm intelligence). What was historically missing was a sufficiently capable **reasoning engine** — an "artificial brain" able to adapt to context, synthesize plans, and drive the actions those plans require. Recent generations of models (the chapter names GPT-3.5 and GPT-4 as the pivotal examples) supply, for the first time at meaningful scale and reliability, this missing reasoning component — which is why practical, general-purpose autonomous multi-agent systems are newly viable rather than a decades-old solved problem finally being applied.

2. **Economic value through time arbitrage.** This is a genuinely economic (not purely technical) argument, and worth taking seriously as such: even though an agent may take *longer in wall-clock time* than a human to complete a task, the economics can still favor delegating to the agent, because the *value of human attention* is high while the *marginal cost of agent inference* is comparatively low. The chapter's illustrative figures: a task costing a human roughly $30–100 of their time (one hour, valued at typical knowledge-worker hourly rates) might cost an agent only $1–2 to complete, even if it takes 2–3 hours of (unattended, non-human) wall-clock time. As inference costs continue to fall while human time continues to become relatively more valuable, this arbitrage only widens — this is the economic mechanism that makes "slower but much cheaper and unattended" a genuinely rational trade in many settings.

3. **Self-improving systems.** Unlike traditional software, which requires direct engineering effort to improve, systems built on top of AI models can automatically get better as the underlying model improves — with **no additional engineering investment required on the system builder's part**. The chapter offers a concrete, first-hand data point as evidence: LIDA's initial version, built on the OpenAI davinci model family, had a 20% error rate on its evaluation harness; simply swapping in GPT-3.5-turbo when it became available months later dropped that error rate to roughly 3%, with *no other change to the system*. This "free lunch" property — where your system's competence rides on the trajectory of an externally-improving component — is a genuinely distinctive economic feature of building on top of foundation models, compared to traditional software.

4. **Opportunity to address tacit-knowledge tasks.** Many real-world tasks have historically resisted automation not because they're conceptually hard to *specify*, but because they depend on **tacit knowledge** — knowledge that is difficult to fully articulate or codify into explicit rules, and that is more readily transferred through experience or practice (the classic example type: "you'll know it when you see it" judgment calls). Because LLM-based agents encode vast amounts of implicit, pattern-based knowledge absorbed from training data — rather than requiring every rule to be hand-specified — groups of such agents, collaborating with humans and each other, open a genuinely new path to automating tasks previously considered infeasible to automate, with the practical benefit of reduced human effort and (potentially) reduced error.

5. **Increasing demand for reliable automation.** Businesses and individuals want AI that can autonomously handle sophisticated tasks while maintaining strong reliability and safety — with the explicit caveat that humans should remain in an oversight role to ensure automated outcomes reliably and safely deliver the intended cost and effort savings. Multi-agent systems are positioned as offering the reliability-plus-adaptability combination this demand requires.

6. **Platform economics and integration value.** Multi-agent systems can enable what the text calls **"everything app" economics** — value created through *integration* across many task types within one system, rather than through many narrow, specialized point solutions. Concretely: the same underlying system handling diverse tasks without per-task reconfiguration can capture user preferences and learn across domains, reducing switching costs for users and compounding in value over time — a dynamic familiar from platform-economics literature applied to a new substrate.

7. **Ethical AI deployment.** As deployment of these systems broadens across domains, the need for responsible, secure, human-centric frameworks grows correspondingly — this point is explicitly positioned as necessary infrastructure for trust, not an optional add-on, and it is the direct forward pointer to the dedicated ethics treatment in Chapter 13 ("Ethics and Responsible AI for Multi-Agent Systems").

---

## Algorithm Walkthrough: The Round-Robin Orchestration Loop

The worked example in Section 8 of the source material (poet + critic) instantiates a specific, simple orchestration algorithm worth walking through formally, since it is the reader's first hands-on contact with orchestration mechanics.

**Objective.** Run two (or more) agents in a fixed, cyclic turn order on a shared conversation, until a termination condition is satisfied.

**Inputs.**
- A list of agents (here: `[poet, critic]`)
- A termination condition (here: a **composed** condition — stop after a maximum of 8 messages, *or* stop when the text "APPROVED" appears — combined with the `|` operator, which reads naturally as logical OR)
- A maximum number of orchestration iterations (here: 4, as a hard safety bound independent of the termination condition)
- An initial task/prompt (here: "Write a haiku about cherry blossoms in spring")

**Per-iteration steps:**
1. Select the next agent in cyclic order (poet, then critic, then poet, then critic, ...).
2. Pass that agent the full shared conversation history accumulated so far.
3. Let the agent produce its next message (this internally invokes that agent's own reason → act → observe cycle from Section 5.3, possibly trivially — here neither agent uses external tools, so this reduces to a single model call each).
4. Append the agent's output message to the shared conversation.
5. Check the termination condition against the updated conversation. If satisfied, stop; otherwise, continue to the next agent in the cycle.

**Why compose two termination conditions with OR, specifically.** `MaxMessageTermination(max_messages=8)` is a *safety* condition — it guarantees the loop cannot run forever even if the critic never approves, which matters because autonomous-ish loops (even simple round-robin ones) can otherwise run indefinitely against a badly-specified success criterion. `TextMentionTermination(text="APPROVED")` is the *substantive* success condition — it lets the loop end as soon as the actual goal (an approved haiku) is reached, rather than always running to the message cap. Combining them with OR gives you the best of both: end early on genuine success, but never run unboundedly on failure. This pattern — a substantive success condition paired defensively with a hard safety bound — recurs throughout the more sophisticated orchestration patterns built later in the book (Chapter 7), and is worth recognizing as a general design pattern for *any* agentic loop, not just round-robin orchestration specifically.

**Observed run trace (annotated).** The example transcript shows: the user's task message, then the poet's haiku ("Petals drift like dreams, / Whispers of a fleeting past, / Spring's soft blush unfolds."), then the critic's response, which both offers a specific, actionable suggestion (about strengthening imagery) *and* includes the literal text "APPROVED" — satisfying the `TextMentionTermination` condition after only two agent turns (well under the 8-message cap), so the loop halts early. The reported usage statistics (duration 3.6s, 232 input tokens, 121 output tokens, 2 model calls) make explicit that orchestration cost is measurable and attributable per run — a detail that matters directly for the cost-awareness UX principle referenced elsewhere in the book (Chapter 3) and for the token-efficiency comparisons cited in Section 8 above.

## Code Walkthrough: Building the Poet–Critic System

```python
# Environment setup (run once)
python -m venv multiagent_env
source multiagent_env/bin/activate     # Windows: multiagent_env\Scripts\activate
pip install picoagents
```
A virtual environment isolates this project's package versions from other Python projects on the same machine, avoiding version-conflict bugs; `picoagents` is the teaching library the book builds and uses throughout.

```python
import asyncio
from picoagents import Agent
from picoagents.llm import OpenAIChatCompletionClient
import os
```
- `asyncio` is Python's standard async-programming library, needed because agent operations (model calls) are I/O-bound and the `Agent` API is `async`-first (recall the prerequisites discussion: `await`-ing a model call lets the program yield control while waiting on the network round-trip rather than blocking).
- `Agent` is the core class representing one agent — bundling model, tools, memory, and instructions.
- `OpenAIChatCompletionClient` is a concrete implementation of picoagents' model-client interface, targeting OpenAI-compatible chat completion APIs (recall Section 3.5: this interface is designed so any OpenAI-API-compatible provider, including Azure OpenAI or a self-hosted vLLM server, can be substituted without changing the `Agent` code).

```python
client = OpenAIChatCompletionClient(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)
```
This instantiates a model client bound to a specific model name and an API key read from an environment variable (the recommended pattern, since hardcoding secrets directly in source is a security anti-pattern — an accidentally-committed API key in source control is a common, costly mistake). This single `client` object is shared by both agents below — each agent doesn't need its own separately-configured client.

```python
poet = Agent(
    name="poet",
    description="Haiku poet.",
    instructions="You are a haiku poet.",
    model_client=client
)
```
Constructing an `Agent` requires, at minimum: a `name` (used to identify this agent's turns in a shared conversation, and, in multi-agent settings, potentially used by orchestration logic to route messages), a `description` (a short summary of the agent's role — useful both for humans reading the code and, in more advanced patterns not shown here, for *other agents* deciding whether to delegate to this one), `instructions` (the system-level directive shaping the agent's behavior — here, deliberately minimal, since haiku generation is a well-represented pattern in the model's training distribution and needs little steering), and the `model_client` constructed above.

```python
async def test_poet():
    response = await poet.run("Write a haiku about cherry blossoms in spring")
    print(f"Poet says: {response}")

asyncio.run(test_poet())
```
`poet.run(...)` is an `async` method — invoking it requires `await` inside an `async def` function, and `asyncio.run(...)` is the standard way to execute an async function from otherwise-synchronous top-level script code. `run()` specifically **waits for the complete response** before returning (as opposed to `run_stream()`, mentioned as an alternative for real-time, token-by-token or step-by-step updates — explored in depth in Chapter 4). The printed trace shows a timestamped `[user]` message followed by a timestamped `[poet]` message and a usage-statistics summary line (duration, input/output token counts, finish reason) — this usage reporting is a recurring pattern throughout the book's examples, reflecting the broader engineering principle that agent systems should be observable by default, not just functional.

```python
critic = Agent(
    name="critic",
    description="Poetry critic who provides constructive feedback on haikus.",
    instructions="You are a haiku critic. \
        When you see a haiku, provide 2-3 specific, actionable \
        suggestions for improvement. Be constructive and brief. \
        If satisfied with the haiku, respond with 'APPROVED'",
    model_client=client
)
```
Notice the critic's `instructions` do real structural work beyond describing a persona: they (a) bound the *number* of suggestions (2–3, avoiding an unbounded, rambling critique), (b) set a *tone* constraint ("constructive and brief"), and (c) — crucially for the orchestration logic — specify an **exact termination signal** ("respond with 'APPROVED'"). This is a direct, load-bearing link back to the `TextMentionTermination(text="APPROVED")` condition constructed below: the orchestration's stopping behavior is only reliable *because* the critic's instructions were explicitly engineered to emit that exact string when its actual judgment condition (satisfaction with the haiku) is met. This is a good general lesson in agent design: a termination condition that depends on matching model output text is only as reliable as the prompt engineering that makes the model reliably emit that text under the intended condition — a fragile link if the instruction wording is vague, but effective here because it's explicit and unambiguous.

```python
from picoagents.orchestration import RoundRobinOrchestrator
from picoagents.termination import MaxMessageTermination, TextMentionTermination

termination = (MaxMessageTermination(max_messages=8) |
               TextMentionTermination(text="APPROVED"))

orchestrator = RoundRobinOrchestrator(
    agents=[poet, critic],
    termination=termination,
    max_iterations=4
)
```
This constructs the orchestrator described in the Algorithm Walkthrough above. The `|` operator overload producing a composed termination condition is a small but instructive API-design detail: it lets termination logic be built compositionally (AND/OR-style) from small, independently-testable condition objects, rather than requiring a single monolithic `should_stop()` function to encode all stopping logic — a design that scales much better as termination requirements grow more sophisticated in later chapters (e.g., adding cost-based or time-based termination conditions without modifying existing ones).

```python
async def run_orchestration():
    task = "Write a haiku about cherry blossoms in spring"
    stream = orchestrator.run_stream(task)
    async for message in stream:
        print(f"{message}")

asyncio.run(run_orchestration())
```
Here the orchestrator's `run_stream()` (note: `run_stream`, not `run`, unlike the single-agent test above) returns an **async generator** — an object you iterate over with `async for`, receiving each new message (poet's haiku, critic's feedback) as soon as it's produced, rather than waiting for the entire multi-turn exchange to finish before seeing any output. This streaming pattern is what powers real-time UIs (Section 8's Web UI screenshot, and the full treatment in Chapter 8) — the same underlying data (a growing message list) can be either awaited-in-full (`run()`) or consumed incrementally (`run_stream()`), and the choice depends on whether the calling context is a batch script or an interactive interface.

Finally, wrapping the orchestrator with `serve(entities=[orchestrator], port=8070, auto_open=True)` from `picoagents.webui` launches a local web server providing a browser-based interface for watching the conversation unfold, inspecting individual messages and tool calls, and monitoring termination conditions in real time — turning an otherwise console-only interaction into a debuggable, visual experience, with the full construction of such interfaces deferred to Chapter 8.

---

## Engineering Notes

- **Provider portability is a design goal, not an accident.** Standardizing on the OpenAI API specification, and building `OpenAIChatCompletionClient` as a swappable model-client abstraction, is what lets the same agent code run against OpenAI, Azure OpenAI, other compatible providers, or a locally-hosted vLLM server. When designing your own agent systems, keep the model-client interface narrow and provider-agnostic from the start — retrofitting this later, once agent logic has become entangled with a specific provider's API quirks, is significantly more costly.
- **Observability should be a default, not an afterthought.** Every example trace in this chapter includes duration, token counts, and finish reasons alongside the conversational content. Treat per-call and per-orchestration usage/cost reporting as a baseline requirement for any agent system you build, not an optional debugging feature added later — this is foreshadowed here and formalized in the book's OpenTelemetry treatment (Section 4.10).
- **Termination conditions that depend on model-generated text (like matching "APPROVED") are only as robust as the prompt that produces that text.** Be explicit and unambiguous in agent instructions about exactly when and how such signal text should be emitted, and consider what happens if the model never emits it (this is precisely why the max-message safety bound exists alongside the substantive condition).
- **Match architectural complexity to measured task requirements.** The chapter's own cited evaluation data (9.7/10 accuracy at 24× better token efficiency for direct model calls on simple tasks, versus 3.2/10 vs. 9.0/10 for direct-model vs. tool-enabled agents on search-dependent tasks) is a reminder to validate architecture choices empirically per task type, rather than defaulting to the most sophisticated available pattern.

## Common Mistakes

- **Treating "the model got the current stock price wrong" as a model-quality problem, rather than an architecture problem.** Per Section 3.2's discussion of the sequence-prediction objective, this failure is *structural* — no amount of prompting a raw model will reliably fix a knowledge gap that is, by construction, absent from its training data. The fix is architectural (add a tool), not a better prompt alone.
- **Assuming multi-agent architectures are strictly "more capable" than single agents or direct model calls, and therefore always preferable.** Section 8's decision framework and its supporting evaluation data directly contradict this — multi-agent systems introduce real costs (unpredictability, reproducibility challenges across model versions, more error surface, added latency and token cost) and are only justified when the task actually exhibits the complexity characteristics from Section 6.
- **Conflating "agent-managed memory" with "the agent remembers everything automatically."** Agent-managed memory specifically means the agent *itself* decides what to store and retrieve, typically via explicit memory-operation tools — it is not a synonym for automatic, invisible persistence, which is closer to what application-managed memory provides.
- **Assuming autonomous orchestration is required whenever a task involves "multiple agents."** Per Section 7.1, many legitimate multi-agent systems use fully predetermined workflow orchestration; "multiple agents" and "autonomous/emergent orchestration" are independent design axes, not synonyms.

## Best Practices

- Start with the Section 8 decision framework explicitly, in writing, before choosing an architecture for a new task — resist defaulting to whatever pattern is currently most discussed in the community.
- Design tool access deliberately around the specific reasoning gaps your task exposes (per Section 3.2's discussion of where raw-model reasoning is weakest: current information, precise arithmetic/logic outside common patterns, and anything requiring real-world side effects).
- When context volume is a concern, prefer **selective context provisioning** (Section 6.3) — give each agent only what it needs — over attempting to fit all accumulated context into a single agent's window.
- Pair every substantive, condition-based termination rule with a hard safety-bound termination rule (Section 8's algorithm walkthrough), combined with logical OR, in any agentic loop you build.
- Keep model-client and tool interfaces provider-agnostic from the outset (Section 3.5) to preserve the ability to swap providers or move between cloud and local deployment later.

## Summary

This chapter established the conceptual foundation for the entire book by tracing a single throughline: generative models are powerful but fundamentally limited next-token predictors, grounded only in training data and bounded by a finite context window; wrapping such a model in a perception–action loop with tools (for acting beyond text) and memory (for retaining and reusing information) produces an **agent**, defined precisely as an entity that can reason, act, communicate, and adapt; and orchestrating multiple such agents — either through predetermined **workflows** or emergent, AI-driven **autonomous orchestration** — produces a **multi-agent system**, justified specifically when a task exhibits planning depth, diverse expertise requirements, extensive context, and/or the need for adaptive, exploratory solutions. The chapter closed by grounding the "why now" of this whole endeavor in concrete forces (reasoning-capability advances, time arbitrage economics, self-improving systems, tacit-knowledge automation, reliability demand, platform economics, and the need for responsible deployment), and made every abstraction concrete through a fully worked, runnable two-agent example using round-robin orchestration with composable termination conditions.

## Key Takeaways

- Generative models are next-token predictors; this single fact explains both their surprising task generality (via prompt reframing) and their core limitation (hallucination on out-of-distribution/current-information queries).
- An **agent** = model (reasoning) + tools (acting) + memory (recall/reuse), operating in an iterative perception–action loop, and defined by four capabilities: reason, act, communicate, adapt.
- Memory has two independent axes: **temporal** (short-term working memory vs. long-term persistent memory) and **control** (application-managed vs. agent-managed).
- A **multi-agent system** is distinguished from a collection of agents by its **orchestration mechanism**, which resolves along two questions: how information is shared, and who controls execution order.
- Two orchestration paradigms exist: **workflows** (predetermined, predictable, lower adaptability) and **autonomous orchestration** (emergent, adaptable, less predictable and reproducible).
- Four task properties justify multi-agent architectures: **planning**, **diverse expertise**, **extensive context**, **adaptive solutions** — usually in combination, not isolation.
- Architecture selection should follow a decision framework moving from direct model call → single agent → multi-agent workflow → autonomous multi-agent system, always preferring the simplest sufficient option, validated empirically where possible.
- Multiple economic and technical forces converge to make this the right moment for multi-agent system investment: reasoning-capability advances, time arbitrage, self-improving systems riding on model improvements, tacit-knowledge automation, reliability demand, platform economics, and ethical-deployment needs.

## Concept Relationships

```
Generative Model (next-token prediction)
        │
        ├─ limited by ──> Context Window (quadratic attention cost)
        ├─ fails on ─────> out-of-distribution / current-info queries (hallucination)
        │
        ▼
     Agent  =  Model (reason) + Tools (act) + Memory (recall/reuse)
        │                                         │
        │                                    ┌────┴────┐
        │                              short-term   long-term
        │                              application-managed / agent-managed
        │
        ├─ operates via ──> Perception–Action Loop
        │
        ▼
  Multi-Agent System = {Agent₁, Agent₂, ...} + Orchestration
        │
        ├── Workflows (predetermined control flow)
        └── Autonomous Orchestration (emergent, AI-driven control flow)
                │
                └─ justified by task properties:
                       Planning · Diverse Expertise · Extensive Context · Adaptive Solutions
                │
                └─ selected via: Decision Framework (simplest-sufficient-architecture principle)
```

## Glossary

- **Agent** — An AI system that can reason, act (via tools), communicate, and adapt to solve problems.
- **Agent-managed memory** — Memory whose storage/retrieval decisions are made by the agent itself, via explicit tools, rather than by fixed developer-written logic.
- **Alignment** — Efforts (via fine-tuning, RLHF, etc.) to make model outputs match human intentions, preferences, and values.
- **Application-managed memory** — Memory whose storage/retrieval is controlled entirely by developer-written application code.
- **Autonomous multi-agent orchestration** — Orchestration where control flow and agent collaboration are determined dynamically at runtime by AI-driven decisions, rather than pre-programmed.
- **Chain-of-thought prompting** — A prompting technique that includes worked intermediate reasoning steps to improve multi-step task accuracy.
- **Context window** — The maximum number of tokens a model can process in a single input-plus-output sequence, bounded by architecture-driven compute/memory constraints.
- **Conversational programming** — Implementing an agent's reason-act-observe cycle as a growing list of conversation messages, exploiting chat-fine-tuned models' native message-processing design.
- **Discriminative model** — A model trained to classify or separate existing data points, as opposed to generating new ones.
- **Few-shot prompting** — Including worked task examples directly in a prompt to guide model behavior by analogy.
- **Generative model** — A model trained to learn a data distribution well enough to produce new, similar samples.
- **Hallucination** — A model generating plausible-sounding but factually incorrect output, typically because the query falls outside its training-time knowledge.
- **Long-term memory** — Memory that persists across tasks and sessions, accumulating knowledge and strategies over time.
- **Multi-agent system** — A collection of agents, each with specialized capabilities, collaborating to solve tasks, distinguished by their orchestration mechanism.
- **Orchestration** — The mechanisms and patterns governing how agents share information and who controls execution order/flow.
- **Perception–action loop** — The iterative cycle of taking an action, perceiving the result, and adapting, that defines agentic (as opposed to single-shot) behavior.
- **ReAct prompting** — A prompting approach interleaving explicit reasoning steps with concrete tool-using actions.
- **Sequence prediction** — The core LLM training objective: predict the next token (or fill masked spans) given surrounding context.
- **Short-term memory** — Working memory scoped to the current task: recent actions, conversation history, temporary state; may be shared across agents in multi-agent settings.
- **Society of mind** — Minsky's theory that intelligence emerges from interacting simple agents; used to motivate multi-agent debate/collaboration improving reasoning quality.
- **Tacit knowledge** — Knowledge difficult to explicitly codify into rules, typically transferred through experience rather than instruction.
- **Time arbitrage** — The economic argument that delegating tasks to (slower but far cheaper) agents can be rational because the marginal cost of agent time is much lower than the value of human attention.
- **Tools** — External capabilities (APIs, code execution, web search, etc.) that let an agent act beyond text generation; categorized as general-purpose or domain-specific.
- **Workflow (multi-agent)** — A multi-agent orchestration pattern with predetermined, explicitly programmed control flow and handoffs.

## Self-Assessment Questions

**Conceptual**

1. Explain, from the sequence-prediction training objective, *why* an LLM asked for today's NVIDIA stock price is likely to hallucinate rather than correctly refuse to answer.
2. A single agent has excellent tools and a well-designed model, but is asked to build a full mobile application spanning UI design, backend integration, and tax-compliance logic. Using the four characteristics from Section 6, explain specifically why this task is a strong candidate for a multi-agent (not single-agent) architecture.
3. Distinguish short-term from long-term memory, and separately distinguish application-managed from agent-managed memory. Give an original example (not from the chapter) illustrating a case where agent-managed memory would be preferable to application-managed memory.
4. Explain why "workflows" and "autonomous orchestration" differ specifically in *where* the orchestration decision is made (design time vs. runtime), and why this single difference produces the predictability-vs-adaptability trade-off described in Section 7.1.
5. Using the Section 8 decision framework, walk through which architecture tier (direct model call, single agent, multi-agent workflow, autonomous multi-agent system) you would choose for: (a) drafting a cover letter from a resume, (b) monitoring five different APIs and alerting on the first failure, (c) autonomously researching and writing a competitive analysis report on an unfamiliar market. Justify each choice.

**Implementation**

6. In the poet/critic example, what specific role does the `TextMentionTermination(text="APPROVED")` condition play, and why is it composed with `MaxMessageTermination` using logical OR rather than AND?
7. Why does `critic`'s `instructions` string explicitly specify the exact text "APPROVED" rather than more general language like "let the poet know when you're satisfied"? What would likely break if this instruction were left vague?
8. What is the practical difference between calling `poet.run(...)` and `orchestrator.run_stream(...)` in the picoagents examples, and in what kind of application context would you prefer each?
9. Explain why `OpenAIChatCompletionClient` is designed as a swappable client object shared across multiple `Agent` instances, rather than being embedded directly inside the `Agent` class.

## Further Reading

- Russell, S. and Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* — the source of the classical sensors/actuators/agent-function definition this chapter builds on.
- Vaswani, A. et al. (2017). "Attention Is All You Need" — the original Transformer architecture underlying modern LLMs and their context-window constraints.
- Brown, T. et al. (2020). "Language Models are Few-Shot Learners" — the paper establishing few-shot prompting.
- Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models."
- Yao, S. et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models."
- Ouyang, L. et al. (2022). "Training Language Models to Follow Instructions with Human Feedback" (the InstructGPT/RLHF paper).
- Liu, N. et al. (2024). "Lost in the Middle: How Language Models Use Long Contexts."
- Sweller, J. (1988). "Cognitive Load During Problem Solving: Effects on Learning."
- Minsky, M. (1986). *The Society of Mind.*
- Du, Y. et al. (2023). "Improving Factuality and Reasoning in Language Models through Multiagent Debate."
- Liang, T. et al. (2023). "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate."
- Wu, Q. et al. (2023). "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation."
- Fourney, A. et al. (2024). The Magentic-One project — on planning and tool-selection for multi-agent reliability.
- Dibia, V. (2023). "LIDA: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models."
- Dibia, V. et al. (2022; 2024); Mozannar, H. et al. (2025); Epperson, W. et al. (2025) — the cited developer-tooling and reliability research underlying this chapter's practitioner narrative.
- Official companion repository: `github.com/victordibia/designing-multiagent-systems`, and companion site `multiagentbook.com`, for runnable versions of every example in this chapter.
