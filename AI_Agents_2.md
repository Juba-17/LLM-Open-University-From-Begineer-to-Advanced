# AI Agent Architectures: A Comprehensive Reference

*Source: "AI Agent Mastery" session on Agent Architectures (Arize AI), presented by John (Developer Advocate) and D (Solutions Architect)*

---

## 1. Introduction: Why a Technical Definition of "Agent" Matters

The term "agent" has become heavily overloaded in industry discourse, frequently used in vague, marketing-oriented ways ("autonomous agents," "agentic workflows," "agentic" as a catch-all adjective). Before any serious engineering discussion can occur, it is necessary to strip away the hype and establish a precise, technical working definition.

### 1.1 Why the Word "Agent" Is Used At All

Even though a rigorous technical definition can make an agent sound like "just a piece of software with some conditional logic," the term remains useful as industry shorthand. It:

- Provides an easily recognized reference to an AI system capable of performing multiple actions, interacting with external tools/systems, and looping through steps to reach an answer.
- Carries certain shared industry associations that make communication faster, even though reasonable people can disagree about the precise boundaries of the definition.

**Historical origin of the term:** There is no single authoritative origin, but the term appears to have simply "stuck" as a useful label during the recent wave of LLM-powered systems. There is also a plausible historical echo from earlier (1970s-era, "first or second wave") AI research, where systems were described as acting **on behalf of** a human — a notion of delegated autonomy that maps conceptually onto today's usage. Practically speaking, part of its adoption is also attributable to marketing: naming a new category of technology helps generate interest and hype, and "agent" was the label that achieved critical mass in the community.

**Practical recommendation:** Because the definition is fuzzy and contested, when discussing agentic systems with colleagues or vendors, it is good practice to explicitly ask "what do you mean by agent?" or "what do you mean by an agentic application?" before proceeding, since the term can refer to meaningfully different architectures.

### 1.2 A Note on "Agentic" as an Adjective

The word "agentic" is increasingly applied to almost anything in the AI space. This creates a semantic dilution problem: if *everything* can be described as agentic, then the word ceases to convey any specific information ("if everything is [agentic], then nothing is"). This is a useful heuristic warning sign: broad, elastic terminology is often a symptom of hype rather than precision.

---

## 2. The Historical Evolution of Agent Architectures

### 2.1 First Generation: ReAct Agents (~1 year prior to this talk)

The dominant agent paradigm of the preceding year was the **ReAct** pattern — an acronym derived from **Rea**son + **Act**. These agents operate by:

1. Planning out a sequence of steps to take.
2. Executing those steps.
3. Continually reflecting on the results.
4. Iteratively expanding/adjusting the plan until they converge on a final answer or response.

**Motivating promise:** Because ReAct agents rely on general-purpose LLM reasoning rather than narrowly scoped logic, early proponents believed they could:
- Handle an extremely broad range of questions and task types.
- Operate over a very large "solution space" (the space of possible problems/approaches the agent could address).
- Generalize across many kinds of inputs and features, since the underlying reasoning process was not hard-coded to any particular domain.

**Why they underdelivered in practice:** Despite the lofty promise, ReAct agents mostly produced impressive short demos ("Twitter demos") rather than dependable production systems. The core problem was that the breadth of the solution space directly undermined reliability: the more general and unconstrained the space of possible actions/questions an agent must handle, the more surface area exists for it to fail on some subset of inputs. In other words, ReAct agents "collapsed under the weight of their own promised potential" — the same generality that made them exciting also made them brittle and unreliable for real-world, repeatable use cases, especially as question variety increased.

By the beginning of the year in which this talk was given, industry enthusiasm for ReAct agents (and to some extent for agents generally) had visibly cooled as these reliability problems became apparent at scale.

> **Speaker's editorial note (important nuance):** Although ReAct agents are described here as not working well *today*, both speakers explicitly stated they remain personally bullish on the ReAct paradigm over the **longer term**. The critique is about current reliability, not a permanent judgment on the architecture's future potential — ReAct-style general reasoning agents are viewed as having "a lot further to run."

### 2.2 Second Generation: Narrowly Scoped, Structured Agents (Current State)

In response to the reliability failures of ReAct agents, the industry shifted toward a **second generation** of agent design, characterized by:

- **Deliberately restricted/narrowed solution spaces.** Instead of trying to handle "any possible task," the agent is designed to handle a smaller, well-defined set of tasks or paths.
- **Explicitly defined execution flows.** The set of steps the agent can take, and the order/conditions under which it takes them, is much more rigorously specified in advance.
- **A smaller, well-understood set of possible outputs/paths**, rather than an open-ended output space.
- **An underlying ethos of "doing less, but doing it reliably."** By reducing scope, the system can be made more structured and rigorous, trading some generality for dependability.

**Concrete illustrative example:** Rather than building an agent that can scrape *any* website on the internet for information (a ReAct-style broad approach), a second-generation agent is restricted to scraping a small, fixed set of known websites. Because the structure of those specific websites is known in advance, the scraping logic can be built reliably around that known structure — trading breadth of coverage for reliability of execution.

### 2.3 Summary Comparison: First vs. Second Generation Agents

| Dimension | First Generation (ReAct) | Second Generation (Current) |
|---|---|---|
| Solution space | Broad / general-purpose | Narrow / well-defined |
| Execution flow | Emergent, decided step-by-step by the LLM | Explicitly defined paths and skills |
| Reliability | Low — many edge-case failures | Higher — because scope is constrained |
| Typical outcome | Impressive demos, weak production performance | More usable in real production settings |
| Underlying philosophy | "Do everything" | "Do less, but do it well" |

---

## 3. Formal Definition of an LLM-Based Agent

To avoid the ambiguity discussed above, the presenters propose the following precise, engineering-grounded definition:

> An **LLM-based agent** is a software system that combines multiple processing steps, where those steps include:
> 1. One or more calls to a large language model (LLM),
> 2. Some element of conditional logic or decision-making capability that determines which path/step to follow next, and
> 3. Some form of **working memory** — a shared memory or state accessible across the different steps of execution.

### 3.1 Mapping the Definition onto Familiar Software Engineering Concepts

This definition is deliberately built to map cleanly onto concepts that any software engineer already understands, which helps de-mystify the term:

| Agent Concept | Software Engineering Equivalent |
|---|---|
| "Calling an LLM" | Calling an API to get some input/output from a model |
| "Conditional logic" | Ordinary control flow (if/else, branching) |
| "Working memory" | Application state — carrying state between steps, just as most applications already do |

The broader point is that terms like "autonomous agents," "LLM agents," and "memory" can sound novel or mysterious, but they largely correspond to well-established software engineering primitives. Grounding the vocabulary in software engineering terms removes unnecessary mystique and enables clearer engineering conversations.

---

## 4. The Three Core Components of an Agent

Having defined what an agent *is*, the next step is to examine what an agent is typically *made of*. Nearly all agents can be decomposed into three core structural components:

1. **Router** — decides what step/path to take next.
2. **Skills / execution branches** — the actual logic blocks that perform work.
3. **Memory / shared state** — accessible across all steps.

Each is discussed in detail below.

---

## 5. Component 1: The Router

### 5.1 What a Router Does

The router is the component responsible for deciding, based on either the user's input or the agent's current state, what path to take next — i.e., which function, skill, or step to invoke. Note that **not every agent has a router**; some very simple agents may have a single fixed path with no branching decision to make. But most non-trivial agents include some kind of router logic.

### 5.2 Three Common Implementations of a Router

#### 5.2.1 LLM Function Calling

**Definition:** The developer provides a JSON description (a schema) of each available function to the LLM. Based on the user's input, the model decides whether to invoke one of these functions and, if so, generates the appropriate parameters extracted from the user's query.

**Example:** A `get_current_weather` function is described to the model. If a user asks a weather-related question, the LLM recognizes it should call `get_current_weather` and returns a structured function call (with parameters, e.g., a location) rather than a plain-text answer. The calling application then executes the actual function and returns the result.

**Advantages of the function-calling approach:**
- **Parallelizable tool calls:** If a single query should trigger multiple different tools, those calls can potentially be handled in parallel, which speeds up the overall agent/program.
- **Natural loop termination and unified generation step:** The router step can be called repeatedly in a loop; eventually the model decides not to call any further tools and instead generates a final natural-language response. This effectively merges the "routing" step and the "final answer generation" step into a single mechanism, which is convenient architecturally.
- **Decoupling of the router from function implementation details:** The router only needs to be given a *promise* — a JSON description of what a function does — while the actual implementation/execution of that function can be handled entirely separately from the router logic.

#### 5.2.2 LLM Intent Classification

**Definition:** Rather than exposing full function-calling capabilities to the LLM, the user's input is classified into a discrete category ("intent"), and that classification is then used to determine which skill/function to invoke.

**Why use this instead of function calling:**
- It reduces the complexity of the job given to the model — sometimes it can even be handled by a smaller, non-LLM classifier model rather than a full LLM call.
- Because the model's task is narrower (classify into one of N categories, rather than choose from and populate arbitrary function schemas), this approach tends to reduce a specific category of errors: cases where an LLM function-calling router might call an unexpected/wrong function, or simply generate a free-text response when a tool call was actually appropriate. Constraining the router's job scope reduces the surface area for these "weird" or unreliable responses.

#### 5.2.3 Rules-Based / Code-Based Routing

**Definition:** The simplest form of routing — ordinary deterministic code logic, e.g., "if this specific string appears in the input, follow this path."

**When to prefer this approach:** As a general software-engineering principle emphasized repeatedly in this session: *if you can implement routing logic in deterministic code, you generally should*, because code-based routing is:
- More reliable (deterministic, not subject to model variability),
- Typically faster/lower latency than an LLM call.

The constraint is that this only works when you can actually enumerate/define the routing logic in advance. If the space of possible user inputs is broad or effectively unbounded, deterministic code cannot capture all the necessary branching logic, and an LLM-based router becomes necessary.

> **Practical heuristic offered by the presenters:** *"Only use an LLM when you have to."* A commonly cited anti-pattern is using an LLM to perform simple text-matching tasks that a regular expression (regex) library could perform faster and more cheaply. LLMs should be reserved for cases where deterministic approaches genuinely cannot handle the required flexibility — not used reflexively because they are fashionable.

### 5.3 Choosing Between Function-Calling and Intent-Classification Routers: A Decision Heuristic

The choice between an LLM function-calling router and a simpler NLP/intent classifier is not arbitrary; it depends on the nature of the task:

- **Favor an LLM router (function calling)** when the routing decision is genuinely complex — for example, when the router may need to decide on a *combination* or *set* of tools to call to solve a problem, rather than a single clean category.
- **Favor a simpler NLP classifier** when the number of possible paths is small and fixed (e.g., "only five possible paths"), and **latency is the primary concern**. LLM calls typically operate on the order of **seconds** of latency, whereas NLP classifiers typically operate on the order of **milliseconds** — roughly **two orders of magnitude faster**. The right choice depends entirely on what you are optimizing for in your specific application.

### 5.4 Important Terminology Distinction: "Agent Routing" vs. "Model Routing"

The word "routing" is used in the AI ecosystem in two genuinely different senses, and conflating them causes confusion:

1. **Agent routing** (the subject of this section): routing *within* an agent, to decide which internal step, skill, or function to execute next.
2. **Model routing**: routing a user query to a *different underlying model* entirely, based on the type of query. This is the function performed by gateway-style tools such as **LiteLLM** or **Martian**, which sit as an interface layer between an application and multiple possible LLM backends.

**Why model routing exists — the software engineering rationale:** Rather than having application code call a specific model's API directly, an interface/abstraction layer is introduced between the application and the model. This is a direct application of the classic software engineering principle of **avoiding tight coupling**. The benefit: if you want to change which model handles a given type of query (for example, to call whichever model is currently the cheapest option that still performs adequately — since relative cost/performance of models changes over time), you only need to update the interface/routing layer, not the application code that depends on it.

Some organizations build this interface layer themselves rather than using an existing tool; the underlying motivation is the same either way.

**A related nuance raised in Q&A:** It is entirely possible for *agent routing* to incorporate *model routing* as a sub-decision — i.e., an agent's intent-based router could, as part of its logic, also decide which underlying model to use for a given intent. This blurs the line between the two concepts but is a legitimate design choice; there is no rule preventing an agent from using different models depending on the classified intent or input type.

### 5.5 Minimal Agent Architecture

At its simplest, an agent can consist of just:
- A router, and
- That router choosing to call one of a few available functions, which then return control back to the router.

The router then continues to decide whether to invoke additional functions or to break the loop and return output to the user. This constitutes one of the most basic possible agent designs: a **single-layer router with functions**.

---

## 6. Component 2: Skills and Execution Branches

### 6.1 Definition

**Skills** (also referred to as **execution branches**) are the individual logic blocks that perform the actual work of the agent once a path has been chosen by the router. A skill may be composed of:

- LLM calls,
- Application code,
- API calls (to either internal or external APIs).

### 6.2 Components Within a Skill

A single skill is often itself composed of multiple smaller **components**. The presenters' running example is a **RAG (Retrieval-Augmented Generation) retrieval skill**:

A full RAG retrieval skill might decompose into the following components:
1. Generating an embedding from the user's question.
2. Retrieving relevant documents from a vector database using that embedding.
3. Generating a final response conditioned on the retrieved documents.

Each of these three steps is a *component* within the broader *skill* of "performing RAG retrieval."

### 6.3 Terminology Note: Skills vs. Components in Diagrams

When agents are represented visually:
- Sometimes an entire skill is **collapsed into a single block** in a diagram (e.g., "go handle all my retrieval" shown as one box).
- Other times, the skill is **broken out** into its constituent components, showing the more granular internal logic steps.

Both representations are valid; which one is used typically depends on the level of detail appropriate for the audience or purpose of the diagram.

### 6.4 Updated Minimal Architecture Diagram

Building on the router-only diagram from Section 5.5, a more complete diagram shows the router directing to different skills, each of which is itself composed of individual components performing distinct logic steps, before control (typically) returns to the router.

---

## 7. Component 3: Memory (Shared State)

### 7.1 What Memory Is and Why It Exists

**Memory** refers to shared state that is accessible by each of the different components and skills within an agent as it executes. Its purpose is to allow information to persist and flow between different steps of agent execution, rather than being lost after each individual step completes.

### 7.2 What Gets Stored in Memory

Typical contents of an agent's memory/state include:

- **Retrieved context** — e.g., content pulled via retrieval that needs to be passed back into the agent for later use.
- **Configuration variables** — settings established for a particular run of the agent.
- **A log of previous steps taken** — this is especially relevant for function-calling agents (e.g., using the OpenAI API), where a running list of messages is maintained and appended to as execution proceeds. This message history is important because many router implementations rely on knowing the full sequence of prior steps (not just the most recent one) in order to decide what to do next. This connects directly back to architectural decisions about the router (Section 5): some router designs are built around full historical context, while others only need the most recent step.

### 7.3 Design Choice: Passing Variables Directly vs. Using Shared Memory/State

An initial, naive implementation of an agent might simply pass variables directly back and forth between functions as each step completes. However, it is **generally preferable to use a separate, shared memory/state object** instead, because:

- It enables more **asynchronous** or **parallel execution** of tools/skills, which is harder to achieve cleanly when data is being passed only through direct function arguments.
- It creates a cleaner architectural separation between the logic of a given step and the data that step needs to operate on.

### 7.4 Why State Matters for User Experience ("The Magic" of Good Agent Design)

Beyond the purely technical rationale, state/memory plays a critical role in the perceived *quality* of an AI-powered user experience. The core underlying principle: **users adopt products that feel "magical," and continuity is a major component of that feeling.**

**Illustrative thought experiment:** Imagine using a chatbot (like ChatGPT) that never remembered anything about the ongoing conversation — every message would require the user to repeat all relevant context from scratch. This creates significant friction and directly undermines the feeling of a smooth, low-effort experience.

**General design principle:** The central goal of good LLM application design is to let users move as fast as possible and avoid making them repeat context they've already provided. One of the fastest ways to slow users down and create a poor experience is forcing them to reintroduce information the system should already "know." This connects state management directly to product quality, not merely technical convenience.

**Illustrative example — the "black socks" problem (raised in Q&A):**
Consider an e-commerce chatbot scenario:
1. User: "I want to buy black socks." → The bot surfaces a set of black sock options.
2. User: "No, I want nylon socks." → If the system *only* processes this second message in isolation (i.e., passes it as an isolated prompt parameter with no memory of the prior turn), it will search for nylon socks generically — losing the "black" constraint from the first message. The user's *actual* intent was **black nylon socks**.

This illustrates precisely why *shared state* is conceptually different from merely *passing prompt parameters*: state functions more like a **conversation history** in a **stateful, session-level interaction** (conceptually similar to a **state machine**), where accumulated context (e.g., "black," then "nylon") should combine correctly across turns — not be treated as a sequence of disconnected, independent inputs.

### 7.5 Not All State Is Created Equal: A Hierarchy of State

A common but naive approach to state management is to indiscriminately accumulate *everything* — full conversation history, all API call return values, etc. — into one large blob, and rely on the LLM to parse through it as needed at inference time.

**Why this "brute-force" approach breaks down:**
- It does not scale. As outputs from API calls or LLM calls grow large, indiscriminately stuffing everything into a shared history becomes both computationally wasteful and, in the limit, infeasible (context window and cost constraints).
- Even where technically possible, it is not a *good* design choice, merely an expedient one.

**Better practice — recognizing that state is not homogeneous:** More sophisticated agent design treats different types of state with different levels of importance/persistence. For example:
- **High-value, persistent information** (e.g., a user's personal information/preferences) may be worth explicitly saving so it can be reused in future interactions.
- **Low-value, ephemeral information** (e.g., transient details from a single return/shipping lookup in an e-commerce flow) may not be worth persisting at all, since it adds cost and clutter without corresponding benefit.

**General guidance:** There is no universal rule for exactly what to persist — "it depends" on the specific application — but designers should deliberately think about building a **hierarchy of state importance**, rather than treating all state as equally worth saving, in order to design more intelligent and scalable state systems.

### 7.6 A Caveat on API/Model Constraints

Depending on the specific LLM API or router design being used, some implementations are architected to require the **full message history** to be passed in on every call (e.g., certain function-calling router implementations). In such cases, the developer may be forced to include the full history regardless of preference. However, where avoidable, it is generally better design to avoid blindly passing the entire history and instead apply the more selective, hierarchy-of-state approach described above.

### 7.7 Updated Architecture Diagram Including Memory

Layering memory onto the earlier diagrams: both the router and each of the individual skills/components interact with the shared memory/state object as execution proceeds — i.e., state is not merely passed linearly along a pipeline, but is accessible as a shared resource from multiple points in the architecture.

---

## 8. Agent Architecture Complexity in Real-World Systems

### 8.1 Growth of Complexity

While the diagrams discussed above are intentionally simplified for teaching purposes, real-world agents (e.g., a production chatbot with multiple skills) grow substantially more complex. A realistic example includes:

- A single entry point (user query).
- A router attempting to determine user intent.
- Branching into potentially five or more different execution branches.
- Each branch composed of multiple components, some of which themselves invoke LLMs.
- Control frequently looping back to the router after a branch completes, to decide the next step.

**Key takeaway:** As functionality is added to a real agent, the architecture can "balloon in complexity" quite quickly. This is a natural and expected consequence of building more capable systems, and should be anticipated during design rather than treated as a failure mode.

### 8.2 Important Distinction: Application Flow Diagrams vs. Backend Implementation

A subtle but important clarification: diagrams depicting an agent's decision paths in a clean, chart-like format primarily illustrate **how the application flow feels** from a user-facing perspective — they do **not** necessarily represent the literal backend implementation.

- From the **backend/implementation perspective**, an agent is more accurately modeled as a **state machine**: some components may be cyclic (looping repeatedly), and the actual code structure may look quite different from the clean directional diagram.
- From an **object-oriented vs. state-machine design perspective**, the presenters note that a state-machine framing is often the more natural way to describe agent backends: a user enters at an entry point with some intent, the system attempts an action, and if the result isn't quite right (e.g., the user wants to refine or compare something), the system needs to be able to loop back and update state repeatedly until the desired user experience/outcome is reached.

**Practical implication:** Do not assume that a clean, DAG-like (Directed Acyclic Graph) diagram used to explain an agent to stakeholders is an accurate 1:1 representation of the actual backend control flow — the backend is frequently better modeled as a cyclic state machine, even when the "front-end feel" of the interaction can be usefully diagrammed as a simpler directed flow.

---

## 9. Frameworks for Building Agents (Overview)

> **Scope note:** This session provides only a brief, introductory overview of frameworks. A dedicated follow-up session was planned to go into significantly greater depth, including building the *same* example agent across multiple frameworks for direct comparison.

### 9.1 LangGraph

**Origin and motivation:** LangGraph is described as one of the *oldest* tools in this space (notably, only about a year and a half old at the time of this talk — illustrative of how young this entire field is). It was designed specifically as a response to the limitations of standard **DAGs (Directed Acyclic Graphs)**. Prior to LangGraph, most LLM pipelines/chains were linear and **acyclic** — they could not easily loop back on themselves, making it difficult to code the kind of iterative, cyclic logic that agents often require. LangGraph was created to make it easier to manage cyclic, looping applications while still using a graph-based architecture/representation.

**Core concepts:**
- **Nodes**: individual logic blocks, conceptually similar to the "components" discussed in Section 6. Example: a "classify input" node might function similarly to a router.
- **Edges**: define the paths/transitions between nodes. Edges must be explicitly defined in LangGraph's code.
- **Conditional edges**: a special type of edge in LangGraph representing a branching decision — for example, an edge called "is_greeting" might route to different nodes depending on whether classified input is a greeting or a substantive (e.g., RAG) question.

**Key architectural note:** LangGraph does **not** have an explicit, separate concept of "a router" in the same sense discussed in Section 5. Instead, routing logic is effectively moved into these conditional edges (though some routing logic could still be embedded within nodes). Developers using LangGraph must think in terms of nodes and edges, explicitly wiring them together, including the ability to loop back to earlier nodes to create cyclic execution paths.

### 9.2 LlamaIndex Workflows

**Origin and motivation:** A more recent framework (introduced "earlier this summer" relative to this talk) — essentially LlamaIndex's answer to the same underlying problem LangGraph addresses: making it easier to define cyclic agents that can return to earlier steps in the program, among other motivations.

**Core concepts:**
- **Steps** (the workflows analog to LangGraph's "nodes"): individual units of logic.
- **Events** (the workflows analog to LangGraph's "edges"): rather than explicitly wiring nodes together via edges, steps **broadcast events** when they complete, and other steps **subscribe** to receive particular event types.

**How execution flows:** A step performs its logic and then broadcasts an event (for example, an "answer query" step might broadcast either a `query_failed` or `query_succeeded` event). A different step (e.g., "improve query") may be subscribed to receive the `query_failed` event; upon receiving it, that step executes its own logic and might, in turn, broadcast a new event (e.g., `retry_query`). Execution can still be represented visually as a diagram, but conceptually the connections shown as arrows represent **events being broadcast and received**, not fixed, pre-declared edges in the LangGraph sense.

### 9.3 Other Frameworks (Briefly Mentioned)

Numerous other frameworks exist for building single or multi-agent systems, including:
- **CrewAI**
- **AutoGen** (from Microsoft)
- **Swarm(s)**

Some of these frameworks are themselves built on top of other existing frameworks internally. The presenters note there is a large and growing ecosystem of such tools, with more detailed comparative treatment deferred to a future session.

---

## 10. Should You Even Use an Agent? A Decision Framework

Before adopting agent architecture at all, the presenters recommend explicitly asking the following diagnostic questions:

### 10.1 Question 1: Does your application follow an iterative flow based on incoming data?

If the application's behavior resembles a **state machine** — i.e., it feels like more complex outcomes are built up progressively on top of each other, and it is difficult or impossible to explicitly enumerate every possible path in advance — this is a signal that an agent may be an appropriate abstraction, since it can absorb this complexity without requiring every path to be hand-coded.

### 10.2 Question 2: Does the path need to change and adapt based on previously accumulated state?

This connects directly to the "black socks / nylon socks" example from Section 7.4: does the application genuinely need to adapt its next action based on accumulated context (state) rather than treating each input in isolation? If flexibility of this kind is required, an agent-based approach becomes more attractive.

### 10.3 Question 3: Is there a large, hard-to-enumerate, or unbounded set of possible actions/inputs/outputs?

If all the possible inputs and outputs of a program can be explicitly, exhaustively defined, then in principle all necessary paths could simply be **coded directly** (deterministically), and an LLM/agent-based approach is likely unnecessary — and potentially a symptom of chasing hype rather than solving a genuine engineering need. It is only once enumerating all paths becomes untenable, or the inputs/outputs are fundamentally unbounded or unpredictable, that an LLM-based agent becomes a more justifiable design choice.

### 10.4 Underlying Trade-off: Flexibility vs. Reliability

These three questions collectively point to a single underlying engineering trade-off:

- **Code is deterministic** — by design, it is highly reliable and executes consistently.
- **LLMs are non-deterministic** — by design, this is precisely what gives them flexibility, but it is also exactly what makes building reliable systems around them more difficult.

**General recommendation:** Treat an LLM API (and the surrounding software engineering needed to use it well) as a **tool** — and, as with any tool, use the right tool for the right problem. Do not default to using an LLM/agent architecture simply because it is trendy; use it because the problem genuinely requires the flexibility that only a non-deterministic reasoning component can provide.

---

## 11. Should You Use a Framework? Pros and Cons

### 11.1 Arguments For Using a Framework

1. **Enforced structure/paradigm.** Frameworks impose a particular way of thinking about and structuring agent logic. If a given framework's paradigm matches how you naturally think about the problem, this structure is valuable — a team of experienced framework developers has effectively pre-solved many structural design decisions for you.
2. **Useful abstractions.** Frameworks' abstractions can make certain concepts easier to reason about and implement, once you understand the abstraction the framework has chosen.
3. **Continuous improvement.** Frameworks are actively maintained and improving over time.
4. **Fast onboarding / getting started quickly.** This is highlighted as an especially strong benefit: frameworks typically come with abundant example code, notebooks, tutorials, and an active community of practitioners, making them a strong choice for newcomers to the space or for quickly bootstrapping a first working agent. This is particularly valuable for those newer to the space or less experienced with writing agent code from scratch.

### 11.2 Arguments Against Using a Framework

1. **Paradigm mismatch risk.** Sometimes the framework's underlying paradigm does not fit the actual constraints of the application being built. In such cases, a team may have already invested significant time and learning into the framework, only to discover it cannot accommodate a required use case — creating a form of **technical debt**, since working around the framework's limitations after the fact can be difficult.
2. **Over-abstraction.** The same abstractions that make frameworks easy to use can, in some situations, be "too abstracted away" from the actual thing a developer is trying to accomplish, obscuring rather than clarifying the underlying logic. In practice, this often means the abstraction gets "in the way" when trying to closely control application logic.
3. **Framework lock-in and ecosystem coupling.** Adopting a framework typically means adopting its broader ecosystem and type system as well — e.g., using LangGraph generally implies also using LangChain; using LlamaIndex Workflows implies using LlamaIndex more broadly. This means adopting the framework's own primitives and types (for example, a specific object type representing a "chat message" sent to a model). Developers already experienced within a given framework's ecosystem will find this easy, since the same primitives and logic patterns recur across that ecosystem's tools. However, for developers who need to do something slightly different from what the framework's types directly support, this can require substantial additional investigation to determine exactly what a given type is, what parameters it requires, and how to correctly construct objects the framework's internal steps expect.

### 11.3 Practical Synthesis / Recommendation ("Hot Take")

Drawing on direct production experience, one presenter offers the following practical synthesis:

- Frameworks are an excellent place to **start** and to **learn**, especially for those newer to building agents.
- However, as developers and teams become more experienced and advanced at building these systems, they frequently find themselves **outgrowing** the constraints of available frameworks.
- **Important caveat:** if you find a framework whose paradigm genuinely fits your use case and satisfies all your requirements, you should absolutely use it — "it would be madness to reinvent the wheel when the wheel's already there." Frameworks are also continuing to mature and improve over time, and this recommendation is not a permanent judgment against them.
- That said, in practice, as builders gain more experience, they often discover that the most robust and flexible "framework" is simply **the coding language itself** — using lower-level, targeted abstractions (e.g., a lightweight tool like LiteLLM specifically for model routing) only where genuinely useful, rather than committing to a large, all-encompassing framework that might not scale to future, currently unforeseen requirements.
- Frameworks may make excellent sense for smaller, well-scoped projects (the "nice Twitter demo" case), where hand-rolling everything from scratch may not be worth the effort.
- **Summary recommendation (circa 2024, per the speaker):** "Your best framework today is your coding language" — meaning that for production-grade, evolving systems, minimizing dependence on a heavyweight framework and building more directly in the underlying language often provides the greatest long-term flexibility and control.

### 11.4 Practical Note on IDE/Tooling Support

Both LangGraph and LlamaIndex Workflows were, at the time of this talk, new enough that they were nonetheless already integrated into the knowledge base of some AI-assisted coding tools (e.g., Cursor). One presenter reported:
- LangGraph worked well when used within Cursor.
- LlamaIndex Workflows worked "somewhat" — in one case, an existing pure OpenAI Assistants-based implementation was successfully converted into a Workflows-based implementation with the help of Cursor, though manual tweaking was still required afterward.

---

## 12. Security and Trust Considerations for Agents

### 12.1 Handling Authorization for Sensitive Actions (e.g., Payments/Transactions)

A key audience question concerned how to safely grant an agent access to sensitive operations — for example, authorizing transactions or handling payment methods.

**Core recommendation — treat this as a standard software engineering security problem, not a novel "AI" problem:**
- Apply the same general principle used in non-agentic software: **do not let the LLM itself touch sensitive operations directly.**
- **Logical separation** is a key protective measure. In one internal example (a co-pilot agent the presenters' team built), the agent was logically separated from other systems it could interact with. This separation proved valuable when the team experienced an attempted **jailbreak** of the agent — the logical isolation limited the potential blast radius of that attempt.
- **Reuse existing authentication mechanisms.** Rather than inventing new authorization schemes specifically for agents, teams can extend the same authentication approach used for human users — e.g., issuing the agent an authentication token (stored within shared state, but not directly exposed to or manipulable by the LLM itself) that is then used to authenticate against the relevant systems on the agent's behalf.
- **Assume a bad actor / extensive adversarial testing.** Given the non-deterministic nature of LLMs (and therefore the possibility of jailbreaking), sensitive operations should be heavily gated and guarded, with the assumption that the system could, in principle, be manipulated by a bad actor. Extensive adversarial testing is recommended for any agent capable of triggering consequential actions.

### 12.2 User Consent for Consequential Actions

A related but distinct interpretation of the authorization question concerns **preventing an agent from unilaterally taking an action a user would not want** (e.g., completing a financial transaction automatically). The recommended pattern here is straightforward: **present the proposed action back to the user for explicit confirmation** ("I want to do this thing next — is that okay?") before proceeding, rather than allowing the agent to execute consequential actions autonomously without a human-in-the-loop confirmation step.

---

## 13. Terminology Clarifications and FAQ

### 13.1 "Tool" vs. "Function" — Is There a Real Distinction?

This question was raised directly in the session, and the honest answer is that the industry does not have a universally agreed-upon technical distinction. In practice:

- The two terms are frequently used **interchangeably**.
- As an illustrative data point: OpenAI's system includes both "functions" and "tools" as concepts, where functions are described as *a type* of tool — but at the time of this discussion, there were not yet other established categories of tools in practice, making the distinction largely theoretical rather than operationally meaningful.
- Functionally, both terms typically refer to **some piece of code (or an API call) that performs a useful, callable action** as a component within a larger agent or application.

**Editorial observation from the presenters:** the word "tool," much like "agent" and "agentic," is flagged as another example of AI-industry vocabulary that "sounds fancier than it is." Reduced to underlying software engineering terms, a "tool call" is simply: sending some payload to perform an action, and receiving some result back — conceptually equivalent to an ordinary function call or API call.

### 13.2 Is a Predetermined Chain (No Looping) Really an "Agent"?

This gets into the fuzziness of the agent definition discussed in Section 1. One perspective offered: a router that has exactly one skill available and involves no looping arguably does *not* meet the bar of being a true "agent" under the working definition in Section 3 (which implies conditional branching/looping behavior) — though this remains a definitional/semantic question rather than a hard technical rule.

### 13.3 Is the Code-Based Example Agent a "ReAct Agent" or a "Second-Generation Agent"?

The presenters characterize their example code-based agents (discussed further in the companion "Comparing Agent Frameworks" session) as **second-generation agents**, because they specifically define the available functions/skills and the paths the agent can follow, returning to the router after each step, rather than allowing free-form step-by-step reasoning characteristic of ReAct. That said, the classification is described as somewhat "wishy-washy" in practice, since the LLM router itself was not strictly prevented from responding to arbitrary or unexpected questions — meaning it could, in principle, behave somewhat like a ReAct agent in edge cases, even though its intended design follows the second-generation, structured-skills philosophy.

### 13.4 Do Agents Support Multimodal Inputs/Outputs?

Yes. Depending on the design, an agent can be built to interpret or produce multiple modalities (e.g., interpreting images, surfing the web, or handling voice), and inputs/outputs can be mixed and matched across modalities as appropriate for the use case. As foundation models themselves become more multimodal over time, this capability is expected to become an increasingly natural part of agent design. The choice of modality detected in an input can even itself become a routing signal — for example, an agent might follow a different execution path depending on whether it received an image versus text input.

### 13.5 Can an Agent Router Also Function as a Model Router?

Yes — see Section 5.4 for a full discussion. There is no inherent conflict between an agent's intent-based routing logic and that same routing logic additionally selecting which underlying model to use for a given intent.

### 13.6 Do You Still Need Intent Classification If a Routed Skill's Parameters Are Static / Absent?

Yes. Even when the target skill/function requires no parameters, or only static/fixed parameters, the router still needs some mechanism (intent classification or otherwise) to decide **which** function/skill to invoke in the first place. Parameter extraction is a logically separate concern from the initial routing decision.

### 13.7 Should an Agent's Execution Be Thought of as Parallel Threads/Services?

This question (from Q&A) considered whether each agent could be modeled as a parallel process or listening service/thread. Response:
- It's possible to design systems this way (and this is conceptually similar to approaches taken by some frameworks, such as CrewAI's model of multiple coordinating agents).
- **However, the recommended default is to start with a synchronous design and add asynchronous/parallel complexity later, only if needed.** A single agent architected from the outset for parallel thread-based execution can become complicated very quickly and introduce more problems than it solves.
- **A more moderate middle ground:** A router that determines multiple functions/tools should be called in a given step *can* execute those specific calls in parallel before recombining results — this is a narrower, more manageable form of parallelism than full multi-threaded agent execution.
- **Real-world caution from production experience:** In building an internal co-pilot agent, the team deliberately restricted the router to invoking only **one tool per iteration**, specifically because they encountered significant problems from overlapping/asynchronous tool calls when multiple tools were allowed to run concurrently. This is offered as a concrete caution against defaulting to parallelization purely because it seems possible — added complexity should be justified by genuine need, following the broader engineering principle of starting simple and adding complexity only when required.

### 13.8 Should an Agent's State Machine Be Modeled as a DAG or a More General (Possibly Cyclic) Graph?

A DAG (Directed Acyclic Graph) is technically a special case of a more general graph (one without cycles). However, as discussed in Section 8.2, real agent backends are often better modeled as **general, potentially cyclic graphs / state machines**, since agents frequently need to loop back to earlier states or nodes — a capability that both LangGraph and LlamaIndex Workflows were specifically created to support (see Section 9), in contrast to earlier, purely linear/acyclic chain-based pipelines.

---

## 14. Key Takeaways

1. **"Agent" is a useful but fuzzy term.** Always clarify what a specific speaker or system means by "agent" in a given context; the underlying implementation can vary substantially.
2. **A rigorous working definition:** an agent is a software system combining multiple LLM calls, conditional logic, and shared working memory — concepts that map directly onto ordinary software engineering primitives (API calls, control flow, and application state).
3. **The industry has moved from broad, general-purpose ReAct agents (unreliable at scale) toward narrower, more structured "second-generation" agents** that trade generality for reliability — while still holding long-term promise for more general reasoning-driven approaches.
4. **Every agent can be decomposed into three core components:** a router (decision-making), skills/execution branches (the actual work), and shared memory/state (continuity across steps).
5. **Routers can be implemented via LLM function calling, LLM/NLP intent classification, or deterministic rules-based code** — with the choice driven by the complexity of the decision, latency requirements, and the reliability/flexibility trade-off.
6. **Prefer deterministic code over LLM calls wherever the routing/decision logic can actually be fully enumerated in advance;** reserve LLM-based approaches for genuinely open-ended or unbounded decision spaces.
7. **Shared memory/state is architecturally superior to passing ad hoc parameters between functions,** enabling more asynchronous/parallel execution and delivering a "magical," continuity-driven user experience. Not all state is equally valuable — design a deliberate hierarchy of what is worth persisting.
8. **Architecture diagrams typically represent the "feel" of the application flow, not necessarily the literal backend implementation,** which is often better modeled as a cyclic state machine.
9. **Frameworks (e.g., LangGraph, LlamaIndex Workflows) offer valuable structure and fast onboarding, but risk paradigm mismatch, over-abstraction, and ecosystem lock-in as systems mature** — advanced practitioners often gravitate toward lighter-weight, closer-to-the-language implementations for production systems, while still using frameworks when their paradigm is genuinely a strong fit.
10. **Security for agentic systems should be approached as standard software engineering security practice:** logically isolate sensitive systems from direct LLM access, reuse existing authentication mechanisms, require explicit user consent for consequential actions, and assume adversarial inputs are possible.
11. **Decide whether you need an agent at all** by asking whether your application requires iterative, state-dependent behavior with a large or unbounded space of inputs/outputs — if all paths can be explicitly coded, a full agent architecture may be unnecessary.

---

## 15. Glossary of Key Terms

| Term | Definition |
|---|---|
| **Agent** | A software system combining multiple LLM calls, conditional/decision logic, and shared working memory, used to iteratively work toward a response or outcome. |
| **ReAct agent** | An agent architecture based on iterative "Reason + Act" cycles: planning steps, executing them, and reflecting before continuing; associated with a broad, general-purpose solution space. |
| **Second-generation agent** | An agent design with a deliberately narrowed solution space and explicitly defined execution paths/skills, prioritizing reliability over generality. |
| **Router** | The component of an agent that decides which path, function, or skill to invoke next, based on input or current state. |
| **Agent routing** | Routing decisions made *within* an agent to select the next internal step/skill. |
| **Model routing** | Routing a query to a specific underlying LLM/model (e.g., via a gateway tool like LiteLLM or Martian), typically to optimize for cost/performance. |
| **Function calling** | An LLM capability where the model is given JSON schemas describing available functions and can choose to invoke one, returning structured parameters rather than free text. |
| **Intent classification** | Classifying user input into a discrete category/intent, which is then used to determine the next routing step. |
| **Skill / execution branch** | A logic block responsible for performing the actual work of the agent once a router has selected a path; may itself be composed of multiple components. |
| **Component** | A smaller logical unit within a skill (e.g., embedding generation, document retrieval, and response generation within a RAG skill). |
| **Memory / (shared) state** | Data accessible across the different steps/components of an agent's execution, used to maintain continuity, context, and configuration across a run. |
| **DAG (Directed Acyclic Graph)** | A graph structure with directed edges and no cycles; historically used to represent linear LLM pipelines/chains, but limited in its ability to represent looping agent behavior. |
| **Node** (LangGraph) | An individual logic block within a LangGraph-based agent, roughly analogous to a "component." |
| **Edge** (LangGraph) | A defined transition between nodes in LangGraph. |
| **Conditional edge** (LangGraph) | A special edge type in LangGraph representing a branching/routing decision between multiple possible next nodes. |
| **Step** (LlamaIndex Workflows) | An individual unit of logic within a Workflows-based agent, roughly analogous to a LangGraph "node." |
| **Event** (LlamaIndex Workflows) | A signal broadcast by a completed step and received by subscribing steps, used to move execution between steps (analogous to LangGraph "edges," but decentralized/event-driven rather than explicitly pre-wired). |
| **Agentic** | An adjective increasingly (and often loosely) applied to AI systems exhibiting agent-like behavior; flagged as prone to overuse/dilution. |

---

*End of reference document for "Agent Architectures" session. A companion reference document ("Comparing Agent Frameworks") covers the follow-up session, in which the presenters build the same example agent using a code-based approach, LangGraph, and LlamaIndex Workflows, plus a case study of Arize's production "Copilot" agent.*


# Evaluating AI Agents: A Comprehensive Reference

*Source: "AI Agent Mastery" session on Evaluating Agents (Arize AI), presented by John (Developer Advocate), D (Solutions Architect), and guest Vibhu Sapra (independent AI engineer / researcher, former founder, formerly in foundation model research)*

---

## 1. Introduction: From Building Agents to Evaluating Them

This session builds directly on two prior sessions in the series: an introduction to agent architectures (routers, skills, memory) and a deep dive comparing agent frameworks (LangGraph, LlamaIndex Workflows, and a hand-built co-pilot agent). Having established *how* agents are built, this session addresses *how they should be evaluated and improved*: what kinds of evaluations exist, how to decompose an agent into evaluable chunks, and how traditional LLM evaluation metrics extend (or fail to extend) to agentic systems.

### 1.1 Why Evaluating Agents Differs from Evaluating Ordinary LLM Calls

Traditional LLM evaluation deals with a **non-deterministic system that produces generative answers** from a given input — a comparatively contained evaluation problem (input, output, and a quality judgment). Introducing an agent adds an entirely new **dimension** to this problem, because an agent involves:

- **Multiple steps**, not just one generation.
- **Variable path length** — how many steps did it take to arrive at an answer?
- **Latency and runtime considerations** — how long can an agent be permitted to run?
- **A user-experience (UX) dimension** that does not exist for a single LLM call — e.g., can a long-running agent be interrupted mid-task? Should progress be streamed to the user?

This turns agent evaluation into what one presenter describes as "another modality of evaluation": it is not simply "is the output right or wrong," but a multi-dimensional trade-off space involving accuracy, latency, number of steps, cost, and user experience simultaneously.

**Illustrative example of this trade-off:** Suppose increasing overall answer precision (an accuracy-based metric) requires adding six additional steps to an agent's execution. This may improve correctness but will also increase latency, degrading the user experience. Deciding whether this trade-off is worthwhile is inherently a product/business decision, not a purely technical one — different products and use cases will make this trade-off differently.

### 1.2 A Real-World Motivating Example: Swapping the Underlying Model

A concrete scenario illustrating why agent evaluation matters: suppose a team needs to swap the LLM powering an agent — for example, moving from a proprietary, cloud-hosted model (such as OpenAI's GPT-4o) to a privacy-preserving, self-hosted alternative (such as Llama 3), perhaps due to data-privacy requirements. This kind of substitution can affect:

- Whether the router still reliably selects tools ("what if my agent doesn't use tools as well?").
- Whether the agent now requires more steps to reach the same answer.
- Whether the final answer quality is preserved.

Evaluating and optimizing for these effects — not just "is the final answer correct" — is a core motivation for agent-specific evaluation practices.

---

## 2. When Do Teams Start Evaluating Agents? Observed Patterns

### 2.1 The Typical Adoption Path (Developer-Community Perspective)

Based on direct observation of the developer community, evaluation adoption tends to follow a fairly consistent progression:

1. **Vibe-checking first.** The moment a prototype agent is built, engineers naturally and immediately start informally checking its outputs ("does this look okay?") — described as simply "natural curiosity as an engineer." This is considered the universal **first evaluation step**, and is explicitly endorsed as a legitimate and necessary part of the process, not something to be skipped even by advanced teams ("the first eval is always vibe-checking").
2. **Noticing failure patterns informally.** As a team plays with the prototype more, they start noticing specific failure modes — the agent "goes off the rails" on certain kinds of questions, or produces inconsistent results — often tracked informally (e.g., pasted into a spreadsheet with notes on what worked or didn't).
3. **Building structured test case sets.** Once failure patterns are noticed, teams begin crafting deliberate example sets/test cases so they have something to consistently test against.
4. **Running formal evaluations.** Only once test sets exist do teams begin running consistent, repeatable evaluations against them.

### 2.2 Where Teams Tend to Focus First — and a Common Sequencing Mistake

From the developer-advocacy perspective, most developers **start by evaluating skills and functions/branches** (the individual capabilities an agent has), and only **later** think to evaluate the **router** and the **looping/step-count behavior** of the agent. Many developers do not even initially recognize the router and step-count dimension as something that *can* be evaluated.

**Why this sequencing is suboptimal:** This is described as somewhat backward, because focusing on the router first often yields some of the **largest performance gains** available in an agentic system — the router determines which skills get invoked at all, so router errors can have outsized downstream impact compared to a single skill being marginally suboptimal.

### 2.3 Prioritizing Evaluation Effort: Not All Components Are Equally Important

A related principle: it is not worth investing evaluation effort uniformly across every component. Specifically, there is limited value in evaluating a component that belongs to a branch or downstream path the agent was never supposed to be on in the first place (i.e., if the router misrouted to the wrong skill entirely, evaluating that skill's internal quality is close to moot — the prior routing error is the actual problem to fix).

### 2.4 Organizational Variation: Speed vs. Rigor

How rigorously a team invests in evaluation depends heavily on organizational context and risk tolerance:
- Teams that need to **move extremely fast** often bypass thorough evaluation initially. This choice tends to resurface later in some form of **technical debt**.
- Teams that must be **risk-averse** (e.g., regulated industries, safety-critical use cases) tend to be far more diligent about evaluation from the outset.

There is no universally correct answer here; the right level of evaluation rigor is a function of the team's constraints and risk profile.

---

## 3. Research vs. Enterprise Perspectives on Agent Evaluation

A recurring theme in this session is that evaluation priorities differ substantially depending on whether the practitioner is operating in a **research** context or an **enterprise/production** context.

### 3.1 Research Context

- **Latency and runtime are typically not primary constraints.** If an agent can run for 24 hours and thereby achieve the best possible result, that is often acceptable or even desirable (e.g., in efforts to expand "test-time compute" for tasks like code generation, or benchmarks such as SWE-bench).
- **Optimal performance, independent of practical deployment constraints, is often the primary objective.**
- **Transparency is often valued.** Research settings tend to favor exposing the full chain-of-thought / reasoning trace, since sharing methodology and improving collective understanding is a goal of the research enterprise.
- Getting an agent to run for many steps (e.g., 200 steps) and still successfully self-correct back onto the right track is considered a significant, celebrated research achievement — the opposite of how this would typically be viewed in a production context.

### 3.2 Enterprise Context

- **Latency and cost constraints dominate.** Enterprises typically care a great deal about reducing the number of steps an agent takes, keeping execution efficient, and maintaining consistency.
- **Consistency and staying "on track"** — including reliably invoking the correct tool when relevant — is prioritized far more heavily than squeezing out marginal accuracy gains.
- **Observability constraints differ.** When agent capabilities are exposed behind an API to external developers, questions arise about how much internal reasoning/chain-of-thought to expose. Many production agent providers intentionally do **not** expose their full internal reasoning trace, treating it as **proprietary "secret sauce."** This stands in direct contrast to the research context's preference for transparency.
- **Tolerance for long-running, self-correcting loops is much lower.** Even if an enterprise agent *can* eventually self-correct after 20 steps of looping, users are typically not willing to wait that long, so this behavior is often undesirable even when it "works" in principle.

### 3.3 Underlying Unity: Same Evaluation Framework, Different Trade-offs

Despite these differences, both research and enterprise practitioners are described as ultimately looking at "the same four kinds of processes" for evaluation — the divergence lies almost entirely in **what trade-offs are considered acceptable**, not in the fundamental categories of things being measured. For example:
- Enterprise: if application speed can be roughly doubled at the cost of a slight performance hit, that trade is usually favorable.
- Research: if taking twice as long yields a modest accuracy improvement, that trade is often favorable instead.

### 3.4 The Necessity of Observability Underlying Both Perspectives

A common thread across both perspectives: none of this evaluation is possible without adequate **observability** — the ability to see the loops being taken, the steps executed, and the latency incurred at each stage. Observability is a *precondition* for making any of the trade-off decisions described above, not an optional nicety.

### 3.5 The Added Complexity of "Determinism in Process," Not Just Determinism in Output

Standard LLM evaluation typically asks: "is this specific output correct?" Agent evaluation adds an entirely separate axis: **was the *process* used to reach that output efficient, minimal, and consistent?**

**Illustrative example:** An agent might reach a correct final answer using 3 steps and 2 tool calls in one run, and reach the *same* correct final answer using 6 steps and 3 tool calls in another run. If both runs are evaluated purely on "was the output correct," they look identical — but the underlying process efficiency differs substantially. Tracking this distinction across many samples can reveal important reliability patterns: for instance, a workflow using 3 tools might fail 20% of the time, while a similar workflow using only 2 tools might fail 80% of the time — information that is invisible if only final-answer correctness is measured. This is why the presenters advocate for **tracking every step**, not merely the final output, even if not every tracked signal is immediately acted upon.

---

## 4. Recap: Typical Agent Structure (For Evaluation Context)

Before diving into specific evaluation techniques, it is useful to recall the canonical agent structure established in earlier sessions, since evaluation is organized around this structure:

- **User input** flows into a **router/planner** (which may use function calling, intent classification, or similar mechanisms).
- The router selects a path from a set of **predefined skills**.
- Once a skill completes, control either:
  - Returns directly to the user, or
  - Returns to the router, which may continue looping through additional skills, or
  - Triggers a **self-critique** step, in which the agent evaluates and potentially revises its own output before finalizing a skill's result.
- Throughout, the agent interacts with **shared state/messages** accessible across steps.

### 4.1 What Distinguishes Agents from Traditional (Single-Call) LLM Applications, From an Evaluation Standpoint

Agents are distinguished from simple, single-shot LLM calls primarily by:
1. **A planning stage.**
2. **Multiple internal checks** — e.g., after invoking a tool and receiving output, the agent may make an additional call to assess: "Is this right? Should I proceed? Did I get a valid response?" These checks function as a form of **guardrail**.

**Practical value of tracing this structure:** Good tracing — e.g., a clear "waterfall" diagram of the steps an agent took, including its planning process — allows practitioners to pinpoint exactly where a system breaks down. This, in turn, is where concrete improvements become possible: editing a prompt, improving a specific step, or otherwise refining the system.

### 4.2 Example: Swapping Deterministic Components for LLM Calls (and Vice Versa)

A validity-check step (e.g., "is this valid, structured JSON?") is sometimes implemented as an LLM call, but this can potentially be replaced with a cheaper/faster deterministic mechanism — e.g., a traditional classifier, a fuzzy string match, or a BM25-based search — depending on what the application needs to optimize for (latency vs. flexibility/quality). Having per-step observability and evaluation is precisely what allows a team to identify *which* steps are good candidates for such optimization, and to measure the resulting impact on end-to-end latency and reliability once a swap is made.

### 4.3 Consequence of Poor Observability: Silent Failure Modes

Without granular observability, a system can quietly degrade in ways that are hard to detect. **Illustrative example:** if a change to a main system prompt inadvertently causes the agent to stop invoking a critical first tool that kicks off an entire downstream chain, this failure may not present as a hard error — it may simply produce degraded final answers. Only step-level observability makes it possible to trace this back to its root cause (in this case, potentially suggesting the need for an additional guardrail around that specific step).

**Core underlying problem this solves:** many agents "work well one in a hundred times" and that one success might even go viral (see also Section 6.4 on "Twitter demos"), but this is not the same as being **robust or reliable**. Without the right evaluation/observability mindset, teams cannot systematically identify or improve on the other 99 cases.

---

## 5. Practical Discussion Topics from the Session (Organized by Theme)

The session was structured as a discussion, with many practical sub-topics raised via audience Q&A. These are organized thematically below rather than in their original chat order, since related points were often discussed in a scattered fashion throughout the session.

### 5.1 Monitoring Cost and Runtime Limits

**Question raised:** How should an agent monitor its own cost while running, and should it know to stop before hitting some limit?

**Guidance offered:**
- **Step-count limits are a common, practical mechanism.** Teams often empirically determine (e.g., via testing against a "golden" dataset) that an agent typically requires around a certain number of steps (e.g., ~20) before it meaningfully derails, and use this to set sensible limits. Even if such a limit is technically "workable," a resulting long delay (e.g., 3 minutes for a single response) may still be unacceptable from a UX standpoint, even with good streaming/interruptibility support.
- **A commonly used default limit in practice is 6 steps.** If an agent exceeds this, the process is deterministically cut off — e.g., the accumulated history is passed back to the model along with an explicit note such as "you were unable to complete this in the allotted attempts," and the agent is asked to either try a different approach or communicate back to the user (e.g., asking for clarification). This entire "cut-off" mechanism should itself be deterministic and should be reflected in the evaluation suite so it can be toggled on/off and improved over time.
- **On raw dollar cost specifically:** at least anecdotally, some well-funded agent products have been reported to run at extremely high operating costs per unit time, subsidized by external investment (a "raise money, build something that goes viral, optimize costs later" strategy) — described as a legitimate, if costly, business/product strategy choice, not a technical recommendation per se.
- **General engineering advice on cost optimization:** *do not prematurely optimize for cost.* Initially build with the best available tools and models (e.g., a lengthy reasoning chain using GPT-4o), and only later identify, via good observability, which specific components are good candidates for distillation, cheaper models, or lighter-weight approaches once the system's behavior and requirements are well understood.

### 5.2 Knowing When an Agent Should Stop vs. Keep Trying

**Question raised:** How can an agent's ability to recognize that a requested task is *not possible/available* (versus one that requires continued searching/effort) be tracked and evaluated?

**Guidance offered — a product-design mindset shift:**
- A recommended mental model is to think of the LLM itself as somewhat analogous to an unpredictable **end user** of a product. Just as good product design accounts for a "worst-case" user (e.g., replacing an unconstrained free-text input field with a constrained dropdown menu to avoid unexpected input), agent design should similarly **constrain and gate** the LLM's behavior in areas where it has been observed to go "off the rails."
- **The core mindset shift:** treat the genuinely deterministic parts of a system deterministically (i.e., use ordinary code/control flow for parts of the system where behavior can and should be fully specified), and reserve non-deterministic (LLM-driven) behavior only for the parts of the system that truly require it — applying dedicated tooling/guardrails around those non-deterministic parts specifically.
- **Practically, the most common approach observed among practitioners** (even fairly advanced ones) is simply **limiting the number of retries/loops** the agent can perform, and/or explicitly informing the model as it approaches this limit (e.g., "this is your last call").
- **Not all agents require real-time responsiveness.** For **offline agents** (e.g., an agent that processes an entire email inbox, clicks through links, and generates reports asynchronously), runtime constraints are far less pressing, since there is no user waiting synchronously for a real-time reply. In such cases, a much larger step budget (e.g., 30 steps, or even multiple full re-runs using the best available model) may be entirely acceptable if it saves meaningful human labor cost (e.g., the equivalent of hours of an executive assistant's time).
- **Latency optimization does not always mean swapping in a smaller/faster model or classifier.** In some observed cases, a fine-tuned lightweight classifier (e.g., a BART-based model) can *outperform* an "LLM-as-a-judge" approach even at the LLM's best achievable quality — meaning latency and quality improvements can sometimes be achieved simultaneously, not just traded off against each other.
- **Detecting "going off the rails" ultimately reduces to standard guardrail concepts** already used elsewhere in LLM evaluation: is the agent hallucinating, is it following an incorrect trajectory, is it misusing a tool, is it producing malformed/unparseable output, etc. These are the same underlying failure categories relevant to standard guardrails, just applied at each step of a multi-step agent process.

### 5.3 How Many Tools Should an Agent Have Access To?

**Question raised:** Is there a maximum reasonable number of tools an agent should have? Should agents be able to delegate tasks to other agents and await a response before continuing?

This question produced genuinely differing perspectives among the presenters, both of which are preserved below as they represent complementary, non-contradictory viewpoints:

**Perspective A (data-driven, empirical approach):**
- There is no universally correct number — "it depends," and there is no fixed right or wrong answer.
- A recommended methodology: **measure actual tool usage** over time (e.g., counts of how often each tool is invoked, and whether invocations are appropriate) and use this data to prune underused or misused tools rather than guessing up front.
- If an LLM router struggles to distinguish between similar tools, this is often a sign that the tools have not been **semantically well-defined/described** — i.e., the fix may lie in improving tool descriptions rather than reducing tool count. (An analogy offered: humans might struggle with an analogous ambiguity too, so this is not necessarily unique to LLMs.)
- Recommendation: it is reasonable to **start with more tools than you think you need**, get basic tool-calling working, and then incrementally add or refine tools from there.

**Perspective B (curated, minimal-tool approach):**
- An alternative analogy: if asked a history question, one would rather be given three books containing the correct answer than an entire library of thousands of books to search through — i.e., excessive optionality can itself degrade performance ("you can confuse these systems by giving them more than they need").
- Recommendation: **scope tool access to only what a given agent/sub-task genuinely needs.** For example, a coding-focused agent likely does not need access to a marketing budget database tool.
- That said, this does not mean large tool sets are never appropriate — some tasks genuinely require many tools; the point is to be deliberate rather than indiscriminate.
- Diagnostic questions to apply when troubleshooting tool-selection issues: is the agent using the right tool at the right time? Are tool descriptions sufficiently verbose and semantically distinct/diverse from one another? Improving prompt engineering and tool descriptions ("get good at prompting") is frequently the actual fix, rather than reducing tool count per se.
- **Research angle:** dedicated research exists on tool-use/function-calling optimality at very large scale (e.g., studies scaling to tens of thousands of available tools), examining where systems begin to select the wrong tool as the option space grows. Teams with sufficiently demanding requirements can build a full "optimal tool selection" check as a dedicated guardrail, though this is a more involved, research-grade undertaking.

**Reconciling anecdote (co-pilot build experience):** When building the Arize co-pilot agent, engineers observed a "if you hand someone a hammer, everything looks like a nail" effect: with only one tool available, the agent *always* invoked it (regardless of appropriateness), but as more tools were added, the agent began to discriminate more carefully between them. The team ultimately concluded that their tool **descriptions were initially quite poor**, and that improving these descriptions was central to fixing tool-selection issues — supporting the view that tool-description quality, not merely tool count, is often the operative variable.

**A related critique of frameworks:** One presenter noted a general reservation about relying heavily on agent frameworks specifically because they tend not to explain or surface these subtleties well — frameworks make it easy to keep adding tools/agents/descriptions without providing the tracing needed to notice that a given tool is rarely or never used, always fails, or has an inadequate description. This was offered as a reason to favor building custom systems with dedicated observability/tracing over relying purely on a framework's abstractions, at least once a system moves beyond initial prototyping. Tool usage evaluation specifically is described as something that frequently "makes or breaks an agent," since problems like getting stuck in loops, being unable to parse output, or selecting the wrong tool are common root causes of agent failure.

### 5.4 Where "The Magic" Happens: The Boundary Between Deterministic and Non-Deterministic Components

A recurring, higher-level theme: much of what separates a well-built agentic product from a poorly built one lies in **how skillfully the deterministic and non-deterministic parts of the system are connected**. Specifically:
- Deterministic parts (ordinary code, tool calls with known input/output contracts) should be treated and engineered as deterministic — this is the "easy," well-understood part of software engineering.
- The genuinely hard, "art rather than science" part is managing the **interface** between these deterministic components and the non-deterministic (LLM-driven) components, since the latter cannot be guaranteed to behave consistently.

**Illustrative example of tool-cost/complexity trade-offs:** Consider a tool that verifies generated code by spinning up a sandboxed execution environment (highly reliable, but slower and more resource-intensive) versus a simpler LLM-as-a-judge approach for a lower-stakes task like basic NumPy data analysis code. Depending on the task's stakes, it may be entirely appropriate to trade away some rigor (e.g., skip the sandboxed verification step) for a faster, simpler system. This decision should be driven by **what is actually at stake** in the specific application — the presenters contrast a high-stakes example (a medical co-pilot supporting a surgeon) against a low-stakes example (an introductory coding tutor for a middle-school Java class) to illustrate how differently this trade-off should be resolved depending on context.

### 5.5 Observability Platforms and Frameworks Are Both Abstractions — Use Them Critically

**Key point raised:** Just as agent-building *frameworks* (e.g., LangGraph) are abstractions with both benefits and limitations (as discussed in the prior "Comparing Agent Frameworks" session), **observability/evaluation platforms are themselves abstractions too** — including, explicitly, Arize's own Phoenix product.

**Illustrative example of framework-induced opacity:** A ~20-line LangGraph code snippet (creating an agent with a Tavily web-search tool and a Phoenix-search capability) can produce an extremely long, scrolling trace of internal steps upon a single execution. This demonstrates that frameworks make agent creation fast, but simultaneously **obscure how much is actually happening "under the hood."** This is precisely why establishing dedicated observability is one of the first and most important steps in working with agents, whether or not a framework is used.

**Important caution given directly (including self-critical pushback on Arize's own tooling):** Do not assume that whatever an off-the-shelf observability/evaluation tool happens to trace by default is a complete or sufficient picture. A polished trace visualization can be visually appealing without necessarily surfacing the *specific* signals a given team actually needs to monitor. Recommendation: if there is a useful, non-standard signal specific to your use case that a given observability tool does not capture by default, **build the additional instrumentation yourself** rather than assuming default tracing is sufficient. (Arize's Phoenix product supports **manual instrumentation** precisely for this purpose, and — being open source — also accepts direct community contributions for extending its default tracing capabilities.)

### 5.6 Distinguishing "No Errors" from "Good Responses": The Silent Failure Problem

A critical evaluation insight: an agent can complete execution with **no hard errors** while still producing a genuinely poor or unhelpful response (e.g., an agent that is simply unable to answer a question, or where a tool silently failed without halting execution). These failures typically will not surface as program errors — they surface only as **substantively bad responses**, which is precisely the category of problem best caught by vibe-checking and dedicated evaluation, rather than error monitoring alone.

**However, a counterpoint was raised:** even seemingly "soft" failures like "failed to parse API response" ideally *should* be caught by a well-designed guardrail (e.g., an LLM-based check, or an output-matching mechanism) rather than being allowed to silently pass through as an ordinary-looking response. The suggestion that such cases "shouldn't need vibe-checking" reflects a design goal: a genuinely well-built system should catch most of these failure categories automatically.

**Important nuance — "I don't know" can be the *correct* answer.** Non-deterministic systems complicate this picture because sometimes an agent legitimately lacks the necessary information, and responding "I don't know" is the *correct* behavior, not a failure. Evaluation must be able to distinguish between an agent correctly declining to answer versus an agent failing to find an answer it should have been able to find.

**Illustrative multi-cause example — "failed to connect to vector DB":** Such a message could indicate any of (at least) three distinct underlying situations, each requiring a different response:
1. A genuine agent failure — the agent itself is broken and needs to be fixed.
2. An external system failure — the vector database itself is down, has been deleted, or has an expired credential, in which case the *agent* behaved correctly by surfacing the issue; the fault lies elsewhere.
3. A recoverable UX opportunity — a well-designed agent might surface a clear, actionable message (e.g., "I couldn't connect to the vector DB; here's why") that allows a user to correct course (e.g., "actually I'm using Pinecone, not Chroma — try again"), after which the system succeeds. This mirrors the ordinary experience of interacting with a chat assistant that says "I don't know" but can be productively redirected ("try again," "think step by step," etc.).

**Critical evaluation implication:** Effective agent evaluation must be able to **localize the source of a failure** — was it the agent's own reasoning/execution, or an external dependency, or something correctable via a follow-up interaction? Evaluating "where the problem is coming from" is described as itself a core part of evaluating an agent, not a separate concern.

---

## 6. A Structured Framework for Evaluating Agents

### 6.1 The Recommended Structured Approach

Building on the discussion above, the session proposes the following general structure/process for evaluating an agent:

1. **Add observability** to the agent so its internal steps, tool calls, and paths can actually be seen.
2. **Build a set of test cases** — a representative collection of input types the agent is expected to receive.
3. **Break down the agent into its individual steps**, and create evaluators either for every step or, at minimum, for the most important/most evaluable steps.
4. **Use the resulting test set and evaluators as a consistent benchmark** to iteratively experiment and improve the agent — both to fix outright issues and to improve overall performance, not merely to catch regressions.

### 6.2 Nuance: What Does "Evaluate Each Step" Actually Mean in Practice?

A more nuanced elaboration on step 3 above: it is not sufficient to just say "evaluate each step" as a slogan — the interesting complexity lies in **how** a given step should be evaluated once you consider *both* output correctness *and* process efficiency simultaneously (see Section 3.5's tool-count example: a correct final answer reached via very different numbers of tool calls/steps looks identical if only judged on final-answer correctness).

**Recommended default: track (nearly) everything, but you do not have to act on everything you track.** Good tracing and evaluation infrastructure should, by default, capture data at every step; teams can then selectively decide which of those tracked signals are worth actively optimizing against, without needing to instrument new tracking retroactively once an interesting pattern is suspected.

**Why tracking the process (not just the output) matters — revisited:** If a system tracks both final-answer correctness *and* the specific number/sequence of tool calls used across many samples, it becomes possible to discover, for example, that a 3-tool-call path fails 20% of the time while a 2-tool-call path (also reaching a "correct" answer when it succeeds) actually fails 80% of the time — a pattern that would justify deliberately biasing the agent toward the more reliable (3-tool) path, even though both nominally "work." This kind of insight is only available with full process-level tracking, not final-output-only evaluation.

### 6.3 Avoiding Over-Engineering: A Balanced Perspective on How Much Evaluation Infrastructure to Build

Despite advocating for structured evaluation, the presenters are explicit that this should not be taken as license to over-invest in evaluation tooling prematurely. Key balancing guidance:

- **Do not over-fixate on evaluation and observability for their own sake.** Heavy vibe-checking, basic LLM-as-a-judge evaluation, and simple guardrails are often sufficient at the start; avoid over-complicating things prematurely.
- **A critical prior question: are you even evaluating the right thing?** It is entirely possible to build a rigorous evaluation suite that benchmarks the wrong target — e.g., benchmarking against generic leaderboards (such as MMLU) rather than benchmarking against what actual users are trying to do with the system, or worse, against an *assumed* (but unverified) use case that does not match real user behavior.
- **Recommended sequencing:** vibe-check first and ship quickly; only afterward layer in formal observability and tracing to discover actual failure patterns and actual usage patterns — while still ensuring the initial system itself is well-built (this is not license to "ship trash").
- **General principle emphasized repeatedly: simplicity is usually preferable to premature complexity.** Many production systems today are already *over*-engineered relative to what they actually need (a specific example given: LLM-as-a-judge is frequently overkill for problems that could be solved with a much simpler, reliable system). Increasing system complexity also increases the complexity of the observability/evaluation suite required to maintain it and the overall reliability risk.
- **A friendly disagreement surfaced between presenters:** one presenter argued that even in a fast-moving, advanced technical community, most systems remain over-engineered relative to actual need, and that there is more unclaimed value available from *simple, reliable* systems than from additional sophistication. The counterpoint offered (particularly relevant to research-oriented builders) is that some contexts genuinely reward "going all out" on complexity — deliberately building the most advanced possible system, even at high cost, specifically because doing something no one else has done is itself the goal in some (e.g., research-driven, exploration-for-exploration's-sake, or high-visibility "moonshot") contexts. Both perspectives are presented as valid depending on the goal being pursued — the throughline being: **know which goal you are optimizing for, and choose your level of engineering investment deliberately, rather than defaulting to either extreme out of habit.**

### 6.4 On "Twitter Demos": A More Balanced View Than Simple Dismissal

Although earlier sessions in this series criticized first-generation (ReAct) agents for amounting to little more than viral "Twitter demos" that don't hold up in production, this session offers a more balanced perspective: such demos, despite their reliability limitations, can serve a valuable function by **sparking further investment and progress** in a promising direction. An explicit historical analogy is drawn to early transformer context-length limitations (e.g., early BERT's 512-token limit): early, imperfect demonstrations of longer-context capability (even via "hacky" methods) helped spark broader investment that eventually led to today's much longer context windows. The broader point: an imperfect but eye-catching demo showing something is *possible* can catalyze subsequent investment (including literal fundraising) that leads to that capability eventually being done *well*.

---

## 7. Always Keep the End Goal in Mind

A meta-level point raised explicitly partway through the session, intended to prevent evaluation from becoming an end in itself:

> The ultimate purpose of building and evaluating agents is to produce a **viable product** that delivers real value — whether that value is business-oriented (e.g., revenue, cost savings, customer satisfaction) or research-oriented (e.g., a genuine capability breakthrough).

**Illustrative business example:** Booking.com's publicly known trip-planning agent is cited as an example of an agent designed to feel like a genuine travel agent for the business — supporting bookings, driving revenue, and extending the company's core service offering, rather than being an evaluation exercise pursued for its own sake.

**Important caveat — this end goal is not always a business objective.** For academics and researchers, the relevant "goal" might instead be a genuine capability milestone (e.g., building an agent that can run autonomously for three months and continually self-correct). The specific evaluation techniques used remain largely the same in either case; what differs is the lens through which "success" is defined. The presenters note this framing is broadly applicable across both research and enterprise mindsets, and caution against "getting lost in evaluation for the sake of evaluation" — regardless of whether the underlying goal is research-driven or business-driven, evaluation work should always trace back to that larger goal.

---

## 8. Evaluating the Router: A Closer Look

Having covered broader thematic ground above, the session provides a more focused technical breakdown of one especially high-leverage evaluation target: **the router**.

### 8.1 The Three Core Jobs of a Router

When a router receives an input, it is generally responsible for three distinguishable jobs, each of which can be evaluated somewhat independently:

1. **Skill selection:** choosing the *correct* skill to invoke, given the current input.
2. **Parameter extraction:** correctly extracting the relevant parameters from the input needed to populate that skill's function call.
3. **Function/code generation:** actually generating the correct function call/code to invoke the selected skill with the extracted parameters.

**Concrete illustrative example:** In a customer-service chatbot capable of tracking shipping orders, a user might write: *"Can you help me track the status of my order, order ID 1234?"* To handle this correctly, the router must:
- Recognize that this requires the **order-tracking skill** (skill selection), and
- Recognize that "1234" is the **order ID** parameter and correctly populate it into the function call (parameter extraction).

### 8.2 Where Routers Tend to Struggle in Practice

Based on direct observation, the third job (generating the correct function/code to actually invoke a skill) tends to be handled well by most modern models almost all of the time. **The more common sources of error lie in the first two jobs** — correctly selecting the right skill among the available options, and correctly extracting the right parameters from the user's input. Evaluation effort focused on the router should therefore weight attention toward skill selection and parameter extraction accuracy specifically, rather than assuming function-call generation itself is the primary risk.

### 8.3 Scope Clarification: The Router Is One of Several Evaluable Layers

Evaluating the router in this focused way (skill selection + parameter extraction + function generation) is only one layer of possible agent evaluation. Two additional layers exist:
- **Evaluating individual skills downstream of routing** (i.e., once a skill has been correctly selected, is its own internal execution correct? — largely similar to evaluating that same functionality as a standalone, non-agentic system).
- **Evaluating the overall flow/trajectory of the agent** across multiple steps — this connects to the concept of **convergence**, discussed next.

---

## 9. Evaluating Convergence: Is the Agent Making Progress?

**Convergence**, as a category of agent evaluation, concerns the *path* the agent takes toward a goal, rather than any single step in isolation. Key questions under this category include:
- Is the agent making measurable progress toward its goal over successive steps?
- Could the same correct outcome have been reached in fewer steps (i.e., more efficiently)?

This connects directly to points raised earlier in the session regarding tracking process efficiency (Section 3.5) and step-count/loop-count considerations (Section 5.1–5.2) — convergence can be thought of as the broader evaluative lens that unifies those more specific concerns about step count, tool-selection efficiency, and avoiding unnecessary looping.

---

## 10. Comparing and Combining Multiple Models

### 10.1 Should You Call Multiple Models Simultaneously and Pick the Best Result?

**Question raised:** What are the trade-offs of calling multiple models in parallel to compare results for accuracy/correctness, and selecting the best one (a "mixture of agents"-style or ensemble approach)?

**Key considerations raised:**
- **Cost is the primary and most obvious trade-off** of running multiple models per request.
- **Latency alignment matters.** Different models have meaningfully different latency profiles (e.g., comparing a fast model against a slower, more deliberate reasoning-style model), so if calling multiple models in parallel, it helps to select models with reasonably comparable latencies to avoid one call bottlenecking the others.
- **Observed model-level biases:** in practice, a commonly observed difference between models is a tendency toward differing **verbosity** in output tokens (some models producing consistently longer or shorter responses) — though other model-specific biases likely exist as well beyond this one commonly cited example.
- **In enterprise practice, calling multiple models simultaneously purely for output comparison/selection on a single request is relatively uncommon.** A more commonly observed enterprise pattern is **model routing** — routing different *types* of requests to different models based on cost or performance considerations (e.g., via a gateway tool such as LiteLLM, or a custom-built interface), rather than calling multiple models redundantly on the *same* request.
- **For evaluating accuracy/correctness across models specifically (rather than production routing), direct benchmarking is common and valuable** — different models are documented to score meaningfully differently on the same evaluation tasks, and this is worth verifying empirically (e.g., via a team's own "golden" dataset) rather than assumed.

### 10.2 Reducing Error Rate via Multi-Model Approaches: A Practical Framing

**Question raised:** Is calling multiple models specifically to reduce error rate a good idea (potentially in a "mixture of agents" style)?

**Guidance offered:**
- This is framed as fundamentally an **optimization decision**, and the right first question to ask is: *what are you actually optimizing for* by doing this (speed to first response, redundancy as a guardrail, accuracy, etc.)? For instance, one could imagine routing a query through several models simultaneously (e.g., a large closed-source model, plus several open-source alternatives) and using the first valid, correct-passing response — but this only makes sense once you know what property of the system you are trying to improve (speed, reliability, correctness confidence).
- **Multi-model comparison introduces its own new evaluation dimension specific to agentic systems: prompt sensitivity across models.** Swapping the underlying model in an agentic system (as opposed to a simple, single-shot summarization task) often requires **substantially re-tuning the prompt**, because different models have different comparative strengths (e.g., one model might be relatively stronger at coding, another might be more concise, another might follow tool-use instructions more reliably, another might parse structured outputs better). This means that for advanced, looping, tool-using agentic systems specifically, prompts frequently cannot be assumed to transfer cleanly across models — unlike simpler zero-shot/few-shot single-call use cases, where a prompt swap between models is often comparatively low-risk.
- **General recommendation: build the core system correctly first, and validate your assumptions before optimizing.** For example, a common assumption is that users always want the fastest possible first response — but in practice, users may be perfectly comfortable with a longer wait (e.g., ~10 seconds) if it yields a meaningfully better-quality response; this should be validated rather than assumed.

### 10.3 A Friendly Disagreement: Simplicity vs. Ambitious Complexity (Restated in This Context)

This discussion revisits the simplicity-vs-complexity theme from Section 6.3, applied specifically to multi-model/ensemble approaches:
- One view: nothing beats simplicity; if a simpler solution suffices, it is generally preferable to an over-engineered one, particularly in enterprise contexts where the practical goal is often "time to value" — achieving the desired outcome with the least necessary effort.
- The counter-view (offered as a genuine, non-contradictory alternative rather than a rebuttal): in research-oriented or exploration-driven contexts, deliberately over-engineering the most advanced possible solution — potentially one that does something no other existing system does — can itself be the point, even at high cost, because the objective in that context is fundamentally different (achieving a genuine breakthrough or novel capability, rather than efficient value delivery).
- **Shared underlying conclusion:** the presenters converge on the view that, for most real, practical systems (including the majority of what an "advanced" audience is likely building), there remains substantial unclaimed value available from **well-executed, reliable, simple systems** — and that increasing system complexity should be a deliberate choice tied to a specific goal, not a default behavior.

---

## 11. Key Takeaways

1. **Agent evaluation is a strictly harder, higher-dimensional problem than single-call LLM evaluation.** It must simultaneously account for output correctness, path/step efficiency, latency, cost, and user experience — not accuracy alone.
2. **The natural evaluation adoption path is: vibe-check → notice failure patterns informally → build structured test sets → run formal, repeatable evaluations.** Vibe-checking is not a step to be skipped, even for advanced practitioners.
3. **Router evaluation deserves at least as much early attention as skill evaluation** — arguably more, since router errors (skill selection and parameter extraction, in particular) often account for outsized downstream performance impact, and are commonly under-prioritized relative to skill-level evaluation in practice.
4. **Enterprise and research contexts optimize for genuinely different things** (latency/consistency/cost vs. raw achievable performance/novel capability), but both rely on the same underlying evaluation categories and both fundamentally require good observability as a precondition.
5. **Track process (steps, tool calls, latency), not just final-answer correctness.** Two runs that reach an identical, correct final answer via very different paths are not equivalent from a reliability standpoint, and this difference is invisible without step-level tracking.
6. **There is no universally correct number of tools to give an agent.** Favor a data-driven, iterative approach: track actual tool usage, invest heavily in clear/distinct tool descriptions, and prune or refine based on observed behavior rather than guessing up front.
7. **Not every "no error" response is a good response, and not every unusual response ("I don't know," a connection failure) is a bad one.** Effective evaluation must localize the true source of a failure (agent logic vs. external dependency vs. a recoverable, correctable interaction) rather than treating all anomalies identically.
8. **Frameworks and observability platforms are both abstractions, each with real limitations.** Do not assume default tracing captures everything relevant to your specific use case; build custom instrumentation where needed.
9. **Favor simplicity and avoid premature over-engineering — but remain deliberate about the trade-off.** Whether "just enough" evaluation/complexity or "maximal" complexity is appropriate depends entirely on the underlying goal (business value delivery vs. research/capability breakthrough).
10. **Always tie evaluation efforts back to the ultimate objective** — a genuinely viable product or a genuine capability milestone — rather than treating evaluation as a self-justifying activity.

---

## 12. Glossary of Key Terms

| Term | Definition |
|---|---|
| **Vibe-checking** | The informal, immediate practice of manually inspecting an agent's outputs for reasonableness, typically the very first form of evaluation applied to any new prototype. |
| **Golden dataset** | A curated, trusted set of test cases (inputs, and often expected outputs/behaviors) used as a consistent benchmark for evaluating an agent's performance over time. |
| **Convergence (agent evaluation)** | An evaluation category concerned with whether an agent's multi-step execution path is making measurable progress toward a correct final outcome, and whether that outcome could have been reached more efficiently. |
| **LLM-as-a-judge** | A pattern in which an LLM is used to evaluate or grade the output of another LLM call or system component, as an alternative to hand-written rules or classifiers. |
| **Guardrail** | A mechanism (deterministic or model-based) used to catch, prevent, or correct undesirable agent behavior — e.g., hallucination detection, malformed-output detection, or step/loop limits. |
| **Model routing** | Directing different requests to different underlying LLMs based on criteria such as cost or expected performance, typically via a gateway/interface layer (e.g., LiteLLM). |
| **Mixture of agents / multi-model ensembling** | An approach involving calling multiple models (or agents) on the same request and combining or selecting among their outputs, generally to improve accuracy, reliability, or provide redundancy. |
| **Skill selection** | The specific sub-task, within a router's overall job, of choosing the correct skill/function to invoke for a given input. |
| **Parameter extraction** | The specific sub-task, within a router's overall job, of correctly extracting the relevant parameter values from a user's input to populate a selected skill's function call. |
| **Test-time compute** | A research-context concept referring to allowing a model/agent additional computation (e.g., more steps, more reasoning) at inference time in exchange for improved output quality, often without hard latency constraints. |

---

*End of reference document for the "Evaluating Agents" session. A companion reference document ("Agent Looping") covers the following session in the series, which focuses specifically on how to design, encourage, and safely bound looping behavior in agents, using the Haystack framework as a worked example.*



# Looping in AI Agents: A Comprehensive Reference (with Haystack Framework Examples)

*Source: "AI Agent Mastery" session on Agent Looping (Arize AI), presented by John (Developer Advocate, Arize) with guest Tuana Çelik (Developer Relations Lead, deepset / Haystack)*

---

## 1. Introduction and Context Within the Series

This is the fourth session in a six-part series on agents. Prior sessions covered: (1) agent architectures and core components, (2) a comparison of agent-building frameworks (LangGraph, LlamaIndex Workflows, and a hand-built co-pilot agent), and (3) evaluating agents — including how to evaluate routers and how to assess **convergence** (whether an agent's execution path is making genuine progress toward a goal, and whether it could reach that goal more efficiently). This session focuses specifically on **looping**: how to structure agents so they can loop productively, how to detect when looping is going wrong, and how to bound looping behavior safely.

The session was presented using **Haystack**, deepset's open-source Python framework for building AI applications, with live coding demonstrations traced using **Arize Phoenix**.

---

## 2. Foundational Concepts: Haystack's Architecture

### 2.1 Pipelines and Components

Haystack's architecture is built on two core concepts:

- **Pipelines**: the overall structure that can represent either an entire agent or an entire end-user-facing application. A pipeline is what you build with Haystack.
- **Components**: the intermediate steps/building blocks that make up a pipeline, each performing a discrete unit of work.

### 2.2 Historical Evolution: Haystack 1.x → Haystack 2.0

- **Haystack 1.x** was architected as a **Directed *Acyclic* Graph (DAG)** — components and decision nodes could be connected, culminating in a final response, but the graph could not loop back on itself.
- **Haystack 2.0** (released some months prior to this talk) removed the acyclic constraint, becoming simply a **directed graph** capable of supporting **loops**. This change was motivated directly by the rise of LLMs capable of reasoning and iterating over their own outputs — the framework's maintainers concluded that native looping support needed to be a first-class capability rather than an afterthought, to accommodate self-reflective and multi-step tool-using agent patterns.

This mirrors a broader industry pattern discussed in earlier sessions of this series: multiple frameworks (LangGraph, LlamaIndex Workflows, and now Haystack 2.0) independently converged on removing the acyclic constraint from their underlying graph representations, specifically to support agentic, iterative behavior.

### 2.3 What Makes Something a "Component" in Haystack

To define a custom component in Haystack, only a small number of requirements must be satisfied:

1. A Python **class** decorated as a component (`@component`).
2. A **`run` method** on that class. The parameters of this method become the component's **inputs** within the pipeline graph.
3. **Declared output types** (via an output-types decorator), specifying what the component returns and in what type.

**Why output types matter architecturally:** Declaring output types allows Haystack to **validate that the overall pipeline graph is well-formed** — for instance, if one component is expected to output text/messages but is wired into a component expecting integers, this mismatch constitutes an invalid/malformed connection that Haystack can detect structurally, before runtime.

**A component may validly return nothing (an empty output).** This is explicitly noted as fully valid within Haystack's component model — a component's purpose does not have to be to produce new data; it can serve purely as a control-flow or side-effect step.

**Illustrative example given:** A custom "translator" component accepting a `from_lang`, `to_lang`, and a set of `documents`, and returning `translated_documents`. (In the live demonstration, this returned an empty list for illustrative purposes only, to show the mechanics of defining inputs/outputs rather than actual translation logic.)

### 2.4 Prompting in Haystack: Jinja2 Templating

Haystack's prompt construction relies on **Jinja2 templating**. Any text wrapped in curly braces (e.g., `{{ documents }}`, `{{ question }}`) within a prompt template automatically becomes a **declared input** to the pipeline. For example, a simple question-answering prompt template ("Given the documents, answer the question. Documents: {{ documents }}. Question: {{ question }}") automatically creates two pipeline inputs: `documents` and `question`.

---

## 3. Why Agents Loop: Two Fundamentally Different Motivations

The session organizes the discussion of looping around two distinct underlying reasons an agent might need to loop, which have different design implications.

### 3.1 Motivation 1: Self-Reflection and Iterative Correction

An agent may loop because it is **evaluating and revising its own output** before finalizing an answer — i.e., iterating toward a better or more complete response rather than accepting its first attempt outright.

### 3.2 Motivation 2: Sequential Multi-Tool Resolution

An agent may loop because answering a query requires invoking **more than one tool in sequence** — the output of one tool call may be needed before the agent can decide which tool to call next, or whether it has gathered enough information to produce a final answer. The agent must loop through this tool-selection-and-invocation process until it determines it has what it needs to respond.

**Both motivations are described as legitimate and valuable uses of looping** — looping is fundamentally what allows an agent to (a) self-correct and refine its own output, and (b) chain together multiple tools to resolve more complex queries that a single tool cannot answer alone.

---

## 4. When Looping Becomes a Liability: Failure Modes

While looping is a core enabling capability of agents, it can also become a serious **disadvantage** in numerous failure scenarios. The session identifies several specific, concrete failure patterns:

### 4.1 Failure Mode 1: Repeated Self-Reflection Without Improvement ("Insanity Loop")

In a self-reflecting pipeline, a generation step might produce answer **A**, which a reflection step judges to be incorrect, causing the pipeline to retry — producing answer **B**, which is also judged incorrect, causing another retry — which then produces answer **A** again, and so on, cycling between the same (still-incorrect) results without ever converging on an improvement.

This pattern is explicitly likened to the popular (though apocryphally attributed) quotation associated with Albert Einstein: *"Insanity is doing the same thing over and over again and expecting different results."* The presenter notes this genuinely does happen quite frequently in practice with naive self-reflection setups.

**Root causes and remedies suggested for this pattern:**
- **Insufficient memory of prior attempts.** The pipeline may not actually be aware that answer A was already generated and rejected in a previous iteration — i.e., the reflection step is not being given adequate context about prior attempts, so it cannot recognize it is repeating a previously-rejected answer.
- **Missing exit conditions.** The pipeline may lack a proper mechanism to determine that after a certain number of unsuccessful attempts, the loop should be abandoned entirely (rather than continuing indefinitely).

### 4.2 Failure Mode 2: Hallucinated Tool Selection

An agent with access to a defined set of tools (e.g., Tool 1, Tool 2, Tool 3) may select a **tool that does not actually exist** — a hallucinated tool call. This produces an error, and if the underlying issue is not addressed, the agent may simply hallucinate a (possibly different) nonexistent tool again on the next attempt, repeating the failure indefinitely.

### 4.3 Failure Mode 3: Tool Errors Not Handled Gracefully

A real, existing tool may itself fail — for example, a tool making an external API call might receive an error response (e.g., an HTTP 404). If the system is not designed to properly interpret and handle this kind of error response, the agent may continue to select the same failing tool repeatedly (since, from its perspective, it still needs the information that tool is supposed to provide), becoming stuck in a loop around a tool that will never succeed as currently configured.

### 4.4 Failure Mode 4: Poorly Structured Agent Prompts Lacking Exit Conditions

Perhaps the most persistently frustrating failure mode described: the agent's main/system prompt may fail to include adequate instruction about **when the agent should stop looping**. For instance, if a prompt effectively instructs the agent to "always continue by selecting the next tool" without qualification, the agent may continue selecting tools indefinitely, even after it has already obtained a valid, sufficient answer — because the prompt never instructed it that reaching a sufficient answer is itself a valid reason to exit the loop.

### 4.5 General Principle: Pinpointing and Solving These Issues Requires Proper Tooling

The presenter emphasizes that identifying *which* of these failure modes is occurring in a given case is genuinely difficult without the right observability tooling — this is a primary motivation for using a dedicated tracing tool (in this session, Arize Phoenix) alongside Haystack's pipeline execution.

---

## 5. Determining Whether a Generated Answer Is "Correct" (The Reflection Problem)

### 5.1 There Is No Universal Answer — It Is Fully Application-Dependent

A key audience question addressed directly: **how does a self-reflection step actually know whether a generated answer is correct?** The honest answer given is that this is entirely dependent on the specific application; there is no general-purpose mechanism for this.

### 5.2 A Simple Illustrative Approach: A "Done" Marker

In the live demonstration, a deliberately naive approach was used: the large language model is prompted such that, once it believes it has successfully extracted **all** required entities from a piece of text, its response includes the literal word **"done."** A custom validation component then simply checks for the presence of this marker to decide whether the reflection loop should terminate.

**This is explicitly flagged as a simplified illustrative mechanism, not a general solution.** In real applications, teams typically build their own **custom reflection/validation components**, each with conditions tailored to their specific correctness criteria — the underlying pattern (a dedicated component that decides pass/fail and triggers either continuation or termination of the loop) generalizes, even though the specific correctness check does not.

### 5.3 A Related Structural Point: Components as Simple Conditional Switches

In response to a question about whether this style of pipeline design resembles data-engineering pipelines more than an "event broadcast/receiver" model (as used by some other frameworks, e.g., LlamaIndex Workflows' event-driven model, discussed in an earlier session): the presenter notes that in many cases, a component may simply be a straightforward if/else conditional switch with no LLM involvement at all — in which case, yes, the pipeline does start to resemble a fairly ordinary data-processing/data-science pipeline. However, whenever a step in the pipeline *does* involve a large language model generating a response, the system inherently depends on an upstream, non-deterministic result whose exact content cannot be known in advance — this is the aspect that fundamentally distinguishes such pipelines from purely deterministic data-engineering pipelines, even when much of the surrounding control flow resembles ordinary data engineering.

---

## 6. Worked Example 1: Self-Reflecting Entity Extraction Pipeline

### 6.1 Pipeline Design

The first live-coded example is a **self-reflecting entity extraction pipeline**, designed to extract structured entities (specifically: **person, location, and date**) from an unstructured piece of text (example content: a set of meeting notes), looping until the system determines that all relevant entities have been successfully extracted.

**Pipeline components:**
1. **A `PromptBuilder`** component: constructs the prompt to be sent to the LLM, using Jinja2 templating.
2. **An LLM generator** (`OpenAIGenerator` in the demo, defaulting to **GPT-4o-mini** at the time of the session): generates a response given the constructed prompt.
3. **An `EntitiesValidator` custom component**: inspects the generated response (specifically, checking for the word "done," per Section 5.2) and determines one of two possible outcomes/output branches:
   - If **"done"** is present → the extraction is considered complete; the pipeline outputs the final extracted entities.
   - If **"done" is absent** → the pipeline outputs the entities extracted so far as **"entities to validate,"** which is fed back into the loop for another iteration.

**Prompt structure — conditional templating:** The prompt template used is a single template file, but is designed to render differently depending on the pipeline's current state:
- If there are **no entities to validate yet** (i.e., this is the first pass), the prompt contains the original, base instruction: extract entities from the given text, where entities should be limited to **person, location, and date**.
- If there **are entities to validate** (i.e., this is a reflection/retry pass), a different segment of the same template is used instead — one that explicitly frames the task as a **reflection**: it informs the model that this is a previous attempt, restates the original text, and shows the entities that were previously extracted, prompting the model to reconsider and improve upon them.

### 6.2 A Design Recommendation Noted During the Demo

The presenter notes that, in an ideal design, the list of expected entity categories (person, location, date) could itself be made an **explicit input to the prompt template**, rather than being hard-coded directly into the prompt text — improving the pipeline's flexibility and reusability across different entity-extraction tasks. This was noted as an improvement opportunity rather than something implemented in the specific demo shown.

### 6.3 Visualizing the Pipeline Graph

Before running the pipeline, its structure can be visualized directly, showing the loop explicitly: the `PromptBuilder` feeds into the LLM generator, whose output feeds into the `EntitiesValidator`; if there are still entities to validate, the flow loops back around to the `PromptBuilder`/generator stage again, repeating until the "done" condition is met.

### 6.4 Observed Runtime Behavior (via Arize Phoenix Tracing)

Multiple live runs were shown and traced:

- **In one earlier run** (shown from a prior execution, not live), the pipeline initially extracted an entity of the wrong category (e.g., mistakenly extracting a "quantity" or "miscellaneous" entity, categories that were never requested), before eventually converging, across iterations, on the correct categories (date and person) in a later iteration. Tracing showed failed iterations in **red** and the successful, final iteration in **green** — directly illustrating the reflection loop correcting itself over successive attempts.
- **In a live run during the session**, the pipeline correctly extracted person, location, and date entities directly from the meeting-notes example text.

### 6.5 An Actual Bug Discovered Live, via Tracing

While reviewing the Phoenix trace of the live run, the presenter discovered a genuine implementation bug: the pipeline was intended to feed the previously-extracted entities back into the prompt on each reflection iteration (i.e., "here are the entities you previously extracted"), but tracing revealed that this field was actually being passed through **empty** at each iteration — meaning the reflection step had no actual awareness of what had been extracted previously, despite the prompt template being designed to include this information.

**Root cause identified live:** the code was only checking/extracting the **first character** of the previously extracted entities data structure, rather than the full extracted content, resulting in this data being effectively discarded before being inserted into the prompt.

**Why the pipeline nonetheless appeared to work correctly in several runs despite this bug:** the underlying model (GPT-4o-mini, described as already a fairly strong model for this task) was often able to succeed even without being reminded of its own prior attempt — meaning the bug did not necessarily prevent correct final answers, but it did undermine the intended "true" self-reflective mechanism (i.e., the system was, in effect, sometimes just retrying blindly and succeeding by chance/model competence, rather than genuinely learning from a visible prior attempt as designed).

**Broader lesson illustrated by this discovery:** this is offered as a concrete, live demonstration of why dedicated tracing/observability tooling is essential — a pipeline can *appear* to be functioning correctly based on final outputs alone, while a meaningful implementation bug is silently undermining the intended mechanism. Without step-level tracing, this class of bug would likely go undetected indefinitely.

---

## 7. Worked Example 2: A Multi-Tool Chat Agent (RAG Tool + Weather Tool)

### 7.1 Purpose and Relationship to Prior Sessions

This second example (building on an agent design referenced from the prior "Comparing Agent Frameworks" session) illustrates **Motivation 2** from Section 3: looping to resolve queries that require invoking more than one tool in sequence, rather than self-reflective correction of a single output.

### 7.2 Setup: Populating a Document Store

A small, in-memory document store is populated with dummy data — sentences describing where various named individuals live. Embeddings are generated for these sentences (noted as likely unnecessary/overkill given how simple the example sentences are, but included to demonstrate the mechanism) and added to an in-memory document store.

### 7.3 Tool 1: A RAG Pipeline Wrapped as a Callable Tool

A conventional **retrieval-augmented generation (RAG)** pipeline is built first as a standalone pipeline (a simple instruction to answer questions based on retrieved content). This entire RAG pipeline is then **wrapped in a function** and registered as a **tool** available to the higher-level chat agent, with the description: *"Get information about where people live."*

**Important architectural observation:** wrapping an entire sub-pipeline (here, a full RAG pipeline) as a single tool substantially increases the potential complexity/failure surface of the overall system, since failures can now originate not just from simple tool-call mechanics, but from anywhere within the wrapped RAG pipeline itself (retrieval failures, embedding issues, generation issues, etc.). This is why observability into the *internal* execution of a tool that is itself a sub-pipeline (not just its final output) becomes especially important (see Section 7.6 below, regarding visualizing nested "tool invoker" traces).

### 7.4 Tool 2: A Simulated Weather Tool

A second tool is defined: a simulated (mocked, not a real external API call) weather-lookup tool, providing hardcoded weather data for a small set of cities (Berlin, Paris, Rome, Madrid, London). The presenter explicitly notes that using an actual live weather API would be a more realistic/compelling implementation, but a simulated version was used for the demo. A **fallback default value** (21.8°F) is returned for any city not covered by the simulated dataset — noted humorously as a very cold default value, since the rest of the simulated data is in Celsius while this fallback default happens to be in Fahrenheit.

### 7.5 Memory and Overall Chat Agent Construction

A custom **message memory component** is used to track conversation history. The overall chat agent pipeline is constructed such that:
- The generator always has access to this memory.
- The generator also has access to the defined tools (the RAG tool and the weather tool).
- If the generator decides a tool call is needed, the tool is invoked, and control loops back to the generator with the tool's result.
- This can repeat multiple times (i.e., multiple sequential tool invocations) until the generator determines it has sufficient information to produce a final answer, at which point it exits the loop with a final reply.

### 7.6 Demonstration Runs and Tracing Multi-Tool Sequences

- **Simple case:** *"Where does Mark live?"* → resolved using only the RAG tool.
- **Compound case (single follow-up turn, using memory):** *"What is the weather like there?"* → because the pipeline retains memory of the prior exchange (Mark's location), it is able to resolve "there" and correctly invoke the weather tool for the correct city (Berlin) without needing to re-ask about Mark.
- **Compound case (single query requiring sequential multi-tool resolution):** *"What's the weather like where Harry lives?"* → this single query requires the agent to first invoke the **RAG tool** (to determine that Harry lives in London), then invoke the **weather tool** (using London as the resolved location), before finally producing a combined answer (e.g., "Harry lives in London and the weather there is cloudy, 9°C").

**Value of tracing nested sub-pipeline tools:** Because the RAG tool is itself a full pipeline (per Section 7.3), Phoenix tracing is shown to reveal not just "the RAG tool was called" but the **internal steps within that nested pipeline execution** — specifically, a "tool invoker" trace showing the embedding step (via a sentence-transformers text embedder), the document retrieval step, and the final generation step that produces "Harry lives in London." A second, separate "tool invoker" trace is then shown for the subsequent weather-tool call, including its specific input parameter (location: London) and its resulting output. This level of nested visibility is presented as directly valuable for diagnosing exactly where, within a multi-tool, multi-step chain, any given failure originates.

### 7.7 Deliberately Breaking the Pipeline: A Live Failure-Mode Demonstration

To concretely illustrate Failure Mode 3 (Section 4.3) and Failure Mode 4 (Section 4.4), the presenter deliberately introduces two changes:
1. The simulated weather tool is modified to always return the literal string **"try again"** instead of actual weather data — simulating a real-world scenario where an external API might legitimately return a retry-type error response.
2. The underlying model is switched from a stronger model to **GPT-3.5 Turbo**, specifically to try to induce a failure-looping pattern (since the presenter notes that more capable/recent OpenAI models tend to already handle "try again"-style tool responses well, by recognizing that the user should simply be informed rather than looping indefinitely).

**Result:** With this configuration, the query *"What is the weather like in Berlin?"* caused the agent to repeatedly invoke the (broken) weather tool over and over, visibly producing a long, repeating chain of "tool invoker" trace entries. The presenter manually stopped the run before it completed, explicitly citing concern about running up unnecessary OpenAI API costs — directly illustrating the earlier discussion (in the prior "Evaluating Agents" session) about the real financial risk of unbounded agent looping in production.

### 7.8 Remedies Demonstrated for This Failure Mode

Two remedies are discussed:

1. **Better prompt engineering:** instructing the model explicitly on what to do if it receives a "try again"-style (or otherwise clearly unproductive) tool response — e.g., explicitly instructing it to exit and inform the user, rather than mechanically retrying indefinitely.
2. **A hard, deterministic safety limit: `max_loops_allowed`.** This is a built-in Haystack guardrail mechanism that sets a **maximum bound on how many times the same component can be executed within a single pipeline run**, providing an unconditional, deterministic backstop against runaway looping. In the self-reflection example from Section 6, this limit was set to **10** loops — a last-resort safeguard that was never actually reached during that particular demo, since the self-reflection example converged well before hitting it. The presenter describes this mechanism candidly as fairly "brute" (blunt/unsophisticated) — it is not an elegant or context-aware solution, but rather a simple, hard ceiling intended purely as a final safety net.

---

## 8. Prompt Engineering for Agentic, Looping Systems

### 8.1 There Is No Universal "Good Agent Prompt" Template

In response to a question about what should generally be included in an agent's system prompt (e.g., regarding when it must consult certain resources before proceeding, to avoid hallucination), the presenter explicitly pushes back on the idea of a single, general-purpose answer. The effectiveness of a given prompt formulation is highly dependent on the **specific model provider being used in combination with that specific prompt** — a phrasing or stylistic approach that works well with one model family (e.g., OpenAI) may not transfer well to another (e.g., Gemini), and different model families often respond differently to model-specific formatting conventions (e.g., special tags).

**Recommended approach:** treat effective prompting as requiring **direct research into what a specific model expects**, combined with genuine trial-and-error experimentation — explicitly characterized as **"an art, not a science."**

### 8.2 A Concrete Empirical Observation: Over-Explanation Can Hurt Performance

Based on direct experimentation, the presenter notes that **over-explaining instructions, particularly with OpenAI models, can sometimes yield *worse* results than being concise and direct** — e.g., using clear bullet-pointed statements of what is and is not wanted, rather than lengthy prose explanation. Constructing the specific reflection prompt used in the entity-extraction demo (Section 6) reportedly required substantial trial and error to reliably ensure the model always returned entity categories as complete lists (including explicitly empty lists for categories with no matches), rather than merely "usually" doing so.

### 8.3 Structured Outputs as an Alternative to Custom Validation Logic

**Question raised:** how do structured-output-conforming models compare to the kind of custom validation approach shown in the demo (the naive "done" marker check)?

**Response:**
- Haystack's earliest looping-based tutorial example (still maintained) was actually an **auto-correction** example using **Pydantic** to define an explicit output schema/model (e.g., a `CitiesData` model containing a list of cities) and using Pydantic to validate that a generated structured output actually conforms to that schema — directly addressing the same underlying "how do I know if this is correct" question, but via schema validation rather than an ad hoc text marker.
- OpenAI's **structured outputs** feature (noted as still in beta at the time of this session) is described, based on direct hands-on experience (first used by the presenter only a few days prior to this session, for a query-decomposition use case), as already working very well — and is speculated to likely be implemented internally using a mechanism conceptually similar to providing the model with a Pydantic-style schema, given how closely its behavior parallels that pattern.
- A brief historical aside is offered: the `instructor` Python package existed specifically to provide this kind of structured-output guarantee before native provider support existed, and the emergence of native structured-output support directly from providers (like OpenAI) is noted as something the maintainers of such packages have had to publicly react to — with the broader expectation that native structured-output support will likely become standard across most major model providers "very soon."

---

## 9. Visual / No-Code and Low-Code Pipeline Development

### 9.1 Deepset Studio

In response to a question about whether no-code/low-code visual development tools are relevant to this space, the presenter highlights **Deepset Studio** — a (at the time newly released, still waitlisted) product from deepset (the company behind Haystack) that allows these same kinds of graph/pipeline structures, including loops, to be **built visually** rather than purely in Python code.

**Key workflow characteristics:**
- Pipelines (including looping pipelines) can be constructed visually within Deepset Studio.
- Once built, a pipeline can be **exported to Python or YAML**, producing a ready-to-run pipeline definition.
- **Deepset Studio itself is scoped only to the pipeline-building step** — it does not include built-in tracing/observability. The recommended workflow is to export the pipeline, then connect it to a dedicated observability tool (such as Arize Phoenix) separately, to obtain tracing.

### 9.2 General View on Visual Tooling

The presenter reports finding this graph/component/pipeline visual paradigm genuinely useful and intuitive, particularly as pipeline complexity grows — even with only a modest number of components, a tool-calling chat pipeline (as in Section 7) can become visually complex enough that having a **"show"/visualization functionality** is valuable for confirming that the intended connections and loop structure actually match what was built. This is characterized as a reasonable **middle ground** between full abstraction (a no-code tool that does everything for you) and full code-level control (since the underlying Python/YAML remains fully accessible and editable after export) — a genuine hybrid rather than a strict trade-off between the two.

---

## 10. Memory and State Management in Looping Pipelines

### 10.1 The Ad Hoc Memory Implementation Shown in the Demo

The memory mechanism used in the live demonstration (Section 7.5) is explicitly clarified as **not an official, built-in Haystack memory feature** — it was simply an easy-to-implement mechanism (a running list/queue of prior messages) built specifically for this demo by the presenter and a colleague.

### 10.2 A Noted Limitation Connecting Back to the Entity-Extraction Example

The presenter explicitly connects this memory discussion back to the bug discovered in Section 6.5: the entity-extraction demo's reflection mechanism only retained the **single most recently generated** set of entities, not a full running history of all prior attempts. This means that if the underlying model were changed to one prone to cycling between two incorrect answers (the "insanity loop" pattern from Section 4.1 — alternating between answer A and answer B), the current implementation would likely be vulnerable to exactly that failure mode, since it lacks a full history of previously attempted (and rejected) answers, only the most recent one.

### 10.3 Haystack's Evolving Approach: Reusing Document Stores as Memory

The approach Haystack is currently exploring (and observed to be a pattern other frameworks are converging on as well) is to **extend/reuse existing document store infrastructure to serve as message memory**, rather than building an entirely separate memory subsystem. Conceptually, this means that memory itself can be implemented as a form of retrieval pipeline (i.e., a filtered retriever pipeline over stored messages) — reusing the same retrieval mechanisms already used for RAG-style tools, applied instead to a store of conversational messages/tool-call results.

**Current implementation status:** this capability is currently housed in Haystack's **experimental package** (`haystack.experimental`, installed by default alongside the main Haystack package, but imported from a separate namespace), reflecting that the team is still actively iterating on how far this document-store-as-memory pattern can be extended.

**A notable architectural benefit of this approach:** because memory is implemented via the same document-store abstraction used elsewhere in Haystack, the **backend used for memory storage becomes pluggable/configurable** — for example, a team could choose to back their agent's memory with Redis, or reuse an existing Weaviate instance they already operate elsewhere, rather than being locked into one fixed memory backend.

### 10.4 A Forward-Looking Note: Graph-Structured Memory ("Graph RAG")

The discussion briefly touches on **Graph RAG** — using a graph-structured representation for retrieval-augmented generation — as a potentially interesting direction specifically for memory representation (though the presenter had not personally experimented with this at the time of the session). A brief tangent from the host (Arize side) notes that graph-based approaches like this can carry a real practical barrier to entry: unlike many retrieval approaches that "do the heavy lifting for you," graph-based methods typically require the graph structure itself to be explicitly built upfront, which represents both a modeling and compute cost that needs to be weighed against its benefits — but was flagged as a promising area for longer-running or more complex agent memory needs.

### 10.5 A Practical Observability Gap Noted for Memory/Message Queues

Toward the end of the session, an interesting operational note surfaced live: when the multi-tool chat agent demo (Section 7) was re-run, it appeared at first that a key trace element ("tool invoker") was missing — which briefly caused concern that something had broken. It was ultimately determined that the pipeline had not actually needed to invoke any tools on that particular re-run, because the relevant answer was **already present in the retained memory/message history** from a previous execution (i.e., the pipeline correctly avoided redundant tool calls by using existing memory) — restarting the pipeline's underlying execution context, rather than its retained message memory, was what ultimately caused a subsequent run to correctly exercise the tool-invoking path again. This episode is used to highlight a genuine, acknowledged gap: current tracing tooling does not yet provide an easy, clear way to inspect the contents of an agent's message/memory queue directly, and this is flagged as a worthwhile direction for future observability tooling improvements (e.g., a dedicated "memory pane" or similar).

---

## 11. Key Takeaways

1. **Looping is a double-edged capability.** It is what enables both self-reflective correction and multi-tool sequential resolution — the two core motivations for looping discussed in this session — but it is equally the source of some of the most persistent and costly agent failure modes.
2. **Frameworks across the industry (Haystack 2.0, LangGraph, LlamaIndex Workflows) independently converged on removing the "acyclic" constraint from their underlying graph models**, specifically to support native looping — a strong signal that robust looping support is now considered foundational to agent frameworks generally, not an edge-case feature.
3. **"Is this answer correct?" has no universal mechanism — it is always application-specific**, and typically requires a custom-built validation/reflection component tailored to the specific correctness criteria of a given task.
4. **The most persistent and infuriating looping failures often stem from missing or inadequate exit conditions** — insufficient memory of prior attempts, absent retry limits, poor handling of tool-level errors, or a system prompt that never explicitly instructs the agent that a sufficient answer is itself a valid reason to stop.
5. **Deterministic safety limits (e.g., Haystack's `max_loops_allowed`) are a necessary, if blunt, last-resort safeguard** against runaway/costly looping — particularly important given the real financial risk of unbounded API calls, as concretely demonstrated live in this session.
6. **Observability/tracing is not optional for looping pipelines — it is often the only practical way to detect subtle bugs**, as directly demonstrated by the live discovery of a genuine implementation bug (empty "previous entities" being silently passed into the reflection prompt) that had gone unnoticed despite the pipeline nonetheless producing seemingly correct outputs in several runs.
7. **Prompt engineering for agentic systems is empirical and model-specific ("an art, not a science")** — there is no universal template, and behaviors such as the effect of over-explaining instructions can vary meaningfully between model providers.
8. **Structured-output features (native provider support, or manual approaches via Pydantic) offer a more principled alternative to ad hoc text-marker-based validation** for determining output correctness, and native provider support for this is maturing rapidly.
9. **Memory/state management for looping agents is an active area of framework evolution.** Haystack's emerging approach of reusing document-store infrastructure as pluggable memory (including experimental support for this pattern) illustrates one promising direction, though current tooling still has meaningful observability gaps (e.g., visibility into memory/message-queue contents) that remain to be addressed.
10. **Visual/no-code pipeline builders (e.g., Deepset Studio) can meaningfully coexist with code-first development**, offering a useful middle ground: visual construction for clarity, with full code export for continued flexibility and integration with separate observability tooling.

---

## 12. Glossary of Key Terms

| Term | Definition |
|---|---|
| **Pipeline** (Haystack) | The overall structure representing an agent or application in Haystack, composed of connected components. |
| **Component** (Haystack) | A discrete, reusable unit of logic within a Haystack pipeline, defined by a `run` method and declared output types. |
| **DAG (Directed Acyclic Graph)** | A graph structure with directed edges and no cycles; describes Haystack 1.x's architecture, which could not natively support looping. |
| **Directed graph (cyclic)** | A graph structure that permits cycles/loops; describes Haystack 2.0's architecture, enabling native support for iterative, self-correcting, and multi-tool-looping agent patterns. |
| **Self-reflection (looping)** | A pattern in which an agent evaluates its own generated output and, if judged insufficient, loops back to attempt an improved output. |
| **"Insanity loop"** | An informal term (referencing a popular quotation commonly attributed to Albert Einstein) for a self-reflection failure pattern in which a pipeline cycles between the same (incorrect) outputs repeatedly without converging on an improvement. |
| **Hallucinated tool call** | A failure mode in which an agent attempts to invoke a tool that does not actually exist within its available tool set. |
| **`max_loops_allowed`** | A guardrail mechanism in Haystack that sets a hard upper bound on how many times a given component can be executed within a single pipeline run, used as a last-resort safeguard against runaway looping. |
| **Tool invoker** (as traced in Phoenix) | A traced execution unit representing the invocation of a tool (which may itself be a full sub-pipeline, such as a wrapped RAG pipeline), including its internal steps when applicable. |
| **Jinja2 templating** | The templating engine used by Haystack for constructing prompts; variables enclosed in `{{ }}` become declared pipeline inputs. |
| **Structured outputs** | A model-provider feature (e.g., from OpenAI, noted as in beta at the time of this session) that constrains a model's output to conform to a specified schema, reducing the need for custom post-hoc output validation. |
| **Deepset Studio** | A visual, no-code/low-code pipeline-building tool from deepset (Haystack's parent company) that allows pipelines, including looping pipelines, to be constructed graphically and exported to Python or YAML. |
| **Document-store-as-memory** | An architectural pattern (under active, experimental development in Haystack) in which existing document-store/retrieval infrastructure is reused to implement conversational or tool-call memory, making the memory backend pluggable (e.g., Redis, Weaviate). |
| **Graph RAG** | A retrieval-augmented generation approach using a graph-structured data representation, noted as a promising but more compute/setup-intensive direction, including for potential use in agent memory. |

---

*End of reference document for the "Agent Looping" session. This is the fourth of a six-part series; prior companion reference documents cover "Agent Architectures," "Comparing Agent Frameworks" (referenced but not separately transcribed here), and "Evaluating Agents."*



# Comparing Agent Frameworks: A Comprehensive Reference

*Source: "AI Agent Mastery" session on Comparing Agent Frameworks (Arize AI), presented by John (Developer Advocate) and Sallyanne (Senior Product Manager, Co-Pilot product owner)*

---

## 1. Introduction and Session Structure

This is the second session in a six-part series on agents, following an initial session on agent architectures (covering the definitions, components, and core makeup of agents: routers, skills, memory). This session is split into two parts:

1. **Part 1:** A single example agent, built three separate times — once in pure code, once in **LangGraph**, and once in **LlamaIndex Workflows** — used to directly compare the developer experience, code structure, and trade-offs of each approach.
2. **Part 2:** A case study of Arize's real, production **Co-Pilot agent**, covering its architecture, design decisions, and lessons learned from building and operating it at scale.

The series also references future sessions on evaluating agents, looping/breaking agents out of loops, and later "masterclass" sessions featuring guests from LlamaIndex (Jerry Liu) and AutoGen (Chi Wang).

---

## 2. The Example Agent: A Phoenix Data-Analysis Co-Pilot

### 2.1 Use Case Overview

To enable an apples-to-apples framework comparison, a single example agent was built: a **chatbot that can interact with a Phoenix instance** (Arize's open-source LLM observability/tracing tool) to help a user gain insights into their traced application.

**What Phoenix provides (context for the example):** Phoenix gives observability over an LLM application — log-level trace data, cost tracking, latency tracking, token usage tracking, and the ability to attach evaluations (e.g., for hallucination detection, or — relevant to this very agent — evaluating function-calling/tool-selection quality).

### 2.2 The Agent's Skills

The example agent has three skills:
1. **Query documentation** — answering questions against Phoenix's documentation.
2. **Analyze data** — pulling trends or generating insights from an applied dataset.
3. **Query a trace database** — since Phoenix produces a full database of traces as an application runs, this skill can generate and execute **SQL queries** directly against that trace database.

### 2.3 Overall Architecture

The whole agent is powered by a router using the **base OpenAI API with function calling** to select among the three skills. Once a selected skill completes, control returns to the router, which decides whether to invoke another skill or return a final response to the user.

**Illustrative multi-skill example:** For the query *"What trends do you see in my trace latency?"*, the agent:
1. First calls the **SQL generation/execution skill** to pull the relevant trace latency data.
2. Returns that data to the router.
3. The router then passes the retrieved data to the **data analysis skill** to produce a trends report for the user.

This demonstrates a case where a single user query requires **two chained skills** to fully resolve.

### 2.4 Session Goals (Clarified via Q&A)

In response to a question about whether the course's goal is to teach attendees to build their own custom framework, the presenter clarifies: **the goal is not to build a framework**, but rather to (a) give insight into the benefits/drawbacks of existing frameworks, and (b) more broadly, to equip attendees to build a great *agent* — covering architectural considerations, design decisions, and industry-level context — rather than to build framework tooling itself.

---

## 3. Approach 1: The Pure Code-Based Agent (Baseline)

This implementation serves as the baseline against which the two frameworks are compared.

### 3.1 Router Implementation

- **Message list management:** Following the OpenAI convention, the implementation maintains a running list of messages — user messages, tool-call responses, and tool-call completion messages — appending a system prompt if one hasn't already been added, and keeping this list continuously updated.
- **Router call:** A call is made to the OpenAI **Completions API** to generate the next step. This call is passed a `tools` object, built from a custom abstraction called a **"skill map."**

### 3.2 The "Skill Map" Abstraction (Custom Design Choice)

The skill map is a custom abstraction — conceptually a dictionary (though implemented as its own class) — holding the functions the router/agent can call. **Design motivation:** to create clean separation between the router and the individual skills, so that new skills could be added in the future **without requiring changes to the router itself.** This is explicitly highlighted as an important, generalizable design principle: **building flexibility into agent architecture so the system can evolve over time** without extensive rework — a theme that recurs later in the Co-Pilot discussion (Section 7.1).

### 3.3 Router Response Handling

The router call returns either (a) a tool-call choice made by the model, or (b) a direct response to the user. The implementation splits on this: if any tool-call responses are present, they are handled and another router call is made; otherwise, the response is returned directly to the user.

### 3.4 Handling Tool Calls: Looping Through Multiple Calls

The `handle_tool_calls` method loops through **all** tool calls the model returns in a single turn (since the model might request multiple tools at once), extracting parameters for each and using the skill map to invoke the corresponding function, appending results as messages.

**Design trade-off noted:** In practice, for this specific use case, the model never generated complex enough queries to require two simultaneous tool calls — it tended to call one tool, return to the router, and then call another. The presenter notes explicitly that in other cases, a developer **may want to force the model to make only one tool-call choice per iteration** rather than allowing multiple simultaneous calls, since this can simplify handling; the choice of whether to allow multiple simultaneous tool calls per turn is an open design decision.

### 3.5 Example Skill: SQL Generation

The **"generate and run SQL"** skill is shown as a concrete example of a skill definition. It consists of:
- A **function description object** (a JSON-like schema) — this is what actually gets passed to OpenAI, describing what the function does and what parameters it accepts, so the model can decide when to invoke it.
- The **actual implementation code** that executes the SQL query once the model has decided to call it.

### 3.6 Summary of the Code-Based Approach

Overall, the code-based agent consists of: a straightforward router that manages the message list and delegates to a skill map, and a set of skills each with (a) a description passed to the model, and (b) actual executable logic — with the router able to recurse back into itself as needed until the flow completes.

---

## 4. Approach 2: LangGraph

### 4.1 Core Conceptual Differences: Nodes and Edges

LangGraph is architected around **nodes** and **edges**, structurally similar in overall diagram shape to the code-based version, but implemented very differently:
- A **node** for what was the router.
- A **node** for a generalized "tool call handler," which invokes whatever tools are needed and passes results back to the router node.

### 4.2 Setting Up the Graph

Building the LangGraph implementation requires explicitly defining the graph structure:
- **Tools are passed in and bound to the model.** LangGraph has a built-in concept of a **tool node**, purpose-built to work with tools (for those familiar with LangChain, this parallels the `@tool` decorator convention). This effectively **replaces** the custom skill-map abstraction used in the code-based version — the developer must adapt to LangGraph's own tool-handling convention rather than using an arbitrary custom structure.
- **The graph is composed of two nodes:** an **agent node** (the router) and a **tool node** (the abstracted tool-call handler, which internally absorbs essentially all of the logic that had to be manually written in the `handle_tool_calls` method in the code-based version).
- **Edges are defined explicitly:** a starting edge from user input to the agent node, and a **conditional edge** — a LangGraph-specific construct — to decide, after the agent/router responds, whether to proceed to the tool node or end and return to the user. A final edge routes from the tool node back to the agent node after tools complete.

### 4.3 Conditional Edges: Where Routing Logic Lives

The **`should_continue`** method is the conditional edge implementation: given a response from the router/LLM, it checks whether the message contains a tool call; if so, it routes to `tools`; if not, it ends and returns to the user. **Architecturally significant point:** this conditional edge takes the place of the manual if/else routing logic that, in the code-based version, lived directly within the main control flow (i.e., LangGraph moves this routing decision out of ordinary code and into an explicit, dedicated graph construct).

### 4.4 A Major Adaptation Challenge: Fitting Custom Skills into LangGraph's Tool Structure

This was described as **one of the single biggest differences/challenges** encountered. Because the original functions/skills had been built as their own class using the custom skill-map abstraction, several aspects of that design did not translate cleanly into LangGraph's tool model:

- **Member functions (methods taking `self`) caused problems.** If a function had, say, three parameters — one of them `self`, plus other legitimate parameters — LangGraph (more specifically, the underlying **Pydantic** validation layer) would raise a **validation error**, since the model correctly avoids generating a function call including the `self` parameter, but LangGraph's tool-handling machinery gets confused by its presence in the function signature.
- **Resolution required:** redefining tools as **discrete, standalone methods**, rather than using the original custom class-based skill abstraction.

### 4.5 Personal Assessment: Strengths of LangGraph

- **Clean, top-level graph definition.** Being able to define the full graph structure in a way that's immediately legible (nodes/edges laid out explicitly) was seen as a strong positive — it provides a clear, holistic map of what the whole agent looks like.
- **Useful built-in features**, such as the tool node abstraction, allowed quick initial setup.

### 4.6 Personal Assessment: Challenges of LangGraph

- **Extension friction.** Extending the framework to fit custom skills (as in Section 4.4) became annoying, requiring significant work to get types correct.
- **Debugging difficulty.** Debugging becomes harder because the developer is frequently debugging into the framework's own internal code, rather than purely their own logic.
- **The core, most significant difficulty: an added layer of translation between developer intent and LLM behavior.** One of the fundamentally hardest problems in building any agent is getting the LLM to reliably do what you want at a given step. Adding a framework introduces an **additional translation layer**: the developer's code must be shaped into whatever format LangGraph expects, LangGraph then relays this to the LLM, and the response that comes back may differ from what was intended — creating an extra debugging burden to determine whether/how the framework's own handling altered things along the way.
- **Conditional edges are a matter of taste.** Whether this feels like a benefit or a burden depends on whether a given developer's mental model naturally separates conditional/branching logic into dedicated edge constructs, versus preferring it embedded directly in ordinary code.

---

## 5. Approach 3: LlamaIndex Workflows

### 5.1 Overall Architectural Similarity, With Key Differences

Structurally similar at a high level (again, routing between a router-like node and a tool-call-handling node), but LlamaIndex Workflows introduces **one additional architectural piece: a workflow setup step** — noted as the presenter's own specific architectural choice for this implementation, not an inherent requirement of the Workflows framework itself.

### 5.2 Creating and Initializing the Workflow

The implementation **extends a Workflow class** from LlamaIndex, adds an LLM to it, and sets up memory (functioning similarly to the "state" concept used in the other two approaches).

### 5.3 Tool Setup: Function Tool Objects

Similar to LangGraph, some adaptation work was required to make the custom tool setup work with Workflows — specifically, creating a **"function tool"** object (a LlamaIndex-specific construct) for each function, then passing these into a list. **Notably, this was described as considerably easier than the LangGraph adaptation:** the underlying skill/function definitions themselves did **not** need to be redefined — only the way they were packaged into a list needed to change. This is a meaningful contrast with LangGraph (Section 4.4), where the actual skill classes had to be fundamentally restructured.

### 5.4 Router Code: Closer to the Code-Based Baseline

A key observed takeaway: the router code for Workflows looks **more similar to the original code-based agent than to LangGraph.** The implementation sends a message, keeps track of a message list, and makes a call to LlamaIndex's asynchronous tool-calling method, then updates memory, and finally checks whether tool calls were returned — if so, it triggers a **tool call event**; if not, it triggers a **stop event.**

### 5.5 Event-Driven Architecture: Steps and Events

Workflows is fundamentally an **event-driven architecture**. Instead of nodes and edges, it uses:
- **Steps** — units of logic (conceptually parallel to LangGraph's nodes).
- **Events** — instead of explicitly defined edges, steps **broadcast events** upon completion, and other steps are set up to **receive/subscribe to** particular event types, triggering their own execution when that event fires. For example, triggering a `tool_call_event` causes the defined `tool_call_handler` method (which is set up to receive that specific event type) to be automatically invoked and passed the relevant data by the Workflows framework.

### 5.6 A Significant Practical Limitation: Difficulty Running Synchronously

Workflows is built to operate **asynchronously by default**, and the presenter reports genuine difficulty getting it to run **synchronously** — some underlying framework functions are inherently asynchronous, requiring the developer to manually redefine associated classes/methods as synchronous when extending the Workflow class. There was no simple flag or setting to flip this over; production use is generally expected to be asynchronous anyway, so this limitation is characterized as a relatively minor concern in practice, though notable for anyone specifically requiring synchronous execution.

### 5.7 Tool Call Handler

The `tool_call_handler` closely parallels the code-based agent's equivalent method — looping through tool calls, extracting parameters (using LlamaIndex-specific object/type syntax, though the underlying logic is essentially the same), triggering the relevant function calls, updating the message list, and finally triggering a **router input event** to send flow back to the router (repeating the loop).

### 5.8 Personal Assessment: Strengths of LlamaIndex Workflows

- **No changes needed to the underlying skill/function definitions themselves** — a genuine practical advantage over the LangGraph experience (Section 4.4).
- **Fully step-based logic organization.** The presenter found it conceptually easier to convert an existing method directly into a "step," compared to LangGraph's requirement to split logic across both nodes *and* edges. (Caveat: this preference may be partly an artifact of converting an *existing* code-based agent rather than building fresh from scratch within Workflows.)
- **Easier-to-track message state**, specifically because Workflows does not provide an object that automatically pulls from/updates messages on your behalf — the developer must **explicitly** load and update the messages list at each relevant point. While this means Workflows "handles less" for you compared to LangGraph's automatic tool-node message handling, it also makes debugging comparatively more transparent, since nothing happens implicitly without the developer having explicitly written it.

### 5.9 Personal Assessment: Challenges of LlamaIndex Workflows

- **Debugging remains difficult**, for the same underlying reason as with LangGraph: reliance on another framework's internal abstractions.
- **Synchronous execution is difficult to achieve** (see Section 5.6).
- **Events, like conditional edges, come down to personal taste.** As agents/flows grow more complex, an event-driven architecture may become *easier to code* but potentially *harder to conceptually map out/understand* at a glance, compared to a more linear/explicit graph. Preference here is largely subjective.

---

## 6. Direct Comparative Q&A and Clarifications

### 6.1 How Do Different Approaches "Know" a Function's Description?

- **Code-based agent:** the model's understanding of each function comes from an explicit, hand-constructed JSON object (via a `get_combined_function_description` method) passed directly to OpenAI.
- **LangGraph:** function descriptions are derived from the **docstring** included directly in the function definition — a materially different mechanism from the explicit JSON schema of the code-based approach.

### 6.2 "Less Control Over the LLM Call" in LlamaIndex Workflows — Clarified

A question arose about an offhand comment suggesting "less control" over the LLM call when using Workflows. On reflection, the presenter clarifies this was **not actually reduced control over the LLM call itself** — the developer can still fully control which functions are passed in. Rather, the issue was that **passing function definitions requires navigating multiple additional wrapping objects** (a "function tool" object containing the callable function, plus a separate "tool metadata" object containing the name/description) before these are ultimately serialized into the JSON format OpenAI expects. This extra layering made **debugging** harder (e.g., tracking down parameter-naming issues), even though raw control over which functions are exposed was not actually diminished. This is characterized as an instance of the same broader theme repeated throughout this session: **added abstraction layers between the developer and the LLM increase debugging friction**, even when they don't necessarily reduce control in principle.

### 6.3 Is the Code-Based Agent a "ReAct Agent" or a "Second-Generation Agent"?

All three implementations (code-based, LangGraph, Workflows) are characterized as **second-generation agents** (per the taxonomy from the prior "Agent Architectures" session), because they specifically define available functions/skills and the paths the agent can follow, explicitly returning to the router after each step — rather than the more open-ended, step-by-step improvisational reasoning characteristic of ReAct agents. **Caveat offered:** this classification is described as somewhat "wishy-washy" in practice, since the LLM router itself was never strictly prevented from responding to arbitrary/unusual questions — meaning it could, in some edge cases, behave somewhat like a ReAct agent, even though the intended design follows the more structured, second-generation philosophy.

### 6.4 Does LangGraph Support Multi-Agent Group Chats?

Yes — LangGraph supports multi-agent workflows, with a documented example pattern involving a **supervisor** coordinating multiple agents. However, these multi-agent setups still operate on a **graph-based structure**, meaning the pathways by which agents communicate must be **explicitly defined** in advance. This contrasts with frameworks like **CrewAI**, which allow a more **open-ended** approach to how different agents coordinate and communicate with each other.

### 6.5 Getting LLMs to Behave Better with Function Calling (Deferred, Then Addressed)

This question — about LLMs sometimes mishandling function-call formatting — was explicitly deferred earlier in the session (to preserve time for the Co-Pilot section) and addressed later during the Co-Pilot Q&A (see Section 8.5).

### 6.6 "Messages State" vs. "State Graph" in LangGraph

In response to a question about using `MessagesState` instead of a custom `StateGraph`, the presenter candidly notes this was **not a deliberate architectural decision** on their part, and they had not closely compared the two approaches — an honest acknowledgment that not every implementation choice reflects a deeply considered architectural trade-off.

### 6.7 Is LlamaIndex Workflows "Closer to Pure Python"?

Yes, based on direct personal experience — Workflows was found to be **lighter in terms of what it enforces on the developer**, i.e., closer to plain Python.

**Trade-off framing (a double-edged sword):**
- **LangGraph** is recommended as more helpful for **first-time agent builders**, since it forces a very structured approach, and benefits from a large amount of existing community material/examples.
- **Workflows** is more **open-ended** — a developer needs to already have their own opinion about what an agent's structure should look like, since the framework itself imposes comparatively little predefined structure, offering more freedom in how steps move between each other.

---

## 7. Case Study: Arize's Production Co-Pilot Agent

### 7.1 Context and Framing

Arize's Co-Pilot agent was built **before** many of today's more mature frameworks (LangGraph, LlamaIndex Workflows) were available or well thought out. The presenter explicitly notes that decisions might look different if the team were starting today — but the team remains satisfied with where they landed, and there are **no plans to switch** the underlying approach; Co-Pilot is fully in production, released to all customers who want it.

### 7.2 High-Level Architecture: A Graph-Style Router, Not a Pure Hub-and-Spoke

**Initial idea — "Hub and Spoke":** In this model (similar to some of the frameworks discussed above), the router acts as the sole central point of communication: skills never respond directly to the user; all data flows back through the router, which alone produces the final user-facing response.

**Adopted approach — Graph-style:** The team ultimately chose an architecture where the **router selects a function/tool, and that tool/skill can respond directly to the user** — rather than always routing every response back through the router first. **Motivation:** this was explicitly a **product-driven decision**, chosen because it was more streamlined and because direct responses meaningfully improved the user experience.

### 7.3 Router Flexibility: Multiple Routers, Configurable Scope

- The router can be given the flexibility to choose **multiple functions** in a single pass.
- Co-Pilot uses **multiple distinct routers** for different parts of the product — allowing some routers to have broad latitude to call multiple skills, while others are deliberately restricted to calling only a specific, limited set of (sometimes just one) skill(s), depending on what's appropriate for that part of the product. All routers, however, are designed around the same underlying graph-style approach described above.

### 7.4 Design Philosophy: Built for Adaptability and Iteration

A deliberate, explicit design objective from the outset: to make Co-Pilot **highly adaptable and iterative** — testing ideas, seeing what resonates, and shaping skills based directly on customer feedback, rather than committing early to a fixed, rigid structure. **General recommendation for anyone building an agent:** keep adaptability and iteration in mind as a first-class design concern, since the need for it will arise "sooner rather than later."

### 7.5 State Management and Context: Why Full State Control Was Chosen Over the OpenAI Assistants API

**Context:** Co-Pilot's core use case is helping users debug and gain insights from their AI applications/ML models — a task where **maintaining context/state about where the user is in an ongoing debugging session** is critical.

**Initial approach considered — OpenAI's Assistants API:** Because it has tool-calling and state management built in, this seemed like an obvious initial choice, since it appeared to simplify state management. However, with the Assistants API, **OpenAI manages context and state internally, with no control on the developer's side** — this required passing the full conversation history and ultimately did not work well enough for Co-Pilot's needs.

**Adopted approach — the Completions API with explicit state control:** The team switched to using the **Completions API** directly, which grants **full control** over state. This recurring emphasis on **full production-setting control** is described as a major, recurring theme throughout the Co-Pilot build process, since many "out of the box" options were found unsuitable for the team's specific production requirements.

**How explicit state control is used in practice:**
- Only **relevant context** is included, deliberately reducing unnecessary context that isn't needed.
- The team uses **summarization**: rather than passing full raw conversation history, a **summarized state** is embedded directly into the function/prompt — summarizing the conversation so far, where the debugging session currently stands, and any insights Co-Pilot has already provided.
- **Why this matters:** because users often need to **reset context** entirely mid-session (e.g., moving from one debugging topic to something unrelated within the same session), having full control over state made this kind of context-reset practical; relying on OpenAI's built-in state management would have made this significantly harder.

### 7.6 Choice of Azure OpenAI as the Underlying Engine

Co-Pilot uses **Azure OpenAI** rather than calling OpenAI directly. **Reasons given:**
- Strong **data protections and security** built into Azure (documented separately by Arize, covering Microsoft/Azure-specific security measures).
- Addressing customer concerns about sharing analytics data with OpenAI directly: customers are not required to opt into Co-Pilot if they'd prefer not to, and Arize also supports **custom Azure endpoints**, allowing customers to fully own their own deployment if desired.
- In practice, most customers are comfortable with this arrangement given these protections.

### 7.7 Latency and Guardrails

- **Latency** is a factor explicitly considered when choosing which underlying model/engine to use for a given task — the team does not want users waiting minutes for a response, and **streaming** (discussed further in Section 7.9) helps directly address perceived latency from a UX standpoint.
- **Guardrails:** Co-Pilot includes some guardrails around toxicity and jailbreaking-attempt detection, but the team explicitly notes that, in practice, **they have not needed extensive additional guardrails** — the built-in protections in modern models have generally proven sufficient. The team does still use some dedicated prompting, some genuine guardrails, and ongoing **evals** to continuously monitor these dimensions. It's noted, citing observations from OpenAI's own reporting on their "o1"-class models, that jailbreaking has become "nearly impossible" with the latest generation of models at the time of this session.
- **Additional security/software-engineering measures:** an **authentication token** is associated with the Co-Pilot session/agent (used for the same kind of software-engineering-level security discussed in the "Agent Architectures" session), and the system is designed with **logical separation ("siloing")** so that one customer cannot access another customer's data — tied to the specific model/application in question, with Arize's security team directly involved in this design.

### 7.8 Skills: Structure and the JSON Data-Formatting Lesson

Co-Pilot has a substantial and evolving number of skills (the presenter notes not even being certain of the exact current count). At a high level, skills:
- **Fetch data and process it,** often via external/API calls.
- **Format that data into a prompt** that the LLM then uses to perform analysis.
- Have the ability to **respond directly to the user**, or alternatively call other tools or return control to the router — mirroring the code-based skill pattern discussed earlier in the session (Section 3.5).

**A specific, explicitly named lesson learned — data formatting:** The team experimented extensively with different data formats for the information passed into prompts (e.g., CSV-style formatting), and ultimately found that **Co-Pilot performed best when data was formatted as JSON** — described as a major, directly actionable lesson learned during development, chosen specifically for the **consistency and clarity** it provided the model.

### 7.9 Structured Responses and Streaming: A Detailed Lesson in UX Engineering

This is highlighted as one of the trickiest engineering challenges encountered while building Co-Pilot.

**The core problem:** when streaming a model's output token-by-token, it can be genuinely difficult to know **when the LLM has actually finished responding.**

**Solution — an "EOM" (end-of-message) delimiter:** Co-Pilot's prompting explicitly instructs the model to split its response into (a) raw streamable text, and (b) a distinct trailing section marked by this delimiter. This design:
- Enables **easy UI integration**, since the front end can stream the raw text portion in real time.
- Provides **confidence that the LLM has finished responding**, since the EOM delimiter's appearance is a clear, unambiguous signal of completion.
- Enables **easier parsing in the UI** more generally, since the response is broken into clearly delineated segments rather than one undifferentiated stream.

**Combining streaming with structured output — a real technical constraint:** The team found they **could not use streaming and structured/JSON-mode responses within the same single model call.** Their EOM-delimiter approach was specifically developed as a workaround: it allows the raw text portion of a response to be streamed conversationally, while a separate, clearly demarcated portion of the response (after the EOM marker) can still deliver structured content for other UI purposes — achieving both goals without requiring true native structured-output mode in the same call.

**Handling very large/verbose outputs:** Some skills (e.g., a prompt optimizer, or particularly in-depth analyses) can produce very long responses. The team built additional tooling/formatting layers specifically to manage these large outputs and avoid character-overflow issues in the UI — accepting a small amount of added latency as a worthwhile trade-off for a better-formatted final response.

**Charts and tables — handled largely outside the LLM itself:** When asked whether charts/tables are rendered via a dedicated skill/LLM call or via a structured-data approach, the answer is that this is **handled primarily outside of the LLM/skill layer entirely** — the system leverages the structured-response/EOM approach to extract relevant data, and then relies on **front-end rendering capabilities** to actually style/render that data as a chart or table within the Co-Pilot interface, rather than having the LLM itself directly produce chart/table markup.

### 7.10 Key Reasons the Team Built Co-Pilot From Scratch (Rather Than Using a Framework)

Explicitly stated: at the time Co-Pilot was being built, **no framework was mature enough, or fit the team's specific, highly specialized needs well enough** — the requirements described throughout this section (full state control, custom streaming/structured-output handling, Azure-specific security integration, custom JSON-based data formatting) were each highly tailored to Co-Pilot's particular production context. The team has **no current plans to switch** away from this custom-built approach.

---

## 8. Additional Q&A Themes from the Co-Pilot Discussion

### 8.1 Approach to Building Generative-AI-Based Products (General Product Advice)

Rather than claiming deep domain expertise in this still-nascent area, the guidance offered focuses on **user-need-driven design**:
- Build things users genuinely need and will benefit from.
- Be deliberate about *where* an AI feature is placed within a broader product — Co-Pilot was intentionally **not** inserted into contexts where it "didn't belong" merely for the sake of including it; placement should genuinely facilitate a real user workflow.
- Avoid creating a fragmented experience — the goal is a single, cohesive product (Arize, enhanced by Co-Pilot) rather than treating the AI feature as a wholly separate product.
- **Project management implications:** because there remain many unknowns in building generative-AI applications, teams should **stay closely involved, remain flexible, and be ready to adapt** — for example, skill behavior can shift unexpectedly when an underlying model is swapped, potentially breaking previously working skills. Building flexibility into both architecture and team process is essential, alongside having clear requirements and an "ideal experience" mapped out in advance.

### 8.2 Convincing the LLM to Follow Output Requirements and Formatting Rules

This directly returns to a question deferred earlier in the session (Section 6.5).

**On function-calling reliability specifically:**
- A common misconception is that function calling will simply "just work" reliably — in practice, it often doesn't without deliberate effort.
- The fix largely comes down to **investing real effort in tool/function descriptions**. Specific practices found helpful: providing explicit **function-calling guidelines** within the prompt (e.g., "call this function if X," "this function is helpful when Y").

**On output formatting more broadly:**
- Genuinely difficult and still imperfect in practice — the presenter candidly notes still encountering cases where Co-Pilot produces malformed or unexpected output, even with careful design.
- Approach used: substantial prompt engineering, including providing **explicit examples** of the desired output format (e.g., specifying exactly what should follow the EOM delimiter and how), combined with **additional tooling outside the LLM call itself** to further enforce/correct the desired output shape.

### 8.3 A Specific, Actionable Prompt-Engineering Tip: Positive Instructions Outperform Negative Ones

A concrete, hard-won lesson: attempting to instruct a model with **negative constraints** (e.g., explicitly telling a router *"do not respond with SQL"*) was found to be **unreliable** — the model does not always correctly process this kind of negation, and can paradoxically "latch onto" the very thing it was told to avoid (e.g., seeing the word "SQL" mentioned at all, even in a prohibition, and still producing SQL). **Recommendation:** achieve substantially more reliable results by **telling the model what *to* do**, rather than primarily specifying what *not* to do. This is offered as a genuinely useful, hard-earned piece of practical prompt-engineering guidance (only half-jokingly called "prompt engineering 101"), rather than a purely theoretical recommendation.

### 8.4 On Combining CrewAI (Built on LangGraph) with LlamaIndex Workflows

**Question raised:** what about layering CrewAI (which is itself built on top of LangGraph) as a starting point for a first agent app, and then layering LlamaIndex Workflows on top for scaling?

**Response:** General skepticism about **combining both frameworks simultaneously** — since this effectively means using LlamaIndex, LangChain, *and* LangGraph together at once, which is described as a somewhat unusual/strange approach. A preferred alternative offered: use LangGraph directly and build **custom orchestration on top of it yourself** (the way CrewAI itself does), rather than attempting to combine two separate, independently opinionated frameworks in a single system. CrewAI's underlying approach is acknowledged as interesting and functional in its own right, but not necessarily something to be layered together with an entirely separate framework like Workflows.

---

## 9. Key Takeaways

1. **The same agent design can be implemented in meaningfully different ways across code, LangGraph, and LlamaIndex Workflows** — each introduces a different mental model (custom abstractions and direct control; nodes/edges/conditional-edges; steps/events) with distinct trade-offs in debuggability, flexibility, and onboarding ease.
2. **A recurring theme across both frameworks is "added abstraction = added debugging friction,"** even when raw control over model behavior is not necessarily reduced — every layer between developer intent and LLM behavior is a place where things can silently diverge from what was intended.
3. **LangGraph required more substantial rework of existing custom skill/function code** (to fit Pydantic-based tool validation) than LlamaIndex Workflows did, which mainly required adapting how functions were packaged/listed rather than restructuring the functions themselves.
4. **LlamaIndex Workflows defaults to asynchronous execution**, which can create real friction for developers who specifically need synchronous behavior.
5. **Framework choice is genuinely a matter of personal fit and taste** for certain dimensions (conditional edges vs. events), but also has objective trade-offs: LangGraph offers more built-in structure and community material (better for first-time builders); Workflows is closer to plain Python and more open-ended (requiring the developer to bring their own architectural opinions).
6. **Arize's production Co-Pilot agent was deliberately built from scratch**, prioritizing full control over state, streaming, and structured output — because available frameworks at the time did not adequately support the team's highly specific production requirements (context-resetting mid-session, EOM-based streaming/structured-output hybrid, Azure-specific security integration).
7. **Concrete, transferable lessons from Co-Pilot:** favor JSON over CSV-like formats for data passed into prompts; invest heavily in clear tool/function descriptions to improve function-calling reliability; prefer positive ("do X") over negative ("don't do X") prompt instructions; and design explicitly for adaptability/iteration from the very start of an agent-building project.
8. **Full state control (via the Completions API) was chosen over the OpenAI Assistants API** specifically because production needs (context resetting, summarized state, fine-grained content-window management) exceeded what an out-of-the-box, provider-managed state solution could support.

---

## 10. Glossary of Key Terms

| Term | Definition |
|---|---|
| **Skill map** | A custom abstraction (used in the code-based baseline agent) — essentially a dictionary/lookup of callable functions the router can invoke, designed to decouple the router from the specifics of individual skills. |
| **Tool node** (LangGraph) | A built-in LangGraph object that abstracts away the manual logic of handling tool calls, replacing custom tool-handling code such as the code-based agent's `handle_tool_calls` method. |
| **Conditional edge** (LangGraph) | A LangGraph construct representing a branching decision between nodes, used here to decide whether to proceed to a tool call or return to the user. |
| **Function tool** (LlamaIndex Workflows) | A LlamaIndex-specific wrapper object used to register a callable function (with associated metadata) as a usable tool within a Workflows-based agent. |
| **Tool call event / router input event** (LlamaIndex Workflows) | Custom event types used to move execution between the router and tool-call-handling steps in an event-driven Workflows implementation. |
| **Hub-and-spoke architecture** | An agent design in which the router is the sole point of communication with the user; skills never respond directly, always routing output back through the router first. |
| **Graph-style architecture (Co-Pilot)** | An agent design (adopted by Arize's Co-Pilot) in which the router selects a function/skill, and that skill can respond directly to the user, rather than always routing through the central router. |
| **Completions API** | The underlying OpenAI API used by Co-Pilot (instead of the Assistants API) to gain full, explicit control over conversational state and context. |
| **EOM (end-of-message) delimiter** | A custom marker used in Co-Pilot's prompting strategy to signal that the model has finished its main streamed response, enabling reliable UI parsing and a hybrid streaming/structured-output workflow. |
| **Summarized state** | A condensed representation of a conversation's progress (rather than full raw history) explicitly constructed and passed into prompts, used by Co-Pilot to manage context efficiently and support mid-session context resets. |

---

*End of reference document for the "Comparing Agent Frameworks" session. Companion reference documents in this series cover "Agent Architectures," "Evaluating Agents," and "Agent Looping."*



# Multi-Agent Frameworks and AutoGen: A Comprehensive Reference

*Source: "AI Agent Mastery" masterclass session on Multi-Agent Frameworks (Arize AI), featuring Chi Wang (founder/creator of AutoGen, Microsoft) in conversation with Jason (Co-founder/CEO, Arize) and John (Developer Advocate, Arize)*

---

## 1. Introduction and Session Scope

This session is presented as the culminating "masterclass" session of a core agent-focused course series, featuring a direct conversation with **Chi Wang**, the founder and creator of **AutoGen**. The session covers:

- How AutoGen's multi-agent framework works, including its core design philosophy.
- Common multi-agent organizational patterns (two-agent chat, sequential chat, group chat).
- Termination conditions — how multi-agent conversations know when to stop.
- A direct comparison and discussion of **OpenAI's Swarm** library (released shortly before this session), including how its "handoff" pattern relates to prior work in AutoGen.
- A worked example: the same simple agent built first in a "lower-level framework" style, then rebuilt using AutoGen's native multi-agent idioms — with a notable, concrete performance difference between the two versions.
- Evaluation approaches for multi-agent systems, including live tracing via Arize Phoenix.
- Broader, forward-looking commentary from Chi Wang on the state of multi-agent research and the field's most significant open problems.

---

## 2. AutoGen: Core Concepts and Design Philosophy

### 2.1 The Basic Building Blocks: Agents and Their Interactions

The core idea in AutoGen: developers **define individual agents**, and then **define how those agents interact with each other.** Each agent has:
- Its own **prompt/system message** defining its role and instructions.
- Its own **configuration** (e.g., which underlying model to use).
- Optionally, its own **tools** — and tools can be shared across multiple agents ("cross-pollination"), not necessarily assigned to only one agent each.

Once agents are individually defined, AutoGen provides several **organizational structures** (patterns) governing how these agents communicate and collaborate to reach a solution — ranging from free-flowing joint conversation to more hierarchical, back-and-forth exchange.

### 2.2 AutoGen's Explicit Design Principle: Simple Abstractions for Complex Applications

Chi Wang describes the core design goal directly: to give developers **a very simple concept/abstraction** for reasoning about genuinely complex applications — i.e., developers should be able to build applications **as advanced/complex as they want**, while only needing to learn a **small number of core concepts** to get there.

**Human-team analogy:** The framework is explicitly designed around the intuition of **how a human team works** — just as people reason about how to organize a team of humans to tackle a complex task, AutoGen lets developers **decompose a task across multiple agents** and define how those agents communicate, mirroring familiar human-organizational reasoning rather than requiring an entirely novel mental model.

**Serving both beginners and advanced users:** AutoGen offers both:
- A **simple, high-level interface** for straightforward cases.
- A **lower-level interface** (e.g., send/receive or general-reply-style primitives) for developers who need **fine-grained, custom control** over complex interaction behaviors among agents.

### 2.3 AutoGen as a "Higher-Level" Framework Compared to LangGraph/LlamaIndex Workflows

Drawing a direct comparison to frameworks covered in earlier sessions in this series (LangGraph, LlamaIndex Workflows), AutoGen is described as sitting at a **different, higher level of abstraction**. Whereas LangGraph/Workflows are more "in the weeds" — precisely defining how a *single* agent (or, in some cases, multiple agents) operates step by step — AutoGen operates a layer above that, letting developers define **more holistically**: what the overall goal is, and how multiple agents work together to reach it, rather than micromanaging every internal step.

**A concrete anecdote illustrating this distinction:** When the Arize team's first attempt at porting an existing (LangGraph-style) agent into AutoGen was made, it ended up using only a **narrow slice** of AutoGen's actual capabilities, rather than genuinely leveraging the framework's native multi-agent design — this experience is revisited in detail in Section 6.

### 2.4 Cross-Framework Interoperability: Can Agents from Different Frameworks Talk to Each Other?

**Question raised directly:** can a tool/agent from one framework communicate with an agent from a different framework?

**Response:** Because most of these systems fundamentally operate on **text** as their processing unit (text input, text output), the practical difficulty of interoperability across frameworks is **less severe than one might initially expect** — a plain text interface is a fairly universal "lowest common denominator." It is plausible to compose an agent or multi-agent system from one framework as a callable unit that produces/consumes text, and hand that off to another framework's system.

**An important limitation, though:** while composing whole "units" together in this text-based way is plausible, it is **not straightforward to swap in an agent from a different framework as a direct, individual participant** within, say, an AutoGen group chat — interoperability tends to work at the level of composing whole systems/units together (treating one framework's agent as a tool for another), rather than seamlessly mixing individual agent objects from different frameworks within the same native conversational structure.

---

## 3. Defining Agents in AutoGen: A Concrete Example

An individual AutoGen agent definition typically includes:
- A **system message** defining the agent's role/purpose.
- A **model configuration** (specifying which model to use).
- Optionally, **tools** attached later.

Multiple such agents can then be **composed into a chat** — connecting two (or more) defined agents together, providing an introductory message, and (optionally) a summarization method to condense the eventual result at the end of the exchange.

---

## 4. Termination Conditions: How Does a Multi-Agent Conversation Know When to Stop?

This is described as a genuinely important design consideration — determining when a back-and-forth conversation between agents should end.

### 4.1 Simple Approach: Max Turns

The simplest mechanism: a **`max_turns`** parameter, specifying a fixed cap on how many back-and-forth exchanges are allowed.

**Limitation:** in many cases, the exact number of turns required for a given task is not known in advance, making a fixed cap either wastefully generous or prematurely restrictive depending on the specific task.

### 4.2 More Flexible Approach: Natural-Language Termination Conditions

For the common **two-agent chat** case, a more flexible pattern is available:
1. One **LLM-based agent** is instructed, via its system message, to examine the conversation history and **decide** — based on natural-language criteria specified in its prompt (e.g., *"if the user's requirement/task has been successfully addressed and you have verified it, say TERMINATE"*) — whether to emit a specific **termination word/phrase**.
2. The **other agent** in the exchange (which, notably, need not itself be LLM-based — it could be a simple, rules-based "user proxy" agent) is responsible for **recognizing** that termination signal in messages it receives, using **predefined rules** (e.g., checking for the presence of the termination word).

**Robustness enhancement:** this recognition logic can be made **more general**, so that it can still correctly recognize termination even if the agent doesn't produce the *exact* same termination word every time — for example, by also recognizing related annotations or markers, not just an exact string match. This same general pattern (an LLM decides when to signal completion; a receiving mechanism recognizes that signal, possibly with some tolerance for variation) generalizes to other multi-agent conversation structures beyond the simple two-agent case.

### 4.3 An Interesting Observed Trade-off: Iteration Improves Results, But Reduces Control

A notable, cross-cutting observation raised during the session: when Arize's team compared results from their **lower-level frameworks** against the same problem tackled with AutoGen's genuine multi-agent, back-and-forth structure, the **AutoGen-based results were often qualitatively better** — attributed specifically to the value of **iterative back-and-forth exchange between agents.** This suggests there is something genuinely valuable, at a deep level, about **improving LLM outputs through iteration** — whether through explicit quality checking, or simply through the natural effect of multiple passes/exchanges. However, this same iterative flexibility is also **harder to control precisely** — which is exactly what mechanisms like termination conditions and turn limits are designed to manage.

---

## 5. AutoGen's Organizational Patterns for Multi-Agent Interaction

### 5.1 Two-Agent Chat

The simplest pattern (illustrated in Sections 3–4): two agents conversing directly, with a termination mechanism (max turns, or natural-language-based) governing when the exchange concludes.

### 5.2 Sequential Chat

A more explicitly pipeline-like structure: agents (or pairs of agents) work together on a task, and **at some defined point, control moves over to another set of agents**, continuing down a defined pipeline as steps are completed. Using the earlier "team of people" metaphor, this is analogous to defining explicit **steps in a pipeline** — starting with one pair, then handing off to the next pair as the team progresses. This gets conceptually closer to the more explicitly-defined, lower-level structures seen in frameworks like LangGraph/Workflows, except that **each individual step/stage in the pipeline is itself potentially composed of multiple collaborating agents**, rather than a single node/step.

### 5.3 Group Chat: The Most Autonomous Orchestration Pattern

**Structure:** multiple agents participate in a shared conversation, with a dedicated **"group chat manager"** agent that decides, at each turn, **which agent gets to speak next** — functioning conceptually as a kind of specialized, higher-level **"super router."**

**Why this is considered the most autonomous orchestration pattern (compared to two-agent and sequential chat):** there is **no predefined, fixed order** of which agent speaks when — the group chat manager makes this decision dynamically.

**Advanced group-chat functionality — combining automated and constrained control:** the group chat manager's decisions need not be *purely* left to LLM judgment. Developers can layer in additional **constraints on transition order**, including:
- **Hard-coded (deterministic) constraints** on which agent(s) may speak next in certain situations.
- **Natural-language instructions** specifying conditions under which a transition should occur.

This combination allows developers to strike a **deliberate balance** between fully automated LLM-driven orchestration and more explicit, domain-expertise-driven (or finite-state-machine-like) transition logic, depending on what a given application needs.

### 5.4 A General Design Question: How Many Agents in a Group, and How Many Tools per Agent Manager, Is Too Many?

This is explicitly characterized as a **problem-dependent** question, with the answer depending on at least two key factors:

1. **Model capability.** Stronger/more capable models can handle more complex role definitions across a larger number of agents without becoming confused; weaker models get confused more readily as the amount of context/role-definition grows.
2. **Role distinctiveness among agents.** If agents in a group have highly **distinct, well-separated roles**, it is comparatively easy (even for weaker models) to correctly select the right one for a given situation. If agents perform **very similar work**, this becomes genuinely confusing — even for strong models — since the distinguishing signal between them is weak.

**A recommended mitigation strategy — hierarchical routing:** Rather than presenting a single flat list of many (possibly similar) agents to choose from, developers can introduce a **hierarchical structure**: define a small number of well-separated, high-level categories at the first routing level, and only within a selected high-level category do you then present a further set of (possibly more similar) agents at a second level — allowing much more specific/nuanced instructions to be given about distinguishing between agents *within* a narrower category, rather than needing to distinguish among, e.g., hundreds or thousands of largely similar options all at once. This directly parallels concepts of **hierarchical routing** already observed in production customer systems by the Arize team.

### 5.5 Can Agents Directly Ask the User for Clarification?

Yes — two distinct mechanisms are described:

1. **A dedicated "user proxy" agent within the group chat.** This agent does not perform any task-processing itself; its sole function is to prompt for and relay human input back into the group chat. Conceptually, this is simply treated as **one of several participating agents** in the group — human input is only solicited when the **group chat manager** decides it is this agent's "turn" to speak. **Key design requirement for this pattern:** the group chat manager must be given a **very clear definition** of exactly when this user-proxy agent should be given a turn, so that the manager reliably recognizes the appropriate moment to solicit human input.
2. **A dedicated tool/function** that, when invoked by any LLM-based agent, **pauses the conversation** to directly solicit human input, then returns whatever the human provides back into the group chat. This tool can be attached to **any given agent** (a per-agent design choice), enabling that specific agent to **proactively** decide, on its own, when to request human input.

**Key distinction between the two mechanisms:** in the first (user-proxy-agent) pattern, the decision to solicit human input is made **implicitly**, via the group chat manager's own turn-selection logic. In the second (tool-based) pattern, the decision is made **explicitly**, via a direct function-call decision made by a specific agent. The appropriate choice between these two patterns depends on the specific application.

---

## 6. Comparison of "Handoffs" in OpenAI's Swarm vs. AutoGen's Prior Work

### 6.1 Context: The Release of OpenAI's Swarm

At the time of this session, OpenAI had recently released **Swarm** — a multi-agent framework — just a couple of days prior. The session explicitly frames this as an opportunity to compare Swarm's approach against AutoGen's own, earlier-developed patterns for agent-to-agent task handoff.

### 6.2 How Swarm Implements Agent Handoffs

Chi Wang's assessment: Swarm has a **"reasonably simple"** way of handling agent transitions — the currently active agent decides, via a **function call**, whether it wants to **hand off** the task to the next agent.

### 6.3 AutoGen's Earlier, Related Prior Work: The "Math Problem-Solving" Example

Chi Wang notes that AutoGen had explored a **similar handoff concept very early on** — specifically, in AutoGen's very first published application example (in its original technical report): a **math problem-solving** scenario.

**Design of this earlier example:**
- Initially, a **"student proxy"** agent talks to a **"student assistant"** agent, working together to help answer math problems.
- If the student assistant agent determines it **cannot satisfactorily answer** the question, it has the option to **invoke a function** that **creates a new, nested chat** — between a separate **"human expert user"** agent and a **"human expert assistant"** agent.
- Once this nested chat concludes, the original student assistant **takes the result back** from that nested exchange and returns the final answer to the original student.

### 6.4 Key Structural Difference Between Swarm's Approach and AutoGen's Nested-Chat Approach

- **Shared mechanism:** both approaches use a **function call** to hand off a task to another agent.
- **Key difference:** AutoGen's approach (as illustrated by the math example) uses **nested chats** — meaning the original chat can create a sub-chat, get a result back from it, and then **return control to the original chat** once that nested exchange concludes.
- **Swarm's approach, by contrast, is a linear/one-way chain of transitions:** once one agent hands off to a next agent, that next agent **takes over completely and does not automatically return** to the prior agent. That new agent, in turn, has the option to hand off *again* to a third agent, and so on — a chain of one-directional transitions, each agent deciding (based on available tools/functions) whether and where to hand off next, rather than AutoGen's nested "call and return" structure.

### 6.5 Replicating Swarm's Pattern Within AutoGen

Chi Wang notes that Swarm's specific handoff pattern **can be replicated within AutoGen** using the **group chat + finite-state-machine-style transition condition** approach described in Section 5.3 — specifically referencing a pattern/feature AutoGen calls **"StateFlow."** In StateFlow, the speaker order can be specified based on a **user-defined function**; if that function is designed so that, whenever an agent invokes a function call that returns the identity of another specific agent, the system simply **selects that returned agent as the next speaker** (without needing to separately deliberate about who should speak next) — this achieves the same practical effect as Swarm's handoff mechanism, implemented via AutoGen's more general orchestration primitives.

### 6.6 A Broader Framing: "Flow Control" / "Processing Control" as a Unifying Concept

Jason draws a broader connective observation: across both the **lower-level, single-agent-style frameworks** discussed in prior sessions (where an "LLM router" decides which skill/branch of processing to invoke) and these **higher-level multi-agent frameworks** (where a function call on a specific agent, or a group chat manager, decides where to hand off next), there is a shared underlying concept sometimes called **"flow control"** or **"processing control"** — the general question of **how and where data/execution gets routed next.** Whether that decision is made via a single-agent LLM router, an agent's own handoff function call, or a group chat manager's speaker-selection logic, it is fundamentally the same category of design problem, and this area is expected to be a continuing focus of both research and practical tooling development.

---

## 7. Worked Example: Rebuilding a Simple Assistant in AutoGen (Two Iterations)

### 7.1 The Example Assistant

The example used throughout this comparison: a simple assistant equipped with **three tools/tasks** it can perform — specifically, tools involving a **RAG-style document-query capability**, a **data analysis capability**, and a **SQL generation-and-execution capability** (the same overall example agent concept used in the earlier "Comparing Agent Frameworks" session in this series, applied here to a multi-agent context).

### 7.2 First Iteration: A "Lower-Level-Framework Mindset" Applied to AutoGen

**Approach taken:** because the team had just spent significant time building this same style of agent using lower-level frameworks (LangGraph, LlamaIndex Workflows), their **first pass at porting it into AutoGen carried over that same mental framing** — they created a single **"user proxy"** agent and a single **"assistant"** agent, and gave that one assistant **all three tools** directly (query-docs, data-analysis, SQL-generation), with each of those tools itself internally containing LLM calls (e.g., the query-docs tool contained a full RAG pipeline; the data-analysis tool was itself an LLM call; the SQL-generation tool used an LLM to generate SQL from a table schema before executing it).

**Retrospective characterization:** this first pass is explicitly described as an **interesting near-miss** — the team, in effect, could have mapped this same problem onto a genuine **multi-agent flow** from the start, but initially approached it with an older, single-agent mental model instead, essentially underutilizing what AutoGen actually offers.

### 7.3 Second Iteration: A Genuine Multi-Agent (Group Chat) Implementation

**Approach taken:** the team rebuilt the same assistant using AutoGen's **group chat** pattern (Section 5.3), creating **individual, dedicated agents for each task**: a **SQL generation agent**, a **data analysis agent**, and a **document-querying agent** — each with its own scoped tool access, rather than bundling all three capabilities into one generalist assistant.

**Observed outcome — a concrete, directly attributable improvement:** this restructured, genuinely multi-agent version produced **noticeably better performance and better-quality responses** than the first iteration — attributed specifically to the value of **iteration among the different collaborating agents** as they worked toward a response together (echoing the general observation raised earlier in Section 4.3).

**Further potential extensions noted (not implemented in this example):** additional pieces could plausibly be added to this structure — for example, a dedicated **summarizer agent** at the end of the pipeline, or other more specialized roles — depending on further refinement needs.

### 7.4 Reflection: A Genuine Mindset Shift Was Required

The team candidly notes that building effectively with AutoGen required a **deliberate shift in mindset** — from thinking in terms of a single agent equipped with many tools, toward thinking natively in terms of **multi-agent flows and structures.** Notably, it took **one full round of building it "the wrong way" first** to arrive at the realization of how the problem should actually be structured for AutoGen — after which, the benefits of the multi-agent approach became apparent quite quickly. It's explicitly confirmed that the original single-agent-style version *did* work and produced legitimate responses — but those responses were **meaningfully lower quality** than what the properly multi-agent-structured version was able to produce.

---

## 8. Observability and Tracing for Multi-Agent Systems

### 8.1 General Approach

The team's tracing was performed using **Arize Phoenix** (Arize's open-source observability tool), which has direct integrations for visualizing AutoGen traces specifically.

### 8.2 Example Trace 1: A Simple Calculator Agent

A straightforward example trace shows the flow of a basic calculator-capable agent: constructing the relevant LLM calls, determining which tool to invoke, actually invoking that tool, and eventually producing a final response — presented as a simple, illustrative baseline example of what tracing an agent's flow looks like.

### 8.3 Example Trace 2: The Multi-Agent Assistant (SQL + Data Analysis Flow)

For a query such as *"What trends do you see in my traces table?"*, the trace shows the full multi-agent flow: generating and executing a SQL query (in this specific illustrative trace, humorously noted to have selected "the whole table"), then handing off to conduct data analysis on the retrieved results. The trace reveals **the underlying chain of LLM calls** — for example, the first agent's tool-usage decision, then the second agent generating and executing the actual SQL query, and so on.

**A notable structural observation about how multi-agent traces differ from the single-agent/lower-level-framework traces seen in prior sessions:** rather than seeing the same kind of explicit step-by-step "logic" transitions moving between clearly defined steps (as in LangGraph/Workflows-style single-agent traces), a multi-agent trace like this one is better understood as a **chain of LLM calls passing back and forth** between different agents.

### 8.4 The Practical Value of Tracing During Debugging

Beyond the specific trace examples shown, the presenters emphasize directly that having tracing available was **genuinely helpful during their own debugging process** — specifically for figuring out **where differences in results came from** across the different iterations/versions they built, and for identifying what specifically was fixed between iterations.

---

## 9. Evaluation Approaches for Multi-Agent Systems

### 9.1 General Philosophy: Decompose, Then Evaluate Components

The team's general evaluation approach: **decompose the overall system into a set of components/concerns you care about** — sometimes breaking things down at a granular, low-level component basis — and then build evaluations targeting **how well the system performs on each individual skill/task.**

### 9.2 Types of Evaluations

- **LLM-as-a-judge** — described as a common, widely-used approach.
- **Code-based evaluations** — used depending on how straightforward a given check is to verify programmatically.

**General framing offered:** "evals" are, at their core, simply a "fancy word" for **performance tests** — and, just as with any system, the goal is to identify the specific things worth testing, rather than assuming a single universal evaluation methodology applies everywhere.

### 9.3 Prioritizing Where to Build Evaluations First

An important practical piece of guidance: **you do not need to build evaluations for every part of your system uniformly.** In cases observed with real customers, sometimes just **one specific component** (e.g., SQL generation, in one cited example) was disproportionately important to get right — and was also a component that was seeing a disproportionate number of mistakes. **Recommendation:** prioritize building evaluations first around the skill(s)/task(s) that matter most for a given use case, rather than attempting comprehensive coverage everywhere from the outset.

### 9.4 Evaluating the Router Specifically (Cross-Reference to Prior Sessions)

Consistent with the "Agent Architectures" and "Evaluating Agents" sessions earlier in this series, router-specific evaluations were again highlighted as centering on two key sub-questions: (1) **did the router correctly identify the right intent** (i.e., choose the correct skill/agent to invoke), and (2) **did it correctly extract the right parameters** needed for that call. These are generally referred to as **function-calling evaluations**.

---

## 10. Interoperability Questions: Can LlamaIndex Workflows Be Used With AutoGen?

**Question raised directly.** Response:
- **LlamaIndex *agents*** (as a component/unit) can definitely be used within AutoGen — a concrete example was cited of embedding a LlamaIndex agent to perform RAG, alongside several other native AutoGen agents, all interoperating together within a group chat.
- **However, regarding "state"/memory specifically:** Chi Wang notes this is a **more specific question he did not have a confident answer to** at the time.
- **John's perspective on the state/memory interoperability question:** since AutoGen has its own memory concept (context passed between steps, conceptually parallel to "state" in the lower-level frameworks), and since both frameworks are ultimately **storing similar kinds of information**, it should in principle be possible to **manually reconcile the two objects** at different points in an application, even without a built-in, automatic bridging mechanism between them.
- **Chi Wang's elaboration — a practical litmus test for interoperability:** if a piece of state can plausibly be **"wrapped" as an agent** itself (for example, defining a dedicated agent whose specific job is state management, which then communicates that shared state to other agents), then that state can likely be integrated relatively smoothly into an AutoGen-based system. If, on the other hand, the state is a **highly specialized object type** that other agents wouldn't inherently understand how to consume directly, integrating it becomes considerably more difficult.
- **A broader, closing observation on interoperability:** because the fundamental inputs/outputs of many of these tools are simply **text**, there is a genuine, underlying promise of interoperability across frameworks — for instance, potentially being able to package an agent built in one framework as a usable "tool" within a different multi-agent framework.

---

## 11. Contextual Design Considerations: How Much Information/Context Should Each Agent Have?

### 11.1 Question: Should Different Agents Have Access to Different Levels of Information? (e.g., Separate Retrievers Per Agent)

This is explicitly framed as one of the **key reasons optimal multi-agent workflow design is genuinely difficult** — developers must actively decide how much context/information each individual agent should have access to.

**Context-sharing implications of different AutoGen conversation structures:**
- **Group chat:** every participating agent **shares the same context** by default.
- **Nested chat:** creates **natural context separation** between different chats — the "super agent" containing a nested chat has the ability to extract/aggregate information from within that nested chat, but agents **outside** that nested structure do not automatically have visibility into what happened inside it; they can only consume whatever information the super agent chooses to explicitly share back out.
- **Sequential chat:** only certain **explicitly defined "carryover" information** is retained/passed from one chat/stage to the next, rather than full, unrestricted context sharing.

**A key, nuanced trade-off identified: more context is not automatically better.** It is genuinely **not always obvious** whether giving an agent more context will help or distract it. **General guidance offered for less capable/weaker agents specifically:** to help such an agent succeed at its own specific task, it is often best to give it **only the minimal necessary context/information**, actively removing distracting or irrelevant information, since excess context can measurably hurt task performance for a less capable agent.

**A candid, forward-looking caveat:** it remains an open question whether sufficiently long-context, more capable future models will eventually make this concern less relevant (i.e., letting a sufficiently strong model simply "pick out" the relevant information on its own, even from a large, noisy context). At the time of this session, however, even models with very long context windows are **not necessarily reliably strong at picking out the correct information** from within that large context — meaning excess context can still measurably hurt performance with **current-generation models**, even if this constraint may loosen somewhat as model capability continues to improve. Until (and unless) that underlying model-level limitation is fully resolved, **deliberate, careful context management remains necessary** at the application-design level.

### 11.2 A Related Question: Should Each Agent Have Its Own Dedicated RAG/Retrieval Access?

**Analogy offered in the question:** similar to giving each individual employee their own reference manual specific to their job function.

**Response:** this connects directly to the context-management discussion in Section 11.1 — the underlying reasoning is the same: avoid confusing a given agent, or adding unnecessary noise/context, beyond what is relevant to its specific job. This is offered as **one of the reasons it can be beneficial to combine multiple AutoGen conversational patterns together** within a single overall system — for example:
- Using an **"outer" group chat** so that a defined sub-team of agents shares information amongst themselves, without necessarily exposing that same information to every other agent in a larger system (e.g., high-level "manager" agents overseeing several smaller, more specialized teams, where the managers don't need full visibility into every detail handled within each smaller team).
- Using a **sequential chat as one step within a larger group chat**, and so on.

**General conclusion:** these different conversational/organizational patterns can be **combined in many different ways**, and the right combination should be chosen based on whatever best fits the specific needs of a given application — there is no single universally correct structure.

---

## 12. Chi Wang's Broader Perspective: The State of Multi-Agent Research and Open Problems

### 12.1 Are the Essential "Building Blocks" for Multi-Agent Systems Already Established?

Chi Wang's assessment: once a framework has standardized support for the key organizational **patterns** discussed above (two-agent chat, sequential chat, group chat, handoff-style transitions, etc.), the **next major bottleneck shifts to developers correctly choosing the right patterns and composing them together** to solve a specific task in the *best* possible way — where "best" is explicitly multi-dimensional, including:
- High success rate.
- Low cost.
- Low latency.
- Debuggability / ease of maintenance.
- (And other relevant dimensions, depending on the application.)

**An explicit analogy drawn to traditional machine learning:** this is compared directly to the long-standing problem, in traditional ML, of **choosing the right model and the right hyperparameters** for a given application — i.e., an optimization/search problem over a defined space of choices, not a one-size-fits-all answer.

**A direct, confident assessment on whether the "essential building blocks" already exist:** having examined recent progress (citing both AutoGen's own development and, more recently, OpenAI's Swarm) against this framing, Chi Wang states a genuine belief that the field is likely **already close to having the essential building blocks** needed to represent the most advanced current multi-agent patterns — i.e., that recent state-of-the-art multi-agent approaches largely turn out to be representable using the same small set of core underlying building blocks already established.

### 12.2 A Framing for the Field Going Forward: "Architecture" vs. "Organization"

Chi Wang offers a specific two-part framing (directly tied to the session's stated title, "Architecture and Organization"):
- **"Architecture"** refers to the **basic building blocks** themselves (the core patterns/primitives available).
- **"Organization"** refers to **how one navigates the large space of possible combinations of those building blocks** to find an optimal solution for a specific task.

**Forward-looking expectation:** as "architecture" (the building blocks themselves) becomes increasingly mature and stable, Chi Wang expects the field's focus to increasingly shift toward **optimization within the "organization" space** — i.e., systematically searching for or discovering better combinations/configurations of these established building blocks, rather than continuing to invent fundamentally new primitives.

### 12.3 Trade-offs Introduced by Multi-Agent Approaches (Summary Discussion)

Building on Chi Wang's framing, the presenters (Jason and John) summarize the core practical trade-off of multi-agent systems as follows:

- **The advantage:** multi-agent approaches enable **compartmentalizing a problem**, potentially achieving genuine iteration/improvement in results (as concretely demonstrated in Section 7.3), and allowing focused agents to specialize on the specific skill sets/roles best suited to their sub-task — along with potential benefits like a degree of fault tolerance from having multiple collaborating components.
- **The corresponding disadvantage:** these same benefits come bundled with genuinely **unsolved, harder problems** — specifically around **handoffs/coordination, maintaining control**, and **reliably knowing when a multi-agent process is truly "done."**
- **A specific technical elaboration on control/debugging challenges (from John):** because each agent can independently use **different parameters and potentially different underlying models**, multi-agent systems introduce a **high number of distinct points where non-deterministic responses can occur**, each of which must be handled robustly. There are also correspondingly **more "knobs" to turn/tune** in pursuit of optimal performance. **Tracing is helpful here** (as demonstrated in Section 8), but a large amount of practical effort still tends to go into **fine-tuning exact prompt wording** for individual agents, and figuring out the right settings for termination-related parameters (e.g., appropriate cycle/turn counts) for a given application.

### 12.4 Question: Is a No-Code/Visual UI for Building Multi-Agent Systems Likely to Emerge?

Acknowledged as an area of active interest across the industry — the presenters note being aware of several different existing approaches/levels of visual tooling already emerging elsewhere in the ecosystem (without committing to a specific timeline or claim about AutoGen's own plans in this specific area).

---

## 13. Key Takeaways

1. **AutoGen operates at a genuinely higher level of abstraction than single-agent-focused frameworks like LangGraph or LlamaIndex Workflows** — its core unit of design is *how multiple defined agents interact*, rather than the precise step-by-step internal logic of a single agent.
2. **AutoGen's explicit design philosophy centers on simple, human-team-like abstractions that still permit arbitrarily complex applications**, serving both beginner (high-level) and advanced (low-level, fine-grained-control) use cases.
3. **Termination conditions for multi-agent conversations range from simple fixed turn limits to natural-language-based, LLM-judged completion signals recognized by a receiving agent** — and this same general pattern generalizes across different conversational structures.
4. **AutoGen supports several distinct organizational patterns** — two-agent chat, sequential (pipeline-like) chat, and group chat (with a dedicated group chat manager acting as a "super router") — each suited to different degrees of structure vs. autonomy.
5. **A concrete, directly observed benefit of genuine multi-agent structuring (vs. a single generalist agent with many tools) was materially improved response quality**, attributed to the value of iterative back-and-forth exchange between specialized agents — though this benefit came only after a deliberate mindset shift toward "thinking in multi-agent terms" rather than porting over single-agent habits.
6. **OpenAI's Swarm implements a linear, one-directional "handoff" pattern via function calls**, distinct from AutoGen's earlier nested-chat approach (which supports returning control to an originating chat after a sub-chat concludes) — though Swarm's specific pattern can be replicated within AutoGen's more general group chat + StateFlow-style transition mechanism.
7. **How much context/information to share with each individual agent is a nuanced, genuinely difficult design decision** — more context is not automatically better, particularly for less capable agents/models, and different AutoGen conversational structures (group chat, nested chat, sequential chat) offer different natural levels of context isolation that can be deliberately leveraged.
8. **The number of agents (or tools) a manager/router can effectively handle depends on model capability and role distinctiveness among agents** — with hierarchical routing offered as a concrete mitigation strategy when agents' roles are similar or numerous.
9. **Multi-agent systems trade increased flexibility, iteration benefits, and specialization for increased complexity in control, debugging, and termination management** — a fundamental, currently still partially unsolved trade-off actively being researched.
10. **Chi Wang's assessment is that the field may already be close to having established the essential "building block" patterns for multi-agent systems**, with the frontier of future work increasingly shifting toward the harder problem of *optimally composing and configuring* those already-established building blocks for specific applications (an "organization," rather than "architecture," problem) — directly analogous to model/hyperparameter selection in traditional machine learning.

---

## 14. Glossary of Key Terms

| Term | Definition |
|---|---|
| **AutoGen** | A multi-agent framework (from Microsoft, created by Chi Wang) enabling developers to define individual agents and the organizational patterns by which they communicate and collaborate. |
| **User proxy agent** | An AutoGen agent type that does not perform task-processing itself, but instead solicits and relays human input into a conversation; can also refer more generally to a non-LLM, rules-based agent. |
| **Group chat manager** | A specialized AutoGen orchestration agent that dynamically decides which agent should speak next within a group chat, functioning as a kind of "super router." |
| **Sequential chat** | An AutoGen organizational pattern resembling a defined pipeline, where control moves from one (possibly multi-agent) stage to the next at defined points. |
| **Nested chat** | An AutoGen pattern in which one chat can spawn a sub-chat (e.g., between specialized "expert" agents); once the sub-chat concludes, control and results return to the originating chat. |
| **StateFlow** | An AutoGen feature/pattern allowing the speaker order in a group chat to be governed by a user-defined function, enabling deterministic, handoff-style transitions (e.g., replicating OpenAI Swarm's handoff behavior) within AutoGen's more general orchestration framework. |
| **Handoff (Swarm)** | OpenAI Swarm's pattern in which an active agent, via a function call, transfers control to a next agent, which then takes over completely (a one-directional chain of transitions, without automatic return to the prior agent). |
| **Termination condition** | The mechanism (fixed turn limit, or natural-language/LLM-judged signal recognized by a receiving agent) by which a multi-agent conversation determines it should stop. |
| **Flow control / processing control** | A general, cross-cutting term for the design problem of determining how and where execution/data should be routed next — applicable whether implemented via a single-agent LLM router, an agent's own handoff function call, or a group chat manager's speaker-selection logic. |
| **Hierarchical routing** | A strategy for managing large numbers of similar agents/tools by first routing among a small number of well-separated high-level categories, then routing again within the selected category — reducing confusion compared to a single flat list of many similar options. |
| **Architecture vs. organization** (Chi Wang's framing) | A two-part conceptual distinction: "architecture" refers to the core building blocks/patterns available for multi-agent systems; "organization" refers to the process of navigating the space of possible combinations of those building blocks to find an optimal solution for a specific task. |

---

*End of reference document for the "Multi-Agent Frameworks / AutoGen" masterclass session. Companion reference documents in this series cover "Agent Architectures," "Comparing Agent Frameworks," "Evaluating Agents," "Agent Looping," and the "LlamaIndex Workflows" masterclass session.*
