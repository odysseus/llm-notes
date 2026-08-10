---
title: "Paper Guide: ReAct"
type: paper-guide
status: active
updated: 2026-08-08
paper_year: 2023
tags: [tools, tool-use, agents, reasoning, prompting, action-observation, foundational-papers]
---

# Paper Guide: *ReAct*

**Paper:** Shunyu Yao et al., 2023; first released as a preprint in 2022  
**Full title:** *ReAct: Synergizing Reasoning and Acting in Language Models*  
**Primary link:** [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)  
**Original problem:** Language models had shown useful reasoning behavior and useful action generation, but those abilities were usually studied separately  
**Why it matters:** ReAct demonstrated a simple way to place reasoning, tool calls, and environmental feedback in one repeated trajectory, helping establish the observation–action loop that underlies many modern LLM agents.

## The paper in one sentence

ReAct showed that a prompted language model can alternate between language-based reasoning, actions in an external environment, and returned observations, allowing new evidence to guide later reasoning and allowing reasoning to guide later actions.

## The practical idea in one view

Before ReAct, two common patterns looked roughly like this:

```text
Reasoning only:
question -> internal reasoning in text -> answer

Acting only:
goal -> action -> observation -> action -> observation -> finish
```

ReAct combines them:

```text
goal
  -> reason about the next useful step
  -> take an action
  -> receive an observation
  -> revise the working understanding
  -> take the next action
  -> ...
  -> finish
```

In the paper, these stages were written explicitly as **Thought**, **Action**, and **Observation**. The model generated thoughts and actions. The external environment generated observations.

That distinction is fundamental:

- A **thought** changes the model's working context but not the outside world.
- An **action** requests information or changes something outside the model.
- An **observation** reports what actually happened or what the tool returned.

ReAct's main contribution was not a new model architecture or a new training objective. Its primary experiments used a frozen language model with a few worked examples in the prompt. The novelty was the **trajectory format and control pattern**: reasoning and interaction were allowed to inform each other repeatedly while the task was underway.

## Background topics

- **Autoregressive language models:** Models that generate one token at a time from the preceding context.
- **In-context learning:** Inducing a task from instructions or examples in the prompt without changing model weights.
- **Chain-of-thought prompting:** Eliciting intermediate reasoning text before a final answer.
- **Tool use:** Letting a model request external retrieval, computation, observation, or action.
- **Agents:** Systems that choose actions over multiple steps in pursuit of a goal.
- **Environment:** The external system that accepts actions and returns observations.
- **Policy:** A rule or model that selects the next action from the current context.
- **Trajectory:** The ordered record of a goal, intermediate reasoning, actions, observations, and final result.
- **Closed-loop control:** Choosing the next step using feedback from earlier steps rather than committing to the whole sequence in advance.

These topics are useful, but the paper can be understood through one practical question:

> What if a language model could pause its reasoning, inspect the world, and then continue from what it actually found?

## Before ReAct: reasoning and acting were usually separate

ReAct joined two research lines that had been developing alongside one another.

### Reasoning without interaction

The [chain-of-thought paper](https://arxiv.org/abs/2201.11903) showed that large language models could solve some multi-step problems more successfully when a prompt included intermediate reasoning examples. Instead of jumping directly from a question to an answer, the model produced a sequence of explanatory steps.

This was valuable because intermediate text could function as a temporary scratchpad. The model could decompose a problem, carry forward partial results, and make its route to the answer easier to inspect.

But a normal chain-of-thought trajectory is closed with respect to the outside world:

```text
question -> model-generated reasoning -> model-generated answer
```

If an early step contains a false fact, later steps may build on it. The model cannot search for evidence, test an assumption, inspect a database, or see whether an attempted action succeeded. More reasoning can elaborate the original mistake just as readily as it can correct it.

### Acting without explicit high-level reasoning

Other systems used language models to select browser actions, robot plans, game commands, or API calls. [WebGPT](https://arxiv.org/abs/2112.09332), for example, trained a model to interact with a text-based web browser while researching answers.

These systems supplied a connection to an environment:

```text
goal -> action -> observation -> action -> observation
```

But selecting actions directly from a growing interaction history can be difficult. The model may need to remember the goal, decide which subgoal is complete, infer why an action failed, and reformulate the plan. An action-only trajectory has no dedicated place to write down those intermediate judgments.

### The missing connection

Reasoning-only methods could organize a solution but could not obtain new evidence. Acting methods could obtain evidence and affect an environment but often lacked a flexible language workspace for interpreting feedback and managing a plan.

ReAct's proposal was straightforward: put both capabilities in the same sequence.

```text
reason to decide what to do
    -> act to obtain evidence or change the environment
    -> reason from the resulting observation
```

The paper describes this in both directions:

- **Reasoning supports acting** by decomposing goals, choosing actions, tracking progress, and handling exceptions.
- **Acting supports reasoning** by retrieving facts, revealing state, and testing whether the current understanding is correct.

The importance lies in the loop, not merely in the presence of both ingredients.

## What problem does the loop solve?

The ReAct pattern addresses several related limitations.

### 1. The model's parameters are not the current world

A language model's learned parameters contain statistical knowledge from training. They do not automatically contain a current Wikipedia page, the state of a simulated kitchen, the contents of a user's task database, or the result of the last API call.

Actions let the system obtain such information when it becomes relevant. Observations then place that information into the next model context.

### 2. A plan formed before execution may be wrong

Suppose a system plans:

```text
1. Search for the product.
2. Open the first result.
3. Select the blue option.
4. Finish.
```

The first search might return nothing. The first product might not satisfy the user's constraints. The blue option might not exist. A plan generated all at once has no opportunity to respond.

ReAct instead delays each decision until the relevant observation exists:

```text
search -> inspect results -> choose next step
```

This is the difference between an **open-loop plan** and a **closed-loop policy**.

### 3. Raw interaction history is hard to interpret

An observation may contain many details but only one relevant change. A reasoning step can extract that change and turn it into working state:

```text
Observation: Search returned twelve products, but only result 4 is under $140
             and has both drawers and a nickel finish.

Working conclusion: Result 4 is the only candidate satisfying all stated constraints.
```

The reasoning text does not create new external evidence. It organizes the evidence already returned so that the next action is better targeted.

### 4. Errors need a recovery point

When an action fails, the system needs to recognize the failure and choose a different path. ReAct provides a natural place for that recovery:

```text
Action: search[Front Row]
Observation: No matching page; several unrelated results returned.
Reasoning: The name is ambiguous. Search for "Front Row media center software".
Action: search[Front Row media center software]
```

The first result changes the next query. This is more capable than retrieving once before generation and more robust than pretending the first action succeeded.

### 5. The trajectory becomes inspectable

Because actions and observations are recorded, a developer can often tell whether failure came from:

- a poor plan;
- the wrong tool;
- malformed arguments;
- an unhelpful tool result;
- failure to interpret a useful result;
- repeated actions;
- or an incorrect final answer.

This does not make every model-generated explanation faithful to the model's internal computation. It does make the **system-level sequence of calls and returned evidence** observable, which is crucial for debugging.

## The core mechanism

### The original formal idea

An agent normally selects an external action from an action space $A$. ReAct expands the available choices to include language-based thoughts from a space $L$:

$$
\hat{A} = A \cup L
$$

An item from $A$ affects or queries the environment. An item from $L$ is a reasoning step that only updates the textual context.

This notation can sound more complicated than the idea is. Imagine a task with these external actions:

```text
search[query]
lookup[term]
finish[answer]
```

ReAct adds one more kind of move:

```text
think[text]
```

The model can therefore choose either to organize what it already knows or to request new evidence. Both kinds of move become part of the history used to choose the following step.

### The context grows over time

At step $t$, the next decision is conditioned on the trajectory so far:

```text
original goal
+ previous reasoning
+ previous actions
+ returned observations
```

Conceptually:

$$
a_t \sim \pi(a_t \mid c_t)
$$

where $c_t$ is the current context and $a_t$ may be either a thought or an external action.

The important change is that $c_t$ contains real feedback. The model is not merely extending its own earlier prose; it is also responding to information supplied by the environment.

### Few-shot trajectories teach the pattern

The main ReAct experiments did not retrain the large model. The prompt contained a small number of human-written example trajectories. Each example showed:

1. what a useful reasoning step looked like;
2. which actions were valid;
3. how observations appeared;
4. how to continue after an observation; and
5. how to signal completion.

The model then imitated that pattern on a new task.

This makes ReAct a form of [prompt engineering](prompt-engineering.md) and in-context learning in its original form. The paper also included a smaller fine-tuning experiment, but the widely adopted ReAct pattern came from the prompted loop.

## A paper-style example

The question-answering experiments gave the model a deliberately small Wikipedia interface:

- `search[entity]` returned the opening sentences of a matching page or suggested similar pages.
- `lookup[string]` found successive sentences containing a term on the current page.
- `finish[answer]` ended the task.

Consider a simplified multi-hop question:

> The inventor of the World Wide Web was born in which country?

A ReAct-style trajectory could look like this:

```text
Reasoning: I need to identify the inventor, then find the inventor's birthplace.

Action: search[World Wide Web inventor]

Observation: The World Wide Web was invented by Tim Berners-Lee.

Reasoning: Now I need his place of birth and the corresponding country.

Action: search[Tim Berners-Lee]

Observation: Tim Berners-Lee was born in London, England.

Reasoning: London is in England, which is part of the United Kingdom. The expected
answer is probably the United Kingdom.

Action: finish[United Kingdom]
```

This example shows three separate functions of the reasoning steps:

- decomposing a multi-hop question;
- turning the first observation into the next retrieval goal;
- resolving how the retrieved wording maps to the requested answer.

The tools supply the factual bridge. The model decides which bridge is needed and how to use it.

## Dense reasoning and sparse reasoning

The paper did not use exactly the same rhythm for every domain.

### Dense alternation for knowledge questions

For HotpotQA and FEVER, the prompts commonly alternated every stage:

```text
Thought -> Action -> Observation -> Thought -> Action -> Observation
```

This made sense because each retrieval result could change the next query, and the tasks involved relatively few steps.

### Sparse reasoning for long interactive tasks

ALFWorld tasks could require dozens of environment actions. Writing a new reasoning paragraph before every small movement would be wasteful. Instead, the demonstrations used thoughts at important moments:

- when decomposing the goal;
- when choosing where to search;
- when a subgoal was completed;
- when the plan needed to change;
- when an action failed;
- or when the final state needed checking.

The rest of the trajectory could contain consecutive environment actions.

This is an important practical lesson. ReAct does not require maximum visible deliberation. It requires enough working reasoning to improve decisions. The useful frequency depends on the task.

Too little reasoning can make the agent forget its goal. Too much reasoning increases tokens, latency, and opportunities to wander.

## How ReAct changes the LLM workflow

### A normal generation

A simple LLM application often performs one model call:

```text
instructions + user message + context
    -> model
    -> final text
```

Even if the generated text contains several reasoning steps, the external application sees one input and one completed output.

### A ReAct workflow

ReAct turns the application into an iterative runtime:

```text
instructions + user goal + trajectory
    -> model proposes next action
    -> application parses and validates it
    -> tool or environment executes it
    -> observation is appended to the trajectory
    -> model is called again
    -> repeat until finish or a stopping rule
```

The model is now only one participant in a larger system. The complete behavior emerges from the interaction among:

- the model;
- the prompt or policy;
- the available action space;
- the external tools;
- the current environment state;
- and the application [harness](harness-engineering.md).

### The runtime, not the model, performs actions

The model emits text or structured data representing a proposed action. Software outside the model must:

1. recognize the proposal;
2. confirm that the tool exists;
3. validate the arguments;
4. check permissions;
5. execute the operation;
6. capture success or failure;
7. return a bounded observation; and
8. decide whether another step is allowed.

This division is sometimes obscured by saying that “the LLM used a tool.” More precisely:

```text
model proposes -> harness authorizes and executes -> environment responds
```

ReAct supplies the decision pattern. It does not remove the need for execution infrastructure.

## A framework-neutral implementation sketch

The essential loop can be implemented without a specialized agent framework:

```python
def run_react(goal, model, tools, max_steps=8):
    trajectory = [{"role": "user", "content": goal}]

    for step in range(max_steps):
        decision = model.next_step(
            trajectory=trajectory,
            tool_schemas=tools.schemas(),
        )

        if decision.kind == "final":
            return decision.answer

        call = tools.validate(decision.tool_call)
        tools.authorize(call)

        observation = tools.execute(call)

        trajectory.append({
            "role": "assistant",
            "tool_call": call,
            "state_summary": decision.state_summary,
        })
        trajectory.append({
            "role": "tool",
            "content": observation,
        })

    raise StepLimitReached(trajectory)
```

This differs from the paper's literal text format in several ways:

- calls are structured rather than embedded in free-form text;
- validation and authorization are explicit;
- the loop has a step limit;
- the example stores a compact state summary rather than requiring a detailed public reasoning trace;
- and failure is returned as an observation rather than silently ignored.

Those changes preserve the observation–action structure while adapting it to production engineering.

## Why observations are the decisive ingredient

It is tempting to summarize ReAct as “chain of thought plus tools.” That is incomplete. The most important feature is not that both appear somewhere in the prompt. It is that **tool results enter the context before the next decision**.

Consider three architectures:

### Tool call after all reasoning

```text
make full plan -> call tool once -> answer
```

The tool cannot correct the earlier plan until the end.

### Retrieve before reasoning

```text
retrieve documents -> reason over them -> answer
```

This can work well, but the first retrieval query is chosen before the model learns anything from the results.

### ReAct

```text
reason -> retrieve -> inspect -> reformulate -> retrieve -> answer
```

The retrieval process itself becomes adaptive. One observation can reveal the entity, date, location, or error needed to formulate the next request.

This is why ReAct is especially relevant to multi-hop questions, exploration, troubleshooting, and tasks whose necessary steps cannot be known in advance.

## The experiments

The paper evaluated four domains. They were chosen to test two different sides of the idea:

- **Knowledge-intensive reasoning:** Can external actions ground and improve reasoning?
- **Interactive decision making:** Can language reasoning improve sequences of actions?

### HotpotQA: multi-hop question answering

HotpotQA questions often require joining information from more than one source. The model received only the question and used the small Wikipedia interface to gather evidence.

With PaLM-540B, the reported exact-match scores were:

| Method | HotpotQA exact match |
|---|---:|
| Standard prompting | 28.7 |
| Chain of thought | 29.4 |
| Chain of thought with self-consistency | 33.4 |
| Acting only | 25.7 |
| ReAct | 27.4 |
| CoT-SC with ReAct fallback | 34.2 |
| ReAct with CoT-SC fallback | 35.1 |

ReAct beat acting alone, which supports the claim that reasoning helps guide retrieval and synthesize an answer. But ReAct by itself did **not** beat chain of thought on this benchmark. The strongest results came from combining externally grounded ReAct trajectories with the more flexible internal reasoning of self-consistent chain of thought.

That mixed result is important. The paper did not show that every reasoning problem should be turned into a tool loop. Tool interaction can improve factual grounding while simultaneously imposing a structure that makes some reasoning harder.

### FEVER: fact verification

FEVER asks whether a claim is supported, refuted, or lacks sufficient evidence. The reported accuracies were:

| Method | FEVER accuracy |
|---|---:|
| Standard prompting | 57.1 |
| Chain of thought | 56.3 |
| Chain of thought with self-consistency | 60.4 |
| Acting only | 58.9 |
| ReAct | 60.9 |
| CoT-SC with ReAct fallback | 64.6 |
| ReAct with CoT-SC fallback | 62.0 |

Here ReAct slightly outperformed chain of thought and acting alone. Accurate retrieval is especially valuable when a claim differs from the truth by one name, date, or relation.

Again, the best result was a hybrid. When the chain-of-thought samples did not agree confidently, the system switched to ReAct and gathered external evidence.

### What the error analysis found

The authors manually examined samples of HotpotQA trajectories. The comparison suggested a real tradeoff:

- Chain of thought produced more failures attributed to hallucinated facts.
- ReAct was more grounded in retrieved information.
- ReAct also produced more reasoning errors, including repeated thoughts and actions.
- Unhelpful search results were a significant ReAct failure source.

This is one of the paper's most useful findings. Tools change the error distribution; they do not simply remove errors.

Without retrieval, the model may invent a fact. With retrieval, it may choose a poor query, receive irrelevant evidence, misread the evidence, or become stuck trying the same action. The system becomes more grounded but also more dependent on the quality of its interaction policy and tools.

### Fine-tuning experiment

The main results used prompting, but the paper also generated 3,000 correct ReAct trajectories and fine-tuned smaller PaLM models on them.

Prompting smaller models with the ReAct format worked poorly because they had to infer both the reasoning style and the action protocol from a few examples. After fine-tuning, however, ReAct became the strongest of the compared trajectory formats in that experiment. The reported result suggests that a useful interaction pattern demonstrated through prompting can later become training data for a more specialized model.

This is a secondary contribution, not the paper's central method. ReAct itself is a trajectory pattern; it can be induced by prompting, learned through fine-tuning, or implemented using a combination of model training and runtime control.

### ALFWorld: acting in a simulated household

ALFWorld is a text-based environment in which an agent navigates rooms and manipulates household objects. A goal might require finding an object, cleaning or heating it, and placing it in a particular location.

The challenge is not factual recall alone. The agent must:

- decompose a high-level instruction;
- remember which subgoals are complete;
- infer likely object locations;
- respond when an action fails;
- and maintain the correct sequence over many steps.

Across six prompt variants, the best ReAct prompt achieved a 71% success rate, compared with 45% for the best acting-only prompt and 37% for the strongest reported BUTLER baseline. The average ReAct result was 57%.

The gap between average and best prompt also matters: performance was affected by which demonstrations appeared in context. ReAct was strong in this environment, but prompt composition remained part of the system's behavior.

### WebShop: navigating a shopping environment

WebShop required the model to search a simulated commerce site, inspect products, select options, and choose an item satisfying a natural-language request.

The reported results were:

| Method | Average score | Success rate |
|---|---:|---:|
| Acting only | 62.3 | 30.1% |
| ReAct | 66.6 | 40.0% |
| Imitation learning | 59.9 | 29.1% |
| Imitation + reinforcement learning | 62.4 | 28.7% |
| Human | 82.1 | 59.6% |

Sparse reasoning helped the model connect a user's constraints to noisy product descriptions and option labels. ReAct improved the success rate by about ten percentage points over the strongest listed learned baseline.

The model still remained far below human performance. The authors observed that people explored more products and reformulated searches more effectively. ReAct provided a useful loop, but the base model and a single demonstration did not supply a complete search strategy.

## What the results actually established

The paper supported several claims well.

### Reasoning can improve action selection

On ALFWorld and WebShop, adding sparse language reasoning to comparable action trajectories improved performance. The reasoning gave the model a place to represent subgoals, constraints, likely locations, and progress.

### External observations can reduce some hallucinations

In the knowledge tasks, retrieved Wikipedia evidence made ReAct trajectories more fact-grounded than reasoning from model parameters alone. The effect was clearest when exact facts were central to the task.

### Prompted models can perform closed-loop interaction

Only a few example trajectories were needed to induce multi-step behavior in very large language models. This was an important demonstration that in-context learning could specify not only a text transformation, but also an interactive policy.

### Reasoning and acting are complementary, not interchangeable

The acting-only baseline could obtain information but struggled to synthesize and plan. The reasoning-only baseline could produce coherent solutions but hallucinated facts. ReAct combined their advantages, though not without new constraints.

### Hybrid systems may beat a universal policy

The best knowledge-task results came from switching between ReAct and chain-of-thought self-consistency. This is an early example of a broader architectural lesson: the best system may route among several reasoning modes instead of forcing every request through one loop.

## What the results did not establish

ReAct did not demonstrate that:

- tool-using agents are generally reliable;
- visible reasoning is always correct or causally faithful;
- ReAct always outperforms chain-of-thought prompting;
- a few examples are sufficient for smaller models or large action spaces;
- unrestricted web or computer access is safe;
- tools eliminate hallucination;
- long trajectories remain reliable as the number of steps grows;
- model-generated actions should execute without validation;
- the agent can maintain durable state between separate runs;
- benchmark success transfers directly to consequential real-world actions;
- or the textual `Thought/Action/Observation` syntax is the only way to implement the pattern.

The experiments used controlled interfaces and bounded environments. The Wikipedia API exposed only search, lookup, and finish. The WebShop environment did not make real purchases. Those restrictions made the central interaction pattern easier to study and limited the possible harm.

## ReAct compared with neighboring approaches

### Chain of thought

Chain of thought gives the model extra language space for intermediate reasoning:

```text
question -> reasoning -> answer
```

ReAct adds external feedback:

```text
question -> reasoning -> action -> observation -> revised reasoning -> answer
```

Use plain reasoning when the needed information is already present and the problem is mainly deductive or computational. Use a ReAct-like loop when the next useful step depends on information that must be retrieved, measured, or observed.

### Retrieval-augmented generation

A simple [RAG](retrieval-augmented-generation.md) pipeline retrieves documents once and supplies them before generation:

```text
query -> retrieve -> generate
```

ReAct can make retrieval iterative:

```text
query -> retrieve -> inspect -> reformulate -> retrieve -> generate
```

RAG is usually cheaper and easier to control when one retrieval pass is enough. ReAct earns its extra calls when initial results reveal what must be searched next.

### Plan then execute

A planner can generate a complete sequence before tools run:

```text
plan all steps -> execute them -> report
```

This is efficient when the workflow is stable. ReAct plans incrementally and can recover from unexpected observations. Its advantage increases with environmental uncertainty; its cost increases with every extra model call.

### Deterministic workflow

Code can define the exact order of operations:

```text
validate input -> search database -> write record -> verify record
```

This is usually preferable when the sequence is known, compliance matters, or every request must follow the same procedure. ReAct is valuable when natural-language interpretation or observations determine which branch comes next.

### Toolformer

[Toolformer](paper-toolformer.md) and ReAct are often grouped together because both connect language models to external tools. They solve different primary problems.

| Question | ReAct | Toolformer |
|---|---|---|
| Main contribution | An interleaved reasoning–action–observation trajectory | A method for generating and filtering tool-use training data |
| Original adaptation method | Few-shot prompting of a frozen large model | Fine-tuning on automatically augmented text |
| Typical interaction | Multiple adaptive steps | Primarily single inline calls in the paper |
| Supervision signal | Human-written example trajectories | Tool results that reduce future-token prediction loss |
| Central practical lesson | Let observations guide later decisions | Learn when a tool result helps language generation |

Toolformer asks, “How can a model teach itself useful call behavior at scale?” ReAct asks, “How should reasoning and environmental interaction alternate while solving a task?” A modern tool-using model can be trained with Toolformer-like data and run inside a ReAct-like loop.

### Structured function calling

Function calling defines how a model represents a tool request, often with a name and schema-validated arguments. ReAct defines how repeated requests and observations fit into a task-solving policy.

They are complementary:

```text
ReAct-like control loop
    + structured calls
    + runtime validation
    + bounded observations
```

Using JSON instead of `Action: search[...]` does not stop a system from being ReAct-like. Conversely, supporting function calls does not automatically give a system planning, recovery, or useful stopping behavior.

### Reflection and critique loops

Later approaches added explicit self-critique, memory, or reflection after failure. These may improve recovery, but they also add more model-generated text and calls. ReAct itself already provides the essential feedback channel: an observation can trigger a revised decision. More elaborate reflection should be added only when evaluation shows that the basic loop cannot recover reliably.

## When a ReAct-like loop is useful

ReAct is a strong candidate when several of the following are true:

- The answer depends on information outside the current context.
- The correct next action depends on the previous tool result.
- Retrieval may need query reformulation.
- The environment can reject, fail, or partially complete an action.
- The task requires exploration rather than one known request.
- A high-level goal must be decomposed into changing subgoals.
- The system needs an auditable record of actions and observations.
- Tool results are cheap and safe enough to obtain iteratively.

Typical examples include:

- multi-hop research;
- troubleshooting an unfamiliar system;
- navigating a website or software interface;
- investigating a codebase and running tests;
- querying several data sources until an entity is resolved;
- or completing a task in an environment whose state changes after each action.

## When another approach is better

### One known tool call

If every request should run the same operation, call it directly. A weather question that always maps to `get_weather(location)` does not need an open-ended agent loop.

### Stable multi-step workflow

If the application always performs validation, lookup, write, and verification in that order, encode those stages in code. Use the LLM only for ambiguous fields.

### Pure reasoning over supplied information

If all facts are already in context, additional retrieval can add latency and distract the model. A bounded reasoning or verification strategy may be enough.

### High-risk action without a narrow policy

Open-ended ReAct is a poor default for sending money, deleting data, publishing content, or changing permissions. A model may help prepare a proposal, but deterministic authorization and user approval should control execution.

### Very long tasks without state management

Appending every action and observation indefinitely causes context growth. Long-running work needs explicit state, summaries, checkpoints, and often a workflow engine. ReAct is a local control pattern, not a complete persistence architecture.

## Failure modes

### Repeating the same step

The paper observed loops in which the model repeated thoughts or actions after failing to make progress. A production runtime should detect repeated calls, unchanged observations, and repeated state summaries.

Useful controls include:

- maximum step counts;
- maximum repetitions of the same normalized call;
- a no-progress detector;
- explicit error observations;
- and a fallback or escalation route.

### Poor search or tool arguments

A correct decision to use search can still produce a weak query. If the tool returns irrelevant evidence, later reasoning may become worse rather than better.

Evaluate argument quality separately from tool selection. The model may need examples of query reformulation, not merely examples of calling `search`.

### Misreading the observation

The tool may return the right fact while the model extracts the wrong one. Grounding therefore requires more than access. The system must preserve provenance and test whether the final claim is actually supported by the observation.

### Treating untrusted output as instructions

Web pages, emails, documents, and database fields may contain text that attempts to redirect the model. An observation is data from the environment, not a new source of authority.

The harness should label tool output as untrusted, restrict which instructions can alter behavior, and keep permissions outside the model's context-based judgment.

### Hallucinating actions or results

A model may write as though a tool ran even when it did not. The runtime must distinguish:

```text
proposed call
executed call
returned result
verified state change
```

Only the latter three are evidence about the world.

### Premature completion

The model may issue a final answer before gathering enough evidence or verifying a side effect. Completion should often depend on application state:

```text
not "the model says the task is done"
but "the required postcondition is true"
```

### Excessive deliberation

The loop can spend many tokens restating the plan. Sparse reasoning, compact state summaries, and deterministic branches can reduce cost without removing the ability to respond to observations.

### Context bloat

Every observation increases the trajectory. Large pages, logs, or database results can crowd out the original goal. Tools should return bounded, structured information, and the application should maintain explicit state outside the prompt.

### Unfaithful reasoning traces

The paper presented human-readable reasoning as an interpretability advantage. That advantage should be understood carefully. Generated rationales can be useful for diagnosis, but later research has shown that chain-of-thought explanations are not guaranteed to reveal the actual causes of a model's answer.

For operational systems, the most dependable audit trail is:

- which action was proposed;
- which action was authorized;
- what actually executed;
- what observation was returned;
- which state changed;
- and what final evidence supported completion.

Detailed private deliberation need not be exposed to users or stored as though it were a factual explanation.

## Safety and authority boundaries

The ReAct paper explicitly recognized that connecting a model to an environment introduces risks. Its experiments reduced those risks by using constrained action spaces: Wikipedia lookup, simulated household actions, and a shopping environment that could not complete real purchases.

A production system needs equally explicit boundaries.

### Read and write tools are different

Searching a catalog and purchasing an item should not share the same approval policy. Reading a task list and deleting a task should not be equally easy.

### The model proposes; the application authorizes

The model can interpret intent and choose a candidate action. It should not be the sole authority on whether a consequential action is permitted.

### Observations require provenance

The model should know which tool produced a result, when it was obtained, and whether execution succeeded. The user-facing answer should not blur model inference with externally verified facts.

### Side effects need postconditions

For a write operation, the return value should confirm what changed. Where practical, the application should reread state and verify the intended postcondition.

These controls belong to [harness engineering](harness-engineering.md). ReAct explains how decisions can adapt to feedback; the harness determines what actions exist and what may execute.

## Evaluating a ReAct system

Final-answer accuracy is necessary but insufficient. A useful evaluation separates the trajectory into stages.

| Stage | Evaluation question |
|---|---|
| Need recognition | Was an external action actually necessary? |
| Tool selection | Did the model choose the appropriate capability? |
| Argument construction | Were the query or parameters correct and complete? |
| Execution | Did the tool run successfully within limits? |
| Observation use | Did the next step interpret the result correctly? |
| Recovery | Did the system change strategy after failure? |
| Completion | Was the final answer supported, or was the desired state reached? |
| Efficiency | How many calls, tokens, seconds, and dollars were used? |
| Safety | Were permissions, approvals, and data boundaries respected? |

Trajectory-level metrics can include:

- task success rate;
- unsupported-claim rate;
- unnecessary-call rate;
- malformed-call rate;
- tool error rate;
- recovery rate after an injected failure;
- repeated-action rate;
- average steps to completion;
- context and token growth;
- and verified side-effect accuracy.

A ReAct architecture should be compared against simpler baselines under a similar resource budget:

1. direct answer;
2. one retrieval pass;
3. fixed workflow;
4. plan then execute;
5. ReAct-style adaptive loop.

If the adaptive loop does not improve task success enough to justify its cost and additional failure modes, it is not the right architecture for that task.

## Application to the task-and-idea chatbot

For the task-and-idea application, a ReAct-like loop is useful when the user's message requires both interpretation and inspection of stored state.

Suppose the user says:

> Add “compare cabinet-refinishing quotes,” unless I already saved basically the same idea.

A useful trajectory is:

```text
Goal: Avoid creating a duplicate while preserving the user's new idea.

Action: search_events(query="cabinet refinishing quotes")

Observation: One event exists: "Research whether to refinish kitchen cabinets."
             It has no quote-comparison subtask.

State summary: The existing event is related but does not capture comparing quotes.
               Add the new item as a child or explicit subtask rather than a duplicate root event.

Action: create_event(
    title="Compare cabinet-refinishing quotes",
    parent_id="event_123",
    status="idea"
)

Observation: Event event_456 created under event_123.

Final: I added “Compare cabinet-refinishing quotes” under your existing cabinet-refinishing idea.
```

The first observation changes the correct write operation. A fixed “always create a root event” path would produce a poorer result.

However, the application should not make every capture request an open-ended investigation. Its MVP uses a **save-first** policy for ordinary ideas. A simple message such as:

> Remember to call the contractor.

can follow a deterministic path:

```text
parse skeletal event -> validate -> create -> verify -> confirm
```

ReAct should be reserved for requests in which stored state or an external result materially changes what action should follow.

This yields a hybrid architecture:

```text
simple capture or update
    -> deterministic workflow

ambiguous retrieval-dependent task
    -> bounded ReAct-like loop

consequential external action
    -> proposal + explicit approval + deterministic execution
```

That is more reliable than treating “agentic” as the default for every request.

## Practical lessons for developers

### Design the action space before the prompt

A model cannot select clean actions from a vague or overlapping tool set. Give each tool a distinct purpose, clear arguments, bounded outputs, and explicit side-effect semantics.

### Use observations as typed state, not prose whenever possible

The original paper encoded everything in text because it was demonstrating a prompting method. Production systems often benefit from structured observations:

```json
{
  "status": "not_found",
  "query": "Front Row",
  "suggestions": ["Front Row software", "Front Row Motorsports"]
}
```

Structured results make error handling and evaluation easier while remaining readable to the model.

### Make failures visible

Timeouts, permission denials, empty results, validation errors, and partial writes should become explicit observations. Silent failure encourages the model to continue from an imagined result.

### Bound the loop

Set limits on steps, cost, elapsed time, repeated calls, observation size, and side effects. Define what happens at the limit: answer with partial evidence, ask the user, switch strategies, or hand off for review.

### Preserve the goal separately

Do not rely on a long trajectory to keep the original request salient. Store the goal, constraints, and current subgoal in explicit application state and reconstruct the model context deliberately.

### Use sparse reasoning

Require reasoning at decision points, not before every mechanical action. A concise state summary can often supply the needed continuity without an elaborate rationale.

### Verify outcomes externally

For retrieval, check source support. For code, run tests. For a database write, inspect the resulting record. For a workflow, test the postcondition. A convincing final sentence is not evidence that the environment changed.

### Compare against a simpler baseline

ReAct is most valuable where observations genuinely alter the next decision. If the same sequence works for nearly every request, encode that sequence directly.

## What aged well

- The observation–action loop became a standard model for tool-using LLM applications.
- Tool results are routinely returned to models as new context for subsequent decisions.
- Adaptive retrieval is now a common alternative to one-shot RAG.
- Action traces and environment observations remain essential for debugging agents.
- Separating reasoning-only, acting-only, and combined baselines remains a useful evaluation method.
- Sparse planning and state tracking are still valuable in long interactive tasks.
- ReAct's strongest architectural lesson—reasoning and acting should correct one another—remains broadly applicable.

## What changed

- Tool calls are now commonly represented with structured schemas rather than free-form `Action:` strings.
- Many models receive tool-use training, so few-shot examples are not always needed to teach basic call syntax.
- Production harnesses add permissions, approvals, retries, timeouts, idempotency, and durable state.
- Long-running agents often separate working state from the raw conversation transcript.
- Detailed reasoning traces may remain private; visible action rationales or state summaries can provide safer diagnostics.
- Agent systems frequently combine deterministic workflow stages with bounded model-directed loops.
- Evaluation increasingly considers cost, safety, trajectory quality, and verified state changes in addition to final-answer accuracy.

The modern form of ReAct is therefore often architectural rather than textual. A system can preserve the repeated cycle of **decide, act, observe, and update** without copying the paper's exact prompt format.

## Reading advice

For a first reading of the original paper, prioritize:

1. **Introduction and Figure 1:** The clearest comparison among standard prompting, chain of thought, acting only, and ReAct.
2. **Section 2:** The formal idea of adding language thoughts to the agent's action space.
3. **Section 3.1:** The intentionally limited Wikipedia action interface.
4. **Table 1:** The mixed knowledge-task results and the strength of hybrid methods.
5. **Table 2:** The shift from hallucination failures toward reasoning, search, and loop failures.
6. **Section 4 and Tables 3–4:** The strongest evidence that sparse reasoning helps interactive action.
7. **Appendix C:** The actual prompts, which show how few-shot trajectories specified the behavior.
8. **Conclusion and ethics statement:** The context-window, data, and action-safety limitations recognized by the authors.

The exact prompt wording is useful for historical understanding. The lasting idea is more general: do not require the model to finish reasoning before it can observe the consequences of its actions.

## Recap

ReAct joined two capabilities that had mostly been studied separately. Chain-of-thought prompting gave language models a textual workspace for decomposing problems, but the resulting reasoning remained isolated from external evidence. Action-generating systems could browse or manipulate an environment, but often lacked a flexible place to represent plans, subgoals, and interpretations of feedback.

The paper combined them in a repeated trajectory. The model generated a reasoning step and an action; the environment returned an observation; and the next decision used the entire history. Reasoning could determine what evidence to seek, while evidence could correct or redirect later reasoning.

The experiments support a conditional conclusion. ReAct consistently improved over acting alone. It reduced some factual hallucinations by grounding trajectories in Wikipedia. It produced large gains in the ALFWorld and WebShop interactive environments. Yet it did not uniformly beat chain of thought on knowledge questions, and it introduced failures involving poor search, constrained reasoning, and repeated loops. The strongest knowledge-task results combined ReAct with another reasoning method.

ReAct is therefore best understood as a control pattern, not a complete agent architecture. It changes the LLM workflow from a single generation into a bounded observation–action loop. Modern systems keep that loop while adding structured tool calls, explicit state, validation, permissions, stopping rules, and external verification. The model helps decide what to do next; the harness remains responsible for what is allowed to run and whether the goal was actually achieved.

## Primary and supporting sources

- Yao, S., et al. (2023). [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629). The primary paper; first released in 2022 and published at ICLR 2023.
- [Official ReAct repository](https://github.com/ysymyth/ReAct). Prompts, notebooks, environment wrapper, and reproduced GPT-3 results.
- Wei, J., et al. (2022). [*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*](https://arxiv.org/abs/2201.11903). The reasoning-only prompting lineage used as a major baseline.
- Nakano, R., et al. (2021). [*WebGPT: Browser-Assisted Question-Answering with Human Feedback*](https://arxiv.org/abs/2112.09332). An influential predecessor for language-model interaction with an information environment.
- Schick, T., et al. (2023). [*Toolformer: Language Models Can Teach Themselves to Use Tools*](https://arxiv.org/abs/2302.04761). A complementary training-oriented approach to learned tool use.
- Turpin, M., et al. (2023). [*Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*](https://arxiv.org/abs/2305.04388). Evidence for treating generated rationales cautiously rather than as guaranteed access to a model's causal reasoning.
