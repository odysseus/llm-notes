---
title: "Task Fine-Tuning and Task Specialization"
type: concept
status: active
updated: 2026-08-13
tags: [fine-tuning, specialization, adaptation, peft, prompts, tools, harnesses, evaluation]
---

# Task Fine-Tuning and Task Specialization

**Central idea:** Task specialization means adapting an LLM system to perform a narrower class of work reliably. The specialization may be encoded **inside the model** through training, or **around the model** through prompts, examples, retrieved context, tools, routing, workflows, and harness rules. These approaches solve different problems and are often strongest in combination.

**Why it matters:** A general-purpose LLM may understand a request yet still perform poorly in a real application. It may use the wrong labels, lack private knowledge, make arithmetic mistakes, call an unsafe tool, forget application state, or stop before the requested outcome exists. Fine-tuning can improve some of these failures, but not all. Choosing the correct location for specialization is therefore more important than choosing a fashionable training method.

## Background topics

- **Pretraining:** Learning broad language patterns, knowledge, and capabilities from large corpora.
- **Supervised learning:** Updating model parameters from input–output examples.
- **Instruction tuning:** Training across natural-language requests so a model generalizes its instruction-following behavior.
- **In-context learning:** Temporarily adapting behavior using demonstrations in the current context.
- **Parameter-efficient fine-tuning:** Training a small set of task parameters while keeping most base weights frozen.
- **Retrieval-augmented generation:** Supplying current or private evidence at inference time.
- **Tool use:** Connecting the model to external computation, data, and actions.
- **Harness engineering:** Controlling context, permissions, execution, state, validation, recovery, and stopping around model calls.
- **Evaluation and distribution shift:** Measuring whether a system works on representative cases, including cases unlike its training examples.

These topics describe different places where specialized behavior can be created. Task specialization is the architectural decision of which places should carry which responsibilities.

## Before general-purpose language models

For much of NLP history, the task and the model were tightly coupled. Sentiment classification, translation, named-entity recognition, document retrieval, question answering, and summarization typically used different training pipelines and often different model architectures. A classifier returned one of a fixed set of labels; an extractive question-answering model selected a span; a translation system produced a target-language sequence.

This provided strong task boundaries. A sentiment classifier could not suddenly decide to browse the web or draft an email. The application selected a specialized model, and the model's architecture and training objective constrained what it could produce.

Large-scale pretraining weakened this coupling. Models such as BERT could be pretrained once and then fine-tuned for many downstream tasks. Each task still commonly received its own trained checkpoint, but all began from a shared general representation ([Devlin et al., 2019](https://aclanthology.org/N19-1423/)). T5 pushed further by representing many language problems through a common text-to-text interface ([Raffel et al., 2020](https://jmlr.org/papers/v21/20-074.html)).

GPT-3 then demonstrated that a frozen model could perform many tasks specified through instructions and examples in its input, without task-specific gradient updates ([Brown et al., 2020](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)). The task boundary moved again: instead of residing only in a trained checkpoint, it could be expressed through language at inference time.

Modern systems add retrieval, tools, persistent state, and application-controlled workflows. As a result, specialization is no longer synonymous with fine-tuning. A general model placed inside a narrow, well-designed environment may behave more like a dependable specialist than a fine-tuned model placed inside a vague one.

## The topic in one view

An LLM application can be approximated as:

$$
o = H(M_{\theta,\phi}, P, C, T, S, u)
$$

where:

- $u$ is the user's request;
- $M$ is the model;
- $\theta$ represents shared base-model parameters;
- $\phi$ represents optional task-specific trained parameters;
- $P$ is the prompt and demonstrations;
- $C$ is selected context and retrieved evidence;
- $T$ is the available tool interface;
- $S$ is application and workflow state;
- $H$ is the harness that controls execution;
- $o$ is the resulting answer, action, clarification, or state change.

Task specialization can modify any of these inputs. Full fine-tuning changes much of $\theta$. LoRA, adapters, and soft prompts add or learn a smaller $\phi$. Prompting changes $P$; retrieval changes $C$; tools change $T$; workflow and policy engineering change $H$ and $S$.

A useful practical approximation is:

$$
\text{specialized performance}
\approx
\text{model adaptation}
+
\text{task representation}
+
\text{external capabilities}
+
\text{runtime control}
$$

The terms are not truly additive: a tool the model cannot use is worthless, and a capable model cannot retrieve data from a database it cannot access. The equation emphasizes that specialization is produced by the complete system.

## First diagnose what is missing

“The model is not specialized enough” can describe several different failures.

| Missing ingredient | Example failure | Natural specialization layer |
| --- | --- | --- |
| **Task behavior** | Uses the wrong label boundary repeatedly | Examples, task fine-tuning, or PEFT |
| **Instruction following** | Does not reliably interpret requests as tasks | Instruction tuning or a stronger instruct model |
| **Domain language** | Misunderstands specialized terminology | Continued pretraining, retrieval, or domain examples |
| **Current/private knowledge** | Does not know today's policy or the user's records | Retrieval or database tools |
| **Exact computation** | Makes arithmetic or date-calculation mistakes | Calculator or deterministic code |
| **Output structure** | Produces prose where typed fields are required | Structured output plus validation |
| **External action** | Cannot create a record or send a request | Tool integration |
| **Authority** | Attempts an action the user may not perform | Deterministic permission checks |
| **Persistent state** | Forgets what was created in an earlier session | Database or state store |
| **Process reliability** | Loops, stops early, or mishandles retries | Harness and workflow design |

This table supplies the article's main rule: **place the specialization where the missing information or guarantee naturally belongs**. Do not encode a changing price list into model weights when it can be retrieved. Do not use a longer prompt to simulate a calculator. Do not train “ask before deleting” and then omit the authorization check.

## Approaches that change the model

Training-time specialization changes durable behavior. The adaptation remains available across requests without repeatedly placing all training examples in the context window.

### Full supervised fine-tuning

Full fine-tuning begins from pretrained parameters $\theta_0$ and optimizes most or all of them on a task dataset:

$$
\theta^* = \underset{\theta}{\operatorname{argmin}}
\sum_{(x,y) \in D_{task}}
-\log p_{\theta}(y \mid x)
$$

For a generative model, each example usually contains an input and desired target sequence. For classification, the target may be a label or a short textual representation of one. The resulting checkpoint is optimized toward the distribution represented by $D_{task}$.

Full fine-tuning is attractive when:

- the task is stable and clearly defined;
- a sufficiently large, representative dataset exists;
- the same behavior will be invoked frequently;
- latency or context costs make long prompts undesirable;
- maximum task performance matters more than preserving one universal checkpoint;
- the organization can train, host, version, and evaluate model variants.

It carries important costs. Training and optimizer state consume substantial memory. A separate checkpoint may be required for every specialist. Poor data can teach shortcuts or stylistic artifacts. Narrow training can degrade unrelated capabilities or produce **catastrophic forgetting**, in which later learning damages earlier behavior ([Luo et al., 2023](https://arxiv.org/abs/2308.08747)).

Fine-tuning also cannot create runtime access. A model trained on yesterday's inventory still does not know today's inventory unless the application retrieves it. A model trained to emit database commands still does not possess database credentials or proof that a command succeeded.

### Continued pretraining and domain adaptation

Continued pretraining exposes a pretrained model to additional unlabeled or self-supervised text from a target domain. The objective may remain next-token prediction or masked-language modeling rather than direct task supervision.

This approach is useful when the problem is broad domain familiarity rather than one exact input–output mapping. Legal, biomedical, scientific, financial, or organization-specific corpora may contain terminology, discourse patterns, and relationships underrepresented in general pretraining. Gururangan et al. distinguished **domain-adaptive pretraining** from **task-adaptive pretraining**, showing benefits from continued pretraining on domain and task distributions in their experiments ([Gururangan et al., 2020](https://aclanthology.org/2020.acl-main.740/)).

Continued pretraining teaches statistical patterns in a corpus. It does not inherently teach the model to obey a new tool protocol, return a specific schema, or follow a user-facing workflow. It is often followed by supervised task or instruction tuning.

It should also not be the default way to provide frequently changing facts. Retrieval keeps source material inspectable, updateable, and attributable; parametric knowledge is harder to correct and audit.

### Narrow task-specific fine-tuning

Narrow fine-tuning optimizes one known behavior: ticket classification, query rewriting, entity extraction, reranking, domain summarization, or a fixed response transformation.

Its strength is consistency on a defined distribution. A small specialized model may be cheaper, faster, and easier to evaluate than asking a frontier model to perform the same high-volume classification through a lengthy prompt. The output space can be kept intentionally small.

Its weakness is brittleness beyond that distribution. A ticket classifier trained for six categories should not be assumed to understand a new seventh category. A model trained only on complete records may invent values when fields are missing. The task definition must include ambiguity, abstention, unknown values, and expected shifts—not only ideal examples.

### Domain or task instruction tuning

[Instruction tuning](instruction-tuning.md) is supervised fine-tuning on request–response examples. Its breadth determines its purpose.

- **Task-specific tuning** teaches one narrow mapping.
- **Domain instruction tuning** teaches several request types within a domain.
- **General instruction tuning** aims to make natural-language instructions a transferable interface across domains.

FLAN showed that training on many instructed tasks could improve zero-shot performance on unseen task types ([Wei et al., 2021](https://arxiv.org/abs/2109.01652)). For a specialist, the objective may be narrower: recognize a domain's requests, preserve its terminology, ask appropriate clarifying questions, and express results through its preferred interface.

Breadth introduces a tradeoff. A very narrow dataset can overfit to templates; a broad mixture can dilute the behavior that matters most. Training mixtures therefore need explicit sampling weights, source tracking, and separate evaluations for general capability and specialist performance.

### Parameter-efficient fine-tuning

Parameter-efficient fine-tuning, or **PEFT**, keeps most shared weights frozen and learns a smaller task-specific component.

#### Adapters

Adapter methods insert small trainable modules between or within existing model layers. Each task stores its own adapter while the base model remains shared. Houlsby et al. demonstrated this pattern across NLP tasks, obtaining performance near full fine-tuning in their BERT experiments while adding a small fraction of task-specific parameters ([Houlsby et al., 2019](https://proceedings.mlr.press/v97/houlsby19a.html)).

#### LoRA

Low-Rank Adaptation freezes an existing weight matrix $W$ and learns a low-rank update:

$$
W' = W + \Delta W = W + BA
$$

where $A$ and $B$ have rank $r$ much smaller than the full dimension. This reduces the number of trainable parameters and makes task variants easier to store or swap ([Hu et al., 2022](https://openreview.net/forum?id=nZeVKeeFYf9)).

#### Soft prompts and prefixes

[Soft prompt tuning](soft-prompting.md) leaves the base weights frozen while learning continuous prompt vectors. Prefix-tuning injects learned task vectors more deeply into attention computations. These approaches can be extremely parameter-efficient, but the learned artifacts are not human-readable and remain coupled to a particular base model and implementation.

PEFT reduces trainable and stored parameters; it does not remove the need for training data, optimization, versioning, or evaluation. It may also require infrastructure access that a hosted inference-only API does not expose.

### Task heads and small specialists

Not every task needs a generative LLM. A frozen encoder plus a classification head, a dedicated reranker, an embedding model, a conventional parser, or a smaller distilled model can be the appropriate specialist.

This is especially true when:

- outputs belong to a small fixed set;
- response latency is critical;
- deterministic evaluation data exists;
- explanations are not needed;
- the task occurs at high volume;
- a smaller model can meet the required quality.

The general LLM can remain the conversational interface while delegating a bounded subtask to the specialist. This produces modular specialization rather than forcing every behavior into one generative checkpoint.

### Task-specific preference optimization

Supervised examples teach what a plausible answer looks like. Preference optimization teaches which of several plausible answers should be favored. A domain might prefer concise answers, conservative abstention, a particular escalation style, or stronger adherence to evidence.

Preference data can refine specialist behavior, but it is easy to optimize presentation while neglecting correctness. A fluent preferred response may still contain the wrong event ID. Outcome-based tests and domain assertions must remain separate from preference judgments.

## Approaches that do not change the model

Non-weight specialization leaves the selected model checkpoint unchanged. It adapts the information, capabilities, and execution environment supplied at runtime.

### Hard prompts and explicit task contracts

[Prompt engineering](prompt-engineering.md) is the fastest and most reversible form of specialization. A prompt can define:

- the task and success criteria;
- domain terminology;
- expected inputs and outputs;
- decision boundaries;
- examples of correct and incorrect behavior;
- how uncertainty should be represented;
- when the model should ask for clarification.

For example:

```text
Interpret the message as exactly one of:
- capture: create a new skeletal event;
- retrieve: find existing events;
- update: propose changes to an identified event;
- converse: answer without changing stored state.

Never invent a missing date. Use null for unknown values.
```

Prompts are ideal while the task is changing or while data is scarce. They are inspectable and can be updated immediately. Their costs recur on every call, and behavior may vary with model versions, wording, order, and context competition.

Prompting should establish a baseline before fine-tuning. Without that baseline, it is impossible to know whether training produced enough improvement to justify its expense.

### In-context examples

[In-context learning](in-context-learning.md) specializes a frozen model temporarily using examples placed in the current context. Demonstrations can teach label meanings, local terminology, boundary cases, and output style without a training pipeline.

A fixed example set is simple and reproducible. Dynamically retrieved examples can better match the current request but introduce selection errors and possible exposure to stale or untrusted material. Examples also consume context space and inference tokens on every request.

In-context examples are particularly useful for testing a proposed task definition. If reviewers cannot agree on which examples are correct, the project is not yet ready for training.

### Context engineering and retrieval

[Context engineering](context-engineering.md) assembles the working information for one model call. It can retrieve:

- current domain documents;
- user or application records;
- relevant conversation history;
- approved policies and procedures;
- prior examples similar to the request;
- current tool results.

Retrieval-augmented generation combines parametric model behavior with external non-parametric memory ([Lewis et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)). In application architecture, the practical advantage is not merely additional facts. Retrieved knowledge can be updated without retraining, cited to a source, restricted by access policy, and inspected when the answer is wrong.

Retrieval specializes **what the model knows for this request**, not necessarily how well it follows a domain workflow. Poor retrieval, missing sources, or excessive context can still produce failure. Selection and model use should be evaluated separately.

### Structured output

[Structured output](structured-output.md) specializes the interface between the model and software. A schema can restrict fields, types, enumerations, and required alternatives. Constrained decoding can prevent syntactically invalid structures.

This changes the output space without changing the model weights. It is especially important when the model's interpretation becomes input to another system.

Schema validity is not semantic validity. A generated `event_id` can be a valid string but refer to no event. The application must still resolve identifiers, check dates, preserve uncertainty, and reject invalid field combinations.

### Tools and specialized external capabilities

[Tool use](tool-use.md) allows a general language model to call systems that are already specialized:

- a calculator performs exact arithmetic;
- a database provides authoritative records;
- a search engine supplies current information;
- a compiler or test runner verifies code;
- a calendar API exposes schedules;
- a domain service applies business rules.

ReAct demonstrated an inference-time pattern that interleaves model reasoning, actions, and observations ([Yao et al., 2023](https://openreview.net/forum?id=WE_vluYUL-X)). Toolformer explored training a model to decide when and how to call external tools ([Schick et al., 2023](https://arxiv.org/abs/2302.04761)). These represent two separable layers: the tool supplies capability; prompting or training teaches the model to use it.

Tool design itself is specialization. A narrow `update_event(event_id, changes)` interface is easier to use and authorize than unrestricted database access. SWE-agent provided evidence that changing the agent–computer interface can materially change coding-agent performance without changing the underlying task definition ([Yang et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html)).

### Harness and workflow specialization

[Harness engineering](harness-engineering.md) specializes how work proceeds. The harness can determine:

- which model, prompt, records, examples, and tools are selected;
- which actions require user confirmation;
- how arguments and business invariants are validated;
- how tool results return to the model;
- which state persists across calls;
- when to retry, clarify, escalate, or stop;
- how success is verified;
- what traces are recorded for evaluation.

This layer is the right place for guarantees that must hold even when the model fails. “Ask before deleting” can appear in training examples and prompts, but the delete operation should still require a valid confirmation in code.

Harness specialization often produces large gains because it makes the environment legible and constrains the task. A model solving “update this record through one typed transaction and verify it” faces a narrower problem than a model told vaguely to “manage the user's plans.”

### Deterministic workflows and state machines

A general model need not choose every step. A fixed workflow can place model judgment only where language interpretation is needed:

```text
receive message
  → classify intent
  → load candidate records
  → extract a typed proposal
  → validate deterministically
  → ask for confirmation if required
  → execute transaction
  → reload and verify state
  → generate grounded confirmation
```

This is a form of task specialization even if the model is unchanged. The workflow narrows the model's responsibility and makes completion observable. [LLMs as state machines](llms-as-state-machines.md) explains why durable state should belong to the application rather than be inferred from a transcript.

### Dynamic routing

[Dynamic selection](dynamic-selection.md) routes different requests to different models, contexts, prompts, tools, or workflows. A small classifier may handle common extraction; a stronger model may receive difficult ambiguity; a research workflow may activate only when external evidence is required.

Routing lets one application behave like a family of specialists without maintaining a single model trained for every case. It adds its own failure mode: misclassification before the main task. The router must therefore be evaluated as part of the complete system.

## Comparing specialization strategies

| Strategy | Changes weights? | Best for | Main cost or risk | Update speed |
| --- | --- | --- | --- | --- |
| Full fine-tuning | Yes, most or all | Stable task behavior and maximum specialization | Training cost, checkpoints, forgetting | Slow |
| Continued pretraining | Yes | Domain language and distributions | Data/compute; facts hard to audit | Slow |
| LoRA/adapters | Adds small trained parameters | Modular task variants | Model coupling and training operations | Moderate |
| Soft prompts | Learns prompt vectors | Parameter-efficient conditioning | Low interpretability and API support | Moderate |
| Hard prompts/examples | No | Rapid behavior definition and prototyping | Fragility and recurring tokens | Fast |
| Retrieval/context | No | Current, private, attributable knowledge | Retrieval errors and context cost | Fast |
| Structured output | No | Reliable machine-readable interfaces | Does not ensure correct values | Fast |
| Tools | No model change | Exact computation, live data, actions | Integration and security complexity | Fast to moderate |
| Harness/workflow | No model change | Permissions, state, recovery, verified completion | Engineering complexity | Fast to moderate |
| Routing | No required change | Heterogeneous workloads | Misrouting and observability | Fast to moderate |

The “weights?” column can be misleading for PEFT. LoRA, adapters, and soft prompts often leave the shared base weights frozen, but they still require gradient-based training and produce learned task parameters. Operationally, they belong closer to fine-tuning than to ordinary prompting.

## How the layers combine

The strongest specialist is often layered rather than pure.

Consider a support assistant:

1. A broadly instruction-tuned base model understands user requests.
2. A domain LoRA improves ticket classification and response conventions.
3. Retrieval supplies current product documentation and customer records.
4. Narrow tools query subscriptions and create support actions.
5. Structured output carries proposed actions to the application.
6. The harness checks permissions, requires approval for credits, executes changes, and verifies outcomes.

No single layer is responsible for everything. The trained component improves repeated domain behavior; retrieval supplies changing facts; tools provide effects; the harness provides authority and reliability.

The same principle applies when no training is used. A capable general model with strong examples, a narrow tool interface, and a deterministic workflow may outperform a poorly integrated fine-tuned model.

## A decision framework

### 1. Define the task and outcome

Specify the input distribution, expected outputs, allowed uncertainty, forbidden actions, and observable success condition. “Be better at customer support” is not trainable or evaluable. “Classify these five ticket types with an abstain option” is.

### 2. Establish a no-training baseline

Use the strongest reasonable existing model with a concise prompt, representative examples, correct context, schemas, and tools. This reveals whether the underlying capability already exists.

### 3. Classify the failures

Separate:

- missing or incorrect context;
- misinterpreted task boundaries;
- invalid output shape;
- weak reasoning or domain capability;
- wrong tool selection;
- permission or execution failure;
- state or recovery failure.

Only some of these are model-adaptation problems.

### 4. Fix the cheapest correct layer

- Add retrieval for missing current facts.
- Add a calculator for exact arithmetic.
- Add schema constraints for parse failures.
- Add domain tools for authoritative reads and writes.
- Add deterministic checks for permissions and invariants.
- Add examples for unclear decision boundaries.
- Consider PEFT or fine-tuning for stable residual behavior failures.

“Cheapest” includes maintenance and risk, not only development time. A brittle prompt patched dozens of times may be more expensive than a small trained classifier.

### 5. Train only on a stable contract

Once task rules and examples have stopped changing, construct a representative dataset with provenance, quality review, unknown cases, adversarial cases, and held-out evaluations. Split by meaningful units such as user, document, time period, or organization—not merely random rows—when leakage would otherwise make results misleading.

### 6. Compare complete systems

Compare the fine-tuned candidate with the prompt-only baseline under the same tools, retrieval, and harness. Record quality, latency, cost, reliability, and operational complexity. A two-point accuracy gain may not justify a new model-serving path; a smaller specialist that cuts latency by tenfold might.

## Data design for task fine-tuning

Fine-tuning data is an executable description of the desired task distribution. Its composition matters more than raw row count.

### Include the real decision boundaries

If the task distinguishes “create a new event” from “update an existing event,” include near-neighbor examples. Easy positive examples teach less than contrasts that force the intended distinction.

### Preserve unknowns and abstention

If information may be missing, teach the model to return `null`, `unknown`, or a clarification request. Otherwise every training row implicitly says a complete answer exists.

### Represent tool observations accurately

If the model will receive structured tool results in production, training examples should use the same result shapes and error variants. Do not train on prose summaries and deploy against raw JSON without testing the shift.

### Avoid policy leakage into model authority

Examples can teach a polite confirmation behavior, but they should not be the sole enforcement mechanism. Training data may reflect policy; the harness must enforce policy.

### Track provenance and versions

Record where examples came from, which policy they represent, who reviewed them, and which model/harness version generated synthetic examples. Otherwise corrections cannot be propagated reliably.

### Keep evaluation independent

Do not use the same model, templates, or synthetic-generation procedure to create nearly identical training and evaluation sets. Hold out new phrasings, users, domains, and time periods where those shifts matter.

## Application to the task-and-idea chatbot

The single-user MVP should begin with system specialization, not model specialization.

### Recommended initial architecture

Use an existing instruction-tuned model with:

- a concise prompt defining capture, retrieval, update, and conversation;
- a small reviewed set of boundary examples;
- separate typed schemas for each operation;
- narrow tools such as `create_event`, `find_events`, and `update_event`;
- an authoritative event database;
- deterministic ID resolution, validation, and transactions;
- clarification only when a target or requested change is genuinely ambiguous;
- verification by rereading the stored record;
- traces that record prompt, model, tool calls, outcomes, latency, and corrections.

This design directly tests the product's core hypotheses: can conversation reliably create, retrieve, and update events, and does the interface feel natural?

### What might be fine-tuned later

After enough reviewed interactions accumulate, training could target narrow, measurable behaviors:

1. **Intent classification:** capture, retrieve, update, or conversation.
2. **Field extraction:** title, notes, tentative date, and explicitly unknown fields.
3. **Reference resolution:** whether a phrase refers unambiguously to a stored event.
4. **Clarification decisions:** whether the system has enough information to proceed safely.
5. **Response style:** concise confirmations grounded in actual tool outcomes.

These could use a small classifier, LoRA, or full fine-tuning depending on volume and model access. They should not train the model to become the event store or permission system.

### A staged experiment

```python
systems = {
    "baseline": BaseModel(prompt, schemas, tools, harness),
    "few_shot": BaseModel(prompt + examples, schemas, tools, harness),
    "peft": AdaptedModel(lora, schemas, tools, harness),
}

for case in held_out_cases:
    for name, system in systems.items():
        result = run(system, case)
        score(name, result, expected_database_state=case.goal_state)
```

The harness and tools remain fixed so the test isolates model adaptation. A second experiment can hold the model fixed and compare prompt, retrieval, or workflow variants. This prevents improvements from being attributed to the wrong layer.

### Likely long-term hybrid

A mature version might use:

- a small specialist for routing and field extraction;
- a stronger general model for ambiguous conversation and planning;
- retrieval over user events and preferences;
- tools for storage and later calendar integration;
- a deterministic state machine for mutations and confirmations;
- dynamic routing based on task and risk.

The conversational experience remains unified even though several specialized components cooperate behind it.

## Common failure modes

### Fine-tuning before defining the task

The dataset encodes contradictory product decisions. Training makes those contradictions harder to inspect rather than resolving them.

### Using training for changing facts

Policies, prices, schedules, or user records are baked into weights. They become stale, difficult to cite, and expensive to update. Use retrieval or tools.

### Training away an integration bug

The model appears to select the wrong event because the harness omitted candidate IDs. Fine-tuning on guessed IDs hides the missing context instead of fixing it.

### Assuming PEFT means low operational cost

The task artifact is small, but training still needs data and compute; inference may need adapter management; every base-model upgrade requires compatibility testing.

### Prompt accretion

Every failure adds another rule or example until the prompt becomes long, contradictory, and untestable. Consolidate rules, move hard constraints into code, and consider training only when the residual behavior is stable.

### Over-specialization

The specialist performs well on familiar templates but loses useful general reasoning, multilingual ability, or robustness. Test retained capabilities and out-of-distribution requests.

### Synthetic-data feedback loops

One model generates training examples, labels, and evaluation cases. The specialist learns the teacher's blind spots and appears successful under an evaluation sharing the same biases.

### Conflating schema validity with correctness

The fine-tuned model returns perfect JSON containing nonexistent identifiers or invented dates. Validate semantics against authoritative state.

### Treating tool calls as completed actions

The model proposes `update_event`; the UI reports success before execution. Tool results and verified state—not model intent—determine completion.

### Security by behavior shaping

Training and prompting teach the model to avoid unauthorized actions, but tools accept them anyway. Security boundaries must remain deterministic and least-privileged.

### Evaluating only average quality

The model improves common cases but becomes less reliable on rare destructive actions, unknown inputs, or underrepresented users. Break out high-risk and minority slices.

### Ignoring version coupling

Prompts, adapters, retrieval policies, schemas, and harness logic are tested with one base model and silently reused after an upgrade. The full configuration needs versioned evaluation.

## Evaluating specialization

A specialist should be evaluated as both a model and a deployed system.

### Model-level evaluation

- task accuracy, precision, recall, or exact match;
- calibration and abstention quality;
- field-level extraction performance;
- robustness to paraphrase, order, and formatting;
- out-of-distribution and domain-shift performance;
- retained general capabilities;
- safety and bias slices relevant to the application.

### System-level evaluation

- final external or database state;
- correct and unnecessary tool-call rates;
- permission and confirmation behavior;
- retrieval recall and evidence use;
- schema and semantic validation rates;
- recovery from timeouts, ambiguity, and partial failure;
- latency, token use, infrastructure cost, and training cost;
- reliability across repeated trials;
- user effort and unnecessary clarification.

### Ablate the layers

Run controlled comparisons:

| Comparison | What it tests |
| --- | --- |
| Base model vs. fine-tuned model, same harness | Value of weight adaptation |
| Zero-shot vs. few-shot, same model | Value of demonstrations |
| No retrieval vs. correct retrieval | Value of external knowledge |
| General tools vs. narrow tools | Value of interface specialization |
| Free-form loop vs. fixed workflow | Value of runtime control |
| One model vs. routed specialists | Value and cost of dynamic selection |

Without ablation, a project may credit fine-tuning for a gain caused by better examples, or blame a model for a missing database record.

### Evaluate the update path

Specialization must survive change. Test what happens when:

- a policy changes;
- a label is added;
- the base model is upgraded;
- the document corpus changes;
- a tool schema evolves;
- the input distribution shifts;
- an adapter is removed or replaced.

The easiest system to improve safely may be preferable to the one with the highest score on a static benchmark.

## Practical design principles

- **Specialize the system before assuming the model needs training.** A better task interface often reveals that the base model is already capable.
- **Train behavior; retrieve facts.** Stable patterns belong more naturally in parameters than frequently changing information.
- **Use tools for capabilities the model should not imitate.** Calculation, authoritative data access, and external actions should be real operations.
- **Keep authority outside the model.** Prompts and fine-tuning guide behavior; code grants permissions.
- **Prefer the smallest sufficient adaptation.** Start with prompts and examples, then PEFT or a small specialist, then full fine-tuning when evidence justifies it.
- **Treat schemas as interfaces, not truth.** Validate IDs, dates, cross-field rules, and outcomes.
- **Version every layer.** Model, adapter, prompt, examples, retriever, tools, policies, and harness jointly determine behavior.
- **Measure residual errors before training.** Fine-tune only the stable failures left after context and system defects are corrected.
- **Evaluate final state.** A fluent confirmation is not task completion.
- **Design for reversal.** Prompts and routing can change instantly; adapters can be detached; full checkpoints and baked-in knowledge are harder to unwind.

## What task specialization cannot guarantee

No specialization method guarantees truth or safety by itself. Fine-tuning can reinforce errors in its data. Retrieval can return misleading sources. Tools can contain bugs. Harness logic can enforce the wrong policy. Human reviewers can agree on an incorrect task definition.

Nor does specialization eliminate uncertainty. A narrow model may be more consistent while becoming less capable outside its task. A general model may handle unusual cases better while varying more on routine ones. A router may choose the wrong specialist. A workflow may omit a branch the designers did not anticipate.

The goal is therefore not to produce a model that “knows the task” in some absolute sense. It is to construct a system whose responsibilities are explicit, whose failures are observable, and whose behavior can be improved at the correct layer.

## Recap

Task fine-tuning is one form of task specialization. It changes durable model behavior by updating all weights or learning smaller task-specific parameters. Full fine-tuning offers strong adaptation at high operational cost; continued pretraining teaches domain distributions; instruction tuning teaches request-following behavior; LoRA, adapters, and soft prompts provide parameter-efficient variants.

Non-weight specialization changes the model's runtime environment. Prompts and examples define behavior temporarily. Retrieval supplies current or private knowledge. Structured output constrains interfaces. Tools provide exact computation, authoritative data, and actions. Harnesses and workflows control permissions, state, validation, recovery, and stopping. Routing selects among specialists.

These approaches are complements. The deciding question is not simply “Should we fine-tune?” It is:

> Which part of the system is missing the information, capability, behavior, or guarantee required by the task—and what is the least costly, most inspectable place to add it?

For new applications, begin with a strong existing model inside a narrow, evaluated system. Fine-tune when the task has stabilized, representative data exists, and controlled experiments show that persistent model adaptation improves the complete application enough to justify its cost.

## Key sources

- Devlin et al. (2019), [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://aclanthology.org/N19-1423/).
- Houlsby et al. (2019), [*Parameter-Efficient Transfer Learning for NLP*](https://proceedings.mlr.press/v97/houlsby19a.html).
- Raffel et al. (2020), [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://jmlr.org/papers/v21/20-074.html).
- Brown et al. (2020), [*Language Models Are Few-Shot Learners*](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html).
- Gururangan et al. (2020), [*Don't Stop Pretraining: Adapt Language Models to Domains and Tasks*](https://aclanthology.org/2020.acl-main.740/).
- Lewis et al. (2020), [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).
- Li and Liang (2021), [*Prefix-Tuning: Optimizing Continuous Prompts for Generation*](https://aclanthology.org/2021.acl-long.353/).
- Lester, Al-Rfou, and Constant (2021), [*The Power of Scale for Parameter-Efficient Prompt Tuning*](https://aclanthology.org/2021.emnlp-main.243/).
- Wei et al. (2021), [*Finetuned Language Models Are Zero-Shot Learners*](https://arxiv.org/abs/2109.01652).
- Hu et al. (2022), [*LoRA: Low-Rank Adaptation of Large Language Models*](https://openreview.net/forum?id=nZeVKeeFYf9).
- Yao et al. (2023), [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://openreview.net/forum?id=WE_vluYUL-X).
- Schick et al. (2023), [*Toolformer: Language Models Can Teach Themselves to Use Tools*](https://arxiv.org/abs/2302.04761).
- Yang et al. (2024), [*SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html).
- Luo et al. (2023), [*An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning*](https://arxiv.org/abs/2308.08747).
