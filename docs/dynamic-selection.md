---
title: "Dynamic Selection"
type: concept
status: active
updated: 2026-08-13
tags: [routing, models, context, tools, agents, orchestration, efficiency]
---

# Dynamic Selection in LLM Systems

**Central idea:** Dynamic selection means choosing some part of an LLM system—such as the model, context, tools, workflow, or agent—at runtime according to the current request and system state, rather than using the same configuration for every interaction.

**Why it matters:** LLM requests differ. Some are simple, some require current information, some benefit from specialized tools, and some carry greater risk. Dynamic selection lets an application spend computation and expose capabilities where they are useful, while keeping routine work faster, cheaper, and more focused.

## Background topics

- **Model inference:** How an LLM generates output from the context supplied for one request.
- **Harness engineering:** How application code controls model calls, tools, state, permissions, and verification.
- **Context engineering:** How instructions, records, documents, and history are selected for a model call.
- **Tool use:** How an LLM proposes actions and receives observations from external systems.
- **Routing and classification:** How software assigns an input to one of several available paths.
- **Evaluation:** How quality, cost, latency, and safety are measured across representative requests.

These topics explain the objects being selected and the system that makes the selection.

## Before dynamic selection

The simplest LLM application has one fixed path:

```text
user request → fixed prompt → fixed model → answer
```

This design is easy to understand and test. It also treats every request as if it had identical requirements. A request to correct punctuation may use the same expensive model and lengthy prompt as a difficult research question. Every tool schema may be included even when no tool is relevant. Large amounts of conversation history may be inserted “just in case.”

As LLM applications gained retrieval, tools, multiple models, and agent workflows, this uniform approach became inefficient. The problem changed from merely asking *what should the model answer?* to asking *what execution path should this request receive?*

Dynamic selection is not one standardized algorithm. It is a general systems pattern for making that choice at runtime.

## The topic in one view

A selector, often called a **router**, examines the request and available state, then chooses among candidate resources:

$$
a^* = \underset{a \in A}{\operatorname{argmax}}\;
\bigl[Q(a,x)-\lambda C(a)-\mu L(a)-\rho R(a,x)\bigr]
$$

where:

- $x$ is the current request and state;
- $A$ is the set of possible actions or configurations;
- $Q$ estimates answer quality;
- $C$ is monetary or computational cost;
- $L$ is latency;
- $R$ estimates risk;
- $\lambda$, $\mu$, and $\rho$ represent the application's priorities.

The equation is a design model, not usually a literal calculation. Its important lesson is that “choose the strongest option” is only one objective. The best path is the least costly path that still meets the required quality and safety level.

## What can be selected dynamically?

| Selection target | Example decision | Primary benefit |
| --- | --- | --- |
| **Model** | Send extraction to a small model and difficult reasoning to a stronger model | Lower cost and latency |
| **Context** | Retrieve only records or documents relevant to the request | Less distraction and token use |
| **Prompt or policy** | Load instructions appropriate to classification, research, or editing | Better task specialization |
| **Tools** | Expose search for current facts or a calculator for arithmetic | Access to external capabilities |
| **Workflow** | Use one call for a simple answer or a research-and-verification loop for a complex question | Proportional effort |
| **Agent** | Delegate database analysis to a specialist while keeping routine work with the primary agent | Context separation and specialization |
| **Model parameters** | Route each token through selected experts inside a mixture-of-experts model | Greater model capacity per unit of computation |

The final row occurs **inside the model**. Switch Transformers, for example, use a learned router to send tokens to selected expert networks rather than activating every expert for every token ([Fedus, Zoph, and Shazeer, 2021](https://arxiv.org/abs/2101.03961)).

The other rows usually occur **around the model**. They are responsibilities of the application [harness](harness-engineering.md). This entry is primarily concerned with that application-level form.

## Common selection patterns

### Rules

Deterministic conditions choose the path:

```python
if request.requires_current_information:
    enable("web_search")
if request.risk == "high":
    require_human_approval()
```

Rules are inspectable and appropriate for permissions and hard product boundaries. They become cumbersome when requests cannot be classified reliably from a few explicit fields.

### Learned routing

A classifier or smaller LLM predicts which model or workflow is sufficient. RouteLLM, for example, learns to route requests between stronger and weaker models using preference data ([Ong et al., 2024](https://arxiv.org/abs/2406.18665)). Learned routing adapts to subtle inputs, but its own errors become a new source of failure.

### Cascades and escalation

The system tries an inexpensive path first and escalates when confidence or verification is inadequate:

```text
small model → confidence check → stronger model → human review
```

FrugalGPT studied cascades that learn which models to query and when to stop ([Chen, Zaharia, and Zou, 2023](https://arxiv.org/abs/2305.05176)). Cascades are useful when many easy requests can be resolved cheaply, although failed first attempts add latency to the harder cases.

### Retrieval and capability selection

[Retrieval-augmented generation](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) dynamically selects external passages to place in context. Tool-using systems extend the same principle to capabilities: the system may select search, calculation, database access, or a multi-step [ReAct](paper-react.md) loop according to the task and observations.

## What dynamic selection provides

- **Efficiency:** Simple requests avoid unnecessarily powerful models and long contexts.
- **Specialization:** Different prompts, tools, or agents can be optimized for different work.
- **Context focus:** Only relevant evidence and capabilities occupy the model's limited attention.
- **Scalable quality:** Expensive reasoning and verification can be reserved for requests that need them.
- **Graceful escalation:** Uncertain cases can move to stronger models, additional evidence, or people.

These gains are not automatic. Selection adds a new prediction problem before the original task. A router that sends difficult cases down the easy path can make the complete system worse than a fixed configuration.

## Risks and design boundaries

### Misrouting

The selector may underestimate difficulty, retrieve the wrong evidence, or hide a necessary tool. Evaluation must therefore measure the **combined router-and-worker system**, not just the individual models.

### Hidden complexity

Multiple paths make failures harder to reproduce. Traces should record what was selected, why, which fallback occurred, and what each path cost.

### Latency from escalation

A cascade saves money on easy cases but may make difficult cases wait through several failed attempts. Parallel selection can reduce delay while increasing cost.

### Security mistakes

Dynamic tool selection must not become dynamic permission granting. A model may recommend a tool, but deterministic policy must still decide whether the user and request are authorized to use it. Routing chooses an execution path; it does not create authority.

### Premature optimization

A single model and fixed workflow are often better for a small, homogeneous application. Dynamic selection becomes worthwhile when request types genuinely differ, traffic is large enough for efficiency gains to matter, and representative evaluations can identify safe routing boundaries.

## Application to the task-and-idea chatbot

For the single-user prototype, selection should remain small and inspectable:

| Request | Selected path |
| --- | --- |
| “Remember to buy furnace filters” | Lightweight extraction plus `create_event` |
| “What were the home projects I saved?” | Event retrieval plus response generation |
| “Move that to next month” | Reference resolution, relevant event state, then `update_event` |
| Ambiguous reference | Clarification instead of guessing or escalating indefinitely |

The application can begin with deterministic routing among capture, retrieval, update, and conversation. A learned model router is unnecessary until measured traffic shows that fixed selection is inadequate. The event database remains authoritative regardless of which path is chosen.

## Evaluation

Useful measurements include:

- task success for each request category;
- incorrect route and unnecessary escalation rates;
- quality relative to always using the strongest path;
- average and worst-case cost and latency;
- tool exposure and permission violations;
- performance after models, prompts, or traffic patterns change.

A useful concept is **routing regret**: how much quality, cost, or time was lost compared with the best path that could have been chosen for that request. Evaluating regret by category helps reveal whether the router systematically mishandles unfamiliar, ambiguous, or high-risk inputs.

## Recap

Dynamic selection lets an LLM application assemble an execution path at runtime. It can choose models, context, prompts, tools, workflows, or agents according to the request instead of treating every interaction identically.

Its value comes from matching resources to needs: inexpensive paths for routine work, specialized capabilities for particular tasks, and escalation for uncertainty. Its cost is an additional layer that can itself make mistakes.

The practical rule is:

> Use dynamic selection when the available paths have meaningful, measurable differences. Keep authority and hard constraints outside the selector, and retain a simple fallback when routing confidence is insufficient.

## Key sources

- Lewis et al. (2020), [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).
- Fedus, Zoph, and Shazeer (2021), [*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*](https://arxiv.org/abs/2101.03961).
- Chen, Zaharia, and Zou (2023), [*FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*](https://arxiv.org/abs/2305.05176).
- Ong et al. (2024), [*RouteLLM: Learning to Route LLMs with Preference Data*](https://arxiv.org/abs/2406.18665).
