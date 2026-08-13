---
title: "In-Context Learning"
type: concept
status: active
updated: 2026-08-13
tags: [in-context-learning, prompting, few-shot-learning, demonstrations, context, adaptation]
---

# In-Context Learning

**Central idea:** In-context learning is the ability of a language model to infer a task or pattern from instructions and examples placed in its current context, then apply that pattern to a new input without updating the model's weights.

**Why it matters:** A single pretrained model can perform new or specialized tasks without a separate training run. Developers can change examples at inference time, adapt behavior per request, and prototype tasks before collecting enough data to justify fine-tuning.

## Background topics

- **Pretraining:** How an LLM learns general language patterns before deployment.
- **Autoregressive generation:** How a model predicts each next token from the preceding context.
- **Transformer attention:** How later tokens incorporate information from earlier instructions and examples.
- **Prompt engineering:** How a task is expressed through natural-language instructions and demonstrations.
- **Context windows:** The bounded working information available during one model call.
- **Fine-tuning:** Training that changes model parameters using task-specific data.

These topics explain why examples can influence a model immediately without becoming permanent knowledge.

## Before in-context learning

Traditional supervised NLP normally produced a separate trained model for each task. A sentiment classifier, translator, question-answering system, and information extractor had different datasets, optimization runs, and often different output heads.

Pretrained language models reduced this burden, but the dominant workflow still used task-specific fine-tuning. A general model supplied a starting point; gradient descent then changed its parameters for the downstream task.

GPT-3 made another workflow prominent. The paper evaluated a frozen language model in zero-, one-, and few-shot settings across many tasks. Instructions and examples were placed in the prompt, and no task-specific parameter updates occurred ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)). The context itself became a temporary task specification.

## Zero-shot, one-shot, and few-shot prompting

| Setting | Context provided before the query |
| --- | --- |
| **Zero-shot** | An instruction but no worked example |
| **One-shot** | An instruction and one example |
| **Few-shot** | An instruction and several examples |

The term **in-context learning** most clearly refers to learning from demonstrations, although it is sometimes used more broadly for adaptation from any task information in the current context.

A simple few-shot classification prompt might look like:

```text
Classify each request as CAPTURE, RETRIEVE, or UPDATE.

Request: Remember to buy furnace filters.
Label: CAPTURE

Request: What home projects have I saved?
Label: RETRIEVE

Request: Move the furnace-filter task to September.
Label: UPDATE

Request: Show me my restaurant ideas.
Label:
```

The expected completion is `RETRIEVE`. The examples communicate the label set, task boundary, input format, and desired output style without changing the model beforehand.

## The mechanism in one view

Let $\theta$ represent the model's fixed parameters, $D$ a sequence of demonstrations, and $x^*$ a new query. In-context prediction can be written as:

$$
\hat{y}^* = \underset{y}{\operatorname{argmax}}\;p_{\theta}(y \mid D, x^*)
$$

The demonstrations alter the conditional input but not $\theta$. The model's activations change as it attends to the examples, but its trained weights remain fixed. Remove $D$ from the next request and the adaptation disappears unless the application supplies it again.

The phrase “learning” is therefore unusual. In normal training, information is incorporated into parameters through gradient updates. In-context learning is temporary adaptation performed during inference.

## What might the model be doing?

There is no single complete account of in-context learning in modern LLMs. Several compatible explanations illuminate parts of the phenomenon.

### Inferring a latent task

The model may use the demonstrations to infer which task or data-generating process best explains the prompt. Xie et al. modeled this as a form of implicit Bayesian inference over latent concepts in a controlled theoretical setting ([Xie et al., 2022](https://arxiv.org/abs/2111.02080)).

### Implementing a learning algorithm in its activations

A Transformer can learn during pretraining to process examples as data. Garg et al. trained Transformers that inferred previously unseen linear functions from in-context input-output pairs, with behavior comparable to standard regression algorithms in their experimental setting ([Garg et al., 2022](https://arxiv.org/abs/2208.01066)).

### Recognizing format and distribution

Demonstrations communicate more than the mapping from inputs to correct answers. They also show the input distribution, output vocabulary, formatting convention, and expected response length. Min et al. found that, for several classification and multiple-choice evaluations, these properties explained much of the benefit even when demonstration labels were replaced with random ones ([Min et al., 2022](https://aclanthology.org/2022.emnlp-main.759/)). This does not mean correct labels are generally unimportant; it shows that demonstrations can influence behavior through several channels at once.

These findings are explanatory models and controlled results, not proof that every LLM performs one universal hidden algorithm.

## In-context learning and neighboring techniques

| Approach | When adaptation occurs | What changes | Persistence |
| --- | --- | --- | --- |
| [Prompt engineering](prompt-engineering.md) | Inference time | Human-readable instructions and examples | Only while context is supplied |
| **In-context learning** | Inference time | Behavior conditioned on current demonstrations | Temporary |
| [Soft prompt tuning](soft-prompting.md) | Training time | Learned continuous prompt vectors | Stored as task parameters |
| [Instruction tuning](instruction-tuning.md) | Post-training | Model weights across many instructed tasks | Durable |
| Fine-tuning or LoRA | Post-training | All weights or task-specific parameter modules | Durable |
| Retrieval-augmented generation | Inference time | Evidence supplied to answer this instance | Temporary |

In-context learning is a capability. Few-shot prompting is the common method used to invoke it. Prompt engineering designs the demonstrations, while [context engineering](context-engineering.md) decides which examples and other information reach the model on each call.

## What demonstrations can provide

- **Task identification:** What transformation or decision is expected?
- **Boundary cases:** Which superficially similar inputs belong to different categories?
- **Output contract:** Which labels, fields, format, and level of detail should be returned?
- **Local terminology:** How does this application use words such as “event” or “update”?
- **Temporary customization:** How should this request differ from the model's default behavior?
- **Fast experimentation:** Does the model possess the underlying capability before training is considered?

A demonstration should teach a boundary, not merely repeat the instruction. Three examples that all represent obvious capture requests provide less information than examples contrasting capture, retrieval, update, ambiguity, and missing values.

## Selecting and arranging demonstrations

### Relevance versus coverage

Examples similar to the current query can clarify local patterns. Diverse examples can define the broader task. A useful set often combines both rather than selecting only the nearest examples.

### Correctness and consistency

Examples should reflect the behavior actually wanted. Contradictory labels, outdated policies, or inconsistent formats force the model to infer which precedent to follow.

### Order

Models can be sensitive to demonstration order and recency. Important distinctions may be placed near the query, but the only reliable choice is to test several harmless permutations.

### Context budget

Each example consumes tokens, increases cost and latency, and competes with instructions, retrieved evidence, conversation history, tool schemas, and output space. More examples are not automatically better.

### Dynamic selection

An application can retrieve examples according to the current task instead of sending one fixed demonstration set. This is a form of [dynamic selection](dynamic-selection.md), but it adds a second failure point: the example retriever may choose misleading precedents.

## Limits and failure modes

### No durable learning

The model does not remember a demonstration in later calls unless the application preserves and resupplies it. In-context learning is not a replacement for a database, long-term memory, or model training.

### Sensitivity

Performance may vary with wording, formatting, example choice, order, and label names. A single successful prompt does not establish reliability.

### Context cost

Few-shot examples lengthen every request. Repeated inference costs can eventually exceed the cost of a stable trained adaptation.

### False analogy

The model may copy irrelevant surface patterns from examples rather than the intended principle. Examples need contrast and held-out evaluation.

### Untrusted demonstrations

Retrieved or user-supplied examples can contain prompt injection, private information, or obsolete rules. Demonstrations are context, not automatically trusted instructions.

### No hard guarantees

Examples can improve [structured output](structured-output.md) but cannot guarantee semantic correctness, authorization, or successful execution. Those remain responsibilities of the application [harness](harness-engineering.md).

## Application to the task-and-idea chatbot

For the MVP, a small demonstration set can teach several product-specific behaviors:

- save vague ideas immediately as skeletal events;
- preserve missing dates as unknown rather than inventing them;
- distinguish a request to retrieve records from a request to update one;
- ask for clarification only when the referenced event or requested change is genuinely ambiguous;
- confirm database mutations from tool results rather than model assumption.

These examples should supplement narrow tools and typed schemas, not replace them. Begin with a fixed, reviewed set. Add dynamically retrieved examples only if evaluations show meaningful variation across request types.

## Evaluation

Compare at least zero-shot, fixed few-shot, and dynamically selected few-shot configurations. Measure:

- task and field-level accuracy;
- ambiguity and unknown-value preservation;
- schema and tool-call correctness;
- sensitivity to example order and paraphrasing;
- robustness to irrelevant or conflicting demonstrations;
- context tokens, latency, and cost;
- final database state after any action;
- performance on request types absent from the examples.

Keep evaluation cases separate from demonstrations. Otherwise the system may appear to generalize while merely repeating examples it has already seen.

## Recap

In-context learning allows a frozen LLM to infer a task from information placed in its current context. Instructions and demonstrations can define the task, output format, label space, local vocabulary, and relevant boundary cases without a training run.

The adaptation is immediate and flexible but temporary. Its effectiveness depends on the pretrained model, example quality, selection, order, and available context budget. It does not create persistent knowledge or application-level guarantees.

The practical rule is:

> Use in-context examples to demonstrate the smallest set of boundaries the model cannot reliably infer from instructions alone, then evaluate the complete prompt under variation.

## Key sources

- Brown et al. (2020), [*Language Models Are Few-Shot Learners*](https://arxiv.org/abs/2005.14165).
- Xie et al. (2022), [*An Explanation of In-Context Learning as Implicit Bayesian Inference*](https://arxiv.org/abs/2111.02080).
- Min et al. (2022), [*Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?*](https://aclanthology.org/2022.emnlp-main.759/).
- Garg et al. (2022), [*What Can Transformers Learn In-Context? A Case Study of Simple Function Classes*](https://arxiv.org/abs/2208.01066).
