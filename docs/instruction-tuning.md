---
title: "LLM Instruction Tuning"
type: concept
status: active
updated: 2026-08-07
tags: [instruction-tuning, supervised-fine-tuning, alignment, transfer-learning, datasets, evaluation, LLMs]
---

# LLM Instruction Tuning: Teaching Models to Follow Requests

**Central idea:** Pretraining teaches a language model to continue text. **Instruction tuning** further trains that model on examples in which a natural-language request is followed by an appropriate response. The resulting model is more likely to interpret a new request as a task, select the relevant behavior, and answer in the expected form. Instead of fine-tuning a separate model for every application, one broadly instruction-tuned model can respond to many tasks through the common interface of language.

**Why it matters:** Instruction tuning is one of the main developments that turned general language models into practical assistants. It helps explain why a modern chat model can summarize a document, classify a review, draft an email, answer a technical question, and produce structured data without loading a different task-specific model each time. It does not, however, give the model live knowledge, external tools, reliable permissions, or guaranteed correctness. It changes how the model responds to requests; it does not replace the application system around it.

## Background topics

- **Language-model pretraining:** Learning broad linguistic and factual patterns by predicting tokens in large collections of text.
- **Supervised learning:** Updating model parameters using examples with desired outputs.
- **Transfer learning:** Reusing a model trained on a broad source task as the starting point for other tasks.
- **Fine-tuning:** Continuing training from an existing checkpoint on a smaller, more targeted dataset.
- **Prompting and in-context learning:** Specifying a task through instructions and examples supplied at inference time.
- **Text-to-text modeling:** Representing both task inputs and task outputs as text rather than using a separate output head for every task.
- **Task generalization:** Performing a task, or a type of task, that was not directly represented in the training mixture.
- **Preference training:** Learning which of several plausible responses people prefer, commonly after supervised instruction tuning.
- **Parameter-efficient fine-tuning:** Updating a small set of added or selected parameters instead of every parameter in the base model.
- **Evaluation:** Separating instruction compliance, factual correctness, reasoning, safety, style, and real task success.

These topics describe the machinery and historical setting of instruction tuning. The key conceptual shift is from training one model *for a particular benchmark* to training a model to recognize natural-language task descriptions as a general control interface.

## Before instruction tuning

For much of modern natural-language processing, a “task” was closely tied to a dataset and a model configuration. Sentiment analysis, translation, question answering, named-entity recognition, and textual entailment were treated as different problems. Even when several systems shared an underlying neural architecture, they commonly used different training runs, task-specific output layers, label spaces, preprocessing pipelines, and evaluation code.

The first wave of large-scale pretraining made this process substantially cheaper. A pretrained representation model such as BERT could be adapted to many downstream tasks, while generative models such as GPT could reuse the same language-model architecture. But adaptation usually still meant producing a separate fine-tuned checkpoint for each task. A sentiment classifier learned sentiment labels; a question-answering model learned answer spans; a summarizer learned summaries. The pretrained model was general, but the deployed model remained specialized.

This arrangement created several practical limitations.

First, every new task required labeled data and an optimization pipeline. If a company wanted five slightly different classifiers, it might train and maintain five model variants. A change in labels, domain, or desired output could trigger another data-collection and training cycle.

Second, the task interface was defined by software rather than ordinary language. The application selected a model or task head and converted its output into a domain-specific representation. A user could not simply ask one model to switch from classification to explanation to rewriting unless those behaviors had been deliberately combined.

Third, narrow fine-tuning encouraged narrow evaluation. A model could achieve high accuracy on one benchmark without learning a transferable notion of what the task instruction meant. The model often learned regularities in the dataset, including shortcuts and annotation artifacts, rather than a general ability to follow requests.

Research began to weaken these boundaries before modern instruction tuning had a stable name. The [Natural Language Decathlon](https://arxiv.org/abs/1806.08730) cast ten different NLP tasks as question answering over a context. [T5](https://arxiv.org/abs/1910.10683) generalized the idea into a unified text-to-text framework: translation, classification, summarization, and question answering could all be written as text in and text out. These projects showed that diverse tasks could share an architecture and an interface, although their models still relied heavily on supervised task mixtures and downstream adaptation.

At the same time, generative pretraining suggested that the model itself might infer a task from text. GPT-2 evaluated several tasks without changing its weights, and GPT-3 made **in-context learning** a central method: an instruction or a few demonstrations in the prompt could condition one frozen model to perform many tasks ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf); [Brown et al., 2020](https://arxiv.org/abs/2005.14165)). This was a major breakthrough, but raw pretrained models remained inconsistent instruction followers. They might continue the wording of a prompt, imitate a document, answer the wrong implied question, or require carefully engineered examples.

The remaining problem was an objective mismatch:

> Pretraining rewards plausible continuation of text. Users want appropriate completion of an intended task.

Instruction tuning addresses that mismatch by showing the model many explicit examples of requests and desired responses.

## The topic in one view

A simplified modern training pipeline contains three conceptually different stages:

```text
large corpus of ordinary text
    → pretraining
    → base language model

instructions paired with good responses
    → supervised instruction tuning
    → instruction-following model

comparisons, ratings, safety signals, or other preference data
    → preference optimization
    → assistant tuned toward selected human preferences
```

The stages are often combined in product discussions, but they solve different problems.

| Stage | Typical data | Primary purpose |
| --- | --- | --- |
| Pretraining | Documents, code, conversations, and other large text corpora | Learn language, broad knowledge, representations, and general capabilities |
| Instruction tuning | Request–response examples across one or many tasks | Learn to interpret requests and produce task-appropriate responses |
| Preference training | Rankings, chosen/rejected responses, critiques, or rewards | Prefer some acceptable behaviors over other plausible behaviors |

Instruction tuning is normally a form of **supervised fine-tuning**, often abbreviated SFT. Suppose the dataset contains pairs $(x_i, y_i)$, where $x_i$ is an instruction plus any supplied input and $y_i$ is the desired response. Training minimizes a token-prediction loss such as:

$$
\mathcal{L}_{\text{SFT}}(\theta)
=
-\sum_i \sum_t m_{i,t}
\log p_\theta\!\left(y_{i,t}\mid x_i, y_{i,<t}\right)
$$

Here:

- $\theta$ represents the model parameters;
- $y_{i,t}$ is a response token;
- $y_{i,<t}$ is the response generated before that token;
- $m_{i,t}$ is commonly a mask that gives loss to assistant-response tokens while ignoring system and user tokens.

The mathematics is intentionally ordinary. Instruction tuning usually does not require a new Transformer architecture or a fundamentally new training objective. Its power comes from **what the examples mean, how diverse they are, how they are formatted, and how they are mixed**.

The broad behavior can be summarized as:

$$
\text{pretrained continuation model}
+
\text{many demonstrations of requested behavior}
\longrightarrow
\text{general instruction-following model}
$$

This does not mean the model has acquired a symbolic instruction interpreter. It means that training has shifted the probability distribution so that task-appropriate responses become more likely after instruction-like inputs.

## What instruction tuning is

**Instruction tuning is continued supervised training on examples whose inputs describe what the model should do.** A training example normally contains some combination of:

1. a task instruction;
2. input material to operate on;
3. optional context, constraints, or demonstrations; and
4. a target response exhibiting the desired behavior.

A simple record might look like:

```json
{
  "instruction": "Classify the review as positive or negative and explain briefly.",
  "input": "The camera is excellent, but the battery rarely lasts an afternoon.",
  "response": "Negative. The reviewer likes the camera but identifies battery life as a serious usability problem."
}
```

A chat-oriented version expresses the same relationship through roles:

```text
system: You are a careful assistant. Follow the requested output format.
user: Classify the review as positive or negative and explain briefly.
      Review: The camera is excellent, but the battery rarely lasts an afternoon.
assistant: Negative. The reviewer likes the camera but identifies battery life as a serious usability problem.
```

During training, the response is known. The model is adjusted so that this response, and responses like it, receive higher probability after inputs like the instruction. Across a diverse dataset, the model encounters summarization, extraction, explanation, classification, rewriting, dialogue, planning, coding, and other behaviors expressed through varied language.

### Broad instruction tuning versus narrow supervised fine-tuning

All instruction tuning is supervised fine-tuning, but not all supervised fine-tuning is broad instruction tuning.

- A model trained only to map support tickets into five fixed labels is performing **task-specific fine-tuning**.
- A model trained on many support activities expressed as requests—classifying, summarizing, drafting replies, extracting fields, and asking for missing information—is receiving **domain instruction tuning**.
- A model trained on thousands of tasks from many domains is receiving **general instruction tuning**.

The boundaries are not absolute. What matters is the intended generalization. Narrow task fine-tuning seeks performance on a known task distribution. Broad instruction tuning seeks the transferable behavior of understanding new task descriptions.

### “Instruct” and “chat” are related but not identical

An **instruct model** is generally tuned to respond directly to standalone requests. A **chat model** is additionally trained on role-formatted, often multi-turn conversations. In practice, modern chat models frequently combine:

- supervised instruction tuning;
- multi-turn dialogue examples;
- safety and refusal examples;
- tool-call demonstrations;
- preference optimization such as RLHF or DPO;
- a particular chat template with system, user, assistant, and tool roles.

Model names do not guarantee a precise recipe. Providers sometimes use *instruction tuning*, *alignment*, *post-training*, and *chat tuning* as overlapping umbrella terms. In this article, **instruction tuning** refers specifically to supervised training on desired request–response behavior. Preference optimization is treated as a separate, usually subsequent stage.

## Why instruction tuning works as a general interface

Instruction tuning is valuable because it does more than improve one benchmark. It changes the relationship among the model, the task, and the user.

### 1. It reduces the gap between the pretraining objective and user intent

A base model trained on ordinary text learns that many continuations are plausible. After a question, the next tokens might be an answer, another question, a discussion of the question, or text copied from a web page containing it. Pretraining alone does not establish that a user message is an instruction the system should attempt to satisfy.

Instruction data repeatedly demonstrates a narrower convention:

```text
request → relevant, complete, appropriately formatted response
```

The model learns common interaction norms: answer the question rather than merely continue it, operate on the supplied input, respect explicit constraints, acknowledge uncertainty, and stop after completing the requested output.

### 2. It represents task selection in language

Traditional software selects a task through code: call the sentiment model, translation model, or summarization endpoint. An instruction-tuned model can receive the selection in ordinary text:

```text
Translate this into French.
Summarize this in three bullets.
Extract the dates as JSON.
Explain the argument to a college student.
Compare these designs by reliability and cost.
```

The user does not need to know the application's internal task taxonomy. The model maps varied expressions onto behaviors learned across training. This is the foundation of the general-purpose language interface.

### 3. It supports generalization to unseen tasks

The central research claim of early instruction tuning was not merely that multitask training improves the tasks in the mixture. It was that a model could improve on **held-out task types** when those tasks were described by instructions.

FLAN instruction-tuned a 137-billion-parameter model on more than 60 instruction-formatted NLP tasks and evaluated it on unseen task types. It substantially improved over the untuned model and exceeded zero-shot GPT-3 on 20 of 25 evaluated tasks ([Wei et al., 2021](https://arxiv.org/abs/2109.01652)). T0 independently fine-tuned an encoder–decoder model on a mixture of prompted datasets and demonstrated strong zero-shot performance on held-out tasks, sometimes outperforming models many times its size ([Sanh et al., 2021](https://arxiv.org/abs/2110.08207)).

These results suggested that instruction following could become a transferable skill. The model did not simply memorize one output vocabulary. It learned regularities shared across instructions, demonstrations, inputs, and response types.

Generalization remains conditional. A model is more likely to handle a new task when its concepts, operations, domain, and output conventions are related to patterns encountered during pretraining or post-training. “Unseen task” in an experiment does not mean independent of all prior experience.

### 4. It lowers the burden on prompt engineering

A base model may require an elaborate prompt that frames the desired response as document completion. An instruction-tuned model is more likely to respond to a direct request. It may also be more robust to paraphrases and less dependent on carefully selected demonstrations.

Instruction tuning does not make prompts irrelevant. The current prompt still supplies the particular goal, context, constraints, and data. It changes the baseline: prompt engineering begins with a model already trained to recognize the request–response convention.

### 5. It teaches output and interaction conventions

Many useful behaviors are not new bodies of factual knowledge. They are conventions about how existing capabilities should be expressed:

- produce only the requested JSON object;
- ask a clarifying question when a required fact is missing;
- distinguish a recommendation from a confirmed action;
- explain an answer at the requested level;
- refuse a disallowed request without continuing its harmful content;
- use a tool call instead of claiming that an external action occurred;
- maintain role boundaries across a conversation.

High-quality supervised examples can teach these patterns. LIMA, for example, found that a 65-billion-parameter base model developed surprisingly strong response behavior after supervised fine-tuning on only 1,000 carefully curated examples. Its authors argued that most knowledge and capability came from pretraining while a comparatively small amount of data could teach response style and interaction conventions ([Zhou et al., 2023](https://arxiv.org/abs/2305.11206)). That result should not be interpreted as a universal data requirement, but it illustrates how post-training can strongly change usability without recreating the model's knowledge from scratch.

## The historical development

Modern instruction tuning emerged from several research threads rather than one isolated invention.

### Unified task representations

The Natural Language Decathlon showed that translation, summarization, sentiment, semantic parsing, and other tasks could be recast as question answering. T5 then established a broader text-to-text framework. These systems reduced the architectural differences among tasks and made multitask mixtures easier to construct.

Their contribution was primarily **interface unification**: many problems could share a text input, text output, model architecture, and training objective. They did not yet establish the complete modern assistant experience.

### Zero-shot and in-context task specification

GPT-2 proposed that language modeling at sufficient scale implicitly produces multitask behavior. GPT-3 provided the decisive demonstration that one frozen model could respond to instructions and examples across many tasks without task-specific gradient updates.

This established a crucial separation:

- **training-time adaptation** changes model weights;
- **in-context adaptation** temporarily conditions fixed weights through a prompt.

GPT-3 showed that natural language and demonstrations could specify a task. It did not show that a raw pretrained model would reliably treat every user request as an instruction.

### Explicit multitask instruction tuning

FLAN and T0 made the missing step explicit. Instead of hoping that task interpretation would emerge entirely from pretraining, they fine-tuned models on diverse datasets rendered as natural-language prompts. Both evaluated cross-task generalization by holding out tasks or task families.

The [Super-NaturalInstructions](https://arxiv.org/abs/2204.07705) project expanded the study to 1,616 tasks spanning 76 task types, with expert-written declarative instructions. It made the relationship among instructions, examples, task diversity, and held-out generalization a research object in its own right.

The later FLAN scaling work instruction-tuned models on as many as 1,800 tasks and studied model scale, task scale, and chain-of-thought data. It found broad improvements across model families and prompting settings ([Chung et al., 2022](https://arxiv.org/abs/2210.11416)). The accompanying [FLAN Collection](https://arxiv.org/abs/2301.13688) emphasized that data balancing, enrichment, and a mixture of zero-shot, few-shot, and reasoning-oriented prompts materially affected results.

### From instruction follower to aligned assistant

InstructGPT applied supervised training to prompts drawn from real API use and demonstrations written by human labelers. It then added a reward model and reinforcement learning from human feedback. Human evaluators preferred the 1.3-billion-parameter InstructGPT model to the raw 175-billion-parameter GPT-3 on the studied prompt distribution ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)).

This did not show that a smaller model had more general knowledge than GPT-3. It showed that post-training could make a model far better matched to what users wanted from the interface. InstructGPT also clarified the now-common division between a supervised demonstration stage and a preference-optimization stage.

### Synthetic and curated instruction data

High-quality human demonstrations are expensive. Self-Instruct explored a bootstrapping process in which a model generated new instructions, inputs, and outputs from a small seed set; filtering removed invalid or highly similar examples before fine-tuning ([Wang et al., 2022](https://arxiv.org/abs/2212.10560)). Similar teacher-generated and distilled datasets helped make open instruction tuning widely accessible.

Synthetic data changed the economics of post-training, but it did not remove the need for judgment. A teacher model can propagate factual errors, stylistic habits, safety weaknesses, and narrow task distributions. The 2023 Tülu study found that different open instruction datasets improved different capabilities and that no single dataset or mixture dominated every evaluation ([Wang et al., 2023](https://arxiv.org/abs/2306.04751)). Data composition, not just example count, became a central engineering problem.

## What the model actually learns

It is tempting to describe an instruction-tuned model as having learned an internal program that reads a command and executes it. That metaphor is useful only up to a point.

The training objective still adjusts token probabilities. Across many examples, the model learns statistical regularities such as:

- which parts of an input describe the operation versus the material being operated on;
- how words like *summarize*, *compare*, *extract*, and *rewrite* relate to output transformations;
- how response form depends on requested constraints;
- when examples in the prompt demonstrate a pattern to continue;
- which conversational role should speak next;
- what a plausible high-quality answer looks like in different domains;
- when training data demonstrates clarification, refusal, citation, or tool selection.

These regularities interact with representations learned during pretraining. The base model may already know what Paris is, what JSON resembles, how SQL is written, and how a summary differs from a transcript. Instruction tuning makes it more likely to assemble and express those capabilities in response to a request.

### Capability learning versus capability elicitation

Instruction tuning can do both of the following:

1. **Elicit and organize existing capability.** The base model has relevant knowledge or patterns, but does not reliably expose them through a direct instruction.
2. **Teach additional behavior or skill.** The examples demonstrate task procedures, domain conventions, output formats, or decision rules that were weak or absent before tuning.

The balance varies. General knowledge-heavy tasks rely greatly on pretraining. A specialized extraction format, tool protocol, or organizational writing convention may be learned substantially during fine-tuning. It is therefore too strong to say that instruction tuning only changes style, just as it is too strong to treat it as a dependable database for new facts.

### Generalization depends on diversity and shared structure

A model trained on hundreds of paraphrases of one sentiment task may become a robust sentiment classifier, but it has not necessarily learned broad instruction following. Cross-task generalization improves when the training mixture exposes reusable relationships among:

- many task families;
- varied instruction wording;
- different input and output lengths;
- classification and open-ended generation;
- direct requests and few-shot demonstrations;
- multiple domains and languages;
- straightforward answers, clarification, and appropriate abstention.

Task diversity teaches the model that the instruction is meaningful. Prompt diversity discourages reliance on one template. High-quality responses establish the target behavior. Balance prevents a large easy dataset from overwhelming rarer but important patterns.

## Constructing instruction data

An instruction dataset is an executable behavioral specification written as examples. Its records jointly define what requests look like, which interpretations are preferred, and what counts as a good answer.

### The anatomy of an example

A practical record may include:

| Field | Purpose |
| --- | --- |
| System instruction | Establishes stable role, policy, or environment information |
| User instruction | States the requested operation or goal |
| Input/context | Supplies the text, data, state, or evidence needed for the task |
| Demonstrations | Shows an in-context pattern when few-shot behavior is part of the target |
| Assistant response | Provides the supervised output used for loss |
| Tool call/observation | Demonstrates how external capabilities fit into the dialogue |
| Metadata | Records source, task family, quality, language, license, and split information |

Metadata usually should not be rendered into the prompt, but it is essential for balancing, filtering, provenance, evaluation, and later removal requests.

### Major data sources

| Source | Main advantage | Main risk |
| --- | --- | --- |
| Existing supervised benchmarks converted with templates | Large amount of validated task data | Benchmark language can be artificial; labels may encourage short or brittle behavior |
| Expert-written instructions and responses | High control and domain accuracy | Expensive and slow to produce |
| Real user prompts with human demonstrations | Closely matches deployment demand | Privacy, consent, skewed traffic, and annotator consistency |
| Synthetic instructions from a stronger model | Scales breadth rapidly | Teacher errors, duplicated style, and hidden contamination |
| Self-generated instruction pipelines | Reduces annotation requirements | Quality can drift without strong filtering and evaluation |
| Corrected production failures | Directly targets measured weaknesses | Can overfit recent edge cases or expose sensitive data |

No source is inherently sufficient. A good mixture often combines broad public tasks, carefully authored examples, domain-specific cases, and adversarial or failure-driven examples.

### Quality is not a single score

A polished answer can still be a poor training example. Data quality includes:

- **correctness:** the response is substantively right;
- **instruction fidelity:** it answers the actual request and respects constraints;
- **relevance:** it avoids unrelated material;
- **calibration:** it expresses uncertainty where evidence is insufficient;
- **format validity:** structured outputs and tool calls satisfy their schemas;
- **pedagogical value:** the example demonstrates a transferable behavior rather than a shortcut;
- **provenance:** its source, license, and transformations are known;
- **distributional value:** it adds needed coverage instead of duplicating abundant patterns.

LIMA supplied evidence that a small, highly curated dataset can produce large behavioral changes. FLAN supplied evidence that broad task scale and carefully designed mixtures also matter. These findings are complementary: the useful quantity is not raw examples, but high-quality, diverse behavioral coverage relative to the base model and target distribution.

### Formatting is part of the model contract

Decoder-only chat models are usually trained on a serialized conversation containing special role and boundary tokens. An illustrative template might be:

```text
<|system|>
You are a concise research assistant.<|end|>
<|user|>
Explain why the result does not imply causation.<|end|>
<|assistant|>
The result shows association, but the study does not rule out confounding...<|end|>
```

The exact tokens differ by model. Using the wrong chat template at inference can substantially degrade an otherwise well-tuned checkpoint because the model learned role boundaries through that serialization. End-of-turn and end-of-sequence behavior are likewise part of training; a model that was not taught when an answer ends may ramble or generate another speaker's turn.

### Response-only loss and masking

For a causal language model, the instruction and answer may be concatenated into one token sequence. Many supervised pipelines calculate loss only on assistant tokens:

```python
tokens = render_chat(example)
labels = tokens.copy()
labels[system_and_user_positions] = IGNORE_INDEX

loss = model(input_ids=tokens, labels=labels).loss
loss.backward()
```

This teaches the model to generate the response conditional on the supplied conversation without rewarding it for reproducing user text. Some training recipes use loss over the whole sequence, and encoder–decoder models naturally separate source and target. The correct choice depends on architecture and objective, but accidental masking errors can make a seemingly valid dataset train the wrong behavior.

## The practical training workflow

Instruction tuning should begin with a behavioral goal and an evaluation set, not with an available pile of conversations.

### 1. Define the target behavior

State what should improve and how it will be recognized. “Make the model better” is not sufficient. Useful targets include:

- following compound formatting constraints;
- producing valid calls to a known tool set;
- asking for clarification in defined ambiguous cases;
- using a domain's terminology accurately;
- generating concise explanations for a particular audience;
- handling a stable family of requests at lower latency or cost.

Create held-out tests before training so the same examples do not define both the lesson and the exam.

### 2. Choose an appropriate base checkpoint

A base model offers maximum freedom but requires the dataset to teach basic interaction conventions. An existing instruction-tuned checkpoint already knows the general request–response interface and usually needs less data for domain adaptation. Starting from a chat model may also carry unwanted style, refusal, or template behavior that must be measured rather than assumed.

The checkpoint's license, tokenizer, context length, language coverage, tool format, quantization support, and deployment cost are part of the decision.

### 3. Build and audit the mixture

Normalize records into a common schema while retaining provenance. Deduplicate semantically similar examples, remove leaked evaluation cases, verify licenses, redact sensitive information, and inspect the distribution by task family, response length, source, language, and format.

Balance is normally performed by sampling policy rather than simply concatenating datasets. Otherwise, the largest source defines the model's behavior regardless of its importance.

### 4. Create meaningful splits

A random row split can exaggerate generalization when paraphrases or examples from the same source appear in training and evaluation. Depending on the goal, hold out:

- entire task families;
- domains or data sources;
- instruction templates;
- users or organizations;
- chronological periods;
- combinations of constraints not seen together during training.

This reveals whether the model learned transferable behavior or memorized local surface patterns.

### 5. Render, tokenize, and mask consistently

Apply the same chat template intended for deployment. Check truncation behavior, role tokens, stop tokens, long-example sampling, and which positions receive loss. Validate a small rendered batch manually before spending significant compute.

### 6. Update all or part of the model

Instruction tuning describes the data and objective, not how many parameters must change.

| Update method | Benefit | Tradeoff |
| --- | --- | --- |
| Full fine-tuning | Maximum freedom to adapt the entire model | High memory and storage cost; greater risk of broad regressions |
| LoRA/adapters | Far fewer trainable parameters and small reusable artifacts | Adds configuration choices and may have a lower ceiling for large distribution shifts |
| Quantized parameter-efficient tuning | Makes larger models trainable on limited hardware | More numerical and implementation complexity |
| Hosted fine-tuning API | Removes most training infrastructure | Less control over model internals, data pipeline, and deployment |

[LoRA](https://arxiv.org/abs/2106.09685) freezes the original weights and learns low-rank updates inside selected layers. It is often a practical way to instruction-tune an open model, but LoRA is not itself an instruction-tuning method. The same adapter technique can be used for a classifier, a style model, or another fine-tuning objective.

### 7. Evaluate checkpoints as systems

Monitor training loss, but do not use it as the product criterion. Compare the base and tuned models on held-out instruction tests, general capability suites, safety cases, output schemas, prompt paraphrases, and real workflows. Check repeated runs where stochastic variation matters.

The best checkpoint may occur before the lowest validation loss if later training improves imitation of the dataset while degrading broader behavior.

### 8. Add preference optimization only if it solves a measured problem

Supervised examples show one desired response. Preference data compares alternatives and can teach finer distinctions such as which answer is clearer, safer, more concise, or more useful.

InstructGPT used a learned reward model and reinforcement learning. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) later provided a simpler objective using chosen and rejected responses without a separate reinforcement-learning loop. These methods are related to instruction tuning but should not be conflated with it.

Preference optimization is most useful when several responses are technically valid but differ in quality. It is not a substitute for correct supervised examples, authoritative knowledge, or deterministic validation.

## Instruction tuning and neighboring approaches

Many LLM techniques are described loosely as ways of “teaching the model.” Their mechanisms and persistence differ.

| Approach | What changes | Best suited for | What it does not inherently solve |
| --- | --- | --- | --- |
| Pretraining | Most or all weights, using a very large corpus | Broad language, knowledge, and capability acquisition | A dependable assistant interface |
| Continued pretraining | Weights, using additional unlabeled domain text | Domain language and distribution adaptation | Exact request-following behavior |
| Task-specific fine-tuning | Weights for a narrow labeled task | Stable specialized performance | Broad cross-task generalization |
| Instruction tuning | Weights using request–response examples | General or domain-specific instruction following | Current facts, permissions, or exact execution |
| Prompt engineering | Current input context | Fast, reversible behavior specification | Persistent adaptation across requests |
| In-context learning | Current input context plus demonstrations | Temporary adaptation without training | Permanent behavior change |
| Preference optimization | Weights using comparisons or rewards | Choosing among plausible response behaviors | Ground-truth correctness by itself |
| RAG | Runtime context from retrieved sources | Current, private, attributable knowledge | Learning a durable interaction policy |
| Tool use | Runtime access to external operations | Search, calculation, state, and real actions | Safe execution without a controlling harness |
| Harness engineering | Application control flow, policy, state, and validation | Reliable system behavior around the model | New model capability by itself |

### Instruction tuning versus prompting

Prompting happens at inference time. It is immediate, reversible, and specific to the current request. Instruction tuning happens before deployment and changes future behavior across requests.

Formally, prompting changes $x$ while holding $\theta$ fixed:

$$
y = M_\theta(x)
$$

Instruction tuning produces new parameters $\theta'$ from a dataset $D$:

$$
\theta' = \operatorname{Train}(\theta, D),
\qquad
y = M_{\theta'}(x)
$$

The two approaches are complements. A tuned model still needs a prompt containing the current goal and evidence. Training should absorb stable, repeated behavioral patterns; the prompt should carry information that is specific, changeable, or local to the present request.

### Instruction tuning versus in-context learning

GPT-3's few-shot learning does not update weights. Examples placed in one prompt influence only that inference context. Instruction tuning uses many examples to modify the checkpoint, after which the model can respond zero-shot to related instructions.

This resolves an apparent contradiction in the historical literature:

- GPT-3 showed that a model did not need **per-task fine-tuning** for every new task.
- FLAN and T0 showed that one broad round of **instruction fine-tuning** made that no-update interface much more reliable on unseen tasks.

Modern assistants frequently use both: broad post-training creates the instruction-following model, and the current prompt supplies the particular task.

### Instruction tuning versus continued pretraining

Continued pretraining exposes the model to more raw text from a target distribution, such as legal documents, source code, or medical literature. It is useful when the model needs stronger representations of the domain's language and patterns.

Instruction tuning demonstrates how the model should respond to requests in that domain. If a model knows medical terminology but produces poor clinical summaries, instruction data may address the behavior. If it does not understand the underlying terminology or document genre, continued pretraining may be more relevant. A serious domain adaptation project may use both in sequence.

Neither approach is a dependable way to store frequently changing facts. Retrieval is better suited to facts that need provenance, updates, access control, or deletion.

### Instruction tuning versus preference training

Supervised instruction tuning asks, “What should a good response look like?” Preference training asks, “Which of these plausible responses is better under the chosen criteria?”

SFT normally establishes basic competence and format. Preference optimization can refine helpfulness, brevity, tone, safety, and other difficult-to-specify qualities. Because preferences can conflict, the resulting model reflects the annotators, rubrics, and reward process used to collect them. Calling this entire pipeline “instruction tuning” hides consequential design choices.

### Instruction tuning versus tools and retrieval

Instruction tuning changes how the model maps context to output. It does not create access to information or actions outside the model.

- If the problem is **the model does not know today's schedule**, use retrieval or an API.
- If the problem is **the model cannot calculate an exact total reliably**, use a calculator or program.
- If the problem is **the model repeatedly ignores a stable response convention**, instruction tuning may help.
- If the problem is **a dangerous action must never execute without approval**, enforce that in the [harness](harness-engineering.md), not only in training examples.

Instruction tuning can teach the model *when and how to request a tool*, but [tool use](tool-use.md) still requires an application to validate, authorize, execute, and return the actual observation.

## Benefits and tradeoffs

### The main benefits

**One model can expose many capabilities through one interface.** The application no longer needs a separate endpoint for every linguistic task. Natural language becomes the task-selection mechanism.

**Unseen-task performance can improve.** FLAN, T0, and related work demonstrated that diverse instruction mixtures can transfer to held-out tasks and prompting formats.

**Smaller models can become much more useful.** Post-training can make a smaller model better matched to a user-facing request distribution than a much larger raw base model, as the FLAN and InstructGPT results illustrated. This is behavioral fit, not proof that scale or pretraining capability is unimportant.

**Prompts can become shorter and more natural.** The model already understands common request conventions, reducing the need to disguise a task as text completion or supply many demonstrations.

**Stable formats and domain procedures can be internalized.** A deployed model can require less repeated scaffolding for recurring output structures and interaction patterns.

**An instruction-tuned checkpoint can be a better starting point for later specialization.** The FLAN Collection reported faster and stronger downstream adaptation from FLAN-T5 than from the corresponding untuned T5 checkpoint in its experiments.

### The main tradeoffs

**Behavior depends strongly on the data mixture.** Increasing one task family can improve that family while weakening another. Conflicting examples produce uncertain policies. Raw example count reveals little without composition and quality.

**Training creates a new model artifact to maintain.** Checkpoints or adapters need versioning, evaluation, security review, storage, and compatibility testing with the serving stack.

**The behavior is persistent but harder to inspect.** A prompt rule can be read directly. A behavior absorbed into weights can only be inferred through evaluation. Correcting it usually requires another tuning run or an external policy layer.

**Capabilities can regress.** Overtraining, narrow data, or a large learning rate can reduce general language ability, calibration, multilingual performance, or useful base-model behaviors. Parameter-efficient tuning reduces training cost but does not eliminate behavioral regression.

**Compliance can be mistaken for correctness.** A model may follow the requested format perfectly while supplying false content. Instruction following, factuality, reasoning, and task outcome must be measured separately.

**Synthetic data can narrow the model.** Distillation from one teacher may reproduce its phrasing, blind spots, and value judgments. Repeated generations can create a polished but less diverse training distribution.

**Safety behavior remains probabilistic.** Refusal examples can make unsafe responses less likely, but model training is not a hard authorization boundary. Adversarial inputs, conflicting context, or distribution shift can still elicit unwanted behavior.

## When instruction tuning is the right choice

Instruction tuning earns its cost when a stable, repeated behavioral gap remains after simpler changes have been tested.

It is a strong candidate when:

- the same family of requests appears frequently;
- a general model has the underlying capability but responds inconsistently;
- the desired behavior can be demonstrated with many high-quality examples;
- prompt instructions and examples are becoming large, fragile, or expensive;
- output schemas, domain procedures, or interaction conventions are stable;
- latency or inference cost justifies using a smaller specialized model;
- an owned or fine-tunable checkpoint fits the deployment requirements;
- held-out tests can measure whether tuning actually helped.

Prefer another approach when:

- **the requirement changes frequently:** keep it in configuration, prompts, or policy code;
- **the missing information is current or private:** use retrieval;
- **the task needs exact computation or side effects:** use tools;
- **the problem occurs in only a few prompts:** improve the prompt or add demonstrations first;
- **the desired behavior is a hard security invariant:** enforce it in deterministic code;
- **the available examples are noisy or too few to evaluate:** collect and diagnose before training;
- **a stronger off-the-shelf instruction model already solves the problem:** changing models may be cheaper than maintaining a custom checkpoint.

A practical escalation sequence is:

```text
measure the failure
    → improve the prompt and context
    → add validation, retrieval, or tools where appropriate
    → compare a stronger existing model
    → collect corrected examples
    → instruction-tune only if a stable model-level gap remains
```

This is not a rule that fine-tuning must always come last. It is a way to ensure that training addresses a behavior encoded in weights rather than a missing fact or application-control problem.

## Instruction tuning for the task-and-idea chatbot

For the current single-user task-and-idea prototype, custom instruction tuning should probably not be the first implementation step. A strong existing instruct or chat model, combined with typed event schemas, [context engineering](context-engineering.md), narrow tools, and an explicit [harness](harness-engineering.md), is enough to validate the central product questions:

- Can the chatbot capture a vague idea without forcing unnecessary structure?
- Can it retrieve the correct saved event later?
- Can it update the intended record from conversational references?
- Can it ask only the clarifying questions that materially matter?
- Can it confirm the actual stored result rather than merely claim success?

Fine-tuning before those behaviors and evaluations are stable risks encoding early product assumptions into a checkpoint. It also makes failures harder to attribute: a mistake could come from the model, prompt, context selection, tool contract, database query, or tuning data.

### What to collect during the MVP

The prototype can produce a future instruction dataset without committing to training immediately. For each important turn, retain a privacy-appropriate record of:

- the user request;
- the relevant conversation state supplied to the model;
- the available tool schemas;
- the model's proposed structured action;
- validation or clarification decisions;
- the actual tool result;
- the final answer;
- any user correction;
- the final authoritative event state.

The most valuable examples are not merely successful conversations. They include corrected failures and boundary cases:

- “Remind me about the restaurant sometime” should create a skeletal idea, not invent a date.
- “Move that to Friday” should clarify only when more than one plausible event is in scope.
- “I might visit Montreal” should not be treated as a confirmed booking.
- A database timeout should not produce “Saved.”
- A retrieved event identifier should be used for updates rather than reconstructed from model memory.

These examples teach product-specific interaction behavior. The database and harness still enforce truth, permissions, transactions, and successful completion.

### A possible later tuning record

Once the tool protocol is stable, a supervised example might include a tool call as the target assistant behavior:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Capture ideas without inventing missing dates. Use create_event for persistent items."
    },
    {
      "role": "user",
      "content": "I might want to visit the aviation museum sometime."
    },
    {
      "role": "assistant",
      "tool_call": {
        "name": "create_event",
        "arguments": {
          "title": "Visit the aviation museum",
          "status": "idea"
        }
      }
    }
  ]
}
```

Related examples should show when *not* to call a tool, when to ask for clarification, and how to respond after a real tool result. Otherwise the model may learn that every conversational mention should become a database mutation.

### When custom tuning would become justified

Instruction tuning becomes more attractive if production-like evaluation shows a repeated gap such as:

- a smaller deployable model understands the domain but misses the event schema;
- the same clarification policy requires long few-shot prompts on every request;
- tool selection and argument construction fail in systematic, teachable ways;
- a stable house style or response length matters across most turns;
- inference cost makes a smaller specialized checkpoint economically important;
- enough corrected examples exist to create meaningful held-out scenarios.

A LoRA-style adaptation of an existing instruction model may then be a reasonable first experiment. Compare it with the untuned model using identical prompts, tools, and harness logic. The goal is not a model that sounds more confident. It is one that produces more correct event states with fewer unnecessary questions and no safety regression.

## Common failure modes

### Treating instruction tuning as knowledge storage

The team places current policies, catalog records, or user facts into SFT examples and expects exact recall. Model weights are difficult to update, delete, attribute, or permission. Use retrieval or structured state for knowledge that changes or requires authority.

### Training on answers without teaching the decision boundary

Every example shows a direct answer, so the model never learns when information is insufficient, when clarification is appropriate, or when a tool should be called. Positive examples need neighboring counterexamples and ambiguity cases.

### Template overfitting

All instructions use the same wording and field order. Validation loss looks good, but minor paraphrases cause failure. Vary instruction language and hold out templates during evaluation.

### Dataset imbalance

A large classification dataset dominates a smaller set of open-ended, multilingual, safety, or tool-use examples. The trained model becomes better at the abundant task and worse at the intended product behavior. Track sampling weights and per-category performance.

### Contradictory demonstrations

Different sources encode incompatible expectations about verbosity, refusal, citation, uncertainty, or output format. The model learns an unstable compromise. Establish a target behavior and reconcile conflicts before training.

### Train–test contamination

Benchmark examples, paraphrases, or synthetic derivatives appear in both training and evaluation. Reported generalization reflects memorization. Deduplicate by source and semantics, and use task- or domain-level holdouts.

### Incorrect chat templates or loss masks

Role tokens used in training do not match serving, assistant boundaries are missing, or loss is accidentally calculated on user tokens. The model may echo prompts, generate extra roles, or fail to stop even though the semantic data is sound.

### Excessive adaptation

Too many steps or too aggressive a learning rate improve imitation of the tuning set while degrading broader abilities. Compare checkpoints throughout training and retain regression suites for capabilities that should remain unchanged.

### Synthetic-data monoculture

One teacher generates most examples, and the student inherits its tone, biases, mistakes, and narrow conception of tasks. Use source diversity, filtering, human review, and evaluations that are not generated by the same teacher.

### Optimizing for preference instead of capability

Human or model judges prefer fluent answers, while exact reasoning, coding, multilinguality, or factual knowledge gets worse. The Tülu study found that preference-style evaluation did not expose all capability differences among instruction datasets. Use task-specific and outcome-based tests alongside preference judgments.

### Encoding security only in examples

The dataset teaches the model to request confirmation before a destructive action, and the application assumes this will always happen. The model can still fail. Training may improve the conversation, but the harness must deterministically block unauthorized execution.

## Evaluating instruction tuning

Instruction tuning should be evaluated as a vector of behaviors, not one aggregate score.

| Dimension | Example question |
| --- | --- |
| Instruction fidelity | Did the response perform the requested operation and respect every explicit constraint? |
| Task correctness | Is the classification, answer, transformation, or plan substantively correct? |
| Held-out generalization | Does improvement extend to unseen tasks, templates, domains, and combinations? |
| Format validity | Does JSON parse, do required fields exist, and are tool arguments schema-valid? |
| Robustness | Do paraphrases, ordering changes, distractors, and longer inputs preserve behavior? |
| Calibration | Does the model admit missing evidence rather than inventing an answer? |
| Safety | Are refusals and boundaries appropriate without excessive false positives? |
| Capability retention | Did reasoning, coding, multilingual, or domain performance regress? |
| Efficiency | What training cost, inference latency, prompt length, and serving footprint changed? |
| Workflow outcome | Did the external system end in the correct state? |

[IFEval](https://arxiv.org/abs/2311.07911) evaluates reproducible, verifiable constraints such as required keywords, length conditions, and formatting rules. It is useful because compliance can be checked deterministically. It is necessarily incomplete: an answer can satisfy every surface constraint and still be irrelevant or false.

Open-ended evaluations often use human preference or another model as a judge. These methods capture qualities that exact-match metrics miss, but they introduce position, verbosity, style, and evaluator-model biases. Use multiple evaluators where practical and preserve objective tests for anything that can be checked directly.

### Compare the tuned model with the correct baselines

A useful experiment should compare at least:

1. the original model with the current production prompt;
2. the original model with an improved prompt or few-shot examples;
3. the tuned model with the same deployment prompt;
4. a stronger off-the-shelf instruction model when economically relevant.

Hold retrieval, tools, temperature, context, and harness behavior constant. Otherwise an apparent fine-tuning gain may come from a changed system rather than the checkpoint.

### Evaluate both average performance and regressions

An aggregate improvement can hide a severe loss in one category. Report results by task family, source, language, output type, safety category, and input length. Important cases should run repeatedly if sampling is stochastic.

For an operational assistant, evaluate the final environment as well as the transcript. A perfectly worded confirmation does not compensate for a malformed database write.

## Practical design principles

- **Use instruction tuning for stable behavior, not volatile context.** Put current facts, user state, and changing policies in retrieval, configuration, or tools.
- **Start with a measured failure.** Define the behavior and held-out evaluation before building the dataset.
- **Treat data as product design.** Each example is a decision about how the assistant should interpret and respond.
- **Prefer coverage over repetition.** Diverse task structures and boundary cases teach more than thousands of near duplicates.
- **Preserve provenance.** Source, license, consent, transformations, and removal paths matter operationally.
- **Match the serving template.** Role tokens, tool formats, stop conditions, and loss masks are part of the trained interface.
- **Separate SFT from preference optimization.** Demonstrating a good answer and ranking plausible answers are different supervision signals.
- **Keep hard guarantees outside the model.** Permissions, transactions, schemas, and destructive-action gates belong in code.
- **Compare with prompting and stronger base models.** Custom training is valuable only when it beats simpler alternatives on the target distribution.
- **Measure capability retention.** A more obedient model is not necessarily a more correct or capable one.
- **Version the entire recipe.** Base checkpoint, data snapshot, mixture weights, template, optimizer, adapter configuration, and evaluations jointly define the result.
- **Tune the smallest behavior that needs to persist.** Avoid using weights to absorb information that the runtime can provide explicitly.

## What instruction tuning does not solve

Instruction tuning does not make the model's answer true. It can teach the model to cite evidence, but it does not supply evidence. It can teach the model to produce a tool call, but it does not authorize or execute the tool. It can demonstrate a safety rule, but it does not turn that rule into an unbreakable boundary.

It also does not eliminate prompt engineering. The system must still provide the current goal, relevant context, available tools, and output constraints. A better instruction-following model makes that interface more reliable; it does not remove the need to design it.

Finally, instruction tuning cannot manufacture capabilities arbitrarily. A small model without the necessary representations, knowledge, or computational capacity may imitate the surface form of a good response without acquiring the underlying ability. Better data can elicit and organize what a model knows, teach bounded new behavior, and improve transfer, but the base checkpoint still places meaningful limits on the result.

## Recap

Pretraining gives a language model broad ability to continue text. It does not inherently establish that a user message is a task to complete. Instruction tuning closes much of that gap by fine-tuning the model on natural-language requests paired with desired responses.

The decisive development was not merely learning another task. It was learning a transferable **task interface**. DecaNLP and T5 showed that diverse problems could share a textual representation. GPT-3 showed that prompts could specify tasks without per-task weight updates. FLAN and T0 demonstrated that broad instruction tuning substantially improves zero-shot performance on unseen tasks. InstructGPT combined supervised demonstrations with human-preference optimization to produce the more recognizable assistant paradigm.

The most important distinctions are:

- **Pretraining** supplies broad knowledge and capabilities.
- **Instruction tuning** teaches how requests map to responses.
- **Prompting** specifies the current task without changing weights.
- **Preference training** selects among plausible behaviors.
- **Retrieval and tools** supply external knowledge and capability.
- **The harness** controls state, permissions, execution, and verification.

Instruction tuning is therefore neither a replacement for fine-tuning nor proof that fine-tuning is unnecessary. It is the general fine-tuning stage that reduces the need to build a new model for each individual task. Used well, it turns natural language into a flexible and reusable control surface. Used indiscriminately, it can hide changing facts, policies, and safety requirements inside weights that remain probabilistic and difficult to inspect.

## Key sources

- McCann et al. (2018), [*The Natural Language Decathlon: Multitask Learning as Question Answering*](https://arxiv.org/abs/1806.08730).
- Radford et al. (2019), [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- Raffel et al. (2019), [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://arxiv.org/abs/1910.10683).
- Brown et al. (2020), [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165).
- Wei et al. (2021), [*Finetuned Language Models Are Zero-Shot Learners*](https://arxiv.org/abs/2109.01652).
- Sanh et al. (2021), [*Multitask Prompted Training Enables Zero-Shot Task Generalization*](https://arxiv.org/abs/2110.08207).
- Hu et al. (2021), [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685).
- Ouyang et al. (2022), [*Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/abs/2203.02155).
- Wang et al. (2022), [*Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks*](https://arxiv.org/abs/2204.07705).
- Chung et al. (2022), [*Scaling Instruction-Finetuned Language Models*](https://arxiv.org/abs/2210.11416).
- Wang et al. (2022), [*Self-Instruct: Aligning Language Models with Self-Generated Instructions*](https://arxiv.org/abs/2212.10560).
- Longpre et al. (2023), [*The Flan Collection: Designing Data and Methods for Effective Instruction Tuning*](https://arxiv.org/abs/2301.13688).
- Zhou et al. (2023), [*LIMA: Less Is More for Alignment*](https://arxiv.org/abs/2305.11206).
- Rafailov et al. (2023), [*Direct Preference Optimization: Your Language Model Is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290).
- Wang et al. (2023), [*How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources*](https://arxiv.org/abs/2306.04751).
- Zhou et al. (2023), [*Instruction-Following Evaluation for Large Language Models*](https://arxiv.org/abs/2311.07911).
