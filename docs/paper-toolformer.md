---
title: "Toolformer"
type: paper-guide
status: active
updated: 2026-08-07
paper_year: 2023
tags: [tools, tool-use, self-supervision, APIs, fine-tuning, foundational-papers]
---

# Paper Guide: *Toolformer*

**Paper:** Timo Schick et al., 2023  
**Full title:** *Toolformer: Language Models Can Teach Themselves to Use Tools*  
**Primary link:** [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)  
**Original problem:** Teaching a general-purpose language model when and how to call external tools without constructing a large human-labeled tool-use dataset  
**Why it matters:** Toolformer demonstrated a practical way for a language model to generate and filter its own tool-use examples, then learn from them through ordinary language-model fine-tuning.

## The paper in one sentence

Toolformer showed that a language model can begin with only a few demonstrations of each tool, propose many possible API calls inside ordinary text, retain the calls whose results make the text easier to predict, and fine-tune itself on those automatically selected examples.

## The practical idea in one view

Toolformer starts with a pretrained language model and a collection of ordinary text. It turns that text into a new training set containing useful tool calls:

```text
ordinary text
    -> propose possible API calls
    -> execute the calls
    -> measure whether each result helps
    -> discard unhelpful calls
    -> fine-tune on the surviving examples
```

After fine-tuning, the model can generate a call while writing. The runtime pauses generation, executes the requested tool, inserts its result, and lets the model continue.

The novel part is primarily the **training-data pipeline**. The paper did not invent calculators, search-augmented language models, or API calling. Its contribution was a scalable method for teaching a general language model:

- whether a tool is useful at a particular point;
- which available tool to select;
- what argument to give it;
- where to place the call in generated text; and
- how to continue after receiving the result.

## Background topics

- **Autoregressive language modeling:** Predicting the next token from the preceding tokens.
- **In-context learning:** Learning an apparent task from examples placed in the prompt without updating model weights.
- **Fine-tuning:** Updating a pretrained model on a more specialized collection of examples.
- **Self-training:** Allowing a model to generate candidate labels or examples that are later used as training data.
- **APIs:** Structured interfaces through which software requests an operation and receives a result.
- **Tool use:** Connecting a language model to external systems for retrieval, calculation, observation, or action.
- **Perplexity and cross-entropy:** Measures of how surprising a piece of text is to a language model.
- **Runtime orchestration:** Application logic that pauses generation, executes a call, inserts the observation, and resumes the model.

These topics help explain the mechanism, but the paper can be understood through a simpler question: **How can a model discover which tool calls are useful without people manually labeling millions of examples?**

## The problem Toolformer was solving

By 2023, large language models could perform many tasks from instructions or a few examples. Yet they continued to struggle with capabilities that much simpler software handled well:

- exact arithmetic;
- current dates and changing facts;
- factual lookup;
- information absent from the model's parameters;
- translation involving languages poorly represented in training.

Scaling the model could improve some of these capabilities, but it was an indirect and expensive solution. A calculator already performs arithmetic exactly. A search system can inspect a maintained corpus. A calendar can return the current date. A translation model can specialize in languages for which the general model is weak.

The obvious architectural response was to give the model access to those systems. That created a training problem. The model needed examples showing not only the syntax of a call, but also the judgment surrounding it:

```text
Should I call a tool here?
Which tool should I use?
What should I ask it?
How should I use the result?
```

Earlier systems commonly solved this with human demonstrations, task-specific prompts, or workflows in which the appropriate tool was already known. Those approaches could work well, but they made it expensive to add tools broadly or tied tool use to a narrow benchmark.

Toolformer's goal was more ambitious: add tool use to a general language model while preserving its ordinary language ability and requiring only a few hand-written demonstrations per API.

## The central insight: let future text provide the supervision

Suppose a training document contains this sentence:

> A survey included 1,250 people, and 375—or 30%—selected the first option.

The document already contains the answer to the calculation. That makes it possible to test a proposed calculator call:

```text
A survey included 1,250 people, and 375—or
[Calculator(375 / 1250) -> 0.30]
30%—selected the first option.
```

If seeing `0.29` makes the original continuation `29%` easier for the model to predict, then the calculator call probably supplied relevant information. If a proposed call returns an unrelated number and does not help predict the continuation, it should be discarded.

The original text therefore acts as a kind of answer key. No person has to label the precise location, argument, and result of the call. The model proposes possible annotations, and its own next-token prediction loss supplies a filter.

This is why Toolformer is described as **self-supervised**. The training signal comes from ordinary text rather than a large dataset in which humans explicitly marked correct tool calls.

That description needs one qualification. The system is not created without human guidance. People still:

- choose and implement the tools;
- define how calls are represented;
- write a small number of examples for each API;
- choose sampling and filtering thresholds;
- select source data and heuristics;
- execute the calls and build the runtime.

Toolformer reduces the need for *large-scale example annotation*. It does not remove the surrounding engineering or all forms of supervision.

## How the training method works

### 1. Represent calls as text

Toolformer treats an API call and its returned value as special spans embedded in otherwise ordinary text. In conceptual notation:

```text
<API> ToolName(argument) -> result </API>
```

For example:

```text
The product costs <API> Calculator(19.95 * 3) -> 59.85 </API> $59.85.
```

In the actual implementation, the researchers reused existing token sequences such as brackets and an arrow instead of adding new vocabulary. The exact delimiters are not the important part. The important choice is that tool interaction becomes something the language model can generate and read as part of a token sequence.

This creates a clean division of labor:

- the **model** generates the tool name and argument;
- the **runtime** recognizes the call, executes external software, and supplies the result;
- the **model** reads that result and continues generating.

### 2. Demonstrate each API a few times

For every tool, the researchers wrote a prompt showing examples of where calls could be inserted. A question-answering example might transform:

```text
Ada Lovelace was born in London, England.
```

into something like:

```text
Ada Lovelace was born in
[QA("Where was Ada Lovelace born?")]
London, England.
```

These demonstrations do not become the entire training dataset. They teach the existing model enough of the annotation pattern that it can propose calls over a much larger corpus through in-context learning.

### 3. Sample possible call locations and arguments

Given a document, the model estimates places where an API call might begin. At promising positions, it samples possible tool inputs.

For the text:

```text
Rochester is often called the Flour City.
```

the model might propose:

```text
[QA("What nickname is Rochester known by?")]
[QA("Which state is Rochester in?")]
```

Both calls are syntactically reasonable. Only the first is useful for predicting the nearby phrase `the Steel City`.

At this stage, Toolformer deliberately generates many candidates. The model is being used as a noisy data annotator, not trusted as a final judge.

### 4. Execute every candidate

The external systems run the proposed calls:

```text
QA("What nickname is Rochester known by?") -> Flour City
QA("Which state is Rochester in?") -> New York
```

Execution is essential. A plausible-looking call is not useful merely because the model wrote valid syntax. Toolformer evaluates the real returned result.

This is an early example of **execution feedback**: generated actions are tested against an environment before they become training examples.

### 5. Keep calls whose results improve prediction

For each candidate, Toolformer compares how well the model predicts the tokens following the proposed insertion under three conditions:

1. no API call;
2. the API call without its result;
3. the API call with its returned result.

The third condition must beat the best of the first two by a chosen margin. In plain language:

> Keep the example only if the tool's answer—not merely the wording of the call—makes the nearby continuation easier to predict.

Let $L_{\text{result}}$ be the prediction loss when the call and result are available, and let $L_{\text{baseline}}$ be the better loss obtained without the result. Toolformer's usefulness score can be summarized as:

$$
\Delta = L_{\text{baseline}} - L_{\text{result}}
$$

The example is retained when:

$$
\Delta \geq \tau
$$

where $\tau$ is a filtering threshold.

Lower loss means that the continuation is less surprising to the model. A large positive $\Delta$ therefore indicates that the returned value supplied information relevant to the next few tokens.

The actual paper uses a weighted loss that emphasizes tokens close to the call. This matters because a tool result should help where it is inserted. A fact that happens to improve a distant and unrelated sentence is weak evidence that the call was well placed.

### 6. Fine-tune on the surviving examples

The retained calls and results are inserted into the original documents. The model is then fine-tuned with the standard next-token prediction objective.

Conceptually, the new corpus contains examples like:

```text
Rochester is often called
[QA("What nickname is Rochester known by?") -> Flour City]
the Flour City.
```

From many such examples, the model learns the complete sequence:

1. recognize a context in which outside help is valuable;
2. generate an appropriate call;
3. leave a position for the result;
4. read the result once the runtime inserts it;
5. incorporate it into the continuation.

Because the augmented corpus otherwise preserves the original text, the researchers hoped to teach tool use without replacing the model's general language distribution with a narrow task dataset.

## A small implementation sketch

The core data-generation idea can be expressed without reproducing the paper's full sampling procedure:

```python
def build_tool_examples(model, documents, tools, threshold):
    accepted = []

    for text in documents:
        candidates = model.propose_tool_calls(text, tools)

        for candidate in candidates:
            result = tools.execute(
                name=candidate.tool,
                arguments=candidate.arguments,
            )

            loss_without_result = model.future_loss(
                text=text,
                insertion=candidate.call_only,
            )
            loss_without_call = model.future_loss(
                text=text,
                insertion=None,
            )
            loss_with_result = model.future_loss(
                text=text,
                insertion=candidate.with_result(result),
            )

            baseline = min(loss_without_call, loss_without_result)
            improvement = baseline - loss_with_result

            if improvement >= threshold:
                accepted.append(
                    insert_call(text, candidate, result)
                )

    return accepted
```

This sketch hides batching, token positions, sampling thresholds, tool-specific prompts, loss weighting, malformed calls, and the merging of multiple examples. It nevertheless captures the paper's central algorithm:

```text
generate candidates -> test them -> score their usefulness -> learn from the useful ones
```

## What happens at inference time

After fine-tuning, generation begins like ordinary autoregressive decoding. If the model produces the tokens that mark an API call, it generates the tool name and arguments. When it reaches the arrow indicating that a result should follow, the runtime intervenes:

```text
1. Model generates: [Calculator(400 / 1400) ->
2. Runtime pauses the model.
3. Runtime executes the calculator.
4. Runtime inserts: 0.29]
5. Model continues: 29% of participants...
```

This is a genuine change to the LLM workflow. A normal call is approximately:

```text
context -> model -> completed text
```

Toolformer requires:

```text
context
    -> partial generation
    -> detect requested call
    -> external execution
    -> insert observation
    -> resumed generation
```

The model does not execute the calculator or search engine internally. It only emits a representation of the request. An application around the model must recognize the request, run the correct implementation, and return the result.

This runtime is a compact precursor to the modern tool-use loop discussed in [Tool Use: Extending LLMs Beyond Text Generation](tool-use.md). It is still much narrower than a contemporary agent harness: the experiments generally allowed at most one call per input, and the model could not iteratively inspect, revise, and call another tool.

## The five tools

Toolformer used five kinds of external capability:

| Tool | What it returned | Limitation it addressed |
| --- | --- | --- |
| Question answering | A short answer from an Atlas retrieval-augmented model | Missing factual knowledge |
| Wikipedia search | Relevant snippets from a BM25 Wikipedia index | Broader factual lookup |
| Calculator | Results for basic arithmetic | Unreliable exact calculation |
| Calendar | The current date | Lack of temporal awareness |
| Machine translation | English translations from an NLLB-based system | Weaknesses in multilingual understanding |

These were all **read-oriented tools**: they returned information to the model. None sent a message, modified a database, purchased an item, or changed the outside world.

That distinction is important. Learning when a lookup result helps predict text is not the same problem as learning when an action is authorized, safe, or faithful to a user's intention.

## What the experiments established

The main model was based on GPT-J with 6.7 billion parameters. The researchers created an augmented dataset from a subset of CCNet, fine-tuned the model, and evaluated it zero-shot on factual completion, mathematical reasoning, question answering, multilingual question answering, and time-sensitive tasks.

The strongest results appeared where a tool cleanly matched the benchmark's missing capability.

### Factual completion

On three LAMA subsets, enabled tool use substantially improved the Toolformer model. For example, its T-REx score rose from 34.9 with calls disabled to 53.5 with tools enabled. The corresponding reported score for the 175-billion-parameter base GPT-3 model was 39.8.

The model used the question-answering API on almost every example. This shows the benefit of connecting a smaller model to a system that can directly supply the missing fact. It does not mean that the smaller model had acquired more factual knowledge in its own parameters.

### Mathematical word problems

The calculator produced the clearest gains. Representative results were:

| Benchmark | Toolformer, tools disabled | Toolformer, tools enabled | GPT-3 175B baseline |
| --- | ---: | ---: | ---: |
| ASDiv | 14.8 | 40.4 | 14.0 |
| SVAMP | 6.3 | 29.4 | 10.0 |
| MAWPS | 15.0 | 44.0 | 19.8 |

These results support a practical division of labor. The language model can interpret a word problem well enough to formulate an arithmetic expression, while a deterministic calculator handles the exact computation.

### Open-domain question answering

Wikipedia search improved all three reported QA benchmarks relative to the same model with tools disabled. The gains were real but did not overtake GPT-3:

| Benchmark | Toolformer, tools disabled | Toolformer, tools enabled | GPT-3 175B baseline |
| --- | ---: | ---: | ---: |
| WebQuestions | 18.9 | 26.3 | 29.0 |
| Natural Questions | 12.6 | 17.7 | 22.6 |
| TriviaQA | 46.7 | 48.8 | 65.9 |

The model relied almost entirely on Wikipedia search for these tasks. The authors attributed part of the remaining gap to the simplicity of the search interface and Toolformer's inability to reformulate a poor query or inspect several results.

### Dates and multilingual tasks

The calendar API produced a large gain on DATESET, a synthetic benchmark requiring knowledge of the current date. Machine translation improved results across the tested languages, but the overall multilingual results were mixed because continued training on the chosen corpus sometimes harmed the base model's language ability.

This is a valuable negative result: a useful tool cannot automatically compensate for every change introduced during fine-tuning.

### General language modeling

With API calls disabled, the fine-tuned model's perplexity was comparable to a control model trained on the same text without inserted calls. This supported the authors' goal of adding tool-use behavior without clearly degrading ordinary language modeling.

The paper also found that effective tool use depended on model capacity. In its GPT-2-family experiments, consistent gains appeared around 775 million parameters, although the search tool helped some smaller models. Tool access did not eliminate the need for a model capable of recognizing when and how the tool was relevant.

## How to interpret those results carefully

The paper established a promising mechanism, not a universal law that tool-enabled small models outperform large models.

Several details matter:

- The GPT-3 comparison used the original `davinci` base model, not a later instruction-tuned assistant.
- The evaluation was zero-shot, but the metrics sometimes allowed an answer to occur within the first several generated words rather than requiring an exact response.
- The runtime modified decoding so that a tool call could begin when its start token appeared among the ten most likely next tokens, not only when it was the single most likely token.
- Increasing the model's tendency to call tools improved some benchmark scores but weakened its natural selectivity about when a call was necessary.
- Toolformer was limited to at most one call per input in the downstream experiments to prevent call loops.
- Some improvement remained when tools were disabled, especially on math, suggesting that the fine-tuning data itself—not just external execution—changed the model.
- The tools were selected because they aligned well with the evaluation tasks.

The right conclusion is:

> A modest-sized model can gain large task-specific benefits when it learns to invoke a well-matched external capability at the right point.

That conclusion is both narrower and more useful than saying tools simply make a model equivalent to one many times larger.

## Why the filtering method is clever

The filtering stage solves several problems at once.

### It tests real outputs

The pipeline does not assume that a well-formed call is correct. It runs the tool and evaluates the actual response.

### It selects examples from the model's perspective

A person might think a call is informative while the model already predicts the continuation easily. Toolformer retains calls that reduce the model's own uncertainty.

### It teaches when *not* to use tools

Most proposed calls are rejected. The final corpus therefore does not consist of indiscriminate tool invocation everywhere a call is possible.

### It avoids building a fully labeled dataset

A handful of demonstrations can seed a much larger annotation process. This is the main source of the paper's scalability.

### It preserves the surrounding language distribution

The accepted calls are inserted into ordinary pretraining-like text rather than replacing general language modeling with a small collection of benchmark instructions.

## The filtering objective's hidden limitation

Toolformer defines a useful call as one whose result makes **future tokens easier to predict**. That is a practical self-supervised signal, but it is not identical to every meaning of usefulness.

A tool result can improve prediction while being:

- factually wrong in a way that happens to match the source document;
- redundant for a user who did not ask for it;
- too expensive or slow to justify;
- derived from an untrustworthy source;
- unsafe to expose;
- irrelevant to completing the user's actual goal;
- harmful if the tool performs a side effect.

Conversely, a call may be operationally valuable without helping predict text already present in a document. Consider:

```text
create_event(title="Call the contractor")
```

Its value is that a durable event now exists in a database. The success of that action is not naturally measured by how well its returned ID predicts the next sentence in a web document.

Toolformer's objective is therefore especially well suited to **information tools** whose outputs help complete text. Production assistants that perform actions need additional supervision and evaluation based on authorization, environment state, user intent, and task completion.

## What Toolformer is—and is not

### It is a fine-tuning method

Toolformer does not use prompting alone. Few-shot prompts generate candidate annotations, but the final model is fine-tuned on the automatically augmented corpus. The approach is closely related to [instruction tuning](instruction-tuning.md) in that both update model weights to create transferable behavior, although their training examples and objectives differ.

### It is a form of self-training

The model proposes examples that may later become its own training data. A filtering mechanism is essential because the initial proposals are noisy.

### It is not retrieval-augmented generation by itself

Two of its tools retrieve information, but the general method can call non-retrieval systems such as a calculator or calendar. RAG describes a retrieval-and-generation architecture; Toolformer describes a way of teaching call behavior across several APIs.

### It is not merely structured output

Generating a parseable call is only one step. The call is executed, its result changes the model's context, and the model continues around that observation.

### It is not a full agent

Toolformer does not form a long plan, maintain durable workflow state, use one result to choose the next tool, recover through several interactive attempts, or determine whether a complex goal has been completed. It demonstrates learned tool invocation inside generation, not general autonomous action.

### It does not make the model the execution authority

The model proposes call text. External software still performs the operation. Even in this simple design, model inference depends on a runtime capable of detecting calls and safely supplying results.

## Important limitations identified by the paper

### No chains of tools

Calls for different tools were generated independently during dataset creation. The model therefore did not learn examples in which the result of one call becomes the argument to another.

A desirable sequence such as:

```text
current_date = Calendar()
fact = Search("office holder as of " + current_date)
```

was outside the method.

### No interactive search

The model could not inspect an initial search result, decide it was poor, reformulate the query, and search again. Search was a single insertion rather than a dialogue with an information environment.

### Sensitivity to wording

Small changes in the input could affect whether the model chose to call an API. This makes reliable need recognition difficult, especially when the call/no-call boundary matters.

### Inefficient data generation

Useful examples could be rare. The paper reports that processing more than a million documents produced only a few thousand retained calculator examples.

### Tool cost was ignored

The filtering score measured predictive benefit, not latency, financial cost, rate limits, energy use, or opportunity cost. A call with a tiny prediction benefit could still be undesirable in a production system.

### Calls were treated as trustworthy information

The paper studied benign tools in a controlled research setting. It did not address authorization, data privacy, prompt injection inside search results, malicious tools, approval for consequential actions, or verification of side effects.

## Toolformer compared with neighboring approaches

Toolformer belongs to a larger transition from closed language models to systems that alternate between language generation and external computation.

| Approach | How tool behavior is obtained | Typical strength | Typical limitation |
| --- | --- | --- | --- |
| Fixed application pipeline | Developers decide when every tool runs | Predictable and easy to test | Cannot adapt the sequence to the request |
| Few-shot prompting | Examples in the current prompt show how to call tools | No weight updates required | Consumes context and may be prompt-sensitive |
| Human-labeled fine-tuning | People supply correct calls and responses | Direct control over target behavior | Expensive to scale and maintain |
| Toolformer | Model proposes calls; prediction improvement filters them | Scalable self-generated training data | Optimizes textual predictability and single-step calls |
| ReAct-style loop | Prompted reasoning alternates with actions and observations | Supports adaptive multi-step behavior | More calls, context, latency, and loop failure |
| Deterministic router plus model | Code selects tools; model fills semantic arguments or interprets results | Strong control and permissions | Less flexible when routing cannot be specified in rules |

The alternatives are not stages of maturity. A fixed pipeline may be the best architecture when a tool must always run. Toolformer is most interesting when the system needs learned judgment across many natural contexts and large-scale human annotation is impractical.

## What carried forward into modern tool-calling systems

Several Toolformer ideas became broadly important even where its exact training recipe is not used.

### Tool use is a learned decision

A useful model must learn more than call syntax. It must recognize the gap between what it can generate internally and what should come from an external capability.

### Execution results belong in the next model context

The returned value becomes new evidence for generation. Tool use is therefore an observation loop, not a one-way emission of JSON.

### Training data can include entire trajectories

Calls, arguments, observations, and continuations can all be represented in model-readable sequences. Later systems extend this idea to multi-step conversations and agent trajectories.

### Execution feedback can filter synthetic data

Generated examples become more trustworthy when they are tested by an environment. Compilers, unit tests, calculators, database constraints, and simulators can all provide stronger filters than a model judging its own prose.

### External capabilities can outperform parameter scaling on the right subproblem

The calculator experiments demonstrate a recurring engineering principle: exact specialized software can be more effective than asking a larger general model to approximate the same operation.

## What modern systems usually add

A production tool-using assistant typically moves much of the responsibility outside the model:

- machine-readable schemas for tool arguments;
- deterministic parsing and validation;
- a registry of available tools and permissions;
- explicit separation between read and write operations;
- user approval for consequential actions;
- retries, timeouts, idempotency, and error handling;
- multiple observation-action steps;
- durable task and application state;
- tracing and evaluation of the complete trajectory;
- defenses against untrusted tool output;
- cost-aware routing and stopping rules.

These are responsibilities of the application [harness](harness-engineering.md), not consequences of the model having seen tool-call tokens during training.

A modern runtime often uses a workflow closer to:

```python
while steps < max_steps:
    proposal = model(context, available_tools)

    if proposal.is_final_answer:
        return proposal.answer

    call = validate_schema(proposal.tool_call)
    authorize(call, user, application_state)
    observation = execute_with_timeout(call)
    context.append(observation)
```

Toolformer supplied an important model-training answer to “how might the model learn to request help?” It did not eliminate the systems-engineering questions surrounding what may execute and whether the task actually succeeded.

## Practical lessons for developers

### Match the tool to a real model limitation

The paper's largest improvements came from clean capability matches: calculator for arithmetic, QA for missing facts, and calendar for the current date. Adding a tool without identifying the gap it closes usually adds latency and failure modes without comparable benefit.

### Separate call quality from tool quality

The model can choose the correct search tool and still receive a poor result. Evaluation should distinguish:

- whether a call was needed;
- whether the correct tool was selected;
- whether the arguments were good;
- whether the tool returned reliable information;
- whether the model used the observation correctly.

### Use execution to validate synthetic examples

If training data contains calculator calls, run them. If it contains code, compile and test it. If it contains database operations, execute them in a sandbox and inspect resulting state. Toolformer illustrates the value of grounding synthetic data in something stronger than model confidence.

### Optimize the actual application objective

Future-token loss was an ingenious signal for Toolformer's setting. It is not automatically the right measure for a support agent, coding assistant, scheduler, or purchasing system. Use tests, database state, policy compliance, factual support, or user-confirmed completion when those better represent success.

### Keep the runtime authoritative

Even a model trained specifically for tool use can generate malformed, unnecessary, ambiguous, or unsafe calls. Application code must still validate and authorize every operation.

### Include negative and abstention examples

A good system needs examples in which the model should answer directly, request clarification, or decline to act. Toolformer's filtering discards many unhelpful calls, but production data often needs explicit coverage of high-risk call/no-call boundaries.

### Account for cost

Tool value is not binary. A useful decision policy should consider expected accuracy improvement alongside latency, price, reliability, and risk:

$$
\text{net value}
=
\text{expected task improvement}
-
\text{execution cost}
-
\text{failure risk}
$$

Toolformer's objective modeled mainly the first term.

## Application to the task-and-idea chatbot

For a task-and-idea application, Toolformer's general lesson is valuable but its exact objective would be incomplete.

The model could learn natural places to request operations such as:

```text
search_events(query)
create_event(draft)
update_event(event_id, patch)
```

However, a `create_event` call should not be considered correct because its result helps predict the next sentence. It should be evaluated against the user's intent and the final database state.

For example, the user says:

> I should look into having the kitchen cabinets refinished.

A correct trajectory is:

```text
interpret as a skeletal idea
    -> create_event(title="Research cabinet refinishing")
    -> database returns event_id and stored fields
    -> verify the event exists
    -> tell the user it was saved
```

Useful training and evaluation signals would include:

- Was the create operation appropriate, or was the user merely asking a question?
- Were tentative details preserved as tentative rather than invented?
- Was exactly one event created?
- Did the saved record match the user's wording?
- Did the response accurately describe the database result?
- Did the assistant follow the application's save-first policy?

This is execution feedback in Toolformer's spirit, but the evaluator is durable application state rather than language-model loss.

## What the paper did not establish

Toolformer did not show that:

- a model can safely operate arbitrary APIs;
- self-generated training data removes the need for human review;
- tools eliminate hallucination;
- tool use always beats a larger model;
- models naturally chain many calls into reliable plans;
- a call that improves text prediction satisfies user intent;
- external results are correct or trustworthy;
- model-generated calls should execute without validation;
- read-tool behavior transfers directly to consequential write tools.

The experiments demonstrated learned selection and use of five controlled information tools. Extending that result to an autonomous assistant requires additional training, orchestration, permissions, and state-based evaluation.

## What aged well

- Tool use became a central capability of general-purpose LLM applications.
- Models are commonly trained to emit structured requests for external operations.
- Tool results are routinely returned to the model as new context.
- Synthetic training data and execution-based filtering became important ways to scale post-training.
- Specialized tools remain attractive alternatives to encoding every capability in model parameters.
- Knowing **when not to call** remains as important as knowing call syntax.

## What changed

- Tool schemas are now often supplied dynamically at inference time instead of being limited to a small fixed set learned during one fine-tuning run.
- Production systems commonly support multi-turn, multi-call workflows rather than a single inline insertion.
- Structured function-call protocols often separate tool requests from ordinary prose more clearly than Toolformer's textual bracket format.
- Agent systems add planning, persistence, retries, approvals, and stopping policies around the observation-action loop.
- Tool-use training increasingly combines human demonstrations, synthetic trajectories, execution feedback, preference optimization, and task-specific evaluators.
- Security concerns—especially permissions and hostile retrieved content—have become first-class design requirements.

The exact Toolformer recipe is therefore not a complete blueprint for a modern assistant. Its enduring contribution is the demonstration that models can learn tool-use behavior from automatically generated, environmentally tested examples.

## Reading advice

For a first reading of the original paper, prioritize:

1. **Introduction:** The limitations the authors wanted tools to address.
2. **Section 2, Approach:** The sample–execute–filter–fine-tune pipeline.
3. **Figure 2:** The Pittsburgh example, which captures the method visually.
4. **Section 3, Tools:** The deliberately simple tool interfaces.
5. **Tables 3–7:** Where tools help strongly and where they do not.
6. **Section 5, Analysis:** The importance of decoding policy and imperfect calibration.
7. **Section 7, Limitations:** Especially the lack of tool chains, interactivity, and cost awareness.

The precise sampling hyperparameters matter for reproduction. They are less important for understanding the paper's lasting idea: use the model to propose training examples, but use observable improvement after real execution to decide which examples deserve to survive.

## Recap

Toolformer begins with a straightforward observation: language models are flexible interpreters, but simpler external systems are often better sources of exact calculations, current information, factual lookup, dates, and specialized translations.

Its main contribution is a way to teach that division of labor without manually labeling a massive tool-use dataset. A few demonstrations show the model the form of an API call. The model then inserts many candidate calls into ordinary text. The calls are executed, and only those whose results reduce prediction loss on nearby text are kept. Fine-tuning on the resulting corpus teaches the model to generate calls, accept returned observations, and continue its text around them.

The experiments showed large gains when a tool directly supplied a missing capability, particularly for arithmetic and factual completion. They also exposed the approach's boundaries: results were mixed in some domains, useful examples could be rare, call behavior was sensitive to decoding and wording, and the model could not chain or interactively refine tools.

The paper is best understood as a bridge between language-model training and application execution. It showed that tool behavior could be learned from self-generated examples filtered by real outcomes. Modern systems generalize that insight, but add schemas, multi-step loops, permissions, persistent state, cost controls, and task-specific evaluators. The model learns when to request help; the surrounding software still determines what may run and whether the real goal was achieved.

## Primary and supporting sources

- Schick, T., et al. (2023). [*Toolformer: Language Models Can Teach Themselves to Use Tools*](https://arxiv.org/abs/2302.04761). The primary paper.
- Parisi, A., et al. (2022). [*TALM: Tool Augmented Language Models*](https://arxiv.org/abs/2205.12255). A closely related predecessor using self-supervised tool learning in downstream-task settings.
- Nakano, R., et al. (2021). [*WebGPT: Browser-Assisted Question-Answering with Human Feedback*](https://arxiv.org/abs/2112.09332). An earlier system using human feedback to teach browser-assisted research.
- Yao, S., et al. (2022). [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629). A complementary prompted approach that interleaves reasoning, actions, and observations.
- Karpas, E., et al. (2022). [*MRKL Systems*](https://arxiv.org/abs/2205.00445). A modular account of combining language models with external knowledge and reasoning systems.
