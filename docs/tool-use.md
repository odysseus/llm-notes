---
title: "Tool Use"
type: concept
status: active
updated: 2026-08-05
tags: [tools, agents, function-calling, APIs, orchestration, harnesses, reliability]
---

# Tool Use: Extending LLMs Beyond Text Generation

**Central idea:** A language model can interpret a request and propose what should happen, but its generated tokens do not automatically reveal current facts, perform exact computation, inspect an application, or change the outside world. **Tool use** connects the model to controlled external capabilities. It lets the model request an operation, observe the real result, and revise its response around evidence that did not exist in its original context.

**Why it matters:** Tool use changes an LLM from a self-contained text generator into one component of an interactive software system. It solves several of the model's most important practical limitations, but it also introduces new failure modes, security boundaries, state transitions, and engineering responsibilities. Understanding that changed workflow is essential for distinguishing a chatbot that talks about work from an application that can reliably do work.

## Background topics

- **The LLM request lifecycle:** How instructions, user messages, model outputs, and follow-up calls form one interaction.
- **APIs and functions:** How software exposes named operations with specified inputs and outputs.
- **Structured output and schemas:** How generated tokens can be constrained and validated as data.
- **Context engineering:** How tool descriptions and observations are selected and represented for the model.
- **Retrieval-augmented generation:** How external documents are found and supplied as evidence.
- **Agents and harnesses:** How a model is placed inside a loop that controls execution, state, permissions, and stopping.
- **State machines:** How a request moves among interpretation, action, observation, clarification, and completion.
- **Security boundaries:** Least privilege, confirmation, sandboxing, and the treatment of generated actions as untrusted proposals.
- **Evaluation:** Measuring tool selection, argument accuracy, policy compliance, and final environmental state.

These topics describe the mechanisms around tool use. The central conceptual question is why a probabilistic language model should be combined with ordinary software at all.

## Before tool-using LLMs

Early language-model applications were usually closed systems. The application assembled a prompt, sent it to a model, and displayed the continuation. Everything needed to produce the answer had to come from one of two places:

1. patterns and information encoded in the model's learned parameters; or
2. text explicitly placed into the current prompt.

This was useful for drafting, rewriting, classification, translation, brainstorming, and answering questions about relatively stable knowledge. It also placed a hard boundary around the application. The model could describe how to calculate compound interest, but it was not inherently a calculator. It could suggest what a database query might look like, but it could not see the database. It could explain how to book a flight, but its answer did not create a reservation.

Several limitations followed from this boundary.

First, model parameters are a lossy compression of training experience, not a live knowledge store. A model may recall a common fact, confuse two similar facts, or produce a plausible answer when the relevant event happened after training. Private records—such as a user's tasks, a company's inventory, or a patient's chart—were never present in public pretraining data and should not be embedded in model weights merely to answer a current request.

Second, next-token prediction is not identical to exact execution. A model can infer what arithmetic operation a problem requires and still make a small numerical error. It can understand a sorting or date calculation task but produce an invalid result. A conventional calculator, interpreter, database, or constraint solver may be dramatically more dependable once the problem has been translated into the right representation.

Third, text generation has no intrinsic side effects. A sentence saying “The event has been created” is still only a sentence. Without an external operation and a returned success record, the claimed event does not exist. This distinction between **describing an action** and **causing an action** became central as users began expecting assistants to operate software.

Earlier AI research already offered a broad answer: combine learned components with specialized systems. Search engines retrieve documents; databases hold authoritative state; symbolic programs apply exact rules; robots expose bounded actions. What LLMs added was a flexible natural-language layer capable of deciding which capability a request might require and translating between a user's phrasing and a machine interface.

The modern tool-use lineage became especially visible from 2021 through 2023. WebGPT placed GPT-3 inside a text-based browser so it could search, navigate pages, collect references, and use those references to compose long-form answers ([Nakano et al., 2021](https://arxiv.org/abs/2112.09332)). MRKL proposed a modular architecture combining language models with external knowledge and discrete reasoning modules ([Karpas et al., 2022](https://arxiv.org/abs/2205.00445)). ReAct interleaved reasoning, actions, and new observations rather than treating reasoning and acting as separate phases ([Yao et al., 2022](https://arxiv.org/abs/2210.03629)).

Program-Aided Language Models, or PAL, assigned different parts of a problem to different components: the LLM interpreted natural language and wrote a program, while a Python runtime carried out the exact computation ([Gao et al., 2022](https://arxiv.org/abs/2211.10435)). Toolformer then investigated whether a model could learn not only how to call tools, but whether and when a call would improve its continuation. Its tools included a calculator, search systems, translation, question answering, and a calendar ([Schick et al., 2023](https://arxiv.org/abs/2302.04761)).

Together, these systems established the basic motivation:

> Use the language model for flexible interpretation and decision-making; use external systems for capabilities that should be current, exact, authoritative, specialized, or consequential.

## The topic in one view

Without tools, a model call can be represented as:

$$
y = M(c)
$$

where $M$ is the model, $c$ is the supplied context, and $y$ is generated output. The output is derived only from what the model learned previously and what appears in the current context.

With tools, the model participates in a larger transition process:

$$
a_t = M(c_t, D)
$$

$$
o_t = E(a_t)
$$

$$
c_{t+1} = c_t \oplus a_t \oplus o_t
$$

Here:

- $D$ is a set of tool descriptions and input schemas;
- $a_t$ is a proposed action, such as a search or database lookup;
- $E$ is the external execution environment;
- $o_t$ is the actual observation returned by the tool;
- $\oplus$ means that the action and observation are added to the next model context.

The model can then generate a final answer, request another tool, or ask the user for clarification. The complete behavior is no longer one inference. It is a loop coordinated by the surrounding application:

```text
user request
    → model interprets request and available tools
    → model proposes a tool call
    → application validates and authorizes the proposal
    → tool executes in an external system
    → tool returns an observation
    → model interprets the observation
    → final answer, clarification, or another tool call
```

Three separations are crucial:

1. **The model requests; the application executes.** Generating a tool call does not itself run anything.
2. **The tool returns an observation; the model interprets it.** A successful API response is not automatically a useful user-facing answer.
3. **The external system owns real state.** The database, filesystem, calendar, or service—not the model's prose—determines what actually exists.

Tool use therefore combines two kinds of computation. The model provides flexible semantic reasoning over language. The tool provides a narrower operation with defined behavior. The surrounding [harness](harness-engineering.md) decides how these two are allowed to interact.

## What counts as a tool?

A **tool** is a capability outside the model's ordinary token generation that the application makes available through an interface. The model is told enough about that interface to request its use.

Tools can include:

| Tool category | Example | What it contributes |
| --- | --- | --- |
| Retrieval | Search engine, vector search, document lookup | Information not already in context |
| Computation | Calculator, Python runtime, symbolic solver | Exact or executable operations |
| State inspection | Database query, file read, application status | Authoritative current state |
| State mutation | Create task, update record, send message | Real-world or application effects |
| Perception | Browser screenshot, camera, sensor, OCR | Observations of an environment |
| Specialized model | Translation, speech recognition, vision model | A capability delegated to another learned system |
| Verification | Test runner, schema validator, policy checker | Evidence that a proposed result satisfies a condition |

The word *tool* describes the interface from the model's perspective, not the internal complexity of the implementation. A tool might wrap one deterministic function, an entire SaaS API, a database transaction, another model, or a carefully controlled multi-step service.

### Tool calls are structured model outputs

In many modern APIs, a tool definition resembles a named function with a description and a JSON schema:

```json
{
  "name": "get_event",
  "description": "Return one saved event by its stable identifier.",
  "parameters": {
    "type": "object",
    "properties": {
      "event_id": {
        "type": "string",
        "description": "The identifier returned by search_events."
      }
    },
    "required": ["event_id"],
    "additionalProperties": false
  }
}
```

The model may return something conceptually like:

```json
{
  "tool": "get_event",
  "arguments": {"event_id": "evt_1842"}
}
```

This is still generated output. The application must parse it, validate it, check permissions, call the underlying function, and return the result. A convenient model API can hide some serialization details, but it does not remove this authority boundary.

### Tool use is broader than function calling

**Function calling** is a common protocol for expressing a tool request. **Tool use** is the full behavior around choosing, executing, observing, and incorporating an external capability.

A model that emits valid JSON has demonstrated structured generation. A system that executes a safe lookup and uses the returned record to answer the user has completed a tool-use cycle. A model that operates a browser through repeated screenshots, clicks, and text entry is also using tools even though the interface is not naturally described as one business function.

## The motivations behind tool use

Tool use is sometimes described as giving a model “more capabilities.” That is true but too general to guide architecture. Different tools solve different limitations, and the benefit should be stated precisely.

## 1. Access information that is current, private, or too large to memorize

A model's parameters cannot be the authoritative source for rapidly changing facts or private application data. Search and retrieval tools let the system obtain information at inference time.

This solves several distinct problems:

- **Recency:** prices, schedules, software documentation, policies, and news change after training.
- **Privacy and locality:** personal tasks, internal documents, customer records, and account state should remain in controlled systems.
- **Scale:** a document collection may be much larger than one context window.
- **Attribution:** retrieved sources can be shown or cited, making the evidentiary basis inspectable.
- **Correctable knowledge:** updating a database or document corpus is easier than retraining a model.

WebGPT illustrated the importance of treating retrieval as an activity rather than a static prompt. The model could reformulate searches, visit pages, follow links, and collect references before answering. Each observation influenced what it did next. This is qualitatively different from asking a closed model to recall everything from its parameters.

Retrieval does not guarantee truth. Search results may be irrelevant, outdated, contradictory, or malicious. The model must still compare evidence, and the application must retain source metadata and distinguish retrieved content from instructions.

## 2. Delegate exact computation to systems built for it

Language models often understand the structure of a calculation better than they perform every operation within it. Tools allow a useful division of labor:

- the model maps an informal question into variables, operations, or code;
- a calculator, interpreter, query engine, or solver executes those operations exactly;
- the model explains the result in the user's terms.

PAL made this separation explicit. Rather than relying on a language model both to decompose a problem and to carry out arithmetic through text generation, it used the model to produce runnable steps and delegated execution to Python. This is a general pattern, not merely a math trick.

For example:

| Flexible interpretation | Exact execution |
| --- | --- |
| “Compare these mortgage options after fees” | Financial formulas and numerical calculation |
| “Find orders from customers who bought both products” | SQL query and relational operations |
| “Schedule these jobs without overlapping machines” | Constraint solver |
| “Does this patch preserve existing behavior?” | Compiler and test suite |

The tool does not make the model's interpretation correct. It prevents the system from asking a probabilistic generator to imitate an executor once the correct operation has been identified.

## 3. Observe the actual environment

Many tasks cannot be completed from a request alone. The system must inspect what is currently present.

A coding assistant may need to list files, read source code, search for symbols, run tests, and inspect error output. A support assistant may need to query an account, view an order, or check whether a service is degraded. A browser agent may need to observe the current page before deciding where to click.

This creates a feedback loop:

$$
\text{hypothesis} \rightarrow \text{inspection} \rightarrow \text{observation} \rightarrow \text{revised hypothesis}
$$

The observation can disconfirm the model's initial assumption. That is one of the deepest benefits of tool use. A closed response must commit based on its initial context; a tool-using system can gather evidence before committing.

ReAct formalized this interleaving for language tasks and interactive environments. Actions did not merely implement a completed plan. They supplied new information that changed later reasoning.

## 4. Produce real effects rather than descriptions

Read tools expand what the model can know. Write tools expand what the overall system can do.

Examples include:

- saving or updating an event;
- sending a message;
- creating a calendar entry;
- purchasing an item;
- modifying a file;
- deploying software;
- controlling a device.

This is the transition from an advisory chatbot to an operational assistant. It is also where risk increases sharply. A mistaken answer can misinform a user; a mistaken write can delete data, spend money, expose information, or contact another person.

The model should therefore not be treated as the final authority over side effects. It proposes intent and arguments. Deterministic software checks identity, permissions, policy, confirmations, and invariants before executing. High-impact actions should often require an explicit user approval step.

## 5. Verify claims against external evidence

Tools are not only for gathering inputs or causing effects. They can check whether a proposed result is actually correct.

A model can:

- run tests after editing code;
- query the database after a write;
- validate generated data against a schema;
- ask a calculator to check a numerical result;
- compare a claim with retrieved primary sources;
- render a document and inspect the output.

This changes the standard of completion. Without verification, the model may stop because its answer sounds finished. With verification, the harness can require an observable condition: the tests pass, the record has the intended fields, the file exists, or the cited source supports the claim.

The strongest tool-using systems use **closed-loop execution**:

```text
propose → execute → inspect → correct → verify → report
```

Verification is not infallible. Tests can be incomplete and validators can check the wrong property. It is nevertheless a major improvement over accepting fluent self-report as proof.

## 6. Add or update capabilities without retraining the whole model

Tools make an LLM application modular. A new business operation can often be added by implementing an API, describing it clearly, and granting it under the right conditions. Current product state remains in ordinary software rather than being learned into model weights.

MRKL emphasized this modular neuro-symbolic architecture. Specialized modules could be selected for knowledge or reasoning tasks for which a general language model was poorly suited. Gorilla later focused on accurately generating calls across large collections of changing APIs and showed the importance of retrieving current API documentation rather than assuming the model had memorized it ([Patil et al., 2023](https://arxiv.org/abs/2305.15334)).

Modularity does not eliminate training. Models may be instruction-tuned or specifically trained for tool selection and argument generation, as Toolformer and ToolLLM explored ([Qin et al., 2023](https://arxiv.org/abs/2307.16789)). The architectural benefit is that the external capability and its state do not have to be absorbed into the model itself.

## 7. Turn natural language into a general interface for software

Traditional applications require users to navigate menus, forms, commands, and domain-specific syntax. A tool-using LLM can translate an underspecified natural-language goal into calls to those existing systems.

This does not mean every interface should become a chatbot. Forms remain better when the user already knows the exact operation and needs predictability. Natural language is especially valuable when:

- the user's goal does not map cleanly to one screen;
- several systems must be coordinated;
- relevant parameters are distributed through a conversation;
- the user knows the desired outcome but not the software procedure;
- the system should explain what it did and why.

The model becomes a semantic adapter between human intent and machine contracts. The tools remain ordinary, inspectable software interfaces underneath.

## How the LLM workflow changes

Tool use changes both the data passed to the model and the control flow around the model.

### The one-call workflow

A simple, tool-free application may perform these steps:

1. Combine system instructions, conversation history, and the user message.
2. Send that context to the model.
3. Receive assistant text.
4. Display it.

The model has one opportunity to answer. The application is primarily a message assembler and renderer.

### The tool-aware workflow

A tool-using turn commonly performs these steps:

1. **Interpret the request.** Determine which policies, state, and capabilities may be relevant.
2. **Select available tools.** Expose only the tools permitted and useful for this turn.
3. **Describe their contracts.** Supply names, descriptions, argument schemas, and sometimes examples.
4. **Call the model.** Permit it to return text, a tool request, or another protocol-defined response.
5. **Parse the proposal.** Treat the returned tool name and arguments as untrusted generated data.
6. **Validate structure and meaning.** Check schemas, identifiers, dates, ranges, and domain rules.
7. **Apply policy.** Enforce authorization, confirmation, rate limits, budgets, and risk controls.
8. **Execute the tool.** Call the real service with controlled credentials and timeouts.
9. **Record the observation.** Preserve success, failure, returned data, and state changes.
10. **Call the model again.** Add the observation to context so the model can interpret it.
11. **Continue or stop.** Allow another tool call, request clarification, recover from an error, or return the answer.

The workflow has become a state machine. The application must distinguish among at least these outputs:

| Model output | Harness response |
| --- | --- |
| Final answer | Return it if completion requirements are satisfied |
| Read-only tool request | Validate, execute, and return the observation |
| Consequential write request | Validate, authorize, and possibly ask for approval |
| Missing required information | Ask the user a focused clarification question |
| Invalid tool request | Return a structured error or ask the model to revise |
| Repeated or unproductive calls | Stop according to budgets and recovery policy |

### The tool result becomes new context

Suppose the user asks, “What do I still need to do for the kitchen remodel?” The model does not know the user's saved events. It might first request:

```json
{
  "tool": "search_events",
  "arguments": {"query": "kitchen remodel", "status": "open"}
}
```

The application executes the search and returns:

```json
{
  "matches": [
    {"id": "evt_31", "title": "Choose countertop material", "status": "open"},
    {"id": "evt_44", "title": "Get island electrical quote", "status": "open"}
  ]
}
```

Only then can the model compose a grounded response. The second model call has a different epistemic position from the first: it now has an observation from the authoritative store.

If the user instead says, “Mark the countertop decision complete,” the workflow may require another read to resolve the record, a confirmation if several matches exist, an update call using the stable identifier, and a final read or returned version to verify the change.

### A compact implementation

The orchestration can be expressed without committing to a particular framework:

```python
def run_turn(user, message, state):
    tools = registry.allowed_tools(user=user, message=message)
    context = build_context(user, message, state, tools)

    for step in range(MAX_STEPS):
        response = model.generate(context=context, tools=tools)

        if response.kind == "final":
            return require_supported_completion(response, state)

        call = parse_tool_call(response)
        validate_schema(call)
        validate_domain_rules(call, state)
        authorize(user, call)

        if requires_confirmation(call):
            return ask_for_confirmation(call)

        observation = execute_with_timeout_and_idempotency(call)
        state.record(call, observation)
        context = append_observation(context, call, observation)

    return stop_with_budget_exhausted(state)
```

The model supplies flexible choices inside the loop. The surrounding code owns the loop itself, tool availability, validation, credentials, side effects, budgets, and stopping.

## Common tool-use architectures

Tool use does not require one universal agent loop. The appropriate pattern depends on how predictable the task is.

### Direct tool routing

The model chooses one tool and supplies arguments; the application executes it and returns a final response. This works well for requests such as checking weather, looking up one record, or performing one calculation.

**Benefit:** low latency and a small failure surface.

**Limitation:** cannot naturally adapt when the first observation reveals another necessary step.

### Deterministic workflow with model-filled slots

Application code defines the sequence, while the model performs bounded semantic tasks within it. A support workflow might always identify the customer, retrieve the order, apply policy, and then generate an explanation.

**Benefit:** predictable control flow and easier testing.

**Limitation:** less flexible when users request unusual combinations of operations.

### ReAct-style observation–action loop

The model repeatedly chooses an action based on the request and observations gathered so far. This is useful for research, diagnosis, browser navigation, and exploratory coding, where the correct next step depends on what the previous step reveals.

**Benefit:** adapts to unknown intermediate state.

**Limitation:** adds cost, latency, and the possibility of loops, drift, or unnecessary calls.

### Planner–executor workflow

One model call proposes a plan; deterministic code or another configured model executes each step through tools. The system may replan after failures.

**Benefit:** makes long tasks and progress more explicit.

**Limitation:** initial plans are often invalidated by the first real observation, so rigid adherence can be worse than incremental planning.

### Code as an intermediate tool language

The model writes code that combines computation or several APIs, and a controlled runtime executes it. PAL is a focused example of this pattern.

**Benefit:** concise composition, variables, loops, and exact operations.

**Limitation:** executing generated code creates a much larger security and resource-control problem than calling a narrow function.

The least agentic architecture that handles the task is usually the easiest to make reliable. A tool loop is justified when intermediate observations genuinely determine later actions, not merely because a framework makes looping convenient.

## Tool use, RAG, structured output, and agents

These concepts overlap but should not be collapsed.

| Concept | Main purpose | Does it require external execution? | Who commonly controls the next step? |
| --- | --- | --- | --- |
| **Structured output** | Produce machine-parseable data | No | Application |
| **Function calling** | Express a request to invoke a named capability | Execution occurs outside the model | Application and model |
| **RAG** | Supply retrieved evidence to generation | Yes, retrieval | Often application-defined; sometimes model-selected |
| **Tool use** | Observe or act through external capabilities | Yes | Model proposes; harness controls |
| **Agent** | Pursue a goal through repeated observation and action | Usually | A model–harness loop |

[Retrieval-augmented generation](retrieval-augmented-generation.md) can be implemented as a fixed pipeline: the application always retrieves before generation. It can also be exposed as a tool so the model decides whether to search, how to reformulate a query, and whether more evidence is needed. The former is simpler and predictable; the latter is adaptive but more difficult to evaluate.

Structured output can exist without tools. A model might produce a validated `EventDraft` that the application shows to the user without saving it. Conversely, tool use usually relies on structured output because tool names and arguments must cross a software boundary reliably.

An agent is not simply a model with one function schema. The label becomes useful when the system can perform multiple steps, use observations to revise its behavior, maintain task state, and decide when it has reached a goal.

## Designing tools for models

A tool is an **agent–computer interface**. Its design changes how difficult the model's task is. SWE-agent showed that interfaces designed around model capabilities could materially improve software-engineering performance even without changing the underlying language-model task ([Yang et al., 2024](https://arxiv.org/abs/2405.15793)).

### Prefer clear, task-level operations

Tools should correspond to meaningful actions with unambiguous names. `reschedule_event(event_id, new_start)` is easier to select and safer to validate than a generic `modify(table, filter, values)` operation.

Too-broad tools expose unnecessary power and force the model to reconstruct business rules. Too-narrow tools may require long, fragile sequences. The right level usually matches a domain transaction that the application already understands.

### Treat descriptions as part of the interface

A tool description should state:

- what the operation does;
- when it should and should not be used;
- what each argument means;
- which identifier source is authoritative;
- whether the call reads or changes state;
- what errors and result variants can occur.

Models can hallucinate APIs or use stale calling conventions. Gorilla's retrieval-aware experiments highlight why current tool documentation matters when interfaces change.

### Return semantic observations

Raw API responses are often poor model context. They may contain internal fields, large payloads, unstable error codes, or sensitive information. A tool adapter should return the smallest semantic result needed for the next decision:

```json
{
  "status": "conflict",
  "reason": "Two open events match the title.",
  "candidates": [
    {"id": "evt_31", "title": "Choose kitchen countertop"},
    {"id": "evt_77", "title": "Choose bathroom countertop"}
  ],
  "next_action": "Ask the user which event they mean."
}
```

This is more useful than leaking an ORM exception or an entire database row.

### Separate reads from writes

Read tools gather information; write tools change authoritative state. Keeping them separate supports clearer permission policies, approval rules, tracing, and evaluation. The application can expose broad read access while limiting writes to narrow, reversible operations.

### Make retries safe

Tool calls can time out after succeeding. If the harness blindly retries `send_message` or `create_order`, one user request may produce duplicates. Write tools should use idempotency keys, stable request identifiers, or read-after-write reconciliation when possible.

## Safety and authority

Tool use magnifies both usefulness and risk because generated text can become executable intent.

### Model output is a proposal, not permission

Schema validity is not authorization. A perfectly formed request to delete a file can still be impermissible. The execution layer must check the authenticated user, allowed scope, current state, and action risk independently of the model.

Prompt instructions such as “ask before sending” help the model behave appropriately, but important restrictions must be enforced in code. The application should be able to reject a call even when the model confidently requests it.

### External observations can contain hostile instructions

Retrieval and browsing introduce a distinction that language models do not naturally enforce: external text may be data to analyze, not instructions to obey. Indirect prompt injection attacks place malicious directions inside documents or pages likely to be retrieved. Greshake and colleagues demonstrated that such content could manipulate how LLM-integrated applications use other APIs ([Greshake et al., 2023](https://arxiv.org/abs/2302.12173)).

Mitigation requires defense in depth:

- label tool results as untrusted data;
- expose only necessary tools and fields;
- keep credentials outside model context;
- validate every consequential call;
- require approval for sensitive actions;
- isolate generated code and browser sessions;
- prevent retrieved text from silently widening permissions;
- log both proposed and executed actions.

No prompt can turn arbitrary external content into trustworthy instructions.

### Ambiguity should block consequential action

Natural language is often incomplete. “Cancel my appointment” may refer to several records. A good assistant resolves the ambiguity before acting. The ability to ask a user a clarifying question is itself part of the tool-use policy.

The safest behavior is not always refusal and not always autonomy. It is proportional control: freely perform reversible reads, confirm consequential writes, and stop when the target or intent is not sufficiently identified.

## Failure modes and tradeoffs

### Unnecessary tool use

A model may search for stable information already in context, call a calculator for trivial work, or repeatedly inspect the same state. Every call adds latency, cost, and another failure point. Tools should be used when their external contribution matters.

### Incorrect tool selection

Similar names or overlapping descriptions can cause the model to choose the wrong capability. Limiting the visible tool set by intent is often better than presenting hundreds of tools on every turn.

### Valid syntax with wrong semantics

JSON validation can confirm that `amount` is a number without confirming that it is in the right currency, applies to the right account, or matches the user's intent. Domain validation is indispensable.

### Hallucinated success

The model may claim that an action succeeded even when no call occurred or the tool returned an error. User-facing claims should be grounded in recorded observations. For important writes, completion should correspond to verified external state.

### Error amplification through retries

An agent may respond to a failure by repeating the same call, changing arguments without justification, or taking a more destructive path. Retry count, total calls, elapsed time, and cost require explicit limits.

### Observation overload

Large tool results consume context and obscure the relevant evidence. Adapters should filter and summarize without losing identifiers, provenance, or error details needed for correct decisions.

### Increased latency and cost

One user turn may now require several model calls plus network or program execution. Parallel reads, tool-result caching, smaller routing models, and deterministic workflows can help, but each optimization introduces its own consistency questions.

### New security exposure

A closed chatbot can produce harmful language; a tool-using system may also leak data or cause financial and operational damage. ToolEmu identified realistic high-stakes failure scenarios involving tool-integrated agents, illustrating why safety evaluation must include actions rather than only final text ([Ruan et al., 2023](https://arxiv.org/abs/2309.15817)).

### Capability does not imply reliability

A model may successfully complete a tool task once and fail on a small variation. $\tau$-bench evaluates conversational agents against policies and final database state, emphasizing consistent task completion rather than plausible dialogue ([Yao et al., 2024](https://arxiv.org/abs/2406.12045)). Tool use should be judged as a software behavior under repeated trials, not as an impressive demonstration trajectory.

## When not to use a tool

Tool use is unnecessary when the model already has everything needed and the task is purely generative. Rewriting a paragraph, brainstorming names, explaining a timeless concept, or changing tone may require no external operation.

A deterministic application call may also be better than model-directed tool selection when the workflow is already known. If every uploaded document must be virus-scanned, do not ask the model whether scanning is appropriate. If every account page must load the same customer record, ordinary application logic should load it.

Avoid model-directed tool use when:

- the correct operation can be selected reliably with a simple rule;
- the tool adds no information, precision, verification, or effect;
- the action is too risky to delegate and has no adequate approval boundary;
- the interface cannot return enough information for the model to detect failure;
- the environment is unavailable or too unstable for safe automation;
- tool latency costs more than the expected improvement.

The architectural question is not “Can the model call this?” but “What uncertainty or capability gap does model-directed tool use resolve?”

## Evaluating tool-using systems

Final-answer quality alone hides where a tool workflow failed. Evaluation should separate the stages.

| Evaluation layer | Key question | Example metric |
| --- | --- | --- |
| Need recognition | Did the model know whether a tool was necessary? | Appropriate call/no-call accuracy |
| Tool selection | Did it choose the right capability? | Tool-name accuracy |
| Argument generation | Were the inputs complete and correct? | Exact match, schema and domain validity |
| Policy compliance | Was the call permitted and properly confirmed? | Violation rate |
| Execution handling | Did it interpret success, failure, and partial results correctly? | Recovery success |
| Task outcome | Did the environment reach the intended state? | State-based success rate |
| Response grounding | Does the final explanation match observed results? | Unsupported-claim rate |
| Consistency | Does the system succeed repeatedly? | Pass rate across repeated trials |
| Efficiency | Did it use reasonable resources? | Calls, tokens, latency, and cost per success |

For write operations, the final database or application state is usually a stronger evaluator than the transcript. For research, citations and source support matter. For code, tests and artifact inspection matter. Evaluation should match the reason the tool was introduced.

Tools themselves also need contract tests. The harness should test permission denials, malformed inputs, timeouts, partial failure, duplicate calls, stale identifiers, and unexpected but valid results. A model cannot recover reliably from an error that the tool reports ambiguously.

## Tool use in the task-and-idea chatbot

The single-user task-and-idea prototype provides a useful boundary between language reasoning and application state.

The model is well suited to:

- interpreting whether the user is describing a task, idea, or question;
- extracting tentative fields from informal language;
- recognizing ambiguity;
- choosing whether to search, create, or update;
- explaining what was found or changed.

Tools should own:

- saving an event;
- retrieving events from the database;
- resolving stable identifiers;
- updating fields under domain rules;
- recording exact timestamps and versions;
- returning the authoritative saved result.

A small MVP tool set might be:

```text
search_events(query, status?, date_range?)
get_event(event_id)
create_event(draft, idempotency_key)
update_event(event_id, patch, expected_version)
```

The user can say, “I should look into getting the kitchen cabinets refinished.” The model interprets this as a skeletal idea and proposes `create_event`. The harness validates the draft, executes the write, and returns the new identifier. The model can then truthfully say that the idea was saved and ask whether the user wants to add a target date.

Later, “Add a note that I want Rocky River green” first requires retrieval. The model searches for the relevant event, asks for clarification if several kitchen items match, and updates the selected record using its identifier and expected version. The model should not edit an event reconstructed from conversational memory when the database can supply the current record.

This architecture supports the primary MVP goal: conversational flexibility over structured, durable state. The model interprets the user; the tools protect the data model; the harness controls the transaction.

## Practical design principles

1. **Name the limitation each tool solves.** Current knowledge, exact computation, environmental observation, side effects, and verification require different designs.
2. **Keep authoritative state outside the model.** The model's statement is not the database record, file, booking, or sent message.
3. **Treat tool calls as untrusted proposals.** Parse, validate, authorize, and execute them in application code.
4. **Expose the smallest useful capability set.** Relevant, well-documented tools are easier to select and safer to operate.
5. **Use stable identifiers.** Natural-language names help people; IDs protect mutations from ambiguity.
6. **Return observations, not implementation debris.** Preserve semantic status, relevant data, provenance, and actionable errors.
7. **Separate reads, writes, and approvals.** Their risks and retry behavior differ.
8. **Make writes idempotent where possible.** A retry should not duplicate an external effect.
9. **Require observable evidence of completion.** Verify state instead of trusting the model's claim.
10. **Bound every loop.** Limit steps, time, cost, retries, and tool scope.
11. **Treat external content as data.** Retrieved text does not gain authority by entering the context window.
12. **Evaluate the trajectory and the outcome.** Correct prose can conceal incorrect calls or state.
13. **Prefer deterministic flow when the procedure is already known.** Reserve model choice for genuine semantic uncertainty.

## What tool use does not solve

Tools do not make a model omniscient. A search tool can retrieve poor sources. A calculator can exactly compute the wrong formula. A database can return the wrong record if the query is ambiguous. A test suite can pass while omitting the behavior that matters.

Tools also do not make a model autonomous by themselves. Someone must define the available capabilities, construct context, grant authority, execute calls, preserve state, handle failures, and decide when the process stops. Those are responsibilities of [harness engineering](harness-engineering.md).

Nor does tool use remove the need for model capability. The model must still understand the request, select a suitable operation, form arguments, and interpret results. Tool use is a division of labor, not a replacement for reasoning.

Finally, tools cannot convert an underspecified goal into a safe action without additional information. The right response to uncertainty is often a question, not another API call.

## Recap

A standalone LLM generates an answer from learned parameters and supplied context. Tool use connects that model to external systems that can retrieve information, compute exactly, observe current state, cause real effects, and verify results.

The motivation is not that APIs are fashionable. It is that language models and conventional software have complementary strengths. Models are unusually good at mapping flexible human language onto possible intentions and procedures. Databases, calculators, interpreters, search systems, validators, and transactional services are better at maintaining authoritative state and executing defined operations.

This combination changes the workflow from one generation into an observation–action cycle. The model receives tool contracts, proposes a call, and later sees the returned observation. The application validates and authorizes the proposal, performs the operation, records its effects, and determines whether another step is allowed. Every tool therefore adds both capability and a new systems boundary.

Good tool-use design begins by stating what problem an external capability solves. It then gives the model a clear interface while keeping permissions, credentials, exact state, side effects, retries, verification, and stopping under deterministic control. The goal is not maximum autonomy. It is a reliable division of labor between semantic reasoning and executable software.

## Key sources

- Karpas, E., et al. (2022). [*MRKL Systems: A Modular, Neuro-Symbolic Architecture That Combines Large Language Models, External Knowledge Sources and Discrete Reasoning*](https://arxiv.org/abs/2205.00445).
- Nakano, R., et al. (2021). [*WebGPT: Browser-Assisted Question-Answering with Human Feedback*](https://arxiv.org/abs/2112.09332).
- Yao, S., et al. (2022). [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629).
- Gao, L., et al. (2022). [*PAL: Program-Aided Language Models*](https://arxiv.org/abs/2211.10435).
- Schick, T., et al. (2023). [*Toolformer: Language Models Can Teach Themselves to Use Tools*](https://arxiv.org/abs/2302.04761).
- Patil, S. G., et al. (2023). [*Gorilla: Large Language Model Connected with Massive APIs*](https://arxiv.org/abs/2305.15334).
- Qin, Y., et al. (2023). [*ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs*](https://arxiv.org/abs/2307.16789).
- Yang, J., et al. (2024). [*SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*](https://arxiv.org/abs/2405.15793).
- Greshake, K., et al. (2023). [*Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*](https://arxiv.org/abs/2302.12173).
- Ruan, Y., et al. (2023). [*Identifying the Risks of LM Agents with an LM-Emulated Sandbox*](https://arxiv.org/abs/2309.15817).
- Yao, S., et al. (2024). [*$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*](https://arxiv.org/abs/2406.12045).
