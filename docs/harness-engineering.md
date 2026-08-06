---
title: "Harness Engineering"
type: concept
status: active
updated: 2026-08-05
tags: [agents, harnesses, orchestration, tools, context, evaluation, reliability]
---

# Harness Engineering: Building the System Around an LLM

**Central idea:** A language model proposes interpretations, language, and actions; a **harness** is the surrounding software system that decides what the model sees, which actions it may request, whether those actions are executed, how results return to the model, what state persists, and when the process must stop.

**Why it matters:** Once an LLM application does more than produce a one-shot answer, much of its quality no longer comes from the model alone. Reliability, safety, continuity, tool use, cost, and recoverability depend on the runtime wrapped around it. Harness engineering turns a capable but probabilistic model into a controlled application.

## Background topics

- **The LLM request lifecycle:** How messages, tool definitions, model outputs, and tool results form a multi-call interaction.
- **Prompt and context engineering:** How instructions and working information are selected, represented, and ordered.
- **Structured output and schemas:** How model-generated text becomes machine-validated data.
- **APIs and tool calling:** How software exposes external capabilities to a model.
- **State machines and control flow:** How a process moves among planning, acting, observing, validating, and stopping.
- **Databases and transactional state:** How authoritative application facts persist outside the model's context window.
- **Security boundaries:** Permissions, least privilege, sandboxing, confirmation, and the treatment of model output as untrusted input.
- **Observability and evaluation:** Traces, replay, test environments, graders, and outcome-based metrics.

These topics explain individual parts of a harness. Harness engineering is the systems problem of making those parts work together.

## Before harness engineering became a distinct concern

Early applications built on large language models were often thin wrappers around a single inference call. The application collected a user message, prefixed an instruction, sent both to a model, and displayed the returned text. For drafting, summarization, classification, or question answering over stable information, this could be enough.

The model itself was the center of attention. If an answer was poor, developers adjusted the prompt, changed the sampling parameters, added examples, fine-tuned a model, or upgraded to a more capable one. Application code handled ordinary concerns such as HTTP requests and rendering, but it did not yet appear to be a major source of intelligence.

Tool-using and retrieval systems changed that picture. A useful assistant needed access to information not contained in model weights, and users wanted it to do more than recommend an action: they wanted it to search, calculate, write a file, query a database, or update an application. Research systems such as MRKL proposed combining language models with external knowledge and discrete reasoning modules ([Karpas et al., 2022](https://arxiv.org/abs/2205.00445)). ReAct made a repeated reasoning-and-action pattern explicit, allowing new observations to influence later model calls ([Yao et al., 2022](https://arxiv.org/abs/2210.03629)). Toolformer explored training a model to decide when and how to call APIs ([Schick et al., 2023](https://arxiv.org/abs/2302.04761)).

These developments made the surrounding code impossible to ignore. The model could emit something resembling `create_event(title="Service watch")`, but some other component still had to decide:

- whether `create_event` was available to this user;
- whether the arguments were valid;
- whether an ambiguous date required clarification;
- whether the action needed confirmation;
- how duplicate requests were handled;
- what happened if the database timed out;
- what result the model saw afterward;
- whether another model call was allowed;
- how the entire attempt could be inspected later.

Ad hoc agent demonstrations often placed this logic in one short loop. That was enough to prove that a model could act, but not enough to make the system predictable. As tasks became longer, the system also had to preserve progress across context limits, expose usable views of the environment, recover from partial failure, and distinguish actual completion from a model's claim that it had finished.

The term **harness engineering** became prominent in practitioner writing during 2025–2026, especially around long-running coding agents. Anthropic described harnesses that preserve structured progress and give later sessions a clean environment ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)). OpenAI used the term for work that makes a codebase, its documentation, tools, tests, logs, and development environment legible to coding agents ([Lopopolo, 2026](https://openai.com/index/harness-engineering/)). A 2026 research paper offered a broader definition: harness engineering designs and evaluates the external execution system that controls context, tools, state, validation, recovery, and stopping around a model ([Pan et al., 2026](https://arxiv.org/abs/2603.25723)).

The vocabulary is still settling. *Harness*, *runtime*, *scaffold*, *orchestrator*, and sometimes *agent framework* overlap. The enduring engineering idea is more important than the label: model behavior is shaped by an executable environment, and that environment must be designed as carefully as the prompt.

## The topic in one view

A model call can be written abstractly as:

$$
y = M(c)
$$

where $M$ is the model, $c$ is the supplied context, and $y$ is generated output. The model has no automatic access to the application's database, filesystem, current time, or external services. It only receives the representations supplied in $c$ and returns tokens or structured data.

A harness turns that one call into a controlled process:

$$
(s_{t+1}, o_t) = H(s_t, u, M)
$$

where:

- $u$ is the user's request;
- $s_t$ is the application's current state;
- $H$ is the harness;
- $o_t$ is an answer, action result, clarification request, or error;
- $s_{t+1}$ is the state after validated actions have occurred.

The harness may call the model once or many times. On each step it chooses context, exposes tools, interprets the model's proposed action, applies deterministic policy, executes permitted operations, records observations, and decides whether to continue.

A useful shorthand is:

$$
\text{deployed agent behavior}
\approx
\text{model capability}
+
\text{harness design}
+
\text{environment quality}
$$

This is not a claim that each part contributes independently or additively. They interact. A powerful model can be crippled by vague tools and missing state. A carefully engineered harness cannot supply reasoning abilities the model does not have. An accurate model and sound controller can still fail if the environment provides stale documentation or unverifiable actions.

The central design rule follows:

> Let the model handle interpretation and flexible judgment; let the harness own authority, exact state, execution, and verification.

## Model, prompt, context, agent, and harness

These terms are related but describe different layers.

| Layer | What it is | Primary responsibility |
| --- | --- | --- |
| **Model** | A learned function that generates output from supplied input | Language understanding, generation, flexible inference, and action proposals |
| **Prompt** | A particular instruction, question, or example | Expressing the immediate task or behavior |
| **Context** | Everything available to the model during one call | Providing working instructions, evidence, history, state, and tool descriptions |
| **Harness** | The executable system around model calls | Constructing context, controlling tools, executing actions, persisting state, validating, recovering, and stopping |
| **Agent** | The complete goal-directed process that can observe, act, and revise | Attempting a task through the combined behavior of model and harness |
| **Framework** | Reusable software used to build harnesses or workflows | Supplying abstractions, integrations, persistence, tracing, and deployment support |

The boundaries are not perfectly standardized. Some authors call the complete model-plus-runtime system the agent; others use *agent* for a configured model inside a larger orchestration system. In either vocabulary, the important distinction is operational: the model proposes; external code grants capabilities and produces effects.

### Harness engineering versus prompt engineering

Prompt engineering asks how an instruction should be worded. Harness engineering asks what system should surround that instruction.

For example, the rule “do not delete an event without confirmation” can appear in a prompt. A harness-level implementation also ensures that the delete tool rejects calls without a valid confirmation token. The prompt guides model behavior; the enforcement point prevents an unsafe state change even when the model misunderstands or ignores the guidance.

### Harness engineering versus context engineering

[Context engineering](context-engineering.md) is one major responsibility of the harness. It selects and shapes what the model sees. Harness engineering additionally covers what happens outside the model call: tool execution, permission checks, persistence, retries, budgets, validation, telemetry, and lifecycle control.

### Harness engineering versus orchestration

Orchestration usually emphasizes control flow: which model, tool, or subtask runs next. Harness engineering includes orchestration but also asks whether the environment is understandable, actions are safe, outcomes are verifiable, and failures are recoverable. A graph of model calls can be part of a harness without being the whole harness.

## A harness as a control plane

The best analogy is not a smarter prompt but an operating layer around a probabilistic worker. The worker is capable of interpreting unusual requests, but the control plane defines its workspace, instruments, rules, records, and checks.

For a typical tool-using turn, the harness performs some version of the following sequence:

1. **Accept the request.** Record the user message, identity, conversation, and request metadata.
2. **Load authoritative state.** Fetch the records and policies that determine what currently exists and what is permitted.
3. **Assemble context.** Select instructions, relevant history, memories, evidence, examples, and tool schemas.
4. **Call the model.** Request either an answer or one or more structured action proposals.
5. **Parse and validate.** Confirm that the response matches the expected protocol and that tool arguments satisfy their schemas.
6. **Apply policy.** Check permissions, risk level, approval requirements, rate limits, budgets, and semantic invariants.
7. **Execute permitted actions.** Run tools against the real environment, preferably with timeouts and idempotency protections.
8. **Return observations.** Give the model explicit success, failure, or partial-result data.
9. **Continue or stop.** Permit another model call, ask the user for clarification, retry safely, escalate, or return the final result.
10. **Persist the trace.** Store enough information to debug, evaluate, resume, and explain the run.

The [LLM request lifecycle](llm-request-lifecycle.md) describes how these steps appear within one interaction. Harness engineering treats the full sequence as a product surface that can be designed, tested, and improved.

## The core responsibilities of a harness

### 1. Context construction

The harness determines the model's temporary world. It chooses which instructions are active, which conversation turns remain, what data is retrieved, which tools are visible, and how all of it is labeled.

This is not a clerical task. If an event-updating assistant sees an old conversational summary but not the current database row, it may confidently modify the wrong record. If every available tool schema is loaded on every turn, irrelevant capabilities consume context and make selection harder. If retrieved documents are inserted without clear boundaries, their text may be confused with instructions.

Good harnesses therefore make context assembly explicit and inspectable. The selected objects should exist as data before being rendered into model messages:

```python
turn_context = TurnContext(
    instructions=policy.for_intent(intent),
    records=events.resolve_references(user_text),
    memories=memory.search(user_text, top_k=4),
    evidence=research.retrieve(user_text, top_k=6),
    tools=tool_registry.allowed_for(intent),
)
```

This separation makes it possible to test selection independently from generation. A failed answer can then be classified as missing context, misleading context, incorrect model use, or incorrect execution rather than being called vaguely “a prompt problem.”

### 2. Tool and action contracts

A tool is an interface between probabilistic language generation and deterministic software. Its name, description, argument schema, error behavior, and result shape form an **agent-computer interface**.

SWE-agent demonstrated that changing this interface can materially change agent behavior and performance, even when the underlying model and task are held constant ([Yang et al., 2024](https://arxiv.org/abs/2405.15793)). The lesson generalizes beyond coding. A calendar tool called `change` with loosely described string arguments is harder to use reliably than separate, precisely documented actions such as `find_event`, `reschedule_event`, and `cancel_event`.

A good action contract answers:

- When should this tool be used?
- What do its arguments mean?
- Which fields are required, nullable, or mutually exclusive?
- What identifiers are authoritative?
- Is the operation read-only, reversible, or destructive?
- Can it be retried without duplicating effects?
- What explicit result and error variants can occur?

Schema validation answers only part of this. The JSON `{"date": "2026-02-30"}` may be structurally valid but semantically impossible. The harness must enforce domain rules after parsing.

### 3. Execution and permissions

The model does not receive authority merely because it can name an action. Model output should be treated like untrusted input arriving at a privileged API boundary.

The execution layer should enforce:

- an allowlist of tools for the current user and task;
- least-privilege credentials;
- argument and domain validation;
- approval gates for consequential actions;
- filesystem, network, or process isolation where needed;
- timeouts and resource limits;
- auditable records of attempted and completed mutations.

This creates defense in depth. Natural-language instructions still matter because they help the model choose well, but exact restrictions belong in code. “Never spend more than $100” should be checked against the transaction, not trusted to the model's recollection of the prompt.

Permissions should also be contextual. Searching a private task database, deleting a record, and sending a message are not equally risky. A harness can allow low-risk reads automatically, require confirmation for destructive writes, and prohibit operations that the product does not support.

### 4. State, memory, and persistence

The context window is temporary. A durable agent must distinguish at least three kinds of state:

1. **Authoritative application state:** database records, file contents, bookings, commits, or other facts whose existence is determined outside the model.
2. **Run state:** the current objective, completed steps, pending actions, tool observations, budgets, and errors for this attempt.
3. **Conversational memory:** prior preferences, decisions, and discussion summaries that may help interpret future requests.

Combining all three into a transcript creates fragile systems. Transcripts become long, summaries drift, and the model's earlier statements can be mistaken for real-world state. A robust harness persists each category in an appropriate form and reconstructs a bounded context when needed.

Long-running tasks make this especially visible. Anthropic's experiments used explicit progress artifacts, version control, and separate initialization and continuation behavior so later sessions could resume work instead of guessing what earlier sessions had done. OpenAI similarly describes checked-in plans, repository documentation, tests, and observability as agent-legible sources of truth rather than relying on one enormous instruction file.

The general pattern is **structured handoff**: preserve goals, verified progress, unresolved problems, important decisions, and exact artifact locations; discard most conversational residue.

### 5. Control flow, budgets, and stopping

An agent loop can be deceptively small:

```python
while not done:
    response = model.generate(context, tools)
    observation = run_requested_action(response)
    context.append(observation)
```

The engineering difficulty lies in defining `done` and preventing the loop from becoming unbounded. A production harness needs explicit limits:

- maximum model calls and tool calls;
- token, time, and monetary budgets;
- retry counts by error type;
- no-progress detection;
- duplicate-action detection;
- criteria for asking the user;
- criteria for human escalation;
- task-specific completion checks.

Stopping because the model says “done” is weak evidence. The harness should prefer observable completion: the database contains the expected record, tests pass, a requested file exists, or the user has supplied the missing decision.

Control flow also need not be fully autonomous. Many useful systems are deterministic workflows with one or two model-powered steps. Anthropic's practical guidance distinguishes predefined workflows from agents that dynamically choose their process and recommends adding complexity only when evaluation shows that it helps ([Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)).

### 6. Validation, verification, and recovery

Validation asks whether a proposed action is allowed and well formed. Verification asks whether the intended outcome actually occurred.

These are different checks:

```text
Proposal: create an event for August 12
Validation: the date exists, the user can create events, and required fields are present
Execution: database insert returns event_id = evt_193
Verification: reading evt_193 returns the expected title and date
```

Tool errors should return structured observations rather than vague strings. A retryable timeout, a permanent validation error, an ambiguous match, and a partially completed operation require different recovery behavior.

Recovery should be conservative. Retrying a read is usually safe. Retrying a payment or message send without an idempotency key may duplicate the action. If execution status is unknown, the harness should reconcile against authoritative state before trying again.

For complex work, verification can include compilers, unit tests, browser automation, database assertions, static analyzers, or a second review model. Automated checks are strongest when they measure the actual environment rather than the fluency of the final answer.

### 7. Observability, replay, and evaluation

An agent's final text is insufficient for debugging. The harness should capture a trace containing the configuration and consequential steps of the run:

- model and harness versions;
- context sources and selection decisions;
- tool schemas exposed;
- model calls and structured action proposals;
- policy decisions and approvals;
- tool inputs, outputs, timing, and errors;
- state changes and final outcome;
- cost and latency.

Sensitive values may need redaction, but removing the entire causal history makes systematic improvement impossible.

Evaluation should distinguish the **trace** from the **outcome**. An assistant may say “I created the reminder” even though no row exists; conversely, it may take an unusual but valid path to the correct state. The $\tau$-bench benchmark evaluates conversational tool agents by comparing the final database state with the intended goal state and also measures reliability across repeated trials ([Yao et al., 2024](https://arxiv.org/abs/2406.12045)). ToolSandbox similarly evaluates stateful dependencies and intermediate milestones rather than only a final sentence ([Lu et al., 2024](https://arxiv.org/abs/2408.04682)).

The harness itself must be included in any meaningful evaluation. Changing the available tools, context policy, retry strategy, or stopping criteria can change results without changing the model. AgentBench and SWE-bench helped shift evaluation from isolated language answers toward interaction with environments ([Liu et al., 2023](https://arxiv.org/abs/2308.03688); [Jimenez et al., 2023](https://arxiv.org/abs/2310.06770)).

## A compact harness implementation

The following framework-neutral example shows the core separation of responsibilities for a single-user task assistant. It is intentionally smaller than a general agent framework.

```python
async def handle_turn(user_text: str, conversation_id: str) -> str:
    trace = Trace.start(conversation_id=conversation_id)

    # Deterministic application code loads truth and chooses capabilities.
    intent = await classify_intent(user_text)
    state = event_store.load_scope(intent, user_text)
    tools = tool_registry.allowed(intent=intent, user="local-user")

    messages = context_builder.render(
        user_text=user_text,
        state=state,
        history=conversation_store.recent(conversation_id),
        memories=memory_store.relevant(user_text, limit=4),
        instructions=policy.instructions_for(intent),
    )

    for step in range(MAX_STEPS):
        response = await model.generate(messages=messages, tools=tools.schemas)
        trace.record_model_response(response)

        if response.final_text is not None:
            if policy.may_finish(response, state):
                return response.final_text
            messages.append(observation("Completion is not yet verified."))
            continue

        for proposed_call in response.tool_calls:
            call = tools.parse_and_validate(proposed_call)

            decision = policy.authorize(
                call=call,
                state=state,
                approvals=approval_store.current(conversation_id),
            )

            if decision.requires_user:
                return render_confirmation_request(call, decision)
            if decision.denied:
                messages.append(tool_error(call, decision.reason))
                continue

            result = await tools.execute(
                call,
                timeout=decision.timeout,
                idempotency_key=call.id,
            )
            trace.record_tool_result(call, result)
            messages.append(tool_observation(call, result))
            state = state.apply(result)

    return "I could not complete this safely within the allowed steps."
```

The model still supplies flexible interpretation. It can decide that “move that to August” probably refers to the event mentioned in the previous turn and can propose a suitable update. But the harness resolves candidate records, validates the identifier and date, checks whether ambiguity requires clarification, executes the database update, and returns the actual result.

The example also exposes an important principle: **a harness is policy plus mechanism**.

- Policy decides which tools are allowed, which actions need approval, when the system may stop, and what counts as success.
- Mechanism serializes messages, calls the model, executes tools, stores state, and records traces.

Keeping those concerns distinguishable makes policies easier to inspect and mechanisms easier to reuse.

## Harness design for the task-and-idea chatbot

For the current single-user prototype, harness engineering should produce a small, explicit runtime rather than a highly autonomous multi-agent system.

The essential path is:

1. Receive the user's message.
2. Determine whether it is conversational, a read, or a proposed mutation.
3. Retrieve only the event records and short conversation history needed for the turn.
4. Expose a narrow set of tools such as `create_event`, `find_events`, and `update_event`.
5. Validate tool arguments against the event schema.
6. Ask for clarification when the target or a consequential field is genuinely ambiguous.
7. Execute the mutation in one transaction.
8. return the stored record as the tool observation.
9. Generate a concise confirmation grounded in that observation.
10. Record an evaluation trace.

The model should not be the authoritative event store. It should not infer that an event exists because it remembers mentioning it. Likewise, a natural-language confirmation should not be accepted as evidence that a mutation succeeded.

A useful first tool surface might be:

```json
{
  "create_event": {
    "required": ["title"],
    "optional": ["date", "status", "notes"]
  },
  "find_events": {
    "required": ["query"],
    "optional": ["status", "date_range", "limit"]
  },
  "update_event": {
    "required": ["event_id", "changes"],
    "optional": []
  }
}
```

Deletion can wait or use a separate confirmation workflow. Research, calendars, group planning, delegation, and complex planning loops do not belong in the first harness merely because a framework supports them. The MVP's harness should prove four outcomes: reliable capture, retrieval, updating, and natural conversation around those operations.

This is also why prototyping in a lightweight stack can transfer to PydanticAI, Agno, LangGraph, or another orchestration framework. The durable design is not the framework syntax. It is the set of tool contracts, state boundaries, policies, traces, and evaluation cases. If those are explicit, the outer implementation can change without recreating the product's behavior from memory.

## Architectural choices and tradeoffs

### Deterministic workflow or agent loop?

Use a deterministic workflow when the stages are known and the cost of deviation is high. A fixed sequence such as extract fields → validate → confirm → write is easier to test than an open loop.

Use an agent loop when the required steps depend on discoveries made during the task: research, debugging, navigating an unfamiliar repository, or interacting with a changing environment. The tradeoff is greater flexibility in exchange for less predictable cost and trajectory.

Many systems should combine both. A model may choose among bounded workflows, while each workflow enforces its own exact transitions.

### One model or several roles?

A second model call can critique, verify, summarize, or route. Multiple roles can reduce certain errors, but they also add latency, cost, correlated failure, and more state to coordinate. A “planner” does not automatically improve execution, and a reviewer using the same evidence may repeat the original mistake.

Add a role only when it has a distinct input, authority, or verification method. A test runner that checks the environment contributes independent evidence. A second model merely asked “is this good?” may not.

### Build directly or use a framework?

Frameworks are valuable when they provide capabilities the application would otherwise have to build repeatedly: schema integration, provider adapters, resumable state, tracing, human approvals, or deployment infrastructure.

They can also conceal the exact message sequence, retry behavior, or persistence model. That becomes a problem when debugging requires knowing what the model actually saw and why a tool ran.

A practical criterion is **inspectable leverage**: use an abstraction when it removes routine work while preserving access to context, actions, state transitions, and traces. For a narrow MVP, a direct loop plus typed models may be clearer. A framework becomes more valuable as branching, resumption, integrations, and operational load grow.

### General tools or narrow tools?

A general `execute_sql` or `run_shell` tool is powerful and compact, but grants a large action surface and requires the model to construct low-level operations correctly. Narrow domain tools encode stronger guarantees and are easier to authorize and evaluate.

Broad tools are appropriate inside well-isolated environments, especially for coding and research. User-facing business actions usually benefit from narrower APIs. A hybrid harness may offer broad read capabilities but restrict writes to typed domain operations.

### Natural-language policy or code?

Natural language is well suited to qualitative guidance, exceptions that require interpretation, and procedures that change frequently. Code is suited to exact invariants, security boundaries, resource limits, and transactional behavior.

The best division is not “all rules in prompts” or “all rules in code.” It is:

- explain the desired behavior to the model;
- enforce non-negotiable constraints outside the model;
- make any disagreement visible in the trace.

## Common failure modes

### The thin-wrapper illusion

The system appears to work in demos, so the model is credited with capabilities actually supplied by hidden application assumptions. Production inputs expose missing identity, state, error handling, and policy boundaries.

### Prompt-only enforcement

Permissions, budgets, or destructive-action rules exist only as prose. A single misunderstood instruction can then produce a real-world effect. Critical invariants need deterministic gates.

### Ambiguous tool contracts

Tool names and fields are technically valid but semantically unclear. The model calls the wrong tool, invents identifiers, or cannot distinguish omitted values from intentional nulls.

### Success by assertion

The model says that a task is complete, and the harness accepts the statement without checking the environment. Completion should be tied to observable state or explicit user acceptance.

### Retry amplification

Every error triggers the same retry. Permanent validation errors waste calls, and ambiguous write failures can duplicate effects. Recovery policy should depend on error type and operation semantics.

### Context accumulation

Every observation remains in the prompt forever. Relevant state becomes buried, token costs grow, and stale results compete with current truth. The harness needs compaction and structured handoffs, not unlimited transcript replay.

### Hidden mutable state

The runtime changes data that is neither included in observations nor recorded in traces. Later model calls reason from an incomplete world, and developers cannot reconstruct the failure.

### Framework-shaped architecture

The system inherits planners, memory stores, or multi-agent roles because the chosen framework makes them easy, not because the task requires them. Complexity increases before success criteria exist.

### Harness overfitting

Prompts, retries, and special cases are tuned to a small benchmark until they stop representing real use. Evaluation needs held-out cases, multiple trials, and production feedback. The harness configuration should be versioned alongside its results.

### Stale scaffolding

Documentation, skills, examples, and tool descriptions no longer match the environment. Because agents rely on this material operationally, stale guidance behaves like stale code. It needs ownership, tests where possible, and regular cleanup.

## Evaluating a harness

A useful evaluation suite tests both final outcomes and the path used to reach them.

| Dimension | Example question |
| --- | --- |
| Task success | Did the requested event exist with the correct fields at the end? |
| Action correctness | Were only the necessary tools called with valid arguments? |
| State consistency | Did the transcript, returned result, and database agree? |
| Safety | Were destructive or external actions blocked or confirmed appropriately? |
| Reliability | Does the same case succeed across repeated trials? |
| Efficiency | How many calls, tokens, tools, and seconds were required? |
| Recovery | Did the system handle timeouts, partial results, and stale state safely? |
| User experience | Did it ask only necessary questions and explain the actual outcome clearly? |

Tests should exist at several layers:

1. **Unit tests** for schema validation, authorization, idempotency, and state transitions.
2. **Context tests** that assert which records and tools are included or excluded.
3. **Simulated tool tests** that inject success, timeout, ambiguity, and partial failure.
4. **Trajectory tests** that inspect prohibited calls, unnecessary loops, and stopping behavior.
5. **Outcome tests** that grade the final external state.
6. **Human review** for conversational quality and cases whose success cannot be reduced to one deterministic assertion.

Because generation is stochastic, important cases should run more than once. A system that succeeds once in eight attempts is qualitatively different from one that succeeds reliably, even if both have a nonzero pass rate.

Model and harness changes should also be evaluated separately where possible. Hold the harness fixed while comparing models; hold the model fixed while testing tool descriptions, context policies, or retry strategies. Otherwise an apparent model improvement may actually be a better scaffold, or a stronger model may be hidden by a worse environment.

## Practical design principles

- **Make the model's world explicit.** Context, tools, state, and policy should be inspectable artifacts.
- **Keep truth outside the transcript.** Reload authoritative state after mutations and resumptions.
- **Treat tool calls as proposals.** Validate and authorize them before execution.
- **Design tools for model use.** Clear names, typed arguments, bounded outputs, and explicit errors matter.
- **Prefer observable completion.** Verify the environment rather than trusting a completion statement.
- **Use narrow authority.** Give each run only the capabilities and credentials it needs.
- **Make writes idempotent or reconcilable.** Retries should not silently duplicate effects.
- **Budget every loop.** Calls, tokens, time, and money need limits and stopping policies.
- **Persist structured progress.** Long tasks should survive context resets through artifacts, not conversational hope.
- **Version the whole system.** Model, prompts, tool schemas, policies, and harness code jointly determine behavior.
- **Start with the smallest sufficient harness.** Add planning, delegation, and extra model calls only when measured failures justify them.

## What harness engineering does not solve

A harness cannot make a model understand facts it was never given or perform reasoning beyond its capability. It cannot guarantee correctness in tasks whose success criteria are themselves ambiguous. Verification code can contain bugs, retrieved sources can be wrong, and human approval can become a rubber stamp.

Nor does a harness remove nondeterminism. It contains and observes it. The aim is to turn uncertain model behavior into bounded application behavior: uncertain interpretations can trigger clarification; uncertain actions can be withheld; uncertain outcomes can be checked; repeated failures can stop safely.

Harness engineering therefore does not replace model improvement, prompt design, context engineering, or domain expertise. It connects them to an executable system.

## Recap

A base language model transforms supplied context into generated output. It does not inherently possess persistent memory, tools, permissions, retries, or knowledge of whether an action succeeded. Those capabilities come from the harness.

Harness engineering designs that surrounding execution system. Its major responsibilities are context assembly, tool contracts, permissions, state, control flow, verification, recovery, observability, and evaluation. The harness is where a flexible language interface meets deterministic application boundaries.

The most useful mental division is:

- the **model** interprets and proposes;
- the **harness** controls and verifies;
- the **environment** supplies state and consequences;
- the **agent** is the behavior that emerges from their interaction.

For a practical application, the goal is not maximum autonomy. It is the smallest inspectable system that can complete the intended work reliably, recover safely when it cannot, and make the difference visible.

## Key sources

- Karpas et al. (2022), [*MRKL Systems: A Modular, Neuro-Symbolic Architecture That Combines Large Language Models, External Knowledge Sources and Discrete Reasoning*](https://arxiv.org/abs/2205.00445).
- Yao et al. (2022), [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629).
- Schick et al. (2023), [*Toolformer: Language Models Can Teach Themselves to Use Tools*](https://arxiv.org/abs/2302.04761).
- Liu et al. (2023), [*AgentBench: Evaluating LLMs as Agents*](https://arxiv.org/abs/2308.03688).
- Jimenez et al. (2023), [*SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*](https://arxiv.org/abs/2310.06770).
- Yang et al. (2024), [*SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*](https://arxiv.org/abs/2405.15793).
- Yao et al. (2024), [*$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*](https://arxiv.org/abs/2406.12045).
- Lu et al. (2024), [*ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities*](https://arxiv.org/abs/2408.04682).
- Anthropic (2024), [*Building Effective Agents*](https://www.anthropic.com/engineering/building-effective-agents).
- Anthropic (2025), [*Effective Harnesses for Long-Running Agents*](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents).
- Lopopolo (2026), [*Harness Engineering: Leveraging Codex in an Agent-First World*](https://openai.com/index/harness-engineering/).
- Pan et al. (2026), [*Natural-Language Agent Harnesses*](https://arxiv.org/abs/2603.25723).
