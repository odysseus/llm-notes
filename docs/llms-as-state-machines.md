---
title: "LLMs as State Machines"
type: concept
status: active
updated: 2026-08-07
tags: [state-machines, workflows, agents, orchestration, memory, tools, reliability, LLMs]
---

# LLMs as State Machines: Structuring Stateful Workflows Around Probabilistic Models

**Central idea:** A language model can interpret the current situation and propose what should happen next, but an ordinary model call does not by itself maintain durable application state. In a reliable LLM system, an explicit state machine often surrounds the model: the application stores the current state, determines which transitions are legal, invokes the model for bounded judgments, executes approved actions, and updates state from real observations. The LLM supplies flexible interpretation inside a process whose continuity and authority remain in software.

**Why it matters:** Many failures attributed to “agent reasoning” are actually failures of state management. The system forgets what has already happened, repeats a write after a timeout, treats a plan as a completed action, loses track of an approval, or allows an old tool result to determine a new decision. A state-machine design makes the process inspectable and resumable. It also clarifies which parts should remain probabilistic and which parts should be enforced by code.

## Background topics

- **Finite-state machines:** Models of behavior defined by states, inputs, and allowed transitions.
- **Extended state machines:** State machines whose finite control state is accompanied by structured data such as identifiers, counters, and records.
- **Autoregressive generation:** Producing each token from the preceding token sequence.
- **Conversation context:** The messages, retrieved records, summaries, and instructions supplied to a model call.
- **Dialogue-state tracking:** Maintaining an estimate of a user's goal and relevant slot values across a conversation.
- **Markov decision processes:** Models in which an agent selects actions based on a state and receives subsequent observations or rewards.
- **Tool use:** Allowing a model to request external reads, computations, or mutations and then interpret their results.
- **Harness engineering:** Building the runtime that assembles context, controls tools, persists state, applies policy, and verifies outcomes.
- **Idempotency and transactions:** Techniques for making retries safe and keeping state changes consistent.
- **Event sourcing and checkpoints:** Recording transitions or snapshots so a process can be audited, replayed, and resumed.

These topics supply the vocabulary for separating a language model's transient computation from the persistent process in which it participates.

## Before LLM state machines

State is not a new problem in conversational software. Early task-oriented dialogue systems could not rely on a free-form language model to remember what a user wanted. They commonly separated the application into recognizable components: language understanding extracted an intent and slot values, a dialogue manager maintained a representation of the conversation, a policy chose the next system action, and a language generator rendered that action as a response.

A travel system might store a state such as:

```text
intent: book_flight
origin: Denver
destination: unknown
departure_date: 2026-09-18
next_action: ask_destination
```

The system did not need to reread the entire transcript to know what information was missing. Its dialogue state explicitly represented the facts relevant to the task. Finite-state controllers were easy to inspect, but they could be brittle: an unexpected user request, correction, or topic change might not fit any anticipated branch.

Statistical dialogue research introduced more flexible representations of uncertainty. Rather than assuming that speech recognition and intent classification always produced one correct state, a system could maintain a probability distribution over possible user goals. Williams and Young formalized spoken dialogue as a partially observable Markov decision process, or POMDP, in which the system acts on a **belief state** derived from uncertain observations ([Williams and Young, 2007](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/williams2007csl.pdf)). The Dialog State Tracking Challenges later provided common evaluations for systems that updated these estimates across turns ([Henderson et al., 2014](https://aclanthology.org/W14-4337/)). Neural belief trackers replaced parts of the hand-built language machinery while retaining the idea that a task-oriented conversation needs an explicit, continually updated account of the user's goal ([Mrkšić et al., 2017](https://aclanthology.org/P17-1163/)).

End-to-end neural conversation models weakened this visible separation. A sequence model could receive dialogue history and generate a response directly. The resulting system was more tolerant of linguistic variation, but important state often became implicit in hidden activations or in the text of the transcript. The application could appear stateful while having no canonical representation of what had been confirmed, changed, or completed.

Large language models amplified both sides of this tradeoff. A modern model can infer intents, extract fields, follow corrections, and discuss topics that were never enumerated in a traditional dialogue graph. For a simple chatbot, the application may only append each new message to a conversation and ask the model for the next response. This creates a convincing impression of continuity.

That approach becomes fragile when the conversation performs work. Once the system can search, update records, send messages, edit files, or wait for later input, it must distinguish among several things that prose alone does not reliably separate:

- what the user requested;
- what the model proposed;
- what the application authorized;
- what a tool actually did;
- what remains incomplete;
- what must happen after a pause or failure.

ReAct made the observation–action loop explicit by interleaving model reasoning, external actions, and returned observations ([Yao et al., 2022](https://arxiv.org/abs/2210.03629)). It demonstrated why an LLM agent needs feedback from an environment rather than one static plan. A free-form loop, however, still leaves the process encoded largely in prompts and generated text.

StateFlow later proposed representing LLM task-solving as a state machine. It separated **process grounding**—states and transitions—from **subtask solving** within a state. In its experiments, this structure improved success and reduced inference cost relative to a ReAct baseline on the tested SQL and interactive-environment tasks ([Wu et al., 2024](https://arxiv.org/abs/2403.11322)). The broader lesson is not that every prompt needs a diagram. It is that flexible model behavior becomes easier to control when the surrounding process has explicit progress states.

## The topic in one view

A conventional state machine can be represented as:

$$
s_{t+1} = \delta(s_t, e_t)
$$

where:

- $s_t$ is the current state;
- $e_t$ is an event or input;
- $\delta$ is the transition function; and
- $s_{t+1}$ is the next state.

An output-producing machine may also define:

$$
a_t = \lambda(s_t, e_t)
$$

where $a_t$ is the action or output associated with the state and event. In classical machines, $\delta$ and $\lambda$ are deterministic rules. An LLM system can retain that structure while using a model inside selected decisions:

$$
z_t \sim p_\theta\!\left(\cdot \mid P(s_t, e_t, c_t)\right)
$$

$$
v_t = V(z_t)
$$

$$
o_t = E(v_t)
$$

$$
s_{t+1} = T(s_t, e_t, v_t, o_t)
$$

Here:

- $P$ constructs a state-specific prompt and context;
- the LLM produces a stochastic proposal $z_t$;
- $V$ parses, validates, and authorizes the proposal;
- $E$ executes any approved external action and returns an observation $o_t$; and
- $T$ updates durable state from the old state and verified results.

This decomposition establishes an important authority boundary:

> The model may propose a transition. The application decides whether that transition is legal, performs any real effect, and records the resulting state.

A stateful LLM workflow therefore looks like:

```text
load durable state
    → select the current state handler
    → assemble only the context needed in that state
    → call a model, tool, or deterministic function
    → validate the returned proposal or observation
    → apply one legal transition
    → checkpoint the new state
    → continue, pause, fail safely, or finish
```

Not every state needs an LLM call. A validation state may be ordinary code. An execution state may call a database. A waiting state may do nothing until a user or external service supplies an event. A response state may use the model only to phrase an outcome that has already been established.

## Is an LLM itself a state machine?

The answer depends on the level of abstraction. Saying simply that an “LLM is stateful” or “LLMs are stateless” hides several different meanings of state.

### Within one generation: a stateful computation

Autoregressive generation is sequential. At token step $t$, the model predicts a distribution for the next token from the preceding prefix:

$$
p(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$

The prefix can be treated as the complete logical state of the generation. Transformer inference normally represents much of the reusable computation over that prefix in a key–value cache. As each token is generated, the prefix and cache grow, changing the distribution for the next token.

In this limited sense, one model invocation is a dynamical process with changing internal computational state. It is not normally useful to call it a classical finite-state machine: its representations are high-dimensional, its outputs may be sampled stochastically, and the meaningful logical state can include an enormous number of possible token sequences.

### Between ordinary calls: no inherent durable memory

The base model's weights do not ordinarily change after each conversation turn. They are parameters of the transition function, not a writable memory record for one user's session. Once an inference call ends, its activations and cache are not inherently available to an unrelated later call.

An API may offer a conversation, thread, or session abstraction, and a serving system may retain or reconstruct earlier context. That can make the endpoint convenient and stateful from the application's perspective. Logically, however, some system outside the frozen model is storing identifiers, messages, summaries, or cached representations and making them available to the next invocation.

Transformer-XL is useful historical evidence for this distinction. It added an explicit segment-level recurrence mechanism so information could extend beyond the fixed segments used by a vanilla Transformer language model ([Dai et al., 2019](https://arxiv.org/abs/1901.02860)). Later memory systems such as MemGPT similarly managed information across a limited active context by moving it among memory tiers ([Packer et al., 2023](https://arxiv.org/abs/2310.08560)). Persistent continuity is an architectural feature, not something every Transformer endpoint silently provides.

### In an application: a probabilistic transition component

The most useful engineering interpretation is neither “the LLM has no state” nor “the LLM is the state machine.” It is:

> The LLM is a probabilistic function used inside a larger state-transition system.

It can infer a user intent, summarize evidence, choose among allowed branches, propose a state patch, generate an action, or phrase a response. The surrounding [harness](harness-engineering.md) owns the durable state and the rules that cannot be left to probabilistic generation.

## Six kinds of state in an LLM application

Many design errors come from placing all state into one transcript. A more precise architecture distinguishes at least six layers.

| State layer | Example | Typical owner | Lifetime |
| --- | --- | --- | --- |
| Model computational state | Token prefix, activations, key–value cache | Model runtime | One active generation or retained inference session |
| Conversation state | Messages, summaries, current topic | Conversation service or harness | Across turns |
| Belief state | Inferred intent, possible referenced record, unresolved ambiguity | Dialogue or reasoning layer | Until revised by evidence |
| Workflow state | Current phase, retries, pending approval, next allowed transitions | Orchestrator | Until the run terminates |
| Domain state | Saved event, account balance, order status, document version | Authoritative application database | Product-defined |
| Environment state | Current web page, filesystem, external API, physical world | External system | Changes independently |

These layers interact, but they should not be treated as interchangeable.

### Model computational state

The model's internal inference state exists to predict tokens efficiently. It is opaque, transient, and unsuitable as the application's source of truth. A cached representation cannot serve as a reliable answer to “Was the calendar event actually created?”

### Conversation state

Conversation state preserves the linguistic continuity of an interaction. It includes what the user and assistant said, perhaps compressed into summaries. It is important evidence, but a transcript is a log of claims—not proof that every claim is true.

### Belief state

Belief state represents what the system currently infers. If the user says “move the museum trip to Saturday,” the system may believe that one of two saved events is the intended target. That uncertainty should remain visible rather than being prematurely converted into a confident domain-state mutation.

Belief states need not contain formal probabilities. They may contain candidate identifiers, confidence, assumptions, and a list of missing facts. The POMDP lineage matters because it reminds us that observations of user intent are incomplete and fallible.

### Workflow state

Workflow state answers procedural questions: Are we interpreting, awaiting clarification, requesting approval, executing, validating, or done? It should also record attempt counts, budgets, checkpoints, and any pending work needed after resumption.

### Domain state

Domain state is what the product actually manages. An event exists because the database contains a committed event record, not because the assistant previously said it saved one. Domain state usually outlives a particular workflow and may be changed by other processes.

### Environment state

The environment may change without the agent. A file can be edited by a person, an order can be shipped, or a web page can navigate. Before taking a consequential action, the system may need to reload current state rather than trust an old observation in its context.

## A strict finite-state machine versus an extended state machine

A classical finite-state machine has a finite set of named states. Real LLM applications usually need an **extended state machine**:

```text
control state: AWAITING_APPROVAL

extended data:
    target_record_id: evt_1842
    proposed_changes: {date: 2026-09-18}
    expected_record_version: 7
    approval_requested_at: 2026-08-07T18:42:00Z
    retry_count: 0
```

The control state determines which transitions are allowed. The attached data supplies the values needed to evaluate guards and perform actions. This avoids inventing a separate named state for every record, date, or retry count.

A practical machine can be described by five concepts:

1. **States:** recognizable phases such as `INTERPRETING`, `WAITING`, and `VERIFYING`.
2. **Events:** user messages, tool results, timeouts, approvals, cancellations, and system errors.
3. **Guards:** conditions that must be true before a transition is allowed.
4. **Actions:** model calls, validations, tool executions, notifications, and writes.
5. **Transitions:** the legal movement from one control state to another after an event.

For example:

| Current state | Event | Guard | Action | Next state |
| --- | --- | --- | --- | --- |
| `INTERPRETING` | extraction returned | schema valid | store proposed event | `PERSISTING` |
| `PERSISTING` | tool succeeded | stable event ID returned | checkpoint receipt | `SAVED` |
| `PERSISTING` | tool timed out | outcome unknown | look up by idempotency key | `RECONCILING` |
| `SAVED` | useful fields missing | event already committed | ask optional clarification | `WAITING_FOR_DETAIL` |
| `WAITING_FOR_DETAIL` | user declines | none | preserve skeletal event | `COMPLETE` |
| `WAITING_FOR_DETAIL` | user supplies detail | target ID still valid | propose update | `UPDATING` |

The table communicates behavior more precisely than a single prompt saying “save the idea and ask sensible follow-up questions.”

## Where the LLM belongs in the machine

There are several valid ways to combine language models and state transitions. They provide different balances of flexibility and control.

### 1. The machine selects the model's job

This is the strongest default. Deterministic code selects the state; each state has a bounded model task.

- In `INTERPRETING`, the model extracts intent and fields.
- In `CLARIFYING`, it asks one question about a specific ambiguity.
- In `SUMMARIZING`, it explains a verified result.
- In `RECOVERING`, it classifies an error into a small set of response strategies.

The model does not receive one enormous instruction to manage the entire lifecycle. State-specific prompts are shorter, easier to evaluate, and less likely to mix incompatible responsibilities.

### 2. The model selects among bounded transitions

The controller can present a small set of legal next actions:

```json
{
  "allowed_transitions": [
    "answer_without_action",
    "create_event",
    "search_for_existing_event",
    "ask_which_event"
  ]
}
```

The LLM chooses one and supplies structured arguments. Code rejects any transition not listed for the current state. This preserves natural-language flexibility without letting the model invent arbitrary control flow.

### 3. The model proposes a state patch

Instead of choosing a phase directly, the model may propose semantic updates:

```json
{
  "intent": "update_event",
  "candidate_event_ids": ["evt_1842", "evt_2190"],
  "requested_changes": {"date": "2026-09-18"},
  "ambiguities": ["Two museum events match the request"]
}
```

A reducer validates this patch and decides which workflow transition follows. The distinction matters: the model is good at interpreting text, while code is better at enforcing whether an update is permissible.

### 4. The state machine becomes a domain tool

In some systems, the LLM-facing tool is not a low-level database update but an operation on a business process:

```text
request_order_cancellation(order_id, reason)
```

The operation advances a separate order state machine that contains rules for shipment status, refunds, authorization, and notifications. The LLM translates the user's language into a request; the domain process owns the legal transition. This is especially useful when business logic already exists independently of the chatbot.

### 5. An open agent loop runs inside one bounded state

Some states genuinely require adaptive work. A `RESEARCHING` state may allow repeated search, reading, comparison, and note-taking until defined evidence criteria or budgets are reached. The outer machine still controls entry, permissions, budgets, checkpoints, and exit.

This hybrid design treats autonomy as a local capability, not the structure of the entire application. A flexible loop can investigate an unfamiliar problem without also deciding whether a payment was authorized or whether a record was committed.

### 6. The model generates or edits the workflow

An LLM can propose a new state graph for a novel process. This may help with prototyping or low-risk automation, but generated workflows require the same treatment as generated code: schema checks, unreachable-state analysis, permission review, resource limits, simulation, and human approval before consequential use.

The more authority a transition carries, the less appropriate it is to accept a generated graph without deterministic validation.

## How the LLM workflow changes

### The conversation-only workflow

A simple conversational system often follows this pattern:

```text
append user message to transcript
    → send transcript to model
    → append assistant response
    → wait for next message
```

The transcript implicitly carries the process. This is often sufficient for explanation, brainstorming, rewriting, or other one-turn and low-consequence tasks.

### The state-machine workflow

A stateful operational system adds several steps:

```text
receive event
    → load workflow checkpoint and authoritative domain records
    → identify legal transitions from the current state
    → construct state-specific model context
    → obtain a structured proposal
    → validate schema, policy, versions, and permissions
    → execute an approved effect
    → interpret the real observation
    → commit the next state
    → render a response or pause
```

The most important changes are outside the prompt.

### Context is selected from state

Instead of sending the entire conversation to every model call, the harness can construct a view appropriate to the current state. A clarification prompt may need the ambiguous candidates and the user's latest message. A confirmation prompt may need the committed record and tool receipt. It does not need every abandoned plan and tool schema from earlier in the session.

This makes state machines a form of [context engineering](context-engineering.md). The state identifies what kind of reasoning is occurring; the context policy supplies the minimum information needed for that reasoning.

### Tool results become transition events

A tool response is not merely more prose. It is an event that may or may not satisfy the guard for a transition.

```text
"Event saved successfully" generated by the model
    ≠ evidence of persistence

{status: "committed", event_id: "evt_1842", version: 1}
    = observation that can permit PERSISTING → SAVED
```

This eliminates a common category error in [tool use](tool-use.md): treating a model's statement of success as if it were an external effect.

### Pauses become first-class states

Waiting for a user, approval, scheduled time, webhook, or long-running tool is not a failed model call. It is a legitimate state. The workflow can checkpoint the reason for the pause and the exact event that will resume it.

### Failures become transitions rather than improvised prose

A timeout can lead to `RETRYING`, `RECONCILING`, `WAITING`, or `FAILED`, depending on whether the operation was read-only, idempotent, or potentially completed. The model may explain the failure, but code determines which recovery paths are safe.

## A compact implementation

The following framework-neutral Python sketch shows the division of responsibility. It is intentionally small: production code would add persistence, transactions, version checks, tracing, and richer error handling.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    RECEIVED = "received"
    INTERPRETING = "interpreting"
    PERSISTING = "persisting"
    SAVED = "saved"
    WAITING_FOR_DETAIL = "waiting_for_detail"
    UPDATING = "updating"
    COMPLETE = "complete"
    RECONCILING = "reconciling"
    FAILED = "failed"


@dataclass
class CaptureState:
    run_id: str
    phase: Phase = Phase.RECEIVED
    draft: dict[str, Any] = field(default_factory=dict)
    event_id: str | None = None
    event_version: int | None = None
    missing_optional_fields: list[str] = field(default_factory=list)
    idempotency_key: str | None = None
    attempts: int = 0
    last_error: str | None = None


def interpret_message(model, message: str) -> dict[str, Any]:
    # The model proposes semantic data, not a workflow phase.
    return model.generate_structured(
        task="Extract one task or idea without inventing missing fields.",
        input=message,
        schema={
            "title": "string",
            "date": "optional string",
            "notes": "optional string",
            "missing_optional_fields": "list[string]",
        },
    )


def advance(state: CaptureState, event: dict[str, Any], store) -> CaptureState:
    if state.phase == Phase.RECEIVED and event["type"] == "user_message":
        state.phase = Phase.INTERPRETING
        return state

    if state.phase == Phase.INTERPRETING and event["type"] == "interpreted":
        proposal = event["proposal"]
        state.draft = validate_event_draft(proposal)
        state.missing_optional_fields = proposal["missing_optional_fields"]
        state.idempotency_key = f"capture:{state.run_id}"
        state.phase = Phase.PERSISTING
        return state

    if state.phase == Phase.PERSISTING and event["type"] == "execute":
        receipt = store.create_event(
            state.draft,
            idempotency_key=state.idempotency_key,
        )
        state.event_id = receipt.event_id
        state.event_version = receipt.version
        state.phase = Phase.SAVED
        return state

    if state.phase == Phase.SAVED and event["type"] == "continue":
        state.phase = (
            Phase.WAITING_FOR_DETAIL
            if state.missing_optional_fields
            else Phase.COMPLETE
        )
        return state

    raise InvalidTransition(state.phase, event["type"])
```

Several details carry more architectural weight than the code's size suggests:

- the model returns an interpretation, not permission to mutate the database;
- only the reducer changes the phase;
- the database receipt supplies the event identifier and version;
- an idempotency key makes a repeated create request reconcilable;
- missing optional fields do not block the initial save;
- an impossible state/event pair raises an error instead of being improvised by the model.

The execution loop can call different handlers depending on the committed phase:

```python
while state.phase not in {Phase.COMPLETE, Phase.FAILED}:
    checkpoint.save(state)

    if state.phase == Phase.INTERPRETING:
        proposal = interpret_message(model, latest_user_message)
        state = advance(state, {"type": "interpreted", "proposal": proposal}, store)

    elif state.phase == Phase.PERSISTING:
        state = advance(state, {"type": "execute"}, store)

    elif state.phase == Phase.SAVED:
        state = advance(state, {"type": "continue"}, store)

    elif state.phase == Phase.WAITING_FOR_DETAIL:
        checkpoint.save(state)
        break  # Resume when a new user event arrives.

    else:
        state = recover_or_fail(state)
```

The loop is deterministic at the control level even though `interpret_message` is probabilistic. This is a recurring design goal: contain uncertainty inside operations whose inputs, outputs, and consequences are explicit.

## State, memory, context, and history

These terms are related but answer different questions.

| Concept | Primary question |
| --- | --- |
| State | What is true now, and what transitions are allowed next? |
| Memory | What past information should remain available later? |
| Context | What information does this particular model call receive? |
| History | What events or messages occurred previously? |
| Domain record | What does the authoritative application say exists? |

### A transcript is history, not sufficient state

The transcript may say that the user approved an action, the assistant started it, and the tool returned an error. A model rereading that text may infer the correct status, but the application should not repeatedly reconstruct critical workflow state from prose if it can store `approval_granted: true` and `execution_status: failed` directly.

History is valuable for audit and reinterpretation. Structured state is valuable for control. Mature systems often retain both.

### Memory is selected evidence

Long-term memory systems such as those explored in Generative Agents and MemGPT store, summarize, and retrieve information beyond the active prompt ([Park et al., 2023](https://arxiv.org/abs/2304.03442); [Packer et al., 2023](https://arxiv.org/abs/2310.08560)). Memory can tell a system that the user prefers afternoon appointments. It does not by itself establish that the current workflow is authorized to book one.

### Context is a view over state and memory

The prompt should usually be derived from durable state, selected history, retrieved memory, current environment observations, and the policy for the active state. It is a temporary working view—not the canonical container for all of those sources.

### Checkpoints are not always the source of truth

A workflow checkpoint may say that it is about to update event version 7. Before executing, the database may reveal version 8 because another request changed the record. The workflow must reconcile with the authoritative domain state. Resumption means continuing safely from stored progress, not assuming the world froze while the process slept.

## Persistence strategies

### Snapshot state

Store the latest complete workflow object after each transition. This is simple and efficient for most applications.

```json
{
  "run_id": "run_93a1",
  "phase": "waiting_for_detail",
  "event_id": "evt_1842",
  "event_version": 1,
  "missing_optional_fields": ["date"],
  "updated_at": "2026-08-07T18:44:12Z"
}
```

Snapshots make resumption easy but do not explain every intermediate change unless a separate trace is retained.

### Event log

Record immutable events such as `MessageReceived`, `DraftExtracted`, `CreateRequested`, `EventCommitted`, and `ClarificationOffered`. Current state can be rebuilt by reducing the event sequence.

This provides strong auditability and replay, but it increases schema-versioning and operational complexity. Replaying model calls is also not equivalent to replaying deterministic code: model versions, prompts, tools, and sampling can change. A reliable event log should preserve the accepted model output and external observations, not merely instructions to regenerate them.

### Hybrid checkpoint plus trace

Many LLM applications benefit from a current snapshot for fast resumption and an append-only trace for debugging and evaluation. The trace can include:

- previous and next state;
- triggering event;
- prompt and model version;
- structured model proposal;
- validation outcome;
- tool request and observation;
- latency, tokens, and cost;
- user approvals and policy decisions.

Sensitive data should be minimized or redacted according to the application's retention policy. Observability is not permission to log every private input indefinitely.

## What state machines provide

### Explicit progress

The system can distinguish “planned,” “awaiting approval,” “executing,” “committed,” and “verified.” This prevents a fluent response from collapsing several materially different stages into one vague sense of completion.

### Controlled flexibility

The model can handle varied language inside a bounded set of transitions. New phrasings do not require hand-authoring every utterance, while critical business rules do not depend on the model remembering a paragraph of policy.

### Resumption and human participation

A process can wait for a user, survive a worker restart, or continue after an external callback. Human approval becomes a real state transition with recorded scope instead of a sentence buried in conversation history.

### Safer retries

The state can record whether an operation is read-only, definitely failed, definitely succeeded, or has an unknown outcome. Idempotency keys and reconciliation states prevent a network timeout from automatically becoming a duplicate write.

### Better context control

Each state can use its own prompt, tools, memory policy, token budget, and output schema. Irrelevant history and tools do not need to accumulate through the entire run.

### Inspectable permissions

Authority can vary by state. A research state may receive read-only search tools. An execution state may receive one narrow write operation only after approval. A response state may have no tools at all.

### Targeted evaluation

Instead of asking only whether the final response sounds good, tests can measure extraction accuracy in one state, transition selection in another, idempotent execution in a third, and end-to-end outcome across the whole machine.

## Alternatives and neighboring approaches

A state machine is one control abstraction among several.

| Approach | Best fit | Main benefit | Main limitation |
| --- | --- | --- | --- |
| One model call | Independent, low-consequence transformation | Minimal cost and complexity | No durable multi-step control |
| Linear pipeline | Known sequence with little branching | Easy to implement and test | Awkward recovery and conditional behavior |
| Router plus handlers | One early classification selects a bounded workflow | Clear specialization | Router mistakes affect the entire path |
| Directed acyclic graph | Parallel stages without cycles | Visible dependencies and parallelism | Poor fit for retries, dialogue, and recurring loops |
| State machine | Branching, cyclic, resumable processes | Explicit legal transitions and recovery | State design and migration overhead |
| Planner–executor | Novel tasks whose steps are not known in advance | Adaptive decomposition | Plans become stale; execution still needs state |
| Free-form agent loop | Open-ended exploration in unfamiliar environments | Maximum local flexibility | Unpredictable paths, cost, and stopping |
| Behavior tree | Hierarchical, reactive control such as games or robotics | Composable fallback and retry behavior | Less natural for data-rich business processes |
| Multi-agent system | Work requiring real parallelism or isolation | Separate contexts, tools, or authorities | Coordination and shared-state complexity |

These approaches can be nested. A state may execute a DAG of read-only searches. A planner may propose tasks that the outer machine validates. A [multi-agent system](multi-agent-systems.md) may give each agent its own state machine while a coordinator manages shared transitions.

The right question is not whether state machines are more advanced. It is whether the task has recognizable phases, legal and illegal transitions, pauses, retries, or consequences that benefit from explicit control.

## When to use a state-machine design

A state machine becomes attractive when several of the following are true:

- the task spans multiple model or tool calls;
- later steps depend on verified results from earlier steps;
- the process can branch, retry, pause, cancel, or resume;
- tools can change durable or external state;
- approval is required before a consequential operation;
- the application must distinguish proposed, attempted, completed, and verified work;
- different phases need different tools, permissions, prompts, or context;
- a timeout may leave the outcome uncertain;
- multiple requests may modify the same records;
- auditability and reproducible recovery matter;
- recurring failures can be described as illegal or mistaken transitions.

It may be unnecessary when:

- the request is one self-contained generation or classification;
- no persistent state or side effect exists;
- a short linear function already describes the entire workflow;
- the work is deliberately exploratory and no stable phase structure is known;
- the cost of designing and maintaining states exceeds the risk of failure.

A useful progression is:

```text
one model call
    → linear pipeline
    → router or small state machine
    → state machine with bounded adaptive loops
    → more elaborate planning or multi-agent coordination only if needed
```

The structure should grow in response to observed failure modes, not because an orchestration framework makes more nodes available.

## Applying the pattern to the task-and-idea chatbot

For the single-user prototype in this research project, a small explicit state machine would be useful, but it should reflect the product's actual policy: **save an idea immediately, then offer clarification rather than blocking persistence until every field is known.**

A capture workflow could use:

```text
RECEIVED
    → INTERPRETING
    → PERSISTING_SKELETON
    → SAVED
    → OFFERING_CLARIFICATION
    → WAITING_FOR_DETAIL or COMPLETE
    → UPDATING
    → COMPLETE
```

This sequence provides several practical guarantees:

1. A vague idea such as “maybe visit the aviation museum” is not lost while the system asks questions.
2. Missing date, duration, cost, or location details remain explicit unknowns rather than invented values.
3. The database receipt establishes the event's stable identity before later clarification attempts to update it.
4. The user can decline clarification, leaving a valid skeletal event.
5. A later conversation can resume enrichment by loading the event record rather than relying on the earlier transcript.

Retrieval and updating need their own ambiguity states. If “move the museum trip to Saturday” matches two records, the correct transition is not directly to `UPDATING`. It is:

```text
SEARCHING
    → AMBIGUOUS_MATCH
    → WAITING_FOR_SELECTION
    → RELOADING_SELECTED_EVENT
    → UPDATING
    → VERIFYING
    → COMPLETE
```

The distinction between `UPDATING` and `VERIFYING` is important. The tool call may return an error, a version conflict, or an ambiguous timeout. A final confirmation should be grounded in the reloaded record or a definite transactional receipt.

The MVP does not need a huge graph. Capture, retrieval, and update can each be a small bounded workflow sharing the same event schema and database. Advanced planning, group interaction, autonomous research, and calendar integrations should introduce new states only when those features themselves enter scope.

## Common failure modes

### Treating the transcript as the database

The system assumes that because a prior assistant message said an event was saved, it exists. A later update operates on a hallucinated record. Authoritative domain state must be loaded from the actual store.

### Letting the model name arbitrary next states

The prompt asks the LLM to return `next_state`, and the controller accepts any string or uses broad fallback behavior. The model can bypass approval or execution states accidentally. Present an allowlist and enforce it in code.

### Hidden state outside the checkpoint

Important information lives only in process memory, a temporary prompt, or one worker's local variables. After restart, the phase is restored but the identifiers, pending action, or tool receipt are missing. A checkpoint must contain everything required for safe resumption or point to durable artifacts that do.

### State explosion

Every combination of data becomes a named control state: `WAITING_FOR_DATE_WITH_LOCATION_WITHOUT_COST`, and so on. Use a small finite control state plus structured extended data.

### One giant `AGENT_RUNNING` state

The application technically has a state machine, but all meaningful behavior occurs inside one unbounded model loop. The graph adds ceremony without control. Split states where permissions, context, verification, waiting, or recovery materially change.

### Overconstraining language behavior

The controller enumerates every conversational path and destroys the main benefit of an LLM: flexible interpretation and response. Use code to constrain effects and lifecycle; leave linguistic understanding and explanation to the model where appropriate.

### Confusing belief with truth

An extracted date or candidate record is written into domain state before ambiguity is resolved. Mark proposed values, sources, and confidence separately until a transition or user confirmation establishes them.

### Retrying unknown writes blindly

A network timeout is classified as failure, so the machine repeats a create, payment, or send action. Unknown outcomes need reconciliation through idempotency keys, lookup operations, or manual review.

### Stale resumption

A paused workflow resumes against an old record version or changed external environment. Reload current state and re-evaluate guards before executing.

### Non-atomic checkpointing

The external write succeeds, but the workflow crashes before recording the success. On restart it repeats the action. Transactional boundaries, idempotency, and outbox or reconciliation patterns are required when workflow and domain state cannot be committed atomically.

### Unversioned state schemas

Deployed code changes the meaning of a field or removes a state while old runs remain paused. Durable workflows need migrations and compatibility rules just like databases.

### Believing explicit control guarantees correctness

The machine reliably advances through the wrong interpretation. A valid trajectory can still produce a bad outcome if model extraction, tool behavior, policy, or state design is wrong. Structure makes errors visible and containable; it does not eliminate them.

## Evaluating stateful LLM systems

Evaluation should examine transitions and final environment state, not only the final response.

| Dimension | Example measure |
| --- | --- |
| Transition validity | Percentage of attempted transitions legal from the current state |
| State estimation | Accuracy of intent, slots, candidates, and ambiguity representation |
| Guard enforcement | Rate at which missing approval, stale versions, or invalid inputs are blocked |
| Action correctness | Correct tool, arguments, order, and idempotency behavior |
| Outcome accuracy | Whether the authoritative environment reaches the target state |
| Completion honesty | Whether success is claimed only after sufficient evidence |
| Recovery | Safe handling of timeouts, partial failure, and unknown outcomes |
| Resumability | Correct continuation after checkpoints, restarts, and long pauses |
| Efficiency | Model calls, tokens, tool calls, latency, and cost per successful run |
| Reliability | Success across repeated stochastic trials, not only the best trajectory |
| Coverage | States, transitions, guards, and error paths exercised by tests |
| User experience | Necessary questions, understandable progress, and graceful cancellation |

Testing should occur at several layers:

1. **Reducer tests:** Given a state and event, assert the exact next state or rejection.
2. **Guard tests:** Try missing approvals, wrong record versions, malformed proposals, and disallowed transitions.
3. **Model-node tests:** Evaluate each bounded interpretation or generation task independently.
4. **Tool simulation:** Inject success, definite failure, timeout-before-write, timeout-after-write, and conflicting updates.
5. **Checkpoint tests:** Stop and resume from every durable state.
6. **Trajectory tests:** Detect unnecessary loops, repeated actions, skipped verification, and budget violations.
7. **Outcome tests:** Compare the final database or environment with the intended result.

Stateful benchmarks reinforce this emphasis. ToolSandbox includes state dependencies among tools and evaluates intermediate as well as final milestones in conversational trajectories ([Lu et al., 2024](https://arxiv.org/abs/2408.04682)). $\tau$-bench evaluates tool-using agents against the final database goal state and reports reliability across repeated trials, showing why a plausible transcript is an insufficient measure of operational success ([Yao et al., 2024](https://arxiv.org/abs/2406.12045)).

## Practical design principles

- **Keep durable state outside the model.** Treat the model's output as a proposal or interpretation.
- **Use a small control graph plus structured data.** Avoid both state explosion and one opaque agent state.
- **Define events as carefully as states.** User messages, approvals, tool receipts, timeouts, and cancellations have different meanings.
- **Enforce transition allowlists.** Prompts can explain policy; code must reject illegal movement.
- **Advance from observations, not assertions.** A tool receipt or reloaded record is stronger than generated success text.
- **Separate belief state from domain state.** Preserve ambiguity until evidence resolves it.
- **Make waits and unknown outcomes explicit.** Pausing and reconciling are legitimate phases.
- **Checkpoint before losing required context.** Resumption should not depend on hidden worker memory.
- **Reload the world before consequential writes.** Stored workflow state may be older than authoritative domain state.
- **Design retries together with idempotency.** Recovery logic is part of the state machine, not an infrastructure afterthought.
- **Give each state the minimum context and authority it needs.** State boundaries are also security and context boundaries.
- **Version prompts, tools, models, and state schemas.** All influence transition behavior.
- **Evaluate individual nodes and whole trajectories.** Local accuracy and end-state correctness are both necessary.
- **Prefer the smallest useful machine.** Add structure where it buys control, recovery, or clarity.

## What state machines do not solve

A state machine does not make an ambiguous user request unambiguous. It gives the system a place to represent ambiguity and a safe transition for resolving it. It does not make an LLM deterministic. It limits where stochastic decisions occur and what consequences they may have.

It also does not supply memory, tools, or knowledge automatically. Those are separate components whose results can become state-machine events. A badly designed tool remains dangerous, a stale database remains stale, and an incorrect model interpretation can still send the process down a legal but wrong branch.

Nor is a state graph a substitute for product understanding. If designers cannot say what “complete” means, adding `COMPLETE` to an enum does not solve the problem. Terminal states need evidence-backed criteria.

Finally, explicit state introduces ordinary software obligations: persistence, concurrency control, migrations, monitoring, and cleanup of abandoned runs. The approach trades some prompt-level simplicity for operational clarity. That trade is worthwhile when the process has real duration, branching, authority, or consequences—not for every act of text generation.

## Recap

An LLM participates in several forms of state. Within one generation, its token prefix and cached activations evolve step by step. Across ordinary calls, however, the frozen model does not inherently own durable conversation or application memory. Persistent continuity comes from systems that store messages, summaries, records, checkpoints, and environmental observations and then supply relevant views to future calls.

The most useful architectural pattern treats the LLM as a probabilistic component inside an explicit transition system:

- the **model** interprets language and proposes semantic decisions;
- the **state machine** defines legal progress and recovery;
- the **harness** validates, authorizes, persists, and assembles context;
- the **tools and environment** provide real observations and effects;
- the **domain store** determines what actually exists.

State machines are especially valuable for multi-step work involving tools, approvals, waiting, retries, resumptions, or external writes. They make the distinction among planned, attempted, completed, and verified work visible. They also permit adaptive model behavior within bounded states rather than forcing a choice between a rigid old-style dialogue tree and an unconstrained agent loop.

For the task-and-idea chatbot, the right machine is small and product-specific: interpret the request, save a skeletal event immediately, record the real event identifier, offer optional clarification, and update only against authoritative state. The purpose is not to turn conversation into bureaucracy. It is to give flexible language behavior a reliable memory of where the work truly stands.

## Key sources

- Mealy (1955), [*A Method for Synthesizing Sequential Circuits*](https://onlinelibrary.wiley.com/doi/10.1002/j.1538-7305.1955.tb03788.x).
- Williams and Young (2007), [*Partially Observable Markov Decision Processes for Spoken Dialog Systems*](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/williams2007csl.pdf).
- Henderson, Thomson, and Williams (2014), [*The Second Dialog State Tracking Challenge*](https://aclanthology.org/W14-4337/).
- Mrkšić et al. (2017), [*Neural Belief Tracker: Data-Driven Dialogue State Tracking*](https://aclanthology.org/P17-1163/).
- Dai et al. (2019), [*Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*](https://arxiv.org/abs/1901.02860).
- Yao et al. (2022), [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629).
- Park et al. (2023), [*Generative Agents: Interactive Simulacra of Human Behavior*](https://arxiv.org/abs/2304.03442).
- Packer et al. (2023), [*MemGPT: Towards LLMs as Operating Systems*](https://arxiv.org/abs/2310.08560).
- Wu et al. (2024), [*StateFlow: Enhancing LLM Task-Solving through State-Driven Workflows*](https://arxiv.org/abs/2403.11322).
- Yao et al. (2024), [*$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*](https://arxiv.org/abs/2406.12045).
- Lu et al. (2024), [*ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities*](https://arxiv.org/abs/2408.04682).
