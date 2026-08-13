---
title: "Multi-Turn Dialogue and Conversational Memory"
type: concept
status: active
updated: 2026-08-13
tags: [dialogue, conversation, memory, context, state, coreference, chatbots]
---

# Multi-Turn Dialogue and Conversational Memory

**Central idea:** Most LLM chatbots do not possess one continuous, durable stream of consciousness. On each turn, an application reconstructs a temporary working context from the current message, selected conversation history, structured state, retrieved memories, instructions, and tool results. The model then interprets the new message relative to that context.

**Why it matters:** A chatbot can sound as if it remembers everything while actually depending on a lossy transcript, a summary, or a retrieval system. Understanding the boundary between model context and application state is essential when a user says “change that,” “use the earlier version,” or “what did I tell you last week?”—especially if the answer will cause a real action.

## Background topics

- **Autoregressive generation:** How an LLM predicts each next token from the tokens already supplied or generated.
- **Transformer attention:** How token representations incorporate information from other positions in the active input.
- **Context windows:** The bounded amount of input and output a model can process in one request.
- **[In-context learning](in-context-learning.md):** How instructions and examples in the current context can temporarily shape model behavior.
- **[Context engineering](context-engineering.md):** How an application selects, represents, orders, and budgets information for a model call.
- **[Harness engineering](harness-engineering.md):** How application code surrounds a model with persistence, tools, validation, control flow, and observability.
- **Coreference and ellipsis:** Linguistic mechanisms that connect expressions such as “it,” “the second one,” and omitted phrases to earlier discourse.
- **Dialogue state:** A representation of the user's current goal, known facts, unresolved questions, and possible referents.

These topics explain why multi-turn dialogue is both a language-model problem and a software-state problem.

## The illusion of one continuous conversation

From the user's perspective, a chat has an obvious shape:

```text
User: Help me plan a weekend in Montreal.
Assistant: What kind of trip do you want?
User: Mostly food and architecture. No rental car.
Assistant: ...
User: Make the second day less ambitious.
```

The last request is meaningful only in relation to earlier turns. “The second day” refers to an itinerary the assistant proposed, “less ambitious” contrasts with the current schedule, and the car constraint still applies even though the user did not repeat it.

It is natural to imagine that a chatbot remains mentally present between messages. The ordinary implementation is less mysterious. A service stores some representation of the interaction. When a new message arrives, it constructs another model request containing enough prior material for the model to continue.

A minimal implementation is close to:

```python
history = conversation_store.load(conversation_id)
history.append({"role": "user", "content": new_message})

response = model.generate(messages=history)

history.append({"role": "assistant", "content": response.text})
conversation_store.save(conversation_id, history)
```

The model receives a sequence that resembles a conversation, but the persistence belongs to `conversation_store`. If the next request contains no history, the model normally cannot recover the omitted exchange merely because it produced an earlier response.

This yields the first important rule:

> A model call uses conversational context; the surrounding application creates conversational continuity.

Some APIs expose a conversation, thread, session, or previous-response identifier. Such handles can make the persistence mechanism less visible to application code, but they do not remove the underlying problem. The service still has to associate the new input with earlier state and decide what the model can use.

## What the model actually receives

Chat interfaces usually represent a request as ordered messages with roles. A simplified turn might be serialized as:

```json
[
  {
    "role": "system",
    "content": "You are a task-and-idea assistant. Do not invent dates."
  },
  {
    "role": "user",
    "content": "Remind me to service the Orient."
  },
  {
    "role": "assistant",
    "content": "I saved a task to service the Orient."
  },
  {
    "role": "user",
    "content": "Move that to August."
  }
]
```

The runtime converts these messages into the model's input representation, including role and boundary markers. During generation, self-attention lets later positions use information represented at earlier positions. This makes it possible for the model to connect “that” with the task about servicing the Orient and “August” with a scheduling field.

For fixed model parameters $\theta$, a reply can be described as a conditional distribution:

$$
p_\theta(y_t \mid I, H, x, y_{<t})
$$

where:

- $I$ is the active instruction set;
- $H$ is the supplied conversational history and state;
- $x$ is the current user message;
- $y_{<t}$ is the reply generated so far.

The important word is **supplied**. The model is conditioned on the history included in this request, not automatically on every interaction the user has ever had.

During one generation, runtimes commonly cache internal attention computations so that each new token does not require recomputing the whole prefix. This key–value cache is computational state, not durable conversational memory. It accelerates the active inference process. A provider may retain or reuse related computation, but application designers should not treat opaque inference caches as a canonical record of the conversation.

Model architectures can also extend dependency across segments. Transformer-XL, for example, introduced segment-level recurrence to carry representations beyond a fixed text segment ([Dai et al., 2019](https://arxiv.org/abs/1901.02860)). That is a model architecture for longer-range sequence processing. It is conceptually different from a product remembering a user's decision for weeks, enforcing a pending approval, or checking the current version of a database row.

## History, context, state, and memory are different things

The vocabulary around chat systems is loose. Four terms are especially easy to collapse.

### Conversation history

The history is a chronological record of conversational events: user messages, assistant responses, tool calls, tool results, edits, and perhaps interface events. It answers:

> What happened in the interaction?

History can be complete even when some claims in it are wrong. An assistant message saying “I booked the table” remains part of history if the booking tool later failed.

### Active context

The active context is the bounded information supplied to the model for one inference. It may contain recent history, summaries of older history, retrieved records, instructions, and tool definitions. It answers:

> What can the model use for this turn?

The context is a temporary view constructed from other sources. It is rarely identical to the complete stored conversation.

### Conversational state

Conversational state represents the current interactional situation: the active topic, what the user is trying to accomplish, which question is pending, which objects are salient, and what remains ambiguous. It answers:

> Where are we in the dialogue?

This state may be implicit in recent messages, explicitly stored in fields, or divided between both.

### Long-term memory

Long-term memory is selected information intended to outlive the immediate turn or session: stable preferences, prior decisions, important episodes, names, or summaries. It answers:

> What from earlier interactions may be useful again?

Long-term memory is not necessarily a raw transcript. It usually requires a policy for extraction, storage, retrieval, correction, expiry, and user control.

These categories overlap, but they are not interchangeable. The most robust systems keep the distinctions visible.

## Six layers of state

Multi-turn applications often need more than conversation text. A useful architecture distinguishes at least six layers:

| Layer | Example | Typical owner | Reliability role |
| --- | --- | --- | --- |
| Model computational state | Token prefix, activations, attention cache | Model runtime | Efficient generation, not durable truth |
| Conversation history | User and assistant turns, tool observations | Conversation service | Audit trail and linguistic evidence |
| Dialogue or belief state | Active goal, candidate referents, unresolved slot | Dialogue layer or harness | Current interpretation, possibly uncertain |
| Workflow state | Pending approval, retry count, completed steps | Orchestrator | Controls what may happen next |
| Domain state | Task row, booking, file version, account record | Authoritative database or external service | Source of truth for real application facts |
| Long-term memory | Preference for afternoons, prior project decision | Memory service | Retrieved background, subject to provenance and correction |

The transcript may help reconstruct any of these, but reconstruction is not always safe. If a task database says event `evt_193` is scheduled for September while an old assistant message says August, the database should normally win.

The distinction is especially important for tool-using systems. A conversational statement can propose an action, claim an action succeeded, or report a tool result. Only the actual environment establishes the final outcome.

This is one reason [LLM applications are better understood as state-transition systems](llms-as-state-machines.md) than as isolated text generators.

## The historical lineage: dialogue state before LLM chat

Multi-turn state is not a new problem. Earlier task-oriented dialogue systems often separated four components:

1. language understanding extracted an intent and entities;
2. a dialogue-state tracker updated what the system believed about the user's goal;
3. a dialogue policy chose the next action;
4. a language generator produced the response.

A restaurant-booking state might look like:

```json
{
  "intent": "find_restaurant",
  "constraints": {
    "cuisine": "Japanese",
    "area": "downtown",
    "price": null
  },
  "requested_fields": ["address"],
  "pending_question": "price"
}
```

The system did not need to reread every word to know that price remained unknown. Dialogue-state tracking research emphasized updating a user's estimated goal across turns, including uncertainty introduced by speech recognition or ambiguous language. The Dialog State Tracking Challenges provided shared evaluations for this problem ([Henderson et al., 2014](https://aclanthology.org/W14-4337/)). Neural belief trackers later learned more of the language mapping while preserving the explicit notion of a continually updated goal state ([Mrkšić et al., 2017](https://aclanthology.org/P17-1163/)).

End-to-end neural dialogue systems made state less visible. A model could consume the history and generate the next reply without exposing an intermediate belief state. LLMs greatly improve this approach because they can interpret open-ended language, corrections, topic changes, and novel tasks. They can often infer the necessary state directly from prose.

The older architecture nevertheless preserves a valuable lesson:

> If the application must make an exact decision across turns, represent the decision-relevant state explicitly enough to inspect and validate it.

LLMs reduce the amount of hand-authored dialogue logic needed. They do not eliminate the value of explicit state.

## How references to earlier turns are interpreted

Users rarely repeat complete instructions. Natural dialogue depends on shared context.

### Coreference

Coreference occurs when two expressions refer to the same entity:

```text
I saved the furnace-filter task.
Move it to Saturday.
```

Here, “it” and “the furnace-filter task” share a referent. Coreference resolution has long been studied as the problem of linking a mention to a possible antecedent; neural approaches can score candidate spans and antecedents directly ([Lee et al., 2017](https://aclanthology.org/D17-1018/)).

Dialogue makes the problem broader than pronouns. A user might say:

- “that one”;
- “the earlier suggestion”;
- “both of those”;
- “the cheaper hotel”;
- “what you said before the table”;
- “the task we discussed last Tuesday.”

Some references depend on recency. Others depend on semantic type, contrast, time, visual selection, or a durable object in the application.

### Ellipsis

Ellipsis omits material that the listener is expected to recover:

```text
User: Move the furnace-filter task to Saturday.
User: And the grill cover to Sunday.
```

The second message means approximately “move the grill-cover task to Sunday.” The operation is inherited from the prior discourse even though it is not repeated.

Other common examples include:

```text
Assistant: Would you like the short or detailed explanation?
User: Short.

Assistant: Should I update both records?
User: Just the first.
```

### Discourse focus and salience

Not every earlier noun is an equally likely referent. Recent mentions, the subject of the previous turn, objects just created by a tool, and items explicitly selected in the interface tend to be more salient.

A system can therefore maintain an **entity focus stack** or a simpler list of active referents:

```json
{
  "active_entities": [
    {
      "entity_id": "evt_193",
      "type": "event",
      "label": "Service the Orient",
      "introduced_by": "tool_result_81",
      "last_mentioned_turn": 42
    }
  ]
}
```

When the user says “move that to August,” the model or resolver can consider this structured candidate alongside the raw wording.

### Corrections and reversals

References can point to prior values rather than prior objects:

```text
User: Put it on Saturday.
Assistant: Done.
User: Actually, change it back.
```

“Back” requires the previous value, not merely the current one. A database audit log or reversible command record is safer than hoping a summary preserved the exact earlier date.

### Quoted and metalinguistic references

Sometimes the user refers to language itself:

```text
What did you mean by “bounded context”?
Use the wording from your first answer, not the revised one.
```

These requests require turn identity and version history. A search over semantic memory may retrieve the right topic but the wrong revision. Exact references benefit from stable turn IDs, branch IDs, and source pointers.

## Reference resolution is a ranking problem

For a new referring expression $r$ and a set of candidates $C$, a system can think in terms of a score:

$$
\operatorname{score}(c \mid r) =
w_1 \cdot \text{semantic-match}
+ w_2 \cdot \text{recency}
+ w_3 \cdot \text{type-match}
+ w_4 \cdot \text{discourse-focus}
+ w_5 \cdot \text{interface-selection}
+ w_6 \cdot \text{task-compatibility}
$$

An LLM may estimate these relationships implicitly. Application code can also generate and filter candidates deterministically. The strongest design is often hybrid:

1. collect plausible candidates from recent turns, selected UI objects, and the database;
2. eliminate incompatible types or inaccessible records;
3. ask the model to interpret the expression against the bounded candidate set;
4. validate the returned stable identifier;
5. clarify if the remaining ambiguity matters.

For example, “delete that” should not allow the model to invent an identifier. The application can present three real candidate records and require a typed selection:

```json
{
  "operation": "resolve_reference",
  "expression": "that",
  "candidates": [
    {"id": "evt_193", "label": "Service the Orient"},
    {"id": "evt_201", "label": "Order furnace filters"}
  ]
}
```

A possible structured model result is:

```json
{
  "resolved_id": "evt_193",
  "confidence": "low",
  "reason": "Both records were mentioned in the previous turn."
}
```

The correct next step is clarification, not deletion. Fluency should not hide uncertainty.

## The simplest memory strategy: replay the transcript

The simplest multi-turn chatbot stores every message and resends the whole sequence on each turn.

This works surprisingly well when conversations are short. It preserves exact wording, turn order, conversational tone, and local references without additional retrieval logic. It is also easy to inspect.

However, full replay has several limits:

- **Context capacity:** Eventually the transcript no longer fits alongside instructions, tool schemas, retrieved evidence, and output space.
- **Cost and latency:** Reprocessing an ever-growing prefix makes later turns more expensive.
- **Attention dilution:** Relevant details compete with large amounts of harmless but irrelevant text.
- **Contradictions:** Old plans and corrected values remain in the history.
- **Security:** Every retained turn is another opportunity for stale or malicious content to influence behavior.
- **Privacy:** Sending unnecessary past material increases exposure and complicates retention policies.

Long context is capacity, not perfect recall. *Lost in the Middle* showed that model performance can vary with the position of relevant information and may degrade when evidence is buried within a long input ([Liu et al., 2024](https://aclanthology.org/2024.tacl-1.9/)). The exact behavior differs by model and task, but the engineering consequence is durable: do not assume that fitting history into the window means the model will use every part reliably.

## Common history-management strategies

Real systems combine several techniques.

### Sliding windows

A sliding window keeps the most recent $n$ turns or tokens and drops older material.

```python
recent = transcript.last_tokens(max_tokens=8_000)
```

This is simple and often effective because recent turns carry strong discourse salience. It fails when the user refers to an older decision:

```text
Use the dietary restriction I mentioned at the start.
```

Recency is a good default signal, not a complete memory policy.

### Summarization

Older turns can be compressed into a running summary:

```text
Conversation summary:
- User is planning a three-day Montreal trip.
- Priorities: food and architecture.
- Constraint: no rental car.
- Current itinerary draft: v3.
- Unresolved: choose between two hotels.
```

Summaries preserve high-level continuity at lower token cost. They are lossy. They may omit the exact wording, remove uncertainty, merge separate events, or preserve a model's earlier mistake as if it were fact.

Repeatedly summarizing summaries creates **summary drift**. A safer design retains the raw log, attaches source turn IDs, and periodically rebuilds important summaries from original or authoritative records.

Summaries should also separate kinds of information:

```json
{
  "confirmed_user_constraints": ["no rental car"],
  "assistant_proposals": ["draft itinerary v3"],
  "open_questions": ["hotel choice"],
  "superseded_values": ["draft itinerary v2"],
  "source_turn_ids": [11, 14, 18, 21]
}
```

This avoids turning every sentence into one undifferentiated paragraph.

### Retrieval over older history

The system can index past turns or memory records and retrieve those relevant to the current message. Retrieval may combine embeddings, keyword search, timestamps, entity IDs, conversation IDs, and memory types.

```python
memories = memory_store.search(
    user_id=user_id,
    query=new_message,
    filters={
        "kind": ["preference", "decision", "episode"],
        "status": "active",
    },
    top_k=6,
)
```

This follows the broader retrieval-augmented pattern of combining a model with external non-parametric memory ([Lewis et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)). In a conversation system, the corpus may be the user's own prior interactions rather than a knowledge base.

Retrieval solves capacity, not truth. Similarity search can return an obsolete plan, another person's data, an assistant speculation, or an episode that shares vocabulary but not intent. Memory records need metadata, provenance, access control, and correction policies.

### Structured dialogue state

Instead of asking the model to infer everything from prose, the application can update explicit fields after each turn:

```json
{
  "active_goal": "reschedule_event",
  "target_candidates": ["evt_193"],
  "resolved_target": "evt_193",
  "requested_changes": {"target_month": "2026-08"},
  "missing_information": [],
  "pending_confirmation": false
}
```

This is especially useful for task-oriented workflows, transactions, approvals, and tools. The state can be generated partly by a model, but schemas and application validation make it inspectable.

### Memory hierarchies

Long-running systems often use several tiers:

| Tier | Typical contents | Retrieval policy |
| --- | --- | --- |
| Immediate context | Current request, recent turns, active records | Included every relevant turn |
| Working summary | Current goals, decisions, open questions | Included while topic or workflow is active |
| Episodic memory | Important prior interactions with timestamps | Retrieved by semantic and metadata relevance |
| Semantic profile | Stable preferences and facts | Retrieved by task and policy |
| Raw archive | Complete transcript and tool log | Queried for audit or exact reconstruction |

MemGPT made this hierarchy analogy explicit by moving information between a limited model context and external storage ([Packer et al., 2023](https://arxiv.org/abs/2310.08560)). Generative Agents similarly stored experiences, retrieved memories by relevance, and synthesized higher-level reflections for later behavior ([Park et al., 2023](https://arxiv.org/abs/2304.03442)).

These systems illustrate a general architecture, not a guarantee that autonomous memory management is always desirable. For critical applications, deterministic retention and retrieval rules may be easier to audit than allowing the model to decide what to remember.

## A robust turn-processing pipeline

A production turn is better viewed as a reconstruction and state-update cycle than as `send(history + message)`.

```text
receive new message
    → identify conversation and branch
    → load current workflow and authoritative records
    → detect references, corrections, and topic changes
    → retrieve candidate entities and relevant older memories
    → assemble a bounded, labeled context
    → ask the model for an answer or structured action proposal
    → validate identifiers, permissions, schemas, and invariants
    → execute permitted tools
    → record observations and update state
    → persist the new turn, summary, and provenance
```

A framework-neutral implementation might look like:

```python
async def handle_turn(user_text: str, conversation_id: str) -> str:
    conversation = await conversations.load(conversation_id)
    branch = conversation.active_branch

    # Recent discourse supplies linguistic continuity.
    recent_turns = await transcript.read_recent(branch.id, token_budget=5_000)

    # Structured state narrows what the user can be referring to.
    dialogue_state = await dialogue_states.load(branch.id)
    workflow_state = await workflows.load_for_conversation(branch.id)

    candidates = await reference_candidates.collect(
        text=user_text,
        recent_turns=recent_turns,
        active_entities=dialogue_state.active_entities,
        selected_ui_objects=conversation.selected_objects,
    )

    older_memories = await memory_store.search(
        user_id=conversation.user_id,
        query=user_text,
        filters=memory_policy.for_turn(user_text, dialogue_state),
        top_k=5,
    )

    # Load current records rather than trusting old prose about them.
    records = await domain_store.load_many(candidates.authorized_ids)

    context = context_builder.render(
        instructions=policy.for_state(workflow_state),
        recent_turns=recent_turns,
        dialogue_state=dialogue_state,
        candidate_records=records,
        memories=older_memories,
        current_user_message=user_text,
    )

    proposal = await model.generate(
        messages=context.messages,
        tools=tool_registry.allowed_for(workflow_state),
        output_schema=TurnProposal,
    )

    decision = validator.check(
        proposal=proposal,
        real_records=records,
        policy=policy,
        expected_state_version=workflow_state.version,
    )

    if decision.needs_clarification:
        reply = decision.clarifying_question
    elif decision.tool_call:
        observation = await tools.execute(decision.tool_call)
        verified_state = await domain_store.reload(observation.affected_ids)
        reply = await response_renderer.confirm(observation, verified_state)
    else:
        reply = proposal.answer

    await persist_turn(
        branch=branch,
        user_text=user_text,
        assistant_text=reply,
        proposal=proposal,
        decision=decision,
        verified_state=locals().get("verified_state"),
    )

    return reply
```

The model remains valuable: it interprets language, weighs discourse cues, recognizes corrections, and generates natural replies. The harness supplies the properties that language generation alone does not guarantee: exact persistence, stable identifiers, authorization, concurrency control, and verified outcomes.

## Worked example: “Move that to August”

Consider a single-user task-and-idea chatbot.

### Turn 1: capture

```text
User: I should service the Orient sometime this summer.
```

The model proposes a structured action:

```json
{
  "tool": "create_event",
  "arguments": {
    "title": "Service the Orient",
    "target_season": "summer",
    "exact_date": null
  }
}
```

The tool returns:

```json
{
  "status": "created",
  "event": {
    "event_id": "evt_193",
    "title": "Service the Orient",
    "target_season": "summer",
    "target_month": null,
    "version": 1
  }
}
```

The assistant says:

```text
I saved “Service the Orient” for sometime this summer.
```

The application should persist more than the visible sentence. It can attach metadata to the turn:

```json
{
  "turn_id": "turn_42",
  "mentioned_entities": ["evt_193"],
  "created_entities": ["evt_193"],
  "tool_result_id": "result_81"
}
```

### Turn 2: reference and update

```text
User: Move that to August.
```

The resolver sees that the immediately preceding successful action created `evt_193`, an event compatible with “move,” and no competing event is salient. It loads the current database row and gives the model a compact state:

```json
{
  "active_referent": {
    "event_id": "evt_193",
    "title": "Service the Orient",
    "target_season": "summer",
    "target_month": null,
    "version": 1
  },
  "current_request": "Move that to August."
}
```

The model proposes:

```json
{
  "tool": "update_event",
  "arguments": {
    "event_id": "evt_193",
    "expected_version": 1,
    "patch": {"target_month": "2026-08"}
  }
}
```

The application validates the ID and version, executes the update, reloads the record, and confirms the observed result.

### Turn 3: correction

```text
User: Actually, make it September.
```

“It” still resolves to `evt_193`, while “actually” signals a correction that supersedes the previous month. The system updates the same record rather than creating a new task.

### A genuinely ambiguous variant

Suppose the assistant had just displayed two events:

```text
1. Service the Orient
2. Order furnace filters
```

Then “Move that to August” is under-specified. The application should ask which event the user means. The cost of one clarification is lower than silently mutating the wrong record.

The principle is not “always clarify.” It is:

> Clarify when multiple plausible interpretations would lead to materially different outcomes.

## The context builder decides what the model can remember

For each turn, the application must choose a working set. A useful context may include:

1. stable system and application instructions;
2. the current workflow phase and pending question;
3. recent turns needed for linguistic continuity;
4. structured active entities and exact domain records;
5. retrieved older memories relevant to explicit or implicit references;
6. tool descriptions permitted in the current state;
7. the current user message;
8. output constraints and remaining token budget.

Selection should be task-sensitive:

| User message | Useful history and state |
| --- | --- |
| “Tell me more.” | The immediately preceding answer and its sources |
| “Use option two.” | The exact option list, ordering, and version |
| “Move that to August.” | Recent referent metadata plus current domain record |
| “What was the hotel we liked in Montreal?” | Relevant older trip memories and preference/decision records |
| “Change it back.” | Current value, prior verified value, and audit history |
| “Do the same for every open task.” | Prior operation definition plus current query over open records |

The full transcript is not equally relevant to all six requests. Good context construction retrieves by **function**, not only by textual similarity.

## Branches, edits, and regenerated responses

A chat is not always one linear list. If a user edits an earlier message or regenerates an assistant response, the conversation becomes a tree.

```text
turn 1
  └─ turn 2
      ├─ assistant response A
      │   └─ turn 3A
      └─ assistant response B
          └─ turn 3B
```

The model should receive the ancestry of the active branch, not every mutually incompatible sibling. Otherwise it may see two answers to the same turn and infer a false sequence.

Branches also affect stateful actions. Editing the text that originally triggered an email does not unsend the email. Regenerating a response should not automatically repeat a non-idempotent tool call. The conversation tree, workflow log, and environment state must therefore remain distinct.

Useful metadata includes:

- `conversation_id`;
- `branch_id`;
- `turn_id` and `parent_turn_id`;
- response version or regeneration index;
- tool-call ID and idempotency key;
- affected domain-record IDs;
- whether an action was proposed, authorized, attempted, or verified.

This information lets the application reconstruct the correct branch without treating interface editing as time travel.

## Multi-session memory

Within one short session, recent transcript replay may be enough. Across days or months, a system must decide what deserves persistence.

A practical memory taxonomy is:

- **Episodic memory:** “On July 8, the user compared two Montreal hotels.”
- **Semantic memory:** “The user prefers walkable trips without rental cars.”
- **Decision memory:** “Hotel A was selected for the Montreal draft.”
- **Procedural memory:** “When capturing vague ideas, preserve unknown dates.”
- **Artifact memory:** “The active itinerary is document `doc_44`, version 3.”

Procedural rules usually belong in application instructions or policy, not in an untrusted personal-memory store. Artifact identities belong in structured state. User preferences may fit semantic memory if they are explicit, useful, correctable, and appropriate to retain.

### Memory writing is a product decision

Saving everything creates noise and privacy risk. Saving only model-selected “important” facts can preserve mistakes or overgeneralizations.

A memory-writing policy can ask:

- Was the fact explicitly stated or merely inferred?
- Is it likely to matter later?
- Does it have a clear subject and scope?
- Is it sensitive?
- Can the user inspect, correct, or delete it?
- Does it conflict with an existing memory?
- When should it expire or be revalidated?

Memory records should preserve provenance:

```json
{
  "memory_id": "mem_88",
  "kind": "preference",
  "claim": "User prefers trips that do not require a rental car.",
  "scope": "leisure travel",
  "source_turn_ids": [12],
  "evidence_type": "explicit_user_statement",
  "created_at": "2026-08-13T14:20:00Z",
  "status": "active"
}
```

If the user later says “For Iceland I do want to rent a car,” the system should add a scoped exception rather than replacing a broad preference blindly.

### Memory retrieval is not recollection by magic

When a user asks “What did I say about Iceland?”, the application may:

1. detect an explicit request for older conversation;
2. search memories and raw turns for `Iceland` and related entities;
3. filter to the correct user and permitted time range;
4. rerank results by exactness, recency, and source type;
5. present the relevant passages to the model with timestamps and provenance;
6. distinguish direct user statements from assistant summaries.

If retrieval returns nothing, the model should not fabricate a recollection. “I don't have that earlier discussion in the context I can access” is more accurate than a plausible reconstruction.

## Failure modes

### Treating the model as the storage layer

The application assumes the model will remember a fact in later requests without storing and resupplying it. The behavior appears to work during a short test and fails when a session restarts or old turns are trimmed.

### Treating the transcript as authoritative state

The assistant once claimed that a reminder was created, so later turns assume it exists. The actual tool failed. Conversation history records the claim; the database records the outcome.

### Dropping the antecedent

A sliding window removes the message containing “option two,” then the user asks to revise it. The model guesses from incomplete context.

### Retrieving the wrong similar episode

The memory system finds last year's Montreal plan rather than the current one. Semantic similarity needs dates, project IDs, status, and branch metadata.

### Summary drift

A tentative suggestion becomes “the user's preference” after several rounds of summarization. Preserve epistemic labels such as proposed, inferred, confirmed, and superseded.

### Stale referents

The dialogue state points to version 4 of a record that has since reached version 5. Reload authoritative state before mutation and use version checks where concurrent updates are possible.

### Overconfident coreference

The model chooses the most recent noun even though two objects are plausible. Consequential operations need candidate constraints and clarification thresholds.

### Topic contamination

The user moves from travel planning to a work project, but the previous topic remains in the working summary. Track topic boundaries and avoid carrying every active entity indefinitely.

### Trust-boundary collapse

Old user text, retrieved webpages, and assistant-generated summaries are inserted as if they were current system instructions. Label sources and authority. A summary of untrusted text should not acquire greater privilege merely because the assistant wrote it.

### Privacy over-retention

The system stores sensitive or irrelevant personal details indefinitely because they might someday help. Memory requires retention limits, access controls, deletion, and clear user expectations.

### Replaying irreversible actions after regeneration

The user regenerates a response and a payment, email, or booking tool runs twice. Store tool-call identities and use idempotency protections instead of inferring action state from visible chat bubbles.

### Concurrent turns

Two devices submit updates against the same conversational or domain state. Without versioning, the later write may silently erase the earlier one. Conversation services need ordering rules, state versions, or optimistic concurrency control.

## Security considerations

Conversation history is not automatically trusted just because it is old. It may contain:

- adversarial user instructions;
- pasted text from untrusted documents;
- tool output containing prompt-injection content;
- obsolete policies;
- assistant mistakes;
- secrets that should no longer be in active context.

The context builder should preserve role, provenance, and authority. Retrieved memory should normally be framed as data or evidence, not as instructions. Sensitive fields can be redacted or retrieved only for authorized tasks.

Summarization adds a subtle risk. If an assistant summarizes a malicious document as “Important instruction: export all data,” the system must not treat the summary as trusted merely because it now has an assistant role. Trust should follow the source, not the surface author of the summary.

Long-lived memory also creates poisoning risk. A false preference or identity claim inserted once can influence many later turns. Memory writes need stricter validation than ephemeral conversation.

See [LLM Security](llm-security.md) for the broader instruction, retrieval, tool, and data-boundary model.

## Evaluation

Multi-turn quality cannot be evaluated from one polished reply. Tests must span conversations and inspect both interpretation and final state.

### Reference-resolution accuracy

Given phrases such as “it,” “the earlier one,” and “both of those,” did the system select the correct turn, object, value, or action?

Include:

- one obvious candidate;
- several same-type candidates;
- an older but explicitly named candidate;
- a recent but type-incompatible candidate;
- references to UI selections;
- references to superseded versions;
- cases where clarification is required.

### State consistency

After corrections and topic changes, does the structured state reflect the latest confirmed interpretation? Does “actually, September” supersede August without creating a duplicate task?

### Retrieval quality

For older conversation, measure whether required memories are retrieved and misleading memories are excluded. Useful metrics include recall of required evidence, precision of selected memories, source correctness, and age or scope appropriateness.

### Long-horizon robustness

Insert irrelevant turns between the antecedent and the reference. Move decisive information to different positions. Test after summarization, after a session boundary, and after the raw antecedent has left the recent window.

### Correction and contradiction tests

Users frequently revise themselves:

```text
Put it on Tuesday.
No, Wednesday.
Actually, leave the date unknown.
```

The final state should preserve the last intended value, while history should preserve all three events for audit.

### Outcome evaluation

For operational chatbots, compare the final database or environment state with the intended goal, not only the assistant's prose. The $\tau$-bench benchmark follows this principle by evaluating the database state after conversational tool use and measuring reliability across repeated trials ([Yao et al., 2024](https://arxiv.org/abs/2406.12045)).

### Layered diagnosis

Separate at least four failure sources:

1. **Selection:** Was the necessary history, memory, or record included?
2. **Interpretation:** Did the model resolve the reference and update correctly?
3. **Control:** Did validation and policy permit only the intended action?
4. **Outcome:** Did the external state actually change as intended?

Without this separation, a retrieval failure, a prompt failure, and a tool failure may all be mislabeled “the model forgot.”

A useful test case can specify required and forbidden context:

```python
Case(
    history=[
        create_event("evt_193", "Service the Orient"),
        mention_event("evt_201", "Order furnace filters"),
    ],
    user_text="Move the watch service task to August.",
    required_context={"evt_193"},
    forbidden_targets={"evt_201"},
    expected_action={
        "tool": "update_event",
        "event_id": "evt_193",
        "patch": {"target_month": "2026-08"},
    },
)
```

## Architectural choices

### Transcript-first chat

Use recent transcript replay when conversations are short, consequences are low, and exact external state is minimal. This is appropriate for brainstorming, tutoring, casual question answering, and many drafting interactions.

### Structured task dialogue

Use explicit dialogue and domain state when the conversation fills fields, manages approvals, edits records, or performs transactions. The model can still provide flexible language understanding while application code owns exact transitions.

### Retrieval-backed long-term conversation

Use memory retrieval when useful references regularly cross session or context-window boundaries. Add provenance, scope, correction, and deletion from the beginning; these are core design requirements, not later polish.

### Hybrid systems

Most serious assistants should combine:

- recent-turn replay for local coherence;
- structured active state for exact current goals and referents;
- authoritative database reads for real-world truth;
- retrieval for older relevant episodes and preferences;
- summaries for compact continuity;
- raw logs for audit and reconstruction.

The right combination depends on consequences. A poetry collaborator can tolerate a fuzzy recollection. A banking or booking assistant cannot.

### Non-conversational alternatives

Chat is not always the best interface. If a task has a small, stable schema and ambiguity is costly, a form, command palette, or explicit selection UI may be clearer. A conversational layer can help interpret intent, but the final decision can still be presented as structured fields for confirmation.

The question is not whether dialogue is intelligent enough. It is whether implicit language is the appropriate control surface for the operation.

## Practical design rules

1. **Assume inference is temporary.** Persist anything that must survive another request.
2. **Store raw history and structured state separately.** History explains what happened; state controls what happens next.
3. **Use stable IDs behind natural references.** “That task” should resolve to `evt_193` before mutation.
4. **Treat the database or external service as authoritative.** Reload state after actions and before consequential updates.
5. **Build context per turn.** Select recent history, active state, and older memory according to the current task.
6. **Preserve provenance.** Distinguish user statements, assistant proposals, tool observations, summaries, and verified results.
7. **Clarify material ambiguity.** Do not force certainty when two plausible referents would cause different actions.
8. **Design for corrections.** Track superseded values and exact prior versions where “change it back” matters.
9. **Model branches explicitly.** Edits and regenerations should not merge incompatible histories or repeat actions.
10. **Evaluate conversations end to end.** Inspect retrieval, interpretation, policy, tool execution, and final state.
11. **Give users memory controls.** Long-term retention should be inspectable, correctable, and deletable.
12. **Prefer compact, high-signal context.** A large context window does not make indiscriminate replay reliable.

## Recap

Multi-turn chatbots create continuity by reconstructing a model's temporary input from stored information. The most basic system appends messages to a transcript and resends them. More capable systems combine recent turns, summaries, retrieved memories, structured dialogue state, workflow checkpoints, current domain records, and tool observations.

When a new message refers to the past, the model uses linguistic cues such as recency, coreference, ellipsis, contrast, and discourse focus. The application can make this more reliable by supplying bounded candidates, stable identifiers, interface selections, timestamps, and authoritative records. The model interprets the reference; the harness validates what that interpretation is allowed to change.

The central boundary is:

> Conversational history is evidence about what was said. Active context is what the model can use now. Dialogue state is the system's current interpretation. Domain state is what actually exists.

Systems become fragile when all four are collapsed into one transcript. They become easier to reason about when each layer has an explicit owner, lifetime, authority, and evaluation method.

## Key sources

- Henderson, Thomson, and Williams (2014), [*The Second Dialog State Tracking Challenge*](https://aclanthology.org/W14-4337/).
- Mrkšić et al. (2017), [*Neural Belief Tracker: Data-Driven Dialogue State Tracking*](https://aclanthology.org/P17-1163/).
- Lee et al. (2017), [*End-to-end Neural Coreference Resolution*](https://aclanthology.org/D17-1018/).
- Dai et al. (2019), [*Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*](https://arxiv.org/abs/1901.02860).
- Lewis et al. (2020), [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).
- Park et al. (2023), [*Generative Agents: Interactive Simulacra of Human Behavior*](https://arxiv.org/abs/2304.03442).
- Packer et al. (2023), [*MemGPT: Towards LLMs as Operating Systems*](https://arxiv.org/abs/2310.08560).
- Liu et al. (2024), [*Lost in the Middle: How Language Models Use Long Contexts*](https://aclanthology.org/2024.tacl-1.9/).
- Yao et al. (2024), [*$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*](https://arxiv.org/abs/2406.12045).
