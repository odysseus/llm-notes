---
title: Context Engineering
type: concept
status: pilot
updated: 2026-07-30
tags: [context, prompts, memory, rag, agents]
---

# Context Engineering

Context engineering is the design of the information environment in which a model answers one request.

A prompt is part of that environment, but not the whole of it. A useful approximation is:

$$
\text{context} =
\text{instructions}
+ \text{current request}
+ \text{conversation state}
+ \text{retrieved evidence}
+ \text{application state}
+ \text{tool descriptions}
+ \text{examples}
+ \text{output constraints}
$$

The aim is not to fill the context window. It is to construct the smallest working set that gives the model the right evidence, priorities, and capabilities for the current task.

## A working-memory analogy

Model parameters contain broad learned patterns and knowledge. The context window is closer to temporary working memory: it contains the particular material available during this inference.

An application usually has much more potential context than will fit—or than should be sent:

- months of conversation;
- every task and idea in a database;
- user preferences;
- documents and search results;
- tool descriptions;
- business policies;
- examples of desired behavior.

Context engineering chooses which pieces enter the working set and how they are represented.

This resembles memory hierarchy in ordinary software. MemGPT made that analogy explicit by treating limited LLM context as a fast memory tier and moving information between context and external storage ([Packer et al., 2023](https://arxiv.org/abs/2310.08560)).

## Context engineering versus nearby concepts

| Concept | Primary question |
| --- | --- |
| Prompt engineering | How should an instruction or example be worded? |
| RAG | Which external evidence should be retrieved for this query? |
| Conversation memory | Which prior interactions should persist and be recalled? |
| Tool design | Which actions are available and how are they described? |
| Context engineering | How should all relevant inputs be selected, shaped, ordered, and budgeted for this call? |

RAG, memory, and prompting are therefore context-engineering techniques, not synonyms for the whole discipline.

## The six core operations

### 1. Select

Include information because it has a reason to affect this answer.

Useful selection signals include:

- semantic relevance to the request;
- recency;
- authority;
- the type of task being performed;
- explicit references such as “the plan we made yesterday”;
- exact identifiers already known by the application.

A current database record should normally outrank an old conversational statement about that record.

### 2. Represent

The same fact can be presented as prose, JSON, a table, or a compact summary. Representation changes how reliably it can be used.

For exact application state, structured data is usually preferable:

```json
{
  "event_id": "evt_193",
  "title": "Service the Orient",
  "status": "idea",
  "target_month": "2026-08",
  "exact_date": null
}
```

For qualitative preference, concise prose may be better:

```text
User preference: prioritize long-term serviceability over novelty in watch recommendations.
Source: explicit user statement.
```

### 3. Order

Put stable rules where the runtime and model treat them as instructions. Clearly delimit retrieved evidence. Keep the current request easy to identify.

Position also matters empirically. *Lost in the Middle* found that several long-context models performed better when relevant material appeared near the beginning or end than when it appeared in the middle ([Liu et al., 2024](https://aclanthology.org/2024.tacl-1.9/)). The exact behavior changes across models, but the engineering lesson is durable: do not bury the decisive evidence inside undifferentiated bulk.

### 4. Compress

Compression can be:

- extractive: retain the most relevant original passages;
- abstractive: summarize older material;
- structural: convert verbose text into fields;
- lossy but targeted: preserve decisions and discard conversational filler.

Compression creates risk. A summary can erase uncertainty, exceptions, or provenance. High-stakes or exact claims should retain links back to source records.

### 5. Separate authority

Not all context should be obeyed.

- Application instructions are directives.
- The current user message is a request.
- Retrieved documents are evidence.
- Tool results are observations.
- Earlier assistant statements may be fallible history.

Clear labels reduce the chance that quoted or retrieved text is mistaken for an instruction. This is particularly important when external documents may contain prompt-injection text.

### 6. Refresh

Context is created per call. After a tool updates the database, the next call should use the returned result or reload the authoritative record. Continuing with stale pre-action state can make the model contradict a successful mutation.

## A context builder

A context builder can be explicit application code rather than a single hand-written prompt:

```python
@dataclass
class TurnContext:
    instructions: list[Message]
    current_state: dict
    memories: list[Memory]
    evidence: list[Passage]
    tools: list[ToolSchema]


def build_context(user_text: str, user_id: str) -> TurnContext:
    intent = classify_intent(user_text)

    state = load_authoritative_state(
        user_id=user_id,
        scope=state_scope_for(intent),
    )

    memories = memory_store.search(
        user_id=user_id,
        query=user_text,
        filters={"kind": memory_kinds_for(intent)},
        top_k=5,
    )

    evidence = []
    if intent.requires_external_knowledge:
        evidence = research_index.search(user_text, top_k=6)

    return TurnContext(
        instructions=rules_for(intent),
        current_state=state,
        memories=deduplicate(memories),
        evidence=rerank(evidence),
        tools=tools_for(intent),
    )
```

Serialization is a separate step:

```python
def render(ctx: TurnContext, user_text: str) -> list[dict]:
    return [
        *ctx.instructions,
        {
            "role": "developer",
            "content": "<current_state>\n"
                       + json.dumps(ctx.current_state)
                       + "\n</current_state>",
        },
        {
            "role": "developer",
            "content": render_memories(ctx.memories),
        },
        {
            "role": "developer",
            "content": render_evidence_with_sources(ctx.evidence),
        },
        {"role": "user", "content": user_text},
    ]
```

Separating selection from rendering makes the system testable. You can evaluate whether the right memories were selected independently from whether the model used them correctly.

## Context for the task-and-idea chatbot

The application should not send every stored event on every turn. A better policy is intent-sensitive:

| User request | Context to retrieve |
| --- | --- |
| “Add ‘replace grill cover’” | Event schema, creation rules, perhaps nearby duplicate titles |
| “What should I do this weekend?” | Incomplete/open events, relevant preferences, date/location state |
| “Move that to August” | The specifically referenced event and the immediately preceding turn |
| “What were the Miyota watches?” | Relevant discussion memory or research notes, not all events |
| “Delete the reminder” | Candidate matching records, plus confirmation policy if ambiguous |

Structured event state should come from the database. Semantic retrieval is useful for resolving fuzzy references and recalling qualitative discussion, but it should not become the only record of whether an event exists.

## Budgeting the context window

A practical budget might reserve capacity by function:

```python
budget = {
    "instructions": 2_000,
    "current_state": 3_000,
    "conversation": 4_000,
    "retrieved_evidence": 8_000,
    "tool_schemas": 3_000,
    "generation": 4_000,
}
```

The numbers are application-specific. The important point is that output tokens, tool definitions, and safety margins compete with retrieved text.

More context can increase:

- latency and inference cost;
- distraction from the relevant evidence;
- exposure to conflicting or malicious text;
- the difficulty of tracing which source drove the answer.

A long supported context window is capacity, not a mandate.

## Common failure modes

### Context dumping

The application inserts all available history “just in case.” The relevant material becomes hard to find, cost grows, and stale statements conflict with current state.

### Summary drift

Repeatedly summarizing summaries turns qualified decisions into absolute facts. Preserve source pointers and periodically rebuild summaries from authoritative material.

### Authority inversion

An old assistant message or retrieved web page overrides current application policy. Label context by role and trust level.

### Retrieval without task awareness

Similarity search finds passages that share vocabulary but do not answer the task. Add metadata filters, task-specific query rewriting, and reranking.

### Hidden state

Important application decisions live only in the transcript. Convert durable facts into structured records.

### Non-deterministic enforcement

Critical rules such as budgets or permission boundaries are expressed only in prose. Enforce exact invariants in code as well as describing them to the model.

## How to evaluate context engineering

Evaluate at least three layers:

1. **Selection:** Did the builder retrieve the necessary facts and exclude misleading ones?
2. **Use:** Given correct context, did the model apply it?
3. **Outcome:** Was the final answer or action correct?

This separation prevents every failure from being mislabeled “a prompt problem.”

A small test fixture can specify required context:

```python
Case(
    user_text="Move that to August",
    conversation=["... created event evt_193 ..."],
    required_records={"evt_193"},
    forbidden_records={"evt_122"},
    expected_action={"tool": "update_event", "event_id": "evt_193"},
)
```

## Practical conclusions

- Build context as data, then render it for the model.
- Retrieve by task, not merely by textual similarity.
- Keep authoritative structured state separate from conversational memory.
- Prefer a small, high-signal working set over maximum context utilization.
- Preserve provenance through summaries and retrieval.
- Test selection, model use, and final outcome independently.

## Key sources

- Packer et al. (2023), [*MemGPT: Towards LLMs as Operating Systems*](https://arxiv.org/abs/2310.08560).
- Liu et al. (2024), [*Lost in the Middle: How Language Models Use Long Contexts*](https://aclanthology.org/2024.tacl-1.9/).
- Dong et al. (2022/2024), [*A Survey on In-context Learning*](https://arxiv.org/abs/2301.00234).
- Lewis et al. (2020), [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).

