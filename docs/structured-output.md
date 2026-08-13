---
title: "Structured Output"
type: concept
status: active
updated: 2026-08-13
tags: [structured-output, schemas, json, tool-use, validation, constrained-decoding, harnesses]
---

# Structured Output in LLM Systems

**Central idea:** Structured output asks an LLM to return information in a machine-readable form—usually JSON conforming to a schema—so application code can interpret the response as typed data rather than prose.

**Why it matters:** Ordinary language is flexible but ambiguous. Software needs stable field names, known types, explicit alternatives, and predictable errors. Structured output creates a contract between probabilistic language generation and deterministic application logic.

## Background topics

- **Autoregressive generation:** How an LLM produces one token at a time from a large vocabulary.
- **JSON and schemas:** How software represents objects, types, required fields, enumerations, and null values.
- **Parsing and validation:** How a program distinguishes readable data from valid domain data.
- **Tool calling:** How structured model output can propose an external action.
- **Harness engineering:** How application code authorizes, executes, verifies, and records model proposals.
- **Information extraction:** How concepts expressed in language become fields in a data model.

These topics explain both the format the model produces and the system that consumes it.

## Before structured output

Early LLM applications commonly asked the model to embed data inside prose:

```text
The task is “Buy furnace filters,” and no date was specified.
```

Application code then searched the response with regular expressions, string splitting, or another model call. Small stylistic changes could break the parser. Asking the model to “return JSON” improved consistency, but ordinary prompting could still produce missing fields, unexpected labels, invalid punctuation, or commentary surrounding the object.

This exposed a fundamental mismatch:

```text
LLM strength: flexible language
Application need: stable interface contract
```

Structured-output systems address that mismatch by making the permitted output shape explicit and, in stronger implementations, constraining generation so invalid structures cannot be produced.

## The topic in one view

Suppose an application needs an event proposal rather than a paragraph. It can define a schema conceptually equivalent to:

```json
{
  "operation": "create_event",
  "title": "Buy furnace filters",
  "date": null,
  "needs_clarification": false,
  "unknown_fields": ["date"]
}
```

The schema can require `operation` and `title`, restrict `operation` to an enumeration, allow `date` to be either a string or `null`, and require `unknown_fields` to be a list. The model still decides what the user's language means, but it must express that interpretation through a defined interface.

The application workflow becomes:

```text
request
  → choose schema
  → generate structured response
  → parse and validate
  → apply domain and permission rules
  → execute if appropriate
  → verify the resulting state
```

Structured output therefore changes an LLM response from an answer to a **typed proposal** that software can inspect before trusting or acting on it.

## Levels of structure

Not every feature described as “structured output” provides the same guarantee.

| Approach | What it provides | Remaining weakness |
| --- | --- | --- |
| **Prompted formatting** | An instruction such as “return JSON” | The model may ignore or partially follow it |
| **JSON mode** | Syntactically valid JSON | The object may not match the intended schema |
| **Schema-constrained output** | Required structure, types, and enumerated values | Values can still be factually or semantically wrong |
| **Tool or function call** | A structured proposal tied to a named capability | The application must authorize and execute it |
| **Parse, validate, and retry** | Recovery when unconstrained output fails | Adds latency and may repeat the same mistake |

Provider terminology and supported schema features vary. For example, OpenAI distinguishes JSON mode from schema-constrained Structured Outputs: JSON mode guarantees a valid JSON object, while Structured Outputs targets conformance to a supplied JSON Schema ([OpenAI, 2024](https://openai.com/index/introducing-structured-outputs-in-the-api/)). The general architectural distinction applies beyond any one API.

## How constrained decoding works

During normal generation, the model assigns probabilities to possible next tokens. A constrained decoder tracks the output produced so far and removes tokens that would make the required grammar or schema impossible to complete.

If $V$ is the model's vocabulary and $G(y_{<t})$ is the set of tokens allowed by the schema after the current prefix, generation is restricted to:

$$
y_t \in V \cap G(y_{<t})
$$

For example, after producing an object key whose schema type is Boolean, the decoder can permit `true` or `false` while preventing an arbitrary sentence. PICARD demonstrated this general idea by incrementally rejecting invalid continuations during text-to-SQL generation ([Scholak, Schucher, and Bahdanau, 2021](https://aclanthology.org/2021.emnlp-main.779/)). Later work showed that grammar-constrained decoding can support a wider range of structured NLP tasks without task-specific fine-tuning ([Geng et al., 2023](https://aclanthology.org/2023.emnlp-main.674/)).

Constraints reduce the model's possible surface forms. They do not tell it which valid value is true. If an `event_id` must be a string, the model can still produce a well-formed identifier that does not exist.

## The five validation layers

Structured output is most reliable when the application treats correctness as a sequence of layers.

| Layer | Question | Typical owner |
| --- | --- | --- |
| **Syntax** | Can the response be parsed? | Decoder or parser |
| **Schema** | Are the required fields, types, and enums correct? | Schema validator |
| **Semantics** | Does the date exist? Does the identifier resolve? | Domain code |
| **Policy** | Is this user allowed to perform the proposed action? | Authorization layer |
| **Outcome** | Did execution produce the intended external state? | Tool and verification logic |

A strict schema can largely solve the first two layers. It cannot replace the other three. This is the same boundary emphasized in [Harness Engineering](harness-engineering.md): the model interprets and proposes; the application controls and verifies.

## Designing useful schemas

### Represent uncertainty explicitly

If every field is required and no unknown state exists, the model is pressured to invent values. Use `null`, an `unknown` enum, or an explicit clarification field when missing information is legitimate.

### Prefer narrow alternatives

Enums such as `create`, `retrieve`, and `update` are easier to validate than an unrestricted operation string. Do not make the schema so narrow that common valid cases cannot be represented.

### Separate identity from description

Use authoritative IDs for updates and human-readable labels for display. Never assume that a generated title uniquely identifies a database record.

### Separate proposals from results

The model may produce:

```json
{"operation": "update_event", "event_id": "evt_193", "date": "2026-09-01"}
```

That object means “request this update,” not “the update happened.” The tool result must report actual success, failure, or ambiguity.

### Keep schemas task-specific

One enormous schema covering conversation, research, calendars, editing, and administration makes selection difficult and creates many meaningless field combinations. Choose a small schema appropriate to the current workflow, as described in [Dynamic Selection](dynamic-selection.md).

### Version the contract

Changing field names, nullability, or enum meanings can break downstream consumers even if the model continues producing valid objects. Treat schemas like APIs: version them, test migrations, and evaluate model behavior before deployment.

## What structured output solves

- Reliable handoff from language interpretation to application code
- Easier parsing, validation, logging, and testing
- Explicit alternatives and missing-value handling
- Safer tool arguments and bounded action proposals
- Field-level evaluation instead of judging only whole paragraphs
- Less brittle dependence on phrasing and punctuation

## What it does not solve

- **Factual correctness:** A schema-valid answer can contain invented facts.
- **Semantic correctness:** Types may be correct while meanings are wrong.
- **Authorization:** Valid tool arguments do not grant permission.
- **Prompt injection:** Untrusted content can still influence selected values or actions.
- **Business rules:** Cross-field constraints often require domain code.
- **Verified completion:** A proposed action is not evidence that execution succeeded.

The dangerous mistake is to equate “parsed successfully” with “safe and correct.” Structured output improves the interface, not the truthfulness of everything passing through it.

## Application to the task-and-idea chatbot

The prototype can use separate schemas for its three core operations:

| User intent | Structured result |
| --- | --- |
| Capture | Proposed title, notes, optional date, and unresolved fields |
| Retrieve | Search query, filters, and result limit |
| Update | Authoritative event ID, proposed changes, and clarification status |

For “Remind me to buy furnace filters,” the model can produce a skeletal event immediately, leaving the date `null`. For “Move that to next month,” the system must first resolve “that” to a real stored event. The model may propose the update, but application code validates the ID and date, writes the transaction, reloads the record, and only then confirms success.

This preserves the MVP's save-first behavior without forcing the model to manufacture completeness.

## Evaluation

Measure structured output at several levels:

- parse and schema-conformance rate;
- field-level precision, recall, and enum accuracy;
- correct preservation of unknown or ambiguous values;
- semantic and cross-field validation failures;
- correct schema or tool selection;
- unauthorized or unnecessary action proposals;
- final database state after execution;
- retry count, latency, and token cost.

Tests should include missing information, contradictory requests, long inputs, invalid dates, nonexistent IDs, and schema-version changes. A perfect parse rate with poor field accuracy is not a successful system.

## Recap

Structured output gives an LLM a typed interface to the rest of an application. Prompting can encourage a format; JSON mode can guarantee parseable JSON; schema-constrained decoding can enforce a particular structure. Tool calling uses similar structures to represent proposed actions.

These mechanisms make model responses easier to integrate and evaluate, but they guarantee shape more readily than meaning. Semantic validation, permissions, execution, and verification remain responsibilities of the surrounding software.

The practical rule is:

> Treat structured output as validated input to your application—not as trusted truth or proof that an action occurred.

## Key sources

- Scholak, Schucher, and Bahdanau (2021), [*PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models*](https://aclanthology.org/2021.emnlp-main.779/).
- Geng et al. (2023), [*Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*](https://aclanthology.org/2023.emnlp-main.674/).
- Beurer-Kellner, Fischer, and Vechev (2022), [*Prompting Is Programming: A Query Language for Large Language Models*](https://arxiv.org/abs/2212.06094).
- OpenAI (2024), [*Introducing Structured Outputs in the API*](https://openai.com/index/introducing-structured-outputs-in-the-api/).
