---
title: AI Research Library
slug: /
sidebar_position: 1
type: index
status: active
updated: 2026-07-30
---

# AI Research Library

This is a selective research library about large language models and LLM application architecture. It is not intended to teach every prerequisite or preserve every detail from prior research. The goal is to build a compact set of articles that make the important ideas understandable, show how they appear in code, and clarify the engineering decisions they create.

## Editorial approach

Each entry should generally proceed in this order:

1. **Conceptual model:** What is the idea, and what intuition makes it useful?
2. **Mechanism:** What actually happens inside the model or application?
3. **Worked example:** What does it look like in pseudocode or a small implementation?
4. **Architectural choices:** When should it be used, and what are the alternatives?
5. **Failure modes:** What commonly goes wrong?
6. **Research support:** Which primary papers establish or test the important claims?

Academic papers are evidence rather than the organizing principle. A paper entry explains the parts that changed how LLMs work or are built; it does not reproduce every experiment or section.

## Current entries

### Concepts

- [The LLM Request Lifecycle](llm-request-lifecycle.md) — How instructions, user messages, retrieved material, tool definitions, tool results, and generation fit into one end-to-end cycle.
- [Context Engineering](context-engineering.md) — How an application chooses and arranges the model's temporary working information.
- [Retrieval-Augmented Generation](retrieval-augmented-generation.md) — How retrieval and generation are combined, where RAG is useful, and where an ordinary database is the better tool.

### Paper guides

- [Attention Is All You Need](paper-attention-is-all-you-need.md) — The Transformer paper, explained through the architectural change that made modern LLMs possible.
- [Chatbot Evolution](chatbot-evolution.md) — How zero-shot transfer, universal text interfaces, instruction tuning, and conversational alignment converged into the general-purpose chatbot.

## Suggested reading paths

**For building the current application**

1. [The LLM Request Lifecycle](llm-request-lifecycle.md)
2. [Context Engineering](context-engineering.md)
3. [Retrieval-Augmented Generation](retrieval-augmented-generation.md)
4. *Prompt, Context, Agent, and Harness* — planned
5. *Structured Output and Tool Calling* — planned
6. *Memory for a Persistent Conversational Application* — planned

**For understanding the research lineage**

1. [Attention Is All You Need](paper-attention-is-all-you-need.md)
2. *Language Models are Few-Shot Learners* — planned
3. *Training Language Models to Follow Instructions with Human Feedback* — planned
4. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* — planned
5. *ReAct: Synergizing Reasoning and Acting in Language Models* — planned
6. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — planned
7. *Toolformer: Language Models Can Teach Themselves to Use Tools* — planned

## Priority backlog

Priority reflects recent questions, relevance to the single-user task-and-idea chatbot, and influence on modern LLM development.

| Priority | Entry | Kind | Why it belongs early |
| --- | --- | --- | --- |
| 1 | Prompt, context, agent, and harness | Concept | Establishes the vocabulary needed to compare frameworks and architectures. |
| 2 | Structured output and tool calling | Concept | Directly supports reliable event creation and updating. |
| 3 | Memory for persistent conversational applications | Concept | Central to user-specific storage, retrieval, and continuity. |
| 4 | ReAct | Paper | Connects model reasoning to external actions and observations. |
| 5 | Model Context Protocol | Concept | Explains a standard interface between models and external capabilities. |
| 6 | RAG paper guide | Paper | Gives the research foundation beneath the practical RAG article. |
| 7 | InstructGPT / RLHF | Paper | Explains why an instruction-following assistant differs from a base language model. |
| 8 | Toolformer | Paper | Provides an influential formulation of learned tool selection and invocation. |
| 9 | Agent orchestration patterns | Applied guide | Helps choose among PydanticAI, Agno, LangGraph, Vercel AI SDK, and simpler prototypes. |
| 10 | Evaluation for conversational task capture | Applied guide | Defines how to test extraction, storage, retrieval, updating, and conversational quality. |

## Entry types

- **Concept:** A durable mental model that may draw on several papers and implementations.
- **Paper guide:** An explanation of one influential paper, emphasizing what changed and what remains relevant.
- **Applied guide:** A design or implementation decision grounded in the concepts and papers.
- **Comparison:** A criteria-driven evaluation of competing architectures, frameworks, or providers.

## Maintenance rules

- Prefer updating an existing entry over creating a near-duplicate.
- State whether a claim comes from a paper, a current implementation, or an engineering inference.
- Keep implementation examples framework-neutral unless a framework is itself the subject.
- Put volatile details—prices, model names, API limits, current defaults—in clearly dated sections.
- Use links between entries so the library can be read selectively rather than linearly.
