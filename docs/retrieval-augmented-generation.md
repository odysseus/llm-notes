---
title: Retrieval-Augmented Generation
type: concept
status: pilot
updated: 2026-07-30
tags: [rag, retrieval, embeddings, context, architecture]
---

# Retrieval-Augmented Generation

Retrieval-augmented generation, or RAG, lets a language model consult an external body of information while answering a request.

The simplest intuition is an open-book exam:

- The model's parameters are what it learned during training.
- The retrieval system finds relevant pages from an external collection.
- The model receives those pages in its context and constructs an answer.

The defining feature is not a vector database. It is the combination of **retrieval** with **generation**.

## Why RAG exists

Knowledge stored only in model parameters has practical limitations:

- updating it usually requires additional training;
- it is difficult to identify the source of a claim;
- domain-specific facts may be absent or weakly represented;
- the model may produce plausible but unsupported details.

The original RAG paper described this as combining **parametric memory** in the generator with **non-parametric memory** in an external dense index ([Lewis et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)).

Modern application RAG is broader than the exact trainable architecture in that paper. Many systems use an off-the-shelf embedding model, a search service, and a separately prompted LLM without jointly training the retriever and generator.

## The two pipelines

RAG has an offline ingestion pipeline and an online query pipeline.

### Ingestion

1. Acquire documents.
2. Parse and normalize them.
3. Divide them into retrievable units.
4. Attach metadata and source identity.
5. Compute a searchable representation.
6. Store the units in an index.

```python
for document in corpus:
    clean_text = normalize(parse(document))

    for chunk in chunk_document(clean_text, target_tokens=500, overlap=60):
        index.add(
            id=stable_chunk_id(document.id, chunk),
            text=chunk.text,
            vector=embed(chunk.text),
            metadata={
                "document_id": document.id,
                "title": document.title,
                "updated_at": document.updated_at,
                "permissions": document.permissions,
            },
        )
```

### Query

1. Interpret or rewrite the user's request for retrieval.
2. Apply permissions and metadata filters.
3. Retrieve candidate passages.
4. Optionally rerank and deduplicate them.
5. Put selected passages into the generation context.
6. Generate an answer with source references.

```python
def answer(question: str, user_id: str) -> Answer:
    query = rewrite_as_standalone_query(question)

    candidates = index.hybrid_search(
        text=query,
        vector=embed(query),
        filters={"readable_by": user_id},
        top_k=30,
    )

    passages = reranker.rank(query, candidates)[:6]

    response = model.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer from the supplied evidence. "
                    "If it is insufficient, say what is missing. "
                    "Cite passage IDs for factual claims."
                ),
            },
            {
                "role": "developer",
                "content": render_passages(passages),
            },
            {"role": "user", "content": question},
        ]
    )

    return attach_source_links(response, passages)
```

## Sparse, dense, and hybrid retrieval

### Sparse retrieval

Methods such as BM25 emphasize lexical overlap. They are strong when exact terms matter:

- product numbers;
- names;
- error codes;
- unusual phrases;
- identifiers.

### Dense retrieval

Dense retrieval embeds queries and passages as vectors. Semantically similar wording can be close even without exact token overlap. Dense Passage Retrieval demonstrated that learned dense representations could outperform a strong BM25 system on several open-domain question-answering benchmarks ([Karpukhin et al., 2020](https://arxiv.org/abs/2004.04906)).

### Hybrid retrieval

Hybrid search combines lexical and semantic evidence. It is often a strong default for application data because user requests contain both fuzzy descriptions and exact entities.

For example, “the blue Orient with the power reserve” benefits from semantic matching, while “WZ0351-EL” benefits from exact lexical matching.

## Chunking is an information-design decision

A retriever does not retrieve “knowledge” in the abstract. It retrieves whatever units were indexed.

Chunks that are too small may omit necessary context. Chunks that are too large may match several unrelated ideas and waste context tokens.

Useful boundaries include:

- headings and sections;
- individual records;
- paragraphs with their parent heading attached;
- question-answer pairs;
- code functions or classes;
- conversation episodes.

For structured event data, one complete event record is often a better retrieval unit than arbitrary 500-token slices.

Chunk overlap can prevent information at a boundary from being lost, but excessive overlap produces duplicate evidence and can cause one source to dominate the context.

## Retrieval is not proof

Similarity answers “Which indexed items resemble this query?” It does not establish that:

- the passage is true;
- the passage is current;
- the passage actually entails the answer;
- the user has permission to see it;
- the source is more authoritative than a conflicting record.

A production system therefore needs:

- source and timestamp metadata;
- permission filters before generation;
- reranking or relevance checks;
- explicit handling of conflicting evidence;
- answerability behavior when evidence is insufficient.

Self-RAG explored adaptive retrieval and model-generated reflection signals because retrieving a fixed number of passages whether or not they are needed can reduce quality ([Asai et al., 2023](https://arxiv.org/abs/2310.11511)).

## RAG versus databases, tools, and fine-tuning

| Need | Usually prefer |
| --- | --- |
| Find a fuzzy passage in unstructured notes | RAG |
| Get the exact current status of event `evt_193` | Database query/tool |
| Enforce a unique parent or maximum budget | Application code/database constraint |
| Teach a stable output style or domain behavior | Prompting or fine-tuning |
| Incorporate frequently changing documents | RAG |
| Perform an external action | Tool call |

RAG should not replace ordinary data access.

For the task-and-idea chatbot, the event table is the source of truth. If the user asks, “Which tasks are incomplete?”, the application should query `status = incomplete`, not retrieve semantically similar event descriptions and hope none are missed.

RAG is valuable for:

- recalling fuzzy prior discussion;
- searching research notes;
- connecting an undeveloped idea with supporting documents;
- finding passages that explain why a recommendation was made;
- resolving references whose exact record is not yet known.

The two approaches can cooperate:

```python
event = db.get_event(event_id)          # exact state
discussion = memory.search(event.title) # qualitative history

context = {
    "authoritative_event": event,
    "relevant_discussion": discussion,
}
```

## Query rewriting

Conversational questions are often poor standalone search queries.

Conversation:

```text
User: Tell me about the Orient WZ0351-EL.
Assistant: ...
User: What about its amplitude when fully wound?
```

Retrieval query:

```text
Orient WZ0351-EL 40N50 movement amplitude fully wound
```

The rewriter should preserve exact names and constraints. Over-expansion can be harmful: adding a guessed caliber or location may exclude the correct material. A safe rewriter distinguishes known context from inferred search terms.

## Reranking and context construction

Initial retrieval is optimized for recall: find a candidate pool likely to include the needed evidence. Reranking is optimized for precision: choose the most useful candidates.

The final context builder may also:

- remove near-duplicates;
- prefer newer versions;
- diversify across subtopics;
- group passages by source;
- place the strongest evidence prominently;
- reserve space for the actual answer.

This is where RAG becomes part of [context engineering](context-engineering.md).

## Evaluation

End-to-end answer quality alone is not diagnostic. Evaluate each stage.

### Retrieval

- **Recall@k:** Was a passage containing the answer retrieved?
- **Precision@k:** How many retrieved passages were actually useful?
- **Ranking quality:** Were the best passages near the top?
- **Permission correctness:** Was any forbidden material retrieved or exposed?

### Generation

- Is each factual claim supported by the supplied passages?
- Are citations attached to the correct claims?
- Does the model abstain when evidence is insufficient?
- Does it distinguish source statements from its own inference?

### System

- latency;
- token and retrieval cost;
- freshness after source updates;
- duplicate and stale-index behavior;
- traceability from answer to source and source version.

A useful evaluation corpus includes adversarial cases:

- two sources that disagree;
- a newer source superseding an older one;
- a query that should be answered from a database rather than retrieval;
- no relevant document;
- a relevant passage containing hostile instructions;
- exact identifiers that embedding-only retrieval might miss.

## Common failure modes

### The wrong source is indexed

No prompt can recover information that never entered the corpus.

### Retrieval finds the topic, not the answer

A passage may discuss “watch amplitude” without containing the threshold or condition the user asked about.

### The generator ignores evidence

Relevant retrieval does not guarantee faithful generation. Strong answer instructions, clear passage labels, and claim-level evaluation are still needed.

### Too many passages

Adding more candidates can introduce contradictions and bury the strongest evidence. Long-context research has shown that position and context structure can affect information use ([Liu et al., 2024](https://aclanthology.org/2024.tacl-1.9/)).

### Stale or duplicate chunks

Re-indexing without stable identities can preserve obsolete versions and multiply near-identical passages.

### Vector-only absolutism

Semantic search is not inherently superior to lexical search. Exact codes, rare names, and quoted phrases frequently favor sparse retrieval.

## Practical defaults

For an initial application:

- index at natural semantic boundaries;
- store stable source IDs, timestamps, and permissions;
- use hybrid retrieval when both natural-language descriptions and exact names matter;
- retrieve a moderate candidate pool, then rerank to a small final set;
- require citations for research answers;
- make “insufficient evidence” an allowed result;
- use database queries for exhaustive or exact structured-state questions;
- log retrieved passage IDs and source versions for debugging.

## Key sources

- Lewis et al. (2020), [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).
- Karpukhin et al. (2020), [*Dense Passage Retrieval for Open-Domain Question Answering*](https://arxiv.org/abs/2004.04906).
- Asai et al. (2023), [*Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*](https://arxiv.org/abs/2310.11511).
- Liu et al. (2024), [*Lost in the Middle: How Language Models Use Long Contexts*](https://aclanthology.org/2024.tacl-1.9/).

