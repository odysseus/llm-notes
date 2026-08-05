---
title: "Attention Is All You Need"
type: paper-guide
status: pilot
updated: 2026-07-30
paper_year: 2017
tags: [transformers, attention, architecture, foundational-papers]
---

# Paper Guide: *Attention Is All You Need*

**Paper:** Ashish Vaswani et al., 2017  
**Primary link:** [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)  
**Original problem:** Neural machine translation  
**Why it matters:** It introduced the Transformer architecture, the foundation from which modern large language models developed.

## The paper in one sentence

The paper showed that sequence modeling did not require recurrent or convolutional processing: a network built around attention could model relationships among tokens, train with far more parallelism, and achieve state-of-the-art translation results.

## The problem it was solving

Before Transformers, strong sequence models were commonly recurrent neural networks, including LSTMs and GRUs.

A recurrent network processes a sequence through a changing hidden state:

```text
state_1 -> state_2 -> state_3 -> ... -> state_n
```

That design matches the sequential nature of language, but creates two major difficulties:

1. **Training is inherently sequential within each example.** The computation for later positions depends on earlier hidden states, limiting parallelism.
2. **Long-range relationships have long computational paths.** Information from a distant token must pass through many intermediate states.

Attention already existed, but was generally used alongside recurrence. Vaswani and colleagues asked whether attention could become the central computational mechanism and recurrence could be removed.

## The conceptual leap

Instead of carrying one evolving state through the sentence, let every token construct a new representation by consulting other relevant tokens.

Consider:

> The animal didn't cross the street because **it** was tired.

To represent “it,” the model should use information from “animal.” Self-attention lets the token at “it” assign weight to other positions and combine their information.

The computation is repeated at every position, in parallel. That parallelism was a decisive training advantage.

## Queries, keys, and values

Each token representation is projected into three vectors:

- **Query:** What information is this position looking for?
- **Key:** What kind of information does this position offer?
- **Value:** What information should be transferred if this position is selected?

The query at one position is compared with the keys at other positions. Similarity scores become weights, and the output is a weighted sum of the values.

The paper's scaled dot-product attention is:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The pieces have direct interpretations:

1. $QK^T$ measures query-key compatibility.
2. Dividing by $\sqrt{d_k}$ prevents large dot products from pushing the softmax into poorly behaved regions.
3. Softmax turns scores into weights.
4. Multiplication by $V$ combines the selected information.

A minimal implementation is:

```python
def attention(Q, K, V):
    scores = Q @ K.T
    scores = scores / math.sqrt(K.shape[-1])
    weights = softmax(scores, axis=-1)
    return weights @ V
```

This code omits masks, batching, multiple heads, and numerical-stability details, but it contains the core mechanism.

## Why multiple heads?

One weighted average can collapse different kinds of relationships. Multi-head attention creates several separately learned query, key, and value projections.

```python
heads = [
    attention(Q @ Wq[i], K @ Wk[i], V @ Wv[i])
    for i in range(num_heads)
]

output = concatenate(heads) @ Wo
```

Different heads can represent different relationship patterns or operate in different learned subspaces. The original model used eight heads in its base and large configurations, with each head working at a reduced dimensionality so total attention cost stayed comparable to full-dimensional single-head attention.

It is useful to say that heads *can* specialize. It is too strong to assume that every head has a clean human-interpretable linguistic role.

## Attention does not know order by itself

If attention receives only a set of token vectors, it has no inherent concept of first, second, or third position.

The original Transformer added sinusoidal positional encodings to token embeddings:

$$
PE_{(pos,2i)} = \sin\left(pos/10000^{2i/d_{\text{model}}}\right)
$$

$$
PE_{(pos,2i+1)} = \cos\left(pos/10000^{2i/d_{\text{model}}}\right)
$$

The exact sinusoidal scheme was not the enduring requirement. The enduring requirement was to provide position information somehow. Later models introduced learned positions and relative or rotary position methods.

## The encoder–decoder architecture the Transformer inherited

The Transformer did not invent the idea of dividing translation into an **encoder** and a **decoder**. It inherited that overall shape from earlier sequence-to-sequence translation systems built with recurrent neural networks, especially LSTMs.

Understanding that inheritance helps separate two ideas that are easy to blend together:

- **Encoder–decoder** describes how the translation problem is divided into a source-reading stage and a target-writing stage.
- **Recurrence, convolution, and attention** describe mechanisms the stages can use to process and exchange information.

The Transformer replaced the recurrent mechanism. It did not discard the useful division between reading the source and generating the translation.

### Why translation needed more than one ordinary RNN

A basic language-model RNN predicts the next token from the tokens that came before it. Translation is harder to fit into that shape:

- the input and output are in different languages;
- their lengths may differ;
- corresponding words may appear in different orders;
- the system normally needs the whole source sentence before deciding how to begin the target sentence.

The early sequence-to-sequence solution was to connect two recurrent networks:

1. An **encoder RNN** reads the source tokens in order and updates its hidden state after each token.
2. The encoder's final state acts as a representation of the source sentence.
3. A **decoder RNN** begins from that representation and predicts the target sentence one token at a time.
4. Each decoder prediction also depends on the target tokens produced previously, until it emits an end-of-sequence token.

In the influential LSTM system of [Sutskever, Vinyals, and Le (2014)](https://arxiv.org/abs/1409.3215), the encoder mapped a variable-length source sequence into a fixed-dimensional vector and a second LSTM decoded a variable-length target sequence from it. The model learned the conditional probability

$$
p(y_1,\ldots,y_m \mid x_1,\ldots,x_n)
=
\prod_{t=1}^{m} p(y_t \mid y_{<t},\,x_1,\ldots,x_n)
$$

This factorization says: generate each target token using the source sentence and the target prefix already generated.

An LSTM is a gated kind of RNN, not an alternative to recurrence. Its gates and memory state made long-range information easier to preserve than in a simple recurrent cell. The overall computation was still sequential: the encoder had to process source position $t-1$ before position $t$, and the decoder had to produce target token $t-1$ before token $t$.

### The fixed-vector bottleneck and the arrival of attention

The simplest encoder–decoder asked one fixed-size vector to carry everything the decoder might need about the source sentence. That becomes an information bottleneck, especially for long or complicated inputs.

[Bahdanau, Cho, and Bengio (2014)](https://arxiv.org/abs/1409.0473) relaxed this bottleneck with attention. Instead of relying only on one final encoder state, their decoder could examine all encoder states at every output step and form a new weighted context:

$$
c_t = \sum_i \alpha_{t,i}h_i
$$

Here, $h_i$ is the encoder representation at source position $i$, and $\alpha_{t,i}$ expresses how relevant that position is while producing target token $t$.

For a rough English-to-French example, the decoder might focus strongly on the source noun while generating its French counterpart, then shift attention toward a source verb while generating the translated verb. The source and target do not need to line up position by position.

That produced the immediate ancestor of the Transformer:

| Stage | Recurrent attention model | Original Transformer |
| --- | --- | --- |
| Read the source | RNN/LSTM encoder states | Encoder self-attention |
| Read the target prefix | RNN/LSTM decoder state | Masked decoder self-attention |
| Connect source to target | Attention over encoder states | Encoder–decoder cross-attention |
| Represent position | Recurrence supplies order implicitly | Positional encodings supply order explicitly |

### Why the Transformer retained the division

The paper opens by describing the strongest prior sequence-transduction systems as recurrent or convolutional networks in an encoder–decoder configuration, often connected by attention. Its central proposal was narrower and more radical than “discard the whole translation architecture”: **remove recurrence and convolution from that architecture and use attention instead.**

Keeping the encoder and decoder was a natural fit for machine translation:

- **They have different information-access rules.** The encoder can examine the entire completed source sentence. The decoder must be causally masked so it cannot inspect future target words during training.
- **They perform different jobs.** The encoder builds contextual source representations; the decoder turns those representations into a target-language sequence.
- **Cross-attention preserves flexible alignment.** At every output position, the decoder can retrieve whichever source representations are currently relevant.
- **The source can be encoded once.** During autoregressive generation, the decoder repeatedly consults stable encoder outputs; the source-side computation does not need to be repeated for each longer target prefix.
- **Input and output remain independently sized.** The architecture naturally handles different sequence lengths and word orders.

This rationale is partly an architectural inference from the translation task and the system the paper describes, rather than a separate experiment proving that an encoder and decoder were universally necessary. Later work made that limitation clear: encoder-only models became useful for representation tasks, while decoder-only models became dominant for general-purpose generative LLMs.

The clean historical takeaway is:

> Earlier work established **encode the source, then decode the target** and added attention between the two. The Transformer kept that problem decomposition but rebuilt all three interactions—source-to-source, target-to-target, and target-to-source—around attention.

## The complete original Transformer

The 2017 system was an encoder-decoder model for translation.

### Encoder

The encoder reads the source sentence. Each layer contains:

- multi-head self-attention;
- a position-wise feed-forward network;
- residual connections;
- layer normalization.

### Decoder

The decoder generates the target sentence one token at a time. Each layer contains:

- masked self-attention over the generated prefix;
- attention over encoder outputs;
- a position-wise feed-forward network;
- residual connections and layer normalization.

The causal mask prevents a decoder position from attending to future output tokens.

This is important because “Transformer” does not mean one exact modern LLM design:

| Architecture | Typical role | Examples of descendants |
| --- | --- | --- |
| Encoder-only | Represent or classify input | BERT-like models |
| Decoder-only | Generate continuations autoregressively | GPT-like LLMs |
| Encoder-decoder | Transform one sequence into another | T5-like models |

The paper introduced the common architectural family; contemporary chat models are usually not literal copies of its six-layer translation model.

## Why training became more scalable

For a self-attention layer, every token can compute its attention relationships in parallel. The paper contrasted this with the $O(n)$ sequential operations of a recurrent layer.

The tradeoff is that full self-attention compares all pairs of positions:

$$
O(n^2d)
$$

where $n$ is sequence length and $d$ is representation width. This quadratic sequence-length cost became increasingly important as context windows grew. Much later research has focused on more efficient attention, sparse patterns, recurrence-like memory, compression, and optimized kernels.

So the Transformer removed one scaling barrier and exposed another:

- **Benefit:** highly parallel training and short paths between positions.
- **Cost:** attention memory and computation grow quickly with sequence length.

## What the experiments established

On the WMT 2014 translation tasks, the paper reported:

- 28.4 BLEU for English-to-German, more than two BLEU above the prior best results including ensembles;
- 41.8 BLEU for English-to-French in the abstract;
- substantially more parallelizable training than recurrent alternatives.

The large models were trained for 3.5 days on eight NVIDIA P100 GPUs. Those numbers are historically interesting, but the lasting result was architectural: attention-only sequence models were not merely viable; they were highly competitive and efficient to train ([Vaswani et al., 2017](https://arxiv.org/html/1706.03762v7)).

## What the paper did not invent

The paper did not invent:

- attention itself;
- neural language models;
- encoder-decoder sequence modeling;
- embeddings;
- residual connections;
- layer normalization;
- autoregressive generation.

Its importance comes from combining and extending existing ideas into an architecture centered on self-attention, particularly scaled dot-product attention and multi-head attention, while eliminating sequence-aligned recurrence and convolution.

## What it did not explain about today's assistants

The paper is necessary background, but it does not explain most end-user assistant behavior:

- instruction tuning;
- reinforcement learning from human feedback;
- system and user message hierarchies;
- tool calling;
- RAG;
- long-term memory;
- agent loops;
- multimodal input;
- modern long-context position schemes.

Those are later model-training or application-runtime developments. A chatbot built from a Transformer is analogous to an application built on a CPU architecture: the foundation matters, but it does not by itself describe the whole software system.

## Practical implications for application developers

### Context is processed, not “understood once”

Each generation step is conditioned on the available token context. Application behavior therefore depends on which instructions, examples, records, and retrieved passages are presented.

### Context length has real costs

Although inference systems use caching and many architectural optimizations, long prompts still affect latency, memory, and cost. Context engineering remains an application concern.

### Attention is not guaranteed retrieval

A model having room for a fact in its context does not guarantee robust use of that fact. Later empirical work found position-sensitive performance in long contexts ([Liu et al., 2024](https://aclanthology.org/2024.tacl-1.9/)). An attention mechanism offers a way to relate positions; it does not guarantee perfect information access.

### The architecture is only one layer of the stack

The Transformer explains the neural computation. The request lifecycle explains how product instructions, tools, external data, and state are assembled around it. See [The LLM Request Lifecycle](llm-request-lifecycle.md).

## What aged well

- Attention-centered architectures became dominant in language modeling.
- Parallel training proved central to scaling.
- Multi-head attention remained a standard component.
- Causal masking remained fundamental to decoder-only language models.
- The tension between global information flow and sequence-length cost became even more important.

## What changed

- Modern LLMs can be hundreds or thousands of times larger than the paper's 65M-parameter base model.
- Decoder-only architectures became especially prominent for general-purpose generation.
- Position encoding, normalization placement, activation functions, attention implementations, and training objectives evolved.
- Tool use, retrieval, alignment, and instruction following moved much of the practical intelligence into training procedures and application harnesses around the core model.

## Reading advice

For a first reading, prioritize:

1. Introduction.
2. Section 3.2 on attention.
3. Section 3.5 on positional encoding.
4. Section 4 on why self-attention.
5. The architecture figure.

The exact optimizer schedule and translation-specific setup matter historically, but are less important for understanding why the paper changed LLM development.

## Primary and supporting sources

- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762); [full HTML](https://arxiv.org/html/1706.03762v7).
- Sutskever, Vinyals, and Le (2014), [*Sequence to Sequence Learning with Neural Networks*](https://arxiv.org/abs/1409.3215), for the fixed-vector LSTM encoder–decoder approach.
- Bahdanau, Cho, and Bengio (2014), [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473), for attention connecting a recurrent encoder and decoder.
- Liu et al. (2024), [*Lost in the Middle: How Language Models Use Long Contexts*](https://aclanthology.org/2024.tacl-1.9/), for a later empirical limitation relevant to long-context use.
