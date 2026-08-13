---
title: "T5 Text-to-Text Transformer"
authors: [Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu]
year: 2020
topics: [T5, transfer learning, text-to-text, pretraining, span corruption, C4, encoder-decoder transformers]
status: active
type: paper-guide
updated: 2026-08-13
---

# Paper Guide: *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*

**Paper:** Colin Raffel et al., 2020  
**Primary link:** [JMLR, volume 21](https://jmlr.org/papers/v21/20-074.html)  
**Common name:** T5, for **Text-to-Text Transfer Transformer**

**Thesis:** A wide range of NLP tasks can use the same encoder–decoder Transformer, training objective, and output interface if every task is represented as text input mapped to text output. A systematic comparison of transfer-learning choices then points to a strong general recipe: pretrain on a large, filtered web corpus using span corruption, fine-tune the complete model for each downstream task, and scale the model and training substantially.

## Background

By 2019, transfer learning had rapidly changed NLP. Instead of training a complete model from scratch for every labeled task, researchers could pretrain a large neural network on abundant unlabeled text and adapt it to smaller supervised datasets.

The field nevertheless lacked one standard recipe. BERT used a bidirectional Transformer encoder with masked-token pretraining, then attached task-specific output heads for classification or span selection. GPT-style models used a causal decoder trained to predict the next token. Sequence-to-sequence systems used separate encoders and decoders for generative tasks such as translation and summarization. Researchers also varied the pretraining data, corruption objective, adaptation method, and amount of supervised multi-task training.

This made results difficult to compare. A new system might change its architecture, training data, objective, parameter count, and fine-tuning procedure simultaneously. If it improved, it was unclear which choice mattered.

Tasks also required different interfaces:

- sentiment analysis returned a class ID;
- extractive question answering selected positions in a passage;
- translation generated a sequence;
- semantic similarity predicted a numeric score;
- summarization generated a shorter document.

Earlier projects had already pursued unification—most notably DecaNLP, which cast diverse tasks as question answering. T5 adopted a broader and simpler abstraction: every task would be a conditional text-generation problem.

The paper is therefore best understood as both a model recipe and a large controlled study of transfer learning. The authors explicitly say that their primary goal was not to claim an entirely new method, but to compare existing choices under a common framework and combine the strongest findings at scale ([Raffel et al., 2020](https://jmlr.org/papers/v21/20-074.html)).

## Core Idea

T5 converts every task into:

$$
\text{input text} \longrightarrow \text{output text}
$$

A textual prefix tells the model which task to perform.

| Task | Input | Target output |
| --- | --- | --- |
| Translation | `translate English to German: That is good.` | `Das ist gut.` |
| Summarization | `summarize: <article>` | `<summary>` |
| Sentiment | `sst2 sentence: The film was wonderful.` | `positive` |
| Question answering | `question: ... context: ...` | `<answer>` |
| Similarity scoring | `stsb sentence1: ... sentence2: ...` | `3.8` |

The same model always learns a conditional distribution:

$$
p_\theta(y \mid x)
$$

where both $x$ and $y$ are token sequences. Classification no longer requires a special classification layer; the model generates a label such as `entailment`. Question answering no longer requires a special span-selection head; the model generates the answer text.

This does not mean that all tasks become conceptually identical. Their data, metrics, and error conditions remain different. The unification occurs at the model interface and learning objective. That gives researchers a consistent testbed and makes one pretrained checkpoint usable across many task types.

The distinction from modern instruction-following is important. A T5 task prefix such as `summarize:` identifies a task the model is subsequently fine-tuned to perform. Original T5 was not primarily trained to follow arbitrary natural-language requests zero-shot. Later instruction-tuning work extended the text-to-text foundation in that direction.

## Method

### A mostly standard encoder–decoder Transformer

T5 uses an encoder–decoder Transformer. The encoder can attend bidirectionally across the complete input. The autoregressive decoder attends to the encoder representation and generates the target from left to right.

Its implementation makes several changes to the original Transformer: layer normalization is placed before each subcomponent, its additive bias is removed, and learned relative-position biases replace sinusoidal position encodings. These details became influential in later T5-family implementations, but the paper does not present the underlying Transformer as a new invention.

The authors compare encoder–decoder, decoder-only language-model, and prefix-language-model configurations. Under their experimental setup, the full encoder–decoder trained with a denoising objective performs best overall. A parameter-shared encoder–decoder is nearly as effective, while reducing encoder and decoder depth hurts more noticeably.

### The C4 pretraining corpus

The authors create the **Colossal Clean Crawled Corpus**, or **C4**, from Common Crawl. Their pipeline extracts English text and applies heuristic cleaning, including deduplication and the removal of pages that appear not to contain natural prose. The released corpus is approximately 750 GB in the paper's accounting.

C4 mattered for two reasons. It supplied enough diverse text that large models could train without repeatedly cycling through a small corpus, and it made an important component of the recipe public. The experiments found that the filtered C4 version consistently outperformed a much larger unfiltered variant, although narrower in-domain corpora could be better for particular downstream tasks.

### Span-corruption pretraining

The best-known T5 pretraining objective corrupts contiguous spans rather than masking isolated tokens. Fifteen percent of the source tokens are selected, with an average corrupted span length of three. Each removed span is replaced in the input by a unique sentinel token. The target contains only the removed spans, separated by the same sentinels.

For example:

```text
Original:
The task should be saved before asking for more details.

Corrupted input:
The task <X> before asking <Y> details.

Target:
<X> should be saved <Y> for more <Z>
```

The decoder learns to reconstruct the missing material. Predicting only corrupted spans produces shorter target sequences than reconstructing the entire input, improving training efficiency.

The paper compares causal language modeling, deshuffling, and several denoising variants. Denoising performs clearly better than language modeling and deshuffling in this setup, but differences among many denoising variants are relatively modest. Span corruption with average length three provides a useful balance of performance and efficiency—not evidence that those exact numbers are universally optimal.

### Systematic comparisons

Using a common baseline, the authors vary:

- Transformer architecture and attention pattern;
- unsupervised pretraining objective;
- corruption rate and span length;
- pretraining corpus and corpus size;
- full fine-tuning, adapters, and gradual unfreezing;
- supervised multi-task training and task-mixture strategies;
- model size, training duration, batch size, and ensembling.

They evaluate on GLUE and SuperGLUE, CNN/Daily Mail summarization, SQuAD question answering, and English-to-German, French, and Romanian translation. This breadth is central to the paper: a choice that helps one benchmark may not be a generally strong transfer-learning strategy.

### Scaling the final recipe

The final family contains models of roughly 60 million, 220 million, 770 million, 3 billion, and 11 billion parameters. The strongest models combine C4, span-corruption pretraining, supervised multi-task pretraining, and task-specific fine-tuning. The largest model required model and data parallelism and represented unusually large-scale training for its time.

## Results

The paper's most durable empirical findings are broader than any one benchmark score.

### Text-to-text worked across task types

One encoder–decoder interface successfully handled classification, similarity prediction, question answering, summarization, and translation. Generating a class name rather than using a dedicated output head did not prevent competitive performance. This validated text-to-text as a practical general-purpose abstraction.

### Architecture and objective both mattered

In the controlled comparisons, the encoder–decoder architecture with denoising pretraining was strongest overall. Explicit encoder–decoder attention helped relative to a comparable prefix language model. Denoising objectives consistently beat causal language modeling for the downstream suite, while fine distinctions among denoising variants mattered less.

These findings are conditional on T5's tasks and compute-matched setup. They do not establish that encoder–decoder models dominate every use case; decoder-only models later became especially successful for open-ended generation and in-context learning.

### Data quality, diversity, and quantity mattered

Heuristic filtering improved performance over unfiltered Common Crawl text. Domain-matched corpora sometimes helped particular tasks, but a large diverse corpus provided better generality. Artificially shrinking and repeatedly recycling the pretraining corpus eventually degraded transfer, consistent with increasing memorization.

### Ordinary full fine-tuning was hard to beat

Adapters and gradual unfreezing reduced the number of updated parameters, but full-model fine-tuning achieved the best overall results in the paper. This was a performance comparison, not a rejection of parameter-efficient methods: storing a separate full checkpoint per task is expensive, and later adapter, prompt-tuning, and low-rank methods improved the trade-off.

### Scale was the decisive final ingredient

The final T5 system reached state-of-the-art results on 18 of the 24 tasks considered. T5-11B performed best across the model-size variants, with the paper describing the increase from 3B to 11B parameters as the most important ingredient in its best results. It reported a 90.3 GLUE average and approached the then-estimated human level on SuperGLUE.

The results were not uniformly dominant. T5 did not set the state of the art on the WMT translation tasks; the authors point to English-only pretraining and the absence of specialized techniques such as backtranslation as likely factors.

## Limitations

### It did not create a general conversational instruction follower

T5 used task prefixes, but downstream tasks still normally required labeled examples and fine-tuning. It should not be confused with later assistants that interpret a wide variety of unseen instructions through prompting alone. FLAN and related work specifically added instruction tuning to improve that capability.

### Text generation removes hard output constraints

A generative classifier can emit a word outside the legal label set. A generated question answer can differ from an exact source span. The unified interface is elegant, but applications may still need constrained decoding, validation, or task-specific evaluation.

### The strongest evidence depended heavily on compute

The best results came from an 11-billion-parameter model and extensive experimental comparisons. This demonstrated scaling, but made the top result difficult and expensive to reproduce. The paper also tested many alternatives on a smaller baseline, so an experimental ranking might change at much larger scales.

### C4 was large, English-centered, and imperfectly documented

C4's filtering improved benchmark performance, but web-scale cleaning is not neutral. A later audit found unexpected source domains, machine-generated text, benchmark examples, and disproportionate removal of text about minority identities by blocklist filters ([Dodge et al., 2021](https://aclanthology.org/2021.emnlp-main.98/)). These findings complicate the idea that a corpus is simply “clean” and emphasize provenance, bias, and contamination analysis.

English-only pretraining also limited the paper's claim to linguistic generality. The follow-up mT5 work expanded the approach to 101 languages ([Xue et al., 2021](https://aclanthology.org/2021.naacl-main.41/)).

### Benchmark breadth is not open-world generality

The suite was broad by the standards of the time, but still consisted of established supervised NLP benchmarks. Success did not establish factual reliability, robust long-form dialogue, tool use, safety, or adaptation to arbitrary real-world requests.

## Impact

T5 made text-to-text modeling a standard design pattern. Its importance lies less in one novel layer than in showing how a uniform interface simplifies transfer learning across heterogeneous tasks. That abstraction influenced later encoder–decoder models, dataset frameworks, and instruction-tuning pipelines.

The span-corruption objective became a widely reused pretraining recipe. C4 became one of the major public web corpora, while criticism of C4 helped motivate better documentation and auditing of large training datasets.

The T5 family also became a platform for follow-up work:

- **mT5** extended the recipe to many languages.
- **ByT5** replaced subword tokens with byte-level input and output.
- **UL2** mixed several denoising modes to support a wider range of downstream settings.
- **FLAN-T5** instruction-tuned T5 checkpoints on large mixtures of prompted tasks, moving the family closer to general instruction following ([Chung et al., 2022](https://arxiv.org/abs/2210.11416)).

For current LLM design, T5 leaves three especially useful lessons:

1. A common input/output interface can simplify training and transfer without making the underlying tasks identical.
2. Architecture, objective, data, adaptation method, and scale should be separated experimentally when possible.
3. A straightforward recipe applied to high-quality data at sufficient scale can outperform a more ornate method—although the cost and dataset assumptions remain part of the result.

## Further Reading

### Primary and follow-up papers

- Raffel et al. (2020), [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://jmlr.org/papers/v21/20-074.html).
- Xue et al. (2021), [*mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*](https://aclanthology.org/2021.naacl-main.41/).
- Dodge et al. (2021), [*Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus*](https://aclanthology.org/2021.emnlp-main.98/).
- Wei et al. (2021), [*Finetuned Language Models Are Zero-Shot Learners*](https://arxiv.org/abs/2109.01652).
- Tay et al. (2022), [*UL2: Unifying Language Learning Paradigms*](https://arxiv.org/abs/2205.05131).
- Chung et al. (2022), [*Scaling Instruction-Finetuned Language Models*](https://arxiv.org/abs/2210.11416).

### Related library entries

- [Attention Is All You Need](paper-attention-is-all-you-need.md)
- [Instruction Tuning](instruction-tuning.md)
- [Task Fine-Tuning and Task Specialization](task-specialization.md)
- [Prompt Fine-Tuning and Soft Prompting](soft-prompting.md)
- [In-Context Learning](in-context-learning.md)
- [Structured Output](structured-output.md)
