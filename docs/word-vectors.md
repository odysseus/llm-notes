---
title: Word Vectors
type: concept
status: active
updated: 2026-08-05
tags: [embeddings, word2vec, glove, distributional-semantics, nlp-foundations]
---

# Word Vectors: Word2Vec, GloVe, and the Geometry of Meaning

**Central idea:** A word vector represents a word as a learned point in a continuous space, so words used in similar linguistic environments acquire similar—and sometimes systematically related—representations.

**Why it matters:** Word vectors replaced isolated symbolic word IDs with reusable learned features. That shift made semantic similarity computationally useful, helped neural NLP generalize across related words, and established the embedding layer that remains part of modern language models.

## Background topics

- **Vectors and vector spaces:** Coordinates, dimensions, dot products, length, and direction.
- **Probability and conditional probability:** Especially the idea that one word changes the probability of seeing another.
- **Linear algebra:** Matrix multiplication, low-rank approximations, and geometric similarity.
- **Language modeling:** Predicting a word from its context or predicting context from a word.
- **Gradient-based learning:** Adjusting numerical parameters to reduce prediction error.
- **Tokenization and vocabulary construction:** Deciding which units receive learned representations.
- **Bag-of-words and n-gram models:** The main statistical representations that word vectors partly displaced or extended.
- **Dimensionality reduction:** Compressing large, sparse descriptions into smaller, dense ones.

These topics explain the mechanics beneath word embeddings, but none is required in full before the main intuition becomes useful.

## Before learned dense word vectors

Early natural-language systems often treated words as symbols whose identities were known but whose relationships were not. A system could record that `cat` and `dog` were different vocabulary entries, yet that representation alone did not say that both were animals, that both commonly appear near words such as *pet* and *feed*, or that either was more similar to *wolf* than to *democracy*.

The simplest numerical version of this approach is a **one-hot vector**. If a vocabulary contains 50,000 words, each word receives a 50,000-dimensional vector with one coordinate equal to 1 and every other coordinate equal to 0. The vector identifies the word perfectly, but every pair of distinct words is equally dissimilar. The system receives no useful notion of semantic proximity from the representation itself.

Statistical NLP added information through counts. Bag-of-words models represented documents by the words they contained. N-gram language models estimated the probability of a word from a short preceding sequence. Distributional methods built large word-context matrices in which a row described how often a word appeared near thousands of possible context words. Weighting schemes such as pointwise mutual information could emphasize unexpectedly strong associations, and dimensionality-reduction methods could compress the resulting matrices.

These approaches were important predecessors, not failed detours. They already embodied a powerful idea: a word can be characterized by the company it keeps. The difficulty was practical and representational. Raw count vectors were usually enormous and sparse. Similarity depended heavily on choices about context windows and weighting. N-gram models generalized poorly to word sequences they had never observed. Manually designed lexicons and ontologies could encode richer relationships, but were expensive to build, difficult to maintain, and incomplete outside the domains for which they were written.

Before effective learned embeddings, several capabilities were possible only in restricted or labor-intensive forms:

- assigning related words similar reusable features without manually listing their relationships;
- finding semantic neighbors across vocabularies containing millions of items;
- transferring what a model learned about one word to other words used in similar ways;
- discovering broad syntactic or semantic patterns directly from unlabeled text;
- using the same compact representation as input to many different neural models.

The key change was therefore not that computers first became able to store words numerically. It was that the numbers themselves began to encode useful regularities learned from language data.

## The topic in one view

A word embedding system begins with a corpus and a definition of **context**. Context might mean the nearby words within a window, the words in the same sentence, or words connected by a grammatical relationship. Training then adjusts a vector for each vocabulary item so that the vectors are useful for reconstructing or predicting those contextual patterns.

Two families became especially influential:

- **Word2Vec** learns vectors through small prediction tasks. Its continuous bag-of-words model predicts a target word from surrounding words; its skip-gram model predicts surrounding words from a target word.
- **GloVe** first summarizes a corpus as global word-word co-occurrence counts, then learns vectors whose dot products reproduce meaningful relationships in those counts.

The training procedures differ, but both compress distributional evidence into a dense geometric space. Similar words often become nearby points. Some recurring relationships appear as approximately similar directions. The resulting vector can be passed into another model as a compact set of learned features.

Static word vectors also have a central limitation: each word type normally receives one vector regardless of context. The word *bank* therefore has the same representation in *river bank* and *bank loan*. Contextual representation methods later addressed this by generating a different representation for each occurrence of a token. Modern LLMs retain learned embedding tables at their input, but transform those initial vectors through many context-sensitive layers.

## The conceptual leap: identity becomes geometry

Suppose a vocabulary contains the words *cat*, *dog*, *wolf*, *keyboard*, and *piano*. A symbolic representation treats those five entries as unrelated labels. A useful embedding might place *cat*, *dog*, and *wolf* in one region; *keyboard* and *piano* may share another relationship because both are instruments with keys, even though *keyboard* also appears in computing contexts.

The coordinates are learned rather than assigned human-readable meanings. A 300-dimensional vector is not normally organized so that coordinate 17 means “animal” and coordinate 93 means “plural.” Information is distributed across many coordinates, and each coordinate participates in many patterns. For this reason, word vectors are called **distributed representations**.

This design supports a form of generalization. If a classifier has learned that texts containing words near *excellent* are often positive, it may need less evidence to handle *wonderful* than it would if every word were represented as an unrelated ID. Similar representations let downstream models reuse statistical structure.

The central assumption comes from distributional semantics: words used in similar contexts tend to have related meanings or linguistic functions. Zellig Harris's account of distributional structure provided an early formal statement of how linguistic elements could be studied through their environments ([Harris, 1954](https://doi.org/10.1080/00437956.1954.11659520)). Modern embedding methods operationalize a version of that idea at corpus scale.

The qualification matters. Similar context does not guarantee identical meaning. Antonyms such as *hot* and *cold* can occur in very similar sentences. Related words such as *doctor* and *hospital* may be close without being synonyms. An embedding space captures statistical patterns of use, not a complete dictionary or a logically precise theory of meaning.

## What an embedding vector is

For a vocabulary of size $|V|$ and an embedding width $d$, an embedding table can be written as a matrix:

$$
E \in \mathbb{R}^{|V| \times d}
$$

Each row is the vector for one vocabulary item. If $d=300$, every word is represented by 300 learned real numbers rather than by a sparse vector with one coordinate for every word in the vocabulary.

The individual values are **not inherently normalized to a fixed interval**. Word2Vec and GloVe do not generally require every coordinate to lie between 0 and 1 or between -1 and 1. The values are ordinary learned real numbers. Their scale depends on the objective, optimization process, data, and implementation.

Applications often normalize the completed vectors to unit length when they want to compare directions:

$$
\hat{v} = \frac{v}{\lVert v \rVert}
$$

That operation makes the entire vector have length 1; it does not independently force each coordinate into a universal semantic range. Training-time vectors and released pretrained vectors may therefore look numerically different even when they support similar comparisons.

### Similarity

The most common comparison is **cosine similarity**:

$$
\cos(u,v) = \frac{u \cdot v}{\lVert u \rVert\lVert v \rVert}
$$

It measures the angle between two vectors while ignoring their overall lengths. A value near 1 indicates similar directions, 0 indicates orthogonal directions, and -1 indicates opposite directions. In practice, the interpretation depends on the particular space; cosine similarity is a ranking signal, not a calibrated probability that two words mean the same thing.

The dot product $u \cdot v$ is also important because many embedding objectives train related word-context pairs to have large dot products. Dot products preserve information about both angle and magnitude, whereas cosine similarity removes magnitude.

### Neighborhoods and directions

An embedding can be explored through nearest neighbors. The neighbors of *violin* may include other instruments, while the neighbors of *walked* may reflect both meaning and past-tense grammar. The exact results depend on the corpus and training choices.

One pattern discovered in trained word vectors was mathematical analogies such as:

$$
v_{\text{king}} - v_{\text{man}} + v_{\text{woman}}
\approx v_{\text{queen}}
$$

The important idea is not that the model performs symbolic algebra over dictionary definitions. Repeated relationships in the corpus can produce approximately parallel geometric differences. Mikolov, Yih, and Zweig showed that continuous word spaces could capture several syntactic and semantic regularities ([Mikolov et al., 2013](https://aclanthology.org/N13-1090/)).

Analogies are useful demonstrations, but they should not be treated as universal laws of embedding geometry. Results are sensitive to the search rule, vocabulary, corpus, frequency, and preprocessing. Many relationships do not form one clean direction, and impressive selected examples can overstate average reliability. While these examples are impressive, many relationships do not hold in the same way. The results are selected to be impressive, making them a curiosity rather than broadly useful. 

## Word2Vec

Word2Vec is best understood as a family of efficient training methods rather than as one particular set of vectors. Tomas Mikolov and colleagues introduced two architectures designed to learn high-quality representations from very large corpora at much lower computational cost than earlier neural language models ([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)).

Earlier neural language models had already learned continuous word representations as part of predicting language. Bengio and colleagues used a shared distributed representation for words to address the poor generalization of traditional n-gram models ([Bengio et al., 2003](https://www.jmlr.org/papers/v3/bengio03a.html)). Word2Vec's major contribution was a simpler and highly scalable route to useful standalone embeddings.

### Context windows

Training begins by moving a window across text. In the sentence:

> The small dog chased the red ball.

if *chased* is the center word and the window extends two positions in each direction, the nearby context is *small*, *dog*, *the*, and *red*. Repeating this process across a corpus produces large numbers of word-context observations.

The window defines what “similar use” means. A narrow window tends to emphasize syntactic roles and close functional substitutes. A wider window includes more topical information. Neither is universally better; the context definition shapes the resulting geometry.

### Continuous bag of words

The **continuous bag-of-words** architecture, usually called CBOW, uses the surrounding words to predict the center word:

$$
\{\text{small},\text{dog},\text{the},\text{red}\}
\longrightarrow \text{chased}
$$

The context vectors are combined—classically by summing or averaging—and the model assigns probabilities to candidate target words. The order of words within the local context is discarded in the basic architecture, which explains the phrase *bag of words*.

CBOW pools evidence from several context words for each prediction. It is often computationally efficient and can work well for frequent words. The averaging step also smooths away some information that might distinguish different arrangements of the same words.

### Skip-gram

The **skip-gram** architecture reverses the prediction direction. It uses the center word to predict words that occur nearby:

$$
\text{chased}
\longrightarrow
\{\text{small},\text{dog},\text{the},\text{red}\}
$$

Each center-context pair becomes a training relationship. A word appearing in many informative environments is gradually adjusted so that it scores its observed contexts more highly than implausible contexts.

Skip-gram creates more training examples from each position and often gives rare words more direct learning opportunities than CBOW. That benefit comes with additional training work. The practical comparison depends on corpus size, vocabulary frequency, and implementation.

### Why ordinary full prediction is expensive

If the model computed a normalized probability for every vocabulary word on every training example, a vocabulary of millions of entries would make each update expensive. The original Word2Vec work used hierarchical softmax as one efficient alternative. A follow-up paper introduced subsampling for frequent words and **negative sampling**, a simpler objective that became closely associated with skip-gram ([Mikolov et al., 2013](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality)).

Negative sampling converts a large multiclass prediction into several binary comparisons. For an observed pair $(w,c)$, training should score the real context $c$ highly for word $w$. It also samples several noise contexts $n_1,\ldots,n_k$ and teaches the model to score those pairs lower.

A simplified skip-gram with negative sampling objective for one observed pair is:

$$
\log \sigma(v_c^\top v_w)
+
\sum_{i=1}^{k}
\log \sigma(-v_{n_i}^\top v_w)
$$

where:

- $v_w$ is the center-word vector;
- $v_c$ is the observed context vector;
- $v_{n_i}$ is a sampled noise-context vector;
- the dot product measures compatibility;
- $\sigma$ converts a score into a value between 0 and 1;
- $k$ is the number of negative examples.

The first term rewards the real pair. The remaining terms reward the model for rejecting sampled false pairs. Across many updates, words seen with similar contexts receive similar pressures and tend to acquire related representations.

### Two representations per vocabulary item

The simplified equations reveal a detail that introductory explanations often hide: skip-gram normally learns one matrix for center words and another for context words. A vocabulary item can therefore have an “input” vector and an “output” vector. Many applications use the input vectors as the final word embeddings; some combine the two spaces.

There is no contradiction here. Training needs separate roles for the item making a prediction and the item being predicted. The final downstream representation is a design choice built from those learned parameters.

### Frequent-word subsampling

Very common words such as *the*, *of*, and *and* create enormous numbers of training pairs while often contributing little information about a word's specific meaning. Word2Vec can randomly discard a portion of frequent-token occurrences. This speeds training and prevents highly frequent function words from dominating the learning signal.

Discarding is probabilistic rather than absolute. Common words still matter for syntax, but the model does not need to process every occurrence to learn that information.

## GloVe

GloVe—**Global Vectors for Word Representation**—begins from a different presentation of the same broad evidence. Instead of generating local prediction examples one window at a time, it constructs a global word-context co-occurrence matrix and learns a compact factorization of its structure. Pennington, Socher, and Manning introduced the method in 2014 ([Pennington et al., 2014](https://aclanthology.org/D14-1162/)).

Let $X_{ij}$ be the number of times context word $j$ occurs near word $i$. The conditional probability

$$
P(j\mid i)=\frac{X_{ij}}{\sum_k X_{ik}}
$$

describes how characteristic context $j$ is for word $i$.

The GloVe paper motivates its model through **ratios** of such probabilities. Suppose *ice* and *steam* are compared across several context words:

- *solid* should be much more characteristic of *ice* than of *steam*;
- *gas* should be much more characteristic of *steam* than of *ice*;
- *water* may be associated with both;
- an unrelated word such as *fashion* may be associated with neither.

The ratio $P(k\mid\text{ice})/P(k\mid\text{steam})$ distinguishes the informative cases. It is large for an ice-related context, small for a steam-related context, and closer to 1 when the context does not distinguish the two. GloVe seeks a vector space in which these corpus-level relationships can be expressed through linear operations.

### The GloVe objective

The model learns a word vector $w_i$, a context vector $\tilde{w}_j$, and bias terms so that their combined score approximates the logarithm of the co-occurrence count:

$$
J = \sum_{i,j}
f(X_{ij})
\left(
w_i^\top \tilde{w}_j
+ b_i + \tilde{b}_j
- \log X_{ij}
\right)^2
$$

The pieces are interpretable:

1. $X_{ij}$ records how often the pair occurs.
2. $\log X_{ij}$ compresses the very large range between rare and frequent counts.
3. $w_i^\top\tilde{w}_j$ expresses the pair's relationship through a low-dimensional dot product.
4. The bias terms absorb broad frequency effects.
5. $f(X_{ij})$ reduces the influence of extremely rare and extremely frequent pairs.

Training minimizes the weighted difference between the geometric score and the transformed corpus count. The final embedding is commonly formed from the word vector, the context vector, or their sum. As with Word2Vec, the exact released representation is partly an implementation choice.

### What “global” means

GloVe's name can create a misleading contrast. Word2Vec does not ignore the global corpus: repeated local predictions across the entire corpus clearly reflect global statistics. GloVe makes those statistics explicit by building the co-occurrence matrix first and optimizing directly against its nonzero entries.

The practical distinction is therefore about the training formulation:

- Word2Vec samples predictive events from the corpus.
- GloVe aggregates co-occurrences, then fits vectors to the aggregate structure.

Both still depend on local definitions of context, corpus composition, vocabulary rules, and frequency weighting.

## Word2Vec and GloVe are closer than they first appear

Word2Vec was often described as a neural predictive method and GloVe as a count-based matrix-factorization method. That distinction is useful operationally but too sharp theoretically.

Levy and Goldberg showed that skip-gram with negative sampling implicitly factorizes a word-context matrix whose cells are shifted pointwise mutual information values ([Levy and Goldberg, 2014](https://proceedings.neurips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)). In other words, the prediction process is another route to compressing corpus co-occurrence statistics.

This does not make the methods identical. Their objectives weight observations differently, their optimization procedures behave differently, and training choices can matter as much as the method name. It does reveal the common foundation: both learn low-dimensional structure from how words co-occur with contexts.

| Question | Word2Vec | GloVe |
| --- | --- | --- |
| Basic training view | Predict words or contexts | Reconstruct transformed co-occurrence counts |
| Main data interface | Streamed local windows | Aggregated sparse co-occurrence matrix |
| Influential variants | CBOW; skip-gram; negative sampling; hierarchical softmax | Weighted least-squares factorization |
| Corpus information | Accumulated through sampled prediction events | Made explicit before vector training |
| Typical output | One learned vector per vocabulary item, selected or combined from two roles | One learned vector per vocabulary item, selected or combined from word and context roles |
| Shared principle | Similar distributions of context should produce related vectors | Similar distributions of context should produce related vectors |

For many applications, corpus quality, domain match, context definition, dimension, frequency handling, and downstream evaluation matter more than choosing a winner by algorithm name alone.

## A small worked example

Imagine a corpus containing many sentences like these:

- *The dog chased the ball.*
- *The puppy fetched the ball.*
- *The dog ate its food.*
- *The puppy ate its meal.*
- *The violin played a melody.*

The words *dog* and *puppy* share contexts involving chasing, eating, balls, and meals. Even if they rarely appear beside each other, their context distributions are similar. A distributional method can therefore place them nearby.

The words *food* and *meal* may also become close. *Dog* and *food* are related but not interchangeable: they appear together or participate in the same topic, yet occupy different grammatical and semantic roles. Depending on context-window size and training objective, an embedding may mix **similarity** with broader **relatedness**.

Now add:

- *The dog barked loudly.*
- *The puppy barked softly.*
- *The violin sounded loudly.*

The word *loudly* connects otherwise different regions of the corpus. The model must compress many overlapping relationships into a fixed number of dimensions. No coordinate needs to become a literal “dogness” or “sound” axis. Instead, many coordinates jointly arrange the words so that useful co-occurrence relationships can be recovered.

This example also shows why corpus scale matters. One sentence supplies ambiguous evidence; repeated patterns across millions or billions of tokens make the relationships more stable.

## What the vectors capture

### Semantic similarity

Words with overlapping meanings often occur in substitutable contexts. *Car* and *automobile* may therefore become close. This supports lexical search, clustering, feature transfer, and exploratory analysis.

### Semantic relatedness

Words participating in the same subject can also become close even when they are not synonyms. *Doctor*, *hospital*, and *patient* are related through events and institutions. Applications must decide whether this topical proximity is desirable.

### Syntactic behavior

Words with similar grammatical roles share contexts. Past-tense verbs, plural nouns, adjectives, and country names can form recognizable patterns. A single space can therefore mix semantic and syntactic organization.

### Corpus perspective

An embedding describes the corpus, not language in the abstract. Vectors trained on medical writing, news, historical fiction, and software documentation will organize some words differently. This makes embeddings useful for studying language use, but dangerous to treat as neutral definitions.

## Important design choices

### Corpus

More text is not automatically better. A large general corpus improves coverage, while a smaller domain corpus may represent specialized terminology more accurately. Duplicate, low-quality, or systematically skewed text can distort the geometry.

### Tokenization and vocabulary

Classic Word2Vec and GloVe generally assign one vector to each retained vocabulary item. Rare words may be dropped into an unknown-word category. Capitalization, punctuation, phrase detection, and spelling normalization determine which items exist at all.

### Context definition

Nearby surface words emphasize one kind of distributional evidence. Dependency-based contexts can emphasize functional similarity. Window size changes whether the space is more syntactic, topical, or mixed.

### Vector dimension

A wider vector can preserve more distinctions but increases memory, compute, and the risk of fitting noise. A narrower vector compresses more aggressively. Dimension is a capacity choice, not a direct count of human-interpretable features.

### Frequency treatment

Rare words provide too little evidence; extremely frequent words provide too much low-information evidence. Minimum-count thresholds, subsampling, negative-sampling distributions, and GloVe's weighting function all manage this imbalance.

### Training objective and evaluation

The best embedding is task-dependent. Nearest-neighbor quality, analogy scores, downstream classification, domain retrieval, and human similarity judgments test different properties. Strong intrinsic benchmark performance does not guarantee a downstream application improvement.

## Limitations and failure modes

### One vector must represent every sense

In a static embedding, *bank* combines evidence from finance and rivers. Its neighbors may reflect the more common sense or a blend of both. The model cannot produce a finance-specific representation for one sentence and a geography-specific representation for another.

### Unknown and rare words

Words absent from the training vocabulary have no learned vector. Rare words have noisy vectors because the corpus supplies little evidence. FastText later represented a word as a combination of character n-gram vectors, allowing morphology to be shared and vectors to be constructed for some unseen words ([Bojanowski et al., 2017](https://aclanthology.org/Q17-1010/)).

### Word order is weakly represented

Basic CBOW discards order within the window, and basic skip-gram learns from pairwise proximity. *Dog bites man* and *man bites dog* contain the same words but describe different events. Static word vectors can supply ingredients to a sequence model; they do not themselves represent the sentence's structure.

### Similarity is not equivalence

Antonyms often share contexts. Topically related words may be closer than synonyms. Cosine similarity can reveal a distributional relationship without explaining its type.

### Geometry inherits corpus bias

Embeddings can encode social stereotypes and historical inequalities present in their training text. Bolukbasi and colleagues demonstrated gender stereotypes in widely used news-trained word vectors and proposed geometric debiasing methods ([Bolukbasi et al., 2016](https://arxiv.org/abs/1607.06520)). Later work showed that removing an obvious bias direction can leave recoverable bias structure in the space, so a clean-looking analogy test is not evidence that downstream harms have been removed ([Gonen and Goldberg, 2019](https://arxiv.org/abs/1903.03862)).

### Spaces are not naturally aligned

Two runs can learn geometrically rotated or reflected spaces while preserving similar neighbor relationships. Coordinate 12 in one model does not necessarily correspond to coordinate 12 in another. Comparing embeddings across time periods, languages, or corpora requires an explicit alignment method.

### Vector arithmetic can be overinterpreted

An analogy result is a nearest-neighbor calculation in a statistical space. It is not proof that the model possesses a symbolic concept, understands the relationship, or would use it reliably in another task.

### Static embeddings are not sentence embeddings

Averaging word vectors can provide a useful lightweight baseline, but it removes much of word order, syntax, negation, and context. A system designed for semantic search over passages usually benefits from a model trained to represent sentences or passages, not merely a collection of isolated word vectors.

## Practical implications

Static word vectors remain reasonable when the task values low computational cost, lexical relationships, transparent nearest-neighbor exploration, or compatibility with a simple model. They can be effective features for small classifiers, clustering, terminology exploration, corpus comparison, and systems operating under tight memory or latency constraints.

They are less suitable when meaning depends strongly on sentence context, word sense, compositional structure, or long passages. For a modern retrieval system, a static word-vector average should usually be treated as a baseline rather than as the default semantic representation.

For an LLM application, three different uses of the word *embedding* should be kept separate:

| Embedding | What it represents | Typical role |
| --- | --- | --- |
| Static word embedding | A vocabulary word independent of its sentence | Lexical similarity and lightweight NLP features |
| Contextual token representation | One token occurrence after surrounding tokens have influenced it | Internal representation inside Transformers and other contextual models |
| Sentence or document embedding | A complete span of text mapped to one vector | Retrieval, clustering, deduplication, and semantic search |

These are related technologies, but they are not interchangeable. A vector database used in retrieval-augmented generation normally stores sentence-, passage-, or document-level embeddings produced by a model trained for that purpose. See [Retrieval-Augmented Generation](retrieval-augmented-generation.md) for the application architecture around those vectors.

## How word vectors changed NLP

Word vectors made representation learning a standard part of NLP. Instead of designing every linguistic feature by hand, researchers could pretrain reusable numerical representations on unlabeled text and transfer them into supervised systems. The representation itself became a learned artifact.

This enabled several larger developments:

1. **Pretrained features became normal.** A model could begin with knowledge of broad corpus regularities rather than learning every word relationship from a small task dataset.
2. **Representation quality became separable from task architecture.** Researchers could improve a general embedding and reuse it across tagging, classification, parsing, and other tasks.
3. **Geometric analysis became a tool for language.** Similarity, clustering, analogy structure, and bias could be studied directly in learned spaces.
4. **Embedding layers became foundational neural components.** Modern sequence models still begin by mapping discrete token IDs into continuous vectors.

What static word vectors still lacked was equally important. They did not resolve polysemy, model sentence structure, or create a representation that changed with context. ELMo addressed this directly by deriving each word occurrence's representation from the internal states of a bidirectional language model and reported improvements across six NLP tasks ([Peters et al., 2018](https://aclanthology.org/N18-1202/)). Transformer models then made contextualization deeper and more scalable: token embeddings enter the network, attention mixes information across positions, and each layer constructs representations conditioned on the surrounding sequence.

The path is therefore not:

> Word2Vec was discarded and Transformers replaced vectors.

It is:

> Static word vectors showed that useful linguistic regularities could be learned as geometry; contextual models made that geometry depend on the particular sentence; large language models scaled contextual representation learning and generation together.

Word2Vec and GloVe are not miniature LLMs. They do not follow instructions, generate open-ended text, maintain a context window, or reason over a prompt. Their importance is more fundamental: they helped establish how discrete language could enter neural systems as learned continuous representations.

## Recap

- A word vector is a dense learned representation, not merely a numeric word ID.
- Words become close when training gives them similar distributional evidence.
- The individual coordinates are real-valued parameters and are not inherently constrained to a fixed interval.
- Cosine similarity compares vector direction, but does not identify the type of linguistic relationship.
- CBOW predicts a target from surrounding words; skip-gram predicts surrounding words from a target.
- Negative sampling makes skip-gram training efficient by contrasting observed pairs with sampled noise pairs.
- GloVe learns from an explicitly aggregated co-occurrence matrix and fits vector dot products to transformed counts.
- Skip-gram with negative sampling can also be understood as implicit matrix factorization, so the predictive-versus-count distinction is not absolute.
- Static embeddings are compact and useful, but combine all senses of a word and weakly represent order.
- Contextual models did not abandon embeddings; they transformed initial token vectors into context-dependent representations.

## Reading guidance

For a focused path through the literature:

1. Read the introduction and model overview of [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/v3/bengio03a.html) to see why distributed word representations helped language models generalize.
2. Read [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781) for CBOW, skip-gram, and the scaling motivation.
3. Read [*Distributed Representations of Words and Phrases and their Compositionality*](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality) for negative sampling, frequent-word subsampling, and phrase handling.
4. Read [*GloVe: Global Vectors for Word Representation*](https://aclanthology.org/D14-1162/) for the co-occurrence-ratio motivation and weighted factorization objective.
5. Read [*Neural Word Embedding as Implicit Matrix Factorization*](https://proceedings.neurips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf) to connect predictive embeddings with count-based methods.
6. Continue to [*Enriching Word Vectors with Subword Information*](https://aclanthology.org/Q17-1010/) and [*Deep Contextualized Word Representations*](https://aclanthology.org/N18-1202/) to see how the field addressed unknown words and context dependence.
7. Then read [Paper Guide: *Attention Is All You Need*](paper-attention-is-all-you-need.md) to follow the transition from static embeddings to fully contextual sequence representations.

## Primary and supporting sources

- Harris, Zellig S. (1954), [*Distributional Structure*](https://doi.org/10.1080/00437956.1954.11659520).
- Bengio, Yoshua, Réjean Ducharme, Pascal Vincent, and Christian Jauvin (2003), [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/v3/bengio03a.html).
- Mikolov, Tomas, Kai Chen, Greg Corrado, and Jeffrey Dean (2013), [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781).
- Mikolov, Tomas, Wen-tau Yih, and Geoffrey Zweig (2013), [*Linguistic Regularities in Continuous Space Word Representations*](https://aclanthology.org/N13-1090/).
- Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeffrey Dean (2013), [*Distributed Representations of Words and Phrases and their Compositionality*](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality).
- Pennington, Jeffrey, Richard Socher, and Christopher D. Manning (2014), [*GloVe: Global Vectors for Word Representation*](https://aclanthology.org/D14-1162/).
- Levy, Omer, and Yoav Goldberg (2014), [*Neural Word Embedding as Implicit Matrix Factorization*](https://proceedings.neurips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf).
- Bolukbasi, Tolga, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai (2016), [*Man Is to Computer Programmer as Woman Is to Homemaker? Debiasing Word Embeddings*](https://arxiv.org/abs/1607.06520).
- Bojanowski, Piotr, Edouard Grave, Armand Joulin, and Tomas Mikolov (2017), [*Enriching Word Vectors with Subword Information*](https://aclanthology.org/Q17-1010/).
- Peters, Matthew E., et al. (2018), [*Deep Contextualized Word Representations*](https://aclanthology.org/N18-1202/).
- Gonen, Hila, and Yoav Goldberg (2019), [*Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But Do Not Remove Them*](https://arxiv.org/abs/1903.03862).
