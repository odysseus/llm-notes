# From Symbols to Geometry: Word Vectors, word2vec, and GloVe

**Topic:** Static word representations and the development of scalable dense word embeddings  
**Scope:** Distributional word vectors, word2vec (CBOW and Skip-gram), negative sampling, GloVe, their relationship, limitations, and their role in the path to modern language models  
**Prepared:** August 2026

---

## 1. Relevant background topics

- **Vectors, dot products, and cosine similarity.** These provide the basic language for representing words as points and measuring their geometric similarity.
- **Probability and conditional probability.** Word-vector methods learn from how often words occur and co-occur, either explicitly as probabilities or implicitly through prediction objectives.
- **The distributional hypothesis.** The idea that linguistic items appearing in similar contexts tend to have related functions or meanings is the conceptual basis of most corpus-derived word representations.[^harris1954]
- **Sparse versus dense representations.** This distinction explains why one-hot vectors and raw count tables behave differently from compact learned embeddings.
- **n-gram language models and smoothing.** These describe the dominant statistical language-modeling approach that neural distributed representations initially sought to improve or replace.
- **Matrix factorization and singular value decomposition.** Earlier methods such as latent semantic analysis created lower-dimensional semantic spaces by factorizing large count matrices.[^deerwester1990]
- **Neural networks, embeddings, and gradient descent.** word2vec learns vectors as trainable parameters optimized for a prediction task.
- **Softmax, logistic regression, and sampling-based approximations.** These are needed to understand why full-vocabulary prediction is expensive and why hierarchical softmax and negative sampling matter.
- **Tokenization and vocabulary construction.** Static word embeddings assign vectors to vocabulary entries, so segmentation choices, frequency thresholds, and out-of-vocabulary handling directly determine what the model can represent.
- **Intrinsic versus downstream evaluation.** Nearest-neighbor, similarity, and analogy tests measure properties of the vector space, but their results do not always predict performance in real NLP applications.[^faruqui2016]

---

## 2. NLP before scalable dense word embeddings

Before word2vec and GloVe, mainstream NLP systems commonly represented words as discrete symbols. A word might be stored as an integer vocabulary index, a one-hot vector, a sparse feature, or an entry in a hand-built lexicon. In a one-hot representation, *cat* and *dog* are exactly as different from one another as *cat* and *thermodynamics*: every distinct word occupies an independent axis. The representation says that the words are different, but it says nothing about *how* they are related.

Statistical NLP compensated for this lack of shared structure through counts, smoothing, clustering, linguistic annotation, and feature engineering. n-gram language models estimated the probability of a word from a short preceding context. Taggers and parsers relied on manually designed features such as the current token, neighboring tokens, prefixes, suffixes, capitalization patterns, part-of-speech tags, and lexicon membership. Class-based models grouped words to share statistics, and count-based vector-space methods represented words or documents using their observed distributions.[^brown1992] Latent semantic analysis showed that matrix factorization could uncover lower-dimensional structure from term-document data, but these methods were often expensive, task-specific, sensitive to weighting decisions, or awkward to incorporate into end-to-end neural systems.[^deerwester1990]

The main problem was **generalization across discrete symbols**. Suppose a supervised named-entity recognizer had learned useful behavior around the word *France* but had seen very few examples involving *Portugal*. A sparse lexical model had little basis for transferring what it knew from one word to the other unless a human designed a shared feature, a clustering algorithm placed them together, or a separate lexical resource stated their relationship. Likewise, a language model that had observed “the dog is running” did not automatically gain much confidence in “the cat is walking” merely because *dog* resembles *cat* and *running* resembles *walking*. Bengio and colleagues framed this as a manifestation of the curse of dimensionality: a model should generalize to unseen sequences through similarity among learned distributed word representations rather than through exact reuse of short observed strings.[^bengio2003]

Several capabilities were therefore effectively impractical at broad scale. It was difficult to pretrain a compact representation on billions of words and then reuse it as a standard input feature across unrelated NLP tasks. It was difficult to retrieve semantically related words with a fast geometric operation over a small dense vector. It was unusual for a representation learned without explicit linguistic labels to encode many syntactic and semantic relationships as approximately consistent directions. Cross-lingual dictionary induction using a simple learned linear map between monolingual spaces was not a standard, broadly effective technique. These ideas were not logically impossible—earlier distributional and neural models anticipated many of them—but they were not yet cheap, scalable, simple, and reliable enough to become default infrastructure.

The change produced by word2vec was therefore not the invention of the idea that words could be vectors. Its contribution was to strip the learning problem down to shallow objectives that could be trained efficiently on enormous corpora. GloVe then offered a complementary formulation that trained directly on aggregated global co-occurrence statistics. Together, these methods made dense pretrained word vectors ordinary components of NLP systems.

---

## 3. Executive summary

A **word vector** is a numerical representation of a word. In distributional semantics, the vector is learned from the contexts in which the word appears. Words with similar contextual behavior tend to receive vectors that are close under measures such as cosine similarity. Dense word vectors compress a very large set of co-occurrence relationships into tens or hundreds of learned dimensions.

**word2vec** is a family of shallow prediction models introduced by Mikolov and colleagues in 2013. Its two principal architectures are **Continuous Bag of Words (CBOW)**, which predicts a center word from nearby context words, and **Skip-gram**, which predicts nearby context words from a center word.[^mikolov2013efficient] The models became especially influential because computational shortcuts—hierarchical softmax, negative sampling, and subsampling of frequent words—made it practical to train high-quality vectors from billions of tokens.[^mikolov2013phrases]

**GloVe**, introduced by Pennington, Socher, and Manning in 2014, learns vectors from a global word-word co-occurrence matrix. It optimizes a weighted least-squares objective so that a word vector’s dot product with a context vector approximates the logarithm of their co-occurrence count.[^pennington2014] GloVe was presented as a bridge between global matrix-factorization methods and local context-window prediction methods.

Although their training procedures look different, word2vec with Skip-gram Negative Sampling (SGNS) and count-based matrix methods are closely related. Levy and Goldberg showed that SGNS implicitly factorizes a word-context matrix whose entries are shifted pointwise mutual information values.[^levy2014] This result is important: the apparent divide between “neural prediction” and “count-based factorization” is partly a difference in optimization, weighting, sampling, and parameterization rather than a difference in the underlying corpus information.

Static word embeddings transformed NLP by making transfer from unlabeled text straightforward. They improved sequence tagging, parsing, classification, information retrieval, machine translation, and many other tasks. However, a static embedding assigns one vector to a word type regardless of context. Thus *bank* has the same vector in “river bank” and “investment bank.” Static vectors also inherit social and historical biases from their training corpora, struggle with rare and unseen words, discard word order inside their training window, and cannot by themselves represent sentences or perform general language reasoning. These shortcomings led directly to subword embeddings, contextual embeddings such as ELMo, and Transformer-based pretrained models such as BERT. Modern LLMs still contain learned embedding layers, but they transform token embeddings through many context-sensitive layers rather than treating the initial vector as the final representation.

---

## 4. Full report

### 4.1 Terminology: word vectors, embeddings, and representations

The terms are often used loosely, so several distinctions are useful.

A **word vector** is any vector associated with a word. It could be:

- a one-hot identity vector;
- a row of raw co-occurrence counts;
- a TF-IDF- or PMI-weighted context vector;
- a low-rank vector obtained through singular value decomposition;
- a learned neural-network parameter;
- or a contextual vector produced for one occurrence of a word inside a sentence.

A **word embedding** usually means a learned, dense, lower-dimensional vector. “Dense” means that many dimensions may contain nonzero real values. “Lower-dimensional” means that the vector dimension, perhaps 100–1,000, is far smaller than the vocabulary size or the number of possible contexts.

A **static embedding** assigns one stored vector to each vocabulary item. word2vec and GloVe are static embedding methods. A **contextual representation** computes a different vector for each token occurrence as a function of the surrounding text. ELMo, BERT, and modern LLM hidden states are contextual.

Finally, **word2vec** can refer to both the software released by Google and the CBOW/Skip-gram model family. It is not one single objective. Skip-gram may be trained with hierarchical softmax, negative sampling, or another approximation; CBOW may use similar output objectives.

### 4.2 The distributional idea

The foundation is the observation that linguistic behavior is reflected in distribution. Harris’s 1954 account described language in terms of the environments in which its elements occur.[^harris1954] In modern computational terms, we operationalize this by collecting contexts for every word.

Consider these sentences:

> The **cat** slept on the sofa.  
> The **dog** slept on the rug.  
> The **cat** chased a mouse.  
> The **dog** chased a ball.

The words *cat* and *dog* occur near similar verbs and syntactic frames. A distributional model can therefore infer a relationship even if no dictionary states that both are animals.

The meaning of “context” is a modeling choice. It can be:

- neighboring words within a fixed window;
- preceding words only;
- syntactic dependencies;
- an entire document;
- subword units;
- or representations computed by another model.

Different context definitions produce different notions of similarity. A narrow window often emphasizes syntactic or functional similarity: *eat* may be near *consume*. A broad window often emphasizes topic or association: *doctor* may be near *hospital*. The vector space does not discover a single philosophically complete definition of meaning; it summarizes the regularities made visible by the selected corpus, context definition, objective, and weighting scheme.

### 4.3 From one-hot symbols to distributional vectors

Assume a vocabulary of size $V$. A one-hot representation for word $w_i$ is

$$
\mathbf{e}_i \in \mathbb{R}^{V},
$$

where the $i$-th coordinate is 1 and every other coordinate is 0. For two different words $i \neq j$,

$$
\mathbf{e}_i^\top \mathbf{e}_j = 0.
$$

Every distinct pair is orthogonal. One-hot vectors are useful as identifiers and as inputs to an embedding lookup, but their geometry contains no learned lexical similarity.

A count-based distributional representation instead creates a word-context matrix $X$:

$$
X_{ij} = \text{number of times context } c_j \text{ occurs near word } w_i.
$$

The row $X_{i:}$ is a high-dimensional sparse vector describing the contexts of $w_i$. Similarity is often measured by cosine similarity:

$$
\cos(\mathbf{x},\mathbf{y}) =
\frac{\mathbf{x}^\top \mathbf{y}}{\lVert \mathbf{x}\rVert_2\lVert \mathbf{y}\rVert_2}.
$$

Raw counts overemphasize common words. Pointwise mutual information (PMI) instead compares observed co-occurrence with what would be expected under independence:

$$
\operatorname{PMI}(w,c)
= \log \frac{P(w,c)}{P(w)P(c)}.
$$

Positive PMI commonly clips negative values:

$$
\operatorname{PPMI}(w,c) = \max(\operatorname{PMI}(w,c), 0).
$$

A count matrix may then be factorized. If

$$
M \approx U_k\Sigma_k V_k^\top,
$$

then a compact word representation can be derived from $U_k$, $U_k\Sigma_k$, or a related scaling. Latent semantic analysis used this general strategy on term-document matrices, demonstrating that lower-rank structure could improve retrieval by representing latent associations rather than exact lexical matches.[^deerwester1990]

This family of methods already contains the essential idea of semantic geometry. The later breakthrough was largely about learning useful low-dimensional geometry more directly and efficiently from large streams of local contexts.

### 4.4 Neural distributed representations before word2vec

Bengio et al.’s neural probabilistic language model jointly learned a vector for each word and a neural function predicting the next word from a fixed-length preceding context.[^bengio2003] If $C(w)$ denotes the embedding of word $w$, the model conceptually learns

$$
P(w_t \mid w_{t-1},\ldots,w_{t-n+1})
= g(C(w_{t-1}),\ldots,C(w_{t-n+1})).
$$

This creates parameter sharing. Similar words can acquire similar embeddings because substituting one for another produces similar useful predictions. The approach offered a principled answer to sparse n-gram generalization, but computing and normalizing probabilities over a large vocabulary was expensive.

Subsequent work showed that pretrained unsupervised representations could improve supervised NLP systems when supplied as additional features.[^turian2010] Collobert and colleagues demonstrated a unified neural architecture for several NLP tasks that learned internal word representations instead of relying primarily on task-specific feature engineering.[^collobert2011] These papers established much of the neural representation-learning program before word2vec.

word2vec’s innovation was radical simplification. Instead of training a deep or nonlinear language model and receiving word vectors as a useful byproduct, it trained shallow models whose main purpose was to produce the vectors.

### 4.5 word2vec

#### 4.5.1 CBOW and Skip-gram

The original word2vec paper introduced two log-linear architectures designed to reduce computational complexity.[^mikolov2013efficient]

**Continuous Bag of Words (CBOW)** predicts the center word from surrounding words. For a context window $C_t$ around position $t$, a simple form averages context embeddings:

$$
\mathbf{h}_t = \frac{1}{|C_t|}\sum_{j\in C_t}\mathbf{v}_{w_j}.
$$

The model then predicts $w_t$ from $\mathbf{h}_t$. Because the context vectors are averaged or summed, the local word order is discarded—hence “bag of words.”

**Skip-gram** reverses the prediction direction. Given a center word $w_t$, it predicts each context word $w_{t+j}$ within a window of radius $c$. The basic objective is

$$
\max_\theta \frac{1}{T}\sum_{t=1}^{T}
\sum_{-c\le j\le c,\ j\ne 0}
\log P(w_{t+j}\mid w_t).
$$

With full softmax,

$$
P(w_o\mid w_i)
=
\frac{\exp({\mathbf{v}'_{w_o}}^\top\mathbf{v}_{w_i})}{\sum_{w=1}^{V}\exp({\mathbf{v}'_w}^\top\mathbf{v}_{w_i})}.
$$

The model maintains two parameter sets:

- an **input** vector $\mathbf{v}_w$ used when a word is the center/input;
- an **output or context** vector $\mathbf{v}'_w$ used when a word is a predicted context.

The denominator sums over the entire vocabulary. With hundreds of thousands or millions of words, this makes exact training expensive.

#### 4.5.2 Hierarchical softmax

Hierarchical softmax replaces the flat vocabulary classifier with a binary tree whose leaves are words. Predicting a word becomes predicting a sequence of left/right decisions along the root-to-leaf path. If the tree is balanced, the number of decisions falls from $O(V)$ to roughly $O(\log V)$. Mikolov et al. used a Huffman tree so frequent words receive shorter codes.[^mikolov2013phrases]

Hierarchical softmax remains a normalized probabilistic model: it defines a probability distribution over the vocabulary. Its performance depends partly on the tree structure.

#### 4.5.3 Negative sampling

Negative sampling changes the task. Instead of calculating a normalized probability over all words, the model learns to distinguish observed word-context pairs from randomly generated pairs.

For an observed pair $(w,c)$, Skip-gram Negative Sampling maximizes

$$
\log \sigma(\mathbf{v}'_c{}^\top\mathbf{v}_w)
+
\sum_{i=1}^{k}
\mathbb{E}_{n_i\sim P_n}
\left[
\log \sigma(-\mathbf{v}'_{n_i}{}^\top\mathbf{v}_w)
\right],
$$

where

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

and $k$ is the number of negative samples. A positive pair should have a large dot product; sampled noise pairs should have small or negative dot products.

The original experiments found that sampling words from a unigram distribution raised to the $3/4$ power worked better than either the raw unigram or uniform distribution.[^mikolov2013phrases] This smooths the frequency distribution: common words remain likely negatives, but not in direct proportion to their overwhelming corpus frequency.

A compact PyTorch-like training fragment makes the objective concrete:

```python
# center_ids:    [batch]
# context_ids:   [batch]
# negative_ids:  [batch, k]

center = input_embeddings(center_ids)              # [batch, dim]
positive = output_embeddings(context_ids)          # [batch, dim]
negative = output_embeddings(negative_ids)         # [batch, k, dim]

positive_score = (center * positive).sum(dim=-1)   # [batch]
negative_score = (negative * center[:, None, :]).sum(dim=-1)

loss = -(
    log_sigmoid(positive_score)
    + log_sigmoid(-negative_score).sum(dim=-1)
).mean()
```

This is not a full implementation: a production trainer also constructs dynamic context windows, samples negatives efficiently, handles batching, applies subsampling, and updates sparse rows. But the code exposes the central mechanism. Training moves observed center-context vectors together and sampled non-context vectors apart.

#### 4.5.4 Subsampling frequent words

Extremely common words produce enormous numbers of low-information training pairs. Observing *France* near *the* is less discriminative than observing *France* near *Paris*. word2vec therefore probabilistically discards frequent tokens. This both accelerates training and can improve representations of less frequent words.[^mikolov2013phrases]

Subsampling is more than an engineering trick. It changes the effective distribution of contexts and therefore the semantics encoded by the model. Many practical differences among embedding systems come from such weighting choices rather than from the headline architecture alone.

#### 4.5.5 Why the vectors develop geometry

Suppose *cat* and *dog* predict many of the same context words. Gradient updates repeatedly push their input vectors toward configurations that produce similar output scores. Their vectors need not be identical—each has distinctive contexts—but they become geometrically close because they serve similar predictive roles.

The dot product controls association:

$$
\{\mathbf{v}'_c}^\top\mathbf{v}_w.
$$

For normalized vectors, dot product and cosine similarity coincide. In practice, cosine similarity became the standard way to inspect nearest neighbors because it discounts vector magnitude and emphasizes direction.

A crucial theoretical result clarified what SGNS learns. Levy and Goldberg showed that, under an idealized high-dimensional optimum, SGNS implicitly factorizes a shifted PMI matrix:[^levy2014]

$$
\mathbf{v}_w^\top\mathbf{v}'_c
\approx
\operatorname{PMI}(w,c)-\log k.
$$

Thus negative sampling does not merely produce mysterious “neural features.” Its dot products approximate a familiar measure of statistical association, shifted by the number of negative samples. Low dimensionality forces a compressed factorization, and stochastic optimization plus frequency-dependent weighting determines how that compression generalizes.

This bridges two traditions:

- **explicit methods** construct a word-context matrix, transform it, and perhaps factorize it;
- **implicit methods** learn low-dimensional factors directly by optimizing predictions or classifications over sampled pairs.

The information source remains word-context co-occurrence.

#### 4.5.6 Analogies and linear relationships

word2vec drew widespread attention because some relationships appeared as approximately consistent vector offsets. For an analogy

$$
a:b::c:?,
$$

a common query is

$$
\mathbf{q}=\mathbf{v}_b-\mathbf{v}_a+\mathbf{v}_c,
$$

followed by nearest-neighbor search for the vector closest to $\mathbf{q}$. Examples in the original work included country-capital and morphological relationships.[^mikolov2013regularities]

A minimal implementation is:

```python
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def analogy(a: str, b: str, c: str, vectors: dict[str, np.ndarray]) -> str:
    """Solve a:b :: c:? using 3CosAdd."""
    query = vectors[b] - vectors[a] + vectors[c]
    excluded = {a, b, c}

    candidates = (
        (word, cosine_similarity(query, vector))
        for word, vector in vectors.items()
        if word not in excluded
    )
    return max(candidates, key=lambda item: item[1])[0]
```

The famous arithmetic should not be interpreted as symbolic reasoning encoded perfectly in Euclidean space. Analogy results depend on corpus, preprocessing, vocabulary, dimension, similarity rule, and evaluation dataset. Relations are not uniformly linear, and nearest-neighbor structure can create misleading successes. Nevertheless, the finding demonstrated that unsupervised learning could organize many lexical regularities into reusable geometric structure.

### 4.6 GloVe

#### 4.6.1 Motivation: use global counts directly

GloVe—**Global Vectors**—starts from a word-word co-occurrence matrix $X$. Let

$$
X_{ij}=\text{number of times word } j \text{ appears in the context of word } i,
$$

$$
X_i=\sum_k X_{ik},
\qquad
P_{ij}=P(j\mid i)=\frac{X_{ij}}{X_i}.
$$

Pennington et al. motivate the model using ratios of co-occurrence probabilities.[^pennington2014] Consider *ice* and *steam* with probe words:

- *solid* is strongly associated with *ice* but less with *steam*;
- *gas* is strongly associated with *steam* but less with *ice*;
- *water* is associated with both;
- an unrelated word is associated with neither.

The ratio

$$
\frac{P(k\mid \text{ice})}{P(k\mid \text{steam})}
$$

is large for *solid*, small for *gas*, and near 1 for words that do not distinguish the two. The ratio can therefore isolate a particular semantic contrast better than either raw probability alone.

GloVe seeks a vector-space function that captures these ratios through vector differences and dot products. The derivation leads to the relation

$$
\mathbf{w}_i^\top\tilde{\mathbf{w}}_j+b_i+\tilde b_j
\approx \log X_{ij}.
$$

Here $\mathbf{w}_i$ is a word vector, $\tilde{\mathbf{w}}_j$ a context vector, and $b_i,\tilde b_j$ bias terms.

#### 4.6.2 The GloVe objective

The objective is weighted least squares:

$$
J=
\sum_{i,j=1}^{V}
f(X_{ij})
\left(
\mathbf{w}_i^\top\tilde{\mathbf{w}}_j
+b_i+\tilde b_j
-\log X_{ij}
\right)^2.
$$

The weighting function used in the paper is

$$
f(x)=
\begin{cases}
(x/x_{\max})^\alpha, & x<x_{\max} \\
1, & x\ge x_{\max}.
\end{cases}
$$

with $\alpha=3/4$ and $x_{\max}=100$ in the reported experiments.[^pennington2014]

The weighting function serves three purposes:

1. zero entries contribute no loss;
2. rare co-occurrences are not trusted as much as well-observed ones;
3. extremely frequent pairs do not dominate without bound.

GloVe trains only on nonzero co-occurrences. It first pays the cost of scanning the corpus and building an aggregated sparse matrix, then can perform repeated optimization passes over the unique nonzero pairs rather than over every token window.

A simplified training expression looks like this:

```python
# X_ij > 0 is an observed co-occurrence count.
weight = min((X_ij / x_max) ** alpha, 1.0)
prediction = dot(word_vector[i], context_vector[j]) + word_bias[i] + context_bias[j]
loss = weight * (prediction - log(X_ij)) ** 2
```

After training, implementations commonly use $\mathbf{w}_i+\tilde{\mathbf{w}}_i$, or sometimes one of the two matrices, as the final embedding.

#### 4.6.3 Is GloVe fundamentally different from word2vec?

At the implementation level, yes:

- SGNS processes sampled local word-context events and applies a logistic classification loss.
- GloVe aggregates counts first and applies weighted regression to log co-occurrences.

At the statistical level, the distinction is smaller. Both learn low-rank factors of information derived from word-context counts. SGNS implicitly targets shifted PMI; GloVe explicitly fits log counts with learned biases that absorb marginal frequencies. Both use weighting schemes to prevent common or noisy observations from dominating.

It is therefore misleading to summarize the comparison as “word2vec uses local context, while GloVe uses global context,” as though word2vec had no access to global corpus statistics. Repeated local samples *are* the corpus statistics, encountered stochastically. The more precise distinction is:

- word2vec’s global behavior emerges from online optimization over local events;
- GloVe explicitly aggregates those events into a global matrix before optimization.

The resulting quality depends heavily on corpus, preprocessing, vector dimension, context window, vocabulary threshold, frequency weighting, and downstream task. Neither objective is universally superior.

### 4.7 Comparison

| Property | Count/PPMI + SVD | word2vec CBOW | word2vec Skip-gram with negative sampling | GloVe |
|---|---|---|---|---|
| Basic training signal | Explicit word-context or term-document matrix | Predict center word from context | Distinguish observed center-context pairs from noise | Regress on log global co-occurrence counts |
| Corpus access pattern | Aggregate matrix, then factorize | Stream local windows | Stream local windows and sampled negatives | Aggregate sparse matrix, then optimize nonzero entries |
| Main objective type | Matrix approximation | Predictive classification | Binary logistic objective | Weighted least squares |
| Frequent-word handling | Count weighting such as TF-IDF/PMI | Subsampling and output approximation | Subsampling plus noise distribution | Explicit weighting function and biases |
| Typical strengths | Interpretable connection to counts; strong baselines | Fast; effective for frequent words | Strong general-purpose embeddings; scalable; good rare-word behavior relative to CBOW | Efficient reuse of aggregate statistics; strong similarity and analogy results |
| Typical weaknesses | Large sparse matrices; weighting/factorization choices | Bag-of-words context; may smooth rare distinctions | Static word types; sampling and hyperparameter sensitivity | Requires co-occurrence construction; static word types; memory for sparse matrix |
| Out-of-vocabulary words | None unless composed from subwords/features | None | None | None |
| Context-sensitive meaning | No | No | No | No |

The table should not be read as a fixed ranking. Well-tuned explicit PPMI/SVD models can rival neural embeddings on some similarity tasks, a point reinforced by the analysis of SGNS as matrix factorization.[^levy2014] The advantage of word2vec was a combination of scale, simplicity, released code, and empirically useful geometry.

### 4.8 How embeddings were used

#### 4.8.1 As pretrained input features

A supervised model can initialize its embedding table with pretrained vectors. For token $w_t$, the system retrieves

$$
\mathbf{x}_t=E[w_t],
$$

then supplies $\mathbf{x}_t$ to a classifier, convolutional network, recurrent network, parser, or tagger. The embeddings may be:

- **frozen**, preserving the pretrained geometry;
- **fine-tuned**, allowing task-specific updates;
- or concatenated with manually designed features.

This was one of the first broadly successful forms of pretraining in NLP: large unlabeled corpora supplied lexical knowledge, while smaller labeled datasets trained task-specific models. Turian et al. documented the general value of unsupervised word representations as additional features, and Collobert et al. helped normalize learned embeddings inside neural NLP architectures.[^turian2010][^collobert2011]

#### 4.8.2 Semantic search and retrieval

Given an embedding matrix $E$, semantic nearest-neighbor retrieval computes words maximizing cosine similarity to a query. Approximate nearest-neighbor indexes can scale this operation to large vocabularies or document collections. Modern embedding-based retrieval systems use much more capable sentence or document encoders, but the basic idea—map symbols or texts into a shared metric space and search by geometry—is continuous with word-vector methods.

#### 4.8.3 Composition

Simple systems represent a sentence by summing or averaging its word vectors:

$$
\mathbf{s} = \frac{1}{n}\sum_{t=1}^{n}\mathbf{v}_{w_t}.
$$

This can work surprisingly well for topic or coarse similarity because frequent semantic directions accumulate. However, it discards word order and interaction. “Dog bites man” and “man bites dog” have the same average. Compositional neural models, RNNs, CNNs, attention, and Transformers were needed to transform token vectors into order- and context-sensitive sequence representations.

#### 4.8.4 Cross-lingual geometry

If two monolingual embedding spaces encode comparable relations, a linear transformation can be learned from a seed dictionary:

$$
W^* = \arg\min_W \lVert WX-Y\rVert_F^2.
$$

The mapped source vector $W\mathbf{x}$ can then retrieve nearby target-language words. Mikolov, Le, and Sutskever showed that simple linear mappings between monolingual word-vector spaces could support dictionary and phrase-table induction.[^mikolov2013crosslingual] This became an important line of cross-lingual representation research.

### 4.9 Evaluation

#### 4.9.1 Intrinsic evaluation

Common intrinsic tests include:

- **word similarity:** correlate cosine similarities with human ratings;
- **word relatedness:** test broader association rather than strict similarity;
- **analogy completion:** solve vector-offset questions;
- **nearest-neighbor inspection:** qualitatively examine local neighborhoods;
- **clustering:** test whether categories emerge.

These tests are cheap and diagnostic, but they have serious limitations. Human similarity judgments can be ambiguous, datasets are often small, repeated tuning on standard benchmarks encourages overfitting, and high intrinsic scores may not correlate with downstream performance.[^faruqui2016]

#### 4.9.2 Downstream evaluation

A stronger test asks whether embeddings improve a real task: named-entity recognition, part-of-speech tagging, parsing, sentiment analysis, information retrieval, machine translation, or question answering. Downstream evaluation is more expensive and entangles the embedding with model architecture and training choices, but it measures utility rather than aesthetic geometry.

A complete evaluation should therefore include both:

- intrinsic probes to understand the representation;
- controlled downstream experiments to determine whether the representation helps the intended application.

### 4.10 Limitations and failure modes

#### 4.10.1 One vector per word type

The largest conceptual limitation is polysemy. Static models compress every occurrence into one vector:

$$
\mathbf{v}_{\text{bank}}
$$

must summarize river edges, financial institutions, aircraft maneuvers, and any other usage. Frequent senses dominate; rare senses may disappear. Contextual models later addressed this by computing

$$
\mathbf{h}_{t}=f(w_1,\ldots,w_n,t),
$$

so the representation of the token at position $t$ depends on its sentence.[^peters2018]

#### 4.10.2 Out-of-vocabulary and morphology

A word-level table cannot represent an unseen token. It also fails to share structure among morphologically related forms unless corpus contexts happen to produce it. *walk*, *walked*, *walking*, and *walker* occupy independent vocabulary rows.

fastText directly extended Skip-gram by representing a word as a sum of character n-gram vectors. This enables sharing across morphological forms and permits approximate vectors for unseen words.[^bojanowski2017]

#### 4.10.3 Loss of word order

CBOW explicitly averages context. Skip-gram predicts neighboring words but does not inherently represent the order in which they occurred. Position-sensitive variants exist, but canonical static vectors summarize a word type’s distribution, not a sentence’s ordered structure.

#### 4.10.4 Corpus bias becomes geometry

Embeddings reflect social, cultural, and historical patterns in their training data. Bolukbasi et al. showed that word embeddings trained on news text encoded harmful gender stereotypes and demonstrated that analogy-like geometry could expose them.[^bolukbasi2016] This is not an accidental exception: distributional learning is designed to preserve corpus associations. Removing one measured bias direction does not guarantee that all harmful associations or downstream effects are removed.

#### 4.10.5 Similarity is not full meaning

Distributional similarity can conflate synonyms, antonyms, topical associates, and words sharing grammatical roles. *Hot* and *cold* occur in similar contexts and may therefore be close even though they contrast. The geometry represents use patterns, not a complete symbolic theory of reference, truth, causality, pragmatics, or world knowledge.

#### 4.10.6 Frequency and domain sensitivity

Rare words have few reliable observations. Domain-specific meanings can be overwhelmed by a general corpus. A vector trained on news may encode *cell* differently from one trained on biomedical literature. Corpus choice is therefore part of the model definition, not merely a data-volume decision.

#### 4.10.7 Instability and non-identifiability

Embedding coordinates have no fixed semantic identity. If all vectors are rotated by the same orthogonal matrix, dot products are unchanged. Separate training runs can produce spaces with equivalent neighborhoods but different axes. This is expected: the meaningful objects are relative positions and relationships, not coordinate 17 in isolation.

#### 4.10.8 Analogy overinterpretation

Successful analogies made embeddings vivid, but analogy accuracy should not be treated as a general intelligence measure. Results can depend on frequency, neighborhood density, relation regularity, and exclusion rules. The benchmark tests whether a specific geometric heuristic retrieves a word, not whether the model can explain or robustly reason about the relation.

### 4.11 Practical modeling choices

Several choices matter as much as the algorithm name.

**Window size.** Small windows emphasize local syntactic substitutability; larger windows emphasize topical relatedness.

**Dynamic windows.** Randomly varying the effective radius gives nearer contexts more opportunities to be sampled than distant ones.

**Vector dimension.** Too few dimensions underfit; too many increase memory, training cost, and the ability to memorize noise. Values around 100–300 were historically common, but the best choice is empirical.

**Minimum count.** Removing very rare words improves speed and statistical reliability but creates more out-of-vocabulary cases.

**Subsampling threshold.** Stronger subsampling reduces the dominance of function words and speeds training, but can discard useful syntactic signal.

**Negative-sample count.** More negatives produce a stronger discrimination problem at greater cost. The appropriate number depends on corpus size and frequency distribution.

**Context distribution.** The negative-sampling distribution changes the implicit matrix being factorized and therefore the resulting geometry.

**Preprocessing.** Lowercasing, punctuation handling, phrase detection, number normalization, and tokenization can materially alter the vocabulary and co-occurrences.

**Final vector choice.** SGNS and GloVe learn word and context matrices. Using the input matrix, context matrix, their sum, or their concatenation can affect results.

**Evaluation target.** Hyperparameters that maximize analogy accuracy may not maximize named-entity recognition, retrieval, or domain adaptation performance.

---

## 5. Recap and synthesis

The history of word embeddings is a progression rather than a single invention. Distributional linguistics supplied the core premise that linguistic environments reveal structure.[^harris1954] Count-based vector-space models and latent semantic analysis showed that co-occurrence data could be organized geometrically and compressed through low-rank factorization.[^deerwester1990] Class-based language models and related statistical techniques provided earlier forms of lexical parameter sharing.[^brown1992]

Neural language models then made word vectors trainable internal parameters. Bengio et al. learned distributed vectors jointly with a next-word probability model, explicitly using continuous representations to generalize beyond observed n-grams.[^bengio2003] Turian et al. showed that unsupervised representations could transfer into supervised NLP systems, while Collobert et al. demonstrated broad neural NLP architectures built around learned word representations.[^turian2010][^collobert2011]

word2vec’s CBOW and Skip-gram architectures removed expensive nonlinear hidden computation and focused directly on scalable representation learning.[^mikolov2013efficient] Hierarchical softmax reduced normalized prediction to a tree traversal; negative sampling converted vocabulary prediction into discrimination between observed and noise pairs; frequent-word subsampling improved speed and vector quality.[^mikolov2013phrases] The resulting embeddings exposed syntactic and semantic regularities through nearest neighbors and approximate vector offsets.[^mikolov2013regularities]

GloVe approached the same broad goal from explicit global co-occurrence statistics. Its weighted regression objective fits vector dot products and biases to log counts, using a weighting function to balance rare and frequent pairs.[^pennington2014] Levy and Goldberg later showed that SGNS implicitly factorizes shifted PMI, demonstrating that prediction-based and count-based methods are deeply connected.[^levy2014]

The enduring contribution was not that a vector could perfectly encode a word’s meaning. It was that large amounts of unlabeled text could produce compact, reusable parameters with useful lexical geometry. This made pretraining, transfer learning, semantic retrieval, and neural end-to-end NLP substantially easier. The limits—static senses, unknown words, lost order, corpus bias, and weak composition—then defined the research agenda that followed.

---

## 6. How the field changed and what followed

### 6.1 Immediate changes

Dense pretrained word vectors became a standard interface between unlabeled corpora and labeled tasks. Researchers could download an embedding matrix, look up vectors by token, and improve a model without designing a full unsupervised learning pipeline. This lowered the barrier to neural NLP and reduced dependence on manually engineered lexical features.

The change also altered how researchers thought about linguistic knowledge. Instead of representing categories only through discrete labels or rules, models could encode graded similarity and multiple overlapping properties in a shared continuous space. A word could simultaneously participate in dimensions associated with topic, syntax, sentiment, geography, morphology, and many other regularities. No individual coordinate needed a human-assigned meaning; the distributed pattern mattered.

Embedding pretraining helped establish a reusable workflow:

1. learn representations from a large unlabeled corpus;
2. transfer those representations to a smaller supervised problem;
3. optionally fine-tune them for the task.

That workflow became central to later pretrained language models.

### 6.2 Direct technical descendants

**Subword embeddings.** fastText preserved the efficient Skip-gram framework while constructing words from character n-grams, directly addressing morphology and out-of-vocabulary words.[^bojanowski2017]

**Sequence encoders.** RNNs, CNNs, and encoder-decoder models took embeddings as token inputs and learned how to combine them across ordered sequences. Sequence-to-sequence neural models made end-to-end translation possible without a phrase table as the central representation.[^sutskever2014]

**Contextual embeddings.** ELMo replaced the one-vector-per-word assumption with vectors derived from a bidirectional language model. The same word received different representations in different sentences, directly targeting polysemy and contextual use.[^peters2018]

**Transformer representations.** The Transformer provided a parallel, attention-based architecture for repeatedly mixing token representations according to context.[^vaswani2017] BERT then paired deep bidirectional Transformer representations with large-scale pretraining and downstream fine-tuning.[^devlin2019]

These developments did not discard embeddings. They changed their role. A modern Transformer begins with a learned token embedding plus position-related information, but that initial vector is only layer zero:

$$
\mathbf{h}^{(0)}_t = E[w_t] + P[t].
$$

Each layer then computes a new context-dependent representation:

$$
\mathbf{h}^{(\ell+1)}_t
= \operatorname{TransformerLayer}
(\mathbf{h}^{(\ell)}_1,\ldots,\mathbf{h}^{(\ell)}_n)_t.
$$

The final meaning-like representation of a token is therefore a function of the whole processed context, not a fixed row in an embedding table.

### 6.3 What word vectors enabled for LLMs

Word-vector research contributed several ideas that remain foundational:

- **learn representations rather than hand-design all features;**
- **use unlabeled text as a training signal;**
- **share statistical strength through continuous geometry;**
- **optimize embeddings jointly with predictive objectives;**
- **treat similarity and relation structure as properties that can emerge from scale;**
- **transfer pretrained parameters into downstream tasks.**

LLMs generalize these ideas from isolated lexical types to deep contextual states. Their vocabulary usually consists of subword tokens rather than words. They model long sequences rather than fixed local windows. They repeatedly transform representations with attention and feed-forward layers. Autoregressive LLMs train to predict the next token, while encoder models such as BERT use objectives such as masked-token prediction. Yet the initial act—mapping a discrete token ID into a learned continuous vector—is directly continuous with the embedding tradition.

### 6.4 What was still lacking

Static word vectors did not provide:

- sentence-level compositional understanding;
- sensitivity to long-distance context;
- robust handling of word order;
- a distinct representation for each word sense in use;
- generation of coherent multi-sentence text;
- instruction following;
- broad in-context task adaptation;
- grounded knowledge of the physical or social world;
- or reliable multi-step reasoning.

Even contextual pretraining did not automatically solve these problems. Larger architectures, longer contexts, improved optimization, massive datasets, instruction tuning, reinforcement-based post-training, retrieval, tools, and system-level orchestration were later added. Word2vec and GloVe should therefore be understood as critical infrastructure in the transition from symbolic sparse NLP to representation learning—not as small LLMs or as complete language-understanding systems.

---

## 7. Future reading and directly connected topics

1. **Distributional semantics and explicit count models**  
   Study PMI, PPMI, entropy weighting, dependency-based contexts, LSA, and the effect of context definitions. This clarifies what neural embedding objectives inherit from classical corpus statistics.

2. **Neural probabilistic language models**  
   Read Bengio et al. (2003) closely to see how embeddings first functioned as part of a language model and how continuous parameter sharing addressed n-gram sparsity.[^bengio2003]

3. **The full word2vec paper sequence**  
   Read the CBOW/Skip-gram architecture paper, the negative-sampling and phrase-composition paper, and the linguistic-regularities paper together.[^mikolov2013efficient][^mikolov2013phrases][^mikolov2013regularities]

4. **SGNS as implicit matrix factorization**  
   Levy and Goldberg’s analysis is the most direct route to understanding why word2vec and PMI-based methods are mathematically related.[^levy2014]

5. **GloVe derivation and weighting**  
   Focus on the co-occurrence probability-ratio argument, the bias terms, and the role of the weighting function rather than memorizing only the final objective.[^pennington2014]

6. **Subword representations and fastText**  
   This is the most direct extension of word2vec for morphology, rare words, and unseen forms.[^bojanowski2017]

7. **Embedding evaluation**  
   Compare intrinsic similarity and analogy benchmarks with downstream evaluation, including critiques of small human-judgment datasets.[^faruqui2016]

8. **Bias in representation spaces**  
   Examine how corpus associations become geometric relationships, how bias is measured, and why post-hoc debiasing is incomplete.[^bolukbasi2016]

9. **Contextual embeddings**  
   ELMo is the cleanest next step because it explicitly contrasts context-independent word vectors with context-dependent representations.[^peters2018]

10. **From embeddings to Transformers and pretrained language models**  
    Read *Attention Is All You Need* and BERT after contextual embeddings to see how token vectors become deep, reusable sequence representations.[^vaswani2017][^devlin2019]

11. **Sentence and document embeddings**  
    Explore why averaging word vectors is limited, how contrastive objectives learn text-level spaces, and how modern embedding models support semantic search and retrieval-augmented generation.

12. **Tokenization and subword vocabularies**  
    Study byte-pair encoding, WordPiece, unigram language-model tokenization, and byte-level methods. Modern LLMs embed tokens that often do not correspond one-to-one with words.

---

## References

[^harris1954]: Zellig S. Harris. 1954. “Distributional Structure.” *WORD* 10(2–3):146–162. [DOI: 10.1080/00437956.1954.11659520](https://doi.org/10.1080/00437956.1954.11659520).

[^brown1992]: Peter F. Brown, Peter V. deSouza, Robert L. Mercer, Vincent J. Della Pietra, and Jenifer C. Lai. 1992. “Class-Based n-gram Models of Natural Language.” *Computational Linguistics* 18(4):467–480. [ACL Anthology](https://aclanthology.org/J92-4003/).

[^deerwester1990]: Scott Deerwester, Susan T. Dumais, George W. Furnas, Thomas K. Landauer, and Richard Harshman. 1990. “Indexing by Latent Semantic Analysis.” *Journal of the American Society for Information Science* 41(6):391–407. [DOI](https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9).

[^bengio2003]: Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. 2003. “A Neural Probabilistic Language Model.” *Journal of Machine Learning Research* 3:1137–1155. [JMLR](https://www.jmlr.org/papers/v3/bengio03a.html).

[^turian2010]: Joseph Turian, Lev-Arie Ratinov, and Yoshua Bengio. 2010. “Word Representations: A Simple and General Method for Semi-Supervised Learning.” In *Proceedings of ACL 2010*, 384–394. [ACL Anthology](https://aclanthology.org/P10-1040/).

[^collobert2011]: Ronan Collobert, Jason Weston, Léon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel Kuksa. 2011. “Natural Language Processing (Almost) from Scratch.” *Journal of Machine Learning Research* 12:2493–2537. [JMLR](https://www.jmlr.org/papers/v12/collobert11a.html).

[^mikolov2013efficient]: Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. “Efficient Estimation of Word Representations in Vector Space.” [arXiv:1301.3781](https://arxiv.org/abs/1301.3781).

[^mikolov2013phrases]: Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. “Distributed Representations of Words and Phrases and their Compositionality.” *Advances in Neural Information Processing Systems 26*. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546).

[^mikolov2013regularities]: Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. 2013. “Linguistic Regularities in Continuous Space Word Representations.” In *NAACL-HLT 2013*, 746–751. [ACL Anthology](https://aclanthology.org/N13-1090/).

[^mikolov2013crosslingual]: Tomas Mikolov, Quoc V. Le, and Ilya Sutskever. 2013. “Exploiting Similarities among Languages for Machine Translation.” [arXiv:1309.4168](https://arxiv.org/abs/1309.4168).

[^pennington2014]: Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. “GloVe: Global Vectors for Word Representation.” In *EMNLP 2014*, 1532–1543. [ACL Anthology](https://aclanthology.org/D14-1162/).

[^levy2014]: Omer Levy and Yoav Goldberg. 2014. “Neural Word Embedding as Implicit Matrix Factorization.” *Advances in Neural Information Processing Systems 27*. [NeurIPS](https://proceedings.neurips.cc/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html).

[^faruqui2016]: Manaal Faruqui, Yulia Tsvetkov, Pushpendre Rastogi, and Chris Dyer. 2016. “Problems With Evaluation of Word Embeddings Using Word Similarity Tasks.” In *RepEval 2016*, 30–35. [ACL Anthology](https://aclanthology.org/W16-2506/).

[^bolukbasi2016]: Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai. 2016. “Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings.” *Advances in Neural Information Processing Systems 29*. [NeurIPS](https://proceedings.neurips.cc/paper/2016/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html).

[^bojanowski2017]: Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017. “Enriching Word Vectors with Subword Information.” *Transactions of the Association for Computational Linguistics* 5:135–146. [ACL Anthology](https://aclanthology.org/Q17-1010/).

[^sutskever2014]: Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. “Sequence to Sequence Learning with Neural Networks.” *Advances in Neural Information Processing Systems 27*. [NeurIPS](https://proceedings.neurips.cc/paper/2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html).

[^vaswani2017]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. “Attention Is All You Need.” *Advances in Neural Information Processing Systems 30*. [NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).

[^peters2018]: Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. “Deep Contextualized Word Representations.” In *NAACL-HLT 2018*, 2227–2237. [ACL Anthology](https://aclanthology.org/N18-1202/).

[^devlin2019]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” In *NAACL-HLT 2019*, 4171–4186. [ACL Anthology](https://aclanthology.org/N19-1423/).
