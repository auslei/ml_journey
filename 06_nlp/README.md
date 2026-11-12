# 06 NLP

This section covers natural language processing from raw text to the Transformer encoder. The thread is representation: how text is turned into numbers, what those numbers capture, and how sequence models read them in order. All neural models use native PyTorch.

## Concept map

```
raw text
    |
    v
preprocessing: tokenise, clean, stopwords, stem/lemmatise (01)
    |
    v
representations
    |--- bag-of-words, TF-IDF: fast, sparse, no word order (01)
    |--- word embeddings: dense, capture similarity, trained by context (02)
    |--- topic models: LDA, unsupervised, documents as mixtures of topics (03)
    |
    v
sequence models: text in order
    |--- RNN, LSTM: autoregressive language model, perplexity (04)
    |--- LSTM classifier: sequence -> label, sentiment (05)
    |
    v
attention: every position can query every other position directly (06)
    |
    v
Transformer encoder block: attention + residuals + feed-forward (06)
```

## Reading order

1. `01_text_preprocessing.ipynb`: tokenisation, stopword removal, stemming and lemmatisation by hand and with NLTK and spaCy, then bag-of-words and TF-IDF from scratch and with sklearn.
2. `02_word_embeddings.ipynb`: why one-hot vectors have no geometry, CBOW in NumPy then PyTorch, skip-gram idea, gensim Word2Vec on the Gutenberg corpus, t-SNE visualisation of the learned space.
3. `03_topic_models.ipynb`: the LDA generative story, Dirichlet priors and plate notation, bag-of-words preprocessing with gensim, training LDA on a small corpus, topic coherence over different values of k.
4. `04_sequence_models_text.ipynb`: the autoregressive objective, character-level language model, vanilla RNN cell in PyTorch, LSTM swap, perplexity as a measure of surprise, text sampling.
5. `05_sentiment_classification.ipynb`: sequence classification with one label per document, LSTM classifier with `pack_padded_sequence`, logistic bag-of-words baseline, pre-trained gensim embeddings frozen then fine-tuned.
6. `06_attention_transformer.ipynb`: the fixed-hidden-state bottleneck in RNNs, scaled dot-product attention from scratch, multi-head attention, sinusoidal positional encoding, Transformer encoder block with Add and Norm, attention weight visualisation.
7. `intuition_test.ipynb`: predict first, then run. Questions covering all six notebooks.

Dependencies: `uv sync --extra torch`. No TensorFlow needed. NLTK data is downloaded by the notebooks on first run (punkt, stopwords, gutenberg, wordnet, movie_reviews). The spaCy model is `en_core_web_sm`; install with `uv run python -m spacy download en_core_web_sm`.

## What you should be able to explain at the end

- Why TF-IDF scores a rare word higher than a common one, and what the inverse document frequency term is doing.
- Why cosine similarity between any two one-hot vectors is 0, and what a dense embedding fixes.
- What the distributional hypothesis says, and why context predicts meaning.
- The difference between CBOW and skip-gram: which direction the prediction runs, and which trains faster on large corpora.
- What the Dirichlet parameter α controls in LDA: how spread out each document's topic mix is, and what a low α implies about document structure.
- Why coherence is a better measure of topic quality than raw likelihood.
- What perplexity measures: the geometric mean probability the model assigns to each token, and why lower is better.
- Why vanilla RNNs struggle with long sequences, and what the gates in an LSTM each control (forget, input, output).
- Why padding sequences to the same length and using `pack_padded_sequence` matters for the LSTM hidden state.
- Why freezing pre-trained embeddings for the first few epochs often helps, and when fine-tuning them makes sense.
- What the query, key and value are in scaled dot-product attention, and why the dot products are divided by sqrt(d_k).
- What positional encoding adds, and why it is necessary: attention has no notion of order without it.
- What Add and Norm (residual connection + layer normalisation) does for gradient flow.
- One concrete failure mode for each representation type: TF-IDF (word order), word2vec (polysemy), LDA (topic count is a hyperparameter), RNN sentiment (negation), attention (quadratic memory in sequence length).

The next section builds on the Transformer encoder: decoder blocks, autoregressive generation, tokenisation at scale, and fine-tuning pre-trained language models.
