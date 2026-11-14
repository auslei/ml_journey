# Section 06 NLP review, round 1 (2026-11-14)

Response to the Cursor review in `ml06_review_round1.md`. All nine findings
were accepted and fixed. None were pushed back.

## How each finding was resolved

1. Root `README.md`: section 06 marked `done` with a link to `06_nlp/`, the
   layout tree drops the `(planned)` tag on `06_nlp/`, Future-plans item 1 was
   removed and the list renumbered, and the Setup block gained the section 06
   sync line plus the spaCy model and NLTK download notes.
2. `02_word_embeddings.ipynb`: the gensim run now passes `sg=1`, so it really
   trains skip-gram. Prose and the `06_nlp/README.md` reading-order bullet were
   aligned to say skip-gram with negative sampling. Chosen over relabelling the
   run as CBOW because the section leads with skip-gram.
3. `intuition_test.ipynb`: the mid-notebook `**Answer.**` cells after Q4 and Q8
   were deleted. The Q4 and Q8 entries under `## Answers` were expanded to carry
   the full answer, so predict-first discipline holds.
4. `pyproject.toml`: `gensim`, `nltk` and `spacy` moved from core into a new
   `nlp` extra (torch kept separate). `uv.lock` regenerated with `uv lock`.
   Setup for section 06 is now `uv sync --extra torch --extra nlp`, documented
   in both READMEs.
5. `06_attention_transformer.ipynb`: the "order-invariant / shuffle" claim was
   replaced with a permutation-equivariant statement (no built-in notion of
   position), then used to motivate positional encoding.
6. `02_word_embeddings.ipynb`: the maths was transposed to match the code
   ($W_2 \in \mathbb{R}^{d \times V}$, $\mathrm{mean}(\ldots)^\top W_2$) rather
   than changing the working code.
7. Narrative em dashes were removed from markdown prose across the section 06
   notebooks and the root README.
8. `03_topic_models.ipynb`: `preprocess` now lemmatises only (the `PorterStemmer`
   import was dropped), matching the either/or lesson in notebook 01.
9. `03_topic_models.ipynb`: a sentence was added to the LDA section stating that
   LDA trains on `bow_corpus` and the TF-IDF corpus is shown only as a contrast.

## Assumptions

- Finding 7 scope follows the judge ruling: only section 06 files and the root
  README. Sections 01 to 05 were left untouched for a later polish feature.
- Em dashes left in code `print(...)` output and plot titles were kept, which
  the reviewer allowed. En dashes in numeric ranges (`150–200`, `83–85%`,
  `1–2`) are standard range notation and were kept.
- Changing notebook 03 preprocessing (finding 8) and the gensim `sg` flag
  (finding 2) shifts printed topic words and similarity numbers. Outputs are
  stripped, so only reproducibility matters; both notebooks still run clean.
