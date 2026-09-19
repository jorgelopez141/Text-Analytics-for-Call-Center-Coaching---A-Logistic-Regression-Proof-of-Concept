# Text Analytics for Call Center Coaching — A Logistic Regression Proof of Concept

**Author:** Jorge D. Lopez Saavedra **Course:** DS 745 — Data Visualization, MS in Data Science, University of Wisconsin

A text analytics study that treats *word choice* as a measurable performance signal. The question behind the project is an operational one from the collections industry: **what actually separates a top-performing call center agent from an underperforming one?**

The answer this project proposes: not just the KPIs, but the words. And word choice can be modeled.

------------------------------------------------------------------------

## The Problem

Collections call centers measure agents with quantitative KPIs:

$$\frac{Customer\ Contact}{Number\ of\ Calls} \qquad \frac{Promises\ to\ Pay}{Customer\ Contact}$$

These tell you *that* an agent performs well. They never tell you *why*. Two agents make the same number of calls, reach the same number of customers, and one consistently secures more promises to pay. The difference lives in the transcript — in the verbs, the framing, the timing of specific words.

The operational gap is worse than it looks. KPIs arrive at month's end, and collections is a cyclical business — nobody collects well over spring break — so a soft month is ambiguous evidence. By the time a manager can distinguish "seasonal" from "struggling," the coaching window has closed.

## The Approach

Real call transcripts are proprietary, so this project builds the full pipeline on a public proxy with the same structure: **two bodies of text by different authors, with different vocabularies, that a model must tell apart.**

-   [*Moby Dick*](https://www.gutenberg.org/ebooks/2701) by Herman Melville — stands in for the prime-agent corpus
-   [*Romeo and Juliet*](https://www.gutenberg.org/ebooks/1513) by William Shakespeare — stands in for the sub-prime corpus

Two authors are to this project what two performance tiers are to a call center: distinct speakers whose separability lives entirely in their word distributions.

------------------------------------------------------------------------

## What Was Used

**Languages:** Python and R, interoperating inside a single Quarto document via `reticulate` — dataframes pass back and forth between the two runtimes in the same render.

| Layer | Tooling |
|------------------------------------|------------------------------------|
| Document / reporting | Quarto (`.qmd`) → self-contained HTML, floating TOC, tabsets, callouts |
| Text ingestion & tokenization | `nltk` (`punkt` sentence tokenizer, `word_tokenize`) |
| Stemming | `nltk.stem.PorterStemmer` |
| Vectorization | `sklearn.feature_extraction.text.CountVectorizer`, `ENGLISH_STOP_WORDS` |
| Modeling | `sklearn.linear_model.LogisticRegression`, `train_test_split`, `accuracy_score`, `classification_report` |
| Data wrangling (R) | `dplyr`, `tidyr` (`pivot_longer` / `pivot_wider`) |
| Visualization (R) | `ggplot2`, `ggiraph` (interactive tooltips), `wordcloud`, `scales` |
| Tables | `knitr::kable`, `kableExtra` |
| Bridge | `reticulate` (`py$object` ↔ `r.object`) |

**Files in this repo**

| File | Purpose |
|------------------------------------|------------------------------------|
| [FourthProject_JDLS.qmd](FourthProject_JDLS.qmd) | The full analysis — source document |
| [FourthProject_JDLS.html](FourthProject_JDLS.html) | Rendered, self-contained report |
| [RJ_cleaning_andGrouping.py](RJ_cleaning_andGrouping.py) | Sentence chunking for *Romeo and Juliet*, imported as a module by the `.qmd` |
| [Moby Dick.txt](Moby%20Dick.txt), [RomeAndJuliet.txt](RomeAndJuliet.txt) | Project Gutenberg source texts |

------------------------------------------------------------------------

## Pipeline

**1. Ingest and vectorize.** Both books are read as raw strings and passed through `CountVectorizer`. Raw vocabulary counts establish the baseline difference in lexical range between the two corpora.

**2. Strip stop words.** English stop words are removed, because the goal is the vocabulary that makes a speaker *distinctive*, not the vocabulary every English speaker shares. A regex filter (`^\d|^\_`) additionally drops chapter numbers and formatting artifacts that survive as pseudo-tokens.

**3. Stem.** `whale` and `whales` are the same signal counted twice. Porter stemming collapses inflections, which raises the count on the true signal words by hundreds of occurrences and sharpens the contrast between corpora.

**4. Visualize the vocabulary gap.** Word clouds per corpus, and a scatter plot of word frequency in one book against the other with a 45° reference line. Two views are provided:

-   *True scale* — surfaces the extreme discriminators (`whale`, `romeo`, `juliet`), the words that are essentially fingerprints.
-   *Log scale with jitter* — surfaces the more interesting middle band: words used by **both** speakers but at meaningfully different rates. `dead`, `night`, and `heart` skew to *Romeo and Juliet*; `world`, `head`, `thought`, and `said` skew to *Moby Dick*.

That second view is the one that matters operationally. Shared-but-skewed vocabulary is where coachable behavior lives — nobody needs a model to notice an agent never says "payment." The signal is in the ratios of words *everyone* uses.

**5. Chunk into observations.** Each book is split into sentences, cleaned of front matter and structural noise, then grouped into **chunks of 10 consecutive sentences**. This yields 849 *Moby Dick* observations and 297 *Romeo and Juliet* observations — 1,146 total.

**6. Model.** Chunks are stemmed, vectorized into a \~10,479-column document-term matrix, and split 80/20. Logistic regression predicts `1 = Romeo and Juliet`, `0 = Moby Dick`.

------------------------------------------------------------------------

## The Novelty: What Logistic Regression Is Doing Here

Logistic regression on bag-of-words is a textbook classifier. The novelty is not the algorithm — it is **the reframing of what the classifier is for.**

### 1. The unit of observation is a conversation, not a document

Most authorship classification asks "who wrote this book?" — a question with one answer per book and therefore no operational value. This project **deliberately fragments each corpus into 10-sentence chunks and treats each chunk as an independent observation.** That single design decision converts a document-level curiosity into a stream of scoreable events.

In the call center mapping, a chunk is a call. You do not want to know whether Agent X is a good agent in aggregate. You want to know whether *this call, yesterday at 2:15pm*, sounded like a good one.

### 2. The prediction of interest is the probability, not the label

The classifier separates the two corpora cleanly on held-out data, which makes the hard label uninformative. The project's actual output is `predict_proba()` — the **continuous confidence score.**

This is the conceptual pivot. A binary label says "this is Moby Dick." A probability says "this is 78.2% Moby Dick" — and *that* is a measurement you can track, trend, and coach against. The report demonstrates the point by pulling two chunks that were both classified correctly:

-   **Chunk 218** — scored above 99%. Dense with the book's signature vocabulary.
-   **Chunk 277** — scored 78.24%. Same book, correctly labeled, but the word `whale` never appears and the register turns emotional, drifting toward the other corpus.

Chunk 277 is the whole thesis in one example. In the call center analogy, that is a prime agent having an off call — visible in the score *while the label still reads "prime,"* and visible immediately, not at month's end.

### 3. The coefficients are the coaching curriculum

Because the model is linear in the vocabulary, every feature weight is a word with a signed, interpretable contribution. The model is not a black box that ranks agents; it is a **ranked list of the words that move the outcome.** A neural classifier might score better and would be useless for this purpose — you cannot hand a manager an embedding and call it feedback. Interpretability is not a tradeoff accepted here; it is the deliverable.

### 4. A leading indicator that is immune to seasonality

Revenue-based KPIs conflate agent skill with market conditions. Language does not fluctuate with the collections calendar. Scoring transcripts produces a performance signal that is **available daily and decoupled from the revenue cycle** — the gap that the KPIs in the problem statement cannot close.

------------------------------------------------------------------------

## Applying This to a Real Call Center

Swap the two books for two transcript corpora and the pipeline runs unchanged:

| This project | Production deployment |
|------------------------------------|------------------------------------|
| *Moby Dick* corpus | Transcripts from prime agents (highest collection rate) |
| *Romeo and Juliet* corpus | Transcripts from sub-prime agents (growth phase) |
| 10-sentence chunk | One call, or a segment of a long call |
| `predict_proba()` output | **Conversation Quality Score**, 0–100 |
| Model coefficients | Ranked phrases to reinforce or retrain |

**What the score answers, in order of operational value:**

1.  **Are underperforming agents improving?** Trend each agent's mean score weekly. Rising scores mean their language is converging on the prime pattern — improvement visible weeks before it reaches the revenue line.
2.  **Is a prime agent slipping?** A falling score is an early warning. Burnout and disengagement show up in word choice before they show up in a monthly report.
3.  **Which specific call should we review?** The score is per-conversation. A supervisor opens the ten lowest-scoring calls of the week instead of sampling at random — coaching aimed at the exact moment it was needed.

**What the coefficients teach.** The report's worked example:

> **Prime agent:** "I *require* payment in full today." **Sub-prime agent:** "I would like to *see* if you could *make* a payment."

Both are polite. Both request payment. The first uses a directive verb and a deadline; the second hedges. That is a specific, teachable, five-minute coaching conversation — and it is exactly the kind of difference the coefficients surface without anyone having to hypothesize it in advance.

**Domain-specific stop words.** Vocabulary that saturates every call in an industry — `debt`, `amount`, `pay`, `payment` — carries no discriminating information and must be suppressed, or it will crowd out the real signal:

``` python
new_stop_words = ENGLISH_STOP_WORDS.union(['debt', 'amount', 'pay', 'payment'])
```

**Beyond collections.** Any operation with (a) recorded conversations and (b) an outcome metric that sorts staff into tiers fits the same template: sales (closed vs. lost), technical support (first-call resolution vs. escalation), retention (saved vs. churned), healthcare intake (appointment kept vs. no-show).

------------------------------------------------------------------------

## Caveats Before Anyone Deploys This

Stated plainly, because the proof of concept is a proof of *concept*:

-   **The 100% held-out accuracy is a property of the proxy, not a forecast.** Melville and Shakespeare are maximally separable — different centuries, genres, and registers. Two tiers of agents reading from the same call script are a far harder problem, and real-world accuracy will be substantially lower. The pipeline is what transfers; the number is not.
-   **Class imbalance is present and unaddressed** (849 vs. 297 chunks). Production work needs balanced sampling or class weighting, and evaluation by precision/recall and AUC rather than accuracy.
-   **Correlation, not causation.** The model learns words that *co-occur* with high performance. Whether coaching an agent into those words causes better collection outcomes is an empirical question requiring a controlled trial.
-   **Bag-of-words discards order, negation, and tone.** "I can help you today" and "I can't help you today" are near-identical to this model after stop-word removal.
-   **This is monitoring of employee speech.** Deploying it carries real obligations around consent, transparency, and using scores for development rather than discipline. Worth settling before the first model runs, not after.

------------------------------------------------------------------------

## Reproducing the Analysis

**Requirements**

``` bash
# Python
pip install pandas numpy scikit-learn nltk
python -c "import nltk; nltk.download('punkt')"
```

``` r
# R
install.packages(c("reticulate", "dplyr", "tidyr", "ggplot2", "ggiraph",
                   "wordcloud", "RColorBrewer", "scales", "knitr", "kableExtra"))
```

Plus [Quarto](https://quarto.org/docs/get-started/).

**Run**

``` bash
quarto render FourthProject_JDLS.qmd
```

> **Note:** [FourthProject_JDLS.qmd](FourthProject_JDLS.qmd) and [RJ_cleaning_andGrouping.py](RJ_cleaning_andGrouping.py) both set an absolute working directory via `os.chdir()`. Update those paths to your local clone before rendering.

------------------------------------------------------------------------

## Closing Note

The report ends on the line that motivates the whole project:

> There is power in the language we use.

The contribution here is making that power *measurable* — and measurable early enough to act on.

------------------------------------------------------------------------

## Data Source

Both texts are public domain via [Project Gutenberg](https://www.gutenberg.org/).