# Data

This project implements the **inference** (forward pass) of an already-trained model, not training from
scratch, so no training corpus is needed. These files exist only for two purposes: sanity-checking
generations by eye, and measuring standard metrics against a held-out set.

## `sherlock_holmes.txt`

*The Adventures of Sherlock Holmes* by Arthur Conan Doyle, plain text, public domain
(source: [Project Gutenberg #1661](https://www.gutenberg.org/ebooks/1661)).

Use for: Day 2 attention parity test (feed real sentences through your implementation and the HF
reference, compare per-layer max-abs-error), and any qualitative "does the generation look sane"
check. Coherent prose makes it easy to eyeball whether sampling/greedy decoding is behaving.

Note: the file has a Project Gutenberg license header/footer — strip those before tokenizing if you
want clean prose (search for `*** START OF THE PROJECT GUTENBERG EBOOK` / `*** END OF...`).

## `wikitext-2-raw/wiki.test.raw`

The WikiText-2 (raw) **test split**, converted from the Hugging Face parquet mirror
(`Salesforce/wikitext`, config `wikitext-2-raw-v1`). This is the standard small benchmark corpus used
for language-model perplexity in the literature ("raw" = untokenized, punctuation and casing intact,
unlike the older word-level WikiText-2 which replaces rare words with `<unk>`).

Use for: Week 4 Day 27 (perplexity on a held-out set, compared across precisions), and any other place
the plan asks for a "held-out set" for quality measurement.

The original S3 mirror the plan-era tooling used
(`s3.amazonaws.com/research.metamind.io/wikitext/...`) is dead; the Hugging Face parquet mirror is the
current canonical source.
