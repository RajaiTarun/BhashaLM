# BhashaLM

**Small GPT-style language models for Hindi and Nepali, built from first principles in PyTorch.**

*bhāṣā* (भाषा) means "language" in both Hindi and Nepali.

BhashaLM is two separate ~25M-parameter decoder-only transformers, **BhashaLM-Hi** (Hindi) and **BhashaLM-Ne** (Nepali). Everything after the raw text is implemented by hand: the byte-level BPE tokenizer, multi-head causal attention, RoPE, the KV cache, the training loop and the evaluation suite. The project does not use `nn.Transformer`, Hugging Face model classes, or any pretrained tokenizer or weights.

Both languages are written in **Devanagari**, which makes the comparison controlled. Script is held constant, so differences between the two models come from data scale and data quality, not from the writing system.

---

## Highlights

- **Built from scratch.** Byte-level BPE with Devanagari-aware pre-tokenization, multi-head causal self-attention, RoPE / sinusoidal / learned positional encodings, SwiGLU, RMSNorm, weight tying and a KV cache.
- **Training that survives interruptions.** Mixed precision (fp16 + GradScaler), gradient accumulation, warmup + cosine schedule, and atomic checkpoints that store optimizer, scheduler, scaler and RNG state. Runs resume automatically after a Kaggle or Colab session dies.
- **2×2 data-budget study.** Hindi and Nepali are each trained at 125M and 500M tokens. This separates the effect of *data scale* from the effect of *language*.
- **Synthetic reasoning benchmark.** Templates written natively in Hindi and Nepali for comparative and transitive reasoning, with a hop-count generalization split: train on 2–3 hops, test on 4–5.
- **Interpretability.** Attention heatmaps, per-head entropy and attention distance, head ablation and the logit lens, applied before and after finetuning.
- **Engineering.** Typed configs, pytest suite (including a causality test), GitHub Actions CI, ruff, pre-commit, W&B tracking, model cards, and a Gradio demo.

---

## Results

| Model | Params | Train tokens | Val loss | Perplexity | Bits-per-byte | Reasoning acc. (FT) |
|---|---|---|---|---|---|---|
| **BhashaLM-Hi** | ~25M | 500M | X.XX | X.XX | X.XXX | XX.X% |
| **BhashaLM-Ne** | ~25M | 500M | X.XX | X.XX | X.XXX | XX.X% |
| Hi-125M | ~25M | 125M | X.XX | X.XX | X.XXX | — |
| Ne-125M | ~25M | 125M | X.XX | X.XX | X.XXX | — |

Bits-per-byte (BPB) is the main cross-language metric because it does not depend on the tokenizer. Devanagari takes 3 bytes per character in UTF-8, and the report discusses chars-per-byte alongside BPB.

---

## Data

| | Hindi (train) | Nepali (train) |
|---|---|---|
| Documents | 853,505 | ~955,000 |
| Words | ~351M | ~343M |
| Sources | Wikipedia, Wikisource, news, web crawl | HPLT v2 web crawl, news, Wikipedia, Wikisource |

Each document keeps its provenance (`source_type`, `source_name`). Before training, the splits were audited for:

- Unicode normalization (NFC)
- leftover markup
- Hindi↔Nepali contamination
- exact-duplicate leakage across train, val and test

Tokenized corpora are stored as flat `uint16` memmaps (`train.bin`, `val.bin`, `test.bin`) so batches load quickly.

---

## Architecture

```
tokens ──► Embedding (tied) ──► [ Pre-Norm Block × N ] ──► RMSNorm ──► LM Head
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  x + MHA(RMSNorm(x))  (RoPE, causal mask, KV cache)
                    │  x + SwiGLU(RMSNorm(x))
                    └───────────────────────────────────┘
```

| d_model | n_layers | n_heads | d_ff | context | vocab | params (tied) |
|---|---|---|---|---|---|---|
| 384 | 8 | 6 | 1536 | 512 | 32k | ~26.5M |

- Manual attention is the reference implementation. A `F.scaled_dot_product_attention` (FlashAttention) fast path is used for training, and tests check that it matches the manual path numerically.
- Initialization follows GPT-2: N(0, 0.02), with residual projections scaled by 1/√(2·n_layers).
- The vocabulary size was chosen with a sweep (8k–48k) that measured fertility, bytes per token and vocabulary usage against the parameter cost of the embedding table.

---

## Reasoning finetuning

A programmatic generator builds the reasoning tasks from native Hindi and Nepali templates. Ground truth is computed from a graph or total order, then turned into a sentence. Task families:

- direct comparison
- transitive chains
- multi-hop ordering (3–5 entities)
- numeric comparison
- relation inversion

The splits use **disjoint name pools** and **held-out templates**. A **hop-count split** tests compositional generalization. Finetuning masks the loss so only answer tokens count. Evaluation uses both exact match and log-likelihood scoring over the candidate answers. It also checks for **catastrophic forgetting** on the general LM validation set.

---

## Repository layout

```
src/bhashalm/
├── data/        # audit, stats, splits, memmap writer
├── tokenizer/   # from-scratch byte-level BPE + SentencePiece baseline + analysis
├── model/       # embeddings, attention, MLP, block, GPT, KV cache
├── train/       # trainer, checkpointing, schedulers, memmap dataloader
├── eval/        # loss / PPL / BPB, generation metrics, diversity, reasoning eval
├── reasoning/   # synthetic dataset generator (per-language templates)
├── interp/      # attention heatmaps, entropy, distance, head ablation, logit lens
├── inference/   # sampling (greedy / temperature / top-k / top-p), KV-cache decoding
└── viz/         # plotting only
configs/{hindi,nepali}/   # data, tokenizer, model, pretrain, finetune YAMLs
scripts/                  # CLI entrypoints
tests/                    # shapes, causality, tokenizer round-trip, resume, metrics
report/                   # phase reports, final report, figures
```

---

## Quickstart

```bash
git clone https://github.com/RajaiTarun/BhashaLM && cd BhashaLM
uv sync

make data      LANG=hindi
make tokenizer LANG=hindi
make train     LANG=hindi
make eval      LANG=hindi
```

Generate text:

```bash
python scripts/generate.py --lang hindi --prompt "भारत की राजधानी" --temperature 0.8 --top-p 0.9
```

Run the tests:

```bash
uv run pytest
```

---

## Models and demo

- Checkpoints and tokenizers are on the Hugging Face Hub with model cards. They load through BhashaLM's own code, not through HF model classes.
- An interactive Gradio demo lets you pick a language, set sampling parameters, and view attention heatmaps.

---

## Acknowledgements

The corpora are built from HPLT v2, Hindi and Nepali Wikipedia and Wikisource, and public news sources. See the model cards for licensing and intended use.

## License

MIT
