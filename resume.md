# BhashaLM — Resume Entry

## Resume section

**BhashaLM: Hindi and Nepali Language Models from Scratch** | *PyTorch, Python, W&B, Hugging Face Hub, Gradio* | [github.com/RajaiTarun/BhashaLM](https://github.com/RajaiTarun/BhashaLM)

- Built two **~25M-parameter GPT-style language models from scratch in PyTorch** for Hindi and Nepali without using `nn.Transformer` or pretrained weights. The implementation includes custom multi-head causal attention, RoPE, SwiGLU, RMSNorm and a KV cache, plus a FlashAttention (SDPA) fast path that tests show matches the manual path numerically.
- Implemented a **byte-level BPE tokenizer** with Devanagari-aware pre-tokenization. Chose the vocabulary size with an 8k–48k sweep over fertility, bytes per token and embedding parameter cost, and benchmarked it against SentencePiece.
- Pretrained on **~700M words of Devanagari text (1.8M documents)** with mixed-precision (fp16) training on free Kaggle/Colab T4 GPUs. Atomic checkpoints and automatic resume let training survive session kills.
- Designed a **2×2 data-budget experiment** (Hindi/Nepali × 125M/500M tokens). Because both languages share a script, it separates the effect of data scale from the effect of language; reached **X.XX bits-per-byte** on Hindi and **X.XX** on Nepali.
- Built a **synthetic reasoning benchmark** from native-language templates for comparative and transitive reasoning, with a hop-count generalization split. Supervised finetuning with answer-only loss raised accuracy from **XX% to XX%**, with a catastrophic-forgetting check on the general LM.
- Analyzed model internals with **attention entropy and distance, head ablation and the logit lens**, and identified the heads that drive comparison reasoning. Released the models on Hugging Face Hub with model cards and a **Gradio demo**.

---

## Short version (3 bullets, for a one-page resume)

- Built two ~25M-parameter GPT-style LMs **from scratch in PyTorch** (custom attention, RoPE, KV cache, byte-level BPE tokenizer) for Hindi and Nepali. Pretrained on ~700M words with fault-tolerant mixed-precision training on free GPUs.
- Ran a controlled **data-scale × language study** on two same-script languages, reaching **X.XX / X.XX bits-per-byte** (Hi / Ne).
- Built a native-language **synthetic reasoning benchmark** with a hop-count generalization split; finetuning lifted accuracy **XX% → XX%**. Explained the gains with head ablation and the logit lens.

---

## One-liner (for a LinkedIn headline or projects list)

> Built Hindi and Nepali GPT-style language models from scratch in PyTorch: tokenizer, transformer, training, reasoning finetuning and interpretability, with no pretrained components.

---

## Interview talking points

| Topic | What to say |
|---|---|
| Why Hindi and Nepali | Both use Devanagari, so script is held constant and data scale becomes the variable under study. |
| Why ~25M params / 500M tokens | About 20 tokens per parameter, which is roughly Chinchilla compute-optimal. That is ~7.5×10¹⁶ training FLOPs per model, which fits on a free T4. |
| Why BPB over perplexity | Perplexity depends on the tokenizer. BPB normalizes by UTF-8 bytes, so the two languages can be compared fairly. |
| Vocab trade-off | At d=384, a 32k vocabulary costs ~12M of the 25M parameters, so vocabulary size competes directly with model depth. |
| Causality test | Change token t+1 and check that the logits at positions ≤ t stay the same. |
| Fault tolerance | Checkpoints are written atomically (tmp file, then rename) and include optimizer, scheduler, scaler and RNG state, so a resumed run continues exactly where it stopped. |
| ROUGE gotcha | `rouge_score`'s default tokenizer drops non-ASCII characters, so Devanagari scores 0 unless you use a custom tokenizer. |
