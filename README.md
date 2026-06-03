# ActivationMap

A single-file, in-browser **GPT-2 attention visualizer**. Type a prompt, hit
**Run**, and see a token×token heatmap of where the model is attending — pick
any of GPT-2's 12 layers and 12 heads from dropdowns.

No backend. No build step. No install. Everything runs client-side in one HTML
file: [`attention-viz.html`](attention-viz.html).

---

## What it does

- Runs **GPT-2 small** entirely in the browser via [Transformers.js].
- Extracts the **raw per-layer, per-head attention matrices** from a forward
  pass and renders them with [D3].
- Each cell `(row i, col j)` = how much query token *i* attends to key token
  *j*. Because GPT-2 is a decoder, the pattern is **causal** — the upper
  triangle is zero (tokens only attend backward), and each row is a softmax
  distribution that sums to 1.

### Controls
- **Prompt + Run** — tokenizes and runs a fresh forward pass.
- **Layer / Head** — switch instantly; results are cached, so no re-inference.
- **Color** — Blues / Viridis / Magma / Greens / Inferno.
- **Hover** any cell for the exact `row → col` token pair and weight.
- Token labels are cleaned BPE: a leading space is shown as `·` (e.g. `·cat`).

---

## Running it

The CDN ES-module imports and the model download both need HTTP, so opening the
file directly (`file://`) won't work. Serve the folder with any static server:

```bash
python3 -m http.server 8123
# then open http://localhost:8123/attention-viz.html
```

> **First load downloads ~653 MB** (the fp32 model weights). The browser caches
> them afterward, so subsequent loads are fast. A progress bar is shown, and the
> loader retries transient network failures.

---

## How it works

Transformers.js's high-level `pipeline()` API does **not** expose attention
weights, and — importantly — the standard [`Xenova/gpt2`] ONNX export doesn't
either: its only outputs are `logits` plus the key/value cache. There is no
attention tensor in the graph to read.

This project uses [`damoncrockett/gpt2-with-attentions-onnx`], a GPT-2 re-export
made with `output_attentions=True` / `use_cache=False`. Its graph exposes
`attentions.0 … attentions.11`, each shaped `[batch, heads, seq, seq]`. The app
loads it with `AutoModelForCausalLM`, calls the model with
`{ output_attentions: true }`, and reconstructs the ordered
`[layer][head][N][N]` tensor (the attentions come back as separate per-layer
keys, not a single array).

### Stack
- [Transformers.js] `@huggingface/transformers@4.2.0` (ONNX runtime, WASM)
- [D3] v7
- Both loaded from CDN as ES modules — no bundler.

---

## Limitations

- Visualizes a single prompt's raw attention only — no head/layer aggregation,
  no generation, no export.
- The model is tuned for prompt inspection, not fast decoding.
- Attention maps are a useful lens, not a complete explanation of model
  behavior.

[Transformers.js]: https://github.com/huggingface/transformers.js
[D3]: https://d3js.org
[`Xenova/gpt2`]: https://huggingface.co/Xenova/gpt2
[`damoncrockett/gpt2-with-attentions-onnx`]: https://huggingface.co/damoncrockett/gpt2-with-attentions-onnx
