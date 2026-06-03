# GPT-2 Attention Visualizer — Design

**Date:** 2026-06-03
**Status:** Approved

## Summary

A single-file, no-build-step HTML app that visualizes GPT-2 self-attention
weights in the browser. The user types a prompt, hits Run, and sees an N×N
attention heatmap where rows and columns are tokens and color intensity
encodes attention weight. Dropdowns select which layer and head to display.

- **Inference:** Transformers.js (`@huggingface/transformers`) running
  `Xenova/gpt2` (base, 12 layers × 12 heads) entirely client-side.
- **Visualization:** D3.js heatmap.
- **Delivery:** One file, `attention-viz.html`. No backend, no build step.
  All dependencies loaded as ES modules from a CDN.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model | `Xenova/gpt2` (base, 12×12) | Canonical GPT-2; gives 12 layers and 12 heads to explore. ~250MB first-load download, cached afterward. |
| Prompt length | No cap; cells shrink to fit | Maximum flexibility; user accepts smaller cells for long prompts. |
| Token labels | Cleaned | Strip the BPE `Ġ` space marker and render a `·` space indicator (e.g. `·dog`) so axis labels read naturally. |
| Color scheme | TBD — chosen visually after Milestone 1 | Look-and-feel choice; deferred until the spike proves the data is real. |

## The Primary Technical Risk

Transformers.js's high-level `pipeline()` API does **not** expose attention
weights. Attentions are only available if **both**:

1. We call the model directly via `AutoModelForCausalLM` and pass
   `output_attentions: true` to the forward call, **and**
2. The ONNX export of `Xenova/gpt2` actually includes attention tensors in its
   output graph.

Whether (2) holds for the stock export is unknown until tested. This is why the
build is gated: **Milestone 1 is a pure feasibility spike that must pass before
any UI is written.**

Possible outcomes:

- **A (best case):** `Xenova/gpt2` already exports attentions. Running with
  `{ output_attentions: true }` returns an `attentions` array of 12 tensors,
  each shaped `[batch=1, heads=12, seq, seq]`. This is the first thing to test.
- **B:** Export lacks attentions but a specific revision/subfolder or option
  enables them. Fallback if A fails.
- **C (worst case):** No usable attention output is available. We would have to
  reconstruct attention manually from hidden states and QK projections — a
  much larger effort. If we land here, **stop and reassess scope** rather than
  silently building it.

## Architecture

Single file `attention-viz.html` containing:

- An `<head>` with minimal CSS.
- A `<body>` with the UI shell (built in Milestone 2).
- One `<script type="module">` that imports dependencies from a CDN:
  - `import { AutoTokenizer, AutoModelForCausalLM, env } from '@huggingface/transformers'`
  - `import * as d3 from 'd3'`

### Components

1. **Model layer** — loads the tokenizer and model once (lazy, on first Run),
   exposes a `runInference(prompt)` function that returns
   `{ tokens: string[], attentions: number[][][][] }` where `attentions` is
   indexed `[layer][head][row][col]`.
2. **State / cache** — holds the most recent inference result in memory so that
   changing the layer/head dropdowns re-renders without re-running inference.
3. **Heatmap renderer (D3)** — given a single `N×N` matrix plus the token list,
   draws the heatmap with axis labels, color-scaled cells, and a hover tooltip.
4. **UI controller** — wires the prompt textarea, Run button, and dropdowns to
   the model layer, cache, and renderer; manages loading/running/error states.

### Data flow

```
Run clicked
  → tokenize(prompt)
  → model forward pass with output_attentions: true
  → cache full [layers][heads][N][N] in memory
  → render(currentLayer, currentHead)

Dropdown changed
  → render(currentLayer, currentHead)   // from cache, no re-inference

New prompt + Run
  → re-run inference, replace cache, render
```

## Milestones (gated)

### Milestone 1 — Spike (console only, no UI)

Hardcode a prompt (e.g. `"The cat sat on the mat"`). Load the tokenizer and
model, run one forward pass with `output_attentions: true`, and `console.log`:

- the token strings and count `N`
- `attentions.length` (expect 12 layers)
- one layer tensor's `.dims` (expect `[1, 12, N, N]`)
- a sample `N×N` slice for layer 0 / head 0, to confirm it is a valid
  attention matrix (causal → lower-triangular, rows summing to ~1)

**Go/no-go gate:** stop and confirm the shape together before writing any UI.

### Milestone 2 — UI (only after the gate passes)

- **Layout:** prompt textarea + Run button at top; layer dropdown (0–11) and
  head dropdown (0–11) below; D3 heatmap fills the remaining space.
- **Heatmap:** two band scales (token index → x/y) and a sequential color scale
  for weight. Axis labels are cleaned tokens. Cells shrink to fit (no length
  cap). Hover tooltip shows `rowToken → colToken` and the weight value.
- **States:** model-loading indicator (first load downloads ~250MB), running
  indicator, and an error display.

## Out of Scope (YAGNI)

- Multi-prompt comparison
- Attention rollup / aggregation across heads or layers
- Saving or exporting images/data
- Text generation
- Anything beyond visualizing one prompt's raw per-head attention
