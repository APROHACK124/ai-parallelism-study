NOTE: this is 100% written by chatGPT and currently not verified

# Phase 2 — Deep Learning Primitives and a Pre-Norm Transformer Block

**Author:** [your name]  
**Date:** [YYYY-MM-DD]  
**Repository:** [local path or URL]  
**Evaluated commit:** [hash]  

---

## 1. Executive Summary

In this phase, I reimplemented, on top of PyTorch but without using its high-level layers inside the main implementation, a minimal set of primitives required to build a functional Transformer block. The goal was not yet to train a full model, but to recover manual fluency in writing the core components, validate them rigorously against PyTorch, and leave the codebase ready for Phase 3.

The implemented components were:

- `Linear`
- `Embedding`
- `LayerNorm`
- `RMSNorm`
- `GELU`
- `log_softmax`
- `cross_entropy`
- `AdamW`
- `CausalSelfAttention`
- `TransformerMLP`
- `PreNormTransformerBlock`

The phase exit criteria were:

- all tests pass,
- forward and backward computations are reasonably consistent with PyTorch,
- the end-to-end sanity run executes without `NaN`s,
- `state_dict` roundtrip works correctly,
- and the implementation decisions, bugs, and validation strategy are documented.

---

## 2. Goal of the Phase

The goal of this phase was to build a small deep learning primitives library and a pre-norm Transformer block that serve as the bridge between a purely mathematical/autograd phase and a later phase focused on tokenization, mini-LLM training, and scaling.

More specifically, this phase aimed to:

1. regain manual control over the most important pieces of a Transformer stack;
2. reduce dependence on high-level APIs for basic components;
3. validate each module against PyTorch;
4. design a reusable and testable codebase;
5. prepare the foundation for real training in Phase 3.

---

## 3. Scope

### Included in this phase

- dense layers and embeddings;
- `LayerNorm` and `RMSNorm`;
- `GELU`;
- core loss functions for autoregressive classification;
- `AdamW`;
- causal self-attention;
- Transformer MLP;
- pre-norm Transformer block;
- unit tests;
- comparisons against PyTorch;
- end-to-end sanity run with save/load and one optimization step.

### Explicitly out of scope

- tokenizer;
- real dataset;
- long training loop;
- pretraining;
- distributed training;
- custom CUDA kernels;
- mixed precision;
- dropout different from `0.0`;
- serious performance benchmarking.

---

### Global Conventions

- All sequence tensors follow the `batch_first` convention.
- The base sequence shape is `[B, T, C]`.
- All modules were validated on CPU first.
- `dropout` was fixed at `0.0` to preserve determinism.
- The main implementation does not use `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.MultiheadAttention`, `F.cross_entropy`, or `torch.optim.AdamW`.
- Those APIs were allowed in the test suite as reference implementations.

### Shape Convention

- `x`: `[B, T, C]`
- `input_ids`: `[B, T]`
- `Linear.weight`: `[out_features, in_features]`
- `Embedding.weight`: `[num_embeddings, embedding_dim]`

---
## 6. Global Design Decisions

### 6.1 Using `nn.Module` and `nn.Parameter`

Phase 2 no longer required implementing the backward pass manually. The chosen strategy was to rely on `torch.autograd` for gradients while implementing the forward logic and explicit parameter handling by hand. This allowed me to focus on the structure of the layers and the correct semantics of shapes, normalization, attention, and optimization.

### 6.2 `batch_first`

I chose `batch_first` across the entire phase to reduce mental overhead and stay consistent with the tensor layout I plan to use later in autoregressive models. This simplified tests, masking, and code readability.

### 6.3 `dropout = 0.0`

I fixed `dropout=0.0` in this phase so that any observed numerical difference came from the implementation itself rather than from stochasticity. Dropout will be reintroduced in a later phase.

### 6.4 Pre-Norm

The final block was implemented as:

\[
x = x + \mathrm{Attn}(\mathrm{Norm}_1(x))
\]

\[
x = x + \mathrm{MLP}(\mathrm{Norm}_2(x))
\]

The motivation was to adopt a more stable variant and one that is closer to what is typically convenient when moving toward deeper stacks.

### 6.5 RMSNorm as the Default Option

Both `LayerNorm` and `RMSNorm` were implemented, but the final block was left configurable with `RMSNorm` as the default option. The reason was that `RMSNorm` is simpler, avoids explicit recentering, and forced me to separate more clearly which part of the behavior comes from mean/variance normalization and which part comes purely from rescaling.

---

## 7. Implementations

## 7.1 Linear

### Definition

I implemented an affine layer:

\[
y = xW^T + b
\]

with:

- `weight.shape = [out_features, in_features]`
- `bias.shape = [out_features]`

### Implementation Decisions

- `weight` and `bias` were declared as `nn.Parameter`;
- the forward pass supports tensors of shape `[..., in_features]`;
- the multiplication was implemented explicitly using basic tensor operations;
- the initialization used was [describe initialization].

### Validation

- forward comparison against `nn.Linear`;
- comparison of gradients for input, weight, and bias;
- `state_dict` save/load test.

### Result

- **max abs diff forward:** [fill in]
- **max abs diff grad input:** [fill in]
- **max abs diff grad weight:** [fill in]
- **status:** [pass/fail]

---

## 7.2 Embedding

### Definition

I implemented an embedding lookup table:

\[
Y_{b,t} = W[\text{input\_ids}_{b,t}]
\]

### Implementation Decisions

- the expected input is a `LongTensor` with shape `[B, T]`;
- the output has shape `[B, T, C]`;
- the lookup is performed via direct indexing into the embedding matrix;
- [indicate whether `padding_idx` was implemented].

### Validation

- forward comparison against `nn.Embedding`;
- backward behavior with repeated indices;
- shape test;
- `padding_idx` test if applicable.

### Result

- **max abs diff forward:** [fill in]
- **max abs diff grad weight:** [fill in]
- **status:** [pass/fail]

---

## 7.3 LayerNorm and RMSNorm

### LayerNorm

Implemented as:

\[
\mu = \frac{1}{d}\sum_i x_i
\]

\[
\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2
\]

\[
y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\]

### RMSNorm

Implemented as:

\[
\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}
\]

\[
y = \gamma \cdot \frac{x}{\mathrm{RMS}(x)}
\]

### Implementation Decisions

- both normalizations operate over the last dimension;
- `LayerNorm` uses `weight` and `bias`;
- `RMSNorm` uses only `weight`;
- `eps = [fill in]`.

### Conceptual Comparison

`LayerNorm` recenters and rescales; `RMSNorm` only rescales. In practice, `RMSNorm` simplified the implementation and made the distinction between normalization and shifting more explicit.

### Validation

- `LayerNorm` vs `nn.LayerNorm`;
- `RMSNorm` vs a reference implementation or `nn.RMSNorm`;
- gradient comparisons;
- shape tests.

### Result

| Module | Forward max abs diff | Grad max abs diff | Status |
|---|---:|---:|---|
| LayerNorm | [ ] | [ ] | [ ] |
| RMSNorm | [ ] | [ ] | [ ] |

---

## 7.4 GELU

### Definition

I implemented `GELU` in the [exact / tanh approximation / both] form.

### Validation

- comparison against `torch.nn.functional.gelu`.

### Result

- **max abs diff:** [fill in]
- **status:** [pass/fail]

---

## 7.5 log_softmax and cross_entropy

### Stable log_softmax

I implemented a numerically stable `log_softmax` by subtracting the row-wise maximum before exponentiation.

### cross_entropy

I implemented `cross_entropy` for integer targets using `log_softmax` plus selection of the correct class.

### Validation

- comparison against `F.log_softmax`;
- comparison against `F.cross_entropy`;
- stability test with large logits.

### Result

| Function | Max abs diff | Stability case | Status |
|---|---:|---|---|
| log_softmax | [ ] | [pass/fail] | [ ] |
| cross_entropy | [ ] | [pass/fail] | [ ] |

---

## 7.6 AdamW

### Definition

I implemented `AdamW` with:

- first moment `exp_avg`,
- second moment `exp_avg_sq`,
- bias correction,
- decoupled weight decay.

### Implementation Decisions

- in-place updates under `torch.no_grad()`;
- explicit step counter;
- minimal support for simple parameter groups [describe whether yes or no].

### Validation

- comparison against `torch.optim.AdamW`;
- one-step test;
- multi-step test;
- `weight_decay=0` test.

### Result

| Case | Final parameter difference | Status |
|---|---:|---|
| 1 step | [ ] | [ ] |
| 5 steps | [ ] | [ ] |
| weight_decay=0 | [ ] | [ ] |

---

## 7.7 CausalSelfAttention

### Definition

Attention was implemented as:

\[
Q, K, V = XW_{QKV}
\]

followed by head splitting and score computation:

\[
A = \frac{QK^T}{\sqrt{d_h}}
\]

Then a causal mask was applied, followed by softmax over the valid positions, and finally a weighted sum with `V`.

### Implementation Decisions

- input and output use shape `[B, T, C]`;
- a single `qkv` projection is used;
- reshaping to `[B, n_heads, T, head_dim]`;
- lower-triangular causal mask;
- `dropout=0.0`.

### Validation

- shape test;
- causality test;
- finite backward test;
- comparison against a reference.

### Causality Test

I verified that changing future tokens does not modify outputs at positions earlier than or equal to `t`.

### Result

- **shape correct:** [pass/fail]
- **causality:** [pass/fail]
- **finite backward:** [pass/fail]
- **max abs diff vs reference:** [fill in]

---

## 7.8 TransformerMLP

### Definition

Implemented as:

\[
\mathrm{MLP}(x) = W_2(\mathrm{GELU}(W_1x + b_1)) + b_2
\]

with expansion to `d_ff` and projection back to `d_model`.

### Validation

- shape test;
- backward test;
- integration test inside the Transformer block.

### Result

- **status:** [pass/fail]

---

## 7.9 PreNormTransformerBlock

### Structure

The final block is:

\[
x = x + \mathrm{Attn}(\mathrm{Norm}_1(x))
\]

\[
x = x + \mathrm{MLP}(\mathrm{Norm}_2(x))
\]

### Implementation Decisions

- support for `norm_type in {"layernorm", "rmsnorm"}`;
- explicit residual connections;
- no dropout;
- `state_dict` compatibility.

### Validation

- shape test;
- finite backward test;
- `state_dict` roundtrip;
- `named_parameters()` verification.

### Result

- **forward:** [pass/fail]
- **backward:** [pass/fail]
- **state_dict roundtrip:** [pass/fail]
- **named_parameters consistent:** [pass/fail]

---

## 8. Global Validation Strategy

The overall validation strategy was:

1. implement each module in isolation;
2. compare it against its PyTorch reference;
3. verify forward;
4. verify backward;
5. integrate everything into a minimal sanity run;
6. test save/load;
7. confirm the absence of `NaN`s and shape mismatches.

### Tolerances Used

- **atol:** [fill in]
- **rtol:** [fill in]

### Aggregate Summary

| Module | Forward | Backward | Save/Load | Status |
|---|---|---|---|---|
| Linear | [ ] | [ ] | [ ] | [ ] |
| Embedding | [ ] | [ ] | N/A | [ ] |
| LayerNorm | [ ] | [ ] | N/A | [ ] |
| RMSNorm | [ ] | [ ] | N/A | [ ] |
| GELU | [ ] | N/A | N/A | [ ] |
| log_softmax | [ ] | [ ] | N/A | [ ] |
| cross_entropy | [ ] | [ ] | N/A | [ ] |
| AdamW | [ ] | N/A | N/A | [ ] |
| CausalSelfAttention | [ ] | [ ] | N/A | [ ] |
| TransformerMLP | [ ] | [ ] | N/A | [ ] |
| PreNormTransformerBlock | [ ] | [ ] | [ ] | [ ] |

---

## 9. End-to-End Sanity Run

I ran a sanity-check script that:

1. instantiates a `PreNormTransformerBlock`,
2. creates a random input,
3. runs a forward pass,
4. builds a simple loss,
5. calls `backward()`,
6. performs one `AdamW` step,
7. saves the `state_dict`,
8. loads it into a new instance,
9. and confirms that the model still produces finite outputs.

### Sanity Run Result

- **input shape:** [fill in]
- **output shape:** [fill in]
- **initial loss:** [fill in]
- **loss after one step:** [fill in]
- **NaNs detected:** [yes/no]
- **save/load correct:** [yes/no]
- **final status:** [pass/fail]

---

## 10. Bugs Found and How I Resolved Them

### Bug 1 — [short title]

**Symptom:** [what I observed]  
**Root cause:** [what was wrong]  
**Fix:** [how I corrected it]  
**Lesson:** [what I learned]

### Bug 2 — [short title]

**Symptom:** [what I observed]  
**Root cause:** [what was wrong]  
**Fix:** [how I corrected it]  
**Lesson:** [what I learned]

### Bug 3 — [short title]

**Symptom:** [what I observed]  
**Root cause:** [what was wrong]  
**Fix:** [how I corrected it]  
**Lesson:** [what I learned]

### 11. AI usage, and daily notes

#### `nn.Embedding`

I didn't know how to implement the padding_idx functionality, because i didn't knew that I could add a "hook" so that when the gradient for a tensor is calculated, some function I wrote can be called. So chatGPT told me about the existence of hooks, and how could I use it here so that at the padding_idx the gradient remains $0$.

Other than that, everything was done by me.

### `nn.LayerNorm`

I had to read the paper on [Layer Normalization](https://arxiv.org/abs/1607.06450) to really understand what I was doing. 

I had to look up how can I calculate the mean and variance on the last dimentions of a tensor. I learned about fundamental Numpy/PyTorch broadcasting.

On AI review, chatGPT says that even if letting `self.bias = None` when `elementwise_affine=False` will behave correctly in the forward function, it's cleaner to still register it as a parameter with `self.register_parameter("bias", None)`.

The same goes for self.weight.

Also, in the tests, it's better to also check the gradient in the input tensors, that is, to put `requires_grad = True` when creating input tensors and checking that the gradients were correctly calculated.

## 11. Final Decision: LayerNorm vs RMSNorm

Both normalization layers were implemented and validated. However, for the final block I chose `[LayerNorm / RMSNorm]` as the default.

### Justification

- [reason 1]
- [reason 2]
- [reason 3]

### Personal Observation

[Write here whether you noticed differences in stability, simplicity, readability, or debugging difficulty.]

---

## 14. Exit Checklist

- [ ] `pytest -q` passes completely
- [ ] `Linear` validated against `nn.Linear`
- [ ] `Embedding` validated against `nn.Embedding`
- [ ] `LayerNorm` validated
- [ ] `RMSNorm` validated
- [ ] `GELU` validated
- [ ] `log_softmax` validated
- [ ] `cross_entropy` validated
- [ ] `AdamW` validated against `torch.optim.AdamW`
- [ ] `CausalSelfAttention` passes the causality test
- [ ] `TransformerMLP` works correctly
- [ ] `PreNormTransformerBlock` runs forward/backward without `NaN`s
- [ ] `state_dict` roundtrip works
- [ ] `run_phase2_sanity.py` completes successfully
- [ ] `compare_to_torch.py` prints reasonable differences
- [ ] the report is fully updated

---

## 15. Conclusion

Phase 2 achieved its goal of turning a still-partial understanding of Transformer components into a concrete, validated, and reusable implementation. The main result of this phase was not only “having working code,” but also recovering manual fluency in the components that will appear repeatedly in autoregressive models, training loops, and systems work.

The correct output of this phase is a small, clear, and reliable base. From here, the natural next step is to assemble a full mini-LLM and train it end to end.

---