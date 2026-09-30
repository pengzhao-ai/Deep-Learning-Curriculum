# Learning Notes — Day 13: Batch Normalization & Learning Rate Scheduling

> **Date:** 2026-09-25
> **Curriculum:** 30-Day Deep Learning Curriculum
> **Topics Covered:** BatchNorm fundamentals (math, placement, why it helps), `model.train()` vs `model.eval()`, Dropout vs BatchNorm at inference, BatchNorm parameter count, a controlled learning-rate experiment

---

## 1. BatchNorm Fundamentals

### 1.1 The core idea

> **BatchNorm re-centers and re-scales the activations entering each layer so that every layer always sees inputs with mean ≈ 0 and variance ≈ 1 — then lets the network *learn back* whatever scale/shift it wants via two trainable parameters.**

It is a normalization layer that **also preserves expressiveness**. That is the whole trick.

### 1.2 The math (readable form)

For a mini-batch of `m` samples, computed **independently for each feature/channel**:

```
Step 1 — batch statistics
    mu_batch  = (1/m) * sum_i( x_i )
    var_batch = (1/m) * sum_i( (x_i - mu_batch)^2 )

Step 2 — normalize
    x_hat = ( x - mu_batch ) / sqrt( var_batch + eps )

Step 3 — scale & shift (learnable)
    y = gamma * x_hat + beta
```

- `eps` (default `1e-5`) prevents divide-by-zero.
- `gamma` (weight) and `beta` (bias) are **trainable, one per feature**, initialized to `gamma = 1, beta = 0`.

**Why `gamma` and `beta` matter:** without them the layer would force every input to be unit-normal, destroying the network's ability to represent non-normalized functions. With `gamma = std` and `beta = mean`, the layer reproduces its input exactly (an **identity escape hatch**). So BatchNorm only *reparameterizes* — it can never make the network less expressive.

### 1.3 Shapes: which statistics are computed over what

| Layer | Input shape | Statistics computed over | Meaning |
|---|---|---|---|
| `nn.BatchNorm1d(C)` | `(N, C)` | the **N** (batch) samples, per feature | one `mu`, `var` per feature |
| `nn.BatchNorm2d(C)` | `(N, C, H, W)` | **N, H, W**, per channel | one `mu`, `var` per feature map |

So `nn.BatchNorm1d(256)` in `MLP_BN` computes **256 means and 256 variances** — one per hidden unit — across the images in the batch.

### 1.4 What it does *between* Linear and ReLU

```
Linear  ->  BatchNorm  ->  ReLU
```

**Why exactly here?** The Linear layer produces a **pre-activation** `z = Wx + b`, whose scale drifts during training. ReLU then sees this raw, unnormalized distribution. BatchNorm fixes two problems:

1. **ReLU is scale-sensitive in a bad way.** If pre-activations drift positive, every unit is "on" and the layer becomes effectively linear; if they drift negative, units die. BatchNorm pins the pre-activation distribution near mean 0, variance 1, so ReLU operates in its useful regime — and `gamma`/`beta` give the network explicit control over *what fraction* of units fire.
2. **The nonlinearity is where distributions distort.** A linear layer preserves mean/variance linearly; ReLU truncates at 0 and shifts the mean. So normalize **after** the layer that produced the drift and **before** the nonlinearity that would amplify it.

A useful rewrite: `BatchNorm -> ReLU` computes `ReLU(gamma*x_hat + beta)` — a ReLU whose *input* is normalized. Since `ReLU(c*u) = c*ReLU(u)` for `c > 0`, controlling the scale of the input to ReLU controls the scale of its output and gradient — this is why BatchNorm tames gradients so well.

### 1.5 Why it helps (mechanisms)

| Mechanism | Explanation |
|---|---|
| **Stabilizes activations ("internal covariate shift")** | Original 2015 motivation (Ioffe & Szegedy): as early layers change, every later layer's input distribution keeps shifting, forcing it to re-adapt. BatchNorm removes the shift. |
| **Smoother loss landscape -> allows bigger LR** | Modern explanation (Santurkar et al., 2018): the covariate-shift story is incomplete; BatchNorm makes the loss surface smoother, so gradient steps are more reliable -> you can use **much larger learning rates**. This is the headline benefit. |
| **Better-conditioned gradients** | Normalization makes each layer **scale-invariant** (scaling incoming weights by a constant leaves the output unchanged) -> robust to weight scale, no exploding/vanishing activations. |
| **Mild regularization** | Each sample is normalized using statistics of *other* samples in its batch -> small noise. Not a replacement for dropout. |
| **Faster convergence** | Reaches a given accuracy in fewer epochs. |

**Optics analogy:** BatchNorm is like **auto-exposure / per-channel auto-gain** in a camera pipeline — every downstream stage always receives a signal normalized into its linear range, regardless of upstream gain drift. It adds no information; it keeps the signal in the regime where the next stage behaves well.

### 1.6 Train vs eval — two *different* formulas (the classic gotcha)

| | Training (`model.train()`) | Evaluation (`model.eval()`) |
|---|---|---|
| Statistics used | **batch** `mu_batch`, `var_batch` of the current mini-batch | **running** statistics accumulated during training |
| Formula | `y = gamma*(x - mu_batch)/sqrt(var_batch+eps) + beta` | `y = gamma*(x - running_mean)/sqrt(running_var+eps) + beta` |
| Running buffers | **updated** each forward pass | **frozen** (not updated) |

During training, PyTorch updates two **buffers** (not parameters) with an exponential moving average, default `momentum = 0.1`:

```
running_mean <- (1 - 0.1) * running_mean + 0.1 * mu_batch
running_var  <- (1 - 0.1) * running_var  + 0.1 * var_batch
```

**Why two formulas?** At inference you may have a single image (batch of 1), where batch statistics are undefined. So you use the population statistics learned during training. This is why you **must call `model.eval()`** before inference.

### 1.7 How many parameters does BatchNorm have?

For `nn.BatchNorm1d(256)` (verified live):

```
num trainable parameters : 512          <- weight (gamma) 256 + bias (beta) 256
named parameters         : [('weight', (256,)), ('bias', (256,))]
named buffers            : [('running_mean', (256,)), ('running_var', (256,)),
                            ('num_batches_tracked', ())]
state_dict entries       : weight, bias, running_mean, running_var, num_batches_tracked
```

So per feature `C`: **2C trainable parameters** (`gamma`, `beta`) **+ 2C buffers** (`running_mean`, `running_var`) **+ 1 scalar** (`num_batches_tracked`).
For `C = 256`: **512 trainable + 512 buffers + 1 scalar**. Negligible cost.

---

## 2. `model.train()` vs `model.eval()` — What They Really Do

### 2.1 They are mode switches, not commands

```python
model.train()   # sets model.training = True  (recursively, on every submodule)
model.eval()    # sets model.training = False (recursively, on every submodule)
```

That is the whole implementation — a boolean flag. **The flag does nothing by itself**; it only matters if a layer's `forward()` reads `self.training` and branches on it. Layers that don't read it behave identically in both modes (verified live):

```
Linear eval output == train output ?  True
```

`train()` / `eval()` are a **contract**: "hey mode-aware layers, behave like this now."

### 2.2 Which layers care about the mode?

| Layer family | Mode-sensitive? | Notes |
|---|---|---|
| **Dropout** (`nn.Dropout`, `Dropout2d`, `AlphaDropout`) | Yes | randomness only during training |
| **BatchNorm** (`BatchNorm1d/2d/3d`) | Yes | batch stats vs. running stats |
| `nn.InstanceNorm` | Yes (only if `track_running_stats=True`) | otherwise agnostic |
| RNN/LSTM built-in dropout | Yes | same idea as Dropout |
| `nn.Linear`, `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.Softmax` | **No** | identical in both modes |

**Dropout and BatchNorm are the two "special" layers** whose behavior flips with the mode.

### 2.3 Dropout: YES, it turns **OFF** in eval

```
train:  output = input * mask / (1 - p)     # random mask + rescale (inverted dropout)
eval:   output = input                      # identity, no mask, no scaling
```

Live demo (input = all ones, `p = 0.5`):

```
TRAIN mode, 3 forward passes:
    [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0]
    [2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0]
    [2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0]
EVAL  mode, 3 forward passes:
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

- **Train:** values are `0.0` (dropped) or `2.0` (kept, because `1/(1-0.5) = 2`), and the pattern differs every call.
- **Eval:** every value is exactly `1.0`, identical every call — dropout is **completely off** (identity).

Expected value check: `E[output] = 0.5*2.0 + 0.5*0.0 = 1.0` = input. The training-time rescaling already made both modes consistent in expectation, so eval needs no compensation.

### 2.4 BatchNorm: NO, it does **not** turn off — it switches formula

Live demo (`BatchNorm1d(3)`, 3-sample batch):

```
TRAIN output:
[[-1.2247,  0.0000, -0.1622],
 [ 0.0000,  1.2247, -1.1355],
 [ 1.2247, -1.2247,  1.2978]]
  per-feature mean of output : [~0, ~0, ~0]
  per-feature var  of output : [1.0, 1.0, 1.0]
  running_mean (buffer)      : [0.5, 0.2, 0.333]      <- updated by this batch
  running_var  (buffer)      : [2.5, 1.3, 1.533]      <- updated by this batch

EVAL output (uses running stats, not batch):
[[ 0.3162,  1.5787,  2.1535],
 [ 2.8460,  3.3328,  0.5384],
 [ 5.3759, -0.1754,  4.5762]]
  eval output per-feature mean: [2.846, 1.579, 2.423]  <- NOT 0 anymore!
```

- In **train**, output is forced to per-feature mean ≈ 0, variance ≈ 1.
- The **running buffers updated** to this batch's statistics.
- In **eval**, output is **no longer zero-mean** — it used the *frozen running* stats.

Batch-size edge case (proves the difference):
```
EVAL with a single sample (batch=1) works
TRAIN with a single sample FAILS: ValueError ->
    Expected more than 1 value per channel when training, got input size torch.Size([1, 3])
```

### 2.5 What BatchNorm does at inference (your question, precisely)

**Yes — BatchNorm stays in the model during inference. It does not disappear.** What it does at inference is a **fixed per-feature affine transform**:

```
y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
```

This can be regrouped into a plain linear operation `y = A*x + B`, with **fixed** per-feature coefficients:

```
A = gamma / sqrt(running_var + eps)                      # fixed scale
B = beta - gamma * running_mean / sqrt(running_var + eps) # fixed shift
```

Verified live:
```
eval output == manual  y = A*x + B  ?  True
running_mean frozen during eval forward ?  True
running_var  frozen during eval forward ?  True
running_mean changes during TRAIN forward ?  True
```

**"Frozen" means:** the buffers `running_mean` and `running_var` are **not updated** during an eval forward pass (they were built during training and are then held fixed). `gamma` and `beta` are ordinary parameters — they are only changed by the optimizer during training anyway, never by a forward pass.

So during inference BatchNorm is a **deterministic, fixed scaling-and-shifting** — no batch statistics, no randomness. Because `y = A*x + B` is itself affine, this transform can be mathematically **fused into the preceding `Linear`/`Conv` weight and bias** at deployment (a common inference optimization).

---

## 3. Dropout vs BatchNorm at Inference (summary table)

| Layer | `train()` behavior | `eval()` behavior | Does it "turn off"? |
|---|---|---|---|
| **Dropout** | random mask + `1/(1-p)` scaling | identity (pass-through) | **Yes, effectively off** |
| **BatchNorm** | batch stats + update running buffers | uses frozen running stats | **No — switches to running stats** |
| `Linear`/`Conv`/`ReLU`/`Pool` | same | same | N/A (mode-agnostic) |

**Evaluation recipe:**
```python
model.eval()
with torch.no_grad():        # or @torch.no_grad() decorator
    ... forward only ...
model.train()                # restore before the next epoch
```

Note: `model.eval()` and `torch.no_grad()` are **independent** — `eval()` controls *layer behavior*, `no_grad()` controls *gradient tracking*. You want both for evaluation.

---

## 4. Does BatchNorm Actually Help? (Experiment)

**Observation:** in the Day 13 notebook (Adam, `lr=1e-3`, 15 epochs) BatchNorm was only *slightly* better.

**Controlled experiment** (`MLP_NoBN` vs `MLP_BN`, same architecture/seed; 12k train subset, 10 epochs, Adam):

| Setting | No BN | With BN |
|---|---|---|
| `lr = 1e-3` — val acc | 96.23% | **97.00%** |
| `lr = 1e-3` — test acc | 96.37% | **96.84%** |
| `lr = 0.1` — val acc | **18.40%** (collapsed) | **95.70%** |
| `lr = 0.1` — test acc | 17.72% | 95.73% |

### Why "slightly better" at the default setting

1. **Adam already does per-parameter adaptive scaling**, which conceptually overlaps with BatchNorm's robustness-to-scale benefit — so at a safe LR the gap stays small. With plain **SGD**, BatchNorm's advantage is much bigger.
2. **`lr=1e-3` is already safe and MNIST is easy** — BatchNorm's main job is *enabling larger LRs / stabilizing optimization*, so there's little to fix.

### The real lesson is the `lr = 0.1` row (Exercise 1)

- **No BN at `lr=0.1` → 18.4%**: unnormalized pre-activations make updates too aggressive; training oscillates and fails.
- **With BN at `lr=0.1` → 95.7%**: normalization keeps every layer well-scaled, so the same large step size now works.

> **BatchNorm's actual value: it buys stability at high learning rates (faster convergence), not necessarily a higher ceiling on an already-easy task.**

---

## 5. Learning Rate Scheduling

> *(Full notebook work-through — constant vs `StepLR` vs `CosineAnnealingLR` — to be added as we cover sections 2 and 3.)*

### 5.1 Adaptive optimizers vs LR schedulers (an important distinction)

Both are called "adaptive learning rate", but they adapt **different things**:

| Aspect | Adaptive optimizer (Adam, RMSprop, Adagrad) | LR scheduler (StepLR, Cosine, ReduceLROnPlateau) |
|---|---|---|
| Trigger | gradient statistics (`m`, `v`) | a plan: epoch/step count, or a monitored metric |
| Granularity | **per parameter** | **global** (one scalar per param group) |
| What it changes | the effective step *per parameter* | the base/global `lr` constant |
| Looks at gradients? | yes | no (except `ReduceLROnPlateau`, which looks at a metric) |
| Time behavior | fluctuates with recent gradients; no plan | planned trajectory |
| Lives in PyTorch at | `optimizer.state[w]['exp_avg_sq']`, `exp_avg` | `optimizer.param_groups[i]['lr']` |

They **compose multiplicatively**:

```
effective step  ≈  lr_t (scheduler)  ×  m_hat / sqrt(v_hat) (optimizer)
```

Adam's update, for contrast:

```
m_t = b1*m_{t-1} + (1-b1)*g_t
v_t = b2*v_{t-1} + (1-b2)*g_t^2
theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```

The `lr` is still a single global scalar; `m/sqrt(v)` supplies the per-parameter scaling, and it keeps the step magnitude ≈ `lr` (so `lr` is like the max step size / envelope).

**Demo evidence:**
```
Adam, gradients = [1, 100] each step:
    global lr stays 1e-2
    exp_avg_sq = [0.001, 10.0]      <- very different per parameter
    both weights move by ~0.01      <- Adam normalized the 100x gradient away
Cosine schedule (SGD, lr=0.1, T_max=5):
    0.10000 -> 0.09045 -> 0.06545 -> 0.03455 -> 0.00955   <- global decay, gradient-independent
Adam + Cosine:
    base lr decays (0.0085 -> 0.0050 -> 0.0015 -> 0.0) while Adam still scales per parameter
```

**Why schedulers still matter with Adam:** adaptive optimizers handle conditioning (different params need different scales) but do **not** anneal the *global* step toward zero. Schedulers (a) lower the gradient-noise floor near convergence, (b) damp oscillation, (c) act as a mild regularizer, (d) allow a large early LR. Empirically cosine decay improves final accuracy even for Adam/AdamW — which is why Day 25's ViT uses AdamW + warmup + cosine.

**Adagrad footnote:** its accumulators grow monotonically, so its effective LR *self-decays* over training — a built-in decay that overlaps with schedulers, but with no plan/control.

**Practical gotchas:**
- Call `optimizer.step()` **before** `scheduler.step()` (otherwise PyTorch skips the first LR value and warns).
- When checkpointing, save `scheduler.state_dict()` too (see Day 14).
- `ReduceLROnPlateau.step(val_loss)` takes a metric; other schedulers take no argument.

---

## 6. Questions for Self-Check

1. In one sentence, what does BatchNorm do to the activations entering a layer?
2. Write the forward formula for BatchNorm in training mode. What do `gamma` and `beta` do?
3. Why does BatchNorm go *between* `Linear` and `ReLU` and not elsewhere?
4. What are the two different formulas BatchNorm uses in `train()` vs `eval()`?
5. What exactly does `model.train()` / `model.eval()` change inside the model?
6. Does Dropout turn off in eval? Does BatchNorm turn off in eval? Explain the difference.
7. What does "frozen running statistics" mean — which tensors are frozen, and who updates them during training?
8. How many trainable parameters does `nn.BatchNorm1d(C)` have? How many buffers?
9. Why does BatchNorm's benefit look tiny at `lr=1e-3` with Adam but huge at `lr=0.1`?
10. Why must you call `model.eval()` before inference if you use BatchNorm or Dropout?

---

## 7. Next Steps

| Day | Topic |
|-----|-------|
| **Day 13 (cont.)** | Learning Rate Scheduling — `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau` |
| **Day 14** | Training Pipeline Best Practices (checkpointing, reproducibility) |
| **Day 15** | Working with Image Data (CIFAR-10, data augmentation) |
| **Day 16** | The Convolution Operation ⚡ |

---

*Last updated: 2026-09-25*
