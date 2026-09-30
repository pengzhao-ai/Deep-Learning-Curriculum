# Learning Notes — Day 17: Build Your First CNN

> **Date:** 2026-09-30
> **Curriculum:** 30-Day Deep Learning Curriculum — Phase 4: CNNs (Days 16–21)
> **Notebook:** `day17/day17_build_first_cnn.ipynb`
> **Worked solutions (companion):** `day17/day17_build_first_cnn_with_exercises.ipynb`
> **Topics Covered:** the CNN block blueprint (Conv→BN→ReLU→Pool), why the stack shrinks resolution while growing channels (receptive fields, pooling, invariance), the classifier head & the Flatten/batch-dimension question, a `BatchNorm2d` statistics recap, learned filters vs feature maps (weights vs activations), the completed exercises, and why an MLP handles MNIST but not CIFAR-10.

---

## 1. The CNN Architecture Blueprint

A CNN splits into two conceptual halves:

1. **Feature extractor** — stacked blocks of `Conv2d → BatchNorm2d → ReLU → MaxPool2d`.
2. **Classifier head** — `Flatten → Linear → ReLU → Dropout → Linear`.

```
Input (3, 32, 32)
 └─ Block 1: Conv(3→32, 3×3, pad 1) → BN(32) → ReLU → MaxPool(2)    → (32, 16, 16)
 └─ Block 2: Conv(32→64, 3×3, pad 1) → BN(64) → ReLU → MaxPool(2)   → (64, 8, 8)
 └─ Block 3: Conv(64→128, 3×3, pad 1) → BN(128) → ReLU → MaxPool(2) → (128, 4, 4)
 Classifier: Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)
```

**Verified shape trace** (dummy input `(1, 3, 32, 32)` through `SimpleCNN`):

| Stage | Shape | Note |
|---|---|---|
| Input | `(1, 3, 32, 32)` | 1 image, 3 channels |
| Block 1 (after pool) | `(1, 32, 16, 16)` | 32 filters; H,W halved |
| Block 2 (after pool) | `(1, 64, 8, 8)` | 64 filters; halved again |
| Block 3 (after pool) | `(1, 128, 4, 4)` | 128 filters; halved again |
| Flatten | `(1, 2048)` | 128 · 4 · 4 |
| Classifier | `(1, 10)` | one logit per class |

**Parameter count:** `620,810`. The first `Linear(2048→256)` alone is `524,544` (**~85%** of the whole network) — the conv layers are cheap thanks to **weight sharing** (Day 16).

---

## 2. Why the Stack Is Built This Way (the big "why")

**Core principle: trade _where_ for _what_.** A single pixel is nearly meaningless; the class label is a high-level concept. The stack climbs a ladder from *low-level / high-resolution* to *high-level / low-resolution*:

```
spatial resolution   32×32  →  16×16  →  8×8  →  4×4     (↓ location precision)
channel count          3    →   32    →   64   →  128     (↑ feature richness)
what it means        pixels     edges    parts   objects
```

**2.1 Conv layers compound the receptive field.** Each conv sees a small patch, but the next layer sees a patch of the previous map — so the effective view grows *without any huge kernel*. Computed for this exact stack (RF = receptive field in input pixels, jump = spacing between neighbours):

| Stage | Operation | RF | Jump |
|---|---|---|---|
| input | — | 1 | 1 |
| Block 1 | Conv 3×3 s1 | 3 | 1 |
| | MaxPool 2×2 s2 | 4 | 2 |
| Block 2 | Conv 3×3 s1 | 8 | 2 |
| | MaxPool 2×2 s2 | 10 | 4 |
| Block 3 | Conv 3×3 s1 | 18 | 4 |
| | MaxPool 2×2 s2 | **22** | 8 |

A neuron in the final 4×4 map "sees" ~**22×22** of the 32×32 image — nearly the whole image — using only 3×3 kernels. **That is the magic of depth.**

**2.2 `padding=1` keeps size flat; `MaxPool` does all the shrinking.** Every conv is "same" padding (Day 16 rule) so 32→32, and only the pool halves it. One job per operation: *conv transforms features, pool downsamples*.

**2.3 Channels = the width of the feature vocabulary.** `3→32→64→128` = 32 pattern detectors → 64 combinations of those → 128 combinations of those (`out_channels` is a *width*, not network depth — Day 16).

**2.4 Why remove spatial detail at all?** (a) **Compute** — a 32×32×128 conv is ~64× the cost of one at 4×4×128. (b) **Translation invariance** — pooling tolerates small shifts ("wing moved 2 px → same answer"). (c) **Generalization** — smaller maps = fewer effective parameters = less overfitting (Day 12).

**Where 4×4 comes from:** three stride-2 pools: `32 / 2³ = 4`. The flatten size is the contract: `128 × 4 × 4 = 2048`. **Coupling trap:** add or remove a pool, and the head's `Linear` in_features must change.

> **🔬 Optics analogy:** read the stack as a **learned image pyramid** (Gaussian/Laplacian). Each pool = low-pass + downsample; the convs encode the residual detail. Early layers = high-spatial-frequency channels; late layers = low-frequency "gist" channels. The network *learns the pyramid* instead of you designing it.

---

## 3. The Classifier Head — Flatten and the Batch Dimension

The head turns one image's **2048-number semantic descriptor** into **10 class logits**.

```python
self.classifier = nn.Sequential(
    nn.Flatten(),                 # (N,128,4,4) -> (N,2048)   start_dim=1
    nn.Linear(128 * 4 * 4, 256),  # 2048 -> 256
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, 10),           # 256 -> 10 logits
)
```

**Does `Flatten` collapse the batch dimension? No.**

- `nn.Flatten()` defaults to `start_dim=1`: it flattens **everything except dim 0**. So `(N,128,4,4) → (N, 2048)`; the batch dim is untouched.
- Verified: `(8,128,4,4) → (8,2048) → (8,10)`. Output is always `(batch_size, num_classes)` — exactly like the MLP. ✅

**⚠️ The `128` trap.** In `nn.Linear(128 * 4 * 4, 256)` the leading `128` is the **number of channels** (from `Conv2d(64,128,...)`), **not the batch size**. It just *coincidentally* equals `BATCH_SIZE = 128`. `128*4*4 = channels × height × width = 2048` — the batch dim is never part of a `Linear`'s `in_features`.

**MLP vs CNN — the batch behaviour is identical; what differs is _when_ flattening happens:**

| | MLP (Day 11) | CNN (Day 17) |
|---|---|---|
| Input layout | already flattened to `(N, 3072)` by the transform | kept spatial `(N,3,32,32)` through the conv stack |
| Flatten happens | before the model / in data pipeline | **inside `forward`** via `nn.Flatten()` |
| `nn.Linear` contract | acts on the **last** dim, preserves dim 0 | same |
| Output | `(N, 10)` | `(N, 10)` ✅ |

**One-liner:** the conv stack converts one `3×32×32` image → one `2048`-length descriptor; the classifier maps that descriptor → 10 scores; the batch just means "do it for N images, row by row."

---

## 4. BatchNorm2d Recap

`BatchNorm2d` is the spatial version of Day 13's `BatchNorm1d`: **normalize each channel independently**, using statistics computed over **(N, H, W)**.

For input `(N, C, H, W)` and each channel `c`:

$$y_c = \gamma_c \cdot \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c$$

- `μ_c`, `σ_c²` = mean/var over **(N, H, W)** of channel `c` during training.
- `γ_c`, `β_c` = **learnable**, one per channel (init 1 and 0) — the "identity escape hatch."

| | `BatchNorm1d` | `BatchNorm2d` |
|---|---|---|
| Input | `(N, C)` | `(N, C, H, W)` |
| Normalize per | each of the C features | each of the C channels |
| Statistics over | **N** | **N, H, W** |
| Learnable params | `γ, β` shape `(C,)` | `γ, β` shape `(C,)` |
| Running stats | `(C,)` | `(C,)` |

**Verified:** input `(4,3,5,5)` → `bn.running_mean.shape == (3,)`, matching `z.mean(dim=(0,2,3))`.

**Train vs eval (a classic gotcha):**
- `model.train()`: normalize with the **current batch's** stats **and** update running buffers as an EMA:
  `running = (1 − momentum)·running + momentum·batch_stat`, default `momentum = 0.1`.
- `model.eval()`: use the **accumulated running stats** (required for single-image inference).

**Verified EMA:** channel-0 batch mean was `−0.1216` and the fresh model's `running_mean[0]` became `−0.0122` — exactly `0.1 × (−0.1216)`. So don't be surprised that `running_mean ≠ batch_mean` after one pass.

**Why per-channel?** Each filter produces its own feature map with its own range/distribution. Normalizing each channel separately puts all maps on a comparable scale before the next conv mixes them.

> **🔬 Optics:** per-channel auto-exposure / per-band gain calibration — normalize each spectral band by its own noise floor and gain before fusing.


---

## 5. Learned Filters — Weights vs Activations (Notebook §4)

**Section 4 plots the layer's _weights_, not its feature maps.** This is the single most misread cell in Day 17.

| Tensor | Shape | What it is | Batch-dependent? |
|---|---|---|---|
| `x` (input) | `(N, 3, 32, 32)` | the images | yes |
| `y = conv1(x)` (activations / **feature maps**) | `(N, 32, 32, 32)` | 32 response maps per image | yes |
| `conv1.weight` (the **filters**) | `(32, 3, 3, 3)` | the layer's 32 learned kernels | **no** |
| `conv1.bias` | `(32,)` | one bias per output channel | no |

```python
filters = model.features[0].weight.data.cpu()   # (32, 3, 3, 3)
f = filters[i]                                  # (3, 3, 3) = [C_in, kH, kW]
f = (f - f.min()) / (f.max() - f.min())         # display-only min-max normalize
ax.imshow(f.permute(1, 2, 0))                   # [C,H,W] -> [H,W,C] for imshow
```

**Each of the 32 output channels = 3 stacked 3×3 kernels (one per R, G, B input), summed:**

$$\text{out}_c(x,y) = \sum_{ic \in \{R,G,B\}} \sum_{i,j} \text{in}_{ic}(x\!+\!i, y\!+\!j)\, W_c[ic,i,j] + b_c$$

**Verified against `nn.Conv2d`** (manual: `−0.50426900`, PyTorch: `−0.50426894`, `match: True`).

- `permute(1,2,0)` makes a **3×3 RGB thumbnail**: the red of a pixel = weight on the R input, etc.
- Min-max normalization is **purely cosmetic** (weights are tiny and signed); it does not change the model.

**What the section is _for_:** it closes the Day 16 loop — §2 was *hand-designed* filters, §6 was *random* filters, and here the filters are **learned from data**. After training they look like oriented edges / colour-opponent blobs, i.e. the network **discovered edge detectors on its own** because edges help classify CIFAR-10.

**Why it matters:** (1) a sanity check — if filters look like noise, learning failed; (2) interpretability; (3) seeds **Day 19** (deeper visualization / CAM / dead neurons) and **Day 20** (these generic edge filters are *why* transfer learning works); (4) **optics**: a filter *is* a learned **PSF**.

**Why only layer 1?** A filter is only `imshow`-able if its **input lives in a viewable space (RGB)**. Layer 2's input is 32 abstract feature maps `(32,16,16)` — there is no natural colour to render. For deeper layers you visualize **activations** instead (that's Exercise 3 here, and Day 19 §2).

> ⚠️ Because inputs are normalized, the colours are only relative — don't over-read absolute hues.

---

## 6. Training Recipe (as used)

| Choice | Value | Why |
|---|---|---|
| Loss | `CrossEntropyLoss` | standard multi-class classification |
| Optimizer | `Adam(lr=1e-3, weight_decay=1e-4)` | good default (Day 8) |
| Scheduler | `CosineAnnealingLR(T_max=30)` | smooth LR decay to fine-tune near convergence (Day 13) |
| Epochs | 30 (original) | enough for ~80%+ on CIFAR-10 |
| Regularization | `Dropout(0.5)` + weight decay + augmentation (Day 15) | combat overfitting (Day 12) |
| Data | CIFAR-10, 45k/5k train/val split (90/10), 10k test touched once | Day 15 |

**Reading the curves:** loss decreasing = learning; train≈val = no overfit; val flattening = the architecture has learned all it can; val loss rising while train falls = **overfitting**.

**This run (companion, 6 epochs):** test accuracy **73.36%** — *not* the 80%+ of a 30-epoch run. Weakest classes: **cat 47.9%**, **bird 59.4%**.


---

## 7. MLP vs CNN — Why MNIST Works but CIFAR-10 Needs a CNN

**The puzzle:** an MLP (Day 11) flattened MNIST and still hit ~97–98%. But the same MLP struggles on CIFAR-10 while a CNN wins. Why?

### 7.1 The two priors an MLP is missing

A conv layer hard-codes two assumptions about images:

1. **Locality** — nearby pixels are related; useful patterns (edges, corners) are *local*. Each output looks only at a small neighbourhood.
2. **Translation equivariance + weight sharing** — the *same* filter is reused at every position, so "an edge is an edge wherever it appears."

An MLP has **neither**: it flattens the image and gives **every pixel position its own private weight**. It's a fixed lookup grid — a pattern learned at one location is not recognized at another.

### 7.2 Why MNIST tolerates this

MNIST is almost a *best case* for a position-locked model:

- **Everything is centered and size-normalized.** Digits are preprocessed into a fixed 28×28 box, so "the position of a stroke" is a **reliable feature**: a horizontal bar near the top = a "7", two loops in the right places = an "8". Per-position weights *can* encode this.
- **Single object, no clutter.** The whole frame is the digit; there's no background to confuse.
- **Shape is enough.** Grayscale, high contrast, no colour/texture needed.
- **Tiny pose variation.** Digits are mostly upright and similarly scaled, so an MLP doesn't need to be shift- or scale-invariant.

Translation invariance is *rarely exercised*, so the missing prior costs little.

### 7.3 Why CIFAR-10 breaks it

CIFAR-10 is the opposite regime:

- **Objects appear anywhere, at any scale and pose**, with **background clutter**. The same "dog" can be shifted, rotated, zoomed, and differently coloured — producing a **completely different flat vector** for the *same* class.
- An MLP must therefore **relearn every pattern at every location** (and every shift), which is hopeless with 50k images. It can't generalize across translations because each position has independent weights.
- **Colour and texture matter** (a "frog" vs "deer" is partly colour/texture), and pixels individually are meaningless.
- **Parameter inefficiency:** the first dense layer (`3072 × hidden`) doesn't share anything, so it both overfits and fails to exploit that an edge detector should be reused everywhere.

### 7.4 The intuition in one picture

- **MLP = a fixed photodiode array, each pixel wired to its own detector.** It works only if the pattern always lands in the same place (MNIST's centered digits).
- **CNN = a sliding stencil / matched filter that scans the whole image.** It detects the pattern *wherever* it appears — exactly the invariant detection your optics experience calls a **PSF / matched filter**.

```
MNIST:  "the stroke is at (x,y)"      → position IS a good feature → MLP OK
CIFAR:  "a wheel-ish texture exists somewhere" → need positional invariance → CNN required
```

### 7.5 Evidence from our exercise (Exercise 2)

Equal parameter budget (~620k), same data, same 3 epochs, same optimizer:

| Model | Params | Test acc |
|---|---|---|
| **SimpleCNN** | 620,810 | **49.76%** |
| MLP | 616,610 | 39.24% |

Same budget, *only architecture* differs → CNN wins by **~10 points**, and the gap widens with full training.

### 7.6 Nuance / caveat

An MLP is not *incapable* on CIFAR-10 — with enormous data, compute, and heavy augmentation it can learn *approximate* invariance from examples. But the CNN **hard-codes the correct prior**, so it reaches good accuracy far more efficiently (this is the "inductive bias" story that Day 20's transfer learning exploits).

> **🔬 Optics takeaway:** data augmentation *simulates* invariance; convolution *builds it in*. Weight sharing is the architectural equivalent of assuming a **shift-invariant (linear) system** — one impulse response reused everywhere.


---

## 8. Worked Exercises (Companion Notebook)

Fully worked, executed solutions live in **`day17/day17_build_first_cnn_with_exercises.ipynb`**. Budget: main model 6 epochs on full data; exercise models 3 epochs on an 8k subset (so the notebook runs in ~1 min). Numbers differ at full budget, but the *relative* lessons hold.

| # | Exercise | Change | Params | Test acc | Takeaway |
|---|---|---|---|---|---|
| — | Main `SimpleCNN` | baseline (6 ep, full data) | 620,810 | **73.36%** | reference |
| 1 | `DeeperCNN` (4th block 128→256) | flatten 2048 → **1024** | 654,346 (+5%) | **54.12%** | +4 pts for +5% params, but `2×2` map = diminishing returns |
| 2 | CNN vs `MLP` | equal params, same budget | 620,810 vs 616,610 | **49.76% vs 39.24%** | architecture (inductive bias) beats equal parameter count |
| 3 | Feature-map visualization | forward hooks on pools 3/7/11 | — | — | `32×16×16 → 64×8×8 → 128×4×4` (resolution↓, channels↑) |
| 4 | `GAPCNN` (`AdaptiveAvgPool2d(1)`) | head 527k → 1.3k | 94,986 (**−84.7%**) | 41.62% | strips the parameter-heavy FC head; small cost at low budget |

**Exercise 1 — exact code insight (verified):**
```python
nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
...
nn.Linear(256 * 2 * 2, 256)   # 2×2 now, not 4×4  →  the coupling trap
```

**Exercise 3 — the hook pattern:**
```python
def make_hook(name):
    def hook(module, inputs, output):
        activations[name] = output.detach().cpu()
    return hook
handles = [model.features[i].register_forward_hook(make_hook(...)) for i in (3, 7, 11)]
# ... run forward pass ...
for h in handles: h.remove()          # ALWAYS remove hooks, or they leak
```
Row 1 maps look image-like (edges/outlines); row 3 `4×4` maps are abstract — you read *which channels fire*, not objects. And note: §4 plotted **weights**, this plots **activations** — you need both views.

**Exercise 4 — why GAP is used in real nets:** (1) near-zero head params → less overfitting; (2) **input-size agnostic** (any `H×W` → `C`, e.g. ResNet's final layer); (3) the bridge to **CAM** in Day 19 — pooled per-channel activations *are* the class activation weights.

---

## 9. Optics Connections (Day 17)

| DL concept | Optics / imaging analogue |
|---|---|
| A conv filter | a **PSF** (impulse response) — here **learned**, not designed |
| Weight sharing across positions | **linear shift-invariant (LSI)** assumption: one PSF convolved everywhere |
| `Conv → Pool` stack | **image pyramid**: low-pass + downsample, convs encode residual detail |
| `BatchNorm2d` per channel | **per-band gain / auto-exposure** calibration |
| MLP vs CNN | fixed position-locked detector array vs a **matched filter that scans** |
| Data augmentation | *simulating* invariance vs convolution *building it in* |


---

## 10. Practical Notes — Data & Companion Notebook

- **Data (no re-download):** every CNN-arc notebook calls `CIFAR10(root="./data", download=True, ...)`, where `./data` resolves *relative to its own folder*. `day17/data` is a **symlink → `../day15/data`**, so CIFAR-10 loads fully offline (`download=True` is a safe no-op). Same pattern for `day18…day21`, `day21_extra`; `day17b_autoencoder_intro/data/` also links MNIST → `day14/data/MNIST`. These `data` links are `.gitignore`d.
- **Companion notebook budget:** `day17_build_first_cnn_with_exercises.ipynb` uses `NUM_EPOCHS = 6` (main) and `EPOCHS = 3` on an 8k subset (exercises) so the whole thing runs in ~1 min. **To reproduce the original/final numbers, set `NUM_EPOCHS = 30` and exercise `EPOCHS ≈ 10` (or use full data).** The ⏱️ speed note appears in the notebook.
- **Device:** `mps` on this Mac (`torch 2.12.0`, `torchvision 0.27.0`).
- **Gotchas:** never overwrite `model` when building exercise variants; remove forward hooks after use; a `Linear`'s `in_features` must equal `channels × H × W` after the last pool.

---

## 11. Questions for Self-Check

1. Write the output-size formula and state what padding keeps 32×32 with a 3×3 kernel.
2. Why does the CNN shrink `32→16→8→4` while channels grow `3→32→64→128`? What are the three reasons to downsample?
3. Compute the receptive field of a neuron in the final `4×4` map for this stack.
4. Does `nn.Flatten()` collapse the batch dimension? What does `start_dim=1` mean?
5. In `nn.Linear(128 * 4 * 4, 256)`, is `128` the batch size or the channel count? What is the flatten size?
6. If you add a 4th pool to `SimpleCNN`, what must change in the head, and why?
7. `BatchNorm2d(32)` on `(N,32,H,W)`: over which axes are mean/var computed, and what are their shapes?
8. Why does `running_mean ≠ batch_mean` after a single forward pass? What is the EMA formula?
9. In §4, what tensor is plotted — the weights `(32,3,3,3)` or the feature maps? What does each of the 32 filters contain?
10. Why can't you `imshow` layer 2's filters directly, and what do you visualize instead?
11. Intuitively, why can an MLP classify centered MNIST digits but fail on CIFAR-10 at equal parameter count?
12. GAP head: how many parameters does it save, and name two reasons real networks use it.

---

## 12. Next Steps

| Day | Topic |
|-----|-------|
| **Day 16** ✅ | The Convolution Operation |
| **Day 17** ✅ | Build Your First CNN — **complete** |
| **day17b** | Autoencoder Bridge (`day17b_autoencoder_intro`) — light ~1 h session: swap head+loss for image reconstruction, `ConvTranspose2d`, latent space, MSE vs L1 |
| **Day 18** | CNN Architectures — VGG blocks & ResNet skip connections |
| **Day 19** | CNN Deep Dive — feature visualization, CAM, dead neurons |
| **Day 20** | Transfer Learning & Fine-Tuning |
| **Day 21** | Build a CNN from Scratch — Complete Project (>90% CIFAR-10) |
| **day21_extra** | CNN Restoration — U-Net denoising / deblurring |

**Reminder — CNN arc ordering:** Day 16 → 17 → **17b** → 18 → 19 → 20 → 21 → **21_extra**.
Do **17b right after Day 17** (it reuses conv/pool mechanics while fresh and teaches `ConvTranspose2d`, needed by the U-Net later).

**Forward hooks:** Exercise 3's forward-hook pattern (and §4's weights-vs-activations distinction) is the direct setup for **Day 19**.

---

*Last updated: 2026-09-30*

