# Learning Notes — Day 18: CNN Architectures — VGG Blocks & ResNet Skip Connections

> **Date:** 2026-10-02
> **Curriculum:** 30-Day Deep Learning Curriculum — Phase 4: CNNs (Days 16–21)
> **Notebook:** `day18/day18_cnn_architectures.ipynb`
> **Prerequisite:** Day 17 (Build Your First CNN) and day17b (Autoencoder Intro — introduces skip connections conceptually)
> **Topics Covered:** the two foundational CNN *design patterns* (VGG-style stacked 3×3 blocks, and ResNet residual/skip connections); the receptive-field + parameter argument for small stacked filters; the **degradation problem** and why deep plain networks fail; **residual learning** `output = F(x) + x`; the **projection shortcut** (1×1 conv) for dimension-changing blocks; the `SimpleResNet` stage pattern (stride-2 + doubled channels, then identity blocks); Global Average Pooling as the head; a **verified** VGG-vs-ResNet parameter and shape comparison; and the recap/preview that frames **why Days 18 → 19 → 20 → 21** must be studied in order.

---

## 📝 Conversation Q&A Log (Day 18 discussion)

> **Purpose:** a record of the *questions raised during the Day 18 session* and the answers given, so the discussion is not lost (distinct from the reference material below).

### Round 1 — Kick-off

**Q.** Recap the Day 18–21 CNN journey: what VGG and ResNet *really* are; why 18/19/20 are needed before the Day-21 wrap-up; and how they connect.
**A.** Captured in §1 (the arc) and §6 (the ordering rationale + the four threads).

### Round 2 — The VGG block & how a `nn.Module` is written

**Context (the "start from fundamentals" request).** A PyTorch model is **just a class**: `__init__` **declares the parts** (where parameters get registered) and `forward` **wires the data flow**; you call `model(x)` and `nn.Module.__call__` runs `forward`. Modules nest — assigning a sub-module to `self.x` *registers* it, so `model.parameters()` finds everything. `nn.Sequential(...)` is a Module whose `forward` applies its children in order (the straight-line case). Three build styles: layers inline in `forward`; `nn.Sequential`; or **split attributes + a custom `forward`** (which Day 17 already used for `features`/`classifier`). The one thing Day 18 adds is **factoring the repeated block into a function**.

**Q1. What does VGG stand for?**
**A.** **V**isual **G**eometry **G**roup — the Oxford computer-vision group; the 2014 paper *"Very Deep Convolutional Networks for Large-Scale Image Recognition"* (Simonyan & Zisserman). The model is named after the group; "VGG16/19" = the number of weight layers. Best remembered as a **design pattern**: *stack many small 3×3 convs in uniform stages, pool between stages, finish with a classifier.*

**Q2. In `vgg_block`, `num_convs=2` — does it run `layers.extend` twice? Is `layers` a dict? Should the 2nd conv's output channel be different?**
**A.**
- `num_convs=2` → `range(2)` → `i = 0, 1` → the `(Conv, BN, ReLU)` trio is appended **twice**, then one `MaxPool` (verified: the resulting `Sequential` has 7 children).
- **`layers` is a plain Python `list`** (`layers = []`), **not** a dict. It grows with **`.extend(...)`, not `.append(...)`**: `extend` unpacks the 3-item list so each layer becomes its own element (after two loops: 6 elements, verified).
- **No — the output channel stays the same** (`out_channels`, here 64) for **both** convs. Only the *input* channel of the **first** conv differs, handled by `in_channels if i == 0 else out_channels` (i=0: 3→64; i=1: 64→64). Rationale: after the first conv the tensor *already* has 64 channels, so the next conv must consume 64. The block is a uniform "stage": introduce a channel width once, then refine within it. Channel count only jumps **between** stages (`vgg_block(3,64)`, `(64,128)`, `(128,256)`); `MaxPool` does the spatial halving.

**Q3. What does `*` mean in `nn.Sequential(*layers)`?**
**A.** Python's **iterable-unpacking** operator: `nn.Sequential(*layers)` ⟺ `nn.Sequential(layers[0], layers[1], …)`. `Sequential` expects **modules as separate positional arguments**, so we unpack the list to satisfy that signature. `nn.Sequential(layers)` (no star) would pass the list as a *single* argument and fail (a list is not an `nn.Module`). Don't confuse the two directions: `def f(*args)` *packs*; `f(*layers)` *unpacks*; `**` is the dict sibling for kwargs.

**Q4. Why define `vgg_block` at all — why not inline the blocks in `VGGNet` like Day 17?**
**A.** (First, a correction to the premise: **Day 17 already split `features`/`classifier`** — it just *inlined* every block as a long literal list.) Day 18 replaces that with a **factory function** because the three blocks are **structurally identical, differing only in numbers** — the classic sign to parameterize (DRY). Benefits: fewer hand-written lines ⇒ fewer channel-typo bugs; trivial to change (add a 4th block = one line; change depth = `num_convs`); and `vgg_block(128, 256, 2)` reads like the paper. Trade-off: it builds the graph dynamically at runtime (minor). It's a **function returning a Module** — perfect for simple repetition; a **class** is the tool when you need custom `forward` logic (which ResNet/U-Net need next).

**Q5. Why not merge `self.features` and `self.classifier` into one `nn.Sequential`?**
**A.** You *could* (`VGGNet`'s forward is a straight line), but keeping them separate buys four things — three of which are the point of the coming days:
1. **Clarity** — `features` = extractor (Conv/BN/ReLU/Pool); `classifier` = head (pool/flatten/linear).
2. **Reusability — Day 20's foundation** — transfer learning *freezes the backbone* and *replaces the head* (`model.fc = nn.Linear(512, 10)`); named/grouped attributes make that a one-liner, whereas one opaque blob has no seam.
3. **Introspectability — Day 19's foundation** — `model.features` lets you register **forward hooks** to grab intermediate feature maps.
4. **Flexibility** — the top-level `forward` is just `self.classifier(self.features(x))`. The moment you need **branching** (ResNet's `out += identity`, U-Net's skips) a single `nn.Sequential` *cannot* express it — you must write a custom `forward`. So the "split + custom `forward`" habit is exactly what Days 18–21 require.

Split line: anything expressible as a straight sequence can live inside a sub-`Sequential` (e.g. `AdaptiveAvgPool2d → Flatten → Linear`), but the top-level `forward` stays hand-written so you can add logic later.

**Q6. What is `AdaptiveAvgPool2d`, vs the `MaxPool2d` we've used so far?**
**A.** Both change spatial size, but they answer different questions:
- **`MaxPool2d(k)`** — *fixed window*; the output size **depends on the input** (`MaxPool2d(2)` on 8×8 → 4×4). A *downsampling* operator used **inside** the body (Day 17) to build translation invariance.
- **`AdaptiveAvgPool2d(output_size)`** — you choose the **output size**; PyTorch chooses the window/stride to hit it. With `output_size=1` it is **Global Average Pooling (GAP)**: each channel collapses to its spatial mean → `(C,1,1)`. Verified resolution-independence: `(1,256,8,8) → (1,256,1,1)` **and** `(1,256,4,4) → (1,256,1,1)` — *the same output regardless of H,W*.

**Why GAP at the head:** (1) **kills the flatten-coupling trap** — `Linear(256,10)` works for any input size (vs Day 17's hard-coded `Linear(128*4*4, 256)`); (2) **far fewer parameters** (~1.3k vs ~527k, −84.7%); (3) conceptually a per-channel "how strongly does this feature appear *anywhere*" summary.

**Semantics:** MaxPool keeps the *strongest* activation ("is this feature present?"); GAP *averages* (overall magnitude). They are not competitors — **MaxPool downsamples in the body; GAP collapses at the head.**

### Round 3 — The residual block, in depth

**Q. What does "residual" mean — what are `x`, `H(x)`, `F(x)`? And what problem does it solve?**
**A.** *Residual = "what is left over after subtracting the obvious part"* (in stats, `observed − predicted`; here, `desired_mapping − identity`). The block learns only the *correction* on top of passing the input through:

| Symbol | Meaning |
|---|---|
| `x` | the block input (also the identity mapping, "do nothing") |
| `H(x)` | the *desired/underlying* mapping the block should compute |
| `F(x) = H(x) − x` | the **residual** — the correction the conv layers learn |
| `output = F(x) + x` | residual learning — re-add the identity |

The reparameterization `H(x) = F(x) + x` is the whole trick: a plain block must synthesize the identity `H(x)=x` *through* nonlinear conv+BN+ReLU layers (awkward, fragile); a residual block gets the identity **for free** via the `+ x` wire. So if the best thing is "do nothing", the net just drives `F(x)→0` (push conv weights toward zero) — easy. It solves the **degradation problem**: very deep *plain* nets get worse on **train and test** error (an *optimization* failure — gradients vanish/explode threading a long chain of matmuls), and the skip gives gradients a direct route through the `+`.

**Q. What is `self.shortcut` — and why is it called a "shortcut" when it sometimes looks like a normal `Conv2d`?**
**A.** It is the branch that carries `x` to the addition, with **two modes**:

```python
self.shortcut = nn.Sequential()
if stride != 1 or in_channels != out_channels:
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels))
```

- **Mode 1 — identity (default):** same shape (`stride=1` and `in==out`) → an **empty** `Sequential` = do-nothing → `shortcut(x) == x`, **0 parameters** (verified). The pure skip — a free wire.
- **Mode 2 — projection:** shape changes (`stride=2` or `in≠out`) → the main output can't be added to `x` (verified `RuntimeError: size of tensor a (16) must match tensor b (8)`), so a **1×1 conv (same stride) + BN** adapts `x` to the same shape/statistics (e.g. `8,448` params for `64→128, s2`). Verified: `shortcut(x)` and the main 3×3 output are both `(1,128,8,8)`.
- The name "shortcut" refers to the **whole bypass path** (it skips the expensive, shape-changing main road), **not** to that conv. The 1×1 conv is only a **shape adapter for the identity** (cheap, per-pixel) — not a feature extractor.
- **Why `bias=False` everywhere:** a conv bias adds a per-output-channel constant, but the following `BatchNorm2d` already has its own learnable shift (`β`), so the bias is redundant.

**Q. (cont.) Walk the `ResidualBlock.forward` wiring line by line — why this order, and what does it buy over the VGG block?**
**A.** The sequence (from my walkthrough):
```python
identity = self.shortcut(x)                        # ① bypass: x, or its projection
out = conv1(x); out = bn1(out); out = relu(out)    # ② main road step 1
out = conv2(out); out = bn2(out)                   # ③ → now `out` is F(x)
out += identity                                    # ④ THE skip: F(x) + x
out = relu(out)                                    # ⑤ final activation
```
- **①②③ compute `F(x)`** (a mini-VGG stack). **④ adds the identity** on the **clean, pre-activation path** (adds to the raw `x`, not an already-ReLU'd tensor) — this is the single line that makes it *residual*. **⑤** is the standard post-add ReLU.
- **Post-add ReLU subtlety:** the true block is `ReLU(F(x)+x)`, so the identity is not *mathematically* exact for negative `x` — but every block input is already non-negative (it is the previous ReLU's output), so in practice the identity holds. Live demo: with `F=0`, the gradient is 1 where `x>0` and 0 where `x<0` (purely the ReLU boundary).
- **What it buys over the VGG block (two guarantees):**
  1. **"Adding layers cannot hurt."** If `F(x)→0` the output equals `x`. Verified by zeroing all weights (`F=0`): `allclose(relu(x), out) == True`, max abs difference `0.0`. A plain block under the same condition has **no path to the output** — zeroed convs kill the signal.
  2. **The gradient highway.** The block's Jacobian is `∂out/∂x = I + ∂F/∂x`; the leading **identity term `I`** guarantees at least one derivative-1 route regardless of the convs. Depth-100 chain demo: **residual** grad-norm at input `5.7446` (survives) vs **plain** `0.0000` (lost). *That is why deep ResNets train and deep VGGs don't* — and why ResNet wins the notebook comparison despite a similar/larger parameter count (equal capacity, far easier to optimize).

**Q. How does a residual block compose into a ResNet — and why is it a class, not a function like `vgg_block`?**
**A.** `SimpleResNet` = **stem** (`Conv 3→64 + BN + ReLU`, stays 32×32) → **stages of `ResidualBlock`** → `AdaptiveAvgPool2d(1)` → `Linear(256→10)`. **Stage rule:** the *first* block of a stage uses `stride=2` + doubled channels (→ projection shortcut); the rest are same-shape identity blocks:
- `layer1`: `ResidualBlock(64,64) ×2` @ 32×32 (identity)
- `layer2`: `ResidualBlock(64,128, stride=2)` + `ResidualBlock(128,128)` @ 16×16
- `layer3`: `ResidualBlock(128,256, stride=2)` + `ResidualBlock(256,256)` @ 8×8

This is the Day 17 resolution↓ / channels↑ ladder re-expressed as *residual stages*. The top-level `forward` is straight-line — **the branching lives inside each block**, not at the top level.

**Class vs function:** `vgg_block` is a **factory function** returning `nn.Sequential` (a straight line → Sequential expresses it, no custom `forward`). `ResidualBlock` is an **`nn.Module` subclass** because it **branches and merges** (`out += identity`) and has **conditional** parts (`if stride != 1 or in != out`). `nn.Sequential` *cannot* express a skip-add, so a hand-written `forward` is mandatory. **Rule of thumb: straight pipeline → function; any branch/merge/conditional → class.** Lego mental model: `ResidualBlock` is a **brick**, `SimpleResNet` is the **assembled model** (repeatedly stacked with `nn.Sequential`).

**Thread:** the skip connection is the single most reused idea ahead — U-Net encoder↔decoder cross-links (`day21_extra`) and the Transformer residual stream `x = x + sublayer(x)` (Day 23+).

### Round 4 — Follow-up clarifications ("loose ends")

**Q1. Does this mean the residual net is superior to VGG in all other ways and completely retires VGG?**
**A1.** Split "model" vs "idea". The VGG *model* is superseded (huge, surpassed). But the VGG *block pattern* (stacked 3×3 convs) is **not** retired — it is the brick *inside* `ResidualBlock` (two 3×3 convs), which ResNet merely wraps in a skip connection. So ResNet = VGG's insight **+** residual, i.e. layering, not replacement. VGG remains the pedagogically simplest (a straight line).

**Q2. In VGG the downsample is via pooling; in ResNet via convolution — is that because stride>1, or kernel>3, and does `conv2d` figure it out automatically?**
**A2.** Downsampling is driven by **stride**, not kernel size. With `stride=1, padding=1` the spatial size is unchanged regardless of kernel (formula `out = (in + 2p − k)/s + 1`). ResNet downsamples with a `stride=2` conv — on **both** paths (main `conv1` stride-2 **and** the projection shortcut's 1×1 stride-2 conv). VGG uses `MaxPool`. `nn.Conv2d` never "auto-decides" the stride — **you** set it.

**Q3. Since it's the same as a VGG block (two small convs), is the reason it acts as a "correction" simply that the identity is added back later?**
**A3.** Yes. The conv output is an ordinary tensor; the `+ x` is what **reparameterizes** it as `F(x) = H(x) − x`. The convs don't "know" they're a residual — the addition changes what the network is asked to learn (and the optimization landscape). The architecture is a mini-VGG; the *meaning* comes entirely from the `+ x`.

**Q4. Mental picture: VGG = straight line of convs; ResBlock = two parallel paths (a shortcut + VGG-like convs) that merge?**
**A4.** Correct. Path A = mini-VGG (two 3×3 convs); Path B = shortcut; they merge by addition, then ReLU. Caveat: in the same-shape case the shortcut is a **plain wire (0 params)**, not a conv — the conv sits on Path A. A 1×1 conv appears on Path B only when the shape changes (projection).

**Q5. Why is ResNet structured `conv1 → layer1 → layer2 → layer3`, not VGG-style `features` + `classifier`?**
**A5.** You *could* name them `features`/`classifier` — it works. The stage naming is a convention with real benefits: (1) each stage is a distinct unit whose **first block is special** (stride-2 + doubled channels → projection); (2) it mirrors `torchvision`'s `resnet18` (`layer1…layer4`, `fc`) used in Day 20; (3) it lets you freeze/replace **by stage** in transfer learning; (4) it documents the resolution/channel milestones. VGG's stages were uniform, so one flat `features` Sequential sufficed.

---

## 1. Where This Sits — the CNN Arc at a Glance

Day 16 taught the *atom* (convolution), Day 17 taught the *first molecule* (Conv→BN→ReLU→Pool stack + head), and day17b taught the *image-to-image turn* (swap head + loss, and hinted at skip connections). **Days 18–21 are about architecture and practice**, not new primitives:

| Day | Notebook | Essence | The single new idea |
|---|---|---|---|
| **18** | `day18_cnn_architectures.ipynb` | VGG blocks & ResNet skip connections | **`output = F(x) + x`** |
| **19** | `day19_cnn_deep_dive.ipynb` | Feature visualization & understanding | The network is **inspectable**, not a black box |
| **20** | `day20_transfer_learning.ipynb` | Transfer learning & fine-tuning | **Reuse** ImageNet features |
| **21** | `day21_cnn_from_scratch_project.ipynb` | Build a CNN from scratch (target >90%) | **Composition** of everything |
| extra | `day21_extra_cnn_restoration.ipynb` | U-Net for denoise/deblur/super-res | Skip connections **across** encoder→decoder |

**One-sentence framing:** Days 16–17 gave us the building block; Days 18–21 teach how to **stack** blocks (18), how to **see** what they learned (19), how to **borrow** someone else's blocks (20), and how to **assemble a winning entry** (21).

**Historical context (from the notebook):**

| Year | Model | Key Idea | Depth | ILSVRC Top-5 Error |
|---|---|---|---|---|
| 2014 | **VGG** | Small (3×3) filters stacked deep | 16–19 | 7.3% |
| 2015 | **ResNet** | Skip connections | 152+ | 3.6% |

Both ideas are still used in essentially every modern architecture.

---

## 2. Part 1 — VGG-Style Blocks

### 2.1 The insight (why small filters, stacked)

Instead of one big filter (5×5 or 7×7), **stack multiple 3×3 filters**:

- Two 3×3 convs have the **same receptive field** as one 5×5.
- Three 3×3 convs = one 7×7.
- But with **fewer parameters** and **more non-linearity**.

**Parameter argument** (for `C` input and output channels):

| Alternative | Effective RF | Parameters | Result |
|---|---|---|---|
| one 5×5 | 5×5 | `25·C²` | 25C² |
| two 3×3 (stacked) | 5×5 | `2·(9·C²) = 18·C²` | **28% fewer** |
| one 7×7 | 7×7 | `49·C²` | 49C² |
| three 3×3 (stacked) | 7×7 | `3·(9·C²) = 27·C²` | **45% fewer** |

And each stacked conv adds a ReLU, so the function is **more non-linear** than one big conv. Same "field of view", less weight, more expressive.

**The VGG block:**

```
Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → MaxPool2d
```

### 2.2 Implementation (this notebook)

```python
def vgg_block(in_channels, out_channels, num_convs):
    """A VGG-style block: num_convs × (Conv → BN → ReLU) + MaxPool."""
    layers = []
    for i in range(num_convs):
        layers.extend([
            nn.Conv2d(in_channels if i == 0 else out_channels,
                      out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class VGGNet(nn.Module):
    """VGG-style network for CIFAR-10."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            vgg_block(3,   64,  2),   # 32→16
            vgg_block(64,  128, 2),   # 16→8
            vgg_block(128, 256, 2),   # 8→4
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # → (256, 1, 1)
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

Three VGG conventions carried over from Day 17: `kernel_size=3, padding=1` (**"same" padding**, so each conv keeps H×W and only MaxPool downsamples), `Conv→BN→ReLU` blocks, and the **resolution↓ / channels↑ ladder** (`32→16→8→4`, `3→64→128→256`). The head uses **Global Average Pooling** (`AdaptiveAvgPool2d(1)`), which collapses any spatial size to `1×1` — removing the "flatten size is hard-coded" coupling trap from Day 17.

### 2.3 Verified shape trace

```
Input:   (1, 3, 32, 32)
block1:  (1, 64, 16, 16)     # 2×(Conv3×3+BN+ReLU) + MaxPool
block2:  (1, 128, 8, 8)      # 2×(Conv3×3+BN+ReLU) + MaxPool
block3:  (1, 256, 4, 4)      # 2×(Conv3×3+BN+ReLU) + MaxPool
head:    (1, 10)             # GAP → Flatten(256) → Linear(256→10)
```

**Parameter count (measured): `1,149,770`**

| Component | Params |
|---|---|
| `features` | `1,147,200` |
| `classifier` (GAP + `Linear(256→10)`) | `2,570` |

The head is now tiny (2,570) compared to Day 17's `~527k` flatten-FC head — the GAP-pooling lesson in action.

---


## 3. Part 2 — ResNet: Skip Connections

### 3.1 The problem: deeper is not always better

Stacking more layers *should* only help (a deeper net can at least represent the shallower one by learning identities). In practice, **very deep plain networks perform worse on both train and test error** — the **degradation problem**. The cause is optimization, not capacity: gradients passed through a long chain of matrix multiplications **vanish or explode**, so the early layers stop learning. (This is not overfitting — train error itself goes up.)

### 3.2 The fix: residual learning

Instead of asking a block to learn the target mapping `H(x)` directly, ask it to learn the **residual** `F(x) = H(x) − x`, then add `x` back:

$$\text{output} = F(x) + x$$

This is the **skip connection** (a.k.a. shortcut connection). If the ideal thing for the block is "do nothing", the network only needs to drive `F(x) → 0` (push the conv weights toward zero) — trivial — and the block becomes an **identity**. So **adding layers can never hurt**: extra blocks can always default to identity instead of degrading the signal.

```
x ──────────────────────┐
│                        │  (skip / shortcut)
├─ Conv → BN → ReLU      │
├─ Conv → BN             │
│                        │
└───── + ◄───────────────┘
       │
     ReLU
```

**Why this helps optimization:** the gradient can flow through the `+` directly (the identity path), giving early layers a low-resistance route that bypasses the conv stack. That is what makes 100+ layer networks trainable.

### 3.3 Implementation (this notebook)

```python
class ResidualBlock(nn.Module):
    """A basic residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # If dimensions change, we need to project the skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)  # Skip connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # ← The skip connection!
        out = torch.relu(out)

        return out
```

**The two shortcut modes (this is the crux):**

| Condition | `self.shortcut` | Params | Meaning |
|---|---|---|---|
| `stride == 1` and `in == out` | `nn.Sequential()` (**empty**) | **0** | pure identity — `identity = x` |
| `stride != 1` **or** `in != out` | `1×1 Conv(stride=stride) + BN` | e.g. `8,448` for `(64→128, s2)` | **projection** — reshapes the skip to match the main branch |

The add `out += identity` requires **identical shapes**. When the block downsamples (`stride=2`) or changes channel count, `x` no longer matches the main-branch output, so the shortcut must be **projected** by a 1×1 conv (with the same stride) to the same shape/statistics. Verified example:

```
ResidualBlock(64, 128, stride=2) → shortcut = Conv2d(64,128,kernel=1,stride=2,bias=False) + BN(128)
projection params = 8,448
```

Why `bias=False` everywhere: a conv bias would be immediately cancelled by the following `BatchNorm2d`'s own shift (`β`), so it is redundant — dropped to save parameters.

**Verified block behavior (from the notebook):**

```
Same dims:     (1, 64, 16, 16) → (1, 64, 16, 16)
Change dims:   (1, 64, 16, 16) → (1, 128, 8, 8)
```

---

## 4. Building a Complete ResNet — the Stage Pattern

```python
class SimpleResNet(nn.Module):
    """A small ResNet for CIFAR-10."""

    def __init__(self, num_classes=10):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.layer1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))                # 32×32
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128))  # 32→16
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256)) # 16→8

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

**The universal ResNet stage rule:**

- **Layer 1**: keeps spatial size (`32×32`), establishes 64 channels.
- **Layer 2**: first block uses `stride=2` (halves `32→16`) **and** doubles channels (`64→128`) → triggers the **projection** shortcut; remaining blocks are same-shape identity blocks.
- **Layer 3**: halves again (`16→8`), doubles channels (`128→256`) → projection; then identity.
- **`AdaptiveAvgPool2d(1)`** → `(256,1,1)`, so the classifier no longer depends on input resolution.
- **`fc`**: single `Linear(256→10)`.

**Verified shape trace:**

```
conv1:    (1, 64, 32, 32)
layer1:   (1, 64, 32, 32)
layer2:   (1, 128, 16, 16)
layer3:   (1, 256, 8, 8)
avgpool:  (1, 256, 1, 1)
fc:       (1, 10)
```

**Parameter count (measured): `2,777,674`**

| Component | Params |
|---|---|
| `conv1` | `1,856` |
| `layer1` | `147,968` |
| `layer2` | `525,568` |
| `layer3` | `2,099,712` |
| `fc` | `2,570` |

Note the parameters **concentrate in the deep stages** (`layer3` alone ≈ 2.1M ≈ 76%) — exactly the "resolution↓ / channels↑ ⇒ params↑" trend from Day 17.

---

## 5. Train & Compare (identical settings)

Both models use the **same** Day-17 CIFAR-10 pipeline (`RandomHorizontalFlip` + `RandomCrop(32, padding=4)`, CIFAR mean/std normalization, 45,000/5,000 train/val split, batch 128) and the **same** training recipe:

- Loss: `nn.CrossEntropyLoss`
- Optimizer: `Adam(lr=1e-3, weight_decay=1e-4)`
- Scheduler: `CosineAnnealingLR(T_max=epochs)`
- 30 epochs, seed 42

```python
h_vgg    = run_training(VGGNet(), "VGG", epochs=30)
h_resnet = run_training(SimpleResNet(), "ResNet", epochs=30)
```

**Parameter comparison:**

| Model | Params | Head |
|---|---|---|
| `VGGNet` | **1,149,770** | GAP → `Linear(256→10)` |
| `SimpleResNet` | **2,777,674** | GAP → `Linear(256→10)` |

**Expected result (notebook narrative):** ResNet *typically converges faster and reaches higher validation accuracy* than VGG **despite having more parameters** — the skip connections make optimization easier. The lesson is not "more parameters win"; it is that **architecture (the skip connection) matters more than raw parameter count**. The comparison plots `h_resnet["val_acc"]` vs `h_vgg["val_acc"]`.

---

## 6. Why Days 18 → 19 → 20 → 21 (the ordering / connections)

Each day deposits exactly one ingredient the Day-21 capstone needs:

| Prerequisite | What it contributes to Day 21 |
|---|---|
| **18 — ResNet blocks** | `MyCNN`'s backbone *is* residual blocks; without them you regress to Day 17's plain CNN that plateaus lower |
| **18 — VGG pattern + GAP head** | The "stack 3×3 convs, then average-pool the head" structuring choice |
| **19 — visualization** | The *diagnostic* mindset for Day 21's ablation exercise: knowing *why* BN/dropout/residual help, not just that they do |
| **20 — transfer learning** | The honest **baseline/contrast**: "can your own design beat pretrained transfer?" — Day 21 Exercise 3 is a from-scratch vs transfer shoot-out |
| **13/14/15 — BN+LR schedule / checkpointing / augmentation** | The machinery that makes 90% accuracy reachable at all |

In short: **18 gives the blocks, 19 gives the microscope, 20 gives the competitor, and 21 makes you the engineer who beats both.**


**Threads to watch across the arc:**

1. **Skip-connection thread:** day17b (foreshadow) → **Day 18** (`out += identity`, projection 1×1) → day21_extra (U-Net encoder↔decoder cross-links) → Days 23+ (Transformer residual stream). *One idea, four appearances.*
2. **Hierarchy thread:** Day 16 (feature maps) → Day 17 (resolution↓ / receptive fields) → **Day 18** (staged ResNet) → Day 19 (hierarchy *visualized*) → Day 20 (hierarchy *reused*).
3. **Swap-the-head-and-loss thread:** Day 17 (classifier) → 17b (autoencoder) → day21_extra (U-Net restoration).
4. **Reuse-vs-build thread:** Day 20 (borrow ImageNet) vs Day 21 (build your own).

**Why ResNet's skip idea is the linchpin:** it is the single most reused construct in the rest of the curriculum — it reappears verbatim in the U-Net and in every Transformer block.

---

## 7. Optics Connections

| DL concept | Optics analogue |
|---|---|
| Conv kernel | PSF / blur filter (Day 16) |
| Small stacked 3×3 convs (VGG) | Composing several small PSFs instead of one large kernel — same footprint, cheaper, more non-linear |
| Receptive field growth | Effective PSF support growing through a cascade |
| Skip connection | A reference / bypass beam; the identity path preserves the signal |
| Projection shortcut (1×1 conv) | A per-pixel channel mixer / reshaping element in the bypass |
| GAP head | Collapsing a focal-plane array to a scalar per band |
| Deep but bad plain net (degradation) | A long cascade where the signal (gradient) is attenuated out of range |

---

## 8. Exercises (from the notebook)

1. **Deeper VGG** — add a 4th VGG block with 512 channels. Does it help or hurt? (Watch the parameter blow-up and possible overfitting.)
2. **Remove skip connections** — comment out `out += identity` in `ResidualBlock` and re-train. Expect the deep plain net to stall / degrade — the *degradation problem* in one line of code.
3. **Bottleneck block** — implement `1×1 (reduce) → 3×3 → 1×1 (expand)`. Compare parameter count with the basic block. (This is what real ResNet-50 uses.)
4. **torchvision ResNet** — load `models.resnet18(num_classes=10)`, adapt the first conv for 32×32 input, and train on CIFAR-10.

---

## 9. Practical Notes — Data & Budget

- **Data (no re-download):** `day18/data` is a **symlink → `../day15/data`**, so `CIFAR10(root="./data", download=True, ...)` loads fully offline (`download=True` becomes a no-op). Same pattern for `day19…day21`, `day21_extra`. These `data` links are `.gitignore`d.
- **Device:** `mps` on this Mac (Apple GPU); CPU fallback is automatic.
- **Budget:** both models train for 30 epochs — the notebook is the long one of the arc. Use the `.venv (3.11.8)` Jupyter kernel, or run headless with `.venv/bin/python`.
- **Gotcha:** when the notebook is open in VS Code while you edit it on disk, saving from a stale editor buffer can overwrite your changes — reopen / `File: Revert File` before saving.

---

## 10. Questions for Self-Check

1. Why do two stacked 3×3 convs match the receptive field of one 5×5 conv, yet use fewer parameters? Give the exact param ratio.
2. What does `vgg_block(in, out, num_convs)` do `num_convs` times? What is the output shape of `vgg_block(64, 128, 2)` applied to a `(N,64,16,16)` tensor?
3. What does `padding=1, kernel_size=3` accomplish, and which layer performs the downsampling in `VGGNet`?
4. State the **degradation problem**. Why is it an *optimization* problem and not overfitting?
5. Write residual learning's core equation. What does the block learn, and what happens if `F(x)=0`?
6. When is `self.shortcut` an empty `nn.Sequential()`, and when does it contain a `1×1` conv? Why is a projection needed exactly then?
7. `ResidualBlock(64, 128, stride=2)` — what are the main-branch and shortcut output shapes, and why must they match?
8. Why does `SimpleResNet` use `stride=2` **and** double channels on the *first block of a stage* (not on every block)?
9. What does `AdaptiveAvgPool2d(1)` do, and why does it make the classifier independent of input resolution?
10. `SimpleResNet` has more parameters than `VGGNet` — so why is it expected to train better?
11. What are the two things that change when you convert a classifier into an image-to-image model (from day17b), and how does the U-Net's skip connection relate to Day 18's skip connection?
12. Name the four "threads" that run from Day 16 through Day 21 and beyond.

---

## 11. Next Steps

| Day | Topic |
|-----|-------|
| **Day 16** ✅ | The Convolution Operation |
| **Day 17** ✅ | Build Your First CNN |
| **Day 17b** ✅ | Autoencoder Intro |
| **Day 18** ✅ | CNN Architectures — VGG blocks & ResNet skip connections — **complete** |
| **Day 19** | CNN Deep Dive — feature maps, filters, Class Activation Maps, dead neurons |
| **Day 20** | Transfer Learning & Fine-Tuning (ResNet-18 on ImageNet → CIFAR-10) |
| **Day 21** | Build a CNN from Scratch — Complete Project (>90% CIFAR-10) |
| **day21_extra** | CNN Restoration — U-Net denoising / deblurring / super-resolution |

**CNN arc ordering:** Day 16 → 17 → **17b** → **18** → 19 → 20 → 21 → **21_extra**.

**Bridge forward:** the residual/skip connection mastered today is the *exact* mechanism the U-Net uses as encoder↔decoder cross-links in `day21_extra`, and the same `x + sublayer(x)` residual stream appears in Transformer blocks starting Day 23. Day 19 next turns the microscope on: visualizing the feature hierarchy these stages build.

---

*Last updated: 2026-10-02*

