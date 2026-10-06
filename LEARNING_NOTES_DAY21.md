# Learning Notes — Day 21: Build a CNN from Scratch — Complete Capstone Project (>90% CIFAR-10)

> **Date:** 2026-10-04
> **Curriculum:** 30-Day Deep Learning Curriculum — Phase 4: CNNs (Days 16–21)
> **Notebook:** `day21/day21_cnn_from_scratch_project.ipynb`
> **Prerequisite:** Day 17 (the CNN block blueprint: Conv→BN→ReLU→Pool, resolution↓/channels↑), Day 18 (`VGGNet` + `ResidualBlock`/`SimpleResNet` — the residual math this note refactors), Day 15 (CIFAR-10 augmentation + normalization), Day 13 (BatchNorm + LR scheduling), Day 14 (train/eval helpers, checkpointing, early stopping)
> **Topics Covered:** the capstone model `MyCNN`; **how the Day-18 `ResidualBlock` was refactored into two single-responsibility blocks** (`ConvBlock` for *shape changes* + `ResBlock` for *constant-shape refinement*), and why the `if stride != 1 or in != out` projection branch **disappears** (it becomes unreachable); why downsampling moved from a **stride-2 conv** (ResNet) to an **out-of-block `MaxPool2d`** (VGG idiom; the same VGG skeleton ResNet kept); the full training pipeline (augmentation, Adam + weight decay, CosineAnnealingLR, checkpointing, early stopping, per-class report); and the through-line to Days 23+ (residual streams / U-Net skips).

---

## 📝 Conversation Q&A Log (Day 21 discussion)

> **Purpose:** a record of the *questions raised during the Day 21 session* and the answers given, so the discussion is not lost (distinct from the reference material below).

### Round 1 — Kick-off: "Day 21 refactors Day 18 but looks different — walk me through it"

**Q1. Day 21's `ResBlock` has *no* `if stride != 1 or in_channels != out_channels:` condition check like Day 18's `ResidualBlock`. Why was it removed?**

**A.** Because the refactor makes it **unreachable**. Day 21 splits the "fat" Day-18 block into two single-responsibility pieces:

- `ConvBlock` owns **changing the shape** (`3 → 64 → 128 → 256 → 512` channels).
- `ResBlock(channels)` owns **refinement at constant shape** (its two convs are hard-wired `kernel=3, stride=1, padding=1` with `channels → channels`).

So a `ResBlock` can **never** see `stride != 1` or `in != out` — the branch would always evaluate false. It is deleted as **dead code**, not for brevity. The shape invariant (`+` needs both sides to match) is now guaranteed **by construction** instead of **by a shortcut module**.

**Q2. Day 18 used a stride-2 conv to downsample inside the block; Day 21 uses `nn.MaxPool2d` *outside* the block. Is that a regression?**

**A.** No — it's a legitimate **hybrid of VGG and ResNet**. There are two ways to shrink resolution:

| | Learnable downsample | Non-learnable downsample |
|---|---|---|
| Mechanism | `Conv2d(stride=2)` | `MaxPool2d(2)` |
| Used by | **ResNet** (Day 18, torchvision) | **VGG** (Day 18 Part 1) |
| Trade-off | learns *what* to keep; +params | fixed rule; 0 params; robust |

Day 18's `VGGNet` already taught **Conv → BN → ReLU → MaxPool**; Day 21's `ConvBlock` **is that VGG block**, with a `ResBlock` snapped in after it. This mirrors reality: per Day 20, `resnet18`'s stem is `Conv2d(7×7,s2) + MaxPool2d(3×3,s2)` — a **MaxPool in the stem** — because ResNet kept VGG's skeleton and bolted skip connections on. Day 21 uses the VGG pooling idiom at *every* stage, so it is **"VGG skeleton + residual refinement at each resolution."**

**Q3. Day 21's `ResBlock` takes a *single* input (`channels`) vs Day 18's `(in_channels, out_channels, stride)`. Why?**

**A.** Same reason as Q1: because the block's shape is **fixed**. `in == out` always, and `stride == 1` always, so the three-argument generality collapses to one number. Fewer args ⇒ fewer ways to misuse the block ⇒ the code reads like the invariant.

**Q4. Is Day 21's design *worse* than Day 18's?**

**A.** It's a **constrained specialization**, with real trade-offs (see §4). Advantages: simpler/DRY, **fewer parameters** (identity shortcuts are *free* in every block; Day 18 paid a 1×1-conv projection at each stage transition), and it matches a widely-used pattern. Costs: no *learnable* downsample; pooling sits *between* stages so the residual stream never sees the resolution change; and the block is less flexible (to add stride-2 you'd have to reintroduce the branch you removed). **Day 18 teaches the general mechanism (so you understand projection shortcuts / the `downsample` tell from Day 20); Day 21 applies a cleaned-up special case at a bigger scale.**

> **One-liner:** Day 21 refactors from **"one smart block"** to **"two dumb blocks"**; the missing `if stride != 1 or in != out` is the *proof* the refactor worked — the shape invariant can no longer be violated inside a `ResBlock`, so the projection branch is unreachable and gone.

---

## 1. Where this sits — Day 21 in the arc

| Day | Role in the CNN arc |
|---|---|
| 16 | Convolution operation (kernel = PSF) |
| 17 | First CNN (`SimpleCNN`) — the block blueprint |
| 17b | Autoencoder bridge (skips foreshadowed) |
| 18 | VGG blocks & ResNet skip connections — build `VGGNet` + `SimpleResNet` |
| 19 / 19b | Visualize the feature hierarchy (behavior / structure) |
| 20 | Reuse a *pretrained* `resnet18` — transfer learning & fine-tuning |
| **21** | **Capstone: build a CNN from scratch, train it well (>90% CIFAR-10) — this note** |
| 21_extra | CNN restoration (U-Net denoising/deblurring/super-resolution) |

Day 21 introduces **no new mechanism** — it **integrates**: Day 18's blocks/stage pattern **+** Day 13's BN & LR scheduler **+** Day 14's checkpointing/early stopping **+** Day 15's augmentation pipeline. The *new skill* is **engineering**: designing the block boundaries, then running a disciplined training loop and evaluating thoroughly. Day 20's "reuse a pretrained backbone" is the natural **competitor** to Day 21's "build it yourself" ethos.

---

## 2. The model, line by line (`day21/day21_cnn_from_scratch_project.ipynb`, cell 5)

```python
class ConvBlock(nn.Module):
    """Conv → BN → ReLU.  [owns: shape change]"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Residual block with skip connection.  [owns: constant-shape refinement]"""
    def __init__(self, channels):                       # ← ONE arg (shape is fixed)
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
    def forward(self, x):
        identity = x                                     # ← ALWAYS identity (0 params, no shortcut module)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)                # ← the skip connection (same math as Day 18)


class MyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 3 → 64, 32×32
            ConvBlock(3, 64),    ResBlock(64),  nn.MaxPool2d(2),   # → 16×16
            # Stage 2: 64 → 128, 16×16
            ConvBlock(64, 128),  ResBlock(128), nn.MaxPool2d(2),   # → 8×8
            # Stage 3: 128 → 256, 8×8
            ConvBlock(128, 256), ResBlock(256), nn.MaxPool2d(2),   # → 4×4
            # Stage 4: 256 → 512, 4×4  (no pool — head GAPs it)
            ConvBlock(256, 512), ResBlock(512),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.head(self.features(x))
```

**Verified (re-run locally):** total **7,827,786 params** (`features` 7,822,656 + `head` 5,130). Shape trace:

```
(1,3,32,32) → (1,64,32,32) → (1,64,32,32) → (1,64,16,16)   # stage 1 + pool
            → (1,128,16,16) → (1,128,16,16) → (1,128,8,8)   # stage 2 + pool
            → (1,256,8,8)  → (1,256,8,8)  → (1,256,4,4)     # stage 3 + pool
            → (1,512,4,4)  → (1,512,4,4)                    # stage 4 (no pool)
            → (1,512,1,1) [GAP] → (1,512) [Flatten] → (1,10)
```

### 2.1 The stage rhythm (two strokes + a coarsen)

```
Stage k:  ConvBlock (C ↑ , H,W same)  →  ResBlock (shape constant)  →  MaxPool (H,W ↓)
          "change channel width, learn new features"   "polish"          "coarsen location"
```

Note the **split of labor** — and contrast with Day 18:

| Responsibility | Day 18 `ResidualBlock` | Day 21 `ConvBlock` + `ResBlock` + `MaxPool` |
|---|---|---|
| Change channels | inside `conv1` (`in→out`) | **`ConvBlock`** |
| Downsample H,W | inside `conv1` (`stride=2`) | **`MaxPool2d`** (between stages) |
| Refine features | inside the block (2 convs) | **`ResBlock`** |
| Shape-match the skip | `self.shortcut` (identity **or** 1×1 projection) | **always identity `x`** (0 params) |

In Day 18 **both** channels and resolution changed at once inside `conv1`; Day 21 splits those into **conv (channels)** and **pool (resolution)**.

### 2.2 Why `bias=False` everywhere (unchanged from Day 18)

A conv bias would be immediately cancelled by the following `BatchNorm2d`'s own shift (`β`), so it is redundant and dropped to save parameters.

---


## 3. The refactor: `ResidualBlock` (Day 18) → `ConvBlock` + `ResBlock` (Day 21)

**Day 18 — the "fat" do-everything block:**

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        ...
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:      # ← the branch
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x):
        identity = self.shortcut(x)                          # ← may be a 1×1 projection
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)
```

**The branch had exactly two modes (Day 18 notes §3.3):**

| Condition | `self.shortcut` | Params | Meaning |
|---|---|---|---|
| `stride == 1` and `in == out` | `nn.Sequential()` (**empty**) | **0** | pure identity — `identity = x` |
| `stride != 1` **or** `in != out` | `1×1 Conv(stride=stride) + BN` | e.g. `8,448` for `(64→128, s2)` | **projection** — reshapes the skip to match |

**Day 21's `ResBlock` lives permanently in the *first* row of that table** — `stride == 1` and `in == out` are guaranteed by the constructor — so:

1. `identity = self.shortcut(x)` simplifies to `identity = x` (the empty `Sequential` isn't even needed).
2. The `if` never fires ⇒ the projection branch is **dead code** ⇒ removed.

**This is the Day 18 "rule of thumb" applied:** straight pipeline → function; **branch/merge/conditional → class**. Day 21's `ResBlock` still needs a hand-written `forward` for the `+` (a merge), but its **conditional** responsibility is gone — that job moved to the *composition* of `ConvBlock` + `MaxPool`.

> **Invariant:** `output = F(x) + x` requires **matching shapes**. Day 18 enforces it **inside** the block via `self.shortcut`; Day 21 enforces it **by construction** — a `ResBlock` is simply not allowed to change shape.

---

## 4. Trade-offs of the split (be precise here)

**Advantages**

- **Simpler, more readable, DRY** — one job per class; no conditional; no projection params.
- **Fewer parameters** — identity shortcuts are **free (0 params)** in *every* block, whereas Day 18 paid a projection conv at each stage transition.
- **Matches a real, widely-used pattern** — VGG skeleton + same-shape residual refinement (also the flavor of "pre-activation"-adjacent designs).

**Costs / things to be aware of**

- **No learnable downsample** — `MaxPool` discards by a fixed rule; a stride-2 conv could *choose* what survives the shrink. For CIFAR-10 this rarely matters, but it is a real modeling difference.
- **Pooling sits *between* stages**, not in the residual path — the residual stream never "sees" the resolution change; shapes match only because the design forbids them from changing together.
- **Less flexible** — Day 18's block is a **general-purpose** brick (drop it anywhere with any stride/channel change); Day 21's needs a separate `ConvBlock` for shape changes. Adding `stride=2` to a `ResBlock` would reintroduce exactly the branch that was removed.

**Bottom line:** Day 18 = the *general mechanism* (with the branch, so projection shortcuts — the `downsample` tell — are understood). Day 21 = a *constrained specialization* of the **identical residual math**, scaled up. Teaching the general tool first, then narrowing it to a tidy special case, is a sound order.

---


## 5. The stage ladder, three ways (Day 17 → Day 18 → Day 21)

All three are the **same "resolution↓ / channels↑" ladder** (Day 17 §2), expressed differently:

| | Day 17 `SimpleCNN` | Day 18 `SimpleResNet` | Day 21 `MyCNN` |
|---|---|---|---|
| Block | `Conv→BN→ReLU→MaxPool` | `ResidualBlock` (stride-2 changes shape) | `ConvBlock` (channels) + `ResBlock` (refine) + `MaxPool` (res) |
| Idea | first CNN — one repeated block | add skip connections | **clean separation of concerns** (capstone) |
| Segments | 3 blocks: 32→64→128 | 3 stages: 64→128→256 | **4 stages: 64→128→256→512** |
| Downsample | `MaxPool` inside block | `stride=2` in `conv1` | `MaxPool` outside, between stages |
| Res of last stage | 4×4 | 8×8 | **4×4** (stage 4 has no pool → GAP) |
| Head | `Flatten→Linear(2048→256)→ReLU→Drop→Linear` | `AdaptiveAvgPool2d(1)→Flatten→Linear(256→10)` | `AdaptiveAvgPool2d(1)→Flatten→Dropout(0.3)→Linear(512→10)` |
| Params | 620,810 (85% in the first Linear!) | 2,777,674 | **7,827,786** |

**Evolutions, each traceable to an earlier day:**

- **GAP head** (Day 18) — this is *why* Day 21 can afford 512 channels × 4 stages: no giant `Flatten→Linear`, so the head is only **5,130 params** (vs Day 17's 524,544 in a single Linear). The freed budget funds depth.
- **Deeper & wider** (4 stages / 512 ch vs 3 / 256) — paralleling Day 20's "`layer4` ≈ 72% of `resnet18`" insight: channel-doubling **quadruples** per-block params.
- **Dropout before the FC** (`nn.Dropout(0.3)`) — regularization carried from Day 13/17.
- **BN everywhere + `bias=False` before BN** — Day 13/18 conventions.

---

## 6. The training pipeline (cells 3, 7, 9)

Every technique from earlier days is assembled into one loop:

| Piece | Setting (Day 21) | Where it came from |
|---|---|---|
| Augmentation | `RandomHorizontalFlip` + `RandomCrop(32, padding=4)` + `ColorJitter(0.1,0.1)` + `RandomErasing(p=0.1)` | Day 15 pipeline; **stronger than Day 18** (adds jitter/erasing) |
| Normalization | CIFAR mean/std `(0.4914,0.4822,0.4465)/(0.2470,0.2435,0.2616)` | Day 15 |
| Split | 45,000 train / 5,000 val (seeded) / 10,000 test | Day 15/18 |
| Optimizer | `Adam(lr=1e-3, weight_decay=5e-4)` | Day 18 Adam; **stronger weight decay** |
| Scheduler | `CosineAnnealingLR(T_max=50)` | Day 13 |
| Loop | checkpoint on best val acc (`best_cnn.pt`), **early stopping** `patience=10`, reload best at end | Day 14 |
| Evaluation | test accuracy + `sklearn.classification_report` (per-class P/R/F1) | new thorough-eval step |

**Diagnostic plots** (cell 11): loss, accuracy, and LR — "val loss up while train loss down ⇒ overfitting"; the LR curve shows cosine annealing; lower LR late = fine-tuning.

**The four exercises** (cell 14) probe the design: (1) **ablation** — remove augmentation / BN / dropout / residual one at a time; (2) **hyperparameter sweep** — 3 LRs × 3 weight decays; (3) **deeper** — 2 `ResBlock`s per stage; (4) **ensemble** — 3 seeds, average predictions. Exercise 1 is the most instructive: **remove `out + identity`** and watch optimization worsen — the Day-18 degradation lesson, re-demonstrated in the capstone.

---

## 7. Key Takeaways

| Concept | Summary |
|---|---|
| **Single-responsibility blocks** | `ConvBlock` = shape change; `ResBlock` = constant-shape refine; `MaxPool` = resolution change |
| **The missing `if` is the point** | A same-shape `ResBlock` can never trigger the projection branch → it's dead code → deleted |
| **VGG idiom + residual** | `MaxPool` between stages = VGG skeleton (which real ResNet kept in its stem) + skip refinement |
| **Same residual math** | `output = F(x) + x`, `identity = x` — identical to Day 18, just guaranteed by construction |
| **GAP funds depth** | Tiny head (5,130 params) enables 512 channels × 4 stages |
| **Engineer the whole loop** | Augmentation + BN + dropout + weight decay + scheduler + checkpoint/early-stop, orchestrated |
| **Deeper ≠ better** | Without the skip `+`, deep nets degrade — the Day-18 lesson, re-run in Exercise 1 |

---

## 8. Questions for Self-Check

1. Why does Day 21's `ResBlock` **omit** the `if stride != 1 or in != out` check? *(The block is same-shape by construction, so the projection branch is unreachable — dead code.)*
2. What two responsibilities did Day 18's single block hold, and which Day-21 module holds each now? *(Shape change → `ConvBlock` + `MaxPool`; refinement → `ResBlock`.)*
3. Why is `identity = x` (not `self.shortcut(x)`) in Day 21's `forward`? *(Because `in == out` and `stride == 1` always, the shortcut is always the empty identity.)*
4. `MaxPool2d` vs `Conv2d(stride=2)` — which model uses which, and what's the trade-off? *(VGG/Day 21 = MaxPool, non-learnable, 0 params; ResNet/Day 18 = strided conv, learnable, +params.)*
5. Why does Day 21 reach **4 stages / 512 channels** where `SimpleResNet` stopped at 3 / 256? *(Deeper ladder; GAP head keeps per-classifier cost flat, so depth is affordable.)*
6. Where do Day 21's identity shortcuts cost parameters — and where did Day 18's? *(Day 21: 0 everywhere; Day 18: a 1×1 projection conv at each stage transition, e.g. 8,448 for 64→128,s2.)*
7. Which single line, if deleted, makes `MyCNN` a *plain* CNN and re-triggers the degradation problem? *(`return torch.relu(out + identity)` — removing the `+ identity`.)*
8. Name three things Day 21 reuses verbatim from Days 13–15. *(CosineAnnealingLR from Day 13; checkpointing + early stopping from Day 14; the CIFAR-10 augmentation + normalization pipeline from Day 15.)*

---

## 9. Next Steps

| Day | Topic |
|-----|-------|
| 16–20 | ✅ done (convolution → first CNN → AE → VGG/ResNet → visualization → transfer learning) |
| **Day 21** | ✅ Build a CNN from Scratch — Capstone (>90% CIFAR-10) — **complete** |
| **day21_extra** | CNN Restoration — U-Net denoising / deblurring / super-resolution |
| **Day 22** | Self-Attention & The Transformer Revolution (Q, K, V, scaled dot-product) |

**Bridge forward:** the **skip connection** built in Day 18 and re-deployed in Day 21 is the through-line — it becomes the **U-Net encoder↔decoder cross-links** in `day21_extra`, and reappears as the **Transformer residual stream** `x = x + sublayer(x)` from Day 23 onward. Day 21 is where the CNN arc's *blocks* (18) meet its *training discipline* (13/14/15) — the from-scratch answer to Day 20's pretrained backbone.

