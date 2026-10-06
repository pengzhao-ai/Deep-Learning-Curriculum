# Learning Notes — Day 20: Transfer Learning & Fine-Tuning (ResNet-18 on ImageNet → CIFAR-10)

> **Date:** 2026-10-02
> **Curriculum:** 30-Day Deep Learning Curriculum — Phase 4: CNNs (Days 16–21)
> **Notebook:** `day20/day20_transfer_learning.ipynb`
> **Prerequisite:** Day 18 (`SimpleResNet` — the exact residual block Day 20 reuses), Day 15 (CIFAR-10 pipeline + normalization), Day 13 (BatchNorm + LR scheduling), Day 14 (train/eval helpers)
> **Topics Covered:** reading `torchvision.models.resnet18` (why `print()` *hides* the skip connection and looks VGG-ish); the layer-by-layer map from the **real** ResNet-18 back to Day 18's `SimpleResNet` (they share `BasicBlock`/`ResidualBlock` — `layer1–3` are **parameter-for-parameter identical**); the `BasicBlock.forward` wiring and the two shortcut modes (`downsample` vs identity); why ResNet beats VGG (the degradation problem → optimization, not capacity) and how ResNet-18 differs from `SimpleResNet` (stem, a 4th stage, pretraining, 1000-class head); and what all of it means for **feature extraction vs fine-tuning** and **differential learning rates**.

---

## 📝 Conversation Q&A Log (Day 20 discussion)

> **Purpose:** a record of the *questions raised during the Day 20 session* and the answers given, so the discussion is not lost (distinct from the reference material below).

### Round 1 — Kick-off: recap + "Section 1 looks like VGG, not ResNet"

**Q. (preface) Recap the codebase and where Day 20 sits.**
**A.** The CNN arc is **Day 16 (convolution) → 17 (first CNN) → 17b (autoencoder; skips foreshadowed) → 18 (VGG & ResNet from scratch) → 19/19b (visualize the hierarchy — behavior/structure) → 20 (reuse a pretrained ResNet-18) → 21 (capstone) → 21_extra (U-Net)**. Day 20 Section 1 loads the **real, pretrained** `torchvision.models.resnet18` — the 2015 paper architecture, trained on ImageNet (1.2M images, 1000 classes).

**Q1. After loading and printing `resnet18`, it shows *many* layers and looks like a VGG rather than the ResNet we learned in Day 18. Why?**
**A.** Three **illusions of `print()`**, laid out in §3:
1. **`print(model)` renders the *module tree*, not the data flow.** It can only show registered sub-modules assigned in `__init__`. The **skip connection is not a module** — it is the tensor op `out += identity` inside `forward()`. I verified it: the printed model is **84 lines** and contains **no `Add` node and no `+`** anywhere (`'Add' in str(m)` → `False`). So each `BasicBlock` prints as a plain `conv1 → bn1 → relu → conv2 → bn2` stack — exactly what a VGG block looks like. **The print flattens the residual away.** (This is the same lesson as Day 19b: a module/`nn.Sequential` diagram cannot show a skip; only `torchview`/Netron's `Add` node can.)
2. **The stem really is VGG-flavored.** `conv1 = Conv2d(3,64,7×7,stride=2)` + `MaxPool2d(3×3,s2)` = a big-conv downsample stem, visually like VGG's conv+pool. ResNet **copied VGG's skeleton** (uniform stages between downsamplings) and bolted the residual wires on — so of course the silhouette matches. ResNet is essentially "VGG + skip connections".
3. **The giveaways that it *is* a ResNet** are (a) stage naming **`layer1…layer4`** instead of one flat `features`/`classifier`, and (b) **`(downsample): Sequential(Conv2d(…,1×1,stride=2), BatchNorm2d)`** inside `layer2/3/4` — the **projection shortcut** from Day 18 (torchvision just names it `downsample`; you named it `shortcut`). It appears exactly when a block changes shape.

> **One-liner:** VGG streams through a single path; ResNet has *parallel* paths (main road + bypass). `print()` only lists the main road, so the bypass — the whole point — is invisible.

**Q2. Connect back to Day 18 and walk the architecture layer by layer.**
**A.** See §4 (the map) and §5 (the walkthrough). The crucial finding: **torchvision `resnet18` and Day 18's `SimpleResNet` use the *same* `BasicBlock`/`ResidualBlock`**, and their `layer1`, `layer2`, `layer3` are **parameter-for-parameter identical** (`147,968 / 525,568 / 2,099,712` — verified). `SimpleResNet` is a **mini-ResNet-18**: an up-scaled article differs only by a 7×7 stem, one extra stage (`layer4`/512ch), and a 10-class head instead of 1000.

**Q3. Why / how is ResNet superior — vs the VGG we built, and vs the `SimpleResNet` we built?**
**A.** See §7. **vs VGG:** the win is **optimization, not capacity** — skip connections fix the *degradation problem* (deep *plain* nets get worse on **train and test** error; an optimization failure, not overfitting), give gradients a low-resistance path through `+`, and give the identity "for free" (`F(x)→0`). ResNet keeps VGG's small-3×3 efficiency and adds a GAP head (in `resnet18`, `fc` is only **4.4%** of params). **vs `SimpleResNet`:** they are the **same family** (`layer1–3` identical); the differences are **scale, depth (`layer4` = 71.8% of all weights), a stronger stem, and — decisively — pretrained ImageNet weights**. That pretraining *is* transfer learning; the topology is the one you already built.

---

## 1. Where this sits — Day 20 in the arc

| Day | Role in the CNN arc |
|---|---|
| 16 | Convolution operation (kernel = PSF) |
| 17 | First CNN (`SimpleCNN`) |
| 17b | Autoencoder bridge (skips foreshadowed) |
| **18** | **VGG blocks & ResNet skip connections — you build `VGGNet` + `SimpleResNet` from scratch** |
| 19 / 19b | Visualize the feature hierarchy (behavior / structure) |
| **20** | **Reuse a *pretrained* `resnet18` — transfer learning & fine-tuning (this note)** |
| 21 / 21_extra | Capstone CNN from scratch; U-Net restoration |

Day 20's architectural content is **not new**: Section 1 is Day 18's residual idea at full scale, plus pretrained weights. The *new* idea is the **transfer-learning workflow** (swap the head, freeze/fine-tune).

---

## 2. Section 1 of the notebook, line by line

`day20/day20_transfer_learning.ipynb`, cell 3:

```python
# Load ResNet-18 with ImageNet weights
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

print("ResNet-18 architecture:")
print(resnet18)                     # ← the 84-line printout you're reading

# Check the final layer
print(f"\nOriginal FC layer: {resnet18.fc}")     # Linear(in_features=512, out_features=1000, bias=True)
print(f"Output classes: 1000 (ImageNet)")
```

- `weights=models.ResNet18_Weights.IMAGENET1K_V1` downloads/loads the **1.2M-image ImageNet** checkpoint — this is the whole point (pretrained features), not the topology.
- `resnet18.fc` is the **ImageNet classifier head**: `Linear(512 → 1000)`. Later sections replace it with `nn.Linear(512, 10)` for CIFAR-10 (§8).
- The `print` output is the **module tree** — and, as §3 shows, it **cannot** render the skip connection.

The notebook then (cells 5–14): resizes CIFAR-10 to 224×224 and normalizes with **ImageNet mean/std** (`(0.485,0.456,0.406)/(0.229,0.224,0.225)`), builds **Strategy A feature extraction** (freeze all, retrain only `fc`) and **Strategy B fine-tuning** (all params trainable, backbone at `1e-4` and `fc` at `1e-3` = **differential LR** to avoid *catastrophic forgetting*).

---

## 3. Why `print(resnet18)` *looks* like VGG (three illusions)

### 3.1 `print(model)` shows the **module tree**, not the **data flow**

`print(model)` recursively walks `named_children()` and indents. It can only show things that are **registered sub-modules** created in `__init__`. Verified on the loaded model:

- The printed `resnet18` is **84 lines** long → the "so many layers" feeling.
- It contains **no `Add` node and no `+`** — verified: `'Add' in str(m)` → `False`.

The **skip connection is not a module**. In Day 18's `ResidualBlock` (and torchvision's `BasicBlock`) it is a plain tensor op:

```python
out += identity        # ← THE skip connection — functional add, invisible to print
```

So each `BasicBlock` prints as `conv1 → bn1 → relu → conv2 → bn2` — a stack of convs — which is exactly a VGG block's silhouette. **`print()` flattens the residual away.** Relates directly to **Day 19b**: an `nn.Sequential`/module diagram can *never* show a skip; only `torchview`/Netron's `Add` node (or an explicit `+` in a hand-drawn graph) can.

### 3.2 The stem genuinely is VGG-flavored

`conv1 = Conv2d(3,64, 7×7, stride=2)` + `MaxPool2d(3×3, s2)` is a **big-conv downsample stem**, visually the same idea as VGG's conv+pool blocks. ResNet **kept VGG's skeleton** (uniform stages separated by downsampling) and added residual wires — so the visual similarity is real, not a misreading. **ResNet ≈ VGG + skip connections.**

### 3.3 The two giveaways that it *is* a ResNet

1. **Stage naming:** `layer1 … layer4` instead of one flat `features` + `classifier`. (Day 18 §A5 explains why stages are named: first-block-is-special, mirrors torchvision, freeze-by-stage, documents milestones.)
2. **`(downsample): Sequential(Conv2d(…, kernel=1, stride=2, bias=False), BatchNorm2d)`** inside `layer2/3/4` — this is the **projection shortcut** from Day 18 (your `self.shortcut`'s non-empty mode). It appears only when a block changes shape (`stride=2` or channel change). **Spotting `downsample` is the tell.**

> **Rule of thumb:** if you see `layer1…layerN` + `AdaptiveAvgPool2d` + `fc` + at least one `downsample`, it's a ResNet, no matter how VGG-ish the flat block list looks.

---

## 4. The map: torchvision `resnet18` ≡ an up-scaled Day 18 `SimpleResNet`

Both were run and compared per stage. **`layer1`, `layer2`, `layer3` are parameter-for-parameter identical** — because both use the *same* `BasicBlock`/`ResidualBlock` math:

| Stage | Day 18 `SimpleResNet` | torchvision `resnet18` | same? |
|---|---|---:|---|
| stem `conv1` | `Conv2d(3,64,3×3)` → **1,856** | `Conv2d(3,64,7×7,s2)` → **9,408** | ✗ (bigger stem) |
| `layer1` | 2× Block(64→64, s1) → **147,968** | 2× Block(64→64, s1) → **147,968** | ✅ identical |
| `layer2` | Block(64→128,s2)+Block(128→128) → **525,568** | same → **525,568** | ✅ identical |
| `layer3` | Block(128→256,s2)+Block(256→256) → **2,099,712** | same → **2,099,712** | ✅ identical |
| `layer4` | — (stops at 256ch) | Block(256→512,s2)+Block(512→512) → **8,393,728** | ✗ (extra stage) |
| `avgpool` | `AdaptiveAvgPool2d(1)` | `AdaptiveAvgPool2d(1)` | ✅ identical |
| `fc` | `Linear(256,10)` → **2,570** | `Linear(512,1000)` → **513,000** | ✗ (ImageNet head) |
| **total** | **2,777,674** | **11,689,512** | — |

**Conclusion:** `SimpleResNet` is `resnet18` with (i) a 3×3 stem instead of 7×7, (ii) one fewer stage (no `layer4`/512ch), and (iii) a 10-class head instead of 1000. You already built the right block in Day 18.

**Param-growth intuition:** a BasicBlock's params ∝ `channels² × 9 × 2`; doubling channels **quadruples** per-block params — hence `148K → 526K → 2.1M → 8.4M`. **`layer4` alone is ~72% of the network** (`8,393,728 / 11,689,512 = 71.8%`), and the ImageNet `fc` head is only **4.4%** — which is exactly why swapping `fc` is cheap and stage-freezing is powerful.

*(For scale: `resnet34` = 21,797,672 params (block pattern `[3,4,6,3]`); `resnet50` = 25,557,032 (uses **bottleneck** blocks, 1×1-reduce → 3×3 → 1×1-expand, and a `fc` of 2,049,000).)*

---

## 5. Layer-by-layer walkthrough of `resnet18` (Section 1)

Verified shape trace for a `(1, 3, 224, 224)` input:

| # | Module | Op | Output shape | What it does |
|---|---|---|---|---|
| 1 | `conv1` | `Conv2d(3,64, 7×7, stride=2, pad=3)` | `(1,64,112,112)` | Big-kernel stem: 224→112, 3→64ch |
| 2 | `bn1` | BatchNorm2d | `(1,64,112,112)` | Normalize (Day 13) |
| 3 | `relu` | ReLU | `(1,64,112,112)` | Non-linearity |
| 4 | `maxpool` | `MaxPool2d(3×3, s2, pad=1)` | `(1,64,56,56)` | 112→56; cheap downsample before the heavy stages |
| 5 | `layer1` | 2× `BasicBlock(64→64, s1)` | `(1,64,56,56)` | Establish 64ch; **no** shape change → shortcut = **identity, 0 params** |
| 6 | `layer2` | `BasicBlock(64→128,s2)` + `(128→128,s1)` | `(1,128,28,28)` | →28, →128ch; first block's shortcut = **1×1 conv `downsample`** |
| 7 | `layer3` | `BasicBlock(128→256,s2)` + `(256→256,s1)` | `(1,256,14,14)` | →14, →256ch; projection shortcut again |
| 8 | `layer4` | `BasicBlock(256→512,s2)` + `(512→512,s1)` | `(1,512,7,7)` | →7, →512ch; projection shortcut again |
| 9 | `avgpool` | `AdaptiveAvgPool2d(1)` | `(1,512,1,1)` | Global Average Pooling → resolution-independent |
| 10 | `fc` | `Linear(512,1000)` | `(1,1000)` | ImageNet classification head |

**The stage rule is identical to Day 18:** the *first block of each stage* uses `stride=2` + doubles channels (→ `downsample` projection shortcut); the *remaining blocks* keep shape (→ empty/identity shortcut, 0 params). Verified: `layer2/3/4` each contain exactly one `(downsample)` submodule; `layer1` contains none.

**Inside every `BasicBlock`** (this is your Day 18 `ResidualBlock`, renamed):

```
identity = self.downsample(x)          # empty Sequential → x  |  or 1×1 conv+BN
out = relu(bn1(conv1(x)))              # main road step 1 (conv1 carries the stride)
out = bn2(conv2(out))                  # main road step 2 → F(x)
out = out + identity                   # ← the skip add (INVISIBLE to print)
out = relu(out)                        # post-add activation
```

Note the **post-add ReLU subtlety** (from Day 18): the true op is `ReLU(F(x)+x)`, so the identity isn't *mathematically* exact for negative inputs — but every block input is already the previous ReLU's (non-negative) output, so in practice the identity holds.

---

## 6. Naming translation table (Day 18 ↔ torchvision)

| Day 18 (yours) | torchvision | Role |
|---|---|---|
| `ResidualBlock` | `torchvision.models.resnet.BasicBlock` | the 2×(3×3 conv+BN)+skip unit |
| `self.shortcut` | `self.downsample` | the bypass path (empty ⇒ identity) |
| `self.conv1` (a `Sequential` stem) | `conv1`+`bn1`+`relu`+`maxpool` (separate attrs) | stem |
| `layer1/2/3` | `layer1/2/3/4` | stages (first block special) |
| `avgpool` + `fc` | `avgpool` + `fc` | GAP head |
| 1×1 `shortcut` conv, `bias=False` | identical | projection shortcut; no bias because BN supplies the shift |

---

## 7. Why / how ResNet is superior

### 7.1 vs VGG (the architectural win)

- **Optimization, not capacity.** VGG fails to get deeper than ~19 layers because of the **degradation problem**: very deep *plain* networks get worse on **train and test** error — a vanishing/exploding-gradient *optimization* failure, not overfitting. The skip gives gradients a **direct low-resistance route** through `+`, so 18/34/50/101/152 layers all train. This is why **ResNet hit 3.6% top-5 vs VGG's 7.3%** (Day 18's historical table).
- **Identity for free.** If a block's best job is "do nothing", it drives `F(x)→0` (weights→0) instead of synthesizing the identity *through* nonlinear conv+BN+ReLU layers. So **extra depth can never hurt**.
- **Small-filter efficiency (VGG's own trick, kept).** Two stacked 3×3 convs match one 5×5's receptive field with **fewer parameters** (`18C²` vs `25C²`, −28%) and more non-linearity; three 3×3 = one 7×7 (`27C²` vs `49C²`, −45%). ResNet keeps VGG's uniform 3×3 blocks and adds the wires.
- **Param-efficient head (GAP).** `AdaptiveAvgPool2d(1)` + a single `Linear` instead of VGG's giant flatten→FC. In `resnet18`, `fc` is only **4.4%** of params — and it's the *only* part swapped for a new task.

### 7.2 vs Day 18's `SimpleResNet` (the subtle bit)

They are the **same family**; `layer1–3` are literally the same parameter counts (`147,968 / 525,568 / 2,099,712` — verified). The differences are **scale and purpose**, not a new mechanism:

1. **Input scale:** yours ran at 32×32 (CIFAR); `resnet18` was trained at 224×224 (ImageNet). Notebook Section 2 handles this with `transforms.Resize(224)`.
2. **A 4th stage (512ch):** `resnet18` adds `layer4` — **~72% of all weights** — the capacity needed for 1000 ImageNet classes.
3. **A stronger stem:** 7×7/stride-2 + maxpool (vs your 3×3) to compress 224→56 quickly.
4. **Pretrained weights:** the decisive Day-20 advantage isn't topology, it's that `IMAGENET1K_V1` already encodes 1.2M-image edge/texture/object features — **that is transfer learning**.

> **Honest framing:** Day 18's `SimpleResNet` is a **mini-ResNet-18**; Day 20's `resnet18` is the **full ResNet-18**. The "superior vs VGG" story (skip-connection optimization) is identical in both; the "`resnet18` vs `SimpleResNet`" difference is **depth + scale + pretraining**, not a new idea.

---

## 8. What this means for the rest of Day 20

- `model.fc = nn.Linear(512, 10)` — you replace only the **ImageNet head** (`512→1000` → `512→10`); everything upstream is the reusable feature extractor.
- **Strategy A — Feature extraction:** freeze *all* params (`requires_grad = False`), train only the new `fc`. Params trained: `512×10 + 10 = 5,130` — a tiny fraction of 11.7M. Use for very small datasets (<1000 images).
- **Strategy B — Fine-tuning:** all params trainable, with **differential learning rates** — backbone `1e-4`, new `fc` `1e-3` — to preserve pretrained features and avoid **catastrophic forgetting** (aggressive updates destroying what was learned). Use for medium datasets / similar domains.
- **Freezing by stage** (Exercise 1, gradual unfreezing) is natural because the network is *named by stage*: unfreeze `layer4` first (biggest, most ImageNet-specific), then `layer3`, etc.
- **Why 224×224?** ImageNet-trained weights expect it; the notebook resizes CIFAR-10 up (alternatively, adapt the stem for 32×32, per Day 18 Exercise 4).
- **Normalization must match pretraining:** use ImageNet mean/std, **not** CIFAR's — otherwise the pretrained BatchNorm running stats see a distribution shift.

---

## 9. Optics connections (Day 20)

- **Pre-trained filters = a library of known PSFs.** Fine-tuning is *re-using a calibrated optical system* and only retouching the final read-out; feature extraction is trusting the existing optics and only recalibrating the sensor readout.
- **Early layers = universal primitives** (edges/Gabor/orientation filters — literally wave-optics-like band-pass responses); **late layers = task-specific** (object parts). That division is *why* transfer works: the first stages of a camera pipeline are reusable across scenes.
- **Skip connections ≈ a bypass optical path** that carries the unmodified wavefront around a distorting element, re-combined coherently at the output — the identity/`+` is interference-free addition of the two paths (Day 18's optics note).
- **The degradation problem is an *alignment/optimization* failure, not a resolution limit:** adding more optical elements should never *hurt* if each can be set to "pass-through" — the residual wire is the pass-through guarantee.

---

## 10. Practical notes — data, device, what was verified

- **Data (no re-download):** `day20/data` is a **symlink → `../day15/data`**, so `CIFAR10(root="./data", download=True)` loads fully offline (`download=True` is a no-op). This 224×224 resize makes Day 20 the **heaviest** CNN-arc notebook — expect slower epochs than Day 18.
- **Device:** `mps` on this Mac (Apple GPU); CPU fallback automatic.
- **Verified in this session (read-only inspection, no notebook changes):**
  - `print(resnet18)` → **84 lines**, contains **no `+`/`Add`**.
  - Shape trace `(1,3,224,224) → (1,64,56,56) → (1,128,28,28) → (1,256,14,14) → (1,512,7,7) → (1,512,1,1) → (1,1000)`.
  - Param breakdown: `conv1 9,408` · `bn1 128` · `layer1 147,968` · `layer2 525,568` · `layer3 2,099,712` · `layer4 8,393,728` · `fc 513,000` · **total 11,689,512**.
  - `SimpleResNet` (`1,856` stem · `layer1–3` identical · `fc 2,570`) → **total 2,777,674**.
  - `resnet34` **21,797,672**; `resnet50` **25,557,032**.
- **No notebook code changes were needed** — Section 1 is correct as written.

---

## 11. Questions for Self-Check

1. Why does `print(resnet18)` **never** reveal a skip connection? *(`print` shows registered sub-modules; the skip is `out += identity` inside `forward()` — a tensor op, not a layer.)*
2. Which submodule name is the giveaway that `resnet18` has **projection** shortcuts, and in which stages does it appear? *(`downsample`; `layer2/3/4` — one each; none in `layer1`.)*
3. Why does `layer1` contribute only **148K** params while `layer4` contributes **8.4M**? *(Conv params ∝ `in×out×k²`; channels double each stage → per-block params quadruple.)*
4. `SimpleResNet.layer2` vs `resnet18.layer2` — same params? *(Yes — `525,568` each; same `BasicBlock`.)*
5. State the **degradation problem** and why it is an *optimization* problem, not overfitting. *(Deep plain nets get worse on train *and* test — vanishing/exploding gradients; not a generalization gap.)*
6. Why can "if `F(x)=0` the block is identity" let adding layers never hurt? *(The block can default to a pass-through instead of degrading the signal.)*
7. In transfer learning, why is the `fc` the *only* layer changed in **feature extraction**? *(It's the 1000-class ImageNet head — `4.4%` of params — and the task-specific part; upstream is reusable.)*
8. Why give the backbone a **10× lower LR** than the new head? *(To avoid catastrophic forgetting — preserve pretrained features while the head adapts.)*
9. Why must Day 20 normalize with **ImageNet** mean/std rather than CIFAR's? *(Pretrained BatchNorm running stats assume the ImageNet activation distribution.)*

---

## 12. Next Steps

| Day | Topic |
|-----|-------|
| 16–19b | ✅ done (convolution → first CNN → AE → VGG/ResNet → visualization) |
| **Day 20** | ✅ Transfer Learning & Fine-Tuning (ResNet-18 → CIFAR-10) — **complete** |
| **Day 21** | Build a CNN from Scratch — Complete Project (>90% CIFAR-10) |
| **day21_extra** | CNN Restoration — U-Net denoising / deblurring / super-resolution |

**Bridge forward:** Day 21 combines everything — Day 18's blocks/stage pattern **plus** Day 13/14's BN + Schedulers + checkpointing — trained from scratch to >90% on CIFAR-10. Day 20's lesson (reuse beats retrain when you have a pretrained backbone) is the natural *competitor* to Day 21's "build it yourself" ethos, and the **skip connection** remains the through-line (U-Net in `day21_extra`, residual streams in the Transformer blocks from Day 23).

---

*Last updated: 2026-10-02*
