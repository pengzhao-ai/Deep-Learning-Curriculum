# AI Context — Deep Learning Curriculum Project

> **For the AI:** Read this file at the start of each new session to understand the project context quickly.

---

## 👤 Learner Profile

- **Background:** Optical engineer with camera & computer vision experience
- **Python/ML/DL experience:** New to all three
- **PyTorch experience:** None (learning from scratch)
- **Goal:** Build and train CNN and ViT models from scratch, understand their differences
- **Learning style:** Benefits from optics analogies (convolution ↔ PSF, attention ↔ adaptive spatial weighting)

---

## 📚 Project Overview

**Repository:** `DL_learning/` (30-day Deep Learning Curriculum)

**Structure:**
- 30 folders: `day01/` through `day30/`
- Each folder contains one Jupyter notebook: `dayXX_topic.ipynb`
- Notebooks have: Markdown explanations → executable code → exercises
- Progressive difficulty: each day builds on previous days

**Teaching Approach:**
- Concepts explained in Markdown *before* code appears
- No assumed knowledge — everything is explained
- Optics analogies used where helpful (not mandatory for every concept)
- From-scratch implementations before using PyTorch built-ins
- Exercises at the end for self-practice

---

## 🗺️ Curriculum Roadmap

### Phase 1: Python, NumPy & ML Basics (Days 1–3)
- **Day 01:** Python Essentials for DL (variables, lists, dicts, functions, classes, `__init__`, `__call__`, imports)
- **Day 02:** NumPy for Tensor Thinking (arrays, shapes, broadcasting, vectorized math)
- **Day 03:** What is Machine Learning? (supervised vs unsupervised, features/labels, train/test split)

### Phase 2: PyTorch Foundations (Days 4–9)
- **Day 04:** Tensors — PyTorch's Building Block
- **Day 05:** Autograd & Computation Graphs
- **Day 06:** Gradient Descent from Scratch
- **Day 07:** Backpropagation Deep Dive
- **Day 08:** `nn.Module`, Layers & Optimizers
- **Day 09:** Loss Functions & Activation Functions

### Phase 3: Building & Training Neural Networks (Days 10–15)
- **Day 10:** Data Loading & Preprocessing (Dataset, DataLoader, transforms)
- **Day 11:** First Neural Network (MLP on MNIST) 🎉
- **Day 12:** Overfitting & Regularization (dropout, weight decay, early stopping)
- **Day 13:** Batch Normalization & LR Scheduling
- **Day 14:** Training Pipeline Best Practices (checkpointing, reproducibility)
- **Day 15:** Working with Image Data (CIFAR-10, data augmentation)

### Phase 4: CNNs — Convolutional Neural Networks (Days 16–21)
- **Day 16:** The Convolution Operation (kernel/filter, stride, padding, PSF analogy)
- **Day 17:** Build Your First CNN
- **Day 18:** CNN Architectures — VGG & ResNet
- **Day 19:** CNN Deep Dive — Feature Visualization
- **Day 19b:** Network Visualization (extension) — `torchinfo`, VisualTorch, Netron, TensorBoard (+ optional torchviz/torchview)
- **Day 20:** Transfer Learning & Fine-Tuning
- **Day 21:** Build a CNN from Scratch — Complete Project (target >90% on CIFAR-10)

### Phase 5: Attention, Transformers & ViT (Days 22–28)
- **Day 22:** Self-Attention & The Transformer Revolution (Q, K, V, scaled dot-product)
- **Day 23:** The Transformer Encoder Block (LayerNorm, FFN, residual connections)
- **Day 24:** Patch Embeddings — Images to Sequences
- **Day 25:** Build a ViT from Scratch
- **Day 26:** ViT Attention Visualization
- **Day 27:** CNN vs ViT — Head-to-Head Comparison
- **Day 28:** Advanced Architectures (hybrid models, ConvNeXt)

### Phase 6: Consolidation & Capstone (Days 29–30)
- **Day 29:** Complete Training Toolkit (Trainer class, model zoo, benchmark)
- **Day 30:** Capstone — From Optics to Deep Learning (image quality/degradation classification)

---

## 🔧 Technical Setup

**Required Packages** (see `requirements.txt`):
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
jupyter
ipykernel
scikit-learn
Pillow
tqdm
tensorboard
einops
```

**Device Support:** MPS (Mac Metal) with CPU fallback
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

**Datasets Used:**
- MNIST (Days 11–12)
- CIFAR-10 (Days 15–21, 25–28)

---

## 🎯 Key Learning Outcomes

By Day 30, the learner will have:
1. Built an MLP, CNN, ResNet, and ViT from scratch
2. Understood convolution vs. attention trade-offs
3. Implemented self-attention and Transformer encoder blocks
4. Completed a capstone connecting optics to deep learning
5. Target: >90% on CIFAR-10 with CNN

---

## 📝 Important Notes for the AI

### Teaching Style Observations:
- **Explanations first, code second** — always explain the "why" before "how"
- **Use optics analogies sparingly** — only when they clearly aid understanding
- **Show from-scratch implementations** — e.g., implement attention manually before using `nn.MultiheadAttention`
- **Encourage experimentation** — "break things on purpose" is a stated tip
- **Connect to prior knowledge** — reference concepts from earlier days
- **Answer directly and once** — avoid duplicating answers across responses

### Common Code Patterns:
```python
# Standard imports across notebooks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

torch.manual_seed(42)  # Reproducibility
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### Typical Training Loop:
```python
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    # Forward pass only, no gradients
```

---

## 🚀 How to Use This File

**At the start of a new session:**
1. User says: *"Read AI_CONTEXT.md and help me with Day X"*
2. Read this file to understand:
   - Where they are in the curriculum
   - Their background and learning style
   - Technical setup and constraints
   - Teaching approach that resonates

**When they ask questions:**
- Use optics analogies only when particularly relevant (not mandatory)
- Build on previous days' knowledge
- Show from-scratch implementations before using PyTorch shortcuts
- Encourage experimentation and "breaking things"

**When they complete a day:**
- Ask if they want to update "Current Progress" section
- Celebrate milestones (e.g., "🎉 You built your first CNN!")

---

## 📍 Current Progress

**Last completed:** Day 20 (Transfer Learning & Fine-Tuning — ResNet-18/ImageNet → CIFAR-10) — notes in `LEARNING_NOTES_DAY20.md`; notebook `day20/day20_transfer_learning.ipynb`.

**Currently on:** Day 21 (Build a CNN from Scratch — Complete Project, >90% CIFAR-10) — Phase 4: CNNs (Days 16–21). Day 19b (Network Visualization) and `day21_extra` (CNN Restoration / U-Net) remain companion/extension notebooks.

**Next session goal:** Day 21 — the capstone: build a from-scratch CNN combining Day 18's blocks/stage pattern with Day 13/14's BatchNorm + LR scheduling + checkpointing/early stopping, targeting >90% on CIFAR-10. Day 20's pretrained-ResNet-18 result is the natural "competitor"/baseline to beat, and `day21_extra` (U-Net restoration) follows it.

**CNN arc ordering (as studied):**
- Day 16 → Day 17 → **day17b (autoencoder bridge)** → Day 18 → Day 19 → **day19b (network visualization)** → Day 20 → Day 21 → **day21_extra (CNN restoration / U-Net)**
- `day17b_autoencoder_intro/` sits between Day 17 and Day 18 by design; `day19b_network_visualization/` sits beside Day 19 (structure vs behavior); `day21_extra_cnn_restoration/` follows Day 21.
- Rationale: 17b reuses Day 17's conv/pool mechanics while fresh; Day 18's skip connections then arrive *before* the U-Net used in day21_extra; Day 19b reuses Day 17's `SimpleCNN` + Day 18's `SimpleResNet` to teach the architecture-visualization toolchain.

**Prerequisites already in hand:** full CIFAR-10 augmentation+normalization pipeline (Day 15), `nn.Module`/autograd/optimizers/losses (Days 4–9), BatchNorm + LR scheduling (Day 13), checkpointing + early stopping (Day 14). CIFAR-10 data cached at `day15/data/cifar-10-batches-py`. Device is `mps` on this Mac.

---

## 🖥️ Environment & Data Setup

- **Venv:** `.venv/` — Python 3.11.8, `torch` + `torchvision 0.27.0`. In Jupyter use the `.venv (3.11.8)` kernel; from the CLI use `.venv/bin/python`.
- **Device:** `torch.device("mps" if torch.backends.mps.is_available() else "cpu")` → Apple GPU.
- **Shared datasets — DO NOT re-download.** CIFAR-10 lives at `day15/data/cifar-10-batches-py` (+ `cifar-10-python.tar.gz`); MNIST lives at `day14/data/MNIST/raw`. Every CNN-arc notebook uses `root="./data"` **relative to its own folder**, so each folder has a `data` symlink to the existing copy:
  - `day16/ … day21/`, `day21_extra_cnn_restoration/`: `data -> ../day15/data`
  - `day17b_autoencoder_intro/data/`: `cifar-10-batches-py` + `cifar-10-python.tar.gz` -> Day 15 copy; `MNIST -> ../../day14/data/MNIST`
- **Verified:** `torchvision.datasets.CIFAR10(root="./data", download=True)` and `MNIST(...)` load fully offline (torchvision skips `download` when data is present). Leave `download=True` in the notebooks — it's a safe no-op. (Day 16 was previously blocked by a fresh 170 MB re-download; the symlink fixed it.)
- These `data` symlinks are intentionally ignored by `.gitignore` (the `data` pattern) so they are never committed.

---

## 💡 Frequently Asked Questions (Anticipated)

### Day 16 Q&A Topics Covered (see `LEARNING_NOTES_DAY16.md`):
- **§1 vs §2 output size:** §1 shrinks (4→2) because `padding=0`; §2 stays 32×32 because `F.conv2d(..., padding=1)`. Same 3×3 kernel, stride 1 — padding is the only difference. Formula: `out = (in + 2p − k)/s + 1`.
- **`F.conv2d` vs `nn.Conv2d`:** function (stateless, fixed kernels) vs Module (owns learnable weights, wraps `F.conv2d`). `nn.Conv2d`'s "4 numbers" are hyperparameters describing the shape `(C_out,C_in,kH,kW)` of ONE 4-D weight tensor — not 4 args.
- **`k.view(1,1,3,3)`:** reshape = `np.reshape`; prepends channel dims so a (3,3) kernel matches F.conv2d's required 4-D weight. Same as `unsqueeze(0).unsqueeze(0)`.
- **`out_channels=16` ≠ 16 layers:** it's 16 parallel filters/feature maps (a *width*). Distinguish tensor channel-depth (C in C×H×W) from network depth (# stacked layers).
- **Float compare gotcha:** `nn.Conv2d` (default bias) vs `F.conv2d` (no bias) differ; even matched, results differ ~2.4e-6 → use `torch.allclose(a, b, atol=1e-5)`, never `==`.
- **§6 feature maps (random conv):** intent = the "before" picture; random filters are meaningless, contrasting §2 (hand-made) and Day 17 §4 (learned). Bug fixed: `range(1,10)` over 8 channels → `IndexError`; now `out_channels=9` + `range(1, features.shape[1]+1)`.

### Day 17 Q&A Topics Covered (see `LEARNING_NOTES_DAY17.md`):
- **Why the stack shrinks 32→4×4 while channels grow 3→128:** trade *where* for *what*. Conv `3×3 + padding=1` keeps size; `MaxPool(2)` halves it; depth compounds the receptive field to ~**22×22** of the 32×32 input. Downsample for (a) compute, (b) translation invariance, (c) generalization. Three stride-2 pools → `32/2³ = 4`; flatten = `128·4·4 = 2048`.
- **Flatten & the batch dim:** `nn.Flatten()` defaults `start_dim=1`, so the **batch dim (axis 0) is preserved**: `(N,128,4,4) → (N,2048) → (N,10)`. Same as MLP (output `(batch, classes)`). **Trap:** the `128` in `nn.Linear(128*4*4, 256)` is the **channel count**, *not* the batch size — it merely coincidentally equals `BATCH_SIZE = 128`. `in_features` = channels × H × W.
- **`BatchNorm2d` stats:** normalize **per channel** using mean/var over **(N, H, W)** (vs BN1d over N). Running stats shape `(C,)`. `train()` uses batch stats and updates running via EMA `running = (1−m)·running + m·batch` (default `m=0.1`); `eval()` uses running stats. Verified: first-pass `running_mean[0] = −0.0122 = 0.1 × (−0.1216)` batch mean.
- **Notebook §4 plots _weights_, not activations:** `model.features[0].weight` is `(32, 3, 3, 3)` = 32 filters, each `(3,3,3)` = three RGB 3×3 kernels **summed** (verified manual `−0.50426900` == PyTorch `−0.50426894`). `permute(1,2,0)` → 3×3 RGB thumbnail; min-max normalize is display-only. Only **layer 1** is `imshow`-able because only its input is RGB; deeper layers → visualize *activations* (forward hooks, Day 19).
- **MLP works on MNIST but not CIFAR-10:** MNIST digits are centered/rigid/clutter-free → pixel *position* is a reliable feature, so position-locked weights suffice. CIFAR objects shift in scale/pose/position with clutter → the MLP lacks **locality** and **translation equivariance** (each position has private weights). CNN weight sharing builds invariance in. Evidence (Ex 2, equal ~620k params): **CNN 49.76% vs MLP 39.24%**.
- **Coupling trap:** adding/removing a conv block or pool changes the flatten size (`128·4·4 = 2048` → `256·2·2 = 1024`), so the head's `Linear` `in_features` must change.
- **Worked exercises (reduced-budget companion):** `DeeperCNN` 654,346 params / 54.12%; `GAPCNN` (`AdaptiveAvgPool2d(1)`) 94,986 params (−84.7%) / 41.62%; feature maps `32×16×16 → 64×8×8 → 128×4×4`.

### Day 17b Q&A Topics Covered (see `LEARNING_NOTES_DAY17B.md`):
- **Classifier → autoencoder:** only the **head** (class logits → image) and the **loss** (CrossEntropy → MSE) change; the backbone is identical. The input is its own label → **self-supervised**.
- **"Latent" and "auto":** the 128-D bottleneck is the *latent* (hidden) code (JPEG-file analogy); **auto** = *self*, because the target is the input itself. An encoder–decoder becomes an *auto*encoder when it reconstructs its own input.
- **Two models & params:** LinearAE **468,368** (784→256→128→256→784, `Sigmoid`; 20 ep: train MSE 0.00335 / val 0.00345). ConvAE **661,795** (encoder 93,696; fc 526,464 ≈ **80%**; decoder 41,635). `ConvTranspose2d(k=2,s=2)` doubles H,W → `4→8→16→32`.
- **Loss:** MSE = pixel-level squared error; PSNR = `10·log10(MAX²/MSE)`; L1 → sharper than MSE.
- **Image-to-image generalization:** same encoder–decoder + per-pixel loss; only the **target slot** changes (denoise→clean, blur→sharp, super-res→HR, colorize, pix2pix, medical). Targets from **synthesized pairs** (`day21_extra`'s `DegradedCIFAR10` returns `(degraded, clean)`) or **unpaired** (CycleGAN). Caveats: MSE is blurry → perceptual/adversarial losses; tight bottleneck → **U-Net skip connections**.
- **Reading the results:** MNIST looks close (easy data, ~6× compression; error concentrated at high-frequency **edges**; `cmap="hot"` exaggerates tiny ~0.003 errors). CIFAR looks fuzzy (3072 values, ~24× compression, MSE averaging → grey, normalize/clamp, ConvTranspose checkerboard, only 20 ep) — expected, and motivates U-Nets / better losses.
- **`requires_grad`/`.numpy()` bug:** `reconstructions` inherits the grad tag; `imshow`→`.numpy()` is forbidden on it. `model.eval()` does **not** help. Fix: wrap inference in `torch.no_grad()` **and** `.detach()` before plotting.

### Day 18 Q&A Topics Covered (see `LEARNING_NOTES_DAY18.md`):
- **VGG insight:** stack small 3×3 convs instead of one big filter. Two 3×3 convs = one 5×5's receptive field but `18C²` vs `25C²` params (−28%); three 3×3 = one 7×7 (`27C²` vs `49C²`, −45%) and more ReLUs (more non-linear). `vgg_block(in,out,num_convs)` = `num_convs × (Conv3×3+BN+ReLU)` + `MaxPool`; `in_channels if i==0 else out_channels` makes the *first* conv change depth and the rest keep it fixed. `nn.Sequential(*layers)` unpacks a **list** into positional args.
- **VGGNet facts (measured):** `1,149,770` params (features `1,147,200`, GAP head `2,570`); trace `(1,3,32,32)→(1,64,16,16)→(1,128,8,8)→(1,256,4,4)→(1,10)`. Head is GAP + `Linear(256→10)` (no flatten-coupling trap).
- **Degradation problem:** deep *plain* nets get worse on **train and test** error — an **optimization** problem (vanishing/exploding gradients), not overfitting.
- **Residual learning:** learn `F(x)=H(x)−x`, output `= F(x)+x`. If `F(x)=0` the block is identity ⇒ *adding layers can never hurt*; the `+` gives gradients a direct path.
- **Shortcut modes:** same shape (`stride=1`, `in==out`) → `nn.Sequential()` empty = **pure identity, 0 params**. Shape change → **projection** shortcut `1×1 Conv(stride=stride)+BN` (e.g. `8,448` params for `64→128,s2`). `bias=False` because the following BN supplies the shift.
- **SimpleResNet stage rule:** first block of each stage uses `stride=2` + doubles channels (→ projection), rest are identity blocks. `1,856+147,968+525,568+2,099,712+2,570 = 2,777,674` params; trace `(1,64,32,32)→(1,64,32,32)→(1,128,16,16)→(1,256,8,8)→(1,256,1,1)→(1,10)`.
- **VGG vs ResNet lesson:** ResNet expected to train *better* despite **more** params — architecture (skip connections) beats raw parameter count. Both trained with Adam lr=1e-3, wd=1e-4, CosineAnnealingLR, 30 ep.
- **Ordering rationale (18→19→20→21):** 18 gives the blocks, 19 the microscope, 20 the competitor, 21 the engineer. Four threads: skip-connection, hierarchy, swap-head-and-loss, reuse-vs-build. The skip connection is the linchpin (reappears in the U-Net and every Transformer block).

### Day 20 Q&A Topics Covered (see `LEARNING_NOTES_DAY20.md`):
- **"Why does `print(resnet18)` look like a VGG, not the ResNet from Day 18?"** Three illusions of `print()`: (1) it renders the **module tree, not the data flow** — the skip connection is `out += identity` inside `forward()`, never a registered module, so it's invisible (verified: the printout is **84 lines** and contains **no `+`/`Add`**); (2) the **stem is genuinely VGG-flavored** (`Conv2d(3,64,7×7,s2)` + `MaxPool2d(3×3,s2)`), because ResNet kept VGG's skeleton ("VGG + skip connections"); (3) the **ResNet tells** are stage naming (`layer1…layer4`) and the **`(downsample)`** submodule (the projection shortcut from Day 18).
- **torchvision `resnet18` ≡ an up-scaled Day 18 `SimpleResNet`** — same `BasicBlock`/`ResidualBlock`; **`layer1/2/3` are parameter-for-parameter identical** (`147,968 / 525,568 / 2,099,712`). `SimpleResNet` = `resnet18` minus `layer4`, with a 3×3 stem and a 10-class head. Totals: **`SimpleResNet` 2,777,674 vs `resnet18` 11,689,512**.
- **`resnet18` param breakdown (verified):** `conv1 9,408` · `bn1 128` · `layer1 147,968` · `layer2 525,568` · `layer3 2,099,712` · `layer4 8,393,728` (≈**71.8%**) · `fc 513,000` (**4.4%**) · total **11,689,512**. Shape trace `(1,3,224,224)→(1,64,56,56)→(1,128,28,28)→(1,256,14,14)→(1,512,7,7)→(1,512,1,1)→(1,1000)`. Also `resnet34` 21,797,672 · `resnet50` 25,557,032 (bottleneck blocks).
- **BasicBlock internals & stage rule:** identical to Day 18 — `out = relu(bn2(conv2(relu(bn1(conv1(x)))))) + downsample(x)`; first block of a stage uses `stride=2` + double channels (→ `downsample` projection), the rest are identity blocks (0 params). `layer2/3/4` each have exactly one `(downsample)`; `layer1` has none.
- **Why/how ResNet beats VGG:** **optimization, not capacity** — skip connections fix the **degradation problem** (deep *plain* nets get worse on **train and test** error; vanishing/exploding gradients, not overfitting); identity is free (`F(x)→0`); keeps VGG's 3×3 efficiency; GAP head (`fc` = 4.4% of params). **vs `SimpleResNet`:** same family — the gap is **depth (`layer4` ≈72% of weights) + 7×7 stem + pretrained ImageNet weights**, not a new mechanism.
- **Transfer-learning workflow:** replace the head (`model.fc = nn.Linear(512, 10)`); **Strategy A** feature extraction (freeze all, train only `fc` — 5,130 params) vs **Strategy B** fine-tuning (all params, **differential LR** backbone `1e-4` / head `1e-3` to avoid **catastrophic forgetting**); freeze **by stage** (unfreeze `layer4` first). Resize to 224×224 and normalize with **ImageNet** mean/std (not CIFAR's) to match pretraining.

### Day 21 extra Q&A Topics Covered (see `LEARNING_NOTES_DAY21_EXTRA.md`):
- **U-Net fundamentals:** encoder (contracting) → bottleneck → decoder (expanding) with **`torch.cat` skip connections** bridging matching resolutions; exists to output a **same-size image** while recovering the **high-frequency detail** pooling destroyed (the fix for Day 17b's fuzzy ConvAE). `ConvTranspose2d(k=2,s=2)` upsampling; `dec*`'s first conv takes `2×` channels because the preceding `cat` doubled them. Params **4,818,051** (`base_channels=64`); the **bottleneck is ~49%** of weights.
- **U-Net skip vs ResNet skip (Day 18):** U-Net = **concatenate** `cat` (cross-network, unequal channels OK, **detail** path); ResNet = **add** `F(x)+x` (within-stage, equal channels → 1×1 projection when shapes differ, **gradient/depth** path). Same family, different tool — and they compose (residual U-Nets).
- **Degradation pipeline:** `add_gaussian_noise` / `apply_gaussian_blur` (PSF) / `bicubic_downscale`; `DegradedCIFAR10` returns `(degraded, clean)` on the fly (denormalize → degrade → renormalize), noise re-randomized each epoch.
- **PSNR & SSIM from scratch:** PSNR = `10·log10(1/MSE)` (dB, = camera SNR; 6 dB = 4× MSE); SSIM via 11×11 Gaussian window (luminance/contrast/structure), C1=0.01², C2=0.03². SSIM is **blur-tolerant but noise-sensitive**.
- **"Why did PSNR/SSIM/visuals show deblur *easier* than denoise despite the 'harder' verdict?"** — the verdict is **printed, not computed**. Absolute PSNR isn't a difficulty measure when baselines differ. **Measured degraded-input baselines** (reproducing the notebook's exact functions, CIFAR-10 test): **noise σ=0.2 → 14.78 dB / SSIM 0.489** vs **blur σ=1.5 → 21.06 dB / SSIM 0.800** (so deblur starts ~6 dB ahead). Four causes: (1) **unmatched severity**; (2) blur is a **fixed deterministic PSF** (one learnable inverse) vs **stochastic noise** (irreducible floor / bias–variance); (3) **metric bias** (PSNR penalizes broadband noise; SSIM is blur-insensitive); (4) the visual grids show the same artifact. Correct signal = **ΔPSNR over the degraded input** (denoiser works harder); match severity + compare matched classical baselines to actually test the claim.


### Day 10 Q&A Topics Covered:
- **torchvision vs torch**: CV toolkit vs core DL framework
- **DataLoader type**: Custom PyTorch iterator (not a basic Python type)
- **Batch vs Epoch**: Model improves at each `optimizer.step()`, not epoch boundaries
- **MLP vs CNN vs ViT**: MLP=Linear only, CNN=Conv, ViT=Attention
- **nn.Linear with batches**: Accepts 2D input (batch, features), preserves batch dim
- **Validation vs Test**: Validation for tuning, Test for final eval only
- **VR/AR Eye-tracking**: Industry uses lightweight CNN (MobileNet)

**"Why do we need both `__init__` and `forward`?"**
- `__init__`: sets up the architecture (layers, parameters)
- `forward`: defines the computation (data flow)
- PyTorch automatically calls `forward()` when you call the model as a function

**"What's the difference between `nn.Sequential` and defining `forward`?"**
- `nn.Sequential`: simple stack of layers, no complex logic
- Custom `forward`: allows branching, multiple inputs/outputs, complex architectures

**"Why divide by √d_k in attention?"**
- Prevents softmax from saturating when d_k is large
- Large dot products → large magnitude → softmax → near-one-hot → gradients vanish

**"When to use CNN vs ViT?"**
- CNN: small datasets, need translation invariance, lower compute
- ViT: large datasets, need global context, have GPU resources

---

## 🔄 Update Log

- **2026-06-01:** Initial creation of AI_CONTEXT.md after digesting the full codebase
- **2026-06-03:** Completed Day 10, updated progress, added Q&A summary from Day 10 session
- **2026-09-27:** Entered Phase 4 (CNNs). Updated progress to Day 15 complete / Day 16 current. Recorded CNN arc ordering incl. extension notebooks (day17b, day21_extra). Opened a running Q&A thread for Days 16–21 to be consolidated into LEARNING_NOTES_DAY16–21.
- **2026-09-29:** Completed Day 16. Wrote `LEARNING_NOTES_DAY16.md` (convolution operation, F vs nn API, padding/stride, channels≠layers, weight sharing, feature maps). Fixed an off-by-one bug in Day 16 §6 (`nn.Conv2d(3,8)` + `range(1,10)` → `out_channels=9` + shape-anchored loop). Set up shared-dataset `data` symlinks for the whole CNN arc; documented env/data setup.
- **2026-09-30:** Completed Day 17. Wrote `LEARNING_NOTES_DAY17.md` (CNN block blueprint, resolution↓/channels↑ rationale + receptive fields, Flatten/batch-dim question, BatchNorm2d stats, weights vs activations, MLP-vs-CNN intuition, optics connections). Created and **executed** worked exercise solutions in `day17/day17_build_first_cnn_with_exercises.ipynb` (DeeperCNN, CNN-vs-MLP, feature-map hooks, GAP head; reduced-budget companion run: main 6 ep, exercises 3 ep on 8k). Added Day 17 Q&A summary here; progress advanced to day17b/Day 18.
- **2026-10-01:** Completed Day 17b (Autoencoder Intro). Fixed a `requires_grad`/`.numpy()` plotting bug in `day17b_autoencoder_intro.ipynb` (wrapped inference in `torch.no_grad()` and added `.detach()` in both visualization cells). Recorded the discussion into `LEARNING_NOTES_DAY17B.md` (classifier→AE swap, latent/auto naming, both models + verified param counts, MSE/L1, image-to-image generalization & target-slot framing, MNIST-vs-CIFAR reconstruction analysis, the autograd-plotting bug, optics connections, self-check). Added Day 17b Q&A summary here; progress advanced to Day 18.
- **2026-10-02:** Completed Day 18 (CNN Architectures — VGG blocks & ResNet skip connections). Wrote `LEARNING_NOTES_DAY18.md` (VGG small-filter insight + param ratios; the degradation problem; residual learning `F(x)+x`; identity vs projection shortcuts; SimpleResNet stage pattern; verified VGG `1,149,770` vs ResNet `2,777,674` param counts and shape traces; the 18→19→20→21 ordering rationale; four cross-arc threads; optics connections; self-check). No notebook code changes needed. Added Day 18 Q&A summary here; progress advanced to Day 19.
- **2026-10-02 (cont.):** Built **Day 19b extension** (`day19b_network_visualization/`) — the *structure* counterpart to Day 19's *behavior* visualization. Added `requirements-viz.txt` (`torchinfo`, `visualtorch`, `netron`, `onnx`; pure-pip, fully encapsulated in `.venv`) and a notebook covering: `torchinfo.summary` tables; VisualTorch `lenet`/`flow`/`graph` figures for `SimpleCNN`/`SimpleResNet`/`resnet18` (paper-quality, no system deps); guarded `torchviz`/`torchview` cells (need Graphviz `dot`); ONNX export → Netron; TensorBoard `add_graph`; and a PlotNeuralNets (LaTeX) appendix. Notebook **executed headless to verify** (all cells pass; 6 figures + `simplecnn.onnx` + TB logs generated). Updated `.gitignore` (`assets/`, `runs/`, `*.onnx`), `CURRICULUM.md`, and this file. Wrote `LEARNING_NOTES_DAY19B.md`.
- **2026-10-02 (cont.):** Completed Day 20 (Transfer Learning & Fine-Tuning). Wrote `LEARNING_NOTES_DAY20.md` (Day 20 Q&A log; why `print(resnet18)` hides the skip and looks VGG-ish; the torchvision `resnet18` ≡ up-scaled Day 18 `SimpleResNet` map with verified per-stage params — `layer1/2/3` identical; layer-by-layer shape trace; Day 18↔torchvision naming table; ResNet-vs-VGG and `resnet18`-vs-`SimpleResNet` superiority analysis; transfer-learning workflow / differential LR; optics connections; self-check). No notebook code changes needed (Section 1 verified read-only: 84-line printout, no `+`/`Add`, total `11,689,512`). Added Day 20 Q&A summary here; progress advanced to Day 21.
- **2026-10-05:** Completed **Day 21 (extra)** — CNN Image Restoration (`day21_extra_cnn_restoration/`). Wrote `LEARNING_NOTES_DAY21_EXTRA.md` with a Q&A log and 12 reference sections: image-to-image framing (swap head + loss, self-supervised `(degraded, clean)` pairs); U-Net fundamentals (encoder/bottleneck/decoder, `torch.cat` skips, verified **4,818,051** params with per-block breakdown, bottleneck ≈49%); **U-Net vs ResNet (Day 18)** — concatenative vs additive skips (cross-level/detail vs within-stage/depth); the degradation pipeline (`add_gaussian_noise`/`apply_gaussian_blur`/`bicubic_downscale`, `DegradedCIFAR10`); PSNR & SSIM from scratch; and the **denoising-vs-deblurring analysis** — explaining why the run's PSNR/SSIM/visuals appeared to contradict the printed "deblur is harder" verdict. **Empirically measured** the degraded-input baselines by re-running the notebook's exact degradation functions over the CIFAR-10 test set: **noise σ=0.2 → 14.78 dB / SSIM 0.489** vs **blur σ=1.5 → 21.06 dB / SSIM 0.800** (~6 dB head start for deblur), identifying the four confounds (unmatched severity, deterministic PSF vs stochastic noise, metric bias, visual artifact) and prescribing **ΔPSNR-over-input** + severity matching as the correct difficulty test. No notebook code changes. Added Day 21-extra Q&A summary here; progress advanced past Day 21.
