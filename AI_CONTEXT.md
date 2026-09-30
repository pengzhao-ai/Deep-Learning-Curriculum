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

**Last completed:** Day 17 (Build Your First CNN) — notes in `LEARNING_NOTES_DAY17.md`; worked exercise solutions in `day17/day17_build_first_cnn_with_exercises.ipynb`.

**Currently on:** day17b (Autoencoder Intro) next — Phase 4: CNNs (Days 16–21)

**Next session goal:** Day 17b — convert the classifier to an image-to-image autoencoder (`day17b_autoencoder_intro/day17b_autoencoder_intro.ipynb`), then Day 18 (VGG/ResNet).

**CNN arc ordering (as studied):**
- Day 16 → Day 17 → **day17b (autoencoder bridge)** → Day 18 → Day 19 → Day 20 → Day 21 → **day21_extra (CNN restoration / U-Net)**
- `day17b_autoencoder_intro/` sits between Day 17 and Day 18 by design; `day21_extra_cnn_restoration/` follows Day 21.
- Rationale: 17b reuses Day 17's conv/pool mechanics while fresh; Day 18's skip connections then arrive *before* the U-Net used in day21_extra.

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
