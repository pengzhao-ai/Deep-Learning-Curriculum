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

**Last completed:** Day 10 (Data Loading & Preprocessing)

**Currently on:** Day 11 (First Neural Network - MLP on MNIST)

**Next session goal:** Complete Day 11 - Build and train MLP on MNIST

---

## 💡 Frequently Asked Questions (Anticipated)

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
