# Learning Notes — Day 10: Data Loading & Preprocessing

> **Date:** 2026-06-03
> **Curriculum:** 30-Day Deep Learning Curriculum
> **Topics Covered:** torchvision, DataLoader, Dataset, MNIST, Batches, Training Loops, MLP vs CNN vs ViT

---

## 1. Environment Fix: torchvision Import Error

### Problem
```
ModuleNotFoundError: No module named '_lzma'
```
This occurred because Python 3.11.8 installed via pyenv was compiled without lzma support.

### Solution Applied
1. Installed `xz` via Homebrew: `brew install xz`
2. Reinstalled Python 3.11.8 with proper flags:
   ```bash
   export LDFLAGS="-L$(brew --prefix xz)/lib"
   export CPPFLAGS="-I$(brew --prefix xz)/include"
   pyenv install 3.11.8 --force
   ```
3. Recreated virtual environment: `rm -rf .venv && python3 -m venv .venv`
4. Reinstalled requirements: `pip install -r requirements.txt`

**Result:** torch 2.12.0 and torchvision 0.27.0 now import successfully.

---

## 2. What is torchvision and Why Is It Needed?

### PyTorch vs torchvision

| Library | Purpose | Provides |
|---------|----------|----------|
| **torch** | Core DL framework | Tensors, autograd, nn.Module, optimizers |
| **torchvision** | Computer vision toolkit | Datasets, transforms, models, utilities |

### Key Components

#### 1. Datasets — Pre-built dataset loaders
```python
import torchvision.datasets as datasets

mnist = datasets.MNIST(root='./data', train=True, download=True)
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True)
```
Automatically downloads and loads standard datasets.

#### 2. Transforms — Image preprocessing pipelines
```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),           # PIL → tensor, [0,255] → [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean & std
])
```

#### 3. Models — Pre-trained architectures
```python
import torchvision.models as models

resnet = models.resnet50(pretrained=True)  # For transfer learning (Day 20)
```

### Why Needed in This Curriculum

| Days | Topic | torchvision Usage |
|------|-------|-------------------|
| Day 10 | Data Loading | `datasets`, `transforms`, `DataLoader` |
| Day 11 | MLP on MNIST | `datasets.MNIST` |
| Day 15+ | CIFAR-10 | `datasets.CIFAR10`, `transforms` |
| Day 18 | CNN Architectures | `models.vgg`, `models.resnet` |
| Day 20 | Transfer Learning | Pre-trained `models` |

---

## 3. MNIST Data Location

In the notebook, `root="./data"` specifies the download location:

**Full path:** `/Users/pengzhao/Documents/Projects/DL_learning/data/`

**Directory structure after download:**
```
data/
  MNIST/
    raw/        ← Downloaded files (train-images-idx3-ubyte.gz, etc.)
    processed/  ← Processed PyTorch tensors
```

---

## 4. Label Type vs Classes

### What's the Difference?

| Concept | What it shows | Example Output |
|---------|---------------|----------------|
| `label type` | Data type of one label (format) | `<class 'int'>` |
| `classes` | List of all category names (meaning) | `['0', '1', ..., '9']` |

### Code from Notebook (Section 4)
```python
print(f"Label type:  {type(train_dataset[0][1])}")   # <class 'int'>
print(f"Classes:     {train_dataset.classes}")         # ['0', '1', ..., '9']
```

### Why Both?

For MNIST, it seems redundant (label `5` → class `'5'`). But for datasets like CIFAR-10:
- `label 2` → `class 'bird'` (not obvious without `classes`)
- Useful for display: `"Predicted: bird"` not just `"Predicted: 2"`
- Enables generic code: `len(dataset.classes)` works for any dataset

---

## 5. Validation Set vs Test Set (Section 6)

### Three Data Splits in the Notebook

| Split | Source | Usage |
|-------|--------|-------|
| **Training subset** (80%) | Split from `train_dataset` | Update model weights via backprop |
| **Validation subset** (20%) | Split from `train_dataset` | Monitor overfitting, tune hyperparameters |
| **Test set** | `torchvision.datasets.MNIST(train=False)` | Final evaluation (once, at end) |

### Key Differences

| Aspect | Validation Set | Test Set |
|---------|----------------|----------|
| **Source** | Original training data | Separate official test split |
| **Usage** | Checked during training | Used only once at end |
| **Risk** | Part of training distribution | Must stay independent |

**They're NOT redundant** — using test set as validation would overfit to the test set, making final evaluation meaningless.

---

## 6. Section 7 Deep Dive: Training Loop Evolution

### 6.1 DataLoader Data Type

**DataLoader is NOT a basic Python type** — it's a custom PyTorch class:
```python
>>> type(train_loader)
<class 'torch.utils.data.dataloader.DataLoader'>
```

| Property | List | DataLoader |
|----------|------|------------|
| Indexable? | `mylist[5]` ✅ | `loader[5]` ❌ |
| Know size? | `len(mylist)` ✅ | `len(loader)` ✅ (number of batches) |
| Memory | Holds all data | Lazy — yields batches on demand |

Think of it as a **lazy iterator** (like a vending machine) that yields batches one at a time.

### 6.2 enumerate() vs next(iter())

```python
# enumerate() — loops through ALL batches
for batch_idx, (images, labels) in enumerate(train_loader):
    print(batch_idx)  # 0, 1, 2, ... 937

# next(iter()) — gets just ONE batch
images, labels = next(iter(train_loader))  # Only first batch
```

### 6.3 Day 09 vs Day 10: The Key Difference

#### Day 09 (Batch Gradient Descent)
```python
for epoch in range(200):           # Loop over epochs
    logits = model(X)              # ALL 300 samples at once
    loss = loss_fn(logits, y)
    optimizer.step()                # 1 weight update per epoch
```
- Samples per iteration: **ALL 300**
- Updates per epoch: **1**

#### Day 10 (Mini-Batch Gradient Descent)
```python
for batch_idx, (images, labels) in enumerate(train_loader):  # Loop over batches
    outputs = simple_model(images)  # 64 samples
    loss = loss_fn(outputs, labels)
    optimizer.step()                 # 1 weight update per batch
```
- Samples per iteration: **64**
- Updates per epoch: **~938**

### 6.4 When Does the Model Improve?

**The model improves at EVERY `optimizer.step()` call, NOT at epoch boundaries.**

```
Day 09: 200 epochs × 1 update = 200 total updates
Day 10: 1 epoch × 938 updates = 938 updates (in ONE epoch!)
```

The epoch is just a concept ("one pass through data"). The learning happens at each batch when `optimizer.step()` is called.

### 6.5 Dimension Trace (Section 7)

For a batch with `BATCH_SIZE = 64`:

| Step | Code | Shape | Explanation |
|------|------|-------|-------------|
| Batch from loader | `images, labels = next(iter(train_loader))` | `(64, 1, 28, 28)` | 64 images, 1 channel, 28×28 |
| Flatten | `images = images.view(64, -1)` | `(64, 784)` | Each image → 784-dim vector |
| After model | `outputs = simple_model(images)` | `(64, 10)` | 64 predictions × 10 class logits |

### 6.6 nn.Linear with 2D Input

**Yes, it's designed for this!** `nn.Linear(784, 10)` works with any input where the last dimension is 784:

| Input Shape | Output Shape | Use Case |
|-------------|---------------|----------|
| `(784,)` | `(10,)` | Single sample (Days 4-9) |
| `(64, 784)` | `(64, 10)` | Batch of 64 (Day 10+) |

The batch dimension is **preserved** — same transformation applied independently to each sample.

---

## 7. MLP vs CNN vs ViT: Architecture Comparison

### Definitions

| Architecture | Core Layers | Description |
|--------------|-------------|-------------|
| **MLP** | `nn.Linear` | Fully connected layers only, input flattened |
| **CNN** | `nn.Conv2d` | Convolutional layers, preserves spatial structure |
| **ViT** | `nn.MultiheadAttention` | Attention-based, preserves spatial structure |

**Note:** MLP is a SPECIFIC architecture (not generic "multi-layer model").

### Complete Comparison

#### MLP (Multi-Layer Perceptron)

| Aspect | Details |
|--------|---------|
| **Pros** | Simplest to understand, works on tabular data, universal approximator |
| **Cons** | Loses spatial structure, parameter explosion, not translation invariant |
| **Best for** | Tabular data, small feature vectors, learning basics (Day 11) |
| **Limitations** | Poor on images, doesn't scale to large images |

#### CNN (Convolutional Neural Network)

| Aspect | Details |
|--------|---------|
| **Pros** | Translation invariant, parameter efficient, hierarchical features |
| **Cons** | Local receptive field only, fixed kernel size |
| **Best for** | Image classification, object detection (Days 17-21) |
| **Limitations** | Struggles with long-range dependencies |

#### ViT (Vision Transformer)

| Aspect | Details |
|--------|---------|
| **Pros** | Global context from layer 1, excellent scaling, multimodal |
| **Cons** | Data hungry, no spatial bias, O(n²) compute |
| **Best for** | Large-scale recognition, global context (Days 25-28) |
| **Limitations** | Poor on small datasets, computationally expensive |

### Quick Decision Guide

```
Small dataset (<10K images)  → CNN
Medium dataset (10K-100K)    → CNN (or CNN + transfer learning)
Large dataset (>100K)        → CNN or ViT (ViT shines)
Tabular/feature data          → MLP
Need global context           → ViT
Limited compute               → CNN
```

---

## 8. Industry Application: Eye-Tracking in VR/AR

### The Problem
Real-time gaze tracking in VR/AR headsets with constrained mobile hardware.

### Constraints

| Constraint | Requirement |
|------------|-------------|
| Latency | <10ms (90+ FPS) |
| Compute | Mobile-class chip on headset |
| Input | Eye region crops (~60×60 to 120×120 pixels) |

### Best Choice: Lightweight CNN (MobileNet-style)

| Model | Verdict |
|-------|---------|
| **MLP** | ❌ Input 64×64×3 = 12K features → first layer too massive |
| **ViT** | ❌ O(n²) attention too heavy for mobile VR chips |
| **CNN (MobileNet)** | ✅ Industry standard — efficient, fast, proven |

### Why CNN Wins
- **Efficient:** ~200K-5M parameters (depthwise separable convolutions)
- **Fast:** Can run 1000+ FPS on mobile GPUs
- **Proven:** Meta Quest Pro, HTC Vive, Tobii all use CNN variants

### Real-World Examples
- **Meta Quest Pro:** Custom lightweight CNN
- **Apple Vision Pro:** Hybrid CNN+Transformer
- **Tobii (industry leader):** Proprietary CNN variants

### Relevance to Curriculum
- **Days 17-21 (CNN):** Directly applicable — this is what industry uses
- **Days 25-28 (ViT):** Overkill for basic tracking, but useful for larger-scale gaze understanding

---

## 9. Key Takeaways from Day 10

| Concept | Summary |
|---------|---------|
| **torchvision** | CV toolkit for datasets, transforms, models |
| **DataLoader** | Lazy iterator yielding batches, not indexable like a list |
| **Batch training** | 938 updates per epoch (vs 1 in Batch GD) |
| **Learning moment** | Model improves after each `optimizer.step()`, not at epoch end |
| **MLP vs CNN vs ViT** | MLP=Linear only, CNN=Conv, ViT=Attention |
| **nn.Linear** | Accepts 2D input (batch, features) — batch dim preserved |
| **Validation vs Test** | Validation for tuning, Test for final eval only |
| **VR/AR Eye-tracking** | Industry uses lightweight CNN (MobileNet) |

---

## 10. Common Code Patterns (Day 10 Style)

### Complete Training Loop Structure
```python
num_epochs = 10

for epoch in range(num_epochs):              # Outer: epochs
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):  # Inner: batches
        # Flatten images
        images = images.view(images.size(0), -1)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch} done")
```

### Data Loading Pattern
```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

---

## 11. Questions for Self-Check

1. **What's the difference between torch and torchvision?**
2. **Where does MNIST download to when using `root="./data"`?**
3. **Why do we need both `label type` and `classes`?**
4. **How is DataLoader different from a Python list?**
5. **When does the model actually improve — at epoch end or during batches?**
6. **What's the shape of images after flattening a batch of 64 MNIST images?**
7. **Why is CNN better than MLP for images?**
8. **What's the best architecture for VR/AR eye-tracking and why?**

---

**Next Up:** Day 11 — First Neural Network (MLP on MNIST) 🎉