# Learning Notes — Day 11: First Neural Network (MLP on MNIST) 🎉

> **Date:** 2026-06-03 to 2026-06-04
> **Curriculum:** 30-Day Deep Learning Curriculum
> **Topics Covered:** CrossEntropyLoss in detail, total loss & accuracy computation, model resetting, MPS/GPU utilization, confusion matrix basics

---

## 1. CrossEntropyLoss & Loss/Accuracy Computation (Section 4 Deep Dive)

### The Two Key Lines

```python
total_loss += loss.item() * labels.size(0)         # Line A
correct += (outputs.argmax(1) == labels).sum().item()  # Line B
```

### Background: What Comes Out of the Model

The final layer outputs **10 numbers per image** (logits for digits 0–9). For a batch of 128 images:

| Tensor | Shape | Content |
|--------|-------|---------|
| `outputs` | `(128, 10)` | Logits: 128 predictions × 10 class scores |
| `labels` | `(128,)` | True digits e.g. `[7, 2, 1, 9, ...]` |

### Line A Breakdown: `loss.item() * labels.size(0)`

**What `loss` is:** `nn.CrossEntropyLoss()` returns the **average loss per sample** across the batch — a single scalar. If the batch has 128 images, `loss` is the mean of all 128 individual losses.

| Expression | Value (example) | Meaning |
|------------|-----------------|---------|
| `loss.item()` | `0.46` | Average loss per sample (Python float) |
| `labels.size(0)` | `128` | Batch size |
| `loss.item() * labels.size(0)` | `58.88` | **Total loss for this batch** |

**Why not just accumulate `loss.item()` directly?** Because the last batch may be smaller than the others. If you simply averaged per-batch averages, a tiny last batch (e.g. 56 samples) would get equal weight as a full batch (128 samples), giving a biased estimate.

**Example — correct vs incorrect averaging:**

| Batch | Samples | Avg Loss | Total Loss |
|-------|---------|----------|------------|
| 1     | 128     | 0.50     | 64.0       |
| 2     | 128     | 0.46     | 58.9       |
| 3     | 56      | 0.48     | 26.9       |

- **Weighted (correct):** (64.0 + 58.9 + 26.9) / (128 + 128 + 56) = **0.484** ✅
- **Unweighted (wrong):** (0.50 + 0.46 + 0.48) / 3 = **0.480** ❌ (biased by small last batch)

At the end:
```python
return total_loss / total  # Divides summed loss by total sample count → true average
```

### Line B Breakdown: `outputs.argmax(1)`

**`argmax(1)` finds the index of the largest value along dimension 1 (the class dimension).**

For each image's 10 logits, it picks the class with the highest score — that's the model's prediction.

```
outputs shape: (128, 10)
                        ┌──────────────────┐
image 0:  [-1.2, 2.3, 0.5, ...] → argmax(1) → 1  (class 1 has highest score)
image 1:  [ 3.1,-0.5,-0.2, ...] → argmax(1) → 0  (class 0 has highest score)
...
```

| Expression | Shape | Content |
|------------|-------|---------|
| `outputs.argmax(1)` | `(128,)` | Predicted class for each image, e.g. `[1, 0, 4, 7, 2, ...]` |
| `(outputs.argmax(1) == labels)` | `(128,)` | Boolean: `[True, False, True, True, ...]` |
| `.sum().item()` | scalar | Count of correct predictions (e.g. `122`) |

**Why dimension 1 and not 0 or 2?**

| Dim | What it indexes | `argmax` would mean |
|-----|-----------------|---------------------|
| **Dim 0** | Batch (128 images) | "For each class, which image scores highest?" — meaningless ❌ |
| **Dim 1** | Classes (10 per image) | "For each image, which class scores highest?" — exactly what we want ✅ |
| **Dim 2** | N/A (no dim 2) | Would error ❌ |

### What CrossEntropyLoss Does Internally

```python
nn.CrossEntropyLoss()  # Combines LogSoftmax + NLLLoss in one class
```

Behind the scenes, it:
1. Applies **Softmax** to convert logits into probabilities (sum to 1 per image)
2. Takes the **negative log** of the probability assigned to the **correct class**

```
logits:  [-1.2, 2.3, 0.5, -1.5, -0.3, 0.1, -2.0, 1.1, 0.8, -0.4]
                    │ true label = 3
                    ▼
Softmax → probabilities:  [0.02, 0.65, 0.11, 0.01, 0.05, 0.07, 0.01, 0.20, 0.15, 0.04]
                                                            ↑
                                                    prob of class 3 = 0.01

loss = -ln(0.01) ≈ 4.6  ← high because the model was very wrong
```

If the model was confident and correct:
```
prob of correct class = 0.92
loss = -ln(0.92) ≈ 0.08  ← low
```

The loss is low when the correct class gets a high probability, high when it doesn't.

---

## 2. Resetting the Model (Question 2)

### The Problem

After running Section 5 (training for 10 epochs), the model's weights have been updated. If you want to retrain with different hyperparameters, you need to **reset the weights**.

### What Each Section Does

| Section | Code | What It Does | Must Re-run? |
|---------|------|-------------|--------------|
| 1 | Imports, device setup | Import libraries, set device | Only if kernel restarted |
| 2 | `class MLP` + `model = MLP().to(device)` | **Creates model with fresh random weights** | **✅ Yes** |
| 3 | `loss_fn = nn.CrossEntropyLoss()` + `optimizer = optim.Adam(...)` | **Creates optimizer with fresh internal state** (momentum buffers, adaptive LR history) | **✅ Yes** |
| 4 | `def train_one_epoch(...)` + `def evaluate(...)` | Defines functions | ✅ Yes (function definitions) |
| 5 | Training loop | Runs the training | **✅ Yes** |

### Why Optimizer State Matters

`optim.Adam` maintains per-parameter **momentum buffers** and **adaptive learning rates** that accumulate over time. Even if you recreate the model with new weights, the Adam optimizer retains its internal state from the previous run. This means:

- **Re-run Sections 2 + 3 + 4 + 5** = truly fresh start
- **Skip Section 2** = continue training the already-trained model (not reset)
- **Skip Section 3** = optimizer carries stale momentum buffers from previous training

### Pro Tip for Experiments

For comparing multiple models, wrap everything in a function:

```python
def train_model(lr=1e-3, hidden1=256, hidden2=128):
    model = MLP(input_dim=784, hidden1=hidden1, hidden2=hidden2, num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # ... training loop ...
    return history
```

This avoids needing to manually re-run cells.

---

## 3. MPS/GPU Utilization on M3 Max

### Initial Observation

Apple MPS utilization was low during Day 11 MLP training. This is **expected and normal** for this stage.

### Root Cause: The Model is Too Small

| Property | Value |
|----------|-------|
| Model parameters | ~270K (very small) |
| Input size | 28×28 grayscale (trivial) |
| Computation per batch | Microseconds on GPU |
| **Bottleneck** | **CPU-side data loading and transfer, not GPU compute** |

The M3 Max's GPU finishes a batch almost instantly, then sits idle waiting for the next batch to be prepared and transferred. This is like asking a Ferrari to drive around a parking lot — the engine barely breaks idle regardless of how fast you load the cargo.

### Recommended DataLoader Settings

| Parameter | Default | Recommended (M3 Max) | What It Does |
|-----------|---------|---------------------|--------------|
| `BATCH_SIZE` | `128` | **`1024` or `2048`** | More samples per GPU launch → amortizes launch overhead |
| `num_workers` | (not set, default 0) | **`8` or `12`** | Parallel CPU data loading — preps batches while GPU works |
| `pin_memory` | (not set, default False) | **`True`** | Faster CPU→GPU memory transfer (benefits MPS too) |

### Updated DataLoader Code

```python
BATCH_SIZE = 1024  # Was 128

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=8, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=8, pin_memory=True)
```

### What You'll Actually See

These changes will **increase CPU utilization** (the workers are loading data in parallel) but **GPU utilization will remain similarly low**. The GPU simply doesn't have enough work to do.

### When GPU Utilization Will Improve

| Curriculum Day | Model | Images | Params | Expected GPU Util |
|---------------|-------|--------|--------|-------------------|
| **Day 11** | MLP | 28×28 gray | 270K | **~5-15%** — CPU/data bound |
| **Day 17** | Simple CNN | 32×32×3 | ~500K | **~20-40%** — more compute |
| **Day 18** | ResNet | 32×32×3 | 11M+ | **~50-70%** — real GPU work |
| **Day 25** | ViT | 32×32×3 | 10M+ | **~60-80%** — attention is compute-heavy |

### On M3 Max with 36GB Unified Memory

Apple's M-series uses **unified memory** — the GPU and CPU share the same 36GB pool. This is advantageous because:
- You don't need to worry about GPU VRAM limits for any curriculum models (they're all small)
- `pin_memory=True` is still beneficial — it pins host memory for faster GPU access
- Larger batch sizes (1024–2048) work fine with MNIST (28×28 images are 784 floats each = ~3KB per image × 2048 = ~6MB — negligible)

### The Real Metric: Wall Clock Time

GPU utilization is a secondary metric. What matters is **how fast training completes**. Even if GPU utilization didn't budge, the training wall clock time likely decreased with `num_workers` and larger batch sizes — that's the real win.

---

## 4. Confusion Matrix (Section 7)

### What Is a Confusion Matrix?

A **confusion matrix** is a table that shows how a classifier's predictions break down **class-by-class**. It tells you **"which classes am I confusing with which?"** — not just "how many did I get right?"

### Structure

For MNIST (10 digits), it's a **10×10 grid**:

```
              Predicted digit →
              0    1    2    3    4    5    6    7    8    9
Actual   0  [ 980    0    1    0    0    0    0    1    0    0 ]
digit    1  [   0 1132    1    0    0    0    1    0    1    0 ]
↓        2  [   1    1 1026    0    0    0    0    2    0    0 ]
         3  [   0    0    1 1002    0    1    0    1    3    0 ]
         4  [   0    0    0    0  978    0    1    0    1    2 ]
         5  [   0    0    0    1    0  885    1    0    1    1 ]
         6  [   1    0    0    0    0    1  950    0    0    0 ]
         7  [   0    1    1    1    0    0    0 1026    0    2 ]
         8  [   0    0    0    1    0    0    1    0  964    1 ]
         9  [   0    0    0    0    1    1    0    2    1  995 ]
```

### How to Read It

| Element | Meaning | Example |
|---------|---------|---------|
| **Rows** | Actual (true) label | Row 4 = all images that are truly the digit "4" |
| **Columns** | Predicted label | Column 9 = all images the model predicted as "9" |
| **Diagonal** | Correct predictions (top-left to bottom-right) | Cell (4, 4) = images of "4" correctly predicted as "4" |
| **Off-diagonal** | Mistakes | Cell (4, 9) = 2 images of "4" that model thought were "9" |

### What You Can Learn

| Question | How to Find It | Example |
|----------|---------------|---------|
| **Which digit is most confused?** | Find row with most off-diagonal entries | "4" and "9" are often confused |
| **What gets confused with what?** | Look at the largest off-diagonal values | 17 images of "4" predicted as "9" |
| **Is the model balanced?** | Compare diagonal values across rows | Some digits may have far fewer mistakes |
| **Overall accuracy** | `sum(diagonal) / sum(all entries)` | ~97.5% for a typical MNIST MLP |

### Why Accuracy Alone is Not Enough

Accuracy hides failure patterns. A confusion matrix reveals the **specific weaknesses**:

| Scenario | Overall Accuracy | Confusion Matrix Reveals |
|----------|-----------------|-------------------------|
| Medical classifier for rare disease | 99.8% (looks great) | Always predicts "no disease" — misses every single case ❌ |
| MNIST digit classifier | 97.5% (looks good) | Digit "4" has 17 errors, digit "0" only 2 errors — not all digits equal |
| Autonomous vehicle pedestrian detector | 99.9% (incredible) | Fails on night-time pedestrians in crosswalks — critical failure mode |

### Connection to Section 8 (Visualize Predictions)

The confusion matrix tells you **quantitative** facts (e.g. "17 images of '4' were mistaken for '9'"). Section 8 complements this by showing you the **actual images** the model got right (green) and wrong (red), so you can visually inspect:

- "Does this '4' genuinely look like a '9' to me too?" (hard example)
- "Is this image blurry or poorly written?" (data quality issue)
- "Is the model making a strange error that no human would make?" (model limitation)

This builds **intuition** for what your model is seeing vs. missing.

---

## 5. Key Takeaways

| Concept | Summary |
|---------|---------|
| **CrossEntropyLoss** | Combines LogSoftmax + NLLLoss; returns average loss per sample across batch |
| **`loss.item() * labels.size(0)`** | Converts per-sample average back to batch total (handles uneven last batch correctly) |
| **`argmax(1)`** | Finds highest-scoring class per image along dimension 1 (the class dimension) |
| **Model reset** | Must re-run Sections 2 + 3 (model creation + optimizer creation) to start fresh |
| **Optimizer state** | Adam maintains momentum buffers — must re-create optimizer for true reset |
| **Low GPU utilization** | Expected for small models (MLP on MNIST); not a problem, just a limitation of the task |
| **DataLoader tuning** | `BATCH_SIZE=1024`, `num_workers=8`, `pin_memory=True` helps but GPU still underutilized |
| **Confusion matrix** | 10×10 grid showing prediction breakdown per class; reveals which digits get confused |
| **Confusion matrix vs accuracy** | Accuracy = single number; confusion matrix = detailed per-class error analysis |

---

## 6. Common Code Patterns (Day 11 Style)

### Complete Train + Evaluate Functions

```python
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss.item() * labels.size(0)   # Weighted sum (handles uneven last batch)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total  # True average


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total
```

### Training Loop

```python
NUM_EPOCHS = 10

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2%} | "
          f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2%}")
```

### Creating a Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = model(images).argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))

fig, ax = plt.subplots(figsize=(8, 7))
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix — MLP on MNIST")
plt.tight_layout()
plt.show()
```

---

## 7. Questions for Self-Check

1. **Why does `total_loss` multiply by `labels.size(0)` instead of just adding `loss.item()`?**
2. **What does `outputs.argmax(1)` return, and why dimension 1?**
3. **What would `argmax(0)` give you instead, and why is it wrong for classification?**
4. **What sections must be re-run to reset the model for a fresh training?**
5. **Why does the optimizer also need to be re-created when resetting?**
6. **Why is GPU utilization low when training an MLP on MNIST?**
7. **What do `num_workers` and `pin_memory` do in DataLoader?**
8. **How do you read a confusion matrix — what do rows, columns, and the diagonal represent?**
9. **What information does a confusion matrix provide that overall accuracy doesn't?**
10. **Why are Sections 7 and 8 complementary — what does each tell you?**

---

## 8. Next Steps

| Day | Topic |
|-----|-------|
| **Day 12** | Overfitting & Regularization — dropout, weight decay, early stopping |
| **Day 13** | Batch Normalization & LR Scheduling |
| **Day 14** | Training Pipeline Best Practices (checkpointing, reproducibility) |
| **Day 15** | Working with Image Data (CIFAR-10, data augmentation) |
| **Day 16** | The Convolution Operation — where GPU really starts working! ⚡ |

---

*Last updated: 2026-06-05*