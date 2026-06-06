# Day 09 Learning Notes - Loss and Activation Functions

## Question 1: Is softmax simply normalization to max value of the input?

**Q:** Is softmax just normalizing to the max value? How is it different?

**A:**
**No!** Softmax is NOT just normalization to the max value. Here's the key difference:

### Max Normalization vs Softmax

**Max Normalization:**
```python
# Just divide by max value
normalized = x / max(x)
# Problem: outputs don't sum to 1, NOT a probability distribution
```

**Softmax:**
```python
# Exponentiate first, then normalize
softmax = exp(x) / sum(exp(x))
# Outputs ARE probabilities (sum to 1)
```

### Why exponentiate?

1. **Makes all values positive** - `exp(x) > 0` for any x
2. **Amplifies differences** - big values become MUCH bigger, small values become VERY small
3. **Creates "winner-take-all" effect** - the largest value gets most of the probability mass

### Example:
```python
import torch
import torch.nn.functional as F

x = torch.tensor([1.0, 2.0, 3.0])

# Max normalization
max_norm = x / x.max()
print(f"Max norm: {max_norm}")  
# Output: [0.333, 0.667, 1.000] → sum = 2.0 (NOT 1!)

# Softmax
softmax = F.softmax(x, dim=0)
print(f"Softmax: {softmax}")     
# Output: [0.090, 0.245, 0.665] → sum = 1.0 (probabilities!)
```

### Key property:
Softmax is **differentiable**, so we can use it with backpropagation! Max normalization is not differentiable at the max point.

---

## Question 2: Why do we call it "logits"?

**Q:** What does "logits" mean? Why use this name?

**A:**
**"Logits" = "Log-odds"** (logarithm of the odds ratio)

### Origin: Logistic Regression (Classical ML)

In logistic regression:
- The model directly predicts **log-odds**
- Log-odds = `log(p / (1-p))` where `p` is probability
- Then apply **sigmoid** to convert log-odds → probability

### In Deep Learning:

The **last linear layer** outputs raw scores (not probabilities)
- These raw scores are called **"logits"**
- Then apply **softmax** to convert logits → probabilities

```python
# Typical classification model
logits = model(x)           # Raw scores, e.g., [2.1, -1.3, 0.8]
probabilities = F.softmax(logits, dim=-1)  # [0.7, 0.05, 0.25]
```

### Why keep the name "logits"?

1. **Historical reason** - from logistic regression
2. **Technical meaning**: logits are **unnormalized log probabilities**
   - If you take `log(softmax(x))`, you get the log probabilities
   - "Logits" are the inputs to softmax (before log)
3. **Mathematical property**: In the limit, softmax of logits approaches one-hot encoding

### Analogy:
- **Logits** = raw scores (like exam scores before grading curve)
- **Softmax** = convert to probabilities (like curving to percentages)

### Connection to Loss Functions:

In practice, we often use **`F.cross_entropy()`** which combines softmax + negative log likelihood:
```python
# These are equivalent:
loss1 = F.cross_entropy(logits, targets)  # Does softmax internally

probs = F.softmax(logits, dim=-1)
loss2 = F.nll_loss(torch.log(probs), targets)  # Manual softmax + NLL
```

**Important:** `F.cross_entropy()` expects **logits** (raw scores), NOT probabilities!

---

## Question 3: Understanding `argmax` and Accuracy Computation

**Q:** In the "Building and Training Classifier" section, what does `argmax` do, and how is `acc` computed?

Code block:
```python
losses.append(loss.item())
if epoch % 40 == 0:
    preds = logits.argmax(dim=1)
    acc = (preds == y).float().mean()
    print(f"Epoch {epoch:3d}  Loss: {loss.item():.4f}  Acc: {acc.item():.2%}")
```

**A:**

### What is `logits`?

- `logits` is the output of the model's last linear layer
- Shape: `(batch_size, num_classes)`
- Example for 3-class classification with batch_size=2:
  ```python
  logits = torch.tensor([[0.1, 0.3, 0.6],   # Sample 1: class 0=0.1, class 1=0.3, class 2=0.6
                         [0.8, 0.1, 0.1]])  # Sample 2: class 0=0.8, class 1=0.1, class 2=0.1
  ```

### What does `argmax(dim=1)` do?

- **`argmax`** returns the **index** of the maximum value along a dimension
- **`dim=1`** means: look across the class dimension (columns) for each sample (row)
- Result: predicted class for each sample

```python
preds = logits.argmax(dim=1)
# preds = tensor([2, 0])
# - Sample 1: max is 0.6 at index 2 → predicted class 2
# - Sample 2: max is 0.8 at index 0 → predicted class 0
```

### How is accuracy computed?

Step by step:

**Step 1: Compare predictions with ground truth**
```python
y = torch.tensor([2, 1])  # Ground truth labels
preds = torch.tensor([2, 0])  # Predictions

correct = (preds == y)
# correct = tensor([True, False])
# - Sample 1: preds=2, y=2 → True (correct)
# - Sample 2: preds=0, y=1 → False (wrong)
```

**Step 2: Convert to float**
```python
correct_float = correct.float()
# correct_float = tensor([1.0, 0.0])
# True → 1.0, False → 0.0
```

**Step 3: Compute mean**
```python
acc = correct_float.mean()
# acc = (1.0 + 0.0) / 2 = 0.5 = 50%
```

### Complete Example:

```python
import torch
import torch.nn.functional as F

# Simulated outputs (logits)
logits = torch.tensor([[0.1, 0.3, 0.6],
                       [0.8, 0.1, 0.1],
                       [0.2, 0.5, 0.3],
                       [0.4, 0.4, 0.2]])

# Ground truth labels
y = torch.tensor([2, 1, 1, 0])

# Get predictions
preds = logits.argmax(dim=1)
print(f"Predictions: {preds}")  # tensor([2, 0, 1, 0])

# Compute accuracy
correct = (preds == y)
print(f"Correct: {correct}")  # tensor([True, False, True, False])

acc = correct.float().mean()
print(f"Accuracy: {acc.item():.2%}")  # 50.00%
```

### Connection to Previous Days:

| Day | Concept | Connection |
|-----|---------|------------|
| 01 | Python basics | `.float()`, `.mean()` are tensor methods |
| 08 | nn.Module | `logits = model(x)` outputs raw scores |
| 09 | Softmax | `argmax` after softmax = same as `argmax` before softmax |

**Note:** We can apply `argmax` directly to logits (without softmax) because softmax preserves the order of values! If `x > y`, then `softmax(x) > softmax(y)`.

---

## Summary Table

| Concept | What it does | Output | Example |
|---------|--------------|--------|---------|
| **Max normalization** | `x / max(x)` | Values in [0,1], don't sum to 1 | `[0.33, 0.67, 1.0]` |
| **Softmax** | `exp(x) / sum(exp(x))` | Probabilities (sum to 1) | `[0.09, 0.25, 0.66]` |
| **Logits** | Raw scores from last layer | Any real number (unnormalized) | `[2.1, -1.3, 0.8]` |
| **argmax(dim=1)** | Index of max value per sample | Predicted class indices | `[2, 0, 1]` |

---

## Key Takeaways from Day 09

1. **Softmax is NOT max normalization** - it exponentiates first to amplify differences
2. **Softmax outputs probabilities** that sum to 1
3. **"Logits" = raw scores** before softmax (historical name from logistic regression)
4. **Use `F.cross_entropy()` with logits** - it handles softmax internally
5. **Softmax is differentiable** - essential for backpropagation
6. **`argmax(dim=1)`** converts logits to predicted class indices
7. **Accuracy computation**: `(preds == y).float().mean()`

---

## Connection to Previous Days

| Day | Concept | How it connects to Day 09 |
|-----|---------|---------------------------|
| 01 | Python classes, methods | `.float()`, `.mean()`, `.item()` are tensor methods |
| 06 | Gradient Descent | Loss functions guide the optimization |
| 07 | Backpropagation | Softmax is differentiable, so gradients flow through it |
| 08 | nn.Module | Loss functions are also nn.Modules (e.g., `nn.CrossEntropyLoss`) |

---

*Notes compiled from Q&A session on Day 09*