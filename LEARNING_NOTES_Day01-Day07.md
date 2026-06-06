# Learning Notes from Chat History — Days 01–07

Collected Q&A from interactive discussion with GitHub Copilot. Reference for concepts that were clarified through questions.

---

## PyTorch Autograd & Computation Graphs

### Does autograd require the model to be differentiable?

Autograd only computes gradients for operations that **have a registered backward function** (a defined gradient rule).

- PyTorch "knows" by storing `grad_fn` in each tensor, which records which backward rule to use.
- Most ops are **piecewise differentiable** (e.g., ReLU). At non-differentiable points, PyTorch uses subgradients/conventions.
- If you use an unsupported op, `.backward()` will error.
- Model must be built from autograd-tracked tensors and supported ops.
- Parameters need `requires_grad=True` (default for `nn.Parameter`).
- Ops under `torch.no_grad()` or on `.detach()`'d tensors won't flow gradients.

### What does "track operations" mean?

When `requires_grad=True`, PyTorch **records every operation** to build a computation graph:
- Nodes = tensors (values)
- Edges = operations (multiply, add, matmul, etc.)
- Stores in `grad_fn` how to compute gradients backward

This graph is then used by `.backward()` to compute all derivatives via chain rule.

### Why no gradient tracking during inference?

Inference = predictions only, no weight updates → no gradients needed.

If tracking was enabled during inference:
- Builds a graph you never use
- Stores intermediate tensors for backward
- Wastes **memory** and **computation time**

Solution: wrap inference with `torch.no_grad()` or use `model.eval()`.

### `.item()` vs `.data`

| Method | Purpose | When to use |
|---|---|---|
| `.item()` | Extract Python scalar from single-value tensor | Printing scalars like `loss`, `w.grad` |
| `.data` | Get tensor view detached from autograd (legacy) | Avoid — use `.detach()` instead (safer) |

Example:
```python
loss = ((y_pred - y_true) ** 2).mean()
print(loss.item())  # Python float — safe for printing
```

**Modern best practice:** use `tensor.detach()` instead of `.data`.

---

## Gradient Computation

### What is `.item()`? (Day 05, Section 7)

The question was about `.item()` (singular), not `.items()`.

**`.item()` converts a 0-dim PyTorch tensor into a plain Python number.**

In your notebook:
```python
y.item()    # scalar tensor → Python float
x.item()    # scalar tensor → Python float
w.grad.item()  # scalar tensor → Python float
```

Key points:
- Works **only when tensor has exactly one element** — throws error otherwise
- Useful for printing/logging or using value in normal Python code
- For multi-element tensors, use `tensor.tolist()` instead

Example:
```python
t = torch.tensor(7.0)
print(type(t), t)          # <class 'torch.Tensor'> tensor(7.)
print(type(t.item()), t.item())  # <class 'float'> 7.0

v = torch.tensor([1.0, 2.0])
# v.item()  # ❌ error: more than one element
v.tolist()  # ✅ [1.0, 2.0]
```

---

### Concrete example: ∂loss/∂w and ∂loss/∂b (Day 05, Section 6)

Given:
- X = [1, 2, 3, 4]
- y_true = [2, 4, 6, 8]  (ground truth: y = 2x)
- w = 0, b = 0 (initial)
- Loss = MSE = mean((wX + b - y_true)²)

**Step 1: Predictions**
- y_pred = [0, 0, 0, 0]  (all zeros)

**Step 2: Errors**
- e = [0-2, 0-4, 0-6, 0-8] = [-2, -4, -6, -8]

**Step 3: Loss**
```
loss = (1/4) × (4 + 16 + 36 + 64)
     = (1/4) × 120
     = 30
```

**Step 4: Gradient ∂loss/∂w**

General formula:
```
∂loss/∂w = (2/N) × Σ (w·xᵢ + b - yᵢ) × xᵢ
```

At w=0, b=0:
```
∂loss/∂w = 0.5 × [(-2)·1 + (-4)·2 + (-6)·3 + (-8)·4]
         = 0.5 × [-2 - 8 - 18 - 32]
         = 0.5 × (-60)
         = -30
```

**Step 5: Gradient ∂loss/∂b**

General formula:
```
∂loss/∂b = (2/N) × Σ (w·xᵢ + b - yᵢ)
```

At w=0, b=0:
```
∂loss/∂b = 0.5 × [(-2) + (-4) + (-6) + (-8)]
         = 0.5 × (-20)
         = -10
```

**Interpretation:** Both gradients are negative → increasing w and b will decrease loss (predictions are too small).

### Do gradients depend on current weight values?

**Yes, on BOTH current weights AND features.**

Formula shows w appears in the prediction term:
```
∂loss/∂w = (2/N) Σ (w·xᵢ + b - yᵢ) × xᵢ
```

Example progression (same data):
```
At w=0, b=0:  ∂loss/∂w = -30
At w=1, b=0:  ∂loss/∂w = -15
At w=2, b=1:  ∂loss/∂w ≈ 0 (optimal)
```

The gradient **shrinks toward zero** as weights approach optimal values. This is why gradient descent naturally converges!

---

## Activation Functions

### What is an activation function?

Without activation functions, stacking layers is useless:

```
Layer 1: z1 = w1 * x + b1
Layer 2: z2 = w2 * z1 + b2
         = w2 * (w1 * x + b1) + b2
         = (w2*w1) * x + (w2*b1 + b2)
         = W * x + B
```

Still just a line! Multiple linear operations = one big linear operation.

**Activation function** = **non-linear function** applied after each layer to break linearity:

```
z1 = w1 * x + b1       ← linear
a1 = activation(z1)     ← non-linear (activation)
z2 = w2 * a1 + b2       ← linear applied to non-linear output
```

Now the network can learn curves.

### What is ReLU?

**ReLU** = Rectified Linear Unit

```
ReLU(x) = max(0, x)

If x > 0   →  output x     (pass through)
If x ≤ 0   →  output 0     (kill it)
```

**Examples:**
```
ReLU(3.5)   = 3.5
ReLU(0.1)   = 0.1
ReLU(0.0)   = 0.0
ReLU(-2.0)  = 0.0
ReLU(-100)  = 0.0
```

### Why does ReLU (a simple max function) make networks behave as non-linear curve models?

**ReLU alone is NOT curvy — it's just a bent line:**

```
        ╱
       ╱
──────╱        ← ReLU is one hinge (two linear pieces)
     0
```

**But combining MANY ReLU neurons creates curves:**

Each ReLU neuron computes:
```
neuron 1: ReLU(w1 * x + b1)   ← hinge at x = -b1/w1
neuron 2: ReLU(w2 * x + b2)   ← hinge at different x
neuron 3: ReLU(w3 * x + b3)   ← hinge at yet another x
```

Each neuron "turns on" at a different point. Then output layer **adds them together with different weights**:
```
output = c1 * neuron1 + c2 * neuron2 + c3 * neuron3
```

**Visual example — approximating a hill (∩) shape:**

```
Neuron 1: slopes up, turns on at x=1
             ╱
────────────╱

Neuron 2: slopes up, turns on at x=3
                       ╱
──────────────────────╱

Neuron 3: slopes up, turns on at x=2
                  ╱
─────────────────╱

Combine with weights:
         ╱‾‾‾‾‾╲
        ╱        ╲
───────╱          ╲──────    ← looks like a hill!
       1    2      3
```

**Key insight:** Each ReLU neuron adds a **new "bend point"**. With enough bend points, you can approximate **any curve** — like connecting dots with short straight segments.

**Real-world analogy — Origami paper folding:**
- One fold = one straight crease (one ReLU neuron)
- Two folds = simple tent shape
- 10 folds = complex shapes
- 1000 folds = can approximate any smooth surface

**Why a plain linear-only network CAN'T do this:**

Without ReLU:
```
Layer 1: z1 = w1*x + b1
Layer 2: z2 = w2*z1 + b2 = w2*w1*x + w2*b1+b2 = W*x + B
```

No matter how many layers, it collapses to **one line**. There are no "bend points."

With ReLU:
```
Layer 1: a1 = ReLU(w1*x + b1)    ← introduces a bend
Layer 2: z2 = w2*a1 + b2          ← weights the bent line
```

The ReLU **prevents the layers from collapsing** because `max(0, x)` cannot be "undone" by multiplication. The bend **survives** through to the output.

**ReLU gradient:**
```
ReLU'(x) = 1 if x > 0
           0 if x ≤ 0
```

**Dying ReLU problem:** If z ≤ 0 always, neuron outputs 0 and gradient is 0 → neuron stops learning. Consider LeakyReLU or ELU instead.

---

## Vectorized Operations & Matrix Dimensions

### What is a hidden layer? Why (2,3) instead of (2,1)?

**Hidden layer** = a layer between input and output. It's called "hidden" because you never directly see its values — they're internal to the network.

The network architecture in Day 07 Section 4:
```
Input (2 features)  →  Hidden Layer (3 neurons)  →  Output (1 value)
     [x1, x2]              [h1, h2, h3]                [out]
```

**Why 3 hidden neurons instead of 1?**

Each hidden neuron learns a **different pattern** from the input. Think of 3 "experts" looking at the same input, each focusing on something different:

```
Neuron h1 = w11*x1 + w12*x2 + b1   ← looks for one pattern
Neuron h2 = w21*x1 + w22*x2 + b2   ← looks for another pattern
Neuron h3 = w31*x1 + w32*x2 + b3   ← looks for yet another
```

That requires **6 weights** (2 inputs × 3 neurons), organized as a (2, 3) matrix:

```
W1 = [[w11, w21, w31],     ← weights FROM input 1 TO each hidden neuron
      [w12, w22, w32]]     ← weights FROM input 2 TO each hidden neuron

Shape: (2 inputs, 3 hidden neurons) = (2, 3)
```

If W1 were (2, 1), you'd have only **1 hidden neuron** — too simple to learn complex patterns.

**More hidden neurons = more capacity to learn** (recall: more ReLU "bends" = better curve approximation).

### Why is b1 shape (3,) and not a single number?

**Each neuron gets its own bias.** Since there are 3 hidden neurons:

```
h1 = w11*x1 + w12*x2 + b1[0]    ← neuron 1's bias
h2 = w21*x1 + w22*x2 + b1[1]    ← neuron 2's bias
h3 = w31*x1 + w32*x2 + b1[2]    ← neuron 3's bias
```

So `b1` has shape `(3,)` — one bias per hidden neuron.

Similarly, `b2` has shape `(1,)` — one bias for the single output neuron.

### In forward pass (4, 2) @ (2, 3) + (3,), does the 2nd term match the first term size?

**Short answer:** No, they don't match initially, but **broadcasting** makes them compatible.

**Step 1: Matrix multiply `(4, 2) @ (2, 3) → (4, 3)`**

```
X @ W1

4 samples, each with 2 features    2 inputs → 3 hidden neurons
┌─────────┐                        ┌──────────────┐
│ 1.0  2.0│                        │ w11  w21  w31│
│ 3.0  4.0│          @             │ w12  w22  w32│
│ 5.0  6.0│                        └──────────────┘
│ 7.0  8.0│                            (2, 3)
└─────────┘
   (4, 2)

Result: (4, 3)  ← 4 samples, each gets 3 hidden values
┌──────────────────────┐
│ h1_s1   h2_s1   h3_s1│   ← sample 1's 3 hidden values
│ h1_s2   h2_s2   h3_s2│   ← sample 2's 3 hidden values
│ h1_s3   h2_s3   h3_s3│   ← sample 3
│ h1_s4   h2_s4   h3_s4│   ← sample 4
└──────────────────────┘
```

The inner dimensions must match: `(4, **2**) @ (**2**, 3)` — the "2" cancels out.

**Step 2: Add bias `(4, 3) + (3,)` — broadcasting!**

The result of `X @ W1` is shape `(4, 3)`. The bias `b1` is shape `(3,)`.

These **don't have the same shape**, but PyTorch uses **broadcasting**:

```
(4, 3) + (3,)

PyTorch treats (3,) as (1, 3), then repeats it 4 times:

┌──────────┐     ┌──────────┐     ┌──────────┐
│ z  z  z  │     │ b1 b2 b3 │     │ z+b1 ... │
│ z  z  z  │  +  │ b1 b2 b3 │  =  │ z+b1 ... │
│ z  z  z  │     │ b1 b2 b3 │     │ z+b1 ... │
│ z  z  z  │     │ b1 b2 b3 │     │ z+b1 ... │
└──────────┘     └──────────┘     └──────────┘
   (4, 3)      (3,) broadcast       (4, 3)
                  to (4, 3)
```

The **same bias** is added to every sample. This makes sense — the bias is a property of the **neuron**, not of the sample.

**Step 3: Same pattern for Layer 2**

```
a1 @ W2 + b2
(4, 3) @ (3, 1) + (1,) → (4, 1)

4 samples, 3 hidden values each → 4 samples, 1 output each
```

### What is the point / purpose of Day 07 Section 4 demo?

**Sections 2-3 used scalar values:** one input, one weight per layer. Good for understanding chain rule, but **real neural networks process multiple samples with multiple features simultaneously using matrices.**

**Section 4 demonstrates that backprop works exactly the same way** — just with matrix operations instead of scalar multiplication.

**Key purposes:**
1. Show how matrix multiply dimensions work in forward pass
2. Demonstrate broadcasting (adding bias across batch)
3. Show that `.backward()` on batched data computes gradients for **all** weights at once
4. Prove autograd scales from scalars to matrices without changing the underlying chain rule logic

```python
# Forward pass (all 4 samples at once)
z1 = X @ W1 + b1          # (4, 2) @ (2, 3) + (3,) → (4, 3)
a1 = torch.relu(z1)        # (4, 3)
z2 = a1 @ W2 + b2          # (4, 3) @ (3, 1) + (1,) → (4, 1)
loss = ((z2 - y_true) ** 2).mean()

# Backward — one call computes ALL gradients (W1, b1, W2, b2)
loss.backward()

print(f"W1.grad shape: {W1.grad.shape}")  # (2, 3)
print(f"W2.grad shape: {W2.grad.shape}")  # (3, 1)
```

The chain rule is the same as Section 2 — just applied to matrices instead of scalars. **Same math, same logic, different data structure.**

---

## Backpropagation & Chain Rule

### Chain rule overview

If y depends on u, and u depends on x:

```
dy/dx = (dy/du) × (du/dx)
```

**In neural networks:** Each layer is a link. Backprop applies chain rule from loss backward through all layers.

### Manual backward pass example (2-layer net)

**Forward:**
```
z1 = w1 * x + b1
a1 = ReLU(z1)
z2 = w2 * a1 + b2
loss = (z2 - y_true)²
```

**Backward (chain rule):**
```
dL/dz2 = 2 × (z2 - y_true)
dL/dw2 = dL/dz2 × a1
dL/db2 = dL/dz2
dL/da1 = dL/dz2 × w2
dL/dz1 = dL/da1 × ReLU'(z1)      ← ReLU'(z1) = 1 if z1>0, else 0
dL/dw1 = dL/dz1 × x
dL/db1 = dL/dz1
```

**Key insight:** Each gradient is a product of local gradients multiplied from loss backward to that parameter. This is the chain rule applied systematically.

---

## Training Loop Structure

```python
for epoch in range(num_epochs):
    # 1. Forward pass
    y_pred = w * X + b
    
    # 2. Compute loss
    loss = ((y_pred - y_true) ** 2).mean()
    
    # 3. Backward pass (compute gradients)
    loss.backward()
    
    # 4. Update weights (don't track this update)
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    # 5. Zero gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()
```

**Critical:** Always zero gradients before backward, else they accumulate.

### Learning rate effects

```
lr = 0.0001  →  Too small, slow convergence
lr = 0.01    →  Good, smooth convergence
lr = 0.1     →  Too large, overshoots/diverges
```

---

## Gradient Descent Intuition

- **Gradient** = which direction to move weight to reduce loss
- **Negative gradient** = increase weight decreases loss
- **Positive gradient** = decrease weight decreases loss
- **Zero gradient** = at a minimum (or saddle point)

Update rule:
```
w_new = w_old - lr × (∂loss/∂w)
```

The gradient **depends on both current w and features X**. As w approaches optimal, gradient shrinks naturally — training slows down near the solution (good!).

---

## Common Pitfalls & Best Practices

| Pitfall | Fix |
|---|---|
| Forgetting to zero gradients | Call `w.grad.zero_()` before `.backward()` |
| Using `.data` unsafely | Use `.detach()` instead |
| Building gradient graph during inference | Wrap with `torch.no_grad()` |
| Learning rate too large | Try smaller values, plot loss curve |
| ReLU neurons dying | Consider LeakyReLU or ELU |
| Confusing tensor shape vs scalar | Use `.item()` only for scalars; print shape with `.shape` |

---

## Quick Reference

```python
# Tensor creation with gradients
param = torch.tensor(1.0, requires_grad=True)

# Forward pass
output = model(input)
loss = criterion(output, target)

# Backward pass
loss.backward()

# Access gradients
grad_value = param.grad.item()

# Update (don't track)
with torch.no_grad():
    param -= lr * param.grad

# Zero for next iteration
param.grad.zero_()

# For inference (no grad tracking)
with torch.no_grad():
    predictions = model(test_input)

# Or detach to break graph
detached = tensor.detach()
```

---

## Matrix shapes quick lookup

**For FC (fully connected) layer:**
- W shape: `(in_features, out_features)`
- b shape: `(out_features,)`

**For batched forward pass:**
```
X @ W + b

X: (batch_size, in_features)
W: (in_features, out_features)
b: (out_features,)

Result: (batch_size, out_features)
```

**Broadcasting rule:** When adding `(batch_size, out_features) + (out_features,)`:
- b broadcasts to (batch_size, out_features)
- Same bias applied to every sample

---

## Tomorrow's topics (preview)

- `nn.Module` — PyTorch's framework for organizing models
- How to structure a reusable model class
- Built-in layers and how they wrap the math we've learned

---


---

## Feedforward Networks vs Other Architectures

### What does "feedforward" mean?

**Feedforward** = data flows in **one direction only: input → hidden layers → output**.

```
Input → Layer1 → Layer2 → Layer3 → Output
  ↓       ↓        ↓        ↓        ↓
  data flows forward, never backward to previous layers
```

**Key characteristics:**
- No loops or cycles
- No memory of past inputs
- Each sample processed independently
- "Feeds forward" the data through each layer sequentially

### In contrast: Other types of networks

#### Recurrent Neural Networks (RNN)

Data can loop back to previous layers (has memory/state):

```
Input → Hidden → Output
         ↑  ↓
         └──┘  ← loops back!

RNN processes sequences by "remembering" previous steps
```

**Use case:** Time series, language, speech — anything where order/history matters.

```python
# Pseudocode
for time_step in sequence:
    hidden_state = RNN(input[t], hidden_state)  # ← uses previous hidden state
    output = output_layer(hidden_state)
```

**Example:**
- Input: "Hello world" (word by word)
- RNN remembers "Hello" when processing "world"
- Understands context because of recurrence

#### Convolutional Neural Networks (CNN)

Feedforward but with **convolution** instead of full connections:

```
Input Image
    ↓
Conv Layer (learns patterns like edges)
    ↓
ReLU
    ↓
Pool Layer (summarize)
    ↓
... more conv layers ...
    ↓
Flatten
    ↓
FC Layers (like your Section 4 network)
    ↓
Output
```

**Still feedforward** (one direction), but uses **local connections** instead of fully connected.

**Use case:** Images, spatial data — learns local features (edges, textures, shapes).

#### Transformer / Attention Networks

Feedforward-ish, but with **self-attention** (can look at all positions at once):

```
Input → Embed → Self-Attention → Feed-Forward → Output
                     ↑________________↑
                 (can attend to any position)
```

**Use case:** Language models (GPT, BERT), Vision Transformers (ViT).

**Key difference from RNN:**
- RNN: processes sequentially (one word at a time)
- Transformer: processes all positions in parallel (all words at once), attends to relevant positions

### Summary table

| Type | Data Flow | Memory | Use Case |
|---|---|---|---|
| **Feedforward (MLP)** | One direction only | No | Tabular data, simple classification |
| **RNN** | Loops back | Yes (hidden state) | Sequences (time series, text, speech) |
| **CNN** | One direction (conv ops) | No | Images, spatial data |
| **Transformer** | One direction (parallel) | No (but self-attention) | Language, vision, long sequences |
| **Hybrid** | Mix of above | Depends | Complex tasks (e.g., image + text) |

### Historical context: Why "feedforward"?

Back in the 1980s-90s, when neural networks were being researched:

- **Feedforward** = explicit designation that there are NO feedback loops (data doesn't cycle back)
- Contrasted with **feedback networks** (like Hopfield networks) that had recurrent connections
- The term stuck!

**Modern usage:**
- "Feedforward" = any network with NO recurrence
- Includes MLPs, CNNs, Transformers (technically)
- **RNN is explicitly NOT feedforward** because of the loops

---

## Fully Connected vs Local Connections

### What does "fully connected" mean?

**Not about samples** — it's about **how neurons connect to each other within a layer**.

**Fully Connected (FC):** Every neuron in layer N connects to EVERY neuron in layer N+1.

```
Layer 1 (Input)          Layer 2 (Output)
   x1 ──┐               ┌─→ y1
        ├─ W[1,1] ─────┤
   x2 ──┤               ├─ y2
        ├─ W[1,2] ─────┤
   x3 ──┘               └─ y3

All connections exist!
Every input connects to every output.
```

**Mathematically:**
```python
# Fully Connected
y = X @ W + b

# X shape: (batch_size, in_features)
# W shape: (in_features, out_features)
# Result: (batch_size, out_features)

# EVERY input feature influences EVERY output
```

**In Day 07 Section 4:**
```python
z1 = X @ W1 + b1   # (4,2) @ (2,3) → (4,3)

# W1 has 2×3 = 6 weights
# Input 1 connects to hidden neurons 1, 2, 3 (3 connections)
# Input 2 connects to hidden neurons 1, 2, 3 (3 connections)
# Total: 6 connections (fully connected!)

W1 = [[w11, w21, w31],   ← from input 1 to all 3 hidden
      [w12, w22, w32]]   ← from input 2 to all 3 hidden
```

### Convolutional Layer (Local Connections)

**Each neuron in layer N connects to ONLY a SMALL REGION of layer N.**

```
Image (5×5 pixels)       Convolution (3×3 filter)
┌─────────────────┐
│ • • ● ● ●       │      ┌─────┐
│ • • ● ● ●       │  ──→ │ y1  │  (one 3×3 window)
│ ● ● ● ● ●       │      └─────┘
│ ● ● ● ● ●       │
│ ● ● ● ● ●       │      The filter only looks at
└─────────────────┘      the 3×3 neighborhood (9 connections)
                         NOT all 25 pixels!
```

**Comparison:**

```
Fully Connected (FC):
- Input: 5×5=25 pixels
- Output neuron: sees all 25 pixels
- Connections per output: 25

Convolutional (Conv):
- Input: 5×5=25 pixels
- Output neuron: sees only 3×3=9 pixels (local window)
- Connections per output: 9
```

### Why local connections for images?

**Problem with Fully Connected on images:**

```
Image (28×28 = 784 pixels)
  ↓
FC layer with 128 hidden neurons
  ↓
Connections needed: 784 × 128 = 100,352 weights!

Too many parameters → overfitting, slow training
```

**Solution with Convolutional:**

```
Image (28×28)
  ↓
Conv with 3×3 filter
  ↓
Connections per output: only 3×3 = 9
All output neurons share the same 9 weights (parameter sharing!)
  ↓
Total weights: 9 (not 100k!)

Much simpler, faster, better for images
```

### Visual example: Convolution vs FC

**Fully Connected on image:**
```
Pixel[0,0] ─┐
Pixel[0,1] ─┼─ Neuron1
Pixel[0,2] ─┤
...all 784─┤
Pixel[27,27]┘

Every pixel directly influences neuron1.
Learns global patterns only.
```

**Convolutional on image:**
```
Pixel[0,0] ─┐
Pixel[0,1] ─├─ Neuron1  (top-left corner)
Pixel[0,2] ─┘

Pixel[0,1] ─┐
Pixel[0,2] ─├─ Neuron2  (one step right)
Pixel[0,3] ─┘

Pixel[1,0] ─┐
Pixel[1,1] ─├─ Neuron3  (one step down)
Pixel[1,2] ─┘
...

Each neuron looks at its 3×3 neighborhood.
All neurons share the same 9 weights (the "filter").
Learns local patterns (edges, textures) that work everywhere!
```

### Key differences

| Aspect | Fully Connected | Convolutional |
|---|---|---|
| **Connections** | Every input → every output | Only local neighborhood |
| **Parameters** | Many (in_features × out_features) | Few (filter_size × filter_size) |
| **Parameter sharing** | No | Yes (same filter everywhere) |
| **Best for** | Tabular data, fully-connected problems | Images, spatial patterns |
| **Example** | Your Day 07 Section 4 network | CNNs (AlexNet, ResNet, etc.) |

---

## Layer Terminology in Day 07 Section 4

### W1: Input → Hidden (expansion)

**W1 is called a "fully connected (FC) layer"** or **"dense layer"**.

- Input: 2 features
- Output: 3 hidden neurons
- Shape: (2, 3) — **expands** the representation (2 → 3)

This is **typical and common**. Why?

```
More neurons = more capacity to learn complex patterns
(Recall: more ReLU bends = better curve approximation)
```

### W2: Hidden → Output (compression)

**W2 is also a "fully connected (FC) layer"** or **"dense layer"**.

- Input: 3 hidden neurons
- Output: 1 value (prediction)
- Shape: (3, 1) — **compresses** the representation (3 → 1)

This is also **typical and common**. Why?

```
Squeeze the 3 learned patterns down to 1 final prediction
(or multiple outputs if multi-class/multi-task)
```

### Why W2 "covers" 1 output per sample

You asked: "W2 seems back coverage for 1 output for each sample."

**Correct intuition!** Here's how it actually works:

```python
a1 shape: (4, 3)   ← 4 samples, 3 hidden values each
W2 shape: (3, 1)   ← 3 hidden → 1 output

z2 = a1 @ W2       ← (4, 3) @ (3, 1) = (4, 1)

Result: (4, 1) ← 4 samples, 1 output per sample
```

**The key insight:** W2 is **shared across all samples**.

Each of the 4 samples goes through the **same W2 weights**:

```
Sample 1: [h1, h2, h3] @ W2 → output_1
Sample 2: [h1, h2, h3] @ W2 → output_2
Sample 3: [h1, h2, h3] @ W2 → output_3
Sample 4: [h1, h2, h3] @ W2 → output_4
```

Matrix multiply does all 4 at once: `(4,3) @ (3,1) → (4,1)`.

### Formal naming

| Term | Meaning |
|---|---|
| **Fully Connected (FC) Layer** | Every input connects to every output. All neurons see all features. |
| **Dense Layer** | Same as FC (PyTorch calls it `nn.Linear`; Keras calls it `nn.Dense`) |
| **Hidden Layer** | Any layer between input and output. "Hidden" = internal, not directly visible. |
| **Layer width** | Number of neurons in that layer (e.g., "layer width = 3") |

In PyTorch:
```python
import torch.nn as nn

# These are all equivalent terms:
fc = nn.Linear(in_features=2, out_features=3)  # Fully Connected
dense = nn.Linear(in_features=2, out_features=3)  # Dense
hidden = nn.Linear(in_features=2, out_features=3)  # Hidden layer
```

### Typical patterns in neural networks

**Pattern 1: Expand → Compress (Day 07 Section 4)**

```
Input (2)  →  Hidden1 (3)  →  Hidden2 (2)  →  Output (1)
              expand          compress

W1: (2,3)
W2: (3,2)
W3: (2,1)
```

This is **very common** — learn a rich representation, then compress to final answer.

**Pattern 2: Expand progressively**

```
Input (2)  →  Hidden1 (8)  →  Hidden2 (16)  →  Output (1)
              expand          expand

W1: (2,8)
W2: (8,16)
W3: (16,1)
```

More layers, more capacity.

**Pattern 3: Keep size constant**

```
Input (10)  →  Hidden1 (10)  →  Hidden2 (10)  →  Output (1)
              same            same

W1: (10,10)
W2: (10,10)
W3: (10,1)
```

---

## Quick Intuition Comparisons

```
Feedforward:     One pass, no memory
                 Input → Forward → Output
                 (stateless)

Recurrent:       Loops, has memory
                 Input → Forward with State → Output
                 (stateful, remembers previous steps)

Fully Connected: All inputs talk to all outputs
                 (dense, many parameters)

Convolutional:   Only local neighborhoods talk
                 (sparse, fewer parameters, parameter sharing)
```

**Analogy for feedforward:**
- Reading a single sentence once (no re-reading)

**Analogy for RNN:**
- Reading a paragraph word-by-word, remembering context as you go

**Analogy for FC:**
- Everyone in a room talks to everyone else (all-to-all communication)

**Analogy for Conv:**
- Everyone only talks to their neighbors (local communication)

---

EOF
