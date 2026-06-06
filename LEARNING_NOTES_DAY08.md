# Day 08 Learning Notes - nn.Module, Layers & Optimizers

## Question 1: Hooks, `model(x)` vs `model.forward(x)`, and `__call__`

**Q:** What are hooks? Why is `model(x)` better than `model.forward(x)`? How does this connect to `__callable__` from Day 01?

**A:**
- **Hooks** are functions that run automatically during forward/backward passes. They're useful for:
  - Debugging (inspecting intermediate activations)
  - Visualization
  - Modifying gradients
  - PyTorch supports forward hooks and backward hooks

- **Why `model(x)` is better than `model.forward(x)`:**
  - `model(x)` calls the `__call__()` method (Python dunder method)
  - `__call__()` does important setup work:
    1. Runs forward hooks (if any are registered)
    2. Calls `forward()` method
    3. Runs backward hooks (if any are registered)
  - `model.forward(x)` skips all hooks - so gradients might not be tracked properly
  - **Always use `model(x)`**, never `model.forward(x)` directly

- **Connection to Day 01 (`__call__`):**
  - `__call__` is a special dunder method in Python (like `__init__`, `__str__`)
  - When you use parentheses on an object (`model(x)`), Python calls `__call__()`
  - This makes the object "callable" (like a function)
  - In PyTorch, `nn.Module` defines `__call__()` to handle hooks and then delegate to `forward()`

---

## Question 2: Understanding `nn.Sequential`

**Q:** How does `nn.Sequential` work in Section 3? How does it connect to previous learning?

**A:**
- **`nn.Sequential`** is a container that chains layers in order
- You pass a sequence of modules, and it runs them sequentially (first layer → second layer → ...)
- **Use case:** Simple feed-forward networks without branching (no if statements, no multiple outputs)

**Example:**
```python
model = nn.Sequential(
    nn.Linear(10, 5),   # Layer 1: 10 inputs → 5 outputs
    nn.ReLU(),           # Activation
    nn.Linear(5, 2)     # Layer 2: 5 inputs → 2 outputs
)
```

**Connection to Day 07 (Backpropagation):**
- Backpropagation still works the same way!
- Autograd tracks the entire computation graph through the sequential layers
- Gradients flow backward through all layers automatically
- `nn.Sequential` is just a convenient way to define the forward pass without writing a custom `forward()` method

**When NOT to use `nn.Sequential`:**
- When you need branching (if statements)
- When you need multiple outputs
- When the forward pass has complex logic
- In these cases, subclass `nn.Module` and write your own `forward()` method

---

## Question 3: Parameter Counting Code (Section 9)

**Q:** How does the code block that computes total parameters work?

**A:**
The code uses a **generator expression** (Day 01 concept!) to sum up parameters:

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

**Breaking it down:**
1. **`model.parameters()`** - Returns an iterator over all `nn.Parameter` tensors in the model (and all submodules)
2. **`p.numel()`** - Returns the number of elements in a tensor ("number of elements")
   - Example: A Linear layer with weight shape `[5, 10]` has `5 * 10 = 50` parameters
3. **Generator expression** `(p.numel() for p in model.parameters())` - Creates an iterator of parameter counts
4. **`sum()`** - Sums up all the counts

**Connection to Days 04-05:**
- `nn.Parameter` is a `Tensor` with `requires_grad=True`
- `nn.Module` automatically tracks all `nn.Parameter` objects defined in `__init__()`
- This is why we use `nn.Parameter()` for learnable weights (Day 05)

**Example output:**
```
Layer: linear1.weight | Shape: torch.Size([5, 10]) | Parameters: 50
Layer: linear1.bias   | Shape: torch.Size([5])      | Parameters: 5
Total parameters: 55
```

---

## Question 4: `model.train()` vs `model.eval()` (Section 8)

**Q:** What are the use cases of `model.train()` and `model.eval()`? Why do they seem separate from previous code?

**A:**

### What they do:
- **`model.train()`** - Sets the model to training mode
- **`model.eval()`** - Sets the model to evaluation mode
- These set an **internal flag** (`self.training`) that some layers check

### Why they matter:

| Layer Type | `model.train()` (training mode) | `model.eval()` (eval mode) |
|------------|--------------------------------|----------------------------|
| **Dropout** | Randomly zeros neurons (prevents overfitting) | Uses ALL neurons (no dropping) |
| **BatchNorm** | Uses current batch statistics (mean, std) | Uses running statistics (learned during training) |
| **Other layers** | Usually no effect | Usually no effect |

### When to use them:

```python
# Training loop
model.train()  # Enable Dropout, BatchNorm uses batch stats
for batch in train_loader:
    # ... training code ...

# Evaluation/Testing
model.eval()  # Disable Dropout, BatchNorm uses running stats
with torch.no_grad():  # Also disable gradient tracking
    for batch in test_loader:
        # ... evaluation code ...
```

### Why separate from previous code?
- In **Day 06-07**, we didn't have Dropout or BatchNorm layers
- Those simple networks behaved the same in training and evaluation
- Once you add Dropout or BatchNorm, you MUST switch between train/eval modes
- **Common bug:** Forgetting `model.eval()` before testing → Dropout still active → bad predictions!

### Connection to Day 06 (Gradient Descent):
- During training: `model.train()` + `optimizer.step()` updates weights
- During testing: `model.eval()` + `torch.no_grad()` evaluates performance
- This separation is crucial for accurate evaluation

---

## Key Takeaways from Day 08

1. **`nn.Module`** is the base class for all neural networks - it organizes parameters, defines forward pass, supports nesting
2. **Always use `model(x)`** not `model.forward(x)` - hooks need to run!
3. **`nn.Sequential`** is great for simple feed-forward networks
4. **`model.train()` / `model.eval()`** are essential when using Dropout or BatchNorm
5. **Parameter counting** uses generator expressions and `p.numel()` - concepts from Day 01 and Day 04-05!
6. **Optimizers** (Section 10) replace manual gradient updates from Day 06:
   - Day 06: `weights -= learning_rate * weights.grad`
   - Day 08: `optimizer.step()` (handles all parameters automatically)

---

## Connection to Previous Days

| Day | Concept | How it connects to Day 08 |
|-----|---------|---------------------------|
| 01 | Python classes, `__init__`, dunder methods | `nn.Module` uses `__init__()` and `__call__()` |
| 04 | Tensors | `nn.Parameter` IS a Tensor with `requires_grad=True` |
| 05 | Autograd | `nn.Module` parameters are tracked by autograd |
| 06 | Gradient descent | Optimizers automate the update step |
| 07 | Backpropagation | Still happens automatically with `loss.backward()` |

---

*Notes compiled from Q&A session on Day 08*