"""
=============================================================================
 01 — Gradient Descent & Backpropagation with PyTorch
=============================================================================

GOAL: Understand how neural networks *learn* by adjusting weights.

KEY CONCEPTS:
  • Forward pass  — compute prediction from inputs
  • Loss function  — measure how wrong the prediction is
  • Backward pass (backpropagation) — compute gradients ∂Loss/∂weight
  • Gradient descent — update weights in the direction that reduces loss

KEY PYTORCH FUNCTIONS:
  • torch.tensor(..., requires_grad=True)  — tells PyTorch to track operations
  • loss.backward()                        — computes all gradients (backprop)
  • optimizer.zero_grad()                  — clears old gradients before next step
  • optimizer.step()                       — updates weights using gradients
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)

# =============================================================================
# PART 1: Manual Gradient Descent (no nn module — pure math)
# =============================================================================
# We'll fit a simple linear function:  y = 2x + 1
# The model will try to learn w=2 and b=1 from data.

print("=" * 70)
print("PART 1: Manual Gradient Descent — learn y = 2x + 1")
print("=" * 70)

# ── 1a. Create synthetic data ──────────────────────────────────────────────
X = torch.linspace(-3, 3, 50)             # 50 evenly-spaced points
y_true = 2 * X + 1 + 0.3 * torch.randn(50)  # y = 2x + 1 + noise

# ── 1b. Initialize learnable parameters ───────────────────────────────────
# requires_grad=True  →  PyTorch will record every operation on these tensors
#                         so it can later compute d(loss)/d(w) and d(loss)/d(b)
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01
losses_manual = []

print(f"\nBefore training:  w = {w.item():.4f},  b = {b.item():.4f}")

for epoch in range(100):
    # ── FORWARD PASS ──────────────────────────────────────────────────────
    # Compute prediction using current w and b
    y_pred = w * X + b

    # ── COMPUTE LOSS ──────────────────────────────────────────────────────
    # Mean Squared Error:  L = (1/N) Σ (y_pred - y_true)²
    loss = ((y_pred - y_true) ** 2).mean()
    losses_manual.append(loss.item())

    # ── BACKWARD PASS (backpropagation) ───────────────────────────────────
    # This single call walks backward through the computation graph and
    # fills in  w.grad  and  b.grad  with  ∂loss/∂w  and  ∂loss/∂b
    loss.backward()

    # ── GRADIENT DESCENT UPDATE ───────────────────────────────────────────
    # torch.no_grad() — we don't want PyTorch to track the update itself
    with torch.no_grad():
        w -= learning_rate * w.grad      # w = w - lr * ∂loss/∂w
        b -= learning_rate * b.grad      # b = b - lr * ∂loss/∂b

    # ── ZERO GRADIENTS ────────────────────────────────────────────────────
    # Gradients accumulate by default!  We must clear them before next step.
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}  |  Loss: {loss.item():.4f}  "
              f"|  w: {w.item():.4f}  b: {b.item():.4f}")

print(f"\nAfter training:   w = {w.item():.4f},  b = {b.item():.4f}")
print(f"Ground truth:     w = 2.0000,  b = 1.0000")


# =============================================================================
# PART 2: Using nn.Module & Optimizer (the PyTorch way)
# =============================================================================
# Same problem, but now using PyTorch's building blocks:
#   nn.Linear   — a layer that does  y = Wx + b  internally
#   nn.MSELoss  — mean squared error loss
#   optim.SGD   — stochastic gradient descent optimizer

print("\n" + "=" * 70)
print("PART 2: Using nn.Module & Optimizer")
print("=" * 70)


class LinearModel(nn.Module):
    """
    A tiny 1-input → 1-output linear model.
    nn.Module gives us:
      • automatic parameter registration (model.parameters())
      • .forward() defines the computation
      • interplay with optimizers, loss functions, etc.
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)
        # nn.Linear already creates weight and bias with requires_grad=True

    def forward(self, x):
        return self.linear(x)


# ── 2a. Reshape data for nn.Linear (expects [N, 1] not [N]) ──────────────
X_2d = X.unsqueeze(1)          # shape: [50] → [50, 1]
y_2d = y_true.unsqueeze(1)     # shape: [50] → [50, 1]

# ── 2b. Create model, loss function, optimizer ───────────────────────────
model = LinearModel()
criterion = nn.MSELoss()                              # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # optimizer

print(f"\nModel parameters before training:")
for name, param in model.named_parameters():
    print(f"  {name:20s} = {param.data.item():.4f}")

losses_nn = []

for epoch in range(100):
    # ── FORWARD PASS ──────────────────────────────────────────────────────
    y_pred = model(X_2d)

    # ── COMPUTE LOSS ──────────────────────────────────────────────────────
    loss = criterion(y_pred, y_2d)
    losses_nn.append(loss.item())

    # ── BACKWARD PASS ─────────────────────────────────────────────────────
    # Step 1: clear old gradients   (otherwise they accumulate!)
    optimizer.zero_grad()

    # Step 2: compute fresh gradients
    loss.backward()

    # ── UPDATE WEIGHTS ────────────────────────────────────────────────────
    # Step 3: optimizer adjusts each parameter by  p = p - lr * p.grad
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}  |  Loss: {loss.item():.4f}")

print(f"\nModel parameters after training:")
for name, param in model.named_parameters():
    print(f"  {name:20s} = {param.data.item():.4f}")
print(f"Ground truth:        weight = 2.0000,  bias = 1.0000")


# =============================================================================
# PART 3: Visualizing the Computation Graph (autograd under the hood)
# =============================================================================
# Let's peek at how PyTorch's autograd tracks operations.

print("\n" + "=" * 70)
print("PART 3: Exploring autograd — the engine behind backpropagation")
print("=" * 70)

a = torch.tensor(3.0, requires_grad=True)
b_val = torch.tensor(2.0, requires_grad=True)
c = a * b_val          # c = a * b
d = c + a              # d = c + a = a*b + a
e = d ** 2             # e = (a*b + a)²

print(f"\n  a = {a.item()},  b = {b_val.item()}")
print(f"  c = a * b     = {c.item()}")
print(f"  d = c + a     = {d.item()}")
print(f"  e = d²        = {e.item()}")

# Backprop from e
e.backward()

# The chain rule gives us:
#   de/da = 2d * (b + 1) = 2*(3*2+3)*(2+1) = 2*9*3 = 54
#   de/db = 2d * a       = 2*9*3             = 54
print(f"\n  de/da (autograd) = {a.grad.item()}")
print(f"  de/da (manual)   = {2 * (a.item()*b_val.item()+a.item()) * (b_val.item()+1)}")
print(f"  de/db (autograd) = {b_val.grad.item()}")
print(f"  de/db (manual)   = {2 * (a.item()*b_val.item()+a.item()) * a.item()}")
print("\n  ✅  autograd matches our manual chain-rule calculation!")


# =============================================================================
# PART 4: Visualize Training
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# ── Plot 1: Loss curves ──────────────────────────────────────────────────
axes[0].plot(losses_manual, label="Manual GD", linewidth=2)
axes[0].plot(losses_nn, label="nn.Module + SGD", linewidth=2, linestyle="--")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Training Loss Over Time")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── Plot 2: Learned line vs data (manual) ────────────────────────────────
with torch.no_grad():
    y_fit_manual = w * X + b
axes[1].scatter(X.numpy(), y_true.numpy(), alpha=0.5, s=20, label="Data")
axes[1].plot(X.numpy(), y_fit_manual.numpy(), "r-", linewidth=2,
             label=f"Fit: y={w.item():.2f}x+{b.item():.2f}")
axes[1].set_title("Manual Gradient Descent Fit")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# ── Plot 3: Learned line vs data (nn.Module) ─────────────────────────────
with torch.no_grad():
    y_fit_nn = model(X_2d).squeeze()
axes[2].scatter(X.numpy(), y_true.numpy(), alpha=0.5, s=20, label="Data")
axes[2].plot(X.numpy(), y_fit_nn.numpy(), "r-", linewidth=2,
             label=f"Fit: nn.Linear")
axes[2].set_title("nn.Module + Optimizer Fit")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("01_gradient_descent_backprop.png", dpi=150)
print("\n📊  Plot saved to 01_gradient_descent_backprop.png")
plt.show()


# =============================================================================
# SUMMARY — Key Takeaways
# =============================================================================
print("""
╔══════════════════════════════════════════════════════════════════════╗
║  KEY TAKEAWAYS                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. requires_grad=True   tells PyTorch to track operations           ║
║  2. loss.backward()      computes all gradients (backpropagation)    ║
║  3. optimizer.zero_grad() clears old gradients (they accumulate!)    ║
║  4. optimizer.step()     updates weights using gradients             ║
║                                                                      ║
║  The training loop is always:                                        ║
║     forward → loss → zero_grad → backward → step                    ║
║                                                                      ║
║  PyTorch autograd builds a computation graph dynamically and uses    ║
║  the chain rule to compute gradients automatically — you never       ║
║  need to derive gradients by hand!                                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
