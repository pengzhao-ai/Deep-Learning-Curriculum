# Learning Notes — Day 12: Overfitting & Regularization

> **Date:** 2026-09-25
> **Curriculum:** 30-Day Deep Learning Curriculum
> **Topics Covered:** why mini-batches, loss vs. accuracy, whether regularization actually helps, and the random-label memorization experiment

---

## 1. Why Mini-Batches Instead of Processing All Samples at Once?

**Terminology trap:** "batch" is overloaded.
- *Batch gradient descent* = the **whole training set** ("full-batch").
- What we set as `batch_size=64` is a **mini-batch**.

**Updates per epoch ≠ `epochs × batch_size`.** The correct count is:

```
updates per epoch = ceil( N / batch_size )
```

For Day 12: 400 train samples, `batch_size=64` → **ceil(400/64) = ceil(6.25) = 7 updates/epoch**. Full-batch (batch_size = 400) = **1** update/epoch.

### The four reasons (roughly in order of importance)

| # | Reason | Concretely |
|---|--------|-----------|
| 1 | **Memory** | Can't hold activations + gradients for all N samples. Full-batch on real data = OOM. |
| 2 | **Noisy gradient → better generalization** | Mini-batch gradient is an **unbiased but noisy estimate** of the true gradient. The noise escapes sharp minima/saddles and prefers flatter basins that generalize better. |
| 3 | **Hardware efficiency** | GPUs/MPS love large parallel matmuls. `batch_size=1` wastes the units; full-batch overflows memory. Mini-batch is the sweet spot. |
| 4 | **Update frequency / convergence speed** | More `optimizer.step()` calls per epoch → weights move more often → faster early progress. |

### The deep one: gradient noise as regularization

- Full-batch: exact gradient, deterministic — can march into a **sharp** minimum with no escape.
- Mini-batch: stochastic jitter (like "speckle" in an optical system) lets it escape narrow minima → prefers **flat** minima → better test performance.

```
batch_size = 1        → very noisy, unstable, strong regularization
batch_size = N (full) → exact gradient, stable, can overfit/get stuck
mini-batch (32–256)   → practical sweet spot
```

**Corollaries:**
- **Gradient accumulation:** sum gradients over several mini-batches before `optimizer.step()` to simulate a large batch on limited memory.
- **`shuffle=True` for train, `False` for val/test.** Shuffling makes each mini-batch a random sample (unbiased noise). You must NOT shuffle val/test — you are evaluating, not training.
- **Linear scaling rule:** larger batches need a proportionally larger LR to keep the same effective step size.

---

## 2. Why Track Both Loss and Accuracy? Are They Redundant?

**Not redundant** — they measure different things and each is blind to something the other sees.

| | Loss (Cross-Entropy) | Accuracy |
|---|---|---|
| **Nature** | Continuous, differentiable | Discrete (0/1), non-differentiable |
| **Signal** | *How wrong, and how confident* | *How often the top prediction is right* |
| **Role** | **The thing you optimize** (gradients flow) | **The thing you report / care about** |
| **Handling** | `loss.backward()` → weights update | Cannot be `backward()`-ed (argmax ≈ zero gradient) |
| **Sensitivity** | Sensitive to confidence (logit magnitudes) | Only cares about the *rank* of the top logit |

### Five diagnostic gaps

1. **Loss ↓ while accuracy flat:** model already predicts correctly but keeps raising confidence — cross-entropy drops without flipping a single prediction.
2. **Loss is smooth, accuracy is coarse:** with 100 val samples accuracy moves in 1% jumps; loss moves continuously → far more sensitive.
3. **Overfitting is seen best in loss:** val loss turns upward *before* val accuracy does — that's why early stopping tracks **val loss**, not val accuracy.
4. **Imbalanced data:** high accuracy is possible by always predicting the majority class, but loss stays terrible. Loss is calibration-aware.
5. **Same accuracy, different loss:** "right but unsure" vs. "right and confident" — matters for deployment (confidence thresholds).

### The one-sentence mental model

> **Loss is what you *optimize* (differentiable, sensitive, calibration-aware); accuracy is what you *care about* (interpretable, discrete, the goal). You watch both because they can disagree — and the disagreements are diagnostic.**

`plot_history` panels: **left = loss = early-warning system** (train/val gap, overfitting onset); **right = accuracy = scoreboard** (practical outcome).

---

## 3. Do the Regularization Techniques Actually Help? (Experiment)

**Observation:** with `BigMLP` on 500 MNIST samples, dropout / weight decay / early stopping seemed to make no meaningful difference in the accuracy gap or the plots.

**Reproduced it** (400 train / 100 val, 80 epochs, seed 42, 567,434 params):

| Variant | Train acc | Val acc | **Acc gap** | Train loss | Val loss | **Loss gap** | Best val loss (epoch) | **Test acc (10k)** |
|---|---|---|---|---|---|---|---|---|
| No regularization | 1.000 | 0.930 | +0.070 | 0.0001 | 0.4294 | **+0.429** | 0.288 (ep 11) | 0.846 |
| Dropout 0.5 | 0.993 | 0.910 | +0.083 | 0.0193 | 0.3572 | **+0.338** | 0.280 (ep 13) | 0.848 |
| Weight decay 1e-2 | 0.998 | 0.930 | +0.068 | 0.0253 | 0.2472 | **+0.222** | 0.247 (ep 80) | 0.848 |
| Dropout 0.3 | 1.000 | 0.930 | +0.070 | 0.0010 | 0.4891 | **+0.488** | 0.269 (ep 13) | 0.854 |

### Why the effect looks invisible — five reasons

1. **The signal lives in the *loss gap*, not the accuracy gap.** Weight decay cut the loss gap from +0.429 → +0.222 (**~half**). Dropout 0.5 cut it ~21%. Real effect — wrong meter.
2. **The task is saturated.** All variants land at **84.6–85.4% test accuracy** (±0.8 pt). A 567K-param MLP on 400 clean MNIST samples already generalizes about as well as it can — no headroom.
3. **Dropout *widens* the accuracy gap on purpose** (+0.083 > +0.070) because it lowers **train** accuracy more than val. Judging dropout by the gap makes it look harmful.
4. **Overfitting IS happening — visible in loss.** Best val loss at **epoch 11** for no-reg, then val loss *rises* to 0.429 while train loss collapses to 0.0001. Weight decay never overfits within 80 epochs (best val loss = epoch 80) and reaches the lowest absolute val loss (0.247).
5. **The 100-sample val set is the hidden bug.** 1% granularity, ~±3% standard error; and val (0.930) sits far above test (0.846) → biased-optimistic measuring stick.

### What to actually look for

- The **val-loss valley-then-rise** (overfitting onset epoch) — not the endpoints.
- Compare **val-loss minima** across techniques.
- Expect **train accuracy to DROP** with dropout — that's success, not failure.

### How to make regularization visibly matter

- Shuffle labels to remove the learnable signal (see §4) — the dramatic demo.
- Sweep hyperparameters: `p ∈ {0.1, 0.3, 0.5, 0.7}`, `weight_decay ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1}`.
- Use a **larger validation set** (e.g. 5,000) so deltas are resolvable.
- Increase capacity/epochs to create overfitting headroom.
- Move to CIFAR-10 (Day 15+) — a harder task where regularization/augmentation finally pay off.

---

## 4. The Random-Label Experiment — The Fundamental Principle

### What "keep 400 samples but shuffle `y`" means

We keep the **same 400 images** but replace each label with a **random** one (a permutation of the true labels). Each 28×28 image now gets some *other* image's digit as its label. The result: there is **no consistent, learnable relationship** between pixels and labels — the mapping `x → y` is, by construction, statistically independent noise. Validation keeps its **true** labels, so we can test whether anything generalizable was learned.

```python
class RandomLabelDataset(Dataset):
    """Wraps a dataset but returns a RANDOM label per sample (permuted)."""
    def __init__(self, base, seed=0):
        self.base = base
        ys = [int(base[i][1]) for i in range(len(base))]
        perm = torch.randperm(len(ys),
                 generator=torch.Generator().manual_seed(seed)).tolist()
        self.ys = [ys[p] for p in perm]          # <-- labels shuffled
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, self.ys[i]
```

### Result (same `BigMLP`, 80 epochs)

| Variant | Train acc | Val acc (true labels) | **Acc gap** | Train loss | Val loss | Test acc (true labels) |
|---|---|---|---|---|---|---|
| Random labels, no reg | 1.000 | 0.130 | **+0.870** | 0.0009 | **7.9366** | 0.093 |
| Random labels, dropout 0.5 | 0.850 | 0.080 | **+0.770** | 0.4486 | 4.9468 | 0.089 |
| Random labels, weight decay 1e-2 | 0.975 | 0.170 | **+0.805** | 0.1095 | 4.2982 | 0.124 |

**Compare:** real labels → gap ≈ **+0.07 (7 pts)**. Random labels → gap ≈ **+0.87 (87 pts)** — **~12× more severe.**

### The fundamental principle

1. **Overfitting magnitude = memorization capacity − learnable signal.** The model's capacity (567K params) is fixed; randomizing labels zeroes the learnable signal, so *all* training progress is memorization → maximal gap.
2. **Low training loss ≠ learning.** A big enough network can fit *arbitrary* labels. Training loss → 0.0009 while test accuracy → 0.093 (chance for 10 classes). Fitting ≠ generalizing. (Core point of Zhang et al., 2017, *"Understanding Deep Learning Requires Rethinking Generalization."*)
3. **Generalization requires a stable `x → y` mapping that holds on unseen data.** Random labels remove it, so no function can beat chance on the held-out set — no matter how well it fits the training set.

### Two sharp extra lessons from the numbers

- **Val loss explodes (7.94).** Cross-entropy punishes **confident wrong** answers. The model memorizes random labels and becomes wildly overconfident on val → loss blows up while accuracy stays at chance. Loss is the more sensitive meter.
- **Regularization constrains memorization but cannot create signal.** Dropout dropped train acc to 0.85 and lowered val loss (7.94 → 4.95, i.e. better calibrated / less overconfident); weight decay lowered it further (4.30). But val/test accuracy stays at **chance**. Regularization limits *effective capacity* to memorize; it cannot manufacture a relationship that isn't in the data.

> **Bottom line:** Overfitting is not "the model is bad" — it's the model using its capacity to memorize instead of generalize. Regularization reduces the *capacity to memorize*; only the *data* can supply something to generalize. Random labels isolate the two and make the distinction impossible to miss.

---

## 5. Questions for Self-Check

1. Why is `batch_size` not the same as "batch" in *"batch gradient descent"*?
2. How many weight updates happen per epoch with N=400 and `batch_size=64`? With full-batch?
3. Why does mini-batch noise improve generalization compared to full-batch?
4. Why must you `shuffle=True` for training but `shuffle=False` for validation?
5. What does "loss decreasing while accuracy is flat" tell you about the model?
6. Why does early stopping track **val loss** rather than val accuracy?
7. Why does adding dropout sometimes *widen* the train–val accuracy gap?
8. Why didn't regularization improve accuracy much in the Day 12 demo?
9. In the random-label experiment, why does training accuracy reach ~100% while test accuracy is ~10%?
10. What does the random-label experiment prove about the relationship between training loss and learning?

---

## 6. Next Steps

| Day | Topic |
|-----|-------|
| **Day 13** | Batch Normalization & LR Scheduling |
| **Day 14** | Training Pipeline Best Practices (checkpointing, reproducibility) |
| **Day 15** | Working with Image Data (CIFAR-10, data augmentation) |
| **Day 16** | The Convolution Operation ⚡ |

---

*Last updated: 2026-09-25*
