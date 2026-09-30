# Learning Notes — Day 16: The Convolution Operation

> **Date:** 2026-09-29
> **Curriculum:** 30-Day Deep Learning Curriculum — Phase 4: CNNs (Days 16–21)
> **Notebook:** `day16/day16_convolution_operation.ipynb`
> **Topics Covered:** the convolution operation from scratch, classical filters vs learned filters, `F.conv2d` vs `nn.Conv2d`, output-size formula (padding & stride), `out_channels` = number of filters (not layers), the 4-D weight layout, weight sharing vs dense layers, feature maps, and a notebook bug we fixed.

---

## 1. The Convolution Operation (the core idea)

**Definition:** slide a small **kernel** (a.k.a. filter) across the image; at each position compute the element-wise product of the kernel and the image patch, then sum it (a **dot product**). The result is one number per position → a new, smaller image called a **feature map**.

Implemented from scratch (no PyTorch), stride 1, **no padding**:

```python
def manual_conv2d(image, kernel):
    """Simple 2D convolution (no padding, stride=1)."""
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+kh, j:j+kw]     # the window under the kernel
            output[i, j] = np.sum(patch * kernel)   # dot product
    return output
```

**Worked example** (verified in the notebook): a 4×4 input and a 3×3 "edge" kernel `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]` produce a **2×2** output:

```
Input (4×4)                Output (2×2)
[[1. 2. 3. 0.]             [[ -4.   4.]
 [0. 1. 2. 3.]     →        [-12.  -4.]]
 [3. 0. 1. 2.]
 [2. 3. 0. 1.]]
```

Note the shrink: 4×4 → 2×2. **Why** it shrinks (and how to stop it) is Section 2.

> **🔬 Optics connection:** a convolution kernel *is* a **PSF** (point-spread function). Applying the same kernel at every location is exactly the assumption of a **linear shift-invariant** optical system — one impulse response, convolved with the scene. The only CNN difference: the kernel is **learned**, not designed.

**Key properties that make convolution ideal for images:**
- **Local receptive field** — each output looks only at a small neighborhood (like a PSF's support).
- **Weight sharing** — the *same* kernel is reused at every position (see Section 6).
- **Translation equivariance** — shift the input, and the feature map shifts with it.

---

## 2. Output Size: Padding & Stride

The one formula that governs everything:

$$\text{out} = \left\lfloor \frac{\text{in} + 2\cdot\text{padding} - \text{kernel}}{\text{stride}} \right\rfloor + 1$$

Verified cases for a 32×32 input (from the notebook's `show_conv_output_size`):

| kernel | stride | padding | output | note |
|---|---|---|---|---|
| 3 | 1 | **0** | **30** | shrinks by 2 (the "valid" default) |
| 3 | 1 | **1** | **32** | preserves size — **"same" padding** |
| 5 | 1 | 2 | 32 | same padding for 5×5 |
| 7 | 1 | 3 | 32 | same padding for 7×7 |
| 3 | **2** | 1 | **16** | stride 2 halves the spatial size |

**"Same" padding rule:** to keep output = input size with stride 1,

$$\text{padding} = \frac{\text{kernel} - 1}{2}$$

So kernel 3 → pad 1, kernel 5 → pad 2, kernel 7 → pad 3.

### Why Section 1 shrank but Section 2 didn't (our Q&A)

Both used a **3×3 kernel, stride 1** — identical. The only difference was **padding**:

- **§1 `manual_conv2d`** → `padding = 0`. For 4×4: `(4 + 0 − 3) + 1 = 2` → **2×2**.
- **§2 classical filters** → `F.conv2d(gray, k4d, padding=1)`. For 32×32: `(32 + 2 − 3) + 1 = 32` → **32×32**.

Section 2 chose `padding=1` for a **reason**: it displays the six filtered results side-by-side with the original 32×32 image, so keeping every panel the same size makes them directly comparable. Without padding they'd all be 30×30.

> **Mental model (optics):** padding is a guard ring of "dark" pixels around the sensor so the sliding PSF can be *centered* on edge pixels instead of hanging off the array — you get an output sample for every input pixel, not just the interior.

**Buffer detail:** without padding you lose `kernel − 1` pixels per side; the output is smaller than the input. This is exactly why CNNs in practice use `padding = kernel_size // 2` for every "same" conv (and why Day 17's `SimpleCNN` writes `padding=1` everywhere).

---

## 3. Classical Filters = Hand-Designed Convolutions

The same convolution math **is** classical image processing — a human picks the kernel:

| Filter | Kernel | What it does |
|---|---|---|
| Identity | `[[0,0,0],[0,1,0],[0,0,0]]` | passes the image through unchanged |
| Blur (box) | `ones(3,3)/9` | local average → smoothing |
| Sharpen | `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | boosts center vs. neighbors |
| Edge (Laplacian) | `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]` | responds to intensity change → edges |
| Sobel X | `[[-1,0,1],[-2,0,2],[-1,0,1]]` | horizontal gradient |
| Sobel Y | `[[-1,-2,-1],[0,0,0],[1,2,1]]` | vertical gradient |

**The punchline of Day 16:**

> *"These are the same convolution kernels used in image processing — but in a CNN, the network **learns** them!"*

So there are two modes for the same operation:

| Mode | Who picks the kernel | API used |
|---|---|---|
| **Classical / fixed** | a human (blur, Sobel, …) | `F.conv2d(x, hand_made_kernel)` |
| **Learned** | gradient descent during training | `nn.Conv2d(...)` |

---

## 4. `F.conv2d` vs `nn.Conv2d` — Function vs Module

PyTorch ships **two parallel APIs** that compute the same convolution:

| | `torch.nn.functional` (`F`) | `torch.nn` (`nn`) |
|---|---|---|
| What it is | a **pure function** | a **class** (an `nn.Module`) |
| Holds parameters? | ❌ No — you pass the weights in | ✅ Yes — owns learnable `weight`/`bias` (`nn.Parameter`) |
| Example | `F.conv2d(x, w, b, padding=1)` | `nn.Conv2d(3, 16, 3, padding=1)` |
| In `model.parameters()`? | No | Yes (the optimizer can train it) |
| Use for | stateless ops; **fixed** kernels | **learnable** layers |

**The relationship:** an `nn` module is a thin wrapper around the `F` function.

```python
# nn.Conv2d.forward() is essentially:
def forward(self, x):
    return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
```

This pairing repeats throughout PyTorch:

| Module (`nn.`) | Functional (`F.`) |
|---|---|
| `nn.Conv2d` | `F.conv2d` |
| `nn.Linear` | `F.linear` |
| `nn.ReLU` | `F.relu` |
| `nn.MaxPool2d` | `F.max_pool2d` |
| `nn.BatchNorm2d` | `F.batch_norm` |
| `nn.Dropout` | `F.dropout` |

### "Is conv2d called as a function in both cases?"

Almost — but the semantics differ:

- `F.conv2d(...)` → a **literal function call** (pure math, no state).
- `nn.Conv2d(...)` → **instantiation**; then `conv(x)` **calls the instance**. That goes through `nn.Module.__call__`, which runs hooks and then dispatches to `forward()` — which calls `F.conv2d` internally.

So `nn.Conv2d(...)` is the `__init__` (builds/stores the parameters) and `conv(x)` is the `forward` (does the math) — the exact Day-8 distinction.

### The signature and the 4-D weight layout

```python
F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
#        ▲       ▲
#      the     the kernel
#     image    tensor
```

`F.conv2d` receives **real tensors**, not hyperparameters. The four "numbers" you write in `nn.Conv2d(in_ch, out_ch, k, ...)` are **not four arguments** to `F.conv2d` — they describe the shape of a **single 4-D weight tensor**:

```
weight shape = (C_out, C_in, kH, kW)
                 ▲      ▲    ▲   ▲
              out_ch  in_ch  kH  kW
```

Verified identity:

```
k.shape                        = (3, 3)
k.view(1,1,3,3).shape          = (1, 1, 3, 3)
nn.Conv2d(1,1,3).weight.shape  = (1, 1, 3, 3)   ← the same shape
```

Input/output conventions: `input = (N, C_in, H, W)`, `weight = (C_out, C_in, kH, kW)`, `output = (N, C_out, H', W')`.

### What `k.view(1, 1, 3, 3)` does

`view` is **reshape** (like NumPy's `np.reshape`): it **reinterprets the same 9 numbers with a new shape, no copy**.

```python
k = torch.tensor([[0,-1,0],[-1,5,-1],[0,-1,0]])   # (3, 3) — 9 values
k4d = k.view(1, 1, 3, 3)                          # (out_ch=1, in_ch=1, 3, 3) — same 9 values
```

- `out_channels = 1` → this one filter makes **one** feature map.
- `in_channels = 1` → the input `gray` is `(1, 1, 32, 32)` (grayscale).
- It simply **prepends the two channel dimensions** a raw `(3,3)` kernel is missing.

Equivalent forms (all identical):

```python
k.view(1, 1, 3, 3)             # fast; needs contiguous data
k.reshape(1, 1, 3, 3)         # same, copies if needed
k.unsqueeze(0).unsqueeze(0)   # add two size-1 dims — torch.equal(...) → True
k[None, None]                 # fancy-index shorthand
```

**The whole point of §2:** you are hand-building the `(1,1,3,3)` weight that `nn.Conv2d` would normally auto-create — because here the "PSF" is **fixed**, not learned.

### Two gotchas we hit live

1. **Bias.** `nn.Conv2d` adds a learnable bias **by default**; `F.conv2d(x, k)` passes `bias=None`. To compare them, set `nn.Conv2d(..., bias=False)`.
2. **Floating point.** Even with identical weights, `nn.Conv2d` and `F.conv2d` differed by `max |Δ| = 2.4e-6` (different conv-algorithm accumulation order), so `torch.allclose` at its default `atol=1e-8` returned **`False`**. It passed at `atol=1e-5`. **Never compare floats with `==` — use `torch.allclose(a, b, atol=1e-5)`.**

---

## 5. What `out_channels` Really Means (Channels ≠ Layers)

```python
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#                       ▲
#                  out_channels = 16
```

**`out_channels = 16` means 16 feature maps / 16 filters — it does NOT mean 16 hidden layers.**

- The layer holds **16 kernels**, each of shape `3×3×3` (= `kH × kW × in_channels`). Verified: `conv.weight.shape = (16, 3, 3, 3)`.
- Each of the 16 kernels is applied **in parallel to the same input**, producing one feature map each.
- Those 16 maps are **stacked along the channel dimension** → the "16" in `(1, 16, 32, 32)`.
- Params: weight `16×3×3×3 = 432` + bias `16` = **448**.

### "Depth" is overloaded — keep two meanings apart

| Term | Meaning | Example here |
|---|---|---|
| **Tensor channel / "depth"** | the `C` in `C × H × W` | `C = 16` |
| **Network depth** | number of **stacked layers** | 3 (if you stack 3 convs) |

`out_channels` increases the *first*; it is unrelated to the *second*. To get "16 layers" you stack 16 conv layers sequentially, each consuming the previous output.

### Analogies

- **Optics:** 16 **parallel filters with 16 different PSFs**, all imaging the *same* scene simultaneously → 16 filtered images stacked. Not 16 sequential relay stages.
- **MLP bridge:** a hidden layer with **N hidden units** → an N-dim vector. A conv layer with **N out_channels** → N *feature maps*. So `out_channels` ≈ "number of hidden units," just kept **spatial** instead of flattened.

### What "depth" looks like in code

```python
# THIS is depth = 3 conv layers (a 3-stage network). Note the second number of each
# line equals the first number of the next — channel counts are the layer interface:
nn.Conv2d(3,  32, 3, padding=1)    # 3  → 32
nn.Conv2d(32, 32, 3, padding=1)    # 32 → 32
nn.Conv2d(32, 64, 3, padding=1)    # 32 → 64
```

**One more detail:** each output channel **sums over all input channels** (its kernel spans `in_channels`). So one feature map is a weighted combination of the filtered R, G, and B — it cannot look at a single color in isolation. That's why `in_channels` must equal the previous layer's `out_channels`.

---

## 6. Weight Sharing: Why Conv Beats Dense on Images

The notebook compares a dense layer that produces 16 features per pixel against a conv layer of the same output width:

```python
dense = nn.Linear(32 * 32 * 3, 32 * 32 * 16)   # 3072 → 16384
conv  = nn.Conv2d(3, 16, kernel_size=3, padding=1)
```

| Layer | Parameters |
|---|---|
| Dense `Linear(3072, 16384)` | **50,348,032** |
| Conv `Conv2d(3, 16, 3, padding=1)` | **448** |
| Ratio | **≈ 112,384× fewer** |

**Why:** the dense layer has a *separate* weight for **every (input-pixel, output-pixel, channel)** pair. The conv layer **shares** the same `3×3×3` kernel across every spatial location — 16 kernels × (9×3 weights + 1 bias).

> **Key takeaway:** weight sharing (translation-invariant reuse of one small kernel) is the biggest reason CNNs are practical for images. It also gives **translation equivariance** for free — a shifted object activates the same filter, just shifted.

---

## 7. Feature Maps & the Random-vs-Learned Arc

Section 6 applies a **randomly initialized** conv layer to a CIFAR image:

```python
conv1 = nn.Conv2d(3, 9, kernel_size=3, padding=1)   # 3 → 9 feature maps
img, label = cifar[42]                              # (3, 32, 32)
img_batch = img.unsqueeze(0)                        # (1, 3, 32, 32)

with torch.no_grad():
    features = conv1(img_batch)                     # (1, 9, 32, 32)

# each channel is one filter's response → one panel per filter
for i in range(1, features.shape[1] + 1):
    fmap = features[0, i-1]                         # feature map of filter i
```

**Shapes:** `(1, 3, 32, 32) → conv1(3→9) → (1, 9, 32, 32)` = **9 feature maps**, each 32×32. `features[0, k]` is the activation map of filter `k` — a grayscale image of "where filter k fired strongly."

### Why random filters? (the intent)

The section title says *"Visualizing Learned Features,"* but `conv1` is **random and never trained** — so nothing has been learned yet. That is deliberate: it's the **"before" picture**.

- Random kernels produce arbitrary/noisy maps — they detect nothing meaningful.
- **That is the teaching point:** *a randomly initialized filter is useless; training is what turns random kernels into meaningful feature detectors.*

| Section | Filters | Meaningful? |
|---|---|---|
| **§2 — classical** | *hand-designed* (blur, Sobel, edge) | ✅ a human crafted them |
| **§6 — this block** | *random, untrained* | ❌ arbitrary patterns |
| **Day 17 §4 — learned** | *trained by gradient descent* | ✅ the **network discovered** them |

So §6 sets up the core question of Phase 4: *how does a network go from random noise to edge detectors?* → **backpropagation** applied to conv weights (your Days 5–7 skills).

**What to look for:** don't try to interpret individual random maps. Confirm the **structure** (one 32×32 map per output channel) and feel the contrast once Day 17 shows the *trained* version.

### 🐞 Bug we found and fixed

The original cell had an **off-by-one** between the plot grid and the channel count:

```python
conv1 = nn.Conv2d(3, 8, ...)          # 8 output channels → valid indices 0..7
fig, axes = plt.subplots(2, 5)        # 10 panels = 1 "Original" + 9 feature maps
for i in range(1, 10):                # i = 1..9 → channel index 0..8  ← index 8 doesn't exist!
    fmap = features[0, i-1]
```

At `i = 9` it raised:

```
IndexError: index 8 is out of bounds for dimension 1 with size 8
```

**Cause:** the grid implies **9** filters (1 original + 9 = 10), but the layer was built with **8**. **Fix applied** — make the layer produce 9 filters, and make the loop robust:

```python
conv1 = nn.Conv2d(3, 9, kernel_size=3, padding=1)     # 9 filters ↔ 9 panels
for i in range(1, features.shape[1] + 1):             # iterate exactly over channels
```

**Lesson:** when looping over channels, anchor the bound to `features.shape[1]`, not a hard-coded number.

---

## 8. Practical Notes — Data Setup (no re-download)

Every CNN-arc notebook calls `torchvision.datasets.CIFAR10(root="./data", download=True, ...)`, where `./data` resolves **relative to each notebook's own folder**. Day 16 originally stalled trying to **re-download 170 MB** into `day16/data`.

**Resolution:** one shared CIFAR-10 copy (in `day15/data`) + per-folder **symlinks** named `data`:

```
day16/data -> ../day15/data     (also day17 … day21, day21_extra)
day17b_autoencoder_intro/data/  -> CIFAR via symlink; MNIST -> ../../day14/data/MNIST
```

A symlink is a tiny file holding only a **path**; the OS transparently redirects reads to the real data. Net effect: one dataset, zero extra copies, no re-download. Verified that `CIFAR10(root="./data", download=True)` and `MNIST(...)` load fully offline — `download=True` becomes a safe no-op when the data is present. These `data` links are ignored by `.gitignore`.

---

## 9. Questions for Self-Check

1. Write the output-size formula. What padding keeps a 32×32 input at 32×32 with a 3×3 kernel?
2. Why did §1's manual convolution shrink 4×4 → 2×2 while §2's filtered a 32×32 image to 32×32, with the same 3×3 kernel?
3. State the "same padding" rule as a function of kernel size.
4. What is the difference between `F.conv2d` and `nn.Conv2d`? Which one owns trainable parameters?
5. What is the required shape of the `weight` argument to `F.conv2d`, and what do its four dimensions mean?
6. What does `k.view(1, 1, 3, 3)` do, and why is it needed? Give one equivalent expression.
7. Does `nn.Conv2d(3, 16, 3)` create 16 layers? Explain what the 16 is, and distinguish "tensor channel depth" from "network depth."
8. If a conv layer has `in_channels=32`, what must the preceding layer's `out_channels` be, and why?
9. Why does this conv layer use ~112,384× fewer parameters than an equivalent-width dense layer?
10. Why do the feature maps in §6 (random conv) look meaningless, and what changes by Day 17?
11. Why should you write `torch.allclose(a, b, atol=1e-5)` instead of `a == b`?

---

## 10. Next Steps

| Day | Topic |
|-----|-------|
| **Day 16** ✅ | The Convolution Operation — **complete** |
| **Day 17** | Build Your First CNN (`SimpleCNN`, conv→pool→FC, train on CIFAR-10) 🚀 |
| **day17b** | Autoencoder Bridge (`day17b_autoencoder_intro`) — light ~1 h session |
| **Day 18** | CNN Architectures — VGG blocks & ResNet skip connections |
| **Day 19** | CNN Deep Dive — Feature visualization, CAM, dead neurons |
| **Day 20** | Transfer Learning & Fine-Tuning |
| **Day 21** | Build a CNN from Scratch — Complete Project (>90% CIFAR-10) |
| **day21_extra** | CNN Restoration — U-Net denoising / deblurring |

**Reminder — CNN arc ordering:** Day 16 → 17 → **17b** → 18 → 19 → 20 → 21 → **21_extra**.
Do 17b right after Day 17 (reuses conv/pool mechanics while fresh, and teaches `ConvTranspose2d` needed by the U-Net later).

---

*Last updated: 2026-09-29*
