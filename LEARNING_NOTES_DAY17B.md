# Learning Notes — Day 17b: Autoencoder Intro

> **Date:** 2026-10-01
> **Curriculum:** 30-Day Deep Learning Curriculum — Phase 4: CNNs (Days 16–21)
> **Notebook:** `day17b_autoencoder_intro/day17b_autoencoder_intro.ipynb`
> **Prerequisite:** Day 17 — Build Your First CNN
> **Topics Covered:** turning a classifier into an image-to-image model by swapping only the head and the loss; the linear (MLP) autoencoder on MNIST and the convolutional autoencoder on CIFAR-10; `ConvTranspose2d` vs `Upsample+Conv2d`; MSE vs L1 on pixels; what "latent" and "auto"-encoder really mean; how the same framework generalizes to *any* image-to-image task (denoising, deblurring, super-resolution, …); why MNIST reconstructs cleanly but CIFAR-10 looks fuzzy; and the `requires_grad` / `.numpy()` visualization bug (`torch.no_grad()` / `.detach()`).

---

## 1. The One-Line Takeaway

> **Core idea:** A classifier and an autoencoder share the *same* backbone. Only two things change — the **head** (class logits → reconstructed **image**) and the **loss** (CrossEntropy on labels → MSE on **pixels**).

| Component | Classifier (Day 17) | Autoencoder (Day 17b) |
|---|---|---|
| **Head** | `Flatten + Linear → 10 logits` | `Linear + Unflatten + ConvTranspose2d → image` |
| **Loss** | `CrossEntropyLoss` (labels) | `MSELoss` / `L1Loss` (pixels) |
| **Output** | 10 class scores | same size as input (`1×28×28`, `3×32×32`) |
| **Supervision** | human labels | **the input is its own label** (self-supervised) |

**Every classification architecture can become an image-to-image architecture. Swap the head and the loss.**

---

## 2. The Two Models in This Notebook

### 2.1 Linear (MLP) autoencoder — MNIST

```
Input (784) → Linear(784,256) → ReLU → Linear(256,128) → ReLU     ← bottleneck / latent = 128
            → Linear(128,256) → ReLU → Linear(256,784) → Sigmoid → reshape (1,28,28)
```

- **Bottleneck:** 128 dims for 784 pixels → ~**6×** compression.
- **`Sigmoid`** at the end squeezes outputs into `[0,1]` (valid pixel range).
- **Params: 468,368** (encoder 233,856 + decoder 234,512).
- Trace: input `(1,1,28,28)` → latent `(1,128)` → recon `(1,1,28,28)` ✅

**Results (MSE, Adam lr=1e-3, 20 epochs):**

| Epoch | Train MSE | Val MSE |
|---|---|---|
| 1 | 0.044831 | 0.021932 |
| 5 | 0.008124 | 0.007741 |
| 10 | 0.005161 | 0.005171 |
| 20 | 0.003352 | 0.003453 |

Train and val track closely → the encoder learns a **generalizable** compact representation, not memorization.

### 2.2 Convolutional autoencoder — CIFAR-10

```
Encoder:  Conv(3→32)+BN+ReLU+Pool → Conv(32→64)+BN+ReLU+Pool → Conv(64→128)+BN+ReLU+Pool  → (128,4,4)
Bottleneck: Flatten(2048) → Linear(2048→128) → [latent 128]     (fc_enc)
            Linear(128→2048) → reshape (128,4,4)                (fc_dec)
Decoder:  ConvTranspose(128→64, k2 s2)+BN+ReLU → (64,8,8)
          ConvTranspose(64→32,  k2 s2)+BN+ReLU → (32,16,16)
          ConvTranspose(32→3,   k2 s2)+Sigmoid → (3,32,32)
```

- **Encoder = exactly Day 17's `SimpleCNN.features`.** Only the head changed.
- **`ConvTranspose2d`** = *learned upsampling* (reverse of pooling); `kernel=2, stride=2` doubles H and W each step → `4 → 8 → 16 → 32`.
- **Params: 661,795** — encoder **93,696**, the two `fc` layers **526,464** (~**80%**!), decoder **41,635**. Same lesson as Day 17: the `Linear` layers are where the parameters live.
- Trace: input `(1,3,32,32)` → `(1,128,4,4)` → latent `(1,128)` → recon `(1,3,32,32)` ✅

**Upsampling comparison (Exercise 2 / §7 of the notebook):**

| Method | How it upsamples | Trade-off |
|---|---|---|
| `ConvTranspose2d` | **learns** the upsampling kernel | flexible, but prone to **checkerboard artifacts** |
| `Upsample + Conv2d` | fixed interpolation (nearest/bilinear) + learned conv | fewer artifacts, less expressive |

Both produce the same `(1,3,32,32)` output — the difference is *how* they fill in the extra pixels.

---

## 3. The Loss for Images — MSE (and L1)

Classification compared **labels**; reconstruction compares **pixels**:

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i)^2$$

- It is the pixel-level analogue of **squared measurement error** — minimise the average pixel error.
- **PSNR** (Notebook 2) is built directly on it: $\text{PSNR} = 10\log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$.
- **MSE** → smooth, slightly blurry reconstructions; **L1 (MAE)** → often **sharper** (Exercise 4).
- Autoencoders need **no labels** → "self-supervised": the data supervises itself.

> **🔬 Optics analogy:** MSE = minimising residual sensor error over the whole frame; the whole autoencoder is a **learned compression → decompression** pipeline (raw sensor data → JPEG → display).

---

## 4. Decoding the Name: "Latent" and "Auto"-Encoder

**"Latent" = hidden / not directly observed.** The 128-number bottleneck is the *latent representation* (a.k.a. latent code / bottleneck / embedding): a compact, abstract description you never see as an image. It is the **JPEG file** in the camera analogy — it keeps the essential structure and throws away the rest.

**"Auto-encoder":**
- **encoder** = a network that compresses the input into a compact code;
- **auto** = *self* — the target is **the input itself**. There is no human label; the network is trained to encode an input and decode it back to *itself*. That's why it is **self-supervised**.

An encoder–decoder is the general shape; it becomes an *auto*encoder because the decoder's goal is to reproduce the encoder's **own input**.

---

## 5. Beyond Reconstruction — Image-to-Image in General

The autoencoder is the **simplest special case** of image-to-image (a.k.a. image translation). The recipe is always the same three ingredients:

1. an **encoder–decoder network** (compress → transform → expand back to image size),
2. a **per-pixel loss** (MSE/L1, or something fancier),
3. a **paired dataset of `(input, target)` images**.

**The only thing that changes for a non-reconstruction task is what you put in the target slot:**

| Task | Input (what the model sees) | Target / ground truth (what you compare against) |
|---|---|---|
| **Autoencoder (this notebook)** | clean image | **the same clean image** |
| **Denoising** | image + Gaussian noise | the clean image |
| **Deblurring** | image blurred by a PSF | the sharp original |
| **Super-resolution** | low-res (downscaled) image | the high-res original |
| **Inpainting** | image with missing patches | the complete image |
| **Colorization** | grayscale | the original color image |
| **pix2pix** (photo↔map, day↔night, edges→photo) | e.g. an edge map | the photo it should have been |
| **Medical imaging** | low-dose / noisy scan | high-dose scan |

**Only the first row has `target == input`.** Rows 2–8 are still autoencoder-shaped networks, but the ground truth is a *different* image. This notebook is literally the degenerate/simplest version of the whole family.

**Where do the targets come from?**
- **Paired data (synthesised):** `day21_extra_cnn_restoration.ipynb` takes a clean CIFAR image, applies noise / blur / downscaling to make the *input*, and keeps the **clean image as the target**. It defines a `DegradedCIFAR10` dataset returning `(degraded_image, clean_image)` pairs — same loss, different target.
- **Unpaired data:** when pairs are impossible, use CycleGAN-style methods (out of scope here, worth knowing).

**Two caveats the later notebooks address:**
- Pure MSE gives **blurry** output (the model hedges toward the average). Advanced methods add **perceptual** and **adversarial** losses to sharpen results.
- A tight bottleneck **loses detail** → the restoration notebook upgrades to a **U-Net with skip connections** that carry fine detail around the bottleneck.

> **Answer to the "will this be covered later?" question:** Yes. Day 21 ext (CNN restoration) and Day 27 ext (ViT restoration) both build directly on this notebook — same encoder–decoder skeleton, new target (and a beefier loss / architecture).

---

## 6. Reading the Results — Why MNIST Looks Close, CIFAR Fuzzy

### 6.1 MNIST / MLP — close to the original, errors only at the **edges**

- **The task is easy:** grayscale, centered, low-resolution, low-complexity → 128 dims for 784 pixels is a comfortable ~6× compression.
- **Interior pixels are trivial; edges are hard.** Inside a stroke, pixels are almost all black or all white — easy to reproduce. Edges are the only place with rapid spatial transitions (high-frequency detail), and that is exactly where compression + a smooth `Sigmoid` output loses the most. → the error map lights up **along the outlines**.
- **The `cmap="hot"` error map exaggerates.** It maps *any* nonzero value to red/yellow. The MNIST MSE is tiny (~0.003) so absolute errors are small, yet the hottest pixels still glow. The visualisation makes it look worse than it is.

### 6.2 CIFAR / CNN — visibly fuzzy / washed-out (expected!)

Several independent effects stack up:

1. **Harder data:** `3×32×32 = 3072` values (4× more than MNIST), full colour, textures, clutter, 10 very different classes.
2. **Tighter compression:** a 128-dim bottleneck for 3072 values is ~**24×** compression vs MNIST's ~6×.
3. **MSE = "hedge toward the average."** When unsure about a pixel, MSE's optimum is to output the *mean* → blurry, desaturated, greyish results. This is the single biggest visual reason.
4. **Normalisation + `clamp(0,1)`** during denormalisation for display can wash out highlights.
5. **`ConvTranspose2d` checkerboard artifacts** (the notebook flags this explicitly).
6. **Only 20 epochs** — neither model is fully converged, CIFAR far from it.

### 6.3 What it is really demonstrating

- **Autoencoders prioritise low-frequency structure and discard high-frequency detail.** Both models keep overall shape / position / rough colour; *detail* suffers — more so with a tight bottleneck and complex data.
- **Difficulty scales with data complexity and compression ratio.** MNIST@6× ≈ easy; CIFAR@24× ≈ hard. It is **not** an apples-to-apples comparison — MNIST is the toy warm-up.
- **MSE is blurry and a plain bottleneck loses detail — which motivates what's next:** U-Net **skip connections** (recover detail) and better evaluation/losses (**PSNR/SSIM**, later perceptual/adversarial).

**Sanity checks that confirm the reasoning (Exercises):** increase `latent_dim` (e.g. 512), train longer, or swap `ConvTranspose2d` for `Upsample+Conv2d` — each should visibly sharpen the result.

---

## 7. The `requires_grad` / `.numpy()` Visualisation Bug

**Symptom:** `RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.`

**Where:** in the plotting cells, when matplotlib's `imshow` tries to turn a tensor into a NumPy array.

**Why it happens:**
- Model outputs (`reconstructions`) inherit `requires_grad=True` from the weights.
- `error = (images[i] - reconstructions[i]).abs()` **inherits that tag too**.
- `imshow` internally calls `.numpy()`, and PyTorch **forbids** that on a tensor that still tracks gradients.
- **Key gotcha:** `model.eval()` does **not** remove the tag — it only changes Dropout/BatchNorm behaviour. Only `torch.no_grad()` or `.detach()` removes it.

**Fix (both layers of safety):**

```python
# Fix 1 — proper inference: no gradient tracking at all
with torch.no_grad():
    reconstructions, latents = model_linear(images)

# Fix 2 — safety net: strip the tag before handing numbers to matplotlib
error = (images[i] - reconstructions[i]).abs().cpu().detach().squeeze()
im = axes[2, i].imshow(error, cmap="hot")
```

Either alone fixes it; both together is bulletproof. The same latent bug existed in the CIFAR-10 cell's `recons_dn` line, so it got the same treatment.

> **Rule of thumb:** any time you plot/inspect a tensor (`.numpy()`, `imshow`, `plt.plot`), do inference under `torch.no_grad()` and/or call `.detach()` first.

---

## 8. Optics Connections (Day 17b)

| DL concept | Optics / imaging analogue |
|---|---|
| Autoencoder | **compression → decompression** pipeline (sensor data → JPEG → display) |
| Latent / bottleneck | the **JPEG file** — a compact code that keeps essential structure |
| `ConvTranspose2d` | **learned upsampling** — reconstructing a larger image from a small map |
| MSE loss | minimising **residual sensor error** over the whole frame |
| PSNR (next notebook) | camera **SNR** characterisation, in dB |
| Denoising / deblurring (Day 21 ext) | content-aware spatial filter / **learned deconvolution** (invert the PSF) |

---

## 9. Practical Notes — Data & Companion

- **Data (no re-download):** `day17b_autoencoder_intro/data/` links **MNIST → `day14/data/MNIST`** and shares `./data` for CIFAR-10, so both load fully offline (`download=True` is a safe no-op). These `data` links are `.gitignore`d.
- **Device:** `mps` on this Mac (`torch 2.12.0`, `torchvision 0.27.0`).
- **Budget:** both models train 20 epochs (Adam, lr=1e-3). MNIST ≈ 2 s/epoch; CIFAR-10 takes a few minutes total.
- **Gotcha — notebooks + the IDE:** if the notebook is open in VS Code while you edit it on disk, saving from the stale editor buffer **overwrites** the disk file. Reopen / `File: Revert File` before saving, or your fix (or the `no_grad`/`detach` change) can silently revert.
- **Gotcha — autograd vs plotting:** wrap inference in `torch.no_grad()`; `eval()` alone does not stop gradient tracking (§7).

---

## 10. Questions for Self-Check

1. What two things change when you turn a classifier into an autoencoder?
2. Why is it called an **auto**encoder, and where does its supervision come from?
3. In plain words, what is the **latent** space, and why is it "hidden"?
4. What does `Sigmoid` at the decoder output accomplish, and why does the linear AE need it?
5. `ConvTranspose2d(kernel=2, stride=2)`: what does it do to H and W, and how does `4×4 → 32×32` happen in three steps?
6. Which layers hold ~80% of the ConvAE's parameters, and what does that echo from Day 17?
7. Give two non-reconstruction image-to-image tasks, and for each state the **input** and the **target**.
8. When the target is *not* the input, where does the ground-truth come from (two ways)?
9. Why does the CIFAR-10 reconstruction look blurrier than MNIST? Name at least three reasons.
10. Why does MSE tend to produce *blurry* images, and what kind of loss/architecture helps?
11. Why does `imshow` fail on a tensor with `requires_grad=True`, and what is the fix? Does `model.eval()` help?
12. Why does the MNIST error map light up **at the edges** specifically?

---

## 11. Next Steps

| Day | Topic |
|-----|-------|
| **Day 16** ✅ | The Convolution Operation |
| **Day 17** ✅ | Build Your First CNN |
| **Day 17b** ✅ | Autoencoder Intro — **complete** (encoder–decoder, `ConvTranspose2d`, latent space, MSE vs L1, image-to-image framing) |
| **Day 18** | CNN Architectures — VGG blocks & ResNet skip connections |
| **Day 19** | CNN Deep Dive — feature visualisation, CAM, dead neurons |
| **Day 20** | Transfer Learning & Fine-Tuning |
| **Day 21** | Build a CNN from Scratch — Complete Project (>90% CIFAR-10) |
| **day21_extra** | CNN Restoration — U-Net denoising / deblurring / super-resolution (**pairs with today's target-slot idea**) |
| **day27_extra** | ViT Restoration — attention-based deblurring |

**CNN arc ordering:** Day 16 → 17 → **17b** → 18 → 19 → 20 → 21 → **21_extra**.
Do **17b right after Day 17** — it reuses the conv/pool mechanics while fresh and introduces `ConvTranspose2d`, which the U-Net needs later.

**Bridge forward:** the "target ≠ input" table in §5 is the direct setup for **Day 21 ext** (synthetic degradation → clean target), and the MSE-is-blurry lesson there motivates skip connections and perceptual losses.

---

*Last updated: 2026-10-01*
