# Learning Notes — Day 21 (Extension): CNN for Image Restoration — U-Net, Denoising & Deblurring

> **Date:** 2026-10-05
> **Curriculum:** 30-Day Deep Learning Curriculum — Phase 4: CNNs (Days 16–21) — **extension notebook**
> **Notebook:** `day21_extra_cnn_restoration/day21_extra_cnn_restoration.ipynb`
> **Prerequisite:** Day 21 (the CNN capstone), Day 18 (`VGGNet` + `ResidualBlock`/`SimpleResNet` — the **skip-connection** idea reused here as U-Net cross-links), Day 17b (autoencoder intro — encoder–decoder + `ConvTranspose2d`, and the *"MSE is blurry"* caveat this notebook sets out to fix)
> **Topics Covered:** turning a classifier into an **image-to-image restorer**; the synthetic degradation pipeline (`add_gaussian_noise`, `apply_gaussian_blur`, `bicubic_downscale`) and the `DegradedCIFAR10` `(degraded, clean)` dataset; the **U-Net** — contracting path, bottleneck, expanding path, and **concatenation skip connections**; U-Net's **relation to Day 18's ResNet** (additive vs concatenative skips); from-scratch **PSNR** and **SSIM**; the **denoising-vs-deblurring** comparison, and — the heart of this note — **why the raw PSNR/SSIM/visualizations appeared to contradict the "deblurring is harder" verdict**.

---

## 📝 Conversation Q&A Log (Day 21 extension discussion)

> **Purpose:** a record of the *questions raised during the Day 21-extension session* and the answers given, so the discussion is not lost (distinct from the reference material below).

### Round 1 — Kick-off: "educate me U-Net — and is it related to the ResNet we learned in Day 18?"

**Q.** What are U-Net fundamentals, and is it similar/related to the ResNet from Day 18?

**A.** A **U-Net** is an **encoder–decoder** (contracting + expanding path) joined by **skip connections between matching resolution levels**; the name comes from the **U shape**. It is **related in spirit** to Day 18's ResNet — both are "shortcut/bypass" architectures (the Day 18 note already anticipates this: *"the residual/skip connection mastered today is the exact mechanism the U-Net uses as encoder↔decoder cross-links"*) — but they are **cousins, not twins**: ResNet skips are **additive** (`x + F(x)`), same-resolution, within a stage, and exist to fix **depth/optimization**; U-Net skips are **concatenative** (`torch.cat`), cross-network, shape-allowing, and exist to **recover the spatial detail that pooling destroyed**. Full treatment in §2 and §3.

### Round 2 — "why did PSNR, SSIM and the visualization show the *opposite* of the verdict — deblur scored better than denoise?"

**Q.** The terminal cell prints a verdict that **deblurring is harder than denoising**, yet the actual PSNR/SSIM numbers and the visual grids made **deblurring look easier**. Why?

**A.** Because the number you read (absolute PSNR/SSIM against the clean image) is **not a difficulty measure when the two tasks start from different baselines**. Four things conspire (full detail in §6):

1. **Unmatched severity.** The notebook used **σ_noise = 0.2** vs **σ_blur = 1.5**. I reproduced the exact degradation functions over CIFAR-10's 10 000 test images and measured the **degraded input itself**: noise → **14.78 dB / SSIM 0.489**, blur → **21.06 dB / SSIM 0.800**. The deblurrer starts **~6 dB ahead**, so of course its final absolute PSNR sits higher.
2. **Deterministic vs stochastic.** Blur here is **one fixed, known PSF** → the net learns **one inverse filter** (well-posed). Noise is **re-randomized every epoch** → the net can only learn a statistical **regularizer** with an **irreducible residual-noise floor**.
3. **The metric itself favors blur.** PSNR penalizes the broadband, variance-heavy noise far more than low-pass blur; **SSIM is designed to be insensitive to blur but harsh on noise** (it lives on local luminance/contrast/structure, which blur preserves).
4. **The visual grids show the same artifact** — the denoiser must *smooth* to kill strong noise (looks soft), the deblurrer just *re-sharpens* using a memorized fixed PSF (looks crisp).

So: the verdict is **true but conditional** (it holds for *matched, destructive, realistic* degradations); the empirical "reversal" is a **measurement/severity confound**, and the honest difficulty signal is **ΔPSNR over the degraded input** (where the **denoiser has to work much harder**). §6 gives the full analysis and how to fix the experiment.

---

## 1. Where this sits — the restoration extension in the arc

`day21_extra_cnn_restoration/` **follows Day 21** and closes the loop on the one thing the whole CNN arc did *not* do: **produce an image instead of a label**.

- **Day 17b** introduced the encoder–decoder and `ConvTranspose2d`, and showed that a plain autoencoder reconstructs CIFAR-10 **fuzzily** (MSE averaging, tight bottleneck). It ended with a promise: *tight bottleneck → **U-Net skip connections***, *MSE is blurry → perceptual losses*.
- **Day 18** built the **skip connection** — but for **classification** (additive, within-stage, `output = F(x) + x`).
- **Day 21** re-deployed that skip as the capstone's `ResBlock`.
- **Day 21 extra (this note)** uses the skip a third way: as **cross-links** from the encoder to the decoder in a **U-Net**, so a *same-size image* can be restored with **both context and fine detail**.

The one-line framing: **same tokens as the classifier, new grammar.**

| Component | Classifier (Day 21) | Restoration (this notebook) |
|---|---|---|
| **Output** | 10 class probabilities | degraded image → restored image (**same size**) |
| **Loss** | CrossEntropy | MSE (pixel comparison) |
| **Architecture** | Encoder → FC/GAP head | Encoder → Decoder (**U-Net**) with skip connections |
| **Evaluation** | Accuracy (%) | **PSNR (dB)**, **SSIM (0–1)** |

The task is **self-supervised**: the "label" is just the **clean image**, and the input is a **synthetically degraded** copy of it — exactly the *"target slot changes"* picture from Day 17b. `DegradedCIFAR10` fabricates the `(degraded, clean)` pairs on the fly, so no new dataset downloads are needed.

---

## 2. What U-Net fundamentally is

### 2.1 The three parts

| Part | What it does | In this notebook |
|---|---|---|
| **Contracting path (encoder)** | Downsample, raise channel count, build *semantic/context* | `enc1→pool1→enc2→pool2→enc3→pool3` |
| **Bottleneck** | Lowest resolution (4×4), highest semantics | `bottleneck` (64→512→256 channels) |
| **Expanding path (decoder)** | Upsample back to input size, reconstruct pixels | `up3→dec3→up2→dec2→up1→dec1` |

The spatial ladder (32×32 input) runs **32 → 16 → 8 → 4 → 8 → 16 → 32**, which is what draws the **U**.

### 2.2 The forward pass (from the notebook)

```python
# --- encoder (save each feature map for the skips) ---
s1 = enc1(x)      # (64, 32, 32)   ← saved for skip
p1 = pool1(s1)    # (64, 16, 16)
s2 = enc2(p1)     # (128, 16, 16)  ← saved for skip
p2 = pool2(s2)    # (128, 8, 8)
s3 = enc3(p2)     # (256, 8, 8)    ← saved for skip
p3 = pool3(s3)    # (256, 4, 4)

# --- bottleneck ---
b  = bottleneck(p3)                    # (256, 4, 4)

# --- decoder (concatenate the matching encoder map at each level) ---
d3 = up3(b);   d3 = torch.cat([d3, s3], 1); d3 = dec3(d3)   # (128, 8, 8)
d2 = up2(d3);  d2 = torch.cat([d2, s2], 1); d2 = dec2(d2)   # (64, 16, 16)
d1 = up1(d2);  d1 = torch.cat([d1, s1], 1); out = dec1(d1)  # (3, 32, 32)
```

- **Upsampling** is `nn.ConvTranspose2d(kernel=2, stride=2)` — *learned* upsampling (the Day 17b mechanism); each one **doubles H and W**.
- **The skip is `torch.cat`, not `+`** — the encoder map is **concatenated** onto the (upsampled) decoder map along the **channel** axis, so `dec*`'s first conv consumes `2×` the channels (e.g. `dec3`: `Conv2d(base*8 → base*2)`). That "double right after the cat" is the fingerprint of a concatenative skip.

### 2.3 Why each piece exists

- **Why encoder→decoder?** An image-to-image task must output a **full-resolution image the same size as the input**. A classifier throws spatial layout away (ends in a label). So "swap only the head + loss" (Day 17b) now means: the head is a **decoder**, not a `Linear`.
- **Why downsample?** Pooling grows each pixel's **receptive field** — a bottleneck pixel "sees" the whole 32×32 image and can reason about *context* (sky vs. grass). Same reason CNNs pool for classification.
- **Why is a plain encoder–decoder not enough?** Downsampling **destroys high-frequency/spatial detail**; the bottleneck is a lossy compression, so the decoder can only recover smooth, low-frequency content — the **fuzzy CIFAR reconstructions** of Day 17b.
- **Why the skips fix that?** `torch.cat([d, s], 1)` hands the decoder the encoder's **own fine-detail map** at that resolution — the edges/texture pooling had discarded — so the decoder gets **both** big-picture context (from the bottleneck) **and** sharp spatial detail (from the skips). This is the single most important idea in U-Net, and it's why the skip-connected model beats `SimpleDenoiser` (4 conv layers, no pooling, no skips — a *single-scale* filter with no way to see context).


### 2.4 Parameters (from the saved checkpoint)

`best_unet_denoiser.pt` / `best_unet_deblur.pt` hold **4,818,051** parameters at `base_channels=64` (99 weight tensors). Breakdown:

| Block | Params | Share |
|---|---|---|
| `bottleneck` (64→512→256 @ 4×4) | 2,363,136 | **49%** |
| encoder (`enc1`+`enc2`+`enc3`) | 1,148,992 | 24% |
| decoder (`dec1`+`dec2`+`dec3`) | 961,411 | 20% |
| up-convs (`up1..up3`) | 344,512 | 7% |

Notes: the **bottleneck holds ~half the weights** — it is where channels are widest (`base*8 = 512`). And each `dec*`'s **first** conv takes `2×` channels *because of the preceding `cat`*. (Contrast with Day 17b's ConvAE, where **80%** of params sat in the two `Linear` layers — here there is no giant FC layer at all; it's fully convolutional.)

---

## 3. U-Net vs ResNet (Day 18) — cousins, not twins

Both are post-2015 **skip-connection** architectures. The difference is in **operation, geometry, and purpose**:

| | **ResNet block (Day 18)** | **U-Net skip (this notebook)** |
|---|---|---|
| Connection | `out = relu(F(x) + identity)` — **addition** | `torch.cat([decoder, encoder], dim=1)` — **concatenation** |
| Spatial relationship | **Within one stage**, same resolution/location | **Across the whole net**: bridges encoder level *k* ↔ decoder level *k* (two opposite arms of the "U") |
| Channel rule | `+` **requires equal channels** → needs a **1×1 projection shortcut** when shapes differ (`ResidualBlock.shortcut`) | `cat` **allows unequal channels** → the following conv absorbs the extra channels |
| Purpose | Fix the **degradation/optimization problem**; give gradients a low-resistance path; make identity `F(x)→0` "free" so **very deep** nets train | **Recover the spatial detail lost by downsampling**; enable **dense, same-size output** |
| Task | Classification (input → **label**) | Dense prediction (input → **image**: segmentation, restoration, …) |
| Mental image | Keeps a **gradient/signal path** to the input | Keeps a **detail path** from the encoder |

More connections worth remembering:

- **The U-Net's encoder *is* essentially the Day 18 VGG stage pattern**: stacked `Conv3×3 → BN → ReLU`, `MaxPool` between stages, channels **doubled** each stage (`64→128→256`); the `bottleneck` is a ResNet-style stage (`256→512→256`).
- **`+` and `cat` are two ends of one spectrum.** Swap the `cat` for `+` (with matching channels) and you get a *residual U-Net*. Modern U-Nets often embed real `ResidualBlock`s inside the encoder/decoder — **the two ideas compose**.
- **Forward link:** the same `x + sublayer(x)` residual stream reappears in **Transformer/ViT** blocks (Days 23–27), and the `day27_extra` Vit restoration notebook reuses this exact restoration framing.

**One-line mnemonic:** *ResNet **adds a sibling** to fix depth; U-Net **concatenates a stranger** to fix detail.*


---

## 4. The degradation pipeline & dataset

Three synthetic degradations, all applied in **[0,1]** and all **optics-ifiable**:

| Function | Math | Optics reading |
|---|---|---|
| `add_gaussian_noise(img, sigma)` | `clamp(img + randn_like(img) * sigma, 0, 1)` | **sensor noise** (temporal/read noise) — broadband, zero-mean |
| `apply_gaussian_blur(img, k=5, sigma=1.5)` | `scipy.ndimage.gaussian_filter` per channel | convolution with a **Gaussian PSF** (defocus) — a low-pass filter |
| `bicubic_downscale(img, factor=2)` | `F.interpolate` down 2× then back up | **resolution loss** (super-resolution target) |

**`DegradedCIFAR10(Dataset)`** wraps a base CIFAR-10 dataset and returns **`(degraded, clean)`** pairs. Crucially, it **denormalizes → degrades → re-normalizes** inside `__getitem__`, so the degradation physics happens in **[0,1]** but the tensors the network sees are **normalized** (same CIFAR mean/std as Day 15/17). Because degradation is applied **on the fly each epoch**, the noise is **different every epoch** and the blur is re-applied — free "augmentation for restoration."

For the two compared tasks the notebook builds two datasets from the same splits:
- **Denoising:** `DegradedCIFAR10(..., degradation_type="noise", sigma=0.2)`
- **Deblurring:** `DegradedCIFAR10(..., degradation_type="blur", blur_sigma=1.5, blur_kernel=5)`

---

## 5. Metrics — PSNR & SSIM (from scratch)

Both are implemented **from scratch** (pedagogical: understand the math before reaching for `scikit-image`).

### PSNR — Peak Signal-to-Noise Ratio (dB)

```python
mse  = F.mse_loss(pred, target)          # mean over all pixels & channels
psnr = 10 * torch.log10(max_val**2 / mse)   # max_val = 1.0 for [0,1] images
```

- The dB scale is exactly the **SNR characterization** an optical engineer already uses.
- **A 6 dB gap = a 4× MSE gap** (each 3 dB doubles the error power). Handy for reading results.
- **PSNR is a function of MSE only** — so it is *monotone in residual error*; it says nothing about *perceptual* quality or *task difficulty*.

### SSIM — Structural Similarity (0–1)

Computed with an **11×11 Gaussian window** slid over the image (via `F.conv2d`), giving local means `μ`, variances `σ²`, and covariance `σ_xy`, then per the original paper:

```
SSIM = ((2·μ_x·μ_y + C1)(2·σ_xy + C2)) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
```

- It decomposes similarity into **luminance** (`μ`), **contrast**, and **structure** (`σ`) terms.
- **Key property for §6:** SSIM is comparatively **tolerant of blur** but **sensitive to noise** — blur preserves `μ` and low-frequency structure, while noise wrecks the local-variance terms.

`max_val=1.0` and `C1=0.01²`, `C2=0.03²` in the notebook (the standard defaults).


---

## 6. Denoising vs deblurring — why the numbers looked "backwards"

The notebook's final comparison cell **hard-codes** the verdict:

```python
print("📝 Which is harder?")
print("Deblurring is typically harder than denoising because:")
print("1. Blur destroys high-frequency information (edges, texture) irreversibly")
print("2. The model must hallucinate (guess) missing high frequencies")
print("3. Denoising preserves edges — just removes random fluctuations")
```

That text is **printed, not computed** — it is a **general statement of theory**, and it is *not* guaranteed to match the run's raw numbers. In this run, deblurring's **absolute** PSNR/SSIM came out **higher** than denoising's. That is **not** a contradiction in the model; it is a contradiction in **what you are comparing**. Four effects explain it.

### 6.1 The two tasks were not severity-matched (the biggest reason)

The notebook uses **σ_noise = 0.2** but **σ_blur = 1.5**. Nothing makes those "equal difficulty." I ran the notebook's **exact degradation functions** over CIFAR-10's 10 000 test images and measured the **degraded input itself** (no model involved):

| Task | **Input PSNR** | **Input SSIM** | Input MSE (per element) |
|---|---|---|---|
| Denoise (σ = 0.2) | **14.78 dB** | **0.489** | **0.0333** |
| Deblur (σ = 1.5, 5×5) | **21.06 dB** | **0.800** | **0.0078** |

(Sanity check: `10·log10(1/0.0333) = 14.78` ✓ and `10·log10(1/0.0078) = 21.06` ✓.)

This is the crux: **the deblurring model starts ~6 dB ahead and from a 0.80-SSIM input, while the denoiser starts from 0.49 SSIM.** The deblurrer only has to beat a much *higher* floor, so its **final absolute PSNR naturally sits higher** — even if it improved *less*. **Absolute PSNR is not a difficulty measure when the input baselines differ.**

Also relevant: **σ = 0.2 is a severe noise** (its std ≈ 0.8 on the standardized scale), whereas **Gaussian blur σ = 1.5 on a 5×5 kernel is mild** — it removes little true information.

### 6.2 Blur is deterministic, noise is stochastic

- **Blur** here = convolution with **one fixed, known PSF** (the same σ = 1.5 kernel on every image). The network only needs to learn **one inverse filter** — a well-posed, consistent mapping. It essentially learns a sharpening / frequency-boost filter and applies it reliably.
- **Noise** is **re-randomized every epoch** (`torch.randn_like`). No fixed inverse exists; the net must learn a statistical **regularizer** balancing **bias vs. variance**. At σ = 0.2 it **cannot fully cancel** the noise, so a **residual-noise floor** is irreducible and caps PSNR.

### 6.3 The metrics themselves favor blur over noise

- **PSNR ∝ −10·log10(MSE).** Blur **removes high frequencies**, and natural images carry little energy up there → small MSE (measured **0.008**). Noise is **broadband** and, although zero-mean, **variance-heavy** → it inflates MSE roughly as σ² (measured **0.033**). So PSNR structurally penalizes the noisy case more.
- **SSIM** works on local luminance/contrast/structure. Blur **preserves** the mean and low-frequency structure → SSIM stays high (**0.80**). Noise **destroys the local variance** `σ_x²` → SSIM collapses (**0.49**). SSIM is documented to be **insensitive to blur but harsh on noise**, so it too rewards deblurring here.

### 6.4 The visualization shows the same artifact

In the denoising grid the restored images look **smoothed/soft** vs. ground truth — because the only way to suppress **strong** noise is to **average locally** (which blurs). In the deblurring grid the restored images look **sharp** — because the model just **re-adds high frequencies** it memorized from the **fixed** PSF. The "winner" in the pictures is therefore also a **baseline/severity** artifact, not proof that deblurring is easier.

### 6.5 Is the verdict *wrong*? No — it's **conditional**

The claim "deblurring is harder" is an **information-theoretic / ill-posedness** statement. It holds when comparing **matched, realistic** conditions:

1. The blur truly **destroys** high-frequency info (a strong PSF / large kernel) — the model must **hallucinate** what is gone. (σ = 1.5, 5×5, 32×32 is *mild*, so little is truly lost.)
2. The PSF is **unknown / motion-varying / large** (real deblurring), not one fixed trivial kernel.
3. You compare **matched severity** and read the **right signal** — not raw PSNR.

### 6.6 How to make the experiment actually *test* the claim

Change the **measure**, not the model:

1. **Report ΔPSNR = PSNR(output) − PSNR(input)** — improvement *over the degraded baseline*:
   - Denoise: ~14.8 dB → (e.g.) ~24 dB ⇒ **Δ ≈ +9 dB** (big gain, hard-fought).
   - Deblur: ~21.1 dB → (e.g.) ~26 dB ⇒ **Δ ≈ +5 dB** (small gain, easy).
   The **denoiser has to work much harder** — the theory's point shows up in **Δ**, not in the absolute number.
2. **Match severity** — sweep noise σ / blur σ until both *inputs* sit at comparable PSNR (e.g. ~21 dB), *then* compare outputs.
3. **Use a matched classical baseline** — Wiener / pseudo-inverse deconvolution for blur vs. a Gaussian smoothing filter for noise — and compare how much the CNN beats *its own* baseline. (Optics reading: deblurrer = learned **deconvolution**; denoiser = content-aware **spatial filter**.)
4. **Sweep blur strength**: as blur σ grows, deblurring's PSNR will fall below denoising's — the verdict emerges once the degradation is genuinely destructive.

**Bottom line:** the numbers aren't lying and the verdict isn't wrong — the run compared **different severities of different degradation types** using a **metric that structurally favors blur**. Absolute PSNR measures *residual error vs. the clean image*; it does **not** measure *which task was harder*. Hardness lives in **ΔPSNR over the degraded input** (and in the perceptual realism of hallucinated detail) — and by that measure, **denoising is the tougher fight** in this setup.


---

## 7. Optics connections

| DL concept | Optics analogue |
|---|---|
| Gaussian blur (`gaussian_filter`) | convolution with a **Gaussian PSF** (defocus) |
| U-Net deblurrer | **learned deconvolution** — the network discovers the inverse of the PSF (adaptive, content-aware — unlike a fixed Wiener filter) |
| Denoiser | **content-aware spatial filter** — behaves differently at edges vs. flat regions |
| Additive Gaussian noise | **sensor noise** (temporal/read noise), zero-mean, broadband |
| PSNR (dB) | **camera SNR** characterization you already use |
| SSIM | **structural** image-similarity metric (luminance/contrast/structure) |
| `bicubic_downscale` | **resolution loss** (MTF cutoff) → super-resolution target |

---

## 8. Exercises (from the notebook)

1. **Noise-level sweep** — train the denoiser at σ = 0.1 / 0.2 / 0.5. How does PSNR change with noise (and why does it *saturate*)?
2. **Remove the skip connections** — build an encoder→bottleneck→decoder with **no** skips and compare PSNR. *How much quality do you lose?* (This isolates the U-Net's whole point.)
3. **L1 vs MSE loss** — train with `L1Loss` instead of `MSELoss`. Which looks **sharper**? (MSE averages → soft; L1 is more robust → crisper.)
4. **Real-image denoising** — take a real photo, add synthetic noise, run it through the trained denoiser. Does it **generalize** to real images (vs. the σ you trained on)?

---

## 9. Practical Notes — Data, Device, Artifacts

- **Data (no re-download):** `day21_extra_cnn_restoration/data` is a **symlink → `../day15/data`** (the shared CIFAR-10 copy), so `root="./data"` loads fully offline. These `data` links are `.gitignore`d.
- **Device:** `mps` on this Mac (Apple GPU) with CPU fallback.
- **Budget:** three models × 30 epochs (Adam `lr=1e-3`, `CosineAnnealingLR(T_max=30)`, `batch_size=64`): `SimpleDenoiser` (baseline), U-Net denoiser, U-Net deblurrer. Checkpoints land beside the notebook.
- **Artifacts present:** `best_unet_denoiser.pt` and `best_unet_deblur.pt` (~19 MB each, 4.82 M params).
- **Checkpointing pattern:** save whenever `psnr > best_psnr` (val PSNR), then reload best before final eval — same discipline as Day 14.
- **Gotcha — notebooks + the IDE:** if the notebook is open in VS Code while you edit it on disk, saving from a stale editor buffer can **overwrite** your changes. Reopen / `File: Revert File` first.

---

## 10. Key Takeaways

| Concept | Summary |
|---|---|
| **Image-to-image = swap head + loss** | classifier head → **decoder**; CrossEntropy → **MSE**; label → **the clean image itself** |
| **U-Net = U-shape + cat skips** | encoder↓ + bottleneck + decoder↑; `torch.cat` bridges matching resolutions to restore detail |
| **Skip in U-Net vs ResNet** | U-Net = **concatenate** (cross-level, unequal channels OK, detail); ResNet = **add** (within-stage, equal channels, depth/optimization) |
| **Degradation is on-the-fly** | `DegradedCIFAR10` returns `(degraded, clean)`; noise re-randomized each epoch |
| **PSNR & SSIM from scratch** | PSNR ∝ −10·log10(MSE) (dB, = SNR); SSIM = local luminance/contrast/structure |
| **Metrics ≠ difficulty** | absolute PSNR compares two tasks fairly **only** when input baselines match; report **ΔPSNR** instead |
| **Why deblur "wins" here** | unmatched severity (14.8 vs 21.1 dB inputs) + fixed deterministic PSF + a metric that favors blur |
| **The verdict is conditional** | "deblur is harder" holds for **matched, destructive, realistic** degradations — not for a mild fixed PSF |

---

## 11. Questions for Self-Check

1. What are the three parts of a U-Net, and what does each contribute?
2. Why does an image-to-image model need a **decoder** rather than a classifier head?
3. What exactly does `torch.cat([d3, s3], 1)` hand the decoder, and why does it fight the fuzzy reconstructions of Day 17b?
4. State the difference between a ResNet skip (`F(x)+x`) and a U-Net skip (`cat`) in **operation**, **geometry**, and **purpose**.
5. Why must `ResNet` insert a 1×1 convolution in the shortcut when shapes change, but a U-Net `cat` needs no such thing?
6. In this U-Net, why does each `dec*` block's **first** conv take `2×` the channels? *(Because the preceding `cat` doubled them.)*
7. Where do most of the U-Net's 4.82 M parameters live, and why? *(The bottleneck — widest channels.)*
8. Write PSNR in terms of MSE. If output MSE drops from 0.04 to 0.01, by how many dB does PSNR rise? *(6 dB.)*
9. Why is SSIM comparatively **insensitive to blur** but **sensitive to noise**?
10. Name the four reasons the deblurring PSNR/SSIM beat the denoising PSNR/SSIM in this run.
11. What was the **input** PSNR/SSIM for each task (measured)? Which model started ahead?
12. Why is **ΔPSNR over the degraded input** a better difficulty signal than absolute PSNR?
13. Under what conditions *does* the "deblurring is harder" claim hold?
14. Why does the denoised image look **softer** than the deblurred one in the visual grids?

---

## 12. Next Steps

| Day | Topic |
|-----|-------|
| 16–20 | ✅ done (convolution → first CNN → AE → VGG/ResNet → visualization → transfer learning) |
| **Day 21** | ✅ Build a CNN from Scratch — Capstone (>90% CIFAR-10) — **complete** |
| **Day 21 extra** | ✅ CNN Restoration — U-Net denoising / deblurring / super-resolution — **complete** (this note) |
| **Day 22** | Self-Attention & The Transformer Revolution (Q, K, V, scaled dot-product) |
| day27_extra | ViT Restoration — attention-based deblurring |

**Bridge forward:** the **skip connection** is the through-line of the whole CNN arc — additive residual in Day 18, reused as `ResBlock` in Day 21, now **concatenative cross-links** in the U-Net here — and it becomes the **Transformer residual stream** `x = x + sublayer(x)` from Day 23 onward. The restoration framing (degradation → clean target, PSNR/SSIM, U-Net) is re-opened with attention in `day27_extra` (ViT restoration).

---

*Last updated: 2026-10-05*

