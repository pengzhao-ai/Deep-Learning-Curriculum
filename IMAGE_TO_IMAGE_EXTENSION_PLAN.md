# Image-to-Image Extension Plan

> **Goal:** Extend the existing 30-day Deep Learning Curriculum with 3 additional notebooks that teach image-to-image tasks (denoising, super-resolution, deblurring) using the same CNN and ViT backbones learned in the main curriculum.

> **Learner:** Optical engineer with camera & computer vision background. Already completed Day 11 (MLP on MNIST, >97% accuracy) and continuing through Days 12-30. The image restoration tasks directly connect to their knowledge of PSFs, spatial filtering, deconvolution, and image quality metrics.

> **Date:** 2026-06-05

---

## Why These Extensions Exist

The main curriculum teaches CNN and ViT exclusively through **classification** (MNIST → CIFAR-10). The user asked: *"Does the knowledge transfer to image-to-image tasks?"* The answer is **yes** — the backbone architectures are identical; only the head and loss function change.

These 3 notebooks bridge that gap:

| Main Curriculum | Extension |
|---|---|
| CNN encoder (classifier) → class label | CNN encoder-decoder → reconstructed image |
| CrossEntropyLoss (compare labels) | MSE/L1 Loss (compare pixels) |
| Classification accuracy | PSNR, SSIM (image quality metrics) |
| ViT for classification | ViT for restoration |

## Connection to Existing Days

| Extension Notebook | Prerequisite Day | What It Reuses | What It Adds |
|---|---|---|---|
| **1. Autoencoder Intro** (after Day 17) | Day 17: CNN conv/pool/ReLU/BN patterns | Same Conv2d, same training loop, same DataLoader | Encoder-decoder concept, ConvTranspose2d, latent space, MSE loss |
| **2. CNN Restoration** (after Day 21) | Day 21: Full CNN pipeline, ResNet skip connections (Day 18) | Skip connections, augmentation, Checkpointing, LR scheduling | U-Net, synthetic noise/degradation, PSNR/SSIM metrics, denoising + super-resolution |
| **3. ViT Restoration** (after Day 27) | Day 27: Self-attention, Transformer encoder, ViT, CNN vs ViT trade-offs | Patch embedding, multi-head attention, position encoding | ViT decoder, hybrid CNN+ViT, deblurring, attention map visualization for restoration |

---

## Notebook 1: `day17b_autoencoder_intro.ipynb`

### Placement
- **Directory:** `day17b_autoencoder_intro/`
- **Prerequisite:** Day 17 (Build Your First CNN)
- **When to study:** After completing Day 17, before Day 18

### What It Teaches

1. **From classifier to autoencoder** — change just the head and loss function
2. **Upsampling methods:**
   - `nn.ConvTranspose2d` (learned upsampling) — from-scratch implementation
   - `nn.Upsample` (nearest/bilinear) — comparison
   - Visual comparison of checkerboard artifacts
3. **Latent space concept** — compress 32×32×3 → bottleneck → reconstruct
4. **Loss functions for pixel-level tasks:**
   - MSE (L2) — why it's the natural choice for images
   - L1 (MAE) — when to use instead
5. **Reconstruction quality evaluation** — visualize input vs. output

### Architecture Progression

```
Section 2: Simple Linear Autoencoder (MNIST)
  784 → 128 → 784  (MLP-based, revisiting Day 11)

Section 3: Convolutional Autoencoder (CIFAR-10)  
  Conv(3→32) → Pool → Conv(32→64) → Pool → Flatten → Linear(4096→128) → 
  Linear(128→4096) → Unflatten → ConvTranspose2d(64→32) → ConvTranspose2d(32→3)

Section 4: Compare upsampling methods side-by-side
```

### Dataset
- **MNIST** (already downloaded from Day 10/11) — for simple linear autoencoder
- **CIFAR-10** (already downloaded from Day 15+) — for convolutional autoencoder
- No new data downloads required

### Training Pipeline
- 20 epochs per model
- Adam optimizer, learning rate 1e-3
- Train/val/test split (same as Day 17)
- History tracking and loss curves
- Visual comparison of original vs. reconstructed images

### Exercises
1. Change bottleneck dimension — how small can you go before quality drops?
2. Replace ConvTranspose2d with Upsample+Conv2d — see the difference
3. Add noise to input — train a **denoising autoencoder** (preview of Notebook 2)
4. Visualize the latent space with PCA (reduce 128D → 2D, color by class)

### Key Takeaways for the Learner
- "I changed the head from Linear(128, 10) to Linear(128, 4096)+ConvTranspose2d and now my CNN outputs images instead of class labels"
- "MSE loss makes intuitive sense — I'm comparing pixel values directly"
- "The autoencoder learns to compress images into a meaningful latent space"

---

## Notebook 2: `day21_extra_cnn_restoration.ipynb`

### Placement
- **Directory:** `day21_extra_cnn_restoration/`
- **Prerequisite:** Day 21 (Build CNN from Scratch — complete project)
- **When to study:** After completing Day 21, before Day 22

### What It Teaches

1. **Synthetic degradation pipeline** — programmatic image corruption:
   - **Gaussian noise** (denoising task) — additive white noise with controllable σ
   - **Gaussian blur** (deblurring task) — PSF convolution with controllable kernel size
   - **Downscaling** (super-resolution task) — bicubic downscale 2×
   - All degradation parameters controllable — tuneable to match real-world scenarios

2. **U-Net architecture** — the canonical encoder-decoder for image-to-image:
   - Encoder: Conv → BN → ReLU → MaxPool (same as Day 17 CNN)
   - Bottleneck: deepest features
   - Decoder: UpConv → BN → ReLU with **skip connections** from encoder
   - Skip connections preserve fine details (extends ResNet concept from Day 18)

3. **Image quality metrics (from scratch):**
   - **PSNR (Peak Signal-to-Noise Ratio)** — dB scale, directly comparable to optics measurements
   - **SSIM (Structural Similarity Index)** — luminance, contrast, structure components
   - Compare metrics to visual perception — what does a 30 dB vs. 25 dB reconstruction look like?

4. **Three restoration tasks with the same backbone:**
   - Denoising: remove additive Gaussian noise
   - Deblurring: invert Gaussian blur (learned deconvolution)
   - Note: super-resolution requires different data loading (downsampled targets)

### Architecture Progression

```
Section 2: Simple CNN Denoiser (baseline)
  Conv(3→32) → ReLU → Conv(32→64) → ReLU → Conv(64→32) → ReLU → Conv(32→3)
  (Same as Day 17's SimpleCNN, but no pooling + classifier replaced by decoder)

Section 3: U-Net Denoiser
  Encoder: Conv → Conv → Pool → Conv → Conv → Pool → Conv → Conv
  Decoder: UpConv(+skip) → Conv → UpConv(+skip) → Conv → Conv → Conv
  (Skip connections bridge each encoder layer to its corresponding decoder layer)

Section 4: Deblurring with U-Net
  Same U-Net architecture, different degradation (blur kernel instead of noise)
  Compare: denoising vs. deblurring — which is harder? Why?
```

### Dataset
- **CIFAR-10** (already downloaded) with **synthetic degradations applied on-the-fly**
- Custom `DegradedCIFAR10` Dataset class that applies noise/blur in `__getitem__`
- No new data downloads
- Visualize degradation samples before training

### Training Pipeline
- 30 epochs per model
- Adam optimizer, CosineAnnealingLR
- MSE loss (for pixel comparison)
- PSNR/SSIM tracked per epoch
- Compare CNN baseline vs. U-Net
- Checkpoint best model by validation PSNR
- Visual grid: input (degraded) → output (restored) → ground truth

### Optics Connections Throughout
- "Gaussian blur = convolution with a PSF. We're learning to invert this PSF."
- "PSNR in dB is the same unit you use for camera SNR characterization."
- "Denoising is like a content-aware spatial filter — adapts to edges vs. flat regions."
- "Deblurring is learned deconvolution — the network figures out the inverse filter."

### Exercises
1. Train at different noise levels (σ=0.1 vs. σ=0.5) — how does PSNR drop?
2. Remove skip connections from U-Net — see the quality difference (fine details lost)
3. Try L1 loss instead of MSE — which produces sharper results?
4. Apply your trained denoiser to a real image (load a photo, add synthetic noise, restore)

### Key Takeaways for the Learner
- "U-Net = same CNN encoder I built on Day 17, plus a decoder with skip connections"
- "Skip connections preserve the high-frequency details that pooling throws away"
- "PSNR and SSIM are the standard metrics — I can now quantify restoration quality objectively"
- "Different degradations need different models, but the architecture is the same"

---

## Notebook 3: `day27_extra_vit_restoration.ipynb`

### Placement
- **Directory:** `day27_extra_vit_restoration/`
- **Prerequisite:** Day 27 (CNN vs ViT — Head-to-Head Comparison)
- **When to study:** After completing Day 27, before Day 28

### What It Teaches

1. **ViT for image restoration** — same patch embedding, same Transformer blocks, but with a decoder head that outputs an image instead of a class logit:
   - Patch embedding → Transformer encoder → reshape patches → Conv2d → output image
   - The decoder is essentially the reversal of patch embedding

2. **Three architectures compared on the same restoration tasks:**
   - **CNN U-Net** (from Notebook 2, reused here as baseline)
   - **Pure ViT** with decoder head
   - **Hybrid:** CNN encoder extracts local features → ViT applies global attention → CNN decoder reconstructs

3. **When does global attention help restoration?**
   - Periodic textures (fabric, grass, bricks) → CNN does well locally
   - Large missing regions or structured noise → ViT uses global context
   - Compare with visual examples and PSNR tables

4. **Attention map visualization for restoration:**
   - Visualize which patches the model attends to when reconstructing a corrupted patch
   - See the model "looking" at clean patches far away for reference
   - Compare attention patterns for denoising vs. deblurring
   - Use Attention Rollout (same technique from Day 26)

### Architecture Progression

```
Section 2: Naive ViT Restoration
  PatchEmbed(3, 16, 256) → TransformerEncoder(L=6, H=8, d=256) → 
  Unpatchify(256, 16, 3, 32) → Output image

Section 3: Hybrid CNN+ViT Restoration (best of both worlds)
  CNN Encoder → Flatten patches → Transformer → Reshape → CNN Decoder
  (CNN handles local features, ViT handles global context)

Section 4: Compare all three on denoising + deblurring
```

### Dataset
- **CIFAR-10** (already downloaded) with synthetic degradations
- Same degradation pipeline as Notebook 2 for direct comparison
- Plus: **deblurring** with a larger PSF (Gaussian blur kernel, σ=1.5, kernel=5×5)
- No new data downloads

### Training Pipeline
- 50 epochs for ViT (ViT needs more epochs to converge)
- AdamW optimizer, linear warmup + cosine decay (standard for Transformers)
- Label smoothing? No — this is pixel-level, not classification
- PSNR/SSIM tracked per epoch
- Comparison table at the end: CNN vs. ViT vs. Hybrid on 3 metrics (PSNR, SSIM, params, speed)
- Visual grid showing all three reconstructions side by side

### Optics Connections
- "ViT treats the image as a set of patches and lets each patch 'talk to' every other patch — like having global context instead of just local neighborhood"
- "For deblurring, ViT can look at sharp edges far away to understand what the blur kernel did — CNN can only see locally"
- "The hybrid model is like having both a local spatial filter AND global context — best of both worlds"

### Exercises
1. Vary patch size (4×4, 8×8, 16×16) — how does it affect restoration quality and compute?
2. Compare attention maps for denoising vs. deblurring — different patterns?
3. Train ViT with different number of transformer layers (L=2, 4, 6, 8) — diminishing returns?
4. Apply the trained ViT denoiser to a real-world noisy photo

### Key Takeaways for the Learner
- "ViT can do image restoration too — same backbone, different head"
- "Global attention helps restoration when the degradation affects large regions"
- "Hybrid CNN+ViT is often the best practical choice for image-to-image tasks"
- "The same attention visualization from Day 26 works for restoration — I can see what the model 'looks at'"

---

## Dependencies

### New Packages Required
No new packages required for core functionality. All notebooks use:
- **torch, torchvision** — already installed
- **numpy, matplotlib** — already installed
- **scipy** — already installed (for Gaussian blur kernel generation)
- **scikit-learn** — already installed (for PCA visualization in Notebook 1)

### Optional (recommended for real-world deployment)
- `scikit-image` — `skimage.metrics.structural_similarity` and `peak_signal_noise_ratio` (mentioned as alternative to from-scratch implementation)
- Update `requirements.txt` after user approval

### Key Design Choices

| Choice | Rationale |
|--------|-----------|
| PSNR/SSIM implemented from scratch | Pedagogical value — understand the math before using library functions |
| Synthetic degradation (not real datasets) | Full control over noise type/level; no new downloads; reproducible |
| CIFAR-10 throughout | Consistent with main curriculum; 32×32 is small enough for fast training |
| MSE loss for image tasks | Direct pixel comparison; intuitive for optics background |
| On-the-fly degradation in DataLoader | Clean separation of concerns; augmentations are random each epoch |

---

## Implementation Order

The notebooks must be written in the order the learner will encounter them:

1. **`day17b_autoencoder_intro.ipynb`** — first, because it's the simplest bridge from classification to image-to-image
2. **`day21_extra_cnn_restoration.ipynb`** — second, because it builds on autoencoder concepts with U-Net and real restoration tasks
3. **`day27_extra_vit_restoration.ipynb`** — third, because it requires ViT knowledge from Phase 5

However, for writing purposes, the implementation can proceed sequentially since the later notebooks reference the earlier ones but don't depend on their code output.

---

## File Structure

```
DL_learning/
├── day17b_autoencoder_intro/
│   └── day17b_autoencoder_intro.ipynb
├── day21_extra_cnn_restoration/
│   └── day21_extra_cnn_restoration.ipynb
└── day27_extra_vit_restoration/
    └── day27_extra_vit_restoration.ipynb
```

---

## Success Criteria

After completing these 3 notebooks, the learner will be able to:

1. **Convert any image classifier into an image-to-image model** by changing the head and loss function
2. **Implement synthetic degradation pipelines** for denoising, deblurring, and super-resolution
3. **Build U-Net from scratch** and understand why skip connections matter
4. **Evaluate restoration quality** using PSNR (dB scale, familiar from optics) and SSIM
5. **Compare CNN vs. ViT for restoration** and choose the right architecture
6. **Visualize attention maps for restoration** and interpret what the model "looks at"
7. **Apply trained models to real-world images** with a deployment-ready pipeline