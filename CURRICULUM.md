# 🎯 30-Day Deep Learning Curriculum

> **Learner profile:** Optical engineer with camera & computer vision background.  
> **Experience:** New to Python, ML/DL, and PyTorch — starting from the basics.  
> **Final goal:** Build and train a CNN and a ViT model from scratch, understand their differences.

---

## 📋 Structure

- **One folder per day:** `day01/`, `day02/`, …, `day30/`
- **One Jupyter notebook per day** with rich Markdown explanations + executable code cells
- **Every concept is explained before code appears** — no assumed knowledge
- **Exercises at the end** of most notebooks for self-practice
- Progressive difficulty — each day builds on the previous

---

## 🗓️ Phase 1: Python, NumPy & ML Basics (Days 1–3)

> *Build the foundation so the rest of the journey is smooth.*

| Day | Topic | What You'll Learn |
| --- | ----- | ----------------- |
| 01 | **Python Essentials for DL** | Variables, types, lists, dicts, loops, functions, f-strings; classes & `__init__`/`__call__`; list comprehensions; importing modules — just the pieces you'll need |
| 02 | **NumPy for Tensor Thinking** | `ndarray` creation, shapes, reshaping, indexing, slicing; broadcasting; vectorized math; random numbers; plotting with matplotlib — this maps directly to PyTorch tensors later |
| 03 | **What is Machine Learning?** | Supervised vs. unsupervised; features, labels, train/test split; what a "model" is; fit a line with sklearn `LinearRegression`; loss = "how wrong am I?"; intuition for *learning from data* |

---

## 🗓️ Phase 2: PyTorch Foundations (Days 4–9)

> *Learn the toolkit you'll use every day for the rest of the curriculum.*

| Day | Topic | What You'll Learn |
| --- | ----- | ----------------- |
| 04 | **Tensors — PyTorch's Building Block** | Tensor creation (`torch.tensor`, `zeros`, `ones`, `randn`, `linspace`); shapes, dtypes, reshaping; indexing & slicing; CPU vs. GPU (`.to()`); comparison to NumPy |
| 05 | **Autograd & Computation Graphs** | `requires_grad=True`; how PyTorch records operations; `.backward()`; `.grad`; `torch.no_grad()`; visualizing a computation graph; the chain rule in plain English |
| 06 | **Gradient Descent from Scratch** | Fit `y = wx + b` manually; forward pass → loss → backward → update; learning rate; watching loss decrease; visualize the loss landscape and the path GD takes |
| 07 | **Backpropagation Deep Dive** | Chain rule with a multi-node graph; manual backprop on a 2-layer network; verify every gradient against autograd; intuition: "each node passes blame backward" |
| 08 | **`nn.Module`, Layers & Optimizers** | `nn.Module` explained (it's just a class!); `nn.Linear`, `nn.Sequential`; `model.parameters()`; `optim.SGD`, `optim.Adam`; the canonical training loop: forward → loss → `zero_grad` → backward → step |
| 09 | **Loss Functions & Activation Functions** | MSE, Cross-Entropy, BCE — when and why; ReLU, Sigmoid, Tanh, Softmax — what they do visually; combine them into a small classifier on toy data |

---

## 🗓️ Phase 3: Building & Training Neural Networks (Days 10–15)

> *Go from toy examples to real datasets and real training practices.*

| Day | Topic | What You'll Learn |
| --- | ----- | ----------------- |
| 10 | **Data Loading & Preprocessing** | `Dataset`, `DataLoader`, `transforms`; batching, shuffling; normalization (why & how); load MNIST images as a first real dataset |
| 11 | **Your First Neural Network (MLP on MNIST)** | Build a multi-layer perceptron; flatten images; train end-to-end; evaluate accuracy; confusion matrix; celebrate your first working model 🎉 |
| 12 | **Overfitting & Regularization** | Train vs. val loss curves; what overfitting looks like; dropout (`nn.Dropout`); weight decay; early stopping; experiment and plot |
| 13 | **Batch Normalization & LR Scheduling** | What BatchNorm does (normalize activations); `nn.BatchNorm1d/2d`; LR schedulers (`StepLR`, `CosineAnnealingLR`); see the effect on convergence speed |
| 14 | **Training Pipeline Best Practices** | Refactor into clean functions (`train_one_epoch`, `evaluate`); checkpointing (`torch.save`/`torch.load`); reproducibility (`manual_seed`); simple logging & plotting |
| 15 | **Working with Image Data** | Images as tensors (C×H×W); RGB channels; `torchvision.datasets` (CIFAR-10); data augmentation (`RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`) — why augmentation matters |

---

## 🗓️ Phase 4: Convolutional Neural Networks — CNN (Days 16–21)

> *The workhorse of computer vision. Convolutions will feel natural to you as an optics engineer — they're spatial filters!*

| Day | Topic | What You'll Learn |
| --- | ----- | ----------------- |
| 16 | **The Convolution Operation** | What a kernel/filter does (PSF analogy from optics!); stride, padding, output size formula; `nn.Conv2d`; apply hand-crafted filters (edge detection, blur) and visualize results |
| 17 | **Build Your First CNN** | Build a CNN from scratch; Conv → BN → ReLU → Pool architecture; train on CIFAR-10; understand how feature extractors + classifier heads work together |
| 18 | **CNN Architectures — VGG & ResNet** | VGG-style blocks (stacked 3×3 filters); ResNet skip connections; implement both patterns; understand the historical evolution of CNN design |
| 19 | **CNN Deep Dive — Feature Visualization** | Visualize learned filters, feature maps at different depths, activation patterns; understand *what* a CNN learns; debug and trust your model |
| 19b | **Network Visualization (extension)** | Draw & inspect a CNN's *structure*: `torchinfo` layer tables, **VisualTorch** flow/graph/lenet figures, **Netron** model-file viewer, TensorBoard graph; optional `torchviz`/`torchview`. Complements Day 19's *behavior* views (filters, feature maps, CAM) |
| 20 | **Transfer Learning & Fine-Tuning** | Use a pre-trained ResNet-18 (ImageNet); feature extraction vs. fine-tuning strategies; adapt to new tasks with minimal data |
| 21 | **Build a CNN from Scratch — Complete Project** | Design, train, evaluate, and debug your own CNN; apply all techniques (augmentation, BatchNorm, Dropout, scheduling); target >90% on CIFAR-10 |

---

## 🗓️ Phase 5: Attention, Transformers & Vision Transformer — ViT (Days 22–28)

> *A completely different paradigm: instead of local filters, use global attention.*

| Day | Topic | What You'll Learn |
| --- | ----- | ----------------- |
| 22 | **Self-Attention & The Transformer Revolution** | Self-attention from first principles; Query/Key/Value; multi-head attention; how attention differs from convolution; the "Attention Is All You Need" paper |
| 23 | **The Transformer Encoder Block** | LayerNorm (Pre-Norm vs Post-Norm); feed-forward network (GELU, expansion factor); residual connections; assemble a complete Transformer Encoder block; stack multiple blocks |
| 24 | **Patch Embeddings — Images to Sequences** | Split images into patches; linear projection and Conv2d trick; the [CLS] token; learnable positional encoding; patch size trade-offs |
| 25 | **Build a ViT from Scratch** | Assemble PatchEmbedding → TransformerBlocks → MLP head; train on CIFAR-10; warmup + cosine decay; label smoothing; AdamW; weight initialization |
| 26 | **ViT Attention Visualization** | Visualize [CLS] attention per layer and per head; attention rollout; positional embedding analysis; compare to CNN feature maps |
| 27 | **CNN vs ViT — Head-to-Head** | Train CNN & ViT with similar parameters; compare accuracy, speed, data efficiency; subset experiments; inductive bias discussion |
| 28 | **Advanced Architectures** | Hybrid CNN+Transformer models; ConvNeXt-style blocks; depthwise separable convolutions; design convergence between CNNs and Transformers |

---

## 🗓️ Phase 6: Consolidation & Capstone (Days 29–30)

> *Bring it all together.*

| Day | Topic | What You'll Learn |
| --- | ----- | ----------------- |
| 29 | **Complete Training Toolkit** | Reusable Trainer class; model zoo (MLP, CNN, ResNet, ViT); grand benchmark; training cheat sheet; architecture selection guide |
| 30 | **Capstone — From Optics to Deep Learning** | Image quality/degradation classification; synthetic degradations (blur, noise, JPEG); end-to-end project; confusion matrix analysis; reflection on 30-day journey; next steps roadmap |

---

## 📦 Required Packages

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
jupyter
ipykernel
scikit-learn
Pillow
tqdm
tensorboard
einops
```

> **Optional — visualization extras (Day 19b):** `requirements-viz.txt` adds `torchinfo`, `visualtorch`, `netron`, `onnx` (pure-pip). `torchviz` / `torchview` additionally need the system Graphviz `dot` binary (`brew install graphviz`).

---

## 📂 Folder Structure (Preview)

```
DL_learning/
├── CURRICULUM.md              ← this file
├── README.md
├── requirements.txt
├── 01_gradient_descent_backprop.py   ← (existing warm-up script)
│
├── day01/
│   └── day01_python_essentials.ipynb
├── day02/
│   └── day02_numpy_matplotlib.ipynb
├── day03/
│   └── day03_what_is_ml.ipynb
├── day04/
│   └── day04_tensors.ipynb
├── day05/
│   └── day05_autograd.ipynb
├── day06/
│   └── day06_gradient_descent.ipynb
├── day07/
│   └── day07_backpropagation.ipynb
├── day08/
│   └── day08_nn_module_layers_optimizers.ipynb
├── day09/
│   └── day09_loss_and_activation_functions.ipynb
├── day10/
│   └── day10_data_loading.ipynb
├── day11/
│   └── day11_first_nn_mlp_mnist.ipynb
├── day12/
│   └── day12_overfitting_regularization.ipynb
├── day13/
│   └── day13_batchnorm_lr_scheduling.ipynb
├── day14/
│   └── day14_training_pipeline.ipynb
├── day15/
│   └── day15_working_with_images.ipynb
├── day16/
│   └── day16_convolution_operation.ipynb
├── day17/
│   └── day17_build_first_cnn.ipynb
├── day18/
│   └── day18_cnn_architectures.ipynb
├── day19/
│   └── day19_cnn_deep_dive.ipynb
├── day19b_network_visualization/
│   └── day19b_network_visualization.ipynb
├── day20/
│   └── day20_transfer_learning.ipynb
├── day21/
│   └── day21_cnn_from_scratch_project.ipynb
├── day22/
│   └── day22_self_attention.ipynb
├── day23/
│   └── day23_transformer_encoder.ipynb
├── day24/
│   └── day24_patch_embedding.ipynb
├── day25/
│   └── day25_build_vit.ipynb
├── day26/
│   └── day26_vit_attention_visualization.ipynb
├── day27/
│   └── day27_cnn_vs_vit.ipynb
├── day28/
│   └── day28_advanced_architectures.ipynb
├── day29/
│   └── day29_training_toolkit.ipynb
└── day30/
    └── day30_capstone.ipynb
```

---

## ⏱️ Estimated Daily Time

- **Days 1–3:** ~45 min – 1 hour (Python/NumPy/ML basics)
- **Days 4–9:** ~1–1.5 hours (PyTorch foundations)
- **Days 10–15:** ~1–1.5 hours (training real models)
- **Days 16–22:** ~1.5–2 hours (CNN deep dive)
- **Days 23–28:** ~1.5–2 hours (Transformer/ViT)
- **Days 29–30:** ~2–3 hours (comparison & capstone)

---

## 💡 Tips

1. **Run every cell.** Don't just read — execute the code and experiment with changes.
2. **Break things on purpose.** Change a learning rate, remove a layer, see what happens.
3. **Read the errors.** PyTorch error messages are verbose but informative. Practice reading them.
4. **Connect to your optics intuition.** Convolution filters ≈ optical PSFs; attention ≈ adaptive spatial weighting.
5. **Don't memorize — understand.** If a concept is fuzzy, re-read the Markdown and tweak the code.
6. **It's OK to re-do a day.** If something didn't click, repeat it before moving on.

---

> **Ready to start?** Review this plan and let me know if you'd like to adjust any topics, pacing, or emphasis areas. Once approved, I'll start generating the daily notebooks!
