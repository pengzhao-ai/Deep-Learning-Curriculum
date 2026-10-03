# Learning Notes — Day 19b (Extension): Network Visualization — Drawing & Inspecting a CNN's Architecture

> **Date:** 2026-10-02
> **Curriculum:** 30-Day Deep Learning Curriculum — Phase 4: CNNs (Days 16–21)
> **Notebook:** `day19b_network_visualization/day19b_network_visualization.ipynb`
> **Prerequisite:** Day 17 (`SimpleCNN`) and Day 18 (`SimpleResNet`); pairs with Day 19 (behavior visualization)
> **Topics Covered:** the **structure vs behavior** distinction; the architecture-visualization toolchain (`torchinfo`, **VisualTorch**, **Netron**, TensorBoard, optional `torchviz`/`torchview`, PlotNeuralNets); a **clean, fully-encapsulated** install strategy; verified outputs; and how to read each diagram.

---

## 📝 Conversation Q&A Log (Day 19b discussion)

> **Purpose:** a record of the *questions raised during the Day 19b session* and the answers given (distinct from the reference material below).

### Round 1 — Kick-off: recap, preview, and "what is Day 19, really?"

**Q. Recap the CNN arc so far, and preview Days 19–21.**
**A.** Recap: **16** (convolution operation) → **17** (first CNN) → **17b** (autoencoder bridge) → **18** (VGG blocks & ResNet skips). Preview: **19** (feature visualization / interpretability) → **20** (transfer learning & fine-tuning) → **21** (CNN-from-scratch capstone, >90% CIFAR-10), with **21_extra** (U-Net restoration) after.

**Q. Can Day 19 be understood as "a standardized framework to inspect a given CNN network" as its main theme?**
**A.** **Yes on the theme, no on "framework."** Day 19's theme is *model interpretability* ("the network is inspectable, not a black box"), but its content is a **collection of ad-hoc recipes**, not a reusable framework/class. More importantly, it inspects **behavior** (*what the network learned*), not **structure** (*what the network is*). Two axes:
- **Axis A — Structure / topology:** "what is this network, how do the pieces connect?" → graph/module diagrams.
- **Axis B — Behavior / features:** "what did it learn?" → activations, filters, saliency. ← **Day 19 lives here.**

**Q. Does Day 19 include code to visualize / plot a neural network?**
**A.** It plots **behavior** (first-layer filters, per-block feature maps, a simplified CAM, activation histograms, dead-ReLU counts) but has **zero network-topology plotting**. Also, Day 19's `simple_cam` is a *simplified* map: it does `last_features.mean(dim=0)` and **never uses `target_class`** — so it's a class-agnostic average-activation map, **not** a true CAM. (Exercise 4 asks the learner to implement proper Grad-CAM.)

### Round 2 — Tool selection, clean install, and "paper-publication views"

**Q. Which tools should we add? You listed Visual Torch, Netron, Torch View, PyTorchViz, PlotNeuralNets, …**
**A.** Split by what they actually do:
| Tool | Axis | Notes |
|---|---|---|
| `torchinfo` | structure | text table of shapes + params (pure-pip) |
| **VisualTorch** | structure | flow / graph / **lenet** styles; **no system deps** |
| **Netron** | structure | interactive viewer for *saved model files* (ONNX/…) |
| TensorBoard `add_graph` | structure | already installed |
| `torchviz` | structure (autograd) | needs Graphviz `dot` |
| `torchview` | structure | needs Graphviz `dot` |
| `hiddenlayer` | structure | **dropped** — last release 2020, stale |
| PlotNeuralNets | structure | **LaTeX/TikZ, not pip** — appendix only |

**Q. I want a clean install that doesn't touch system Python and is encapsulated under the project. Verify that.**
**A.** Verified: `.venv` is **fully isolated** (`include-system-site-packages = false`), project-local, built from pyenv 3.11.8; no `PIP_TARGET`/`PYTHONPATH`/`pip.conf` overrides; system Python (`/usr/bin/python3` 3.9.6) untouched; `.venv/` is git-ignored. So `.venv/bin/pip install …` is **100% encapsulated**. Homebrew has **no brew Python** (won't shadow the venv). The **only** thing that would live outside the project is the Graphviz `dot` binary — and **VisualTorch does not need it** (its deps are just `pillow, numpy, aggdraw, torch`), so the paper-quality figures come with a pure-pip install.

**Q. I want "paper-publication-like" views.**
**A.** **VisualTorch's `lenet` style** gives the classic LeNet-paper figure look (3D slabs + funnel connectors) with **zero system dependencies**; `flow` is compact, `graph` shows layer widths. PlotNeuralNets gives the same aesthetic but needs a LaTeX install, so we keep it as a reference appendix rather than a dependency.

**Decision.** **Tier 1 only** — pure pip (`torchinfo`, `visualtorch`, `netron`, `onnx`), no Homebrew. `torchviz`/`torchview` cells are included but **guarded** so they skip gracefully without Graphviz.

---

## 1. The two axes (the mental model)

```
                       "look inside a CNN"
                ┌──────────────┴───────────────┐
        BEHAVIOR (Day 19)              STRUCTURE (Day 19b)
   what did it LEARN?               what IS it / how connected?
   filters, feature maps,           torchinfo, VisualTorch,
   CAM, activation stats,           Netron, TensorBoard,
   dead ReLUs                       torchviz, torchview
```

Day 19 = the **microscope** (activations on real data). Day 19b = the **blueprint** (the architecture itself). They are complementary, not redundant.

---

## 2. What the notebook covers (§ by §)

| § | Content | Tool | System dep? |
|---|---|---|---|
| Setup | reuse `SimpleCNN` (Day 17) + `SimpleResNet` (Day 18); no training | — | none |
| 1 | Layer-by-layer shapes + param counts | `torchinfo.summary` | none |
| 2 | Publication figures: `lenet`, `flow`, `graph` for `SimpleCNN`, `SimpleResNet`, `resnet18` | VisualTorch | none |
| 3 | Autograd execution graph *(guarded)* | `torchviz.make_dot` | Graphviz |
| 4 | Nested module graph with shapes *(guarded)* | `torchview.draw_graph(device="meta")` | Graphviz |
| 5 | Export to ONNX → open in interactive viewer | `torch.onnx.export` + Netron | none |
| 6 | Graph in the TensorBoard web UI | `SummaryWriter.add_graph` | none |
| 7 | Appendix: bespoke LaTeX figures | PlotNeuralNets (TikZ) | LaTeX |

---

## 3. Verified outputs (headless execution)

The notebook was **executed end-to-end with `jupyter nbconvert --execute`** and passed every cell. Generated into `assets/` (git-ignored):

- `simplecnn_lenet.png`, `simplecnn_flow.png`, `simplecnn_graph.png`
- `simpleresnet_flow.png` (skip connections visible)
- `resnet18_flow.png`
- `simplecnn.onnx` (for Netron)
- `tb/` (TensorBoard event file with the graph)

**Verified facts:** `SimpleCNN` = **620,810** params; `torchinfo` shape trace `32→16→8→4` with channels `3→32→64→128`; ONNX export produces a 13-node graph (opset 17); `torchviz`/`torchview` report **unavailable** without Graphviz and their cells skip cleanly.

---

## 4. Tool comparison (which, when)

| Tool | Input | Output | System dep? | Best for |
|---|---|---|---|---|
| `torchinfo` | live model | text table (shapes, params) | none | exact layer facts, param counts |
| **VisualTorch** | live model | PIL image (flow / graph / lenet) | none | **paper / talk figures**, skip connections |
| **Netron** | model **file** (ONNX/…) | interactive UI | none | inspecting saved/exported models, weights, shapes |
| TensorBoard `add_graph` | live model | web-UI graph | none | alongside your training curves |
| `torchviz` | live model + forward | Graphviz diagram | **Graphviz** | the autograd graph (small nets) |
| `torchview` | live model | Graphviz diagram + shapes | **Graphviz** | nested module hierarchy with shapes |
| PlotNeuralNets | hand-written LaTeX | PDF / TikZ | **LaTeX** | bespoke publication figures |

**Rule of thumb:** `torchinfo` (facts) → **VisualTorch** (the figure) → **Netron** (when you have a saved file). Add `torchviz`/`torchview` only if you want the autograd/module graph and can install Graphviz.

---

## 5. How to read each diagram

- **`torchinfo` table** — read the **shape column top-to-bottom**: spatial dims shrink (`32→16→8→4`), channels grow (`3→32→64→128`). The `num_params` column shows where the parameters *live* (almost all in the first `Linear`, not the convs).
- **`lenet` / `flow` figure** — each box is a feature-map tensor; its **width/height is proportional to channels**, its **depth to spatial size**. Funnels = pooling (spatial down-sampling).
- **`graph` figure** — nodes are neurons; edges are connections; dense vs sparse layers are visually obvious.
- **`torchviz`** — each node is a **tensor op**; edges show data flow; `params=...` annotates weight tensors. A full CNN is a hairball — that clutter is the lesson (use it on small graphs).
- **`torchview`** — a *module*-level graph (your `block1/…`), annotated with shapes, **showing connections** — the one place a skip connection appears as an explicit extra edge.
- **Netron** — click any node to see its attributes, weights and input/output shapes; the `Add` nodes in a ResNet are the skip connections.

---

## 6. Optics connections

- **VisualTorch `lenet` slabs** ↔ an optical **lens train / relay system**: each box is a "surface" (a tensor), and the **funnels** are the stops that shrink the spatial extent (`MaxPool` = down-sampling the field).
- **`torchinfo`'s shape trace** ↔ **ray transfer**: the spatial footprint shrinks (`32→16→8→4`) while the "information width" (channels) grows (`3→32→64→128`) — resolution traded for feature richness, exactly the pattern you met on Day 17.
- **`torchviz` autograd graph** ↔ a **system ray/aberration diagram**: every element that touches the beam (tensor op) is drawn, and the backward pass is the return path.
- **Netron's per-node shapes** ↔ an **optical metrology readout**: click any element to inspect its exact dimensions.

---

## 7. Cross-arc threads

1. **Autograd-graph thread:** Day 5 (computation graph, from scratch) → **Day 19b** (`torchviz` draws it).
2. **Hierarchy thread:** Day 16 (feature maps) → 17 (resolution↓/channels↑) → 18 (staged ResNet) → 19 (hierarchy *visualized* — behavior) → **19b (hierarchy *drawn* — structure)** → 20 (hierarchy *reused*).
3. **Skip-connection thread:** 17b (foreshadow) → 18 (`F(x)+x`) → **19b (skip shows up as an explicit graph edge in VisualTorch / torchview / Netron's `Add`)** → 21_extra (U-Net) → Days 23+ (Transformer residual stream).
4. **Reuse-vs-build thread:** 19b renders `resnet18` as a *preview* of what Day 20 will *reuse*.

---

## 8. Self-check

1. What is the difference between the two axes (behavior vs structure), and which day covers which?
2. Why can't a plain `nn.Sequential` diagram ever show a ResNet skip connection? Which tools can?
3. Which tools need the system Graphviz binary, and which are pure-pip?
4. In `torchinfo`'s table for `SimpleCNN`, why is nearly all the parameter count in the first `Linear`, not the conv layers?
5. In Netron, which nodes correspond to `out = out + identity`?
6. Why is Day 19's `simple_cam` not a "true" CAM? (Hint: is `target_class` used?)

---

## 9. Install & run reference

```bash
# Tier 1 — pure pip, fully encapsulated in .venv (no Homebrew)
.venv/bin/pip install -r requirements-viz.txt        # torchinfo, visualtorch, netron, onnx

# Optional — Graphviz-based tools (torchviz / torchview)
brew install graphviz
.venv/bin/pip install graphviz torchviz torchview

# Run the notebook
.venv/bin/jupyter notebook day19b_network_visualization/day19b_network_visualization.ipynb
```

**Files added/changed in this session:**
- **Added:** `day19b_network_visualization/day19b_network_visualization.ipynb`, `day19b_network_visualization/assets/` (git-ignored), `requirements-viz.txt`, `LEARNING_NOTES_DAY19B.md`
- **Updated:** `.gitignore` (`assets/`, `runs/`, `*.onnx`), `CURRICULUM.md` (Phase-4 row 19b + folder tree + packages note), `AI_CONTEXT.md` (roadmap, progress, arc ordering, update log)

---

## 🔭 Where this goes next

- **Day 19** — the behavior companion (filters, feature maps, CAM, dead ReLUs); optionally upgrade its `simple_cam` to a real Grad-CAM.
- **Day 20** — transfer learning: render `resnet18` to see what gets *reused*.
- **Day 21** — capstone: build a CNN from scratch to >90% on CIFAR-10.
- **Days 25–26** — VisualTorch and the visualization habits return for the ViT (attention maps).
