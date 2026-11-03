# VLA Models: Literature Review and Demonstration

Literature review and hands-on demonstration of **Vision-Language-Action (VLA) models**,
prepared for a Master's-level Machine Learning and Deep Learning course (Module 4,
"Own Topic" presentation). Covers two open-source VLA systems in depth — **Octo** and
**OpenVLA** — including their architectures, reported results, and a working demo that
lets anyone run both models on a free Colab GPU.

## What is a VLA model?

A Vision-Language-Action model is a single neural network that takes a camera image and
a natural-language instruction as input and outputs a robot action directly — replacing
the classic separate perception → planning → control pipeline with one end-to-end
forward pass.

## Models at a glance

| | OpenVLA | Octo |
|---|---|---|
| Parameters | 7B | 27M (Small) / 93M (Base) |
| Backbone | Llama-2, fused with DINOv2/SigLIP visual encoders | Transformer (ViT-S/ViT-B sized) with a diffusion action head |
| Action output | Discretized action tokens, decoded autoregressively | Continuous actions, predicted as a chunk via diffusion |
| Pretraining data | 970k Open X-Embodiment trajectories | ~800k-episode curated subset of 25 Open X-Embodiment datasets |
| Inference hardware | GPU (reported: 15GB VRAM, ~6Hz on a single RTX 4090) | Runs on CPU — checkpoints are small enough to be fast without a GPU |

## Papers covered

- **OpenVLA: An Open-Source Vision-Language-Action Model**
  Kim, Pertsch, et al., 2024. [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) ·
  [project page](https://openvla.github.io) ·
  [code](https://github.com/openvla/openvla)

- **Octo: An Open-Source Generalist Robot Policy**
  Octo Model Team, 2024. [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) ·
  [project page](https://octo-models.github.io) ·
  [code](https://github.com/octo-models/octo)

Paper PDFs are not redistributed in this repo — follow the links above for the original
sources.

## Repository structure

| Path | Contents |
|---|---|
| `Octo_Demo.ipynb` | Colab notebook: loads Octo-Small and Octo-Base (JAX), runs inference on an image + instruction, visualizes the predicted action. |
| `OpenVLA_Demo.ipynb` | Colab notebook: loads `openvla/openvla-7b` (PyTorch), same inference + visualization flow. Kept in a separate notebook/runtime from Octo so the two frameworks don't compete for GPU memory. |
| `test_images/` | Fixed set of real [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) frames used as a controlled comparison set across both models, plus extra sample images for anyone trying the notebooks without a photo of their own. See `test_images/README.md`. |

The presentation notes, slide deck, formal report, and working task log are kept outside
this repository and are not shared here.

## Running the demos

Both notebooks are self-contained — open either one in Google Colab, run the cells top
to bottom, and use the "try it yourself" cell to upload your own photo and type your own
instruction. No local setup is required; each notebook installs its own dependencies in
the Colab runtime.

- `OpenVLA_Demo.ipynb` needs an older Colab runtime image (Python ≤ 3.12) and a T4 GPU —
  see the note in the notebook's first cell.
- `Octo_Demo.ipynb` runs on CPU by design (the checkpoints are small enough that this is
  fast) and works on any runtime.

The pre-loaded comparison cells (`TEST_PAIRS`) read images from each presenter's own
Google Drive cache and are mainly for reproducing the slide-deck results; the
upload-your-own-photo cell has no such dependency.

## What the demo tests

Both notebooks run the same four fixed (image, instruction) pairs from the public
[BridgeData V2](https://rail-berkeley.github.io/bridgedata/) dataset through all three
pretrained checkpoints — OpenVLA, Octo-Small, Octo-Base — with no fine-tuning:

| Pair | Instruction | Scene | What it tests |
|---|---|---|---|
| 1 | "pick up the spatula" | Clean | Baseline: one clear, unambiguous target |
| 2 | "pick up the spoon, not the pot" | Pot, spoon, cans | Distractor / language grounding |
| 3 | "pick up the fruit" | Fruit- and animal-shaped toys | Semantic generalization (category, not literal name) |
| 4 | "pick up the spatula" | Same instruction as pair 1, cluttered scene | Scene robustness: Octo-Small vs Octo-Base |

Each model outputs a single-step, 7-DoF control delta (x, y, z, yaw, pitch, roll, grasp)
per image — not an object label — since both models are designed as closed-loop
controllers re-evaluated on every new camera frame, not single-shot classifiers.

## Understanding the output: the 7-DoF action vector

Every prediction is the same 7-number vector, in this fixed order, representing a
**change** for the next control step — not an absolute position and not an object
location:

| # | Dim | Meaning | Units |
|---|---|---|---|
| 1 | `x` | End-effector movement, left(−)/right(+) | meters |
| 2 | `y` | End-effector movement, back(−)/forward(+) | meters |
| 3 | `z` | End-effector movement, down(−)/up(+) | meters |
| 4 | `yaw` | Wrist rotation about the vertical axis | radians |
| 5 | `pitch` | Wrist rotation, tilting up/down | radians |
| 6 | `roll` | Wrist rotation, twisting about the gripper's own axis | radians |
| 7 | `grasp` | Gripper command (≈0 = closed, ≈1 = open) | unitless |

All values are tiny (millimeter/sub-radian scale) because each one is a single step of
a loop meant to run continuously at a few Hz, not a full motion to the target in one
shot — the model is re-queried on a fresh camera frame after every step. Only `x`/`y`
are drawn as the red arrow in the visualization (they're the two axes that lie in the
image plane); `z`, `yaw`, `pitch`, `roll`, and `grasp` are only shown in the printed
values below each image.

### Our results

Four fixed (image, instruction) pairs (see above), predicted first-step action for
each model — pretrained checkpoints only, no fine-tuning:

| Pair | Instruction | OpenVLA | Octo-Small | Octo-Base |
|---|---|---|---|---|
| 1 | "pick up the spatula" | x=0.004, y=−0.005, z=0.008, grasp=0.996 | x=0.004, y=−0.010, z=0.018, grasp=0.995 | x=0.004, y=−0.001, z=0.004, grasp=0.992 |
| 2 | "pick up the spoon, not the pot" | x=−0.0, y=−0.0, z=−0.001, grasp=0.996 | x=0.005, y=−0.022, z=0.018, grasp=0.998 | x=0.005, y=−0.021, z=0.016, grasp=0.993 |
| 3 | "pick up the fruit" | x=0.0, y=−0.002, z=0.0, grasp=0.996 | x=0.004, y=−0.000, z=0.009, grasp=0.996 | x=0.003, y=−0.002, z=0.003, grasp=0.990 |
| 4 | "pick up the spatula" (cluttered) | x=0.0, y=0.0, z=−0.003, grasp=0.996 | x=0.001, y=−0.000, z=−0.000, grasp=0.995 | x=0.002, y=−0.001, z=0.002, grasp=1.004 |

(`yaw`/`pitch`/`roll` are recorded in each notebook's own output cells; omitted here
for brevity.) What these confirm: predictions are reproducible, non-degenerate, and
genuinely different per instruction/image/model — evidence the pipeline runs a fresh
forward pass each time rather than returning a cached result. What they do **not**
confirm: whether the model identified the correct object — a single still-frame
prediction is only the first step of a closed loop, and that question needs an actual
multi-step rollout (real or simulated robot), which is outside the scope of this demo.
