# VLA Models: Literature Review and Demonstration

Literature review and hands-on demonstration of **Vision-Language-Action (VLA) models**,
prepared for a Master's-level Machine Learning and Deep Learning course (Module 4,
"Own Topic" presentation). Covers two open-source VLA systems in depth — **Octo** and
**OpenVLA** — including their architectures, reported results, and a working demo that
lets anyone run both models on a free Colab GPU.

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
