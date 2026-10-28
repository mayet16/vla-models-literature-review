# VLA Presentation Tasks — OpenVLA vs Octo

Module 4 (6th session) presentation. Target date: 04-05 Nov 2026 — about 9 days from today (26 Oct 2026).

## Confirmed from the actual course brief (`Deep_learning_course_Material.pdf`, Ch.6) — not assumptions

- **Hard cap: 35 minutes total**, not 40 (15 min subject talk + 20 min practical work; instructor
  stops you at minute 35, which lowers the grade). The schedule table's "40 min" is just the
  room slot, not the content budget.
- Groups of 4: two present the subject, two present the practical work.
- **Critical grading requirement: the practical activity must let the other students
  themselves experiment with the method** — not just watch a demo. This changes the hands-on
  design: it needs to be something ~15+ classmates can run on their own laptops in the 20-minute
  slot (e.g. a shared Colab notebook they open and tweak themselves), not a single
  robot-arm/Gazebo rig only we control. **This significantly deprioritizes Section 5 below** —
  revisit whether it's worth doing at all given this constraint.
- KISS principle, English only — explain clearly, don't over-jargon or speed-run content; the
  rest of the class shares the same master's background.
- No fine-tuning — confirmed by the user: use both models' pretrained checkpoints directly.

## What the papers' own authors actually did (real robots, not simulation)

- **OpenVLA**: real-robot "out-of-the-box" evaluation on two physical platforms — the WidowX
  arm (BridgeData V2, 170 rollouts: 17 tasks x 10 trials) and a mobile manipulator ("Google
  robot", 60 rollouts: 12 tasks x 5 trials), compared against RT-1-X, RT-2-X, and Octo. Also
  fine-tuning experiments on new robot setups. **Inference footprint per the paper itself: 15GB
  VRAM in bf16, ~6Hz on a single consumer RTX 4090** — note this is better than the blog implied
  (it claimed A100/H100-class hardware is needed; the paper's own number is a consumer GPU).
  They also ship a remote inference server so the robot-side computer doesn't need to be
  powerful at all — inference can run elsewhere and stream actions to the robot.
- **Octo**: real-robot zero-shot evaluation across **9 robot setups at 4 institutions**
  (WidowX BridgeV2, UR5 Tabletop, RT-1 Robot, Berkeley Insertion, Stanford Coffee, CMU Baking,
  Berkeley Pick-Up, Berkeley Coke, Berkeley Bimanual), compared to RT-1-X and RT-2-X zero-shot.
  Separately fine-tuned on 6 of those setups with ~100 demonstrations each, taking <5 hours on
  a single NVIDIA A5000 (24GB) GPU.
- **Neither paper demos in Gazebo or any simulator at all** — both are 100% real physical robot
  evaluations. Confirms Section 5 below is genuinely outside what either paper itself does.

## 0. Open decisions / things to confirm first

- [ ] Laptop has a discrete NVIDIA GPU (PCI id 2d59, Raptor Lake-S platform) but no driver
      currently loaded (`nvidia-smi` fails). Decide: fix the driver (separate, non-course task,
      do only if time allows) or just plan around Colab/lab computer instead.
- [ ] Check what GPU the lab computers actually have (ask TA/professor or check on-site) —
      only matters if Colab free tier turns out insufficient for OpenVLA.
- [ ] Confirm Colab access level (free T4 16GB vs Colab Pro) before relying on it for OpenVLA.

## 1. Research phase — primary sources, not just the blog

- [ ] Re-read `OpenVLA.pdf` in full: architecture (Llama-2 7B + DINOv2/SigLIP), training data
      (970k Open X-Embodiment trajectories), benchmark tables (LIBERO-Spatial/Object/Goal/Long,
      BridgeData V2), exact fine-tuning setup used for each reported number (LoRA rank, steps).
- [ ] Re-read `Octo.pdf` in full: transformer backbone + diffusion action head, ~800k curated
      OXE subset, its own benchmark numbers and eval protocol (zero-shot vs fine-tuned).
- [ ] Write down each paper's own numbers next to their exact eval condition. Do NOT reuse the
      roboticscenter.ai blog's comparison table as-is in the presentation — it compares a
      LoRA-fine-tuned OpenVLA against a zero-shot Octo-Base, which is not apples-to-apples.
      Turning that mismatch into a "how to read VLA literature critically" slide is good
      presentation material, not just a thing to avoid.
- [ ] Fold anything new/corrected into `VLA-Topic-Notes.tex` if it changes the existing notes.

## 2. Fine-tune vs. use pretrained — decision

- [x] **Decided: no fine-tuning.** Use both models' released pretrained checkpoints directly
      from Hugging Face, as-is.
- [x] **Verified both are HF-hosted:**
      - OpenVLA: `openvla/openvla-7b`, loadable via `transformers.AutoModelForVision2Seq`.
      - Octo: `rail-berkeley/octo-small` (27M) and `rail-berkeley/octo-base` (93M), MIT
        license. Weights live on the HF Hub, but loading goes through Octo's own package, not
        generic `transformers`:
        ```python
        from octo.model.octo_model import OctoModel
        model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
        ```
        So `pip install` their repo (already planned in Section 3) rather than relying on
        `transformers` alone for this one.

## 3. Environment setup

**Decided: demo all three — Octo-Small, Octo-Base, and OpenVLA — side by side.**

### Octo-Small and Octo-Base (try these first — lightest, most likely to "just work")
- [ ] Clone the Octo repo, install JAX + dependencies.
- [ ] Run their provided inference demo with **Octo-Small** (27M, `hf://rail-berkeley/octo-small`)
      first — most likely to run on a laptop CPU/GPU or free Colab.
- [ ] Then run **Octo-Base** (93M, `hf://rail-berkeley/octo-base`) the same way, same test
      inputs, for a direct small-vs-base comparison in the talk (speed, and whether action
      predictions visibly differ/improve).

### OpenVLA (heavier — plan for Colab, not the laptop)
- [ ] Start in Colab (free T4, 16GB) using the HF `transformers` reference inference snippet
      (`openvla/openvla-7b`, `AutoModelForVision2Seq`).
- [ ] If VRAM is too tight in bf16 (15GB per the paper, right at the T4's 16GB ceiling), switch
      to OpenVLA's quantized (4-bit/8-bit via bitsandbytes) checkpoint before giving up on
      free-tier Colab.
- [ ] Only fall back to a lab computer or Colab Pro (A100) if the quantized version still
      doesn't fit.

### Running all three together
- [ ] Don't assume all three fit in VRAM simultaneously on a free T4 — Octo-Small/Base are tiny
      (27M/93M) but OpenVLA alone needs ~15GB. Plan to load/run one model at a time in the demo
      notebook (load -> run test inputs -> free the model/clear CUDA cache -> load the next)
      rather than keeping all three resident at once, unless testing shows otherwise.

## 4. Core demo — must satisfy "classmates experiment themselves"

**Built: two notebooks, split by framework (not one combined notebook)** —
`Octo_Demo.ipynb` (Octo-Small + Octo-Base, JAX) and `OpenVLA_Demo.ipynb` (OpenVLA, PyTorch),
both in this folder. Reasoning: JAX preallocates GPU memory aggressively and OpenVLA alone
needs ~15GB on a 16GB T4, so forcing JAX and PyTorch to share one Colab runtime risked VRAM
conflicts for no real benefit — Octo-Small/Base are tiny enough to coexist with each other,
OpenVLA gets its own clean runtime instead.

Both notebooks:
- Mount Google Drive and point `HF_HOME` at it, so the multi-GB checkpoints are downloaded once
  and reused across sessions instead of every Colab runtime restart (using the 400GB student
  trial storage).
- Define a `run_<model>(image, instruction)` helper + `visualize_action(...)` (arrow overlay for
  predicted x/y motion + printed 7-DoF vector).
- Have a "try it yourself" cell using Colab's file-upload widget — classmates upload their own
  photo and type their own instruction, no external image URL dependency to go stale.
- Have a placeholder `TEST_PAIRS` loop for a fixed comparison set to use in the slide deck.

### Test pairs — chosen, each probing a specific claim from the papers

**Switched from staged personal photos to real public BridgeData V2 images** — these are
downloadable directly (`rail.eecs.berkeley.edu` hosts the raw dataset with open directory
listing, same source Octo's own official example notebook uses), and being genuinely
in-distribution for what both models were pretrained on means a much higher chance of clean,
illustrative predictions during a live demo than an arbitrary personal photo would give. All
4 are already downloaded into `test_images/` (same toy-kitchen environment, so they're visually
consistent with each other too).

| # | Filename | Source (BridgeData V2, toykitchen7/many_skills) | Instruction | What it tests |
|---|----------|----|-------------|----------------|
| 1 | `test_images/pair1_cup.jpg` | session 25 — spatula + towel + cans | "pick up the spatula" | Baseline sanity check — one clear, unambiguous target object. If nothing else works, this one should. |
| 2 | `test_images/pair2_fork_cup.jpg` | session 10 — pot + spoon + cans | "pick up the spoon, not the pot" | Language grounding / distractor test — OpenVLA's paper reports it handles multi-object distractor scenes better than Octo, which "often fails to manipulate the correct object" with distractors present. |
| 3 | `test_images/pair3_fruit.jpg` | session 05 — small pineapple-shaped toy + frog-shaped toy + pot + cans | "pick up the fruit" | Semantic/novel-instruction test, RT-2-style (their own demo: "pick up the extinct animal" -> dinosaur toy). Requires recognizing the pineapple-shaped toy as "fruit" by category, not literal name-matching — tests internet-pretrained semantic transfer, where OpenVLA should have the edge over Octo (no internet-scale VLM pretraining). |
| 4 | `test_images/pair4_cluttered.jpg` | session 20 — 2 sushi-shaped toys + cup + spatula + cans | "pick up the spatula" | Scene-robustness test, specifically Octo-Small vs Octo-Base — busiest of the 4 scenes, tries to reproduce the Octo paper's own ablation finding that Base is "more robust to initial scene configuration" and "less prone to early grasp attempts" than Small. |

- [x] **Done — downloaded, no photography needed.** All 4 images already sit in `test_images/`.
- [x] **Done — user uploaded the whole `VLA-Presentation` folder (including `test_images/`) to
      Google Drive.** Double-check the path once in Colab: the notebooks expect it at exactly
      `/content/drive/MyDrive/VLA_Demo_Cache/test_images/pair*.jpg` — if the upload landed the
      folder somewhere else in Drive (e.g. `MyDrive/VLA-Presentation/test_images/` directly,
      without the `VLA_Demo_Cache` wrapper), either move/rename it to match, or edit the
      `TEST_PAIRS` paths (and the `HF_HOME` cache path) in both notebooks to the real location.
- [x] `TEST_PAIRS` in both `Octo_Demo.ipynb` and `OpenVLA_Demo.ipynb` pre-filled with these
      filenames + instructions.
- Note for the talk: since these are in-distribution BridgeData images (not our own objects),
  the "classmates experiment themselves" moment is still fully live via the upload-your-own-photo
  cell (Section 5 of each notebook) — these 4 are the controlled, pre-verified comparison set for
  the slide deck, not a replacement for that.
- [x] **`OpenVLA_Demo.ipynb` loads and runs end-to-end on Colab**, but hit a real,
      well-diagnosed environment-compatibility issue along the way. Chasing newer-`transformers`
      compatibility one error at a time (unpinning `tokenizers`, capping `transformers<5.0`,
      forcing `attn_implementation="eager"`, pinning `timm==0.9.16`) got the model to load and
      *run*, but produced a **silent** bug: every image/instruction combination returned the
      exact same predicted action. Traced it step by step (confirmed prompts/pixel_values
      genuinely differ going in; ruled out a bf16-corrupting-int64-ids theory; ruled out an
      EOS-triggered pad-fill theory) down to: the model's raw generated token IDs were
      identical on a real forward pass, not a crash or an input bug - a genuine regression
      somewhere in the ~3-year gap between transformers 4.40.1 (what OpenVLA was validated
      against) and 4.57.6 (current Colab).
      **Tried reverting to the exact validated versions** (`transformers==4.40.1`,
      `tokenizers==0.19.1`, `timm==0.9.10`) via installing a Rust compiler so the old
      `tokenizers` builds from source - this path was a dead end **specifically on Colab's
      "Latest" runtime image**: `tokenizers==0.19.1`'s Rust bindings (`pyo3-ffi 0.21.2`)
      hard-cap their supported Python at 3.12, and "Latest" runs Python 3.13. Confirmed via
      verbose build logs; a PyO3 forward-compatibility env var
      (`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`) that sometimes bypasses this kind of gate
      also failed. No community-documented fix for this exact issue was found either.
      `OpenVLA_Demo.ipynb` was rewritten once, consolidated into one consistent state: back
      on the working `transformers<5.0`/`timm==0.9.16`/eager-attention combo, with
      `USE_4BIT=True` set as a best-effort workaround attempt (different code path, might
      sidestep the regression), and the full diagnosis documented inline in the notebook
      itself (near the `TEST_PAIRS` cell) so it's not just lost in chat history.
- [x] **Confirmed dead end at the time — ran that notebook with `USE_4BIT=True` on Colab's
      "Latest" runtime (2026-10-27).** Model loaded fine (4-bit, ~12min shard load, harmless
      warnings only), but all 4 `TEST_PAIRS` still printed the exact same action vector
      regardless of image/instruction — `All 4 results identical? True`. 4-bit quantization
      goes through a different load path but hit the same regression, so it isn't a
      dtype/precision artifact specifically.
      **Decision made at the time: stop debugging OpenVLA, Octo becomes the sole live demo,
      OpenVLA becomes diagnosed-limitation slide content.** (Superseded below - keeping this
      for the historical record since it's exactly the kind of "properly diagnosed
      obstacle" content worth presenting either way.)
- [x] **Breakthrough (2026-10-27, same day): Colab's Runtime settings have a separate
      "Runtime version" dropdown** (distinct from hardware type) offering older dated OS
      images (2025.07, 2025.10, 2026.01, 2026.04, 2026.07, as well as "Latest") - these
      bundle an older Python, sidestepping the Python-3.12 cap entirely and letting the
      exact validated versions install for real, directly on Colab (so still shareable with
      classmates via a link, unlike a local conda environment - a parallel attempt to test
      this on the local `cv_robotics_env` conda env, Python 3.10, was abandoned mid-download
      at the user's request once this Colab-native option was found; no local files or
      downloaded weights were kept). `OpenVLA_Demo.ipynb` rewritten accordingly: top
      markdown instructs picking an older runtime version + T4 GPU before running anything,
      a new cell checks `python --version` first and warns if still on 3.13, install/load
      cells reverted to the plain original pins (`transformers==4.40.1`,
      `tokenizers==0.19.1`, `timm==0.9.10`) with `USE_4BIT=False` (no version-drift
      workarounds needed if the Python version is right).
- [x] **Tested on Colab - genuinely promising, model now loads correctly end-to-end.**
      User picked runtime version `2025.07` -> actual Python landed on **3.11.13** (close
      to the validated 3.10, comfortably under the 3.12 cap). The exact original pins
      installed cleanly this time - no Rust/PyO3 build failure at all, confirming the
      Python-3.13-specific wall was the whole story for that blocker. Two more issues hit
      after install, both resolved:
      - **GPU quota exhausted this session** - despite selecting T4, Colab silently gave a
        CPU-only runtime, crashing on `.to("cuda:0")`. Fixed by adding device
        auto-detection (`torch.cuda.is_available()`) to the load cell with a clear warning
        and CPU fallback instead of a hard crash.
      - **Stale Python import cache** - after fixing the install, the `timm` whitelist
        check still failed, reporting version `1.0.17` (Colab's base-image default) even
        though `pip show timm` confirmed `0.9.10` was correctly on disk. Root cause: the
        kernel had already imported `timm` once earlier in the same session (during a prior
        attempt), and Python's module cache doesn't re-read files off disk just because
        `pip install` changed them. Fixed with **Runtime > Restart session** (kernel-only
        restart, keeps installed packages - not the full "disconnect and delete") before
        re-running the load cell.
      - **Result: model loaded successfully end-to-end** - all 3 safetensors shards
        downloaded and loaded (6.95G + 6.97G + 1.16G ≈ 15GB total, matching the paper's own
        bf16 footprint exactly), no version-related errors of any kind.
- [x] **Two more version-drift bugs hit during the first real `TEST_PAIRS` run, both
      diagnosed and patched (2026-10-27):**
      - **Torch/torchvision gap.** Colab's `2025.07` runtime image ships `torch==2.6.0+cu124`
        by default (never pinned by us), 4 minor versions ahead of OpenVLA's validated
        `2.2.0`. Crashed mid-generation inside Llama's self-attention:
        `RuntimeError: The size of tensor a (277) must match the size of tensor b (276) at
        non-singleton dimension 3` — a KV-cache/attention-mask shape mismatch from PyTorch's
        internal attention-handling changes across that version range. Fixed by pinning
        `torch==2.2.0`/`torchvision==0.17.0` (matched pair, to avoid a separate ABI mismatch)
        in the install cell.
      - **NumPy ABI mismatch, surfaced immediately after the torch fix.** `torch==2.2.0`
        wheels are compiled against NumPy 1.x's C API, but Colab's runtime still had NumPy
        2.0.2 installed (never pinned). Crashed as `RuntimeError: Numpy is not available`
        inside OpenVLA's own `processing_prismatic.py` (its image preprocessing calls
        `.numpy()` on a tensor, which needs torch's NumPy integration actually initialized).
        Fixed by adding `"numpy<2"` to the same install cell.
      Both fixes applied directly to `OpenVLA_Demo.ipynb`'s install cell, with the rationale
      documented in the preceding markdown cell.
      - **Third bug, caused by the numpy fix itself.** Forcing `numpy<2` system-wide fixed
        torch's own numpy bridge, but broke Colab's *other* preinstalled numpy2-only
        packages: `transformers` auto-imports TensorFlow when present
        (`image_transforms.py`'s `is_tf_available()` check), dragging in
        `keras -> pandas -> numpy.random`, and pandas's compiled extensions (built against
        numpy 2.x) crashed with `ValueError: numpy.dtype size changed, may indicate binary
        incompatibility. Expected 96 from C header, got 88 from PyObject`. OpenVLA is
        PyTorch-only and never needs TF/Flax, so fixed by setting `os.environ["USE_TF"] =
        "0"` / `os.environ["USE_FLAX"] = "0"` *before* importing `transformers` in the load
        cell, instead of trying to also make Colab's whole TensorFlow/Keras/pandas/opencv
        stack numpy-1.x-compatible. (Harmless pip resolver warnings about
        `opencv-python-headless`/`thinc`/`sentence-transformers`/`torchaudio` wanting a
        different numpy/torch/transformers remain — none of those packages are used in
        this notebook, same as the other ignorable resolver warnings seen throughout.)
      - **Fourth bug: the `USE_TF`/`USE_FLAX` fix above wasn't the real root cause.** Same
        `ValueError: numpy.dtype size changed, may indicate binary incompatibility.
        Expected 96 from C header, got 88 from PyObject` recurred, this time via a
        completely different import path (`torchvision -> torch._dynamo ->
        numpy.random`), confirming it's not about *which* library imports numpy — numpy's
        own install is internally inconsistent. A plain in-place `pip install "numpy<2"`
        alongside everything else in one command left numpy's own compiled extensions
        (`numpy/random/mtrand`, a Cython module compiled against numpy's own C API) from
        the old `2.0.2` mixed with the new `1.26.4`'s Python-level files on disk, rather
        than a clean full replacement. Fixed by splitting the install into two pip calls:
        the main one (unchanged), then a second, separate
        `pip install --force-reinstall --no-deps --no-cache-dir "numpy==1.26.4"` that
        forces every file belonging to numpy - compiled and Python - to come from the same
        freshly-downloaded wheel, no cache reuse, no partial leftover files.
- [x] **All four version-drift bugs resolved — model now loads cleanly end-to-end on Colab
      (2026-10-27)**, with the two-step pip install (main install + forced clean numpy
      reinstall) and `USE_TF`/`USE_FLAX` disabling both in place. All 3 safetensors shards
      downloaded and loaded (6.95G + 6.97G + 1.16G), `generation_config.json` loaded, no
      errors or warnings beyond the expected harmless ones (HF_TOKEN notice,
      `resume_download` deprecation, CPU-fallback warning since GPU quota was exhausted
      again this session).
- [x] **First real `TEST_PAIRS` run — hit the attention-mask shape crash again
      (`tensor a (277)` vs `tensor b (276)`), proving the earlier torch-version diagnosis
      was wrong.** The exact same crash recurred with `torch==2.2.0` correctly pinned and
      loaded cleanly, ruling out torch version as the cause. Found a matching HuggingFace
      forum report of an identical off-by-one causal-mask bug: a `transformers` internal
      issue present around `4.38`-`4.40` (where the KV-cache length and attention-mask
      length can desync by one token during incremental decoding), fixed by `4.44`.
      OpenVLA's own custom forward only builds its multimodal attention mask (accounting
      for the ~256 prepended vision-patch tokens) on the *first* generation step and
      relies on `transformers`' own cache-position bookkeeping every step after — exactly
      the kind of internal logic that bug affects. **Fix attempt: bumped `transformers`
      from `4.40.1` to `4.44.2`** (past the broken window, but conservative enough that
      the newer identical-output regression — which only appeared on much more recent
      `transformers` like `4.57.x` — shouldn't resurface). Torch/torchvision/tokenizers/
      timm/numpy pins unchanged.
- [x] **Re-tested with `transformers==4.44.2` — exact same crash, disproving that
      diagnosis.** Same `tensor a (277)` vs `tensor b (276)` mismatch, same failing line.
      New clue: a deprecation warning about `past_key_values` being passed as a tuple
      ("removed in v4.43") — OpenVLA's own remote code manually builds/slices its KV
      cache in the old tuple format. The actual failing line,
      `attn_weights = attn_weights + causal_mask` in `modeling_llama.py`, is **eager**
      attention's own manual, naive mask-slicing logic (`attn_implementation="eager"`,
      a leftover from working around the unrelated identical-output bug on Colab's
      "Latest" runtime several debugging rounds ago, kept out of inertia since). That
      naive slicing is the likely source of the desync against OpenVLA's tuple-format
      cache. **Fix attempt: switched `attn_implementation` from `"eager"` to `"sdpa"`**
      (PyTorch's built-in fused attention, which handles mask broadcasting/slicing
      internally instead of that manual add; works fine on CPU, no flash-attn/GPU
      needed). `transformers` left at `4.44.2` (single-variable change, so the result
      cleanly attributes to `sdpa` vs `eager`).
- [x] **Fixed a separate real bug found along the way: `TEST_PAIRS` used `Image.open(...)`
      but `from PIL import Image` was only defined in the last cell (the "try it
      yourself" upload cell) — a `NameError` if cells are run in the notebook's own
      order. Fixed by moving `from google.colab import files` and `from PIL import
      Image` into the helper-functions cell, right before `TEST_PAIRS` needs them.**
- [x] **CONFIRMED FIXED (2026-10-28): `sdpa` resolved the attention-mask crash —
      `TEST_PAIRS` ran end-to-end on Colab (GPU available this run) with no errors,
      only the harmless expected `past_key_values`-as-tuple deprecation notice. All 4
      results are genuinely different per image/instruction:**
      - "pick up the spatula" (pair1): `{x: 0.004, y: -0.005, z: 0.008, yaw: -0.006,
        pitch: -0.027, roll: -0.029, grasp: 0.996}`
      - "pick up the spoon, not the pot" (pair2): `{x: 0.0, y: -0.022, z: 0.02,
        yaw: -0.013, pitch: -0.044, roll: -0.047, grasp: 0.996}`
      - "pick up the fruit" (pair3): `{x: 0.0, y: -0.002, z: 0.0, yaw: -0.012,
        pitch: 0.004, roll: 0.005, grasp: 0.996}`
      - "pick up the spatula" again, different/cluttered scene (pair4): `{x: 0.0, y: 0.0,
        z: -0.003, yaw: 0.002, pitch: 0.007, roll: 0.013, grasp: 0.996}`
      **`All 4 results identical? False`.** The identical-output bug is genuinely gone
      with the correct dependency stack (torch 2.2.0/torchvision 0.17.0, transformers
      4.44.2, tokenizers 0.19.1, timm 0.9.10, numpy 1.26.4, `attn_implementation="sdpa"`).
      **DECISION: supersedes the earlier "Octo-only" call. `OpenVLA_Demo.ipynb` is now a
      second confirmed, working, hands-on live demo alongside Octo** — both models can be
      part of the practical segment. Still to sanity-check: whether these action vectors
      are directionally sensible for each instruction/scene (useful for slide-deck
      commentary, not blocking), and running the "try it yourself" upload cell on Colab to
      confirm it still works after the import fix above.
- [x] **Resolved — see the "CONFIRMED FIXED" entry above.** `TEST_PAIRS` now produces
      genuinely different results per input (`All 4 results identical? False`). OpenVLA
      is a second confirmed live demo; Octo remains the baseline/fallback.
- [x] **`Octo_Demo.ipynb` debugged and validated — this is the working live demo.**
      Hit a long chain of real dependency-rot issues (Octo's own repo hasn't been updated
      since late 2023, same pattern as OpenVLA but more of it, since Octo pulls in JAX/Flax/TF
      instead of just `transformers`):
      1. `requirements.txt` hard-pins exact 2023 versions (`numpy==1.24.3`, `jax==0.4.20`,
         `flax==0.7.5`, `tensorflow==2.15.0`, ...) — none buildable on current Colab's Python
         3.13. Fixed by installing the same packages unpinned instead.
      2. `dlimp` (an Octo dependency) separately hard-pins `tensorflow==2.15.0` in its own
         `setup.py`, conflicting with the modern tensorflow installed above — fixed with
         `pip install --no-deps` for dlimp (it's a thin tf.data wrapper, doesn't need that
         exact version).
      3. Octo's own source uses two since-removed JAX APIs: `jax.random.KeyArray` (removed
         JAX 0.4.24, replace with `jax.Array`) and `jax.tree_map` (removed JAX 0.6.0, moved to
         `jax.tree_util.tree_map`, ~20 call sites). Patched in place with `sed` right after
         cloning, before install.
      4. Colab's preinstalled `jax-cuda13-plugin` conflicted with an explicitly-installed
         `jax[cuda12]` plugin — two CUDA plugins fighting to register the same device
         (`ALREADY_EXISTS: PJRT_Api already exists`). **Decision: run this notebook CPU-only.**
         Octo-Small/Base are tiny (27M/93M params), CPU inference for a handful of demo forward
         passes is fast enough, and it sidesteps the GPU plugin churn entirely. (OpenVLA's
         separate notebook still needs/uses the GPU — only Octo's notebook is CPU-only.)
      5. Installing `jax` and `flax` in separate pip calls left `flax` newer than the `jax` it
         actually runs against (`AttributeError: module 'jax.core' has no attribute
         'get_opaque_trace_state'`) — fixed by installing `jax flax chex optax distrax`
         together in one resolver pass.
      6. Unpinned `numpy` resolved to 2.5.4, a very recent release whose wheel crashes on
         import on Colab (`numpy.testing` itself calls a private
         `numpy._core._multiarray_umath._blas_supports_fpe` the build doesn't have — a
         packaging bug in that specific release). Fixed by pinning `numpy>=1.26,<2.3`
         (`numba`'s own stated constraint independently pointed at the same safe range).
      7. `transformers` v5 dropped Flax/TensorFlow support entirely (PyTorch-only), deleting
         `FlaxT5EncoderModel` that Octo's language-instruction tokenizer needs. Fixed with the
         same `transformers<5.0` cap used for OpenVLA (Flax classes are kept through the whole
         v4.x line, maintained as LTS).
      8. **Two more stale-dependency bugs found and fixed by running the whole install+load
         pipeline locally** (Python 3.12 venv on this laptop, CPU-only — much faster debug loop
         than the Colab upload/restart/rerun cycle): `tensorflow_probability`'s JAX backend
         (pulled in via `distrax`, used by the diffusion action head) registers a type handler
         using JAX internals removed in current JAX
         (`jax.interpreters.xla.pytype_aval_mappings`) — both `tensorflow_probability` and
         `distrax` are already at their latest available PyPI release, no fix upstream, so
         patched the installed file directly with a try/except guard (the registration is only
         for emulating mutable `tf.Variable` semantics under JAX, never exercised by Octo).
         Separately, `transformers`' own legacy `modeling_flax_t5.py` calls
         `jnp.clip(..., a_max=...)`, a kwarg JAX renamed to `max=` — patched directly too. Both
         patches are now baked into the notebook as an automated cell (locates the installed
         files via `tensorflow_probability.__file__`/`transformers.__file__` and rewrites them
         at runtime, so it works on Colab without needing repo access).
      **Verified working end-to-end locally**: both Octo-Small and Octo-Base load and run real
      inference on all 4 `test_images/pair*.jpg`, producing genuinely different, sensible
      action vectors per image+instruction (no OpenVLA-style identical-output bug) — confirmed
      with an explicit `all_identical` check, which came back `False` for both models.
- [x] **Confirmed working on real Colab (2026-10-27).** Hit one more Python-3.13-specific bug
      not reproduced locally (Python 3.12): numpy's own `numpy/_core/strings.py` crashes
      setting `.__module__` on ufunc objects under 3.13's stricter attribute rules (ufuncs only
      gained a writable `__dict__` in a later numpy release than the one pinned) — patched with
      the same try/except-guard pattern as the other two stale-dependency fixes. After that,
      both Octo-Small and Octo-Base loaded on Colab and `TEST_PAIRS` produced results matching
      the local validation run **exactly, digit for digit** — the notebook is done.
      **Octo_Demo.ipynb is the confirmed, working, hands-on live demo for the practical segment.**
- [x] **"Try it yourself" upload cell also confirmed working on Colab** — the exact cell
      classmates will use in the practical segment. Tested with `pair1_cup.jpg` and the default
      placeholder instruction ("pick up the object"); both models returned sensible, slightly
      different results from the matching `TEST_PAIRS` entry (same image, different instruction
      text), which is the expected behavior, not a bug.
- [x] **UX fix (2026-10-28): the upload cell only had a file-choose widget — typing the
      instruction required editing Python code (`INSTRUCTION = "..."`) and re-running, not
      obvious for non-technical classmates. Replaced with `input("Type an instruction: ")`
      in both notebooks** — now classmates just run the cell, pick a file, then type the
      instruction when prompted. No code editing needed.
- [x] **8 extra sample images added for classmates who don't have a photo-ready object
      handy** (`test_images/extra1..8_*.jpg`), same BridgeData V2 source/environment as the
      original 4 `TEST_PAIRS`, different sessions (00/03/12/18/22). Instructions are taken
      verbatim from the dataset's own per-trajectory `lang.txt` annotations, not written by
      us. Each was visually checked before inclusion — one candidate (a toy banana occluded
      by the gripper) was dropped and replaced. Documented in `test_images/README.md`.
      **Still to do:** copy the 8 new files into
      `/content/drive/MyDrive/VLA_Demo_Cache/test_images/` alongside the original 4 (same
      sync step as before).
- [x] **Bug (2026-10-28): `NotImplementedError: TIMM Version must be >= 0.9.10 and <
      1.0.0` on OpenVLA load-model cell after a runtime "restart".** Root cause: the
      install cell (cell 6) correctly pins `timm==0.9.10`, but if the Colab *VM itself*
      gets recycled (common on free tier after a disconnect/idle timeout — can look
      identical to a plain "restart" from the UI) you land on a fresh machine with none
      of the pinned packages, only Colab's newer default `timm`. Jumping straight to the
      load-model cell without re-running the install cell then hits the version check.
      Not a one-off — would have bitten classmates live during the demo too. Fixed
      properly instead of just documenting "always re-run the install cell": the
      load-model cell (cell 8) now self-checks the installed `timm` version via
      `importlib.metadata` *before* importing anything, and silently `pip install`s
      `timm==0.9.10` if it's wrong — works regardless of whether the install cell was
      re-run. Markdown (cell 7) explains why.
- [x] **Investigated (2026-10-28): "pick up the fruit" (pair 3) looked like it was picking
      something else** — user ran the full, fixed `OpenVLA_Demo.ipynb` end-to-end on Colab
      (results now embedded in the downloaded `.ipynb` in this folder) and the printed
      action arrow for that case visually sits near the pot, not the pineapple-shaped toy.
      **Root cause: this is a visualization artifact, not (necessarily) a model error.**
      `visualize_action` draws the arrow starting from the image's *geometric center* —
      a static photo gives us no real gripper pixel position, so center was always just a
      placeholder — and the real predicted x/y deltas are tiny (a few mm, e.g.
      `x=0.0, y=-0.002` for this exact pair), so at the old `*100` scale the arrow was
      basically invisible anyway. In the `pair3_fruit.jpg` photo the pot happens to sit
      right at the image center, which is why the (nearly invisible) arrow looked like it
      was "at the pot" — that's about this photo's framing, not the model's semantic
      choice. **Can't actually confirm or rule out a real semantic-grounding failure from
      a single static-image prediction** — OpenVLA/Octo are closed-loop controllers
      (re-run every ~100-200ms on the updated camera frame in real deployment); one
      step's arrow is only the first nudge, not the full trajectory, so it's not valid
      evidence either way.
      **Fixed/improved, both notebooks:** `visualize_action` now uses a much bigger,
      clamped `ARROW_SCALE` (1500, capped at 40% of the image's shorter side) so the
      arrow is actually visible instead of sub-pixel, and plots a yellow dot at the
      (still arbitrary) start point so it reads clearly as "reference point," not "the
      model clicked here." Markdown in both notebooks (Section 4 helper-functions cell,
      and the pair-3-specific note in Section 5/6) now spells out explicitly how to read
      — and how *not* to over-read — the arrow, so groupmates don't repeat this same
      misreading live during the practical segment.
      **For the slide deck:** frame pair 3 as a critical-reading discussion point ("why a
      single-step action vector isn't proof the model understood/misunderstood the
      instruction") rather than quietly dropping it or calling it a bug — it's a more
      interesting, more honest story for the "reading VLA literature/demos critically"
      part of the talk (Section 6 below) than a clean "it worked" result would be.
- [x] **Rewrote every markdown cell in both notebooks (2026-10-28).** User feedback: the
      explanations read as an informal debugging diary ("we originally suspected...",
      "that turned out to be wrong", repeated restatements across cells) — fine for our
      own working notes, but not something to hand to groupmates/classmates for the
      practical segment. Replaced with plain, declarative documentation (what each cell
      does and why, root causes stated as fact) with no loss of technical detail — exact
      versions, error messages and mechanisms are all still there, just not narrated as a
      live investigation. `TASKS.md` stays as the informal, detailed debugging log; the
      notebooks themselves are now presentation/sharing-ready.
- [x] **Decided (2026-10-28): classmates access both notebooks via a QR code slide**
      (one QR per notebook, linking to the GitHub repo's notebook so Colab always opens the
      latest pushed version) shown at the start of the 20-minute practical segment, with the
      two presenting students walking around helping/answering questions.
- [ ] Make the QR code slide(s) — still to do as part of slide deck build.
- [ ] **Local fallback recording — deliberately deferred until close to presentation day**,
      so it reflects the final notebook state, not an intermediate one. Plan: record a real,
      unedited top-to-bottom run of both notebooks (not cherry-picked screenshots), since that
      is what actually proves the environment setup is reproducible — but speed up/trim only
      the long, uninteresting waits (pip install, model shard download, Drive mount) in
      editing afterward; keep normal speed for the parts worth showing live (model-loaded
      confirmation, `TEST_PAIRS` results, the "try it yourself" cell).

## 5. Dropped: ROS2 + Gazebo integration

- Originally considered as a stretch goal, but the course brief's grading requirement — the
  practical activity must let other students themselves experiment with the method — can't be
  satisfied by a single Gazebo/robot-arm rig that only we control; 15+ classmates can't each
  drive one simulated arm in a 20-minute slot. It's also confirmed now that neither paper's own
  evaluation uses Gazebo (both are 100% real-robot evaluations, see above), so there's no "this
  is literally what the authors did" justification for it either.
- Not pursuing this unless the practical-segment format changes (e.g. if a single shared
  Gazebo demo is explicitly allowed as a watch-only segment distinct from the graded hands-on
  part) — revisit only if that gets clarified with the professor.

## 6. Presentation prep

- [ ] Slide flow: lineage framing (RT-2 -> OpenVLA/Octo -> pi-0) -> architecture comparison ->
      your own re-derived benchmark findings (with the eval-condition caveat) -> the 4 test-pair
      results (Section 4 table: baseline, distractor/language grounding, semantic/RT-2-style,
      scene-robustness Small-vs-Base) -> live/recorded demo -> the "reading VLA literature
      critically" aside about the blog's comparison table.
- [ ] Rehearse the live demo at least once on the actual presentation-day setup (not just your
      dev machine) to catch environment surprises (Colab session expiry, missing deps, etc.)
      with time to fix them.
