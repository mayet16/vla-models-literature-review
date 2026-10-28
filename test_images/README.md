# Test images

Downloaded directly from the public BridgeData V2 raw dataset
(`rail.eecs.berkeley.edu/datasets/bridge_release/...`, toykitchen7/many_skills environment) —
the same source Octo's own official example notebook uses. No photography needed; these are
already in-distribution for both models' pretraining data, which makes clean, reproducible
predictions more likely for the slide-deck comparison set.

- `pair1_cup.jpg` — spatula + towel + cans. Instruction: "pick up the spatula". Baseline check.
- `pair2_fork_cup.jpg` — pot + spoon + cans. Instruction: "pick up the spoon, not the pot".
  Distractor/language-grounding test.
- `pair3_fruit.jpg` — small pineapple-shaped toy + frog-shaped toy + pot + cans. Instruction:
  "pick up the fruit". Semantic/novel-instruction test (RT-2-style).
- `pair4_cluttered.jpg` — 2 sushi-shaped toys + cup + spatula + cans. Instruction: "pick up the
  spatula". Scene-robustness test (Octo-Small vs Octo-Base).

See `TASKS.md` (Section 4) for the full rationale behind each pair.

## Extra images for classmates to try themselves (optional, not in `TEST_PAIRS`)

Same source and environment (toykitchen7/many_skills), different sessions than the 4 pairs
above (00, 03, 12, 18, 22 — not 05/10/20/25). Unlike the 4 pairs, these instructions are
taken **verbatim from the dataset's own `lang.txt` annotations** (the real human-written
instruction recorded for that trajectory when the demo was collected), not written by us —
a nice talking point if asked ("how did you pick instructions?"). Minor grammar is the
original annotator's, left as-is. Meant for the "try it yourself" cell (Section 5/6): upload
one of these instead of a personal photo if you don't have an object handy, or classmates can
use them as inspiration for phrasing their own instruction.

| Filename | Scene | Suggested instruction |
|---|---|---|
| `extra1_red_block.jpg` | red block/sponge + blue spoon + teal cloth + cans | "move the red thing next to the blue spoon" |
| `extra2_pot_lid.jpg` | silver pot + lid, separated | "put the lid on the silver pot" |
| `extra3_can.jpg` | peach can + blue spoon + pot | "move the can to the right side" |
| `extra4_blue_spoon.jpg` | blue spoon + peach can + teal cloth | "take the blue spoon and put it to the left side of the can" |
| `extra5_red_fork.jpg` | red fork + green beans can + pot | "pick up the red fork" |
| `extra6_grasped_spoon.jpg` | spoon already in gripper, pot + cans on table | "put the spoon on the table" |
| `extra7_red_spoon.jpg` | red spoon + cherries can + cloth | "move the red spoon to the upper right of the table" |
| `extra8_spoon_mushroom.jpg` | red spoon + mushroom toy + cloth | "pick up the red spoon" |

All 8 were visually checked (not just picked from metadata) to confirm the described object
is actually clearly visible in frame — one initial candidate (a toy banana, occluded by the
gripper in its first frame) was dropped for that reason.

Next step: copy this whole folder (now including the 8 extras) to
`/content/drive/MyDrive/VLA_Demo_Cache/test_images/` so both Colab notebooks can read them
without re-uploading every session — same as the original 4 pairs.
