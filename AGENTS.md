# AGENTS.md

Guidance for every contributor — human or agent — working in this repository.
`CLAUDE.md` imports this file, so these rules bind Claude Code, Codex and any other agent equally.

---

## 0. What this project is

**EN3150 Assignment 03 — Resource-Constrained CNN for Edge Image Classification.**
Group assignment, 100 marks, deadline **02 October 2026**. The brief is
[`docs/EN3150_Assignment_03_In23.pdf`](docs/EN3150_Assignment_03_In23.pdf) — read it before your
first issue.

Six graded tasks:

| Task | Marks | What it wants |
|---|---|---|
| 1. Data preparation | — | Dataset ≤64×64, split 70/15/15 |
| 2. Custom architecture design | 20 | Model A (standard CNN) + Model B (depthwise separable, ≤100k params), hand-derived parameter counts, hardware-aware activation justification |
| 3. Optimizer selection & tuning | 15 | Chosen optimizer vs SGD vs SGD+momentum, effect of the momentum term |
| 4. Custom training & evaluation | 25 | ≥20 epochs, loss curves, accuracy, confusion matrix, precision, recall, A-vs-B comparison table |
| 5. SOTA fine-tuning | 20 | Two pretrained lightweight nets on the same splits, params + size in MB |
| 6. Final comparison & trade-offs | 20 | Model B vs SOTA: accuracy vs memory vs compute |

**Roughly 60 of the 100 marks are written analysis.** The lecturer grades the PDF only. Code that
runs but produces no table, figure or paragraph in the report earns nothing.

### Locked project decisions

| Decision | Choice |
|---|---|
| Dataset | **EuroSAT** — 27,000 Sentinel-2 RGB images, 10 classes, natively 64×64 |
| Framework | **PyTorch + torchvision** (`torchvision>=0.20` — see below) |
| Compute | **Google Colab canonical**, local optional. All code device-agnostic. |
| Layout | `src/pr_edge/` package + `scripts/` CLIs + `configs/` YAML + `results/` artifacts |
| Report | **LaTeX in this repo**, built from generated tables |
| Visibility | **Private until submission**, flipped to public on 02 Oct |

> **Pin `torchvision>=0.20`.** Earlier versions download EuroSAT from `madm.dfki.de`, which is dead.
> `>=0.20` uses the HuggingFace `torchgeo/eurosat` mirror.

> **EuroSAT is natively 64×64**, so the brief's "downscale to a maximum of 64×64" is satisfied by
> construction. State this explicitly in the report with a sample grid — do not silently skip it.

---

## 1. The loop — follow this in order, every time

### 1. Take an issue

Assign yourself on GitHub and comment that you are starting, so nobody duplicates work.
Read the issue's **Depends on** line and confirm every dependency is already merged into `main`.
If a dependency is still open, pick a different issue — do not build on an unmerged branch.

### 2. Branch fresh off the latest `main`

Never branch off another feature branch.

```bash
git fetch origin
git switch -c <type>/<issue-number>-<slug> origin/main
```

`<type>` is one of `feat`, `fix`, `chore`, `docs`, `exp`. Example: `feat/10-model-b-depthwise`.

### 3. Open a WIP draft PR *before* you start working

Push the branch and open the pull request as a **draft**, immediately — before any implementation.
If you have nothing to push yet, start with an empty commit:

```bash
git commit --allow-empty -m "chore: start #<N>"
git push -u origin HEAD
gh pr create --draft --title "WIP: <issue title>" --body "Closes #<N>" --fill-first
```

Why this comes first: the draft PR is the team's live view of who is working on what and how far it
has got. Four people on one board cannot see branches, but they can see PRs. It also means the diff
is reviewable incrementally instead of landing as a wall of code on the last day.

If the slides gate below blocks you, say so in the draft PR and add the `blocked` label. A stalled
draft PR is visible; a stalled local branch is not.

### 4. STOP GATE — lecture slides

**Before writing any methodological code or prose, confirm the relevant deck exists in
`docs/slides/`.** The issue's **Slide anchor** line names it.

| Topic | Deck |
|---|---|
| Convolution, pooling, parameter counting, depthwise separable, MobileNet, transfer learning | `docs/slides/CNNs.pdf` |
| Optimizers, momentum, Adam, activations, backpropagation, initialisation | `docs/slides/DNNs.pdf` |
| Metrics, confusion matrix, precision/recall, softmax | `docs/slides/Classification 1.pdf`, `Classification 2.pdf` |
| Train/val/test methodology, overfitting, regularisation | `docs/slides/Learning from data.pdf` |

**If the deck you need is missing, stop and ask the human. Do not substitute general machine
learning knowledge for the course's own treatment.**

This is a hard gate, not a preference. The report is graded against the syllabus. A method the
lecturer did not teach is a liability in the report even when it is technically correct, and a
derivation in notation he does not use costs marks for no benefit.

### 5. Implement, grounded in the slides — and log decisions on the PR as you go

Match the decks' notation and terminology. Where the slides give a worked example, mirror its
structure so the report reads as coursework rather than as a blog post. The clearest case:
`CNNs.pdf` derives depthwise separable cost as

```
regular 3x3, 256 -> 256 channels :  9 x 256 x 256  ~= 590,000 ops
depthwise  3x3, per channel      :  9 x 256        ~=   2,300 ops
pointwise  1x1, 256 -> 256       :      256 x 256  ~=  65,500 ops
total                            :                 ~=  67,800 ops   (8.7x speedup)
```

Your Model A vs Model B arithmetic should follow exactly that shape.

**Every non-obvious decision gets a comment on the draft PR, at the moment you make it.** Not at the
end, not in the final description — the value is in capturing the reasoning while the alternatives
are still fresh. Use four headings:

```markdown
**What** — replaced the Flatten -> Dense head with global average pooling.

**Why** — the dense head was 187k parameters on its own, over the 100k budget before a single
convolution was counted. GAP drops it to 1,290.

**Why not the alternatives** — narrowing the conv channels instead would have hit the budget too,
but costs representational capacity in the feature extractor, which is where the accuracy actually
comes from. Reducing the flatten size by adding a fifth pooling stage would leave a 2x2 feature map
and throw away spatial detail that the 64x64 input barely has to begin with.

**How** — nn.AdaptiveAvgPool2d(1) then a single Linear(128, 10). Verified by the sub-100k unit test
in tests/test_model_b.py.
```

This is not bureaucracy. Roughly 60 of the 100 marks are discussion, and these comments are the
raw material for it — the "why not" paragraphs in particular are what turn a results table into an
argument. When you write `report/sections/*.tex`, you should be editing your own PR comments, not
reconstructing forgotten reasoning two weeks later.

Log a decision whenever you: pick a hyperparameter, choose between two architectures or libraries,
deviate from what an issue specified, hit a result you did not expect, or discard an approach that
did not work. **Dead ends are worth recording** — "we tried X and it failed because Y" is a
legitimate and often high-value report sentence.

### 6. Verify before marking the PR ready

Run the script end to end. Confirm the `results/<run_id>/` artifacts actually appear on disk and
match the contract in section 3. An issue is not done because the code imports cleanly.

### 7. Mark the PR ready for review

```bash
gh pr ready
```

Before you do, bring the description up to date so a reviewer never has to read the comment thread
to understand the change:

- Title mirrors the issue; drop the `WIP:` prefix.
- Body contains `Closes #N`, a summary of what changed, the paths of any generated artifacts, and a
  short **Decisions** section distilling the comment thread.
- Request review from one other member.

### 8. Merge and clean up

Never merge your own PR without one approval. After merge, delete the branch and return to `main`:

```bash
git switch main && git pull origin main && git branch -d <branch>
```

---

## 2. Commits and attribution

- Agents commit **on behalf of the group member driving the session**, using that member's
  configured `user.name` / `user.email`.
- **No AI co-authorship trailers.** Do not add `Co-Authored-By: Claude ...`,
  `Co-Authored-By: Codex ...`, any "Generated with" line, or any equivalent tool attribution to
  commit messages or PR descriptions. Commit messages describe the change and nothing else.
- Conventional-commit subjects: `feat(models): add depthwise separable Model B`.
- **Commit regularly.** The brief requires a GitHub link so the lecturer can see the work spread
  over a reasonable duration. One giant commit the night before is visible and looks bad.

---

## 3. The results contract

This is the interface that lets four people work in parallel. Do not change it without an issue.

Every experiment writes `results/<run_id>/`:

| File | Contents |
|---|---|
| `config.yaml` | The fully resolved config for this run, including the seed |
| `history.json` | `[{epoch, train_loss, val_loss, train_acc, val_acc, epoch_seconds}, ...]` |
| `metrics.json` | `{test_accuracy, macro_precision, macro_recall, per_class:{...}, confusion_matrix:[[...]]}` |
| `profile.json` | `{total_params, trainable_params, disk_size_kb, macs, mean_epoch_seconds}` |
| `model.pt` | Checkpoint (gitignored) |
| `figures/*.png` | Loss curves, confusion matrix heatmap |

Run IDs are fixed so issues can reference each other's outputs before they exist:

```
model_a   model_b   model_b_sgd   model_b_sgdm
mobilenetv2_64   efficientnetb0_64   mobilenetv2_224
```

**Numbers reach the report only through generated tables:**

```
results/*/[metrics|profile].json  ->  scripts/build_tables.py  ->  report/tables/*.tex  ->  \input{}
```

Never hand-type a result into LaTeX. If a run is repeated, the report updates itself.

---

## 4. Repo conventions

- Config-driven. No hardcoded hyperparameters, no absolute paths.
- Device-agnostic: always `get_device()`, never a bare `.cuda()`. Members run on Colab, NVIDIA,
  Apple Silicon and CPU.
- Seed every run and record the seed in `config.yaml`.
- Notebooks commit with outputs stripped.
- **Commit** `results/**/*.json` and `results/**/figures/*.png` — they are the report's data source.
- **Do not commit** `data/`, `*.pt` checkpoints, or LaTeX build artifacts.
- Comment the code. The brief requires the submitted code to be "complete, well-documented
  (commented), and ready to run".

---

## 5. Report-first rule

Every issue names the **report artifact** it produces — a table, a figure, or a named section.
If a proposed change produces none of those, question whether it belongs in a 13-day assignment.

---

## 6. Scope guard

The brief defines the scope. No ONNX export, no quantization, no architecture search, no extra
datasets, no hyperparameter search beyond what Task 3 asks for. The edge argument is made
**analytically** — MAC counts, memory footprint in KB, and what would fit in a microcontroller's
SRAM.

Out-of-scope ideas become issues labelled `stretch` and stay unassigned until the graded work is
done.

---

## 7. Open TODOs

These block the final assembly issue and need a human answer:

- [ ] Group number, for the `<groupno>_A03_EN3150` submission filename
- [ ] All four members' names and index numbers, for the report title page
- [ ] Two more collaborator invites (currently `dpiyumal2319`, `himadreee`)
- [ ] Register EuroSAT in the shared course dataset sheet
