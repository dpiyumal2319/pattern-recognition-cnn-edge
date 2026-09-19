---
description: Implement a GitHub issue end to end following the AGENTS.md loop
argument-hint: <issue-number>
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

Implement issue **#$1** in this repository, following the loop in `AGENTS.md` exactly.

Work through the phases below in order. Do not skip ahead, and do not start writing implementation
code before phase 3 has passed.

## Phase 0 — Load context

Read, in this order:

1. `AGENTS.md` — the loop, the conventions, the commit rules
2. `docs/interfaces.md` — **the frozen contract.** Your implementation must match the signatures,
   schemas and CLI flags there exactly. If the issue and this file disagree, this file wins and you
   should say so on the PR.
3. The issue itself:

```bash
gh issue view $1 --json number,title,body,labels,milestone
```

Extract from the body: **Goal**, **Depends on**, **Acceptance criteria**, **Report artifact
produced**, **Slide anchor**, and the test file named in `docs/interfaces.md` section 5.

## Phase 1 — Check dependencies

Every issue named in **Depends on** must already be closed and merged into `main`.

```bash
gh issue view <dep> --json number,state,title
```

If any dependency is still open: **stop.** Report which ones and suggest a different issue. Do not
build on an unmerged branch — that is how this project gets a merge disaster on day 11.

## Phase 2 — Branch and open a WIP draft PR

Before writing any code:

```bash
git fetch origin
git switch -c <type>/$1-<slug> origin/main
git commit --allow-empty -m "chore: start #$1"
git push -u origin HEAD
gh pr create --draft --title "WIP: <issue title>" --body "Closes #$1"
```

`<type>` is `feat`, `fix`, `chore`, `docs` or `exp`, matching the issue title prefix.

Confirm the PR URL back to the user before continuing.

## Phase 3 — STOP GATE: lecture slides

Read the issue's **Slide anchor** line.

- If it says "Not applicable" or "No stop-gate for this issue", continue.
- Otherwise, verify the named deck exists in `docs/slides/`.

**If the deck is missing: stop immediately.** Comment on the draft PR saying which deck is needed,
add the `blocked` label, and ask the user to supply it. Do not substitute general machine learning
knowledge for the course's own treatment — the report is graded against the syllabus.

If the deck is present, read the relevant portion before writing methodology, and match its notation
and terminology in both code comments and any prose you produce.

## Phase 4 — Implement

Work against the frozen contract in `docs/interfaces.md`. Replace `NotImplementedError` stubs; do
not invent new public functions or rename arguments.

**Post a comment on the draft PR for each non-obvious decision, at the time you make it** — not
batched at the end. Four headings:

```bash
gh pr comment --body "**What** — ...

**Why** — ...

**Why not the alternatives** — ...

**How** — ..."
```

Log a decision whenever you pick a hyperparameter, choose between approaches, deviate from what the
issue specified, hit an unexpected result, or abandon something that did not work. Dead ends are
worth recording; "we tried X and it failed because Y" is a legitimate report sentence.

These comments are the raw material for `report/sections/*.tex`. Write them as if they were going
into the report, because they are.

## Phase 5 — Verify

Run it. Actually run it.

```bash
pytest <the test file named for this issue> -v
python scripts/run_experiment.py --config <cfg> --smoke   # if this issue touches the pipeline
```

Confirm on disk that the expected artifacts exist and match the schemas in `docs/interfaces.md`.
Walk the issue's **Acceptance criteria** one by one and check each against something you actually
observed, not something you believe to be true.

If anything fails, fix it before phase 6. Do not mark a PR ready with a failing test and a note
about it.

## Phase 6 — Ready for review

Update the PR description to the template in `.github/pull_request_template.md`: `Closes #$1`, what
changed, generated artifact paths, a **Decisions** section distilling your comment thread, and what
you actually ran under Verification.

```bash
gh pr ready
```

Then report back to the user: the PR URL, what you implemented, the decisions you logged, and
anything you could not complete.

## Rules that apply throughout

- **No AI co-authorship trailers** in commits or the PR description. No `Co-Authored-By: Claude`,
  no "Generated with". Commit as the member driving the session.
- Conventional commit subjects: `feat(models): add depthwise separable Model B`.
- Commit in meaningful increments and push as you go — the draft PR is the team's live view of
  progress.
- Stay inside the issue's scope. If you spot something else worth doing, note it on the PR or open
  a separate issue; do not fold it into this branch.
- Never merge your own PR.
