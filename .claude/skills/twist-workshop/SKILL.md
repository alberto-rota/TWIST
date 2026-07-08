---
name: twist-workshop
description: Build and execute a TWIST workshop verification notebook — REQUIRED for every codebase change. Use when finishing any code addition/change, when asked to demonstrate or verify a component, or before launching a run that depends on new code.
---

# TWIST workshop notebooks

**Every codebase change ships with an executed `workshops/NN_<topic>.ipynb`.**
This is the project's only test suite: the user reviews progress through these,
and the rule is that the notebook must call the EXACT modules real runs use —
`utilities.config.load_and_process_config`, `create_datasets_from_config`,
`create_model_from_config`, `create_loss_from_config`, `Engine`,
`utilities.evaluation.evaluate_and_report` — never a mock or a re-implementation.

## Recipe

1. Pick the next free number: `ls workshops/*.ipynb | tail` (numbering has
   duplicates like 26/27/28 from parallel work — that's tolerated; just don't
   reuse an existing exact filename). `workshops/` is gitignored (local review
   artifacts).
2. Build programmatically with `nbformat` (write a small builder script in the
   scratchpad, not in the repo):

```python
import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = [nbf.v4.new_markdown_cell("# NN — <topic>\nWhat this demonstrates and why."),
         nbf.v4.new_code_cell(
             "import os, sys\n"
             "os.chdir('/anvme/workspace/v120bb18-twist')\n"
             "sys.path.insert(0, os.getcwd())"),
         # ... cells that exercise the real code path ...
        ]
nb["cells"] = cells
nbf.write(nb, "workshops/NN_topic.ipynb")
```

3. Execute in place (login node = CPU: use boot/CNN-encoder paths, tiny slices):

```bash
http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80 \
  .venv/bin/jupyter nbconvert --execute --inplace workshops/NN_topic.ipynb
```

4. Confirm 0 errors and that outputs (metrics, plots, rendered frames) actually
   demonstrate the claim. A notebook that runs but shows nothing decisive is not
   done — include a known-answer or before/after cell whenever possible (the POR
   metric shipped with hand-computed exact cases; the rollout loss shipped with a
   per-horizon-step profile).

## Content conventions that made past workshops useful

- First markdown cell: what changed, why, and what "verified" means here.
- Load the real config (`load_and_process_config("config/train_best.yaml", ...)`)
  and override to CPU scale via the same dotted-override mechanism the CLI uses —
  that also verifies the config path for the new keys.
- For model changes: shape-check both flag states (v2 blocks default OFF must stay
  checkpoint-compatible with v1), then a micro-overfit or known-answer forward.
- For data changes: render a frame with GT overlaid (coordinate-scale bugs are the
  recurring dataset failure — EndoTAPP's prep had one).
- For loss/engine changes: print the per-term `w_*` decomposition on a real batch
  and eyeball position dominance (Law 1 of twist-experiment-playbook).
- CPU end-to-end gate before any GPU launch: `python train.py -b` must stay green.
