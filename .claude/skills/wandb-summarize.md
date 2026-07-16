# wandb-summarize

Fetch, analyze, and summarize W&B experiments (runs or sweeps) with automated anomaly detection, trend analysis, and comparative reports.

## Usage

**Single run summary:**
```bash
/wandb-summarize run-id:<ENTITY>/<PROJECT>/<RUN_ID>
```

**Single sweep summary:**
```bash
/wandb-summarize sweep:<ENTITY>/<PROJECT>/<SWEEP_ID>
```

**Compare multiple runs:**
```bash
/wandb-summarize run-id:<E>/<P>/<R1> run-id:<E>/<P>/<R2> comparative:true
```

**Focus investigation on a specific topic:**
```bash
/wandb-summarize run-id:<E>/<P>/<RUN_ID> investigation:"gate health and visibility prediction"
```

## What it does

- **Fetches** all metrics, config, charts, and system info from W&B API
- **Detects anomalies**: metric collapse, NaN values, unusual spikes, saturation points
- **Trends analysis**: improving vs stuck vs diverging vs collapsed
- **Health verdict**: green/yellow/red with specific findings
- **Comparative mode**: ranks experiments, identifies patterns, surfaces winning configs
- **Actionable recommendations**: specific hyperparameter changes based on observed behavior

## Examples

### Investigation: "Is the gate learning properly?"
Fetches `gate_vis`, `gate_occ` metrics and analyzes whether the gate is discriminating visible vs occluded points.

### Investigation: "Why did training collapse at epoch 15?"
Identifies the collapse point, checks for NaN/inf, correlates with loss terms, config, and suggests what changed.

### Comparative: "Which of these 3 configs is best?"
Ranks them by final metric values, compares learning curves, highlights shared failure modes, recommends which to scale up.

## Output format

For single experiments:
- Health verdict (1 sentence)
- Key metrics table
- Trend narrative
- Anomalies (if any)
- Specific parameter recommendations
- Visual insights

For comparative:
- Ranking by effectiveness
- Side-by-side metrics table
- Per-run status
- Cross-run patterns
- Unified next steps
