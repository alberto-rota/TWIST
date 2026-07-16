"""
W&B experiment analyzer — fetch metrics, configs, and detect anomalies.
Used by wandb-summarize workflow agents to efficiently pull W&B data.
"""

import json
import os
import subprocess
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ExperimentMetadata:
    """Structured experiment data from W&B."""
    type: str  # 'run' or 'sweep'
    id: str
    name: str
    project: str
    entity: str
    config: Dict[str, Any]
    metrics: Dict[str, List[float]]
    summary: Dict[str, Any]
    status: str  # 'finished', 'running', 'failed'
    duration_minutes: float
    charts: List[Dict[str, str]]
    anomalies: List[str]


def fetch_run_data(entity: str, project: str, run_id: str) -> ExperimentMetadata:
    """
    Fetch a single W&B run's config, metrics, and charts.

    Args:
        entity: W&B entity/team
        project: W&B project name
        run_id: Run ID (short name, not full path)

    Returns:
        ExperimentMetadata with full run data
    """
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb not installed. Run: pip install wandb")

    api = wandb.Api()
    run_path = f"{entity}/{project}/{run_id}"

    try:
        run = api.run(run_path)
    except Exception as e:
        raise ValueError(f"Could not fetch run {run_path}: {e}")

    # Extract config
    config = {k: v for k, v in run.config.items() if not k.startswith('_')}

    # Extract metrics — full history, not just summary
    metrics = {}
    for key in run.keys():
        try:
            # Get the full metric history
            history = run.history(keys=[key], pandas=False)
            if history:
                values = [row.get(key) for row in history if key in row]
                # Filter out None/NaN for analysis
                values = [v for v in values if v is not None and isinstance(v, (int, float))]
                if values:
                    metrics[key] = values
        except:
            pass

    # Summary (final values)
    summary = {k: v for k, v in run.summary.items() if isinstance(v, (int, float, str))}

    # Detect anomalies in metrics
    anomalies = _detect_anomalies(metrics)

    # Charts logged
    charts = []
    for key in run.keys():
        if 'chart' in key.lower() or 'plot' in key.lower():
            charts.append({
                'name': key,
                'type': 'custom',
                'description': f"Logged chart: {key}"
            })

    # Runtime
    duration_minutes = (run.metadata.get('runtime', 0) or 0) / 60

    return ExperimentMetadata(
        type='run',
        id=run_id,
        name=run.name,
        project=project,
        entity=entity,
        config=config,
        metrics=metrics,
        summary=summary,
        status=run.state,
        duration_minutes=duration_minutes,
        charts=charts,
        anomalies=anomalies,
    )


def fetch_sweep_data(entity: str, project: str, sweep_id: str) -> Dict[str, Any]:
    """
    Fetch sweep metadata and summarize runs in the sweep.

    Returns dict with:
    - config: sweep config (params, method, etc)
    - runs: list of {id, name, config_params, key_metrics}
    - best_run: {id, name, metric_value}
    """
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb not installed. Run: pip install wandb")

    api = wandb.Api()
    sweep_path = f"{entity}/{project}/sweeps/{sweep_id}"

    try:
        sweep = api.sweep(sweep_path)
    except Exception as e:
        raise ValueError(f"Could not fetch sweep {sweep_path}: {e}")

    runs_data = []
    best_run = None
    best_metric = None

    for run in sweep.runs:
        summary = {k: v for k, v in run.summary.items() if isinstance(v, (int, float))}
        runs_data.append({
            'id': run.id,
            'name': run.name,
            'state': run.state,
            'config_params': {k: v for k, v in run.config.items() if not k.startswith('_')},
            'summary': summary,
        })

        # Track best run (assumes lower metric is better; adjust as needed)
        if run.summary.get('val/epe') is not None:
            metric_val = run.summary['val/epe']
            if best_metric is None or metric_val < best_metric:
                best_metric = metric_val
                best_run = {'id': run.id, 'name': run.name, 'metric': metric_val}

    return {
        'type': 'sweep',
        'id': sweep_id,
        'name': sweep.name,
        'project': project,
        'entity': entity,
        'config': {
            'method': getattr(sweep, 'method', 'unknown'),
            'parameters': getattr(sweep, 'config', {}).get('parameters', {}),
        },
        'runs': runs_data,
        'run_count': len(runs_data),
        'best_run': best_run,
    }


def _detect_anomalies(metrics: Dict[str, List[float]]) -> List[str]:
    """
    Detect common training anomalies in metric trajectories.
    """
    anomalies = []

    for metric_name, values in metrics.items():
        if not values or len(values) < 2:
            continue

        # Check for NaN/Inf
        has_nan = any(v != v for v in values)  # NaN != NaN
        has_inf = any(abs(v) == float('inf') for v in values)

        if has_nan:
            anomalies.append(f"NaN detected in {metric_name}")
        if has_inf:
            anomalies.append(f"Inf detected in {metric_name}")

        # Check for collapse (sudden drop to zero or near-zero)
        if metric_name not in ['epoch', 'step']:
            final_val = values[-1]
            start_val = values[0]
            if start_val != 0 and final_val / max(abs(start_val), 1e-6) < 0.1:
                anomalies.append(f"{metric_name} collapsed ({start_val:.3f} → {final_val:.3f})")

            # Check for divergence (exploding values)
            max_val = max(abs(v) for v in values)
            mean_val = sum(abs(v) for v in values) / len(values)
            if mean_val > 0 and max_val / mean_val > 10:
                anomalies.append(f"{metric_name} shows high variance/divergence (max/mean={max_val/mean_val:.1f}x)")

        # Check for saturation (no improvement in last 20% of training)
        if len(values) > 10:
            late_start = int(len(values) * 0.8)
            early_mean = sum(values[:late_start]) / late_start
            late_mean = sum(values[late_start:]) / (len(values) - late_start)

            if metric_name in ['loss', 'train_loss'] and early_mean > 0:
                improvement = abs(early_mean - late_mean) / early_mean
                if improvement < 0.01:
                    anomalies.append(f"{metric_name} saturated (no improvement in last 20% of training)")

    return anomalies


def export_experiment_summary(exp: ExperimentMetadata, output_file: Optional[str] = None) -> str:
    """
    Export experiment summary as JSON.
    """
    data = asdict(exp)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        return f"Exported to {output_file}"

    return json.dumps(data, indent=2)


if __name__ == '__main__':
    # Example usage for testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python utilities/wandb_analyzer.py <entity>/<project>/<run_id>")
        print("  or:  python utilities/wandb_analyzer.py <entity>/<project>/sweeps/<sweep_id>")
        sys.exit(1)

    path = sys.argv[1]
    parts = path.split('/')

    if 'sweeps' in path:
        entity, project = parts[0], parts[1]
        sweep_id = parts[3]
        data = fetch_sweep_data(entity, project, sweep_id)
    else:
        entity, project, run_id = parts[0], parts[1], parts[2]
        data = fetch_run_data(entity, project, run_id)

    print(json.dumps(asdict(data) if hasattr(data, '__dataclass_fields__') else data, indent=2))
