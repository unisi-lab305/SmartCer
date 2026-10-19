"""
Reference values of SmartCer 0.6.2 for the later phases of the 1.0 refactoring.

Runs the 0.6.2 environment (`smartcer.cer_env.SmartCer`) on a local scenario with every battery
idle (zero actions) and writes, to the output directory:

- `<name>_community.csv`: the hourly community series of `get_sim_results()['cer']`, full precision;
- `<name>_kpis.json`: run metadata, the annual sums of those series and the 0.6.2 KPIs
  (`compute_kpis`) without rounding.

The default output directory `test/fixtures/local/` is gitignored: the hourly series derive from
local, partly confidential data and must not be committed. The script depends on the 0.6.2 modules
and is deleted in phase 3; its outputs stay.

Usage:
    python scripts/make_reference_v062.py
    python scripts/make_reference_v062.py --scenario smartcer/cer_data/Leather --name leather_2022
"""

import argparse
import collections
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Dict

import numpy as np
import pandas as pd

import smartcer
from smartcer.cer_env import SmartCer
from smartcer.cer_metrics.scores import compute_kpis


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--scenario', type=Path,
                        default=REPO_ROOT / 'smartcer' / 'cer_data' / 'Leather',
                        help='scenario folder in the 0.6.2 format (read only)')
    parser.add_argument('--start', default='2022-01-01 00:00:00', help='first timestamp')
    parser.add_argument('--end', default='2022-12-31 23:00:00', help='last timestamp')
    parser.add_argument('--name', default='leather_2022', help='prefix of the output files')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'test' / 'fixtures' / 'local',
                        help='output folder (gitignored)')
    return parser.parse_args()


def git_commit() -> str:
    """
    Commit of the working tree that produced the reference values.

    Returns:
        str: Short commit hash, or 'unknown' outside a git repository.
    """
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    return result.stdout.strip()


def run_idle(scenario: Path, start: str, end: str) -> Dict[str, Any]:
    """
    Run one 0.6.2 episode with every battery idle.

    Args:
        scenario: Scenario folder in the 0.6.2 format.
        start: First timestamp, '%Y-%m-%d %H:%M:%S'.
        end: Last timestamp, '%Y-%m-%d %H:%M:%S'.

    Returns:
        Dict[str, Any]: The community table of `SmartCer.get_sim_results()`.

    Raises:
        RuntimeError: If the episode does not terminate after the expected number of steps.
    """
    run_config = {
        'simulation': {'start': start, 'end': end, 'seed': 0},
        'reward': {
            'type': 'profit',
            'single_agent': False,
            'constraints': {'penalty': True, 'coefficient': 10.0, 'max_ratio': 0.99},
            'scaler': {'apply': False},
        },
    }
    env = SmartCer(cer_path=str(scenario), cer_config=run_config)
    env.reset(seed=0)
    actions = collections.OrderedDict(
        (node_id, np.array([0.0], dtype=np.float32)) for node_id in env.storage_nodes_ids
    )
    terminated = False
    for _ in range(env.sim_hours):
        _, _, terminated, _, _, _ = env.step(actions)
    if not terminated:
        raise RuntimeError(f'episode did not terminate after {env.sim_hours} steps')
    return env.get_sim_results()['cer']


def main() -> None:
    """
    Run the idle 0.6.2 episode and write the reference files.
    """
    args = parse_args()
    if not (args.scenario / 'config.json').is_file():
        raise FileNotFoundError(f'scenario not found: {args.scenario / "config.json"}')

    community = pd.DataFrame(run_idle(args.scenario, args.start, args.end))
    series = [column for column in community.columns if column != 'datetime']

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f'{args.name}_community.csv'
    json_path = args.output_dir / f'{args.name}_kpis.json'
    community.to_csv(csv_path, index=False, float_format='%.17g')

    reference = {
        'metadata': {
            'description': 'SmartCer 0.6.2, every battery idle (zero actions)',
            'scenario': args.scenario.name,
            'start': args.start,
            'end': args.end,
            'steps': len(community),
            'smartcer_version': smartcer.__version__,
            'git_commit': git_commit(),
            'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        },
        'annual_sums': {column: float(community[column].sum()) for column in series},
        'kpis_v062': compute_kpis(results_cer={c: community[c].tolist() for c in series},
                                  rounding=15),
    }
    json_path.write_text(json.dumps(reference, indent=2) + '\n')

    print(f'{len(community)} steps written to {csv_path} and {json_path}')


if __name__ == '__main__':
    main()
