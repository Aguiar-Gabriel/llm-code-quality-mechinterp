from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def join_with_sonar(
    prompts_csv_path: str,
    sonar_metrics_jsonl_path: str,
    output_jsonl_path: str,
    class_path_field: str = "class_path",
) -> None:
    """
    Join prompts with Sonar metrics by class_path, emitting a filtered JSONL.

    Expects `sonar_metrics_jsonl_path` to contain one JSON object per line with a
    `class_path` key. Records that match a class in prompts.csv are written to output.
    """
    df = pd.read_csv(prompts_csv_path, usecols=[class_path_field])
    wanted = set(df[class_path_field].astype(str).tolist())

    src = Path(sonar_metrics_jsonl_path)
    if not src.exists():
        raise FileNotFoundError(f"Sonar metrics file not found: {src}")

    out_path = Path(output_jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in, n_out = 0, 0
    with out_path.open("w", encoding="utf-8") as out:
        for obj in _read_jsonl(src):
            n_in += 1
            cp = str(obj.get(class_path_field, ""))
            if cp in wanted:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_out += 1


def run_join_from_params(params_metrics: dict):
    """Kedro wrapper to run join and then return path content as list of dicts."""
    join_with_sonar(**params_metrics)
    out = Path(params_metrics["output_jsonl_path"]).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in out if l.strip()]
