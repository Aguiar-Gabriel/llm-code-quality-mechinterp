from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional

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

    # Read all source objects first to avoid in-place truncation when src == out
    objs = list(_read_jsonl(src))
    n_in, n_out = 0, 0
    with out_path.open("w", encoding="utf-8") as out:
        for obj in objs:
            n_in += 1
            cp = str(obj.get(class_path_field, ""))
            if cp in wanted:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_out += 1


def run_join_from_params(params_metrics: dict, ready: object | None = None):
    """Kedro wrapper to run join and then return path content as list of dicts.

    `ready` is a dummy dependency to enforce execution order after the optional
    generator node; it is otherwise ignored.
    """
    join_with_sonar(**params_metrics)
    out = Path(params_metrics["output_jsonl_path"]).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in out if l.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_sonar_metrics(
    mode: str,
    output_jsonl_path: Optional[str] = None,
    prompts_csv_path: Optional[str] = None,
    source_jsonl_path: Optional[str] = None,
    sonar_repo_root: Optional[str] = None,
    project_relpath: str = "sonarvsllm-testcases",
    maven_command: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Generate sonar_metrics.jsonl before the join step.

    Modes
    - 'copy': copy an existing JSONL from `source_jsonl_path` to `output_jsonl_path`.
    - 'maven': run `maven_command` inside the sonar repo project directory, then copy
               from `source_jsonl_path` if provided; otherwise, leave the file as produced.
    - 'stub': create a minimal JSONL using `prompts_csv_path` with placeholder metrics.
    """
    # Choose output path, fallback to common default if not provided
    out = Path(output_jsonl_path or "data/02_intermediate/sonar_metrics.jsonl")
    if out.exists() and not overwrite:
        return

    mode = (mode or "").strip().lower()
    if mode == "copy":
        if not source_jsonl_path:
            raise ValueError("metrics_generate.mode=copy requires source_jsonl_path")
        src = Path(source_jsonl_path)
        if not src.exists():
            raise FileNotFoundError(f"source_jsonl_path not found: {src}")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
        return

    if mode == "maven":
        if not sonar_repo_root:
            raise ValueError("metrics_generate.mode=maven requires sonar_repo_root")
        workdir = Path(sonar_repo_root).resolve() / project_relpath
        if not workdir.exists():
            raise FileNotFoundError(f"Sonar project dir not found: {workdir}")
        cmd = maven_command or (
            "./mvnw verify org.sonarsource.scanner.maven:sonar-maven-plugin:sonar"
        )
        # Run Maven; rely on environment for SonarCloud credentials/key
        proc = subprocess.run(cmd, cwd=str(workdir), shell=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Maven/Sonar command failed (exit {proc.returncode}). "
                f"Tried: {cmd} in {workdir}"
            )
        # If a source_jsonl_path is provided, copy it to the output
        if source_jsonl_path:
            src = Path(source_jsonl_path)
            if not src.exists():
                raise FileNotFoundError(
                    f"Sonar metrics export not found after Maven run: {src}"
                )
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            return
        # Otherwise, assume the command already produced `output_jsonl_path`.
        if not out.exists():
            raise FileNotFoundError(
                "No JSONL produced. Provide `source_jsonl_path` or adjust command to write "
                f"to {out}"
            )
        return

    if mode == "stub":
        if not prompts_csv_path:
            raise ValueError("metrics_generate.mode=stub requires prompts_csv_path")
        import pandas as pd

        df = pd.read_csv(prompts_csv_path, usecols=["class_path"])  # type: ignore
        rows = []
        for cp in df["class_path"].astype(str).tolist():
            rows.append(
                {
                    "class_path": cp,
                    "metrics": {
                        "code_smells": None,
                        "cognitive_complexity": None,
                        "complexity": None,
                        "lines": None,
                        "comment_lines_density": None,
                        "sqale_rating": None,
                    },
                    "source": "stub",
                }
            )
        _write_jsonl(out, rows)
        return

    raise ValueError(f"Unknown metrics_generate.mode: {mode}")


def run_generate_from_params(params: dict):
    """Kedro wrapper to generate sonar metrics if needed (optional step).

    Be tolerant to minimal local overrides by providing sensible defaults
    for missing keys like `output_jsonl_path` and `prompts_csv_path`.
    """
    merged = dict(params or {})
    merged.setdefault("output_jsonl_path", "data/02_intermediate/sonar_metrics.jsonl")
    merged.setdefault("prompts_csv_path", "data/01_raw/prompts.csv")
    generate_sonar_metrics(**merged)
    # Return a simple flag to establish dependency in the pipeline
    return {"status": "ok", "path": merged.get("output_jsonl_path")}
