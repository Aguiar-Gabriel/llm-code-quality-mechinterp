from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from feature_lens.settings_prompts import (
    SYSTEM_PROMPT_PAPER_V2 as SYSTEM_PROMPT,
    USER_PROMPT_PAPER_V2 as USER_PROMPT,
)
import pandas as pd


@dataclass
class PromptRecord:
    scenario: str
    class_path: str
    java_code: str
    system_prompt: str
    user_prompt: str


def _iter_java_files(controlled_root: Path) -> Iterable[tuple[str, Path]]:
    """
    Yield (scenario, path) for all .java files under
    classFilesToBeAnalysed/controlled/<scenario>/**.java
    """
    for scenario_dir in sorted((controlled_root).iterdir()):
        if not scenario_dir.is_dir():
            continue
        scenario = scenario_dir.name
        for p in scenario_dir.rglob("*.java"):
            yield scenario, p


def collect_prompts(
    sonarvsllm_repo_root: str,
    output_csv_path: str,
    n_limit: int | None = None,
) -> None:
    """
    Walk the sonarvsllm testcases controlled scenarios and build prompts.csv.

    - scenario: name of subdirectory directly under `controlled/`
    - class_path: path relative to the sonarvsllm repo root
    - java_code: UTF-8 content of the .java file
    - system_prompt, user_prompt: from settings/prompts.py
    """
    repo_root = Path(sonarvsllm_repo_root).resolve()
    controlled_root = (
        repo_root
        / "sonarvsllm-testcases"
        / "src"
        / "main"
        / "resources"
        / "classFilesToBeAnalysed"
        / "controlled"
    )
    if not controlled_root.exists():
        raise FileNotFoundError(f"Controlled root not found: {controlled_root}")

    records: list[PromptRecord] = []
    for scenario, path in _iter_java_files(controlled_root):
        try:
            code = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            code = path.read_text(errors="replace", encoding="utf-8")
        class_path = str(path.relative_to(repo_root)).replace("\\", "/")
        records.append(
            PromptRecord(
                scenario=scenario,
                class_path=class_path,
                java_code=code,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT,
            )
        )
        if n_limit is not None and len(records) >= int(n_limit):
            break

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "class_path",
                "java_code",
                "system_prompt",
                "user_prompt",
            ],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "scenario": r.scenario,
                    "class_path": r.class_path,
                    "java_code": r.java_code,
                    "system_prompt": r.system_prompt,
                    "user_prompt": r.user_prompt,
                }
            )


def run_collect_from_params(params_collect_prompts: dict, params_collect: dict | None = None):
    """Kedro wrapper: merge optional `collect` group (e.g., n_limit) and run.

    Returns a DataFrame loaded from the written CSV so it can be saved by
    a Kedro CSVDataSet if wired that way.
    """
    merged = dict(params_collect_prompts)
    if params_collect:
        merged.update(params_collect)
    collect_prompts(**merged)
    return pd.read_csv(merged["output_csv_path"])  # convenience for catalog writes
