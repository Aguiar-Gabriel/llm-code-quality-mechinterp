from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import AutoTokenizer


def tokenize_prompts(
    prompts_csv_path: str,
    model_name_or_path: str,
    max_length: int,
    output_pkl_path: str,
    pad_to_max_length: bool = True,
    local_files_only: bool = False,
    text_template: Optional[str] = None,
) -> None:
    """
    Tokenize prompts into fixed-length input_ids.

    - Reads CSV with columns: scenario, class_path, java_code, system_prompt, user_prompt
    - Builds text using `text_template` or defaults to: system + two newlines + user + two newlines + code
    - Uses EOS as PAD if tokenizer has no PAD; sets `tokenizer.pad_token = tokenizer.eos_token`
    - Saves a dict with: input_ids (List[List[int]]), attention_mask, meta (scenario, class_path)
    """
    df = pd.read_csv(prompts_csv_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True, local_files_only=local_files_only
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_text(row) -> str:
        if text_template:
            return text_template.format(
                system_prompt=row["system_prompt"],
                user_prompt=row["user_prompt"],
                java_code=row["java_code"],
            )
        return (
            str(row["system_prompt"]).strip()
            + "\n\n"
            + str(row["user_prompt"]).strip()
            + "\n\n"
            + str(row["java_code"]).strip()
        )

    texts = [build_text(r) for _, r in df.iterrows()]
    enc = tokenizer(
        texts,
        padding="max_length" if pad_to_max_length else True,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )

    payload = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc.get("attention_mask"),
        "meta": df[["scenario", "class_path"]].to_dict(orient="records"),
        "tokenizer": tokenizer.name_or_path,
        "max_length": max_length,
    }

    out_path = Path(output_pkl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)


def run_tokenize_from_params(params_tokenize: dict):
    """Kedro wrapper to run tokenization from parameters and return the payload.

    This calls `tokenize_prompts`, then loads and returns the pickled object so
    the catalog can persist it to `prompts_tokenized`.
    """
    tokenize_prompts(**params_tokenize)
    with open(params_tokenize["output_pkl_path"], "rb") as f:
        return pickle.load(f)
