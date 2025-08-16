from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _iter_shards(n_items: int, shard_size: int) -> Iterable[tuple[int, int, slice]]:
    n_shards = max(1, math.ceil(n_items / shard_size))
    for k in range(n_shards):
        start = k * shard_size
        end = min(n_items, (k + 1) * shard_size)
        yield k, n_shards, slice(start, end)


def extract_activations(
    tokenized_pkl_path: str,
    model_name_or_path: str,
    output_root: str,
    layers: Sequence[int] | None = None,
    shard_size: int = 16,
    local_files_only: bool = False,
    use_gpu_for_forward: bool = True,
) -> None:
    """
    Extract MLP activations for specified layers and save to shards.

    Notes
    - Targets GPT-2–style models (module names like `transformer.h.{i}.mlp`).
    - TransformerLens is not used here; only HF forward (GPU allowed).
    - Saves shards under `{output_root}/layer_{i}/shard_{k}.pkl`.
    """
    with open(tokenized_pkl_path, "rb") as f:
        payload = pickle.load(f)

    input_ids: List[List[int]] = payload["input_ids"]
    attention_mask = payload.get("attention_mask")
    device = _device(use_gpu_for_forward)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True, local_files_only=local_files_only
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, local_files_only=local_files_only
    )
    model.eval().to(device)

    # Determine available MLP layers by name to be robust to max layer index
    mlp_modules = {name: module for name, module in model.named_modules() if name.endswith(".mlp")}
    # Map index -> name for GPT-2 style blocks
    index_to_name = {}
    for name in mlp_modules.keys():
        # Expect names like 'transformer.h.0.mlp'
        parts = name.split(".")
        try:
            idx = int(parts[parts.index("h") + 1])
            index_to_name[idx] = name
        except Exception:
            continue

    if layers is None:
        # Default: first 6 layers if available
        layers = [i for i in range(min(6, len(index_to_name)))]

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def run_batch(ids_batch: torch.Tensor, mask_batch: torch.Tensor | None):
        hooks = []
        activations = {}

        def make_hook(layer_idx: int):
            def hook_fn(module, inp, out):
                activations.setdefault(layer_idx, []).append(out.detach().cpu())

            return hook_fn

        # Register hooks for selected layers
        for idx in layers:
            name = index_to_name.get(int(idx))
            if name is None:
                continue
            module = dict(model.named_modules())[name]
            hooks.append(module.register_forward_hook(make_hook(int(idx))))

        _ = model(input_ids=ids_batch.to(device), attention_mask=(mask_batch.to(device) if mask_batch is not None else None))

        for h in hooks:
            h.remove()
        # Concatenate along batch dim for each layer
        for k in list(activations.keys()):
            activations[k] = torch.cat(activations[k], dim=0)
        return activations

    # Convert to tensors
    all_ids = torch.tensor(input_ids, dtype=torch.long)
    all_mask = torch.tensor(attention_mask, dtype=torch.long) if attention_mask is not None else None

    # Iterate shards and save per-layer files
    n_items = all_ids.shape[0]
    for shard_idx, n_shards, sl in _iter_shards(n_items, shard_size):
        ids_batch = all_ids[sl]
        mask_batch = all_mask[sl] if all_mask is not None else None
        acts = run_batch(ids_batch, mask_batch)

        for layer_idx, tensor in acts.items():
            layer_dir = output_root_path / f"layer_{layer_idx}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            out_file = layer_dir / f"shard_{shard_idx}.pkl"
            with out_file.open("wb") as f:
                pickle.dump({
                    "layer": layer_idx,
                    "shard": shard_idx,
                    "n_shards": n_shards,
                    "shape": tuple(tensor.shape),
                    "activations": tensor,
                }, f)


def run_extract_from_params(params_extract: dict):
    """Kedro wrapper to run extraction based on parameters.

    Returns a mapping of partition key to None so the PartitionedDataSet can
    pick up on-disk files without re-serializing large tensors.
    """
    extract_activations(**params_extract)
    root = Path(params_extract["output_root"])
    partitions = {}
    for pkl in root.rglob("*.pkl"):
        # Partition key should be relative path with forward slashes
        key = str(pkl.relative_to(root)).replace("\\", "/")
        partitions[key] = None
    return partitions
