# src/feature_lens/pipelines/extract_activations/nodes.py
from __future__ import annotations
from pathlib import Path
import pickle
from typing import Iterable, List, Dict, Any
from typing import Dict, Optional
import logging
import torch
from transformers import AutoModelForCausalLM

def _batch_iter(n: int, batch_size: int) -> Iterable[range]:
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        yield range(i, j)
        i = j

def extract_activations(
    tokenized_pkl_path: str,
    model_name_or_path: str,
    output_root: str,
    layers: List[int],
    shard_size: int = 16,
    batch_size: int = 8,                 # << agora aceitamos batch_size
    local_files_only: bool = False,
    use_gpu_for_forward: bool = True,
    torch_dtype: str = "float32",
    device_mode: str | None = None,      # "auto" | "cuda" | "cpu"
    allow_cpu_fallback: bool = True,
) -> None:
    """
    Lê o pickle tokenizado, faz forward e salva ativações por camada e shard:
      {output_root}/layer_{L}/shard_{K}.pkl
    Cada shard contém um dict com chaves:
      - "indices": lista de índices das amostras naquele shard
      - "activations": tensor ou lista com ativações daquela camada
    """
    # 1) carregar tokens
    with open(tokenized_pkl_path, "rb") as f:
        enc = pickle.load(f)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)
    n_samples = len(input_ids)

    # 2) modelo e device
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float32)
    # Log de ambiente CUDA/Torch
    try:
        logging.info(
            "env: torch=%s, torch.cuda=%s, cuda_available=%s, devices=%s",
            torch.__version__,
            getattr(torch.version, "cuda", None),
            torch.cuda.is_available(),
            torch.cuda.device_count(),
        )
    except Exception:
        pass

    # Seleção de device
    mode = (device_mode or "auto").lower()
    if mode not in {"auto", "cuda", "cpu"}:
        mode = "auto"
    if mode == "cpu":
        device = torch.device("cpu")
    elif mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device_mode=cuda, mas CUDA não está disponível nesta máquina.")
        device = torch.device("cuda")
    else:  # auto (backcompat com use_gpu_for_forward)
        device = torch.device("cuda") if (use_gpu_for_forward and torch.cuda.is_available()) else torch.device("cpu")
    # Evita dtype incompatível no CPU (ex.: float16)
    if device.type == "cpu" and dtype is torch.float16:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    # Move para device com fallback seguro se CUDA falhar neste ambiente
    try:
        model = model.to(device)
    except Exception as e:
        if device.type == "cuda" and allow_cpu_fallback:
            logging.warning(f"Falha ao mover modelo para CUDA: {e}. Fazendo fallback para CPU.")
            device = torch.device("cpu")
            if dtype is torch.float16:
                dtype = torch.float32
            model = model.to(device)
        else:
            logging.error("Erro ao mover modelo para device=%s: %s", device, e)
            raise
    model.eval()

    gpu_name = None
    if device.type == "cuda":
        try:
            idx = device.index if device.index is not None else 0
            gpu_name = torch.cuda.get_device_name(idx)
        except Exception:
            gpu_name = "unknown"
    logging.info(
        "extract_activations: device=%s%s, dtype=%s, shard_size=%s, batch_size=%s, layers=%s, n_samples=%s",
        device,
        f"({gpu_name})" if gpu_name else "",
        str(dtype).replace("torch.", ""),
        shard_size,
        batch_size,
        layers,
        n_samples,
    )

    # sanity de camadas
    n_layer = getattr(getattr(model, "transformer", model), "n_layer", None)
    if n_layer is None and hasattr(model.config, "n_layer"):
        n_layer = model.config.n_layer
    if n_layer is not None:
        assert max(layers) < n_layer, f"layers={layers} excede n_layer={n_layer} para {model_name_or_path}"

    # 3) hooks para capturar MLP de cada bloco
    captured: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}
    handles = []

    # tenta GPT-2 style: model.transformer.h[L].mlp
    blocks = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers  # fallback arquiteturas diferentes

    if blocks is None:
        raise RuntimeError("Não encontrei blocos de Transformer no modelo para registrar hooks.")

    def _make_hook(layer_idx: int):
        def hook(module, inp, out):
            # out: [batch, seq, hidden] ou similar; guardamos como está
            captured[layer_idx].append(out.detach().to("cpu"))
        return hook

    for L in layers:
        mlp = None
        blk = blocks[L]
        # tentativas comuns de submódulo MLP
        for attr in ["mlp", "mlp.c_fc", "ffn", "feed_forward", "mlp.act"]:
            try:
                mlp = eval("blk." + attr)
                if mlp is not None:
                    break
            except Exception:
                continue
        if mlp is None:
            # se não achar submódulo, pendura no próprio bloco
            mlp = blk
        handles.append(mlp.register_forward_hook(_make_hook(L)))

    # 4) forward em lotes e shards
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for L in layers:
        (out_root / f"layer_{L}").mkdir(parents=True, exist_ok=True)

    idx_all = list(range(n_samples))
    shard_id = 0
    with torch.no_grad():
        for shard_start in range(0, n_samples, shard_size):
            shard_end = min(shard_start + shard_size, n_samples)
            shard_indices = idx_all[shard_start:shard_end]

            # zera buffers de captura para este shard
            for L in layers:
                captured[L].clear()

            # processa este shard em batches menores
            for batch_idx in _batch_iter(len(shard_indices), batch_size):
                sel = [shard_indices[k] for k in batch_idx]
                ids = torch.tensor([input_ids[i] for i in sel], dtype=torch.long, device=device)
                att = None
                if attention_mask is not None:
                    att = torch.tensor([attention_mask[i] for i in sel], dtype=torch.long, device=device)
                try:
                    _ = model(input_ids=ids, attention_mask=att)
                except RuntimeError as re:
                    msg = str(re)
                    logging.error("Erro no forward (device=%s, dtype=%s): %s", device, dtype, msg)
                    if "CUDA out of memory" in msg or "c10::Error" in msg:
                        logging.error("Possível OOM: reduza batch_size/shard_size ou max_length.")
                    raise

            # salva cada camada deste shard
            for L in layers:
                # concatena capturas ao longo de batches
                if len(captured[L]) == 0:
                    acts_cat = None
                else:
                    try:
                        acts_cat = torch.cat(captured[L], dim=0)  # [shard_size, seq, hidden]
                    except Exception:
                        # se forem listas não concatenáveis, salva a lista
                        acts_cat = captured[L]
                payload = {
                    "indices": shard_indices,
                    "activations": acts_cat,
                }
                p = out_root / f"layer_{L}" / f"shard_{shard_id}.pkl"
                with open(p, "wb") as f:
                    pickle.dump(payload, f)
                if isinstance(acts_cat, torch.Tensor):
                    logging.info(
                        "salvo shard=%s layer=%s shape=%s em %s",
                        shard_id,
                        L,
                        tuple(acts_cat.shape),
                        str(p),
                    )

            shard_id += 1

    # limpa hooks
    for h in handles:
        h.remove()


def run_extract_from_params(params_extract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executa a extração e retorna um mapeamento {chave_partição: payload}
    para o PartitionedDataset salvar sem reclamar.
    """
    allowed = {
        "tokenized_pkl_path", "model_name_or_path", "output_root",
        "layers", "shard_size", "batch_size", "local_files_only",
        "use_gpu_for_forward", "torch_dtype", "device_mode", "allow_cpu_fallback"
    }
    safe = {k: v for k, v in params_extract.items() if k in allowed}
    extract_activations(**safe)

    root = Path(params_extract["output_root"])
    partitions: Dict[str, Any] = {}
    for pkl in root.rglob("*.pkl"):
        key = str(pkl.relative_to(root))  # ex: layer_0/shard_0.pkl
        with open(pkl, "rb") as f:
            partitions[key] = pickle.load(f)  # <- devolve o payload
    return partitions
