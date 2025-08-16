#!/usr/bin/env python
from __future__ import annotations
import json
import os
import torch

def main():
    info = {
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_is_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    try:
        if torch.cuda.is_available():
            info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
            try:
                x = torch.randn(1, device="cuda")
                info["cuda_tensor_alloc"] = True
                info["cuda_tensor_device"] = str(x.device)
            except Exception as e:
                info["cuda_tensor_alloc"] = False
                info["cuda_tensor_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        info["cuda_query_error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()

