# src/feature_lens/pipelines/tokenize/nodes.py
from __future__ import annotations
import pickle
import pandas as pd
from transformers import AutoTokenizer

def tokenize_prompts(
    prompts_csv_path: str,
    model_name_or_path: str,
    max_length: int,
    output_pkl_path: str,
    local_files_only: bool = False,
    pad_to_max: bool = True,
    truncate: bool = True,
    pad_token_to_eos: bool = True,
) -> None:
    # 1) carregar prompts
    df = pd.read_csv(prompts_csv_path)
    # escolha da coluna de texto: priorize java_code; fallback se necessário
    for col in ["java_code", "text", "prompt"]:
        if col in df.columns:
            texts = df[col].astype(str).tolist()
            break
    else:
        # se não achar, usa a primeira coluna textual
        texts = df.iloc[:, 0].astype(str).tolist()

    # 2) tokenizer
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    if pad_token_to_eos and tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    enc = tok(
        texts,
        max_length=max_length,
        padding="max_length" if pad_to_max else False,
        truncation=truncate,
        return_attention_mask=True,
        return_tensors=None,
    )

    # 3) salvar
    with open(output_pkl_path, "wb") as f:
        pickle.dump(enc, f)


def run_tokenize_from_params(params_tokenize: dict):
    """
    Lê params:tokenize, chama tokenize_prompts e retorna o objeto
    para o catalog persistir em prompts_tokenized.
    """
    # Opcional: filtrar apenas chaves esperadas, para evitar crash se vier algo extra no YAML
    allowed = {
        "prompts_csv_path",
        "model_name_or_path",
        "max_length",
        "output_pkl_path",
        "local_files_only",
        "pad_to_max",
        "truncate",
        "pad_token_to_eos",
    }
    safe_kwargs = {k: v for k, v in params_tokenize.items() if k in allowed}

    tokenize_prompts(**safe_kwargs)

    import pickle as _p
    with open(params_tokenize["output_pkl_path"], "rb") as f:
        return _p.load(f)
