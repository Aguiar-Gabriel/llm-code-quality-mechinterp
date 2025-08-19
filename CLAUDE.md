# CLAUDE.md - Memória do Projeto Feature-Lens

## Contexto
Repositório: interpretabilidade de LLMs com GPT-2 e Kedro. Extrair ativações, gerar outputs, aplicar técnicas de interpretabilidade e registrar métricas de execução. Infra simples e reprodutível. Foco em rigor e clareza.

## Objetivos
1. Integrar GPT-2 nos pipelines para geração real e extração de ativações.
2. Aplicar interpretabilidade baseada em SHAP e feature-lens sobre as ativações e/ou tokens.
3. Instrumentar uso de GPU com fallback para CPU, mantendo tensors e ativações no mesmo device.
4. Popular `data/02_intermediate/sonar_metrics.jsonl` com métricas de execução e adicionar `vram_alloc_mb` e `vram_reserved_mb`.
5. Deixar tudo rodável via `kedro run --pipeline all` com logs claros de device e VRAM.

## Requisitos técnicos e versões alvo
- torch 2.1.2+cu118
- transformers 4.38.2
- tokenizers 0.15.2
- accelerate 0.26.1
- numpy 1.26.4
- GPU local disponível: NVIDIA GTX 1050 Ti
- Ambiente Linux/WSL com Python 3.10 e Kedro

## Padrões de device
- Env flag `USE_GPU` controla preferência por CUDA. Default 1.
- Seleção de device por arquivo que roda forward
  - `use_gpu = os.getenv("USE_GPU", "1") == "1"`
  - `device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")`
  - dtype: `torch.float16` quando `device.type == "cuda"`, senão `torch.float32`
- Após `from_pretrained`, usar `model.to(device)` e `model.eval()`
- Após tokenizar, mover inputs: `enc = {k: v.to(device) for k, v in enc.items()}`
- Converter para CPU apenas na hora de salvar em disco

## Instrumentação obrigatória
- Após cada `forward` ou `generate`, se `device.type == "cuda"` imprimir linha única
  - `[GPU] device={name} allocMB={allocated_mb:.1f} reservedMB={reserved_mb:.1f}`
- Em métricas JSONL acrescentar
  - `vram_alloc_mb` e `vram_reserved_mb` quando CUDA ativo, senão `null`
- Logar também `model_device={device}`, `dtype={dtype}` no início do node

## Locais a refatorar
- `src/feature_lens/pipelines/extract_activations/nodes.py`
  - Importar `os, torch`
  - Definir `use_gpu` e `device` no início da função
  - Passar `torch_dtype` condicional ao `from_pretrained`
  - `model.to(device); model.eval()`
  - Inputs e ativações no mesmo device
  - `[GPU]` logging após cada forward
  - Converter ativações para CPU somente ao serializar
- `src/feature_lens/pipelines/metrics/nodes.py`
  - Acrescentar campos de VRAM ao objeto antes de escrever no JSONL
- Qualquer outro script que faça forward deve seguir o mesmo padrão

## Status atual do código
- `src/feature_lens/pipelines/extract_activations/nodes.py:162-169` já tem logs GPU básicos
- Linhas 54-64 têm código duplicado de device/dtype
- Existem arquivos .pkl duplicados no diretório de ativações (layer_0/shard_0.pkl.pkl.pkl...)
- PyTorch não está instalado no ambiente atual

## Problemas identificados
1. Código duplicado nas linhas 54-64 (device e dtype definidos duas vezes)
2. Arquivos pickle duplicados sugerem bug no salvamento
3. Logs GPU já implementados mas podem estar melhorados
4. Dependências não instaladas

## Critérios de aceite
- Executar `USE_GPU=1 kedro run --pipeline all` e ver linhas `[GPU]` durante o `extract_activations_node`
- `sonar_metrics.jsonl` contém para cada amostra `vram_alloc_mb` e `vram_reserved_mb`
- Não há `.cpu()` em caminhos críticos antes do salvamento
- Rodar com `USE_GPU=0` funciona sem erros e sem logs `[GPU]`
- Tempo total reduzido quando GPU está ativa em relação ao CPU

## Comandos úteis
- Rodar tudo com GPU
  - `USE_GPU=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 kedro run --pipeline all`
- Rodar só extração de ativações
  - `USE_GPU=1 kedro run --from-nodes extract_activations_node`
- Verificar CUDA rápido
  - `python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.version.cuda); import time; import torch as t; x=t.randn(4096,4096,device='cuda',dtype=t.float16); y=x@x; t.cuda.synchronize(); print('ok')"`

## Estilo de contribuição para a CLI do Claude
- Produzir patches mínimos que não alterem hiperparâmetros
- Preservar assinaturas de funções e contratos do Kedro
- Se `USE_GPU=1` e `cuda` indisponível, levantar `RuntimeError` com mensagem clara
- Não adicionar dependências novas
- Incluir mensagens de commit descritivas em PT-BR e pequenos testes manuais

## Próximos passos
1. Limpar código duplicado em `extract_activations/nodes.py`
2. Verificar e corrigir bug de arquivos pickle duplicados
3. Garantir logs `[GPU]` aparecendo corretamente
4. Enriquecer `sonar_metrics.jsonl` com VRAM
5. Instalar dependências se necessário
6. Integrar SHAP/feature-lens sobre os outputs/ativações
7. Criar relatório de progresso por pipeline

## Hardware disponível
- NVIDIA GTX 1050 Ti (confirmado via contexto do usuário)
- Ambiente WSL2/Linux
- Python 3.10