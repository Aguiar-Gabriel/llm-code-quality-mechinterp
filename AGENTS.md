# Repository Guidelines

## Project Structure & Module Organization
- `src/feature_lens/`: Kedro package with pipelines (collect_prompts, tokenize, extract_activations, metrics) and settings.
- `data/01_raw` … `data/08_reporting/`: Data lifecycle directories (write only under `data/**`).
- `conf/base/`: Parameters and catalog; edit `parameters.yml` for configurable options.
- `tests/`: Pytest suite (e.g., `tests/test_run.py`).
- External test Java sources live in local `sonarvsllm` repo (read-only here).

## Build, Test, and Development Commands
- Activate venv (WSL): `source ../.venv/bin/activate`
- Run pipeline: `kedro run`
- Visualize DAG: `kedro viz`
- Run tests: `pytest -q`
- Lint/format (Ruff+Black via Ruff): `ruff check .` and `ruff format .`
- Type check (optional): `pyright` or `mypy` if installed.

## Coding Style & Naming Conventions
- Python 3.10, Black formatting (via Ruff), import order enforced.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants.
- Prefer type hints and concise docstrings; keep modules focused and small.
- Do not write files outside `data/**`. Log key paths/params to MLflow when applicable.

## Testing Guidelines
- Framework: Pytest. Name tests `test_*.py`; functions `test_*`.
- Keep tests deterministic; mock file system/network where needed.
- Run `pytest -q tests/test_run.py` for the minimal smoke path; add unit tests near changed code.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject (≤72 chars), optional scope, descriptive body when needed.
- Reference issues with `#<id>` and summarize user-visible changes.
- PRs: include purpose, screenshots/logs for UX/CLI changes, steps to reproduce, and any config/data impacts.
- Keep diffs minimal; avoid drive-by refactors unrelated to the change.

## Security & Configuration Tips
- Secrets/config: use `conf/local/` for machine-specific overrides; never commit secrets.
- GPU usage: HF model forwards may use GPU; TransformerLens must run on CPU.
- Large runs: start with small shards; validate outputs before scaling.

## Agent-Specific Instructions
- Run Codex locally from repo root (venv active): `codex`.
- Codex merges this `AGENTS.md` with any global `~/.codex/AGENTS.md` (top-down). Keep this file authoritative for repo-specific rules.


Contexto do projeto

Você está dentro de um projeto de interpretabilidade voltado para código. O objetivo imediato é:

Ler classes Java dos cenários controlados do sonarvsllm.

Montar um CSV data/01_raw/prompts.csv com colunas scenario, class_path, java_code, system_prompt, user_prompt.

Tokenizar e extrair ativações com Hugging Face.

Anexar métricas do Sonar por classe para futuras análises.
Não use saídas de modelos pré-existentes como Claude3-haiku, GPT4o etc. Estes diretórios são outputs do estudo e não fazem parte do insumo. 
GitHub
OpenAI
OpenAI Help Center

Estrutura local esperada

Kedro com pastas data/01_raw até data/08_reporting já criadas.

Ambiente roda em WSL2 com VS Code Remote WSL. Python 3.10.

Venv em ../.venv relativo ao diretório do projeto. Ative com source ../.venv/bin/activate.

GPU GTX 1050 Ti disponível, porém TransformerLens roda no CPU. Use GPU só para forward de modelos HF.

Repositório sonarvsllm disponível localmente para leitura dos .java e execução do Sonar quando necessário.
(Se precisar de caminhos UNC no Windows, abrir por explorer.exe . ou \\wsl$\Ubuntu-22.04\....)

O que fazer primeiro

Criar o dataset bruto de prompts:

Varrer sonarvsllm-testcases/src/main/resources/classFilesToBeAnalysed/controlled/**/**/*.java.

scenario é o nome do subdiretório direto sob controlled.

class_path é o caminho relativo ao repo sonarvsllm.

java_code é o conteúdo do arquivo.

system_prompt e user_prompt devem seguir o texto canônico descrito no paper para avaliação de código.

Salvar CSV em data/01_raw/prompts.csv.
Fontes de referência de onde estão os arquivos e classes utilitárias: 
GitHub

Tokenizar prompts:

Seq length 128. pad_token igual a eos. Persistir em data/02_intermediate/prompts_tokenized.pkl.

Extrair ativações:

Hooks no MLP layers=[0..5].

Salvar shards em data/03_primary/activations/layer_{i}/shard_{k}.pkl.

Ajustar n_prompts e shard_size se memória apertar.

Métricas do Sonar:

Opção A: rodar Sonar via Maven no repo de testes e exportar métricas por classe.

Opção B: usar arquivos de referência do Paper 1 apenas para schema e checagem de campos.

Consolidar em data/02_intermediate/sonar_metrics.jsonl com class_path como chave.
Instrução de execução do Sonar está documentada no README do projeto de testes. 
GitHub

Comandos para eu executar

Ativar venv no WSL:

source ../.venv/bin/activate


Rodar o pipeline mínimo:

kedro run


Visualizar DAG:

kedro viz

Tarefas que você deve executar agora

Implementar nó collect_prompts em src/feature_lens/pipelines/collect_prompts/:

Função Python deve caminhar pelos .java, montar as linhas e persistir CSV.

System prompt e user prompt devem vir de constantes no módulo settings/prompts.py.

Implementar tokenize em pipelines/tokenize/.

Implementar extract_activations em pipelines/extract_activations/ com hooks no MLP.

Implementar join_with_sonar em pipelines/metrics/ para agregar sonar_metrics.jsonl por class_path.

Regras de estilo e qualidade

Python formatado com Ruff e Black.

Nada de escrever arquivos fora de data/**.

Parametrizações no conf/base/parameters.yml.

Logar caminhos chave no MLflow quando aplicável.

Limitações e precauções

TransformerLens deve rodar no CPU por restrições da GTX 1050 Ti.

Evite processar todos os .java de uma vez antes de validar um shard pequeno.

Não consumir pastas de outputs de LLM de terceiros.

Como eu testo que fiz o certo

tests/test_run.py deve passar.

Após kedro run, conferir:

data/01_raw/prompts.csv existe e tem colunas corretas.

data/02_intermediate/prompts_tokenized.pkl criado.

data/03_primary/activations/layer_*/shard_*.pkl criados.

data/02_intermediate/sonar_metrics.jsonl com pelo menos um registro válido por classe processada.

Pronto para Codex

Você, Codex, deve priorizar este AGENTS.md do repo. Além disso pode existir um ~/.codex/AGENTS.md com regras globais. Faça merge top-down como documentado.