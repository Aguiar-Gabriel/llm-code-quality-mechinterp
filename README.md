# llm-code-quality-mechinterp

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

This repository combines:
- A Kedro pipeline for mechanistic interpretability on code prompts (Feature Lens).
- The SonarvsLLM controlled datasets and instructions from Papers 1 and 2.

## Feature Lens (Kedro pipeline)

This is a Kedro project (generated with `kedro 1.0.0`) that builds prompts from the SonarvsLLM controlled scenarios, tokenizes inputs, extracts HF model activations, and joins Sonar metrics.

Basic usage:

```
pip install -r requirements.txt
kedro run
```

See `AGENTS.md` for repo-specific automation guidelines and `conf/base/parameters.yml` for parameters.

### Notes
- Keep credentials only under `conf/local/` (not committed).
- Do not commit data; the `data/**` paths are ignored except directories.
- Tests: `pytest -q` or `pytest -q tests/test_run.py`.

## SonarvsLLM (Papers 1 & 2)

This project has the classes analysed in the paper 1 (DOI [10.48550/arXiv.2408.07082](https://doi.org/10.48550/arXiv.2408.07082)) and paper 2 (DOI [XXX](https://doi.org/XXXXXXX.XXXXXXX)).

---
Paper 2 — Submitted to FSE 2025

Scope: Controlled quasi-experiment evaluating the responses from 8 LLMs and SonarQube to 3 code intervention scenarios.

Details:

Note to Revisors: The https://anonymous.4open.science/ website has a glitch. When you click on any folder link, it may hang. Refresh the page and toggle the folder tree to load contents.

The classes analysed in the second paper are under `sonarvsllm-testcases/src/main/resources/classFilesToBeAnalysed/controlled` with one subfolder per scenario.

LLM analysis classes:
- `sonarvsllm-testcases/src/main/java/science/com/master/sonar/SourceCodeLLM4o.java`
- `sonarvsllm-testcases/src/main/java/science/com/master/sonar/SourceCodeLLMAnthropicClaude.java`
- `sonarvsllm-testcases/src/main/java/science/com/master/sonar/SourceCodeLLMGoogleGemini.java`
- `sonarvsllm-testcases/src/main/java/science/com/master/sonar/SourceCodeLLMMetaLlama.java`

Dataset folders (LLM outputs at time of study):
- `sonarvsllm-testcases/src/main/resources/controlled/Claude3-haiku`
- `sonarvsllm-testcases/src/main/resources/controlled/Claude35-sonnet`
- `sonarvsllm-testcases/src/main/resources/controlled/GPT4o-mini`
- `sonarvsllm-testcases/src/main/resources/controlled/GPT4o`
- `sonarvsllm-testcases/src/main/resources/controlled/Gemini15flash`
- `sonarvsllm-testcases/src/main/resources/controlled/Gemini15pro`
- `sonarvsllm-testcases/src/main/resources/controlled/Llama3-8B`
- `sonarvsllm-testcases/src/main/resources/controlled/Llama31-405B`

Run Sonar for paper 2 (replace project key):

```
./mvnw verify org.sonarsource.scanner.maven:sonar-maven-plugin:sonar -Dsonar.projectKey=igorregis_sonarvsllm
```

---
Paper 1 — Comparative study of code readability using LLM vs SonarQube.

Classes analysed:
- `sonarvsllm-testcases/src/main/resources/classFilesToBeAnalysed/quarkus`
- `sonarvsllm-testcases/src/main/resources/classFilesToBeAnalysed/shattered-pixel-dungeon/core/src/main/java/com/shatteredpixel/shatteredpixeldungeon`

LLM analysis classes:
- `sonarvsllm-testcases/src/main/java/science/com/master/sonar/SourceCodeLLM.java`
- `sonarvsllm-testcases/src/main/java/science/com/master/sonar/SourceCodeLLM4o.java`

JSON datasets (LLM + SonarQube at time of execution):
- `sonarAndLLM35Quarkus.json`
- `sonarAndLLM35Shatteredpixeldungeon.json`
- `sonarAndLLM4oQuarkus.json`
- `sonarAndLLM4oShatteredpixeldungeon.json`

They have been preserved, since SonarQube analysis is updated alongside the respective projects.
