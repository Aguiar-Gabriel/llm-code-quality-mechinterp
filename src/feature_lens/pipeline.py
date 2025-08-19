from __future__ import annotations

from kedro.pipeline import Pipeline, node

from feature_lens.pipelines.collect_prompts.nodes import run_collect_from_params
from feature_lens.pipelines.tokenize.nodes import run_tokenize_from_params
from feature_lens.pipelines.extract_activations.nodes import run_extract_from_params
from feature_lens.pipelines.metrics.nodes import run_join_from_params, run_generate_from_params


def collect_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=run_collect_from_params,
            inputs={
                "params_collect_prompts": "params:collect_prompts",
                "params_collect": "params:collect",
            },
            outputs=None,
            name="collect_prompts_node",
        )
    ])


def tokenize_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=run_tokenize_from_params,
            inputs={"params_tokenize": "params:tokenize"},
            outputs="prompts_tokenized",
            name="tokenize_prompts_node",
        )
    ])


def activations_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=run_extract_from_params,
            inputs={"params_extract": "params:extract_activations"},
            outputs="activations",
            name="extract_activations_node",
        )
    ])


def metrics_pipeline() -> Pipeline:
    return Pipeline([
        # Optional generator: controlled by params:metrics_generate.mode
        node(
            func=run_generate_from_params,
            inputs={"params": "params:metrics_generate"},
            outputs="metrics_generated",
            name="maybe_generate_sonar_metrics_node",
        ),
        node(
            func=run_join_from_params,
            inputs={
                "params_metrics": "params:metrics",
                "ready": "metrics_generated",
            },
            outputs="sonar_metrics",
            name="join_with_sonar_node",
        ),
    ])


def all_pipeline() -> Pipeline:
    # Chain all named pipelines in execution order
    return collect_pipeline() + tokenize_pipeline() + activations_pipeline() + metrics_pipeline()


def register_pipelines() -> dict[str, Pipeline]:
    # Return named pipelines with __default__ for convenience
    return {
        "__default__": all_pipeline(),
        "collect": collect_pipeline(),
        "tokenize": tokenize_pipeline(),
        "activations": activations_pipeline(),
        "metrics": metrics_pipeline(),
        "all": all_pipeline(),
    }
