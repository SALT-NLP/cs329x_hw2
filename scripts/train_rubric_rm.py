import asyncio

from dotenv import load_dotenv
from typing import cast

import chz
from datasets import Dataset, load_dataset
from tinker import types
from tinker_cookbook import cli_utils
from inkling_renderer import get_renderer_name
from rubric_preference_types import (
    LabeledPrometheusEvalComparison,
    PrometheusEvalComparison,
    PrometheusEvalComparisonRenderer,
    PrometheusEvalComparisonRendererFromChatRenderer,
)
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig


@chz.chz
class PrometheusV2PreferenceDataBuilder(ChatDatasetBuilder):
    @property
    def comparison_renderer(self) -> PrometheusEvalComparisonRenderer:
        return PrometheusEvalComparisonRendererFromChatRenderer(self.renderer)

    def __call__(
        self,
    ) -> tuple[SupervisedDatasetFromHFDataset, SupervisedDatasetFromHFDataset | None]:
        dataset = load_dataset("kaist-ai/Preference-Collection", split="train")
        dataset = cast(Dataset, dataset)
        test_ds = dataset.take(1024)
        train_ds = dataset.skip(1024)

        def comparison_to_datum(labeled_comparison: LabeledPrometheusEvalComparison) -> types.Datum:
            model_input, weights = self.comparison_renderer.to_tokens_weights(labeled_comparison)
            return datum_from_model_input_weights(model_input, weights, self.common_config.max_length)

        def example_to_data(example: dict[str, str]) -> list[types.Datum]:
            labeled_comparison = LabeledPrometheusEvalComparison(
                comparison=PrometheusEvalComparison(
                    prompt_conversation=[{"role": "user", "content": example["orig_instruction"]}],
                    completion_A=[{"role": "assistant", "content": example["orig_response_A"]}],
                    completion_B=[{"role": "assistant", "content": example["orig_response_B"]}],
                    rubric=example["orig_criteria"],
                    reference=example["orig_reference_answer"],
                ),
                label=example["output"],
            )
            return [comparison_to_datum(labeled_comparison)]

        return SupervisedDatasetFromHFDataset(
            train_ds,
            batch_size=self.common_config.batch_size,
            flatmap_fn=example_to_data,
        ), SupervisedDatasetFromHFDataset(
            test_ds,
            batch_size=self.common_config.batch_size,
            flatmap_fn=example_to_data,
        )


def build_config(
    model_name: str,
    log_path: str,
    max_length: int,
    learning_rate: float,
    batch_size: int,
    eval_every: int,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> train.Config:
    renderer_name = get_renderer_name(model_name)  # disables thinking for Inkling
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = PrometheusV2PreferenceDataBuilder(common_config=common_config)
    return train.Config(
        log_path=log_path,
        model_name=model_name,
        recipe_name="cs329x_hw2_rubric_rm",
        dataset_builder=dataset,
        learning_rate=learning_rate,
        lr_schedule="linear",
        num_epochs=1,
        eval_every=eval_every,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )


def main(
    model_name: str = "thinkingmachines/Inkling-Small",
    log_path: str = "results/prometheus_v2_preference_rm",
    max_length: int = 32768,
    learning_rate: float = 1e-4,
    batch_size: int = 512,
    eval_every: int = 200,
    wandb_project: str = "tinker_personalization",
    wandb_name: str = "prometheus_v2_preference_rm_inkling_small",
):
    load_dotenv()
    config = build_config(
        model_name=model_name,
        log_path=log_path,
        max_length=max_length,
        learning_rate=learning_rate,
        batch_size=batch_size,
        eval_every=eval_every,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    chz.entrypoint(main, allow_hyphens=True)
