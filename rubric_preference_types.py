"""
Types for Prometheus-Eval that takes in a rubric, two model outputs, and an optional reference
to return reasoning and pairwise preference.

Reference: https://github.com/prometheus-eval/prometheus-eval
"""

import logging
from dataclasses import dataclass

import torch
from tinker import SamplingClient, types
from tinker_cookbook import renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.tokenizer_utils import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class PrometheusEvalComparison:
    prompt_conversation: list[renderers.Message]
    completion_A: list[renderers.Message]
    completion_B: list[renderers.Message]
    rubric: str
    reference: str | None = None

    def swap(self) -> "PrometheusEvalComparison":
        return PrometheusEvalComparison(
            prompt_conversation=self.prompt_conversation,
            completion_A=self.completion_B,
            completion_B=self.completion_A,
            rubric=self.rubric,
            reference=self.reference,
        )


@dataclass
class LabeledPrometheusEvalComparison:
    comparison: PrometheusEvalComparison
    label: str  # "{reasoning} [RESULT] (Either "A" or "B")"


class PrometheusEvalPreferenceModel:
    async def __call__(self, comparison: PrometheusEvalComparison) -> float:
        """
        -1: A is strongly preferred
        0: Tie
        1: B is strongly preferred

        Caveat: Prometheus-eval training data do not include examples that are tied.
        """
        raise NotImplementedError


class PrometheusEvalComparisonRenderer:
    def build_generation_prompt(self, comparison: PrometheusEvalComparison) -> types.ModelInput:
        raise NotImplementedError

    def to_tokens_weights(
        self, labeled_comparison: LabeledPrometheusEvalComparison
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def tokenizer(self) -> Tokenizer:
        raise NotImplementedError


class PrometheusEvalComparisonRendererFromChatRenderer(PrometheusEvalComparisonRenderer):
    def __init__(self, convo_renderer: renderers.Renderer):
        self.convo_renderer = convo_renderer

    def _comparison_to_convo(self, comparison: PrometheusEvalComparison) -> list[renderers.Message]:
        comparison_prompt_template = """\
You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort.

###Task Description:
An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), a reference answer, and an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.
2. Make comparisons between Response A, Response B, and the Reference Answer. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.
3. After writing the feedback, indicate the better response, either "A" or "B".
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"
5. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response A to evaluate:
{completion_A}

###Response B to evaluate:
{completion_B}

###Reference Answer:
{reference}

###Evaluation Criteria:
{rubric}

###Feedback:
"""
        instruction = ""
        for msg in comparison.prompt_conversation:
            if msg["role"] == "user" or msg["role"] == "assistant":
                instruction += f"{msg['role']}: {msg['content']}\n"
        comparison_prompt = comparison_prompt_template.format(
            instruction=instruction,
            completion_A=comparison.completion_A[0]["content"],
            completion_B=comparison.completion_B[0]["content"],
            reference=comparison.reference if comparison.reference else "None",
            rubric=comparison.rubric,
        )
        return [{"role": "user", "content": comparison_prompt}]

    def build_generation_prompt(self, comparison: PrometheusEvalComparison) -> types.ModelInput:
        return self.convo_renderer.build_generation_prompt(self._comparison_to_convo(comparison))

    def to_tokens_weights(
        self, labeled_comparison: LabeledPrometheusEvalComparison
    ) -> tuple[torch.Tensor, torch.Tensor]:
        convo = self._comparison_to_convo(labeled_comparison.comparison)
        convo_with_pref = convo + [{"role": "assistant", "content": labeled_comparison.label}]
        tokens, weights = self.convo_renderer.build_supervised_example(
            convo_with_pref, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
        return tokens, weights

    @property
    def tokenizer(self) -> Tokenizer:
        return self.convo_renderer.tokenizer


class PrometheusEvalPreferenceModelFromChatRenderer(PrometheusEvalPreferenceModel):
    def __init__(self, convo_renderer: renderers.Renderer, sampling_client: SamplingClient):
        self.comparison_renderer = PrometheusEvalComparisonRendererFromChatRenderer(convo_renderer)
        self.sampling_client = sampling_client

    async def __call__(self, comparison: PrometheusEvalComparison) -> float:
        pm_input = self.comparison_renderer.build_generation_prompt(comparison)
        response = await self.sampling_client.sample_async(
            pm_input,
            num_samples=1,
            sampling_params=types.SamplingParams(
                temperature=0.0,
                max_tokens=1024,
                stop=self.comparison_renderer.convo_renderer.get_stop_sequences(),
            ),
        )
        str_output = self.comparison_renderer.tokenizer.decode(response.sequences[0].tokens).strip()
        # Expected output format: "{feedback} [RESULT] (Either "A" or "B")"
        if "[RESULT]" not in str_output:
            logger.warning(f"Invalid output preference model output: '{str_output}'")
            return 0.0
        result = str_output.split("[RESULT]")[-1]
        if "A" in result:
            return -1.0
        elif "B" in result:
            return 1.0
        else:
            logger.warning(f"Invalid output preference model output: '{str_output}'")
            return 0.0
