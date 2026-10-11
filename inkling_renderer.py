"""
Inkling renderer with thinking disabled.

Inkling models are conditioned on a "Thinking effort level" system message (default 0.9).
This module registers a renderer that uses the lowest effort (0.0) for both sampling and
SFT, so the policy answers directly instead of reasoning first. Use `get_renderer_name`
wherever you would call `model_info.get_recommended_renderer_name` for the policy model.
"""

from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer

NO_THINKING_EFFORT = 0.0
INKLING_NO_THINKING_RENDERER = "cs329x/tml_v0_no_thinking"


class TmlV0NoThinkingRenderer(TmlV0Renderer):
    """TmlV0Renderer whose default thinking effort is 0.0 instead of 0.9."""

    def build_generation_prompt(self, messages, role="assistant", prefill=None, effort=NO_THINKING_EFFORT):
        return super().build_generation_prompt(messages, role=role, prefill=prefill, effort=effort)

    def build_supervised_examples(
        self, messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES, effort=NO_THINKING_EFFORT
    ):
        return super().build_supervised_examples(messages, train_on_what, effort=effort)

    def build_supervised_example(
        self, messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES, effort=NO_THINKING_EFFORT
    ):
        return super().build_supervised_example(messages, train_on_what, effort=effort)


renderers.register_renderer(
    INKLING_NO_THINKING_RENDERER,
    lambda tokenizer, image_processor=None: TmlV0NoThinkingRenderer(tokenizer),
)


def get_renderer_name(model_name: str) -> str:
    """Renderer name to use for `model_name`, with thinking disabled for Inkling models."""
    if model_name.startswith("thinkingmachines/Inkling"):
        return INKLING_NO_THINKING_RENDERER
    return model_info.get_recommended_renderer_name(model_name)
