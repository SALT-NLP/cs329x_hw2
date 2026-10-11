from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def build_config() -> train.Config:
    # TODO: Implement this function to set up SFT config.
    # Check out https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/sl_basic.py for example.
    # Use `get_renderer_name(model_name)` from inkling_renderer.py (instead of model_info.get_recommended_renderer_name)
    # for the renderer so that thinking is disabled for Inkling models.
    pass
    

def main():
    load_dotenv()
    config = build_config()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    train.main(config)


if __name__ == "__main__":
    main()