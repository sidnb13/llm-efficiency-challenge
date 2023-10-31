from absl import app, flags
from ml_collections import config_flags

from train_harness.core import train

# define configuration file
_CONFIG = config_flags.DEFINE_config_file(
    "config",
    "config.py",
    "Config file for training.",
    lock_config=True,
)

USE_WANDB = flags.DEFINE_bool(
    "use_wandb",
    True,
    "Whether to use wandb for logging.",
)

WANDB_PROJECT = flags.DEFINE_string(
    "wandb_project",
    "finn-experiments",
    "Name of wandb project.",
)

WANDB_ENTITY = flags.DEFINE_string(
    "wandb_entity",
    "finnd",
    "Name of wandb entity.",
)

WANDB_NOTES = flags.DEFINE_string(
    "wandb_notes",
    None,
    "Notes for wandb.",
)

RUN_ID = flags.DEFINE_string(
    "run_id",
    None,
    "Run id for wandb.",
)

WANDB_TAGS = flags.DEFINE_list(
    "wandb_tags",
    None,
    "Tags for wandb.",
)

DATASET_PATH = flags.DEFINE_list(
    "dataset_path",
    "data/dolly-15k-packed",
    # "data/databricks___databricks-dolly-15k/default-edf378b23813c52d/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96/",
    "Path to dataset.",
)

WANDB_RUN_NAME = flags.DEFINE_string(
    "wandb_run_name",
    None,
    "Name of wandb run.",
)

CHECKPT_PATH = flags.DEFINE_string(
    "checkpt_path",
    None,
    "Path to checkpoint.",
)

INSTRUCTION_COLS = flags.DEFINE_list(
    "instruction_cols",
    None,
    "Instruction columns.",
)
INPUT_COLS = flags.DEFINE_list(
    "input_cols",
    None,
    "Input columns.",
)
OUTPUT_COLS = flags.DEFINE_list(
    "output_cols",
    None,
    "Output columns.",
)


def main(argv):
    del argv
    cfg = _CONFIG.value
    print(f"cfg['training_config']: \n{cfg['training_config']}\n")   
    train(
        cfg["training_config"],
        cfg["data_config"],
        cfg["lora_config"],
        dataset_path=DATASET_PATH.value,
        wandb_project=WANDB_PROJECT.value,
        wandb_entity=WANDB_ENTITY.value,
        wandb_enabled=USE_WANDB.value,
        wandb_run_name=WANDB_RUN_NAME.value,
        wandb_tags=WANDB_TAGS.value,
        wandb_notes=WANDB_NOTES.value,
        checkpt_path=CHECKPT_PATH.value,
        run_id=RUN_ID.value,
        input_cols=INPUT_COLS.value,
        output_cols=OUTPUT_COLS.value,
        instruction_cols=INSTRUCTION_COLS.value,
    )


if __name__ == "__main__":
    app.run(main)
