from absl import app, flags
from ml_collections import config_flags

from train_harness import train

# define configuration file
_CONFIG = config_flags.DEFINE_config_file(
    "config",
    "config.py",
    "Config file for training.",
    lock_config=True,
)

USE_WANDB = flags.DEFINE_bool(
    "use_wandb",
    False,
    "Whether to use wandb for logging.",
)

WANDB_PROJECT = flags.DEFINE_string(
    "wandb_project",
    "llm-efficiency-challenge",
    "Name of wandb project.",
)

WANDB_ENTITY = flags.DEFINE_string(
    "wandb_entity",
    "sidnbaskaran",
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

DATASET_PATH = flags.DEFINE_string(
    "dataset_path",
    None,
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


def main(argv):
    del argv
    cfg = _CONFIG.value
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
    )


if __name__ == "__main__":
    app.run(main)
