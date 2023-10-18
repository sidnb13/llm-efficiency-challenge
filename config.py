import os

from ml_collections import config_dict

from train_harness import load_from_yaml

cfg_file_path = os.environ.get("CONFIG", None)


def get_config():
    config_cls_dict = load_from_yaml(cfg_file_path)

    cfg_dict = config_dict.ConfigDict(
        {
            "lora_config": config_cls_dict["lora_config"],
            "data_config": config_cls_dict["data_config"],
            "training_config": config_cls_dict["training_config"],
        }
    )

    return cfg_dict
