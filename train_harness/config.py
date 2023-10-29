import json
import os
from dataclasses import InitVar, dataclass
from typing import Callable, Dict, Literal, Optional

import yaml
from peft import LoraConfig as LoraConfig_
from peft import PeftType, TaskType

DEFAULT_SYSTEM_PROMPT = """ Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
"""


def load_from_yaml(path: str | os.PathLike) -> Dict[str, Callable]:
    class_dict = {}
    print(f"\n path:{path} \n")
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    for key, cfg in cfg_dict.items():
        if key in _CFG_REGISTRY:
            class_dict[key] = _CFG_REGISTRY[key](**cfg)

    return class_dict


@dataclass
class LoraConfig(LoraConfig_):
    path: InitVar[str] = None
    name: InitVar[str] = "default"

    # run right after initialization, gives the user the option to specify a 
    # custom configuration path and name. If a path is provided, the method reads
    # the configuration from this path and updates the instance attributes accordingly.
    def __post_init__(self, path: Optional[str], name: Optional[str]):
        self.task_type = TaskType.CAUSAL_LM
        self.peft_type = PeftType.LORA
        
        if not path:
            return

        cfg_path = os.path.join(path, "adapter_config.json")

        assert os.path.exists(cfg_path), "Adapter path does not contain config."

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        self.r = cfg["r"]
        self.lora_alpha = cfg["lora_alpha"]
        self.target_modules = cfg["target_modules"]
        self.name = name
        self.path = path


@dataclass
class DataConfig:
    max_length: int = 2048
    template: str = "alpaca_input"
    input_column: str = "input"
    output_column: str = "output"
    instruction_column: str = "instruction"
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
    test_split: float = 0.1
    proc_bsz: int = 1000


@dataclass
class TrainingConfig:
    """Note: all training is on one gpu"""

    batch_size: int = 16 # see if increasing this boosts GPU memory usage. 
    grad_accum_steps: int = 4
    epochs: int = 1
    steps: int = -1
    peak_lr: float = 3e-4
    weight_decay: float = 0.0
    gradient_clip_val: float = 1.0
    warmup_steps: Optional[int] = 0
    warmup_ratio: Optional[float] = 0.3
    seed: int = 42
    use_flash_attn: bool = True
    gradient_checkpointing_enabled: bool = True
    neftune_noise_alpha: int = 5

    hf_model: str = "mistralai/Mistral-7B-v0.1"
    bits: Literal[4, 8, -1] = -1

    log_steps: int = 10
    eval_steps: int = 100
    do_eval: bool = True


_CFG_REGISTRY = {
    "lora_config": LoraConfig,
    "data_config": DataConfig,
    "training_config": TrainingConfig,
}
