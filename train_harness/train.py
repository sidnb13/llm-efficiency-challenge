from os import PathLike

import torch
from peft import get_peft_config, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, MistralForCausalLM, TrainingArguments
from trl import SFTTrainer

from .config import DataConfig, LoraConfig, TrainingConfig
from .data import InstructionDataset


def create_load_peft_model(
    training_config: TrainingConfig, lora_config: LoraConfig, inference: bool = False
):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=training_config.bits == 4,
        load_in_8bit=training_config.bits == 8,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = MistralForCausalLM.from_pretrained(
        training_config.hf_model,
        torch_dtype=torch.bfloat16,
        device_map={"": training_config.device},
        use_flash_attention_2=training_config.use_flash_attn,
        quantization_config=quantization_config,
    )

    if training_config.bits != -1:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=training_config.gradient_checkpointing_enabled,
        )
    elif training_config.gradient_checkpointing_enabled:
        base_model.gradient_checkpointing_enable()

    # adjust padding
    if hasattr(base_model, "config"):
        base_model.config.pad_token_id = base_model.config.eos_token_id

    peft_model = get_peft_model(base_model, lora_config, adapter_name=lora_config.name)

    if training_config.gradient_checkpointing_enabled and not inference:
        peft_model.enable_input_require_grads()

    return peft_model


def train(
    training_config: TrainingConfig,
    data_config: DataConfig,
    lora_config: LoraConfig,
    dataset_path: str | PathLike,
):
    # setup model
    peft_model = create_load_peft_model(training_config, lora_config, inference=False)
    # setup data
    instruction_data = InstructionDataset(
        data_config, dataset_path, training_config.hf_model
    )
    # training
    training_args = TrainingArguments()
    trainer = SFTTrainer()
