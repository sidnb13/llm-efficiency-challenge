from os import PathLike
from typing import List, Optional

import torch
import wandb
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import MistralForCausalLM, TrainingArguments
from trl import SFTTrainer

from .config import DataConfig, LoraConfig, TrainingConfig
from .data import InstructionDataset


def create_load_peft_model(
    training_config: TrainingConfig,
    lora_config: LoraConfig,
    inference: bool = False,
    device: int = 0,
):
    base_model = MistralForCausalLM.from_pretrained(
        training_config.hf_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        use_flash_attention_2=training_config.use_flash_attn,
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
    wandb_project: str,
    wandb_entity: str,
    wandb_enabled: bool,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    wandb_notes: Optional[str] = None,
    checkpt_path: Optional[str] = None,
    run_id: Optional[str] = None,
    device: int = 0,
):
    # setup model
    peft_model = create_load_peft_model(
        training_config, lora_config, inference=False, device=device
    )
    # setup data
    instruction_data = InstructionDataset(
        data_config, dataset_path, training_config.hf_model
    )
    instruction_data.process_dataset()
    # training
    training_args = TrainingArguments(
        report_to="none" if not wandb_enabled else "wandb",
        optim="paged_adamw_32bit",
        bf16=True,
        num_train_epochs=training_config.epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.grad_accum_steps,
        learning_rate=training_config.peak_lr,
        weight_decay=training_config.weight_decay,
        warmup_steps=training_config.warmup_steps
        if training_config.warmup_steps
        else 0,
        warmup_ratio=training_config.warmup_ratio
        if training_config.warmup_ratio
        else 0,
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=training_config.log_steps,
        eval_steps=training_config.eval_steps,
        gradient_checkpointing=training_config.gradient_checkpointing_enabled,
        output_dir="checkpoints",
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        data_collator=instruction_data.collator,
        train_dataset=instruction_data.get_dataset("train"),
        eval_dataset=instruction_data.get_dataset("test"),
        dataset_text_field=data_config.input_column,
        packing=True,
        neftune_noise_alpha=training_config.neftune_noise_alpha,
    )

    if run_id:
        assert checkpt_path, "Must provide checkpt_path with run_id"

    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        mode="online" if wandb_enabled else "disabled",
        name=wandb_run_name,
        tags=wandb_tags,
        notes=wandb_notes,
        resume="must",
        id=run_id,
    ):
        trainer.train(resume_from_checkpoint=checkpt_path)
