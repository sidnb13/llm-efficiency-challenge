import copy
import os
from dataclasses import asdict
from functools import wraps
from os import PathLike
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import torch
from datasets import Dataset, concatenate_datasets
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from torch.cuda import memory_allocated, memory_cached
from transformers import (
    DataCollator,
    EvalPrediction,
    MistralForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)

import wandb

from .config import DataConfig, LoraConfig, TrainingConfig
from .data import InstructionDataset


def neftune_forward(self: nn.Embedding, input: torch.Tensor):
    """
    Implements the NEFTune forward pass for the model. Note this works only for
    torch.nn.Embedding layers. This method is slightly adapted from the original source code
    that can be found here: https://github.com/neelsjain/NEFTune

    Args:
        input (`torch.Tensor`):
            The input tensor to the model.
        noise_alpha (`float`):
            The noise alpha value to use for the NEFTune forward pass.
    """
    embeddings = torch.nn.functional.embedding(
        input,
        self.weight,
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse,
    )

    if self.training:
        dims = torch.tensor(embeddings.size(1) * embeddings.size(2))
        mag_norm = getattr(self, "neftune_noise_alpha") / torch.sqrt(dims)
        embeddings = embeddings + torch.zeros_like(embeddings).uniform_(
            -mag_norm, mag_norm
        )

    return embeddings


class NEFTTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        neftune_noise_alpha: Optional[float] = None,
    ):
        self.neftune_noise_alpha = neftune_noise_alpha

        if self.neftune_noise_alpha is not None:
            model = self._activate_neftune(model)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def _activate_neftune(self, model):
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
        """
        if isinstance(model, PreTrainedModel):
            embeddings = model.get_input_embeddings()
        elif isinstance(model, PeftModel):
            embeddings = model.base_model.get_input_embeddings()

        setattr(embeddings, "neftune_noise_alpha", self.neftune_noise_alpha)
        old_forward = embeddings.forward

        # This hack seems to be needed to properly use a custom forward pass
        # all credits to: https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/11
        bound_method = neftune_forward.__get__(embeddings, embeddings.__class__)
        setattr(embeddings, "forward", bound_method)
        # embeddings.forward = neftune_forward
        setattr(embeddings, "_old_forward", old_forward)

        return model
    
    # ensure train method retains the metadata of 'Trainer.train'
    @wraps(Trainer.train)
    def train(self, *args, **kwargs):
        output = super().train(*args, **kwargs)

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer
        if self.neftune_noise_alpha is not None:
            if isinstance(self.model, PreTrainedModel):
                embeddings = self.model.get_input_embeddings()
            elif isinstance(self.model, PeftModel):
                embeddings = self.model.base_model.get_input_embeddings()

            if hasattr(embeddings, "_old_forward"):
                embeddings.forward = getattr(embeddings, "_old_forward")
                setattr(embeddings, "_old_forward", None)
                setattr(embeddings, "neftune_noise_alpha", None)

        return output


def create_load_peft_model(
    training_config: TrainingConfig,
    lora_config: LoraConfig,
    inference: bool = False,
):
    base_model = MistralForCausalLM.from_pretrained(
        training_config.hf_model,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
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
        peft_model.config.use_cache = False

    return peft_model
 
    def on_log(self, args, state: TrainerState, control: TrainerControl, model, optimizer, **kwargs):
        if torch.cuda.is_available() and state.global_step % args.logging_steps == 0:
            total_memory = torch.cuda.memory_allocated()
            
            # Estimate memory of adapter layers
            adapter_memory = sum(param.nelement() * param.element_size() for name, param in model.named_parameters() if 'adapter' in name)
            
            # Estimate memory used by optimizer state
            optimizer_memory = sum(tensor.nelement() * tensor.element_size() for group in optimizer.param_groups for tensor in group['params'])
            
            logs = {
                "gpu_memory_total_mb": total_memory / (1024 * 1024),
                "gpu_memory_adapter_mb": adapter_memory / (1024 * 1024),
                "gpu_memory_optimizer_mb": optimizer_memory / (1024 * 1024),
                # minibatch memory can be computed during a forward pass, outside of this method
            }
            self._trainer.log(logs)

def train(
    training_config: TrainingConfig,
    data_config: DataConfig,
    lora_config: LoraConfig,
    dataset_path: List[str | PathLike],
    wandb_project: str,
    wandb_entity: str,
    wandb_enabled: bool,
    wandb_run_name: Optional[str] = "mistral_lora",
    wandb_tags: Optional[List[str]] = None,
    wandb_notes: Optional[str] = None,
    checkpt_path: Optional[str] = None,
    run_id: Optional[str] = None,
    instruction_cols: Optional[List[str]] = None,
    input_cols: Optional[List[str]] = None,
    output_cols: Optional[List[str]] = None,
):
    # setup model
    peft_model = create_load_peft_model(training_config, lora_config, inference=False)
    
    print(f"\n core,py line 195 dataset_path: {dataset_path}\n")
    if len(dataset_path) == 1:
        # setup data
        instruction_data = InstructionDataset(
            data_config, dataset_path[0], training_config.hf_model
        )
        instruction_data.process_dataset()
        train_ds = instruction_data.get_dataset("train")
        test_ds = instruction_data.get_dataset("test")
    else:
        # TODO mixed dataset capability
        flattened_datasets = []
        if instruction_cols and len(instruction_cols) < len(dataset_path):
            instruction_cols = instruction_cols[0] * len(dataset_path)
        if input_cols and len(input_cols) < len(dataset_path):
            input_cols = input_cols[0] * len(dataset_path)
        if output_cols and len(output_cols) < len(dataset_path):
            output_cols = output_cols[0] * len(dataset_path)

        for i, ds_path_name in enumerate(dataset_path):
            conf = copy.deepcopy(data_config)
            if instruction_cols:
                conf.instruction_column = instruction_cols[i]
            if input_cols:
                conf.input_column = input_cols[i]
            if output_cols:
                conf.output_column = output_cols[i]
            instruction_data = InstructionDataset(
                data_config, ds_path_name, training_config.hf_model
            )

            instruction_data.process_dataset()
            flattened_datasets.append(
                concatenate_datasets(
                    [
                        instruction_data.get_dataset("train"),
                        instruction_data.get_dataset("test"),
                    ]
                )
            )
        mixed_ds = concatenate_datasets(flattened_datasets)
        mixed_ds = mixed_ds.train_test_split(
            test_size=data_config.test_split, shuffle=True
        )
        train_ds, test_ds = mixed_ds["train"], mixed_ds["test"]

    # training
    _do_eval = training_config.do_eval and data_config.test_split > 0
    training_args = TrainingArguments(
        report_to="none" if not wandb_enabled else "wandb",
        optim="paged_adamw_8bit",
        bf16=True,
        do_eval=_do_eval,
        evaluation_strategy="steps",
        num_train_epochs=training_config.epochs,
        max_steps=training_config.steps,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.grad_accum_steps,
        learning_rate=training_config.peak_lr,
        weight_decay=training_config.weight_decay,
        warmup_steps=training_config.warmup_steps,
        warmup_ratio=training_config.warmup_ratio,
        lr_scheduler_type="cosine",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=training_config.log_steps,
        eval_steps=training_config.eval_steps,
        gradient_checkpointing=training_config.gradient_checkpointing_enabled,
        output_dir="checkpoints",
        load_best_model_at_end=True,
    )

    trainer = NEFTTrainer(
        model=peft_model,
        args=training_args,
        data_collator=instruction_data.collator,
        train_dataset=train_ds,
        eval_dataset=test_ds if _do_eval else None,
        neftune_noise_alpha=training_config.neftune_noise_alpha,
        callbacks=[GPUMemoryLoggerCallback()],
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
        resume="allow",
        id=run_id,
        config={
            "data_config": asdict(data_config),
            "training_config": asdict(training_config),
            "lora_config": asdict(lora_config),
        },
    ) as run:
        trainer.train(resume_from_checkpoint=checkpt_path)
        trainer.save_model(
            os.path.join("checkpoints", f"{run.name}_{dataset_path[0].split('/')[-1]}")
        )
