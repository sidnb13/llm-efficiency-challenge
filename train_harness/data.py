from cgi import print_arguments
import copy
import os
from itertools import chain
from os import PathLike
from typing import Dict

from datasets import DatasetDict, load_dataset
from platformdirs import user_cache_dir
from transformers import AutoTokenizer, default_data_collator

from train_harness.config import DataConfig


class PromptTemplates:
    alpaca_input: str = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )
    alpaca_no_input: str = (
        (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    )


class InstructionDataset:
    def __init__(
        self,
        dataset_config: DataConfig,
        dataset_name: str | PathLike,
        tokenizer: str,
        debug: bool = False,
    ):
        self.config = dataset_config
        self.debug = debug
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if not os.path.exists(dataset_name):
            self.raw_dataset = load_dataset(dataset_name, cache_dir="data")
        else:
            assert dataset_name.endswith(".jsonl")
            self.raw_dataset = load_dataset(
                "json", data_files=dataset_name, cache_dir="data"
            )
        if "train" not in self.raw_dataset:
            self.raw_dataset = DatasetDict({"train": self.raw_dataset})
        # else Trainer defaults to padding
        self.collator = default_data_collator

    def process_dataset(self):
        dataset = self.raw_dataset.map(
            self.process_item,
            batched=True,
            batch_size=self.config.proc_bsz,
            remove_columns=self.raw_dataset["train"].column_names,
            desc="Processing dataset",
            load_from_cache_file=not self.debug,
        )
        dataset = dataset.map(
            self.pack_seqs,
            batched=True,
            batch_size=self.config.proc_bsz,
            desc="Packing sequences",
            load_from_cache_file=not self.debug,
        )

        self.processed_dataset = dataset

        if self.config.test_split > 0 and not isinstance(
            self.processed_dataset, DatasetDict
        ):
            ds = self.processed_dataset.train_test_split(
                test_size=self.config.test_split
            )
            self.processed_dataset = ds

    def get_dataset(self, split: str = None):
        if split:
            return self.processed_dataset[split]
        return self.processed_dataset

    def pack_seqs(self, batch: Dict):
        # Concatenate all texts.
        block_size = self.config.max_length
        concatenated_examples = {k: list(chain(*batch[k])) for k in batch.keys()}
        total_length = len(concatenated_examples[list(batch.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def process_item(self, batch: Dict):
        IGNORE_IDX = -100

        labels, examples, attn_masks = [], [], []

        for instruction, input, output in zip(
            batch.get(
                self.config.instruction_column,
                [self.config.system_prompt] * len(batch[self.config.input_column]),
            ),
            batch[self.config.input_column],
            batch[self.config.output_column],
        ):
            prompt = getattr(PromptTemplates, self.config.template).format(
                instruction=instruction, input=input
            )
            example = self.tokenizer(
                prompt + output,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            label = copy.deepcopy(example["input_ids"].squeeze())
            label[: len(prompt)] = IGNORE_IDX

            labels.append(label)
            examples.append(example["input_ids"].squeeze())
            attn_masks.append(example["attention_mask"].squeeze())

        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask": attn_masks,
        }
