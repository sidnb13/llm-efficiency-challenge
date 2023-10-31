import copy
import os
from itertools import chain
from os import PathLike
from typing import Dict, List

import psutil
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, default_data_collator

from train_harness.config import DataConfig


class PromptTemplates:
    alpaca_input: str = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    )
    alpaca_no_input: str = (
        (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        ),
    )
    chatml: str = "<|im_start|>{input_role}\n{message}\n<|im_end|>{output_role}\n"

# Nice utility class that packs a dataset for us, handles padding and generates attention masks. 
class InstructionDataset:
    def __init__(
        self,
        dataset_config: DataConfig,
        dataset_name: str | PathLike,
        tokenizer: str,
        debug: bool = False,
        pad_last: bool = True,
        packing: bool = True,
    ):
        self.config = dataset_config
        self.debug = debug
        self.pad_last = pad_last
        self.packing = packing
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.raw_dataset = self._load_dataset(dataset_name)
        # else Trainer defaults to padding
        self.collator = default_data_collator

    @staticmethod
    def _load_dataset(dataset_name: str | PathLike):
        print(f"os.curdir: {os.curdir}")
        print(f"os.path.exists(os.curdir): {os.path.exists(os.curdir)}")

        candidate_path = os.path.join(os.curdir, dataset_name)
        absolute_path = os.path.abspath(candidate_path)
        print(f"Absolute Path: {absolute_path}")
        print(f"os.path.exists(absolute_path): {os.path.exists(absolute_path)}")

        print(f"Current Working Directory: {os.getcwd()}")
        print(f"os.path.exists(os.getcwd()): {os.path.exists(os.getcwd())}")

        print(f"\n candidate_path: {candidate_path}\n")
        print(f"os.path.exists(candidate_path): {os.path.exists(candidate_path)}")


        # candidate_path = os.path.abspath(os.path.join(os.curdir, dataset_name))
        if not os.path.exists(candidate_path):
            ds = load_dataset(dataset_name, cache_dir="data")
        else:
            if os.path.isdir(candidate_path):
                ds = load_from_disk(candidate_path)
            else:
                assert dataset_name.endswith(".jsonl")
                ds = load_dataset("json", data_files=dataset_name, cache_dir="data")
        if "train" not in ds:
            return DatasetDict({"train": ds})
        return ds

    def process_dataset(self):
        dataset = self.raw_dataset.map(
            self.process_item,
            batched=True,
            batch_size=self.config.proc_bsz,
            remove_columns=self.raw_dataset["train"].column_names,
            desc="Processing dataset",
            load_from_cache_file=not self.debug,
            num_proc=psutil.cpu_count(),
        )
        if self.packing:
            dataset = dataset.map(
                self.pack_seqs,
                batched=True,
                batch_size=self.config.proc_bsz,
                desc="Packing sequences",
                load_from_cache_file=not self.debug,
                num_proc=psutil.cpu_count(),
            )
        self.processed_dataset = dataset

        if self.config.test_split > 0 and (
            (
                isinstance(self.processed_dataset, DatasetDict)
                and "test" not in self.processed_dataset
            )
            or isinstance(self.processed_dataset, Dataset)
        ):
            if "train" in self.processed_dataset:
                self.processed_dataset = self.processed_dataset.pop("train")
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
        if not self.pad_last:
            total_length = (total_length // block_size) * block_size

        def _pad(x: torch.Tensor | List[int]):
            if not self.pad_last:
                return x

            if isinstance(x, torch.Tensor):
                return torch.nn.functional.pad(
                    x, (0, block_size - len(x)), value=self.tokenizer.pad_token_id
                )
            return x + [self.tokenizer.pad_token_id] * (block_size - len(x))

        # Split by chunks of max_len.
        result = {
            k: [_pad(t[i : i + block_size]) for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def process_item(self, batch: Dict):
        IGNORE_IDX = -100

        labels, examples, attn_masks = [], [], []
        bsz = len(batch[list(batch.keys())[0]])

        for instruction, input, output in zip(
            batch.get(
                self.config.instruction_column,
                [self.config.system_prompt] * bsz,
            ),
            batch.get(self.config.input_column, [""] * bsz),
            batch.get(self.config.output_column, ["output"] * bsz),
        ):
            prompt = getattr(PromptTemplates, self.config.template).format(
                instruction=instruction, input=input
            )
            example = prompt + output

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
