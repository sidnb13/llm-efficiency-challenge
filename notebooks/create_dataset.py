from train_harness.config import LoraConfig, DataConfig
from train_harness.data import InstructionDataset

instruction_dataset = InstructionDataset(
    DataConfig(
        instruction_column="instruction",
        input_column="context",
        output_column="response",
    ),
    "databricks/databricks-dolly-15k",
    "mistralai/Mistral-7B-v0.1",
    debug=True,
)

instruction_dataset.process_dataset()
processed_ds = instruction_dataset.get_dataset()
processed_ds.save_to_disk("data/dolly-15k-packed/")

