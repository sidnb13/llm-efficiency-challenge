# NeurIPS 2023 LLM Efficiency Challenge Attempt

# Setup and Usage

### Bitsandbytes

```shell
git clone https://github.com/automorphic-ai/bitsandbytes
cd bitsandbytes
CUDA_VERSION=117 make cuda11x
python setup.py install
python -m bitsandbytes
cd ..
```

Sample training command:
```shell
CONFIG=configs/dolly.yaml python train.py --config=config.py --use_wandb --dataset_path="databricks/databricks-dolly-15k"
```

Inference:
```shell
TBD
```

## Resources available

https://lightning.ai/pages/community/tutorial/neurips2023-llm-efficiency-guide/#toc12

https://lightning.ai/pages/community/lora-insights/#toc2

Starter - [Llama Recipes](https://github.com/facebookresearch/llama-recipes) loosely based, stripped down + modified. Single A100 40GB. Full finetune requires much more.

> Came across [this](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#anatomy-of-models-memory) after searching for gradients and this helps to explain it, thank you! So based on this thats 16 bytes per param \* 1B = 16GB per param. That's 48GB for a 3B param model and 112GB for a 7B. That's also not counting the forward activations and temporary memory allocations, which appear to be highly dependent on the model architecture and the sequence length. 112GB for a 7B gets us to about 20% the 8x A100 memory and from what I can tell sequence length / architecture can significantly increase memory size. Does it make sense that 80% of the memory usage for training Llama2 goes to forward activations, general usage (temp allocations, etc) and batch size?

Tasks are conversational, QA, reasoning focus.

Datasets: https://github.com/Zjh-819/LLMDataHub for good overview

## Strategy

Regular LoRA fine tuning, optimal ranks seem to be `r/alpha=256/512`. Use Mistral-7B as base model.

> Active learning and some tricks when dealing with large amounts of data:
>
> [1] https://medium.com/@timothylimyonglee/finetuning-llm-efficiently-part-1-simple-fixes-to-the-dataloader-c4eef3e9822a
>
> [2] https://medium.com/@timothylimyonglee/finetuning-llm-efficiently-part-2-sorting-sequences-matters-potential-time-and-money-saver-7e6fa57067b2
>
> [3] https://medium.com/@timothylimyonglee/active-learning-for-llm-part-1-exploration-b8ea6d80f58d
>
> [4] https://medium.com/@timothylimyonglee/active-learning-for-llm-part-2-how-about-chat-994071941f9c

### Optimizations

-   [Sequence Packing](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py#L516)

-   [NEFTune](https://twitter.com/younesbelkada/status/1714283468790935687?t=ouZhU6BUFhLnaisfepDx8g&s=19)

### Datasets

-   https://huggingface.co/datasets/garage-bAInd/Open-Platypus
-   https://huggingface.co/datasets/OpenAssistant/oasst1
-   https://huggingface.co/datasets/GAIR/lima
