python main.py \
    --model mistral-lima \
    --model_args pretrained=mistralai/Mistral-7B-v0.1,peft=/home/sidnbaskaran/llm-efficiency-challenge/checkpoints/dummy-7rv119oc \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0