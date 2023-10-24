python lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=mistralai/Mistral-7B-v0.1,peft=/home/sidnbaskaran/llm-efficiency-challenge/checkpoints/dummy-7rv119oc \
    --tasks hendrycksTest-abstract_algebra,hendrycksTest-college_chemistry,hendrycksTest-computer_security,hendrycksTest-econometrics,hendrycksTest-us_foreign_policy \
    --device cuda:0 \
    --batch_size auto \
    --output_base_path results \
    # --limit 0.2