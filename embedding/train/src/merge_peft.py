import argparse
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModel, LlamaForCausalLM


def save_model(model, to):
    print(f"Saving dequantized model to {to}...")
    model.save_pretrained(to)


# def merge_peft(BASE_MODEL_PATH, PEFT_MODEL_PATH, SAVE_PATH):
def merge_peft(PEFT_MODEL_PATH, SAVE_PATH):
    config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)
    base_model_path = config.base_model_name_or_path
    base_model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    peft_model = PeftModel.from_pretrained(model=base_model, model_id=PEFT_MODEL_PATH)
    merged_model = peft_model.merge_and_unload()  # type: ignore

    llama_causual_model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    llama_causual_model.model = merged_model  # type: ignore
    save_model(llama_causual_model, to=SAVE_PATH)

    return base_model_path


def copy_tokenizer(BASE_MODEL_PATH, SAVE_PATH):
    os.system(f"cp -r {BASE_MODEL_PATH}/tokenizer.json {SAVE_PATH}")
    os.system(f"cp -r {BASE_MODEL_PATH}/tokenizer_config.json {SAVE_PATH}")
    os.system(f"cp -r {BASE_MODEL_PATH}/tokenizer.model {SAVE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_model_path", type=str, required=True, help="Path to the PEFT model")
    parser.add_argument(
        "--merged_model_path", type=str, required=True, help="Path to save the merged model"
    )
    args = parser.parse_args()
    base_model_path = merge_peft(args.peft_model_path, args.merged_model_path)
    copy_tokenizer(base_model_path, args.merged_model_path)
