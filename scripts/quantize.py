"""Load a base model and serialize an HQQ-quantized checkpoint.

Usage:
    python scripts/quantize.py --config configs/hqq_config.json --preset phi3.5_mini_1bit
    python scripts/quantize.py --model_id microsoft/Phi-3.5-mini-instruct --nbits 1 --save_path ./out/
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig
from hqq.models.hf.base import AutoHQQHFModel


def run_quantization(model_id, save_path, nbits=1, group_size=64, quant_zero=False, quant_scale=False, axis=1, token=None):
    quant_config = HqqConfig(
        nbits=nbits,
        group_size=group_size,
        quant_zero=quant_zero,
        quant_scale=quant_scale,
        axis=axis,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        quantization_config=quant_config,
        token=token,
    )

    # Save BEFORE any further patching (e.g. prepare_model_for_kbit_training)
    AutoHQQHFModel.save_quantized(model, save_path)
    tokenizer.save_pretrained(save_path)

    return model, tokenizer


def load_preset(config_path, preset):
    with open(config_path, "r", encoding="utf-8") as f:
        presets = json.load(f)
    if preset not in presets:
        raise KeyError(f"Preset '{preset}' not found in {config_path}. Available: {list(presets)}")
    return presets[preset]


def main():
    parser = argparse.ArgumentParser(description="Quantize a causal LM with HQQ.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON file of named presets.")
    parser.add_argument("--preset", type=str, default=None, help="Preset name inside --config to use.")
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--save_path", type=str, default="./Phi-3.5-mini-instruct_HQQ-1bit/")
    parser.add_argument("--nbits", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--quant_zero", action="store_true")
    parser.add_argument("--quant_scale", action="store_true")
    parser.add_argument("--axis", type=int, default=1)
    args = parser.parse_args()

    if args.config and args.preset:
        preset = load_preset(args.config, args.preset)
    else:
        preset = {
            "model_id": args.model_id,
            "save_path": args.save_path,
            "nbits": args.nbits,
            "group_size": args.group_size,
            "quant_zero": args.quant_zero,
            "quant_scale": args.quant_scale,
            "axis": args.axis,
        }

    token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")

    run_quantization(
        model_id=preset["model_id"],
        save_path=preset["save_path"],
        nbits=preset["nbits"],
        group_size=preset["group_size"],
        quant_zero=preset["quant_zero"],
        quant_scale=preset["quant_scale"],
        axis=preset["axis"],
        token=token,
    )
    print(f"Saved HQQ-quantized checkpoint to {preset['save_path']}")


if __name__ == "__main__":
    main()
