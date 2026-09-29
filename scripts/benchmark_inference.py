"""Footprint reduction, throughput, and latency benchmarks for a quantized checkpoint.

Usage:
    python scripts/benchmark_inference.py --model_path ./Phi-3.5-mini-instruct-1bit-openassistant-guanaco/ \
        --baseline_params 3.82e9 --baseline_dtype_bytes 2
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_dir_size_mb(path):
    total_bytes = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes / (1024 * 1024)


def measure_footprint_reduction(model_path, baseline_params, baseline_dtype_bytes):
    baseline_mb = (baseline_params * baseline_dtype_bytes) / (1024 * 1024)
    quantized_mb = get_dir_size_mb(model_path)
    reduction_pct = ((baseline_mb - quantized_mb) / baseline_mb) * 100
    return {
        "baseline_mb": baseline_mb,
        "quantized_mb": quantized_mb,
        "reduction_pct": reduction_pct,
    }


def benchmark_inference(model, tokenizer, prompt="Explain quantum computing in simple terms:", max_new_tokens=64):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warm-up run
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    generated_tokens = output.shape[1] - inputs.input_ids.shape[1]
    throughput = generated_tokens / elapsed_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0

    return {
        "generated_text": tokenizer.decode(output[0], skip_special_tokens=True),
        "generated_tokens": generated_tokens,
        "elapsed_time_s": elapsed_time,
        "throughput_tokens_per_sec": throughput,
        "peak_vram_mb": peak_vram_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark footprint reduction and inference latency.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--baseline_params", type=float, default=3.82e9, help="Baseline model parameter count.")
    parser.add_argument("--baseline_dtype_bytes", type=int, default=2, help="Bytes/param for the baseline dtype (e.g. 2 for bf16).")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms:")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")

    footprint = measure_footprint_reduction(args.model_path, args.baseline_params, args.baseline_dtype_bytes)
    print(f"Baseline Size    : {footprint['baseline_mb']:.2f} MB (~{footprint['baseline_mb'] / 1024:.2f} GB)")
    print(f"Quantized Size   : {footprint['quantized_mb']:.2f} MB (~{footprint['quantized_mb'] / 1024:.2f} GB)")
    print(f"Footprint Reduction: {footprint['reduction_pct']:.2f}%")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=token)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="cuda", token=token)

    latency = benchmark_inference(model, tokenizer, prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    print("-" * 50)
    print(f"Generated text: {latency['generated_text']}")
    print(f"Tokens Generated     : {latency['generated_tokens']}")
    print(f"Elapsed Time         : {latency['elapsed_time_s']:.3f} s")
    print(f"Inference Throughput : {latency['throughput_tokens_per_sec']:.2f} tokens/s")
    print(f"Peak VRAM Usage      : {latency['peak_vram_mb']:.2f} MB")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"footprint": footprint, "latency": latency}, f, indent=2)


if __name__ == "__main__":
    main()
