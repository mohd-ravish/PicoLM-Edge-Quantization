"""Sliding-window perplexity evaluation on WikiText-2.

Usage:
    python scripts/evaluate_ppl.py --model_path ./Phi-3.5-mini-instruct-1bit-openassistant-guanaco/
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def calculate_perplexity(model, tokenizer, test_dataset, max_length=512, stride=256):
    model.eval()
    device = next(model.parameters()).device

    encodings = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="Evaluating Perplexity"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # tokens we calculate loss on
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # mask context tokens

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss  # mean loss per active token

        nlls.append(neg_log_likelihood * trg_len)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return ppl.item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate WikiText-2 perplexity for a causal LM checkpoint.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to write {'perplexity': ...} to.")
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=token)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", token=token)

    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ppl_score = calculate_perplexity(model, tokenizer, test_data, max_length=args.max_length, stride=args.stride)
    print(f"Perplexity (WikiText-2): {ppl_score:.2f}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"perplexity_wikitext2": ppl_score}, f, indent=2)


if __name__ == "__main__":
    main()
