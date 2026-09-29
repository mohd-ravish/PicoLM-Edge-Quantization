"""QLoRA fine-tuning of an HQQ-quantized model on timdettmers/openassistant-guanaco.

Usage:
    python scripts/train_peft.py --model_id microsoft/Phi-3.5-mini-instruct \
        --nbits 1 --output_dir ./Phi-3.5-mini-instruct-1bit-openassistant-guanaco/
"""
import argparse
import multiprocessing
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer
from hqq.models.hf.base import AutoHQQHFModel


class TimerCallback(TrainerCallback):
    """Periodically checkpoints so long Kaggle/Colab sessions survive a runtime timeout."""

    def __init__(self, save_time_in_seconds, trainer):
        self.start_time = None
        self.save_time_in_seconds = save_time_in_seconds
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.save_time_in_seconds:
            self.trainer.save_model(args.output_dir)
            self.trainer.log({"info": f"Checkpoint saved after {self.save_time_in_seconds}s"})
            self.start_time = time.time()


def add_eos(row, eos_token="<|end_of_text|>"):
    row["text"] = row["text"] + eos_token
    return row


def build_model_and_tokenizer(model_id, nbits, group_size, quant_zero, quant_scale, axis, quantized_save_path, token=None):
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

    # Save the quantized checkpoint BEFORE any further patching
    AutoHQQHFModel.save_quantized(model, quantized_save_path)

    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune an HQQ-quantized causal LM.")
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--nbits", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--axis", type=int, default=1)
    parser.add_argument("--quantized_save_path", type=str, default="./Phi-3.5-mini-instruct-1bit/")
    parser.add_argument("--output_dir", type=str, default="./Phi-3.5-mini-instruct-1bit-openassistant-guanaco/")
    parser.add_argument("--dataset", type=str, default="timdettmers/openassistant-guanaco")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint_interval_hours", type=float, default=9.0)
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")

    model, tokenizer = build_model_and_tokenizer(
        model_id=args.model_id,
        nbits=args.nbits,
        group_size=args.group_size,
        quant_zero=False,
        quant_scale=False,
        axis=args.axis,
        quantized_save_path=args.quantized_save_path,
        token=token,
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )

    dataset = load_dataset(args.dataset)
    ds = dataset.map(add_eos, num_proc=multiprocessing.cpu_count(), load_from_cache_file=False)

    training_arguments = SFTConfig(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        log_level="debug",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=args.learning_rate,
        eval_steps=100,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    trainer.add_callback(
        TimerCallback(save_time_in_seconds=args.checkpoint_interval_hours * 60 * 60, trainer=trainer)
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
