# PicoLM: Edge Quantization

A 1-bit large language model built with half-quadratic quantization (HQQ) and QLoRA fine-tuning, targeting extreme weight compression for edge deployment without sacrificing quality.

## Highlights

- **Engineered a highly compressed 1-bit LLM**: quantizes `microsoft/Phi-3.5-mini-instruct` (and `meta-llama/Meta-Llama-3-8B`) to as low as 1-bit precision using HQQ, then recovers quality with parameter-efficient QLoRA fine-tuning.
- **~83% weight footprint reduction** versus the bf16 baseline, while tracking perplexity (PPL) on WikiText-2 to confirm quality is preserved.
- **Edge-focused evaluation**: on-disk footprint, WikiText-2 perplexity, and generation throughput/VRAM benchmarks, all reproducible from the command line or from the original Kaggle notebooks.

## Repository Structure

```
picolm-edge-quantization/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── hqq_config.json                 # Named HQQ presets (bit-width, group size, etc.)
├── notebooks/
│   ├── quantization_and_training.ipynb        # Full pipeline: HQQ 1-bit quantize -> QLoRA fine-tune -> eval
│   └── multi_model_quantization_comparison.ipynb  # HQQ at 1-bit/4-bit (Phi-3.5) and 2-bit (Llama-3-8B)
├── scripts/
│   ├── quantize.py             # Load a base model, HQQ-quantize it, save the checkpoint
│   ├── train_peft.py           # QLoRA fine-tuning on timdettmers/openassistant-guanaco
│   ├── evaluate_ppl.py         # Sliding-window WikiText-2 perplexity
│   └── benchmark_inference.py  # Footprint reduction + generation throughput/VRAM
```

## Setup

```bash
pip install -r requirements.txt
export HUGGINGFACE_API_KEY=<your_hf_token>   # or HF_TOKEN
export WANDB_API_KEY=<your_wandb_key>        # optional, for experiment tracking
```

A CUDA GPU is required (the scripts and notebooks assume `device_map="cuda"`).

## Usage

**1. Quantize a base model with HQQ**

```bash
python scripts/quantize.py --config configs/hqq_config.json --preset phi3.5_mini_1bit
```

**2. QLoRA fine-tune the quantized model**

```bash
python scripts/train_peft.py \
    --model_id microsoft/Phi-3.5-mini-instruct \
    --nbits 1 \
    --output_dir ./Phi-3.5-mini-instruct-1bit-openassistant-guanaco/
```

**3. Evaluate perplexity on WikiText-2**

```bash
python scripts/evaluate_ppl.py \
    --model_path ./Phi-3.5-mini-instruct-1bit-openassistant-guanaco/ \
    --output_json results/benchmark_metrics.json
```

**4. Benchmark footprint reduction and inference latency**

```bash
python scripts/benchmark_inference.py \
    --model_path ./Phi-3.5-mini-instruct-1bit-openassistant-guanaco/ \
    --baseline_params 3.82e9 --baseline_dtype_bytes 2 \
    --output_json results/benchmark_metrics.json
```

The `scripts/` are extracted, CLI-driven versions of the logic developed interactively in `notebooks/`. The notebooks remain the canonical, step-by-step walkthrough (originally run on Kaggle, using Kaggle Secrets for `HUGGINGFACE_API_KEY`/`WANDB_LOGIN_API`); the scripts are meant for reproducing individual steps outside that environment.

## Method

1. **Quantization**: [HQQ](https://github.com/mobiusml/hqq) quantizes each linear layer to `nbits` precision with a given `group_size`, applied via `transformers.HqqConfig` at load time.
2. **Fine-tuning**: `prepare_model_for_kbit_training` + a LoRA adapter (`r=16`, `alpha=16`, targeting all attention and MLP projections) recovers quality lost to quantization, trained on `timdettmers/openassistant-guanaco` via `trl.SFTTrainer`.
3. **Evaluation**: perplexity is computed with a sliding-window pass over WikiText-2 (`max_length=512`, `stride=256`); inference is benchmarked with a warm-up generation followed by a timed, greedy-decoded generation to report tokens/sec and peak VRAM.

## Notes

- `notebooks/multi_model_quantization_comparison.ipynb` quantizes three configurations (Phi-3.5-mini at 1-bit and 4-bit, Llama-3-8B at 2-bit) purely for footprint/PPL/latency comparison — it does not fine-tune.
- Model checkpoints (`*.safetensors`, `*.bin`, `*.pt`, `checkpoint-*/`) are gitignored; only code, configs, and metrics are tracked in this repository.
