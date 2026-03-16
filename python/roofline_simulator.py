import argparse
import os
import json
from transformers import (
    AutoModelForCausalLM, AutoConfig,
    ViTModel, ViTConfig,
)

# Optional Mixtral
try:
    from transformers import MixtralForCausalLM, MixtralConfig
except ImportError:
    MixtralForCausalLM = None
    MixtralConfig = None

# ------------------------------
# Hardware Specs
# ------------------------------
GPU_SPECS = {
    "A100": {
        "peak_flops": {
            "bfloat16": 0.312e15,   # 312 TFLOPS (BF16 Tensor Core, SXM 80GB)
        },
        "bandwidth": 2.0e12,        # 2.0 TB/s (A100 SXM 80GB HBM2e)
    },
    "H200": {
        "peak_flops": {
            "bfloat16": 1.979e15,   # 1,979 TFLOPS (BF16 Tensor Core, SXM)
            "float8":   3.958e15,   # 3,958 TFLOPS (FP8 Tensor Core, SXM)
        },
        "bandwidth": 4.8e12,        # 4.8 TB/s (HBM3e)
    },
    "B200": {
        "peak_flops": {
            "bfloat16": 2.25e15,    # 2.25 PFLOPS
            "float8":   4.5e15,     # 4.50 PFLOPS
        },
        "bandwidth": 8.0e12,        # 8.0 TB/s
    },
}

DTYPES      = ["bfloat16", "float8"]
BATCH_SIZES = [16, 32, 64, 128]

# ------------------------------
# Models
# ------------------------------
MODELS = {
    "vit-b":        ("google/vit-base-patch16-224",           ViTModel,                                          ViTConfig),
    "vit-l":        ("google/vit-large-patch16-224",          ViTModel,                                          ViTConfig),
    "vit-h":        ("google/vit-huge-patch14-224-in21k",     ViTModel,                                          ViTConfig),
    "gpt2-l":       ("gpt2-large",                            AutoModelForCausalLM,                              AutoConfig),
    "gpt2-xl":      ("gpt2-xl",                               AutoModelForCausalLM,                              AutoConfig),
    "minerva-7b":   ("sapienzanlp/Minerva-7B-instruct-v1.0",  AutoModelForCausalLM,                              AutoConfig),
    "llama3-8b":    ("meta-llama/Meta-Llama-3-8B",            AutoModelForCausalLM,                              AutoConfig),
    "llama3-70b":   ("meta-llama/Meta-Llama-3-70B",           AutoModelForCausalLM,                              AutoConfig),
    "mixtral-8x7b": (
        "mistralai/Mixtral-8x7B-v0.1",
        MixtralForCausalLM if MixtralForCausalLM else AutoModelForCausalLM,
        MixtralConfig      if MixtralConfig      else AutoConfig,
    ),
}

# ------------------------------
# Roofline helpers
# ------------------------------
def roofline_time(flops, bytes_accessed, peak_flops, peak_bw):
    ai = flops / bytes_accessed if bytes_accessed > 0 else float("inf")
    return flops / min(peak_flops, ai * peak_bw)

def bytes_per_element(dtype_str):
    return {"bfloat16": 2.0, "float8": 1.0}[dtype_str]

def count_non_expert_params(model, model_name):
    total = 0
    for name, param in model.named_parameters():
        if "mixtral" in model_name.lower():
            if "block_sparse_moe" in name and "experts" in name:
                continue
        total += param.numel()
    return total

def compute_times(N, L, d, H, E, k, B, dtype_str, gpu_name):
    """Return (forward_time_us, backward_time_us) for one configuration."""
    s          = bytes_per_element(dtype_str)
    peak_flops = GPU_SPECS[gpu_name]["peak_flops"][dtype_str]
    bw         = GPU_SPECS[gpu_name]["bandwidth"]

    # FLOPs
    attn_f = (8 * B * N * d**2 + 4 * B * N**2 * d) * L
    mlp_f  = (4 * B * N * d * H * k) * L

    # Memory bytes accessed
    attn_b = (4 * d**2 * s + 2 * B * N * d * s) * L
    mlp_b  = (2 * d * H * s * E + 2 * B * N * d * s) * L

    t_fwd = (roofline_time(attn_f, attn_b, peak_flops, bw) +
             roofline_time(mlp_f,  mlp_b,  peak_flops, bw))
    t_bwd = 2 * t_fwd

    return round(t_fwd * 1e6, 2), round(t_bwd * 1e6, 2)

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Theoretical Roofline Simulator — A100 / H200 / B200, JSON output"
    )
    parser.add_argument("model_name", choices=list(MODELS.keys()))
    args = parser.parse_args()

    model_key = args.model_name
    hf_name, model_class, config_class = MODELS[model_key]

    # ── Load config ──────────────────────────────────────────────────────────
    print(f"Loading config for {model_key}...")
    config = config_class.from_pretrained(hf_name, trust_remote_code=True)

    # Sequence length
    if hasattr(config, "seq_len"):
        N = config.seq_len
    elif hasattr(config, "n_positions"):
        N = config.n_positions
    elif hasattr(config, "max_position_embeddings"):
        N = config.max_position_embeddings
    elif hasattr(config, "image_size") and hasattr(config, "patch_size"):
        N = (config.image_size // config.patch_size) ** 2 + 1
    else:
        N = 1024

    # Architecture parameters
    L = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))
    d = getattr(config, "hidden_size",       getattr(config, "n_embd", 768))
    H = getattr(config, "intermediate_size", 4 * d)
    E = getattr(config, "num_local_experts", 1)
    k = 2 if "mixtral" in model_key else 1

    # ── Load model to count non-expert parameters ─────────────────────────
    print("Loading model to count non-expert parameters...")
    model = model_class.from_pretrained(hf_name, trust_remote_code=True)
    model_size = count_non_expert_params(model, model_key)
    del model  # free memory

    # ── Build JSON ────────────────────────────────────────────────────────
    output = {
        "model":        model_key,
        "seq_len":      N,
        "embedded_dim": d,
        "num_blocks":   L,
        "model_size":   model_size,
        "ffn": {
            "intermediate_size": H,
            "num_experts":       E,
            "active_experts_k":  k,
        },
        "gpus": {},
    }

    # Sweep GPU x dtype x batch_size (only dtypes supported by each GPU)
    for gpu_name in GPU_SPECS:
        output["gpus"][gpu_name] = {}
        for dtype_str in GPU_SPECS[gpu_name]["peak_flops"]:
            output["gpus"][gpu_name][dtype_str] = {}
            for B in BATCH_SIZES:
                t_fwd, t_bwd = compute_times(N, L, d, H, E, k, B, dtype_str, gpu_name)
                output["gpus"][gpu_name][dtype_str][str(B)] = {
                    "forward_time_us":  t_fwd,
                    "backward_time_us": t_bwd,
                }
                print(f"  {gpu_name} | {dtype_str} | batch={B:3d} → fwd={t_fwd:.2f} us  bwd={t_bwd:.2f} us")

    # ── Write JSON ────────────────────────────────────────────────────────
    out_dir = os.path.expanduser("~/DLNetBench/model_stats")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{model_key}.json")

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved → {out_file}")