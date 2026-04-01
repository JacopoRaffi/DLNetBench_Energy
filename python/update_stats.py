import os
import json
import glob
import argparse

# ------------------------------
# Hardware Specs (complete)
# ------------------------------
GPU_SPECS = {
    "A100": {
        "peak_flops": {
            "bfloat16": 0.312e15,   # 312 TFLOPS (BF16 Tensor Core, Dense)
        },
        "bandwidth": 2.0e12,        # 2.0 TB/s (A100 SXM 80GB HBM2e)
    },
    "H200": {
        "peak_flops": {
            "bfloat16": 0.989e15,   # 989 TFLOPS (BF16 Tensor Core, Dense)
            "float8":   1.979e15,   # 1,979 TFLOPS (FP8 Tensor Core, Dense)
        },
        "bandwidth": 4.8e12,        # 4.8 TB/s (HBM3e)
    },
    "B200": {
        "peak_flops": {
            "bfloat16": 2.25e15,    # 2.25 PFLOPS (BF16 Tensor Core, Dense)
            "float8":   4.50e15,    # 4.50 PFLOPS (FP8 Tensor Core, Dense)
            "float4":   9.00e15     # 9.00 PFLOPS (FP4 Tensor Core, Dense)
        },
        "bandwidth": 8.0e12,        # 8.0 TB/s (HBM3e)
    },
    "GH200": {
        "peak_flops": {
            "bfloat16": 0.990e15,   # 990 TFLOPS (BF16 Tensor Core, Dense)
            "float8":   1.979e15,   # 1,979 TFLOPS (FP8 Tensor Core, Dense)
        },
        "bandwidth": 4e12,        # 4 TB/s (HBM3)
    },
    "H100": {
        "peak_flops": {
            "bfloat16": 0.989e15,   # 989 TFLOPS (BF16 Tensor Core, Dense, SXM)
            "float8":   1.979e15,   # 1,979 TFLOPS (FP8 Tensor Core, Dense, SXM)
        },
        "bandwidth": 3.35e12,       # 3.35 TB/s (H100 SXM5 HBM3)
    },
    "MI250X": {
        "peak_flops": {
            "bfloat16": 0.192e15,   # 0.192 PFLOPS (BF16 Matrix Core, Dense)
        },
        "bandwidth": 1.6e12,        # 1.6 TB/s (HBM2e)
    },
    "B300": {
        "peak_flops": {
            "bfloat16": 3.50e15,    # 3.50 PFLOPS (BF16 Tensor Core, Dense)
            "float8":   7.00e15,    # 7.00 PFLOPS (FP8 Tensor Core, Dense)
            "float4":   14.0e15     # 14.0 PFLOPS (FP4 Tensor Core, Dense)
        },
        "bandwidth": 8.0e12,        # 8.0 TB/s (HBM3e)
    },
    "GB300": {
        "peak_flops": {
            "bfloat16": 2.40e15,    # 2.40 PFLOPS (BF16 Tensor Core, Dense)
            "float8":   4.80e15,    # 4.80 PFLOPS (FP8 Tensor Core, Dense)
            "float4":   9.60e15     # 9.60 PFLOPS (FP4 Tensor Core, Dense)
        },
        "bandwidth": 8e12,        # 7.35 TB/s (HBM3e)
    },
     "GB200": {
        "peak_flops": {
            "bfloat16": 2.40e15,    # 2.40 PFLOPS (BF16 Tensor Core, Dense)
            "float8":   4.80e15,    # 4.80 PFLOPS (FP8 Tensor Core, Dense)
            "float4":   9.70e17     # 9.70 PFLOPS (FP4 Tensor Core, Dense)
        },
        "bandwidth": 8.0e12,        # 7.35 TB/s (HBM3e)
     }
} 
BATCH_SIZES = [16, 32, 64, 128]

# ------------------------------
# Roofline helpers
# ------------------------------
def roofline_time(flops, bytes_accessed, peak_flops, peak_bw):
    ai = flops / bytes_accessed if bytes_accessed > 0 else float("inf")
    return flops / min(peak_flops, ai * peak_bw)

def bytes_per_element(dtype_str):
    mapping = {"bfloat16": 2.0, "float8": 1.0, "float4": 0.5}
    return mapping.get(dtype_str, 2.0)

def compute_times(N, L, d, H, E, k, B, dtype_str, gpu_name):
    """
    Return (forward_time_us, backward_time_us, ffn_forward_time_us, ffn_backward_time_us)
    """
    s          = bytes_per_element(dtype_str)
    peak_flops = GPU_SPECS[gpu_name]["peak_flops"][dtype_str]
    bw         = GPU_SPECS[gpu_name]["bandwidth"]

    # Attention FLOPs and bytes
    attn_f = (8 * B * N * d**2 + 4 * B * N**2 * d) * L
    attn_b = (4 * d**2 * s + 2 * B * N * d * s) * L

    # MLP/FFN FLOPs and bytes
    mlp_f  = (4 * B * N * d * H * k) * L
    mlp_b  = (2 * d * H * s * E + 2 * B * N * d * s) * L

    # Compute times
    t_attn_fwd = roofline_time(attn_f, attn_b, peak_flops, bw)
    t_mlp_fwd  = roofline_time(mlp_f,  mlp_b,  peak_flops, bw)
    
    t_fwd = t_attn_fwd + t_mlp_fwd
    t_bwd = 2 * t_fwd
    
    t_mlp_fwd_us = round(t_mlp_fwd * 1e6, 2)
    t_mlp_bwd_us = round(2 * t_mlp_fwd * 1e6, 2)
    
    return (
        round(t_fwd * 1e6, 2),      # forward_time_us
        round(t_bwd * 1e6, 2),      # backward_time_us
        t_mlp_fwd_us,               # ffn_forward_time_us
        t_mlp_bwd_us                # ffn_backward_time_us
    )

# ------------------------------
# Main Update Logic
# ------------------------------
def update_json_files(stats_dir, skip_sanity_check=False):
    """
    Update all JSON files in stats_dir with:
    1. New GPU data (H100, MI250X, B300)
    2. non_expert_size field
    3. FFN timing fields
    """
    json_files = glob.glob(os.path.join(stats_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {stats_dir}")
        return False
    
    print(f"Found {len(json_files)} JSON files to update\n")
    
    all_sanity_checks_passed = True
    
    for json_path in json_files:
        print(f"Processing: {os.path.basename(json_path)}")
        
        # Load existing data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract model parameters
        N = data["seq_len"]
        L = data["num_blocks"]
        d = data["embedded_dim"]
        H = data["ffn"]["intermediate_size"]
        E = data["ffn"]["num_experts"]
        k = data["ffn"]["active_experts_k"]
        model_size = data["model_size"]
        
        # Add non_expert_size (equals model_size for all models)
        data["non_expert_size"] = model_size
        
        # Get list of GPUs already in the file
        existing_gpus = set(data["gpus"].keys())
        new_gpus = set(GPU_SPECS.keys()) - existing_gpus
        
        print(f"  Existing GPUs: {sorted(existing_gpus)}")
        print(f"  Adding GPUs: {sorted(new_gpus)}")
        
        # Sanity check for existing GPUs
        if not skip_sanity_check and existing_gpus:
            print(f"  Running sanity checks on existing GPUs...")
            file_sanity_passed = True
            
            for gpu_name in existing_gpus:
                for dtype_str in data["gpus"][gpu_name]:
                    for batch_key in data["gpus"][gpu_name][dtype_str]:
                        B = int(batch_key)
                        
                        # Get old timings
                        old_entry = data["gpus"][gpu_name][dtype_str][batch_key]
                        old_fwd = old_entry.get("forward_time_us")
                        old_bwd = old_entry.get("backward_time_us")
                        
                        # Compute new timings
                        new_fwd, new_bwd, new_ffn_fwd, new_ffn_bwd = compute_times(
                            N, L, d, H, E, k, B, dtype_str, gpu_name
                        )
                        
                        # Check if they match (allow small tolerance for floating point)
                        tolerance = 0.01  # 0.01 microsecond tolerance
                        if old_fwd is not None and abs(old_fwd - new_fwd) > tolerance:
                            print(f"    ⚠️  WARNING: {gpu_name}/{dtype_str}/batch={B}")
                            print(f"       forward_time_us mismatch: old={old_fwd}, new={new_fwd}, diff={abs(old_fwd - new_fwd)}")
                            file_sanity_passed = False
                            all_sanity_checks_passed = False
                        
                        if old_bwd is not None and abs(old_bwd - new_bwd) > tolerance:
                            print(f"    ⚠️  WARNING: {gpu_name}/{dtype_str}/batch={B}")
                            print(f"       backward_time_us mismatch: old={old_bwd}, new={new_bwd}, diff={abs(old_bwd - new_bwd)}")
                            file_sanity_passed = False
                            all_sanity_checks_passed = False
            
            if file_sanity_passed:
                print(f"  ✓ Sanity checks passed for existing GPUs")
            else:
                print(f"  ✗ Sanity checks FAILED for some configurations")
        
        # Process all GPUs
        for gpu_name in GPU_SPECS:
            if gpu_name not in data["gpus"]:
                data["gpus"][gpu_name] = {}
            
            # Process each dtype supported by this GPU
            for dtype_str in GPU_SPECS[gpu_name]["peak_flops"]:
                if dtype_str not in data["gpus"][gpu_name]:
                    data["gpus"][gpu_name][dtype_str] = {}
                
                # Process each batch size
                for B in BATCH_SIZES:
                    batch_key = str(B)
                    
                    # Compute all timings
                    t_fwd, t_bwd, t_ffn_fwd, t_ffn_bwd = compute_times(
                        N, L, d, H, E, k, B, dtype_str, gpu_name
                    )
                    
                    # Update or create entry
                    data["gpus"][gpu_name][dtype_str][batch_key] = {
                        "forward_time_us": t_fwd,
                        "backward_time_us": t_bwd,
                        "ffn_forward_time_us": t_ffn_fwd,
                        "ffn_backward_time_us": t_ffn_bwd
                    }
        
        # Save updated data
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  ✓ Updated and saved\n")
    
    return all_sanity_checks_passed

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update model stats JSON files with new GPU data and FFN timings"
    )
    parser.add_argument(
        "stats_dir",
        help="Path to directory containing JSON files (e.g., ~/DLNetBench/model_stats)"
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip sanity checks on existing GPU timings"
    )
    
    args = parser.parse_args()
    stats_dir = os.path.expanduser(args.stats_dir)
    
    if not os.path.exists(stats_dir):
        print(f"Error: Directory {stats_dir} does not exist!")
        exit(1)
    
    print(f"Updating JSON files in: {stats_dir}\n")
    print("=" * 70)
    
    all_checks_passed = update_json_files(stats_dir, args.skip_sanity_check)
    
    print("=" * 70)
    
    if not args.skip_sanity_check:
        if all_checks_passed:
            print("\n✓ All sanity checks passed!")
            print("✓ All files updated successfully!")
        else:
            print("\n⚠️  WARNING: Some sanity checks failed!")
            print("   Review the warnings above. Files have been updated anyway.")
    else:
        print("\nAll files updated successfully!")
        print("(Sanity checks were skipped)")