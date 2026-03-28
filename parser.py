import json
import pandas as pd
import sbatchman as sbm
import sys
import os
from argparse import ArgumentParser
from collections import defaultdict

# Make sure Python can find ccutils
home = os.getenv("HOME")
sys.path.append(f"{home}")

from ccutils.parser.ccutils_parser import *

NUM_GPUS = 4  # Default number of GPUs per node


def extract_dp_metrics_df(dp_section, job_vars, cluster_name='leonardo', filename=''):
    """
    Extract DP metrics from a Section object and job variables into a Pandas DataFrame.
    Returns a DataFrame with one row per rank per run matching ccutils output format.
    """
    json_data = dp_section.json_data

    # Extract job-level parameters
    world_size = int(job_vars.get("nodes", json_data.get("world_size")))
    
    # Try different variable names for model
    model_name = (job_vars.get("models") or 
                  job_vars.get("dp_model") or 
                  job_vars.get("models_fsdp") or
                  json_data.get("model_name") or 
                  "unknown")
    
    local_batch_size = json_data.get("local_batch_size")
    num_buckets = job_vars.get("num_buckets", json_data.get("num_buckets"))
    
    # Get NCCL parameters - try multiple variable names
    protocol = (job_vars.get("protocol") or 
                job_vars.get("n_protocol") or
                json_data.get("protocol") or 
                "Default")
    
    algorithm = (job_vars.get("algorithm") or 
                 job_vars.get("n_algorithm") or
                 json_data.get("algorithm") or 
                 "Default")
  
    if algorithm != "Default":
        protocol = "Simple"

    channels = (job_vars.get("channels") or 
                job_vars.get("n_channels") or
                json_data.get("channels") or 
                "Default")
    
    threads = (job_vars.get("threads") or 
               job_vars.get("n_threads") or
               json_data.get("threads") or 
               "Default")
    
    backend = json_data.get("backend", "NCCL")
    data_type = json_data.get("data_type", "bfloat16")
    gpu_model = json_data.get("GPU model", "A100")
    device = json_data.get("device", "GPU")
    
    # Message size info
    # Note: ccutils reports 2x the actual message size, so we divide by 2
    msg_avg_raw = json_data.get("msg_size_avg_bytes")
    msg_avg = msg_avg_raw / 2 if msg_avg_raw else None
    msg_std = json_data.get("msg_size_std_bytes")
    total_model_size = json_data.get("total_model_size_params")

    rows = []
    rank_outputs = dp_section.mpi_all_prints["ccutils_rank_json"].rank_outputs

    total_ranks = world_size * NUM_GPUS  # Adjust for total ranks

    for rank, json_str in rank_outputs.items():
        parsed = json.loads(json_str)
        runtimes = parsed["runtimes"]
        barrier_times = parsed.get("barrier_time", [])
        device_energies = parsed.get("device_energies_mj", [])
        hostname = parsed.get("hostname", "unknown")
        device_id = parsed.get("device_id", 0)
        power_traces = parsed.get("power_traces", [])

        # Calculate local rank (rank within node)
        local_rank = rank % NUM_GPUS
        node_id = rank // NUM_GPUS

        for run_idx, runtime in enumerate(runtimes):
            barrier_time = barrier_times[run_idx] if run_idx < len(barrier_times) else 0
            device_energy = device_energies[run_idx] if run_idx < len(device_energies) else 0
            power_trace = power_traces[run_idx] if run_idx < len(power_traces) else []

            row = {
                # ccutils standard fields
                "library": backend,
                "collective": "allreduce",  # DP uses allreduce
                "data_type": data_type,
                "message_size_bytes": msg_avg,
                "message_size_elements": msg_avg / 4 if data_type == "float32" else msg_avg / 2,  # Approximate
                "num_ranks": total_ranks,
                "global_rank": rank,
                "local_rank": local_rank,
                "hostname": hostname,
                "total_nodes": world_size,
                "run_id": run_idx,
                
                # Time measurements
                "runtime_s": runtime,
                
                # Energy measurements
                "device_energy_mj": device_energy,
                
                # Power trace (as JSON string)
                "power_trace": json.dumps(power_trace) if power_trace else "",
                
                # Additional metadata
                "model_name": model_name,
                "local_batch_size": local_batch_size,
                "num_buckets": num_buckets,
                "protocol": protocol,
                "channels": channels,
                "threads": threads,
                "algorithm": algorithm,
                "network": cluster_name,
                "gpu_model": gpu_model,
                "device": device,
                "total_model_size_params": total_model_size,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def extract_fsdp_metrics_df(fsdp_section, job_vars, cluster_name='leonardo', filename=''):
    """
    Extract FSDP metrics from a Section object and job variables into a Pandas DataFrame.
    Returns a DataFrame matching ccutils output format.
    """
    global_data = fsdp_section.json_data

    # Global metrics
    world_size = job_vars.get("nodes")
    
    # Try different variable names for model
    model_name = (job_vars.get("models_fsdp") or 
                  job_vars.get("models") or
                  job_vars.get("dp_model") or
                  global_data.get("model_name") or
                  "unknown")
    
    backend = global_data.get("backend", "NCCL")
    data_type = global_data.get("data_type", "bfloat16")
    
    # Get NCCL parameters - try multiple variable names
    protocol = (job_vars.get("protocol") or 
                job_vars.get("n_protocol") or
                global_data.get("protocol") or 
                "Default")
    
    algorithm = (job_vars.get("algorithm") or 
                 job_vars.get("n_algorithm") or
                 global_data.get("algorithm") or 
                 "Default")
    
    channels = (job_vars.get("channels") or 
                job_vars.get("n_channels") or
                global_data.get("channels") or 
                "Default")
    
    threads = (job_vars.get("threads") or 
               job_vars.get("n_threads") or
               global_data.get("threads") or 
               "Default")
    
    global_params = {
        "network": cluster_name,
        "world_size": world_size,
        "sharding_factor": global_data.get("sharding_factor"),
        "num_replicas": global_data.get("num_replicas"),
        "model_name": model_name,
        "model_size_bytes": global_data.get("model_size_bytes"),
        "local_batch_size": global_data.get("local_batch_size"),
        "num_units": global_data.get("num_units"),
        "fwd_time_per_unit_us": global_data.get("fwd_time_per_unit_us"),
        "bwd_time_per_unit_us": global_data.get("bwd_time_per_unit_us"),
        "allgather_msg_size_bytes": global_data.get("allgather_msg_size_bytes"),
        "reducescatter_msg_size_bytes": global_data.get("reducescatter_msg_size_bytes"),
        "protocol": protocol,
        "algorithm": algorithm,
        "channels": channels,
        "threads": threads,
    }

    rows = []

    for rank, json_str in fsdp_section.mpi_all_prints["ccutils_rank_json"].rank_outputs.items():
        parsed = json.loads(json_str)
        runtimes = parsed.get("runtime", [])
        hostname = parsed.get("hostname", "unknown")
        allgather_times = parsed.get("allgather", [])
        reduce_scatter_times = parsed.get("reduce_scatter", [])
        device_energies = parsed.get("device_energies_mj", [])
        power_traces = parsed.get("power_traces", [])
        
        local_rank = rank % NUM_GPUS
        total_ranks = world_size * NUM_GPUS

        num_runs = len(runtimes)

        for run_idx in range(num_runs):
            device_energy = device_energies[run_idx] if run_idx < len(device_energies) else 0
            power_trace = power_traces[run_idx] if run_idx < len(power_traces) else []

            row = {
                # ccutils standard fields
                "library": backend,
                "collective": "mixed",  # FSDP uses allgather + reduce_scatter
                "data_type": data_type,
                "message_size_bytes": global_params["allgather_msg_size_bytes"],
                "message_size_elements": 0,  # Calculate if needed
                "num_ranks": total_ranks,
                "global_rank": rank,
                "local_rank": local_rank,
                "hostname": hostname,
                "total_nodes": world_size,
                "run_id": run_idx,
                
                # Time measurements
                "total_time_ms": runtimes[run_idx] * 1000 if run_idx < len(runtimes) else 0,
                
                # Energy measurements
                "total_device_energy_mj": device_energy,
                
                # Power trace
                "power_trace": json.dumps(power_trace) if power_trace else "",
                
                # FSDP-specific metadata
                **global_params,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def extract_metrics_df(strategy: str):
    """
    Returns the correct extraction function depending on the strategy.
    """
    if strategy == "fsdp":
        return extract_fsdp_metrics_df
    elif strategy == "dp":
        return extract_dp_metrics_df
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def get_config_key(job_vars, json_data=None):
    """
    Create a configuration key from job variables and json_data.
    Jobs with the same configuration will be grouped together.
    """
    # Try job_vars first (with multiple possible names), then json_data, then "Default"
    if json_data:
        algorithm = (job_vars.get("algorithm") or 
                     job_vars.get("n_algorithm") or
                     json_data.get("algorithm") or 
                     "Default")
        
        protocol = (job_vars.get("protocol") or 
                    job_vars.get("n_protocol") or
                    json_data.get("protocol") or 
                    "Default")
        
        channels = (job_vars.get("channels") or 
                    job_vars.get("n_channels") or
                    json_data.get("channels") or 
                    "Default")
        
        threads = (job_vars.get("threads") or 
                   job_vars.get("n_threads") or
                   json_data.get("threads") or 
                   "Default")
        
        model = (job_vars.get("models") or 
                 job_vars.get("dp_model") or
                 job_vars.get("models_fsdp") or
                 json_data.get("model_name") or 
                 "unknown")
    else:
        algorithm = job_vars.get("algorithm", job_vars.get("n_algorithm", "Default"))
        protocol = job_vars.get("protocol", job_vars.get("n_protocol", "Default"))
        channels = job_vars.get("channels", job_vars.get("n_channels", "Default"))
        threads = job_vars.get("threads", job_vars.get("n_threads", "Default"))
        model = job_vars.get("models", job_vars.get("dp_model", job_vars.get("models_fsdp", "unknown")))
    
    # Create a tuple that represents the configuration
    # Format: (model, algorithm, protocol, channels, threads)
    return (model, algorithm, protocol, channels, threads)


def config_key_to_filename(config_key):
    """
    Convert a configuration key to a filename.
    """
    model, algorithm, protocol, channels, threads = config_key
    
    # Clean up the values for filename
    model_str = str(model).replace('/', '_').replace(' ', '_')
    alg_str = f"alg{algorithm}"
    prot_str = f"prot{protocol}"
    ch_str = f"ch{channels}"
    th_str = f"th{threads}"
    
    return f"{model_str}_{alg_str}_{prot_str}_{ch_str}_{th_str}.csv"


def process_jobs_to_csvs(strategy: str = "dp", output_dir: str = "./csv_outputs"):
    """
    Process all completed jobs and group them by configuration.
    Jobs with the same configuration (algorithm, protocol, channels, threads) 
    will be saved to the same CSV file.
    
    Args:
        strategy: Training strategy ('dp' or 'fsdp')
        output_dir: Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to group dataframes by configuration
    # Key: configuration tuple, Value: list of dataframes
    config_groups = defaultdict(list)
    
    jobs = sbm.jobs_list(status=[sbm.Status.COMPLETED])

    model_var_key = "dp_model" if strategy == "dp" else "fsdp_model"

    # Filter jobs based on strategy
    total_jobs = len(jobs)
    jobs = [job for job in jobs if model_var_key in job.variables]
    filtered_jobs = len(jobs)

    print(f"Strategy filter: {strategy.upper()}")
    print(f"Total completed jobs: {total_jobs}")
    print(f"Jobs matching strategy: {filtered_jobs}")
    print(f"Jobs filtered out: {total_jobs - filtered_jobs}")
    print("=" * 80)

    processed_count = 0
    failed_count = 0

    print("Processing jobs...")
    print("=" * 80)
    
    for job in jobs:
        try:
            job_exp_dir = job.exp_dir
            job_output = job.get_stdout()
        #print(job_output)
            mpi_parser = MPIOutputParser()
            parser_output = mpi_parser.parse_string(job_output)
        except Exception as e:
            print(f"✗ Failed to parse job {job}: {e}")
            failed_count += 1
            continue
        
        section = parser_output.get(strategy)
        
        if not section:
            print(f"✗ No {strategy} section found for job {job}")
            failed_count += 1
            continue
        
        cluster_name = job.cluster_name
        job_name = getattr(job, 'name', '') or os.path.basename(job.exp_dir)
        job_vars = job.variables
        
        
        # Extract metrics for this job
        try:
            df = extract_metrics_df(strategy)(section, job_vars, cluster_name, job_name)
        except Exception as e:
            print(f"✗ Failed to extract metrics job {job}: {e}")
        # Get configuration key for grouping (pass json_data for fallback)
        json_data = section.json_data
        config_key = get_config_key(job_vars, json_data)
        
        # Add to the appropriate group
        config_groups[config_key].append(df)
        
        print(f"✓ Processed job: {job_name} → config: {config_key}")
        processed_count += 1

    print("\n" + "=" * 80)
    print(f"Combining jobs by configuration and saving CSVs...")
    print("=" * 80)
    
    # Now combine all dataframes for each configuration and save
    saved_files = 0
    for config_key, df_list in config_groups.items():
        # Concatenate all dataframes for this configuration
        combined_df = pd.concat(df_list, ignore_index=True)
        
        # Select and order columns to match ccutils format
        ccutils_columns = [
            "library",
            "collective",
            "data_type",
            "message_size_bytes",
            "message_size_elements",
            "num_ranks",
            "global_rank",
            "local_rank",
            "hostname",
            "total_nodes",
            "run_id",
            "total_time_ms",
            "total_device_energy_mj",
            "power_trace",
        ]
        
        # Add any additional metadata columns that exist
        additional_cols = [col for col in combined_df.columns if col not in ccutils_columns]
        final_columns = ccutils_columns + additional_cols
        
        # Reorder dataframe
        combined_df = combined_df[[col for col in final_columns if col in combined_df.columns]]
        
        # Generate filename from configuration
        output_filename = os.path.join(output_dir, config_key_to_filename(config_key))
        
        # Save to CSV
        combined_df.to_csv(output_filename, index=False)
        print(f"✓ Saved {len(df_list)} jobs ({len(combined_df)} rows) to {output_filename}")
        print(f"  Configuration: model={config_key[0]}, alg={config_key[1]}, prot={config_key[2]}, ch={config_key[3]}, th={config_key[4]}")
        saved_files += 1

    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Jobs processed: {processed_count}")
    print(f"  Jobs failed: {failed_count}")
    print(f"  Unique configurations: {len(config_groups)}")
    print(f"  CSV files created: {saved_files}")
    print(f"  Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = ArgumentParser(description="Parse distributed training metrics into ccutils format (grouped by configuration)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="dp",
        choices=["dp", "fsdp"],
        help="Training strategy: 'dp' or 'fsdp'"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./csv_outputs",
        help="Output directory for CSV files"
    )
    
    args = parser.parse_args()
    
    process_jobs_to_csvs(strategy=args.strategy, output_dir=args.output_dir)
