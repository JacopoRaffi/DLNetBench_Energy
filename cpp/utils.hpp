/*********************************************************************
 *
 * Description: 
 * Author: Jacopo Raffi
 *
 *********************************************************************/

#ifndef UTILS_HPP
#define UTILS_HPP

#include <assert.h>
#include <cstdint>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <cerrno>
#include <filesystem>
#include <cstdlib>

#ifdef PROXY_ENABLE_CUDA   
    #include <cuda_runtime.h>
    #include <ccutils/cuda/cuda_macros.hpp>
#endif

#ifdef PROXY_ENABLE_HIP
    #include "tmp_hip_ccutils.hpp" 
    #include <hip/hip_runtime.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;


#include <signal.h>
#include <atomic>


volatile sig_atomic_t end = 0;

static void signal_handler(int sig){
    if (sig == SIGINT || sig == SIGQUIT || sig == SIGTERM || sig == SIGUSR1)
        end = 1;
}

void install_signal_handlers(void) {
    struct sigaction s;
    memset(&s, 0, sizeof(s));

    s.sa_handler = signal_handler;
    sigfillset(&s.sa_mask); // block all signals while handler runs

    if (sigaction(SIGINT,  &s, NULL) < 0) { perror("sigaction SIGINT");  exit(EXIT_FAILURE); }
    if (sigaction(SIGQUIT, &s, NULL) < 0) { perror("sigaction SIGQUIT"); exit(EXIT_FAILURE); }
    if (sigaction(SIGTERM, &s, NULL) < 0) { perror("sigaction SIGTERM"); exit(EXIT_FAILURE); }
    if (sigaction(SIGUSR1, &s, NULL) < 0) { perror("sigaction SIGUSR1"); exit(EXIT_FAILURE); }
}

/**
 * Get the base path of the DNNProxy folder.
 * 
 * @param argc Number of command-line arguments.
 * @param argv Command-line arguments.
 * @param rank MPI rank (for error messages).
 * @return The fs::path to the DNNProxy folder.
 * @throws std::runtime_error if the folder does not exist or HOME is not set.
 */
fs::path get_dnnproxy_base_path(char* input_path) {
    fs::path base_path;
    fs::path cwd;
    try {
        cwd = fs::current_path();
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error: cannot determine current working directory.\n";
    }

    base_path = cwd / input_path;
    if (!fs::exists(base_path) || !fs::is_directory(base_path)) {
        std::cerr << "Error: DNNProxy folder does not exist at: " << base_path << "\n";
        throw std::runtime_error("DNNProxy folder not found");
    }
    return base_path;
}

// helper per parsare device list passata come stringa "0,1,2"
std::vector<int> parse_devices(const std::string& s) {
    std::vector<int> result;
    if (s.empty()) return result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stoi(item));
    }
    return result;
}

// function to set the local device for each MPI process based on the local rank and the provided device list (if any)
inline int set_local_device(MPI_Comm global_comm, const char* devices_str){
    MPI_Comm local_comm;
    MPI_Comm_split_type(global_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);

    int local_rank = 0;
    int local_size = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    std::vector<int> device_list;

    if (devices_str && std::string(devices_str).size() > 0) {
        device_list = parse_devices(devices_str);
    } else {
#if defined(PROXY_ENABLE_CUDA)
        int num_gpus = 0;
        cudaGetDeviceCount(&num_gpus);
        if (num_gpus <= 0) throw std::runtime_error("No CUDA devices available");
        for (int i = 0; i < num_gpus; i++) device_list.push_back(i);

#elif defined(PROXY_ENABLE_HIP)
        int num_gpus = 0;
        hipGetDeviceCount(&num_gpus);
        if (num_gpus <= 0) throw std::runtime_error("No HIP devices available");
        for (int i = 0; i < num_gpus; i++) device_list.push_back(i);
#else
        return 0; // CPU case, no device to set
#endif
    }

    assert(local_size <= static_cast<int>(device_list.size()));
    int device_index = device_list[local_rank];

    // --- set device ---
#if defined(PROXY_ENABLE_CUDA)
    CCUTILS_CUDA_CHECK(cudaSetDevice(device_index));
#elif defined(PROXY_ENABLE_HIP)
    CCUTILS_HIP_CHECK(hipSetDevice(device_index));
#endif

    MPI_Comm_free(&local_comm);

    return device_index;
}

// consider the warm-up time and adjust the number of runs accordingly to meet the minimum execution time requirement
// avoid the first 2 warm-up iterations
int estimate_runs(std::vector<float>& warmup_times, uint64_t min_exectime){
    float warmup_time = 0;
    for (size_t i = 2; i < warmup_times.size(); i++) warmup_time += warmup_times[i];
    float avg_time_per_run = warmup_time / (warmup_times.size() - 2);
    
    float global_avg_time;
    MPI_Allreduce(&avg_time_per_run, &global_avg_time, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    global_avg_time /= warmup_times.size();
    int estimated_runs = static_cast<int>(std::ceil(min_exectime / global_avg_time));

    // rank 0 broadcast estimated runs to all other processes
    MPI_Bcast(&estimated_runs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return estimated_runs;
}


/**
 * @brief Computes the average and standard deviation of a list of message sizes.
 *
 * @param sizes A vector containing the sizes of messages.
 * @param sharding_multiplier An optional multiplier to scale each size (default is 1).
 * @return A pair where the first element is the average size and the second is the standard deviation.
 */
std::pair<float, float> compute_msg_stats(const std::vector<uint64_t>& sizes, uint sharding_multiplier = 1) {
    float avg = 0.0f;
    for (uint64_t s : sizes)
        avg += s * sharding_multiplier;
    avg /= sizes.size();

    float stddev = 0.0f;
    for (uint64_t s : sizes) {
        float diff = s * sharding_multiplier - avg;
        stddev += diff * diff;
    }
    stddev = std::sqrt(stddev / sizes.size());

    return {avg, stddev};
}

static char default_devices[] = "";
static char default_gpu[]   = "B200";
static char default_dtype[] = "bfloat16";

std::map<std::string, uint64_t> get_model_stats(std::string filename, 
                                                  std::string gpu, 
                                                  std::string dtype, 
                                                  uint64_t batch_size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << std::strerror(errno) << "\n";
        throw std::runtime_error("Could not open model stats file: " + filename);
    }

    nlohmann::json j;
    file >> j;

    std::string batch_key = std::to_string(batch_size);

    // Validate keys exist
    if (!j["gpus"].contains(gpu))
        throw std::runtime_error("GPU not found in JSON: " + gpu);
    if (!j["gpus"][gpu].contains(dtype))
        throw std::runtime_error("dtype not found in JSON: " + dtype);
    if (!j["gpus"][gpu][dtype].contains(batch_key))
        throw std::runtime_error("batch_size not found in JSON: " + batch_key);

    auto& timing = j["gpus"][gpu][dtype][batch_key];

    std::map<std::string, uint64_t> model_stats;
    model_stats["modelSize"]         = j["model_size"].get<uint64_t>();
    model_stats["nonExpertModelSize"]= j["model_size"].get<uint64_t>();
    model_stats["sequenceLength"]    = j["seq_len"].get<uint64_t>();
    model_stats["embeddedDim"]       = j["embedded_dim"].get<uint64_t>();
    model_stats["batchSize"]         = batch_size;
    model_stats["experts"]           = j["ffn"]["num_experts"].get<uint64_t>();
    model_stats["avgForwardTime"]    = (uint64_t)timing["forward_time_us"].get<double>();
    model_stats["avgBackwardTime"]   = (uint64_t)timing["backward_time_us"].get<double>();

    return model_stats;
}

uint count_layers(std::string filename) {
    std::ifstream f(filename);
    json data = json::parse(f);
    return data["num_blocks"].get<uint>();
}

#endif // UTILS_HPP