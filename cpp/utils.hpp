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

/**
 * @brief Extracts the value part from a line formatted as "key: value".
 *
 * @param line The input line containing a key-value pair.
 * @return The extracted value as a string, trimmed of whitespace.
 */
std::string extract_value(const std::string &line) {
    size_t pos = line.find(':');
    if (pos == std::string::npos)
        return ""; // no delimiter → empty string

    std::string value = line.substr(pos + 1);

    // trim whitespace
    value.erase(0, value.find_first_not_of(" \t\r\n"));
    value.erase(value.find_last_not_of(" \t\r\n") + 1);

    return value;
}

/**
 * @brief Reads model statistics from a stats file and returns them in a map
 *TODO: change thi func and use a JSON instead
 * The file has this format:
 *   - Forward Flops:<value>
 *   - Backward Flops:<value>
 *   - Model Size:<value>
 *   - Average Forward Time (s):<value>
 *   - Average Backward Time (s):<value>
 *   - Batch Size:<value>
 *   - FFN_Average_Forward_Time (us):15125
 *   - FFN_Average_Backward_Time (us):24139
 *   - Experts: 4
 * Each parsed value is stored in the returned map with keys:
 * "forwardFlops", "backwardFlops", "modelSize", "avgForwardTime", "avgBackwardTime", "batchSize", "ffn_avgForwardTime", "ffn_avgBackwardTime", "experts".
 *
 * @param file_name Path to the model statistics file.
 * @return std::map<std::string, float>.
 */
std::map<std::string, uint64_t> get_model_stats(std::string filename){
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << std::strerror(errno) << "\n";
        throw std::runtime_error("Could not open model stats file: " + filename);
    }

    std::map<std::string, uint64_t> model_stats;
    std::string line;

    std::getline(file, line);
    uint64_t forwardFlops = std::stoull(extract_value(line));

    // Backward Flops
    std::getline(file, line);
    uint64_t backwardFlops = std::stoull(extract_value(line));

    // Model Size
    std::getline(file, line);
    uint64_t modelSize = std::stoull(extract_value(line));

    // non expert size
    std::getline(file, line);
    uint64_t non_expert_size = std::stoull(extract_value(line));

    // Average Forward Time  (should be double, not uint64_t)
    std::getline(file, line);
    uint64_t avgForwardTime = std::stod(extract_value(line));

    // Average Backward Time (should be double)
    std::getline(file, line);
    uint64_t avgBackwardTime = std::stod(extract_value(line));

    // Batch size
    std::getline(file, line);
    uint64_t batch_size = std::stoull(extract_value(line));

    // FFN Average Forward Time (us)
    std::getline(file, line);
    uint64_t ffn_avgForwardTime = std::stoull(extract_value(line));

    // FFN Average Backward Time (us)
    std::getline(file, line);
    uint64_t ffn_avgBackwardTime = std::stoull(extract_value(line));

    std::getline(file, line); // Experts (optional)
    uint64_t experts = std::stoull(extract_value(line));

    std::getline(file, line);
    uint64_t sequence_length = std::stoull(extract_value(line));

    std::getline(file, line); 
    uint64_t embedded_dim = std::stoull(extract_value(line));

    model_stats["forwardFlops"] = forwardFlops;
    model_stats["backwardFlops"] = backwardFlops;
    model_stats["modelSize"] = modelSize;
    model_stats["avgForwardTime"] = avgForwardTime;
    model_stats["avgBackwardTime"] = avgBackwardTime;
    model_stats["batchSize"] = batch_size;
    model_stats["ffn_avgForwardTime"] = ffn_avgForwardTime;
    model_stats["ffn_avgBackwardTime"] = ffn_avgBackwardTime;
    model_stats["experts"] = experts;
    model_stats["sequenceLength"] = sequence_length;
    model_stats["embeddedDim"] = embedded_dim;
    model_stats["nonExpertModelSize"] = non_expert_size;
    
    return model_stats;   
}


/**
 * @brief Extracts the number of layers from a model configuration JSON file.
 * The function reads the JSON file and sums up the number of encoder and decoder blocks
 * to determine the total number of layers in the model.
 * @param filename The path to the JSON configuration file.
 * @return The total number of layers in the model.
*/
uint count_layers(std::string filename){
    std::ifstream f(filename);
    json data = json::parse(f);

    uint num_layers = 0;

    if(data.contains("num_encoder_blocks")){
        num_layers += data["num_encoder_blocks"].get<uint>();
    }

    if(data.contains("num_decoder_blocks")){
        num_layers += data["num_decoder_blocks"].get<uint>();
    }

    return num_layers;
}

#endif // UTILS_HPP