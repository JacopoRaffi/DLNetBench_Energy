/********************************************************************* 
 * 
 * Description: C++/MPI proxy for Transformer-based models distributed training
 *              with hybrid data, pipeline, and expert parallelism (DP+PP+EP) Assume 1 PP-stage 1 Transformer block
 * 
 *********************************************************************/ 

#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "../netcommunicators.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

// CCUTILS headers
#include <ccutils/mpi/mpi_timers.hpp>
#include <ccutils/mpi/mpi_macros.hpp>
#include <ccutils/macros.hpp>

#ifdef PROXY_ENABLE_CUDA
#include <ccutils/cuda/cuda_macros.hpp>
#endif

#ifdef PROXY_ENABLE_ONECCL
#include <oneapi/ccl.hpp>
#include <CL/sycl.hpp>
#endif

// Project headers
#include "../utils.hpp"
#include "../data_types.hpp"
#include "../proxy_classes.hpp"

#ifdef PROXY_ENABLE_NCCL
#include <nccl.h>
Proxy_CommType world_comm;
#endif

#ifdef PROXY_ENABLE_RCCL
#include <rccl.h>
Proxy_CommType world_comm;
#endif

// Device to use
#if defined(PROXY_ENABLE_CUDA) || defined(PROXY_ENABLE_HIP)
constexpr Device device = Device::GPU;
#else
constexpr Device device = Device::CPU;
#endif

// Default values
#define WARM_UP 3
#define RUNS 5
#define POWER_SAMPLING_RATE_MS 5

CCUTILS_MPI_TIMER_DEF(runtime)
CCUTILS_MPI_TIMER_DEF(pp_comm)
CCUTILS_MPI_TIMER_DEF(dp_comm)
CCUTILS_MPI_TIMER_DEF(ep_comm) 
CCUTILS_MPI_TIMER_DEF(dp_ep_comm)

/**
 * @brief Simulates one iteration of hybrid DP+PP+EP training using GPipe schedule.
 * 
 * EXPERT PARALLELISM EXPLANATION:
 * - In MoE models, different experts process different tokens
 * - Each device hosts a subset of experts
 * - All-to-All communication routes tokens to appropriate expert devices
 * - Assumption: tokens are evenly distributed among experts
 * 
 * @param num_microbatches Number of micro-batches (gradient accumulation steps)
 * @param stage_id Pipeline stage ID for this process
 * @param num_stage Total number of pipeline stages
 * @param pipe_msg_size Size of activations/gradients passed between stages (per microbatch)
 * @param fwd_rt Forward pass runtime per micro-batch (in microseconds)
 * @param bwd_rt Backward pass runtime per micro-batch (in microseconds)
 * @param grad_ptr Pointer to gradient buffer for DP all-reduce
 * @param sum_grad_ptr Pointer to reduced gradient buffer
 * @param dp_allreduce_size Size of gradient buffer for DP all-reduce
 * @param non_expert_size Size of non-expert parameters (for DP all-reduce inside a EP group)
 * @param fwd_send_buff Forward activation send buffer
 * @param fwd_recv_buff Forward activation receive buffer
 * @param bwd_send_buff Backward gradient send buffer
 * @param bwd_recv_buff Backward gradient receive buffer
 * @param ep_send_buffer Pointer to expert parallel send buffer (for token routing)
 * @param ep_recv_buffer Pointer to expert parallel receive buffer (for token routing)
 * @param ep_alltoall_size Size of expert parallel all-to-all per device
 * @param num_experts Total number of experts in the model
 * @param dp_communicator Communicator for data-parallel all-reduce
 * @param pp_communicator Communicator for pipeline-parallel p2p
 * @param ep_communicator Communicator for expert-parallel all-to-all
 * @return int Always returns 0
 */
int run_data_pipe_expert_parallel(
    int num_microbatches, 
    int stage_id, 
    int num_stage,
    int layers_per_stage,
    uint64_t pipe_msg_size,
    uint64_t fwd_rt,
    uint64_t bwd_rt,
    Tensor<_FLOAT, device>* grad_ptr,
    Tensor<_FLOAT, device>* sum_grad_ptr,
    uint64_t dp_allreduce_size,
    uint64_t non_expert_size,                     
    Tensor<_FLOAT, device>* fwd_send_buff,
    Tensor<_FLOAT, device>* fwd_recv_buff,
    Tensor<_FLOAT, device>* bwd_send_buff,
    Tensor<_FLOAT, device>* bwd_recv_buff,
    Tensor<_FLOAT, device>* ep_send_buffer,
    Tensor<_FLOAT, device>* ep_recv_buffer,
    uint64_t ep_alltoall_size,
    int num_experts,
    ProxyCommunicator* dp_communicator,
    ProxyCommunicator* pp_communicator,
    ProxyCommunicator* ep_communicator){
    
    // GPipe Pipeline Schedule
    // Forward pass for all micro-batches
    for(int i = 0; i < num_microbatches; i++){
        if(stage_id == 0){
            // First stage: compute then send
            usleep(fwd_rt);
            CCUTILS_MPI_TIMER_START(pp_comm)
            pp_communicator->send(fwd_send_buff->data, pipe_msg_size, stage_id+1);
            CCUTILS_MPI_TIMER_STOP(pp_comm)
        } 
        else if(stage_id == num_stage-1){
            // Last stage: receive then compute
            CCUTILS_MPI_TIMER_START(pp_comm)
            pp_communicator->recv(fwd_recv_buff->data, pipe_msg_size, stage_id-1);
            CCUTILS_MPI_TIMER_STOP(pp_comm)
            usleep(fwd_rt);
        } 
        else{
            // Middle stages: receive, compute, send
            CCUTILS_MPI_TIMER_START(pp_comm)
            pp_communicator->recv(fwd_recv_buff->data, pipe_msg_size, stage_id-1);
            CCUTILS_MPI_TIMER_STOP(pp_comm)
            usleep(fwd_rt);
            CCUTILS_MPI_TIMER_START(pp_comm)
            pp_communicator->send(fwd_send_buff->data, pipe_msg_size, stage_id+1);
            CCUTILS_MPI_TIMER_STOP(pp_comm)
        }
        // Expert parallel communication during forward pass
        // In MoE layers: All-to-All to route tokens to experts
        // Assuming MoE layers occur at regular intervals (e.g., every other layer)
        // For simplicity, we do 2 all-to-all operations per microbatch:
        // 1. Route tokens TO experts (before expert computation)
        // 2. Route tokens FROM experts (after expert computation)
        for(int ep_iter = 0; ep_iter < (2*layers_per_stage); ep_iter++){
            CCUTILS_MPI_TIMER_START(ep_comm)
            ep_communicator->Alltoall(ep_send_buffer->data, ep_alltoall_size, ep_recv_buffer->data, ep_alltoall_size);
            CCUTILS_MPI_TIMER_STOP(ep_comm)
        }
    }
    
    // Backward pass for all micro-batches
    for(int i = 0; i < num_microbatches; i++){
        if(stage_id == 0){
            // First stage: receive then compute
            CCUTILS_MPI_TIMER_START(pp_comm)
            pp_communicator->recv(bwd_recv_buff->data, pipe_msg_size, stage_id+1);
            CCUTILS_MPI_TIMER_STOP(pp_comm)
            usleep(bwd_rt);
        } 
        else if(stage_id == num_stage-1){
            // Last stage: compute then send
            usleep(bwd_rt);
            CCUTILS_MPI_TIMER_START(pp_comm)
            pp_communicator->send(bwd_send_buff->data, pipe_msg_size, stage_id-1);
            CCUTILS_MPI_TIMER_STOP(pp_comm)
        } 
        else{
            // Middle stages: receive, compute, send
            CCUTILS_MPI_TIMER_START(pp_comm)
            pp_communicator->recv(bwd_recv_buff->data, pipe_msg_size, stage_id+1);
            CCUTILS_MPI_TIMER_STOP(pp_comm)
            usleep(bwd_rt);
            CCUTILS_MPI_TIMER_START(pp_comm)
            pp_communicator->send(bwd_send_buff->data, pipe_msg_size, stage_id-1);
            CCUTILS_MPI_TIMER_STOP(pp_comm)
        }        
        // Expert parallel communication during backward pass
        // All-to-All for gradient routing back through experts
        for(int ep_iter = 0; ep_iter < (2*layers_per_stage); ep_iter++){
            CCUTILS_MPI_TIMER_START(ep_comm)
            ep_communicator->Alltoall(ep_send_buffer->data, ep_alltoall_size, ep_recv_buffer->data, ep_alltoall_size);
            CCUTILS_MPI_TIMER_STOP(ep_comm)
        }
    }
    CCUTILS_MPI_TIMER_START(dp_ep_comm)
    ep_communicator->Allreduce(grad_ptr->data, sum_grad_ptr->data, non_expert_size);
    CCUTILS_MPI_TIMER_STOP(dp_ep_comm)

    CCUTILS_MPI_TIMER_START(dp_comm)
    dp_communicator->Allreduce(grad_ptr->data, sum_grad_ptr->data, dp_allreduce_size);
    CCUTILS_MPI_TIMER_STOP(dp_comm)
    
    return 0;
}

#define REQUIRED_ARGS                                                                       \
    REQUIRED_STRING_ARG(model_name, "model", "Name of the model to use")                    \
    REQUIRED_INT_ARG(num_stages, "num_stages", "Number of pipeline stages")                 \
    REQUIRED_INT_ARG(num_microbatches, "num_microbatches", "Number of microbatches")        \
    REQUIRED_INT_ARG(num_expert_shards, "num_expert_shards", "Number of expert parallel shards (number of devices per stage that share experts)") \
    REQUIRED_STRING_ARG(base_path, "base_path", "Base path for the repository")  
    
static char default_devices[] = "";

#define OPTIONAL_ARGS                                                                           \
    OPTIONAL_INT_ARG(warmup, WARM_UP, "-w", "warmups", "Number of warm-up iterations")          \
    OPTIONAL_INT_ARG(runs, RUNS, "-r", "runs", "Number of iterations to run")                   \
    OPTIONAL_STRING_ARG(devices, default_devices, "-d", "devices", "Comma-separated list of devices")  \
    OPTIONAL_INT_ARG(min_exectime, 0, "-m", "min_exectime", "Minimum total execution time in seconds (overrides runs)")

#define BOOLEAN_ARGS \
    BOOLEAN_ARG(help, "-h", "Show help")

#include <ccutils/easyargs.hpp>


int main(int argc, char* argv[]) {
    int rank, world_size;
    int num_stage;
    int num_microbatches;
    int num_expert_shards;  // Number of expert parallel groups
    
    args_t args = make_default_args();
    if (!parse_args(argc, argv, &args) || args.help) {
        print_help(argv[0]);
        return 1;
    }
    
    std::string model_name = args.model_name;
    num_stage = args.num_stages;
    num_microbatches = args.num_microbatches;
    num_expert_shards = args.num_expert_shards;
    uint warmup = args.warmup;
    uint runs = args.runs;
    
    // --- Construct model stats file path ---
    fs::path repo_path = get_dnnproxy_base_path(args.base_path);
    fs::path file_path = repo_path / "model_stats" / (model_name + ".txt");
    std::string strip_model_name = model_name.substr(0, model_name.find_last_of('_'));
    strip_model_name = strip_model_name.substr(0, strip_model_name.find_last_of('_'));

    fs::path model_architecture_path = repo_path / "models" / (strip_model_name + ".json");

    uint num_layers = count_layers(model_architecture_path);
    
    if (!fs::exists(file_path)) {
        std::cerr << "Error: model stats file does not exist: " << file_path << "\n";
        return -1;
    }
    
    std::map<std::string, uint64_t> model_stats = get_model_stats(file_path);
    
    // Get model stats from file
    uint64_t fwd_rt_whole_model = model_stats["avgForwardTime"]; // in us
    uint64_t bwd_rt_whole_model = model_stats["avgBackwardTime"]; // in us
    uint local_batch_size = model_stats["batchSize"];
    uint64_t total_model_size = model_stats["modelSize"]; // number of parameters
    uint64_t sequence_length = model_stats["sequenceLength"]; // sequence length
    uint64_t embedded_dim = model_stats["embeddedDim"]; // hidden dimension size
    uint64_t non_expert_model_size = model_stats["nonExpertModelSize"]; // size of non-expert parameters
    uint64_t expert_params = total_model_size - non_expert_model_size;

    float sample_size_bytes = sequence_length * embedded_dim * sizeof(_FLOAT);
    
    // Get number of experts from model stats
    int num_experts = model_stats["experts"];
    if (num_experts <= 0) {
        std::cerr << "Error: Invalid number of experts in model stats: " << num_experts << "\n";
        return -1;
    }
    
    assert(num_layers % num_stage == 0);
    assert(local_batch_size % num_microbatches == 0);
    assert(num_experts % num_expert_shards == 0);
    
#ifdef PROXY_ENABLE_ONECCL
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#else
    MPI_Init(&argc, &argv);
#endif
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Check that world_size = num_stages * num_expert_shards * dp_size
    assert(world_size % (num_stage * num_expert_shards) == 0);
    int dp_size = world_size / (num_stage * num_expert_shards);
    
    CCUTILS_MPI_INIT
    print_topology_graph(MPI_COMM_WORLD);
    // Create DP, PP, and EP communicators
    // Hierarchy: world_size = num_stages * num_expert_shards * dp_size
    // Layout: [DP replicas] x [Pipeline stages] x [Expert shards]
    
    // Calculate position in 3D grid
    int ep_id = rank % num_expert_shards;
    int stage_id = (rank / num_expert_shards) % num_stage;
    int dp_id = rank / (num_expert_shards * num_stage);
    
    // Create EP communicator: groups GPUs with same (stage_id, dp_id)
    // These devices work together to handle all experts
    int ep_color = dp_id * num_stage + stage_id;
    MPI_Comm ep_comm;
    MPI_Comm_split(MPI_COMM_WORLD, ep_color, rank, &ep_comm);
    
    // Create PP communicator: groups GPUs with same (dp_id, ep_id)
    int pp_color = dp_id * num_expert_shards + ep_id;
    MPI_Comm pp_comm;
    MPI_Comm_split(MPI_COMM_WORLD, pp_color, rank, &pp_comm);
    
    // Create DP communicator: groups GPUs with same (stage_id, ep_id)
    int dp_color = stage_id * num_expert_shards + ep_id;
    MPI_Comm dp_comm;
    MPI_Comm_split(MPI_COMM_WORLD, dp_color, rank, &dp_comm);
    
    int dp_rank, pp_rank, ep_rank;
    MPI_Comm_rank(dp_comm, &dp_rank);
    MPI_Comm_rank(pp_comm, &pp_rank);
    MPI_Comm_rank(ep_comm, &ep_rank);
 

    // Compute per-stage and per-microbatch runtimes
    uint64_t fwd_rt_per_stage = fwd_rt_whole_model / num_stage;
    uint64_t bwd_rt_per_stage = bwd_rt_whole_model / num_stage;
    
    uint64_t fwd_rt_per_microbatch = fwd_rt_per_stage / num_microbatches;
    uint64_t bwd_rt_per_microbatch = bwd_rt_per_stage / num_microbatches;
    
    // Pipeline message size: activations for batch_size/num_microbatches samples
    uint64_t samples_per_microbatch = local_batch_size / num_microbatches;
    uint64_t pipe_msg_size = (uint64_t)(sequence_length * samples_per_microbatch * embedded_dim);
    
    // EP all-to-all size calculation:
    // Each device sends tokens to all other expert devices
    // Assuming uniform distribution: total_tokens / num_expert_shards
    // For one microbatch: (samples_per_microbatch * tokens_per_sample) tokens total
    uint64_t tokens_per_microbatch = samples_per_microbatch * sequence_length;
    // Each device sends to every other device (including self)
    // Size per device pair: (tokens_per_microbatch / num_expert_shards) * embedded_dim
    int top_k=2;
    int layers_per_stage = num_layers/num_stage;
    uint64_t ep_alltoall_size = (tokens_per_microbatch * top_k * embedded_dim) / (num_expert_shards);
    
    uint64_t non_expert_size = non_expert_model_size / num_stage;
    uint64_t expert_size = (expert_params / num_stage) / num_expert_shards;
    uint64_t dp_allreduce_size = non_expert_size + expert_size;
    
    int my_device = set_local_device(MPI_COMM_WORLD, args.devices);
    CCUTILS_MPI_ALL_PRINT(fprintf(fp, "Using device %d\n", my_device);)

#ifdef PROXY_ENABLE_CCL
    // Initialize CCL for DP communicator
    ncclUniqueId dp_id_nccl;
    if (dp_rank == 0) {
        ncclGetUniqueId(&dp_id_nccl);
    }
    MPI_Bcast(&dp_id_nccl, sizeof(dp_id_nccl), MPI_BYTE, 0, dp_comm);
    
    Proxy_CommType dp_world_comm;
    ncclCommInitRank(&dp_world_comm, dp_size, dp_id_nccl, dp_rank);
    CCLCommunicator* dp_communicator = new CCLCommunicator(dp_world_comm, 1);
    
    // Initialize CCL for PP communicator
    ncclUniqueId pp_id_nccl;
    int pp_size;
    MPI_Comm_size(pp_comm, &pp_size);
    if (pp_rank == 0) {
        ncclGetUniqueId(&pp_id_nccl);
    }
    MPI_Bcast(&pp_id_nccl, sizeof(pp_id_nccl), MPI_BYTE, 0, pp_comm);
    
    Proxy_CommType pp_world_comm;
    ncclCommInitRank(&pp_world_comm, pp_size, pp_id_nccl, pp_rank);
    CCLCommunicator* pp_communicator = new CCLCommunicator(pp_world_comm, 1);
    
    // Initialize CCL for EP communicator
    ncclUniqueId ep_id_nccl;
    int ep_size;
    MPI_Comm_size(ep_comm, &ep_size);
    if (ep_rank == 0) {
        ncclGetUniqueId(&ep_id_nccl);
    }
    MPI_Bcast(&ep_id_nccl, sizeof(ep_id_nccl), MPI_BYTE, 0, ep_comm);
    
    Proxy_CommType ep_world_comm;
    ncclCommInitRank(&ep_world_comm, ep_size, ep_id_nccl, ep_rank);
    CCLCommunicator* ep_communicator = new CCLCommunicator(ep_world_comm, 1);
    
#elif defined(PROXY_ENABLE_ONECCL)
    // Select GPU device
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    int num_gpus = gpus.size();

    // DP communicator
    sycl::device dev = gpus[rank % num_gpus];
    DeviceManager::init(dev);
    sycl::queue& queue = DeviceManager::get_queue();
    sycl::context ctx = queue.get_context();

    ccl::init();
    ccl::device ccl_dev = ccl::create_device(dev);
    ccl::context ccl_ctx = ccl::create_context(ctx);

    // DP communicator
    ccl::shared_ptr_class<ccl::kvs> dp_kvs;
    if (dp_rank == 0) dp_kvs = ccl::create_main_kvs();

    ccl::kvs::address_type dp_addr;
    if (dp_rank == 0) dp_kvs->get_address();

    MPI_Bcast(dp_addr.data(), dp_addr.size(), MPI_BYTE, 0, dp_comm);

    if (dp_rank != 0) dp_kvs = ccl::create_kvs(dp_addr);

    auto dp_world_comm = ccl::create_communicator(dp_size, dp_rank, ccl_dev, ccl_ctx, dp_kvs);
    OneCCLCommunicator* dp_communicator = new OneCCLCommunicator(std::move(dp_world_comm), ctx, dev, 1);

    // PP communicator
    int pp_size;
    MPI_Comm_size(pp_comm, &pp_size);

    ccl::shared_ptr_class<ccl::kvs> pp_kvs;
    if (pp_rank == 0) pp_kvs = ccl::create_main_kvs();

    ccl::kvs::address_type pp_addr;
    if (pp_rank == 0) pp_addr = pp_kvs->get_address();

    MPI_Bcast(pp_addr.data(), pp_addr.size(), MPI_BYTE, 0, pp_comm);

    if (pp_rank != 0) pp_kvs = ccl::create_kvs(pp_addr);

    auto pp_world_comm = ccl::create_communicator(pp_size, pp_rank, ccl_dev, ccl_ctx, pp_kvs);
    OneCCLCommunicator* pp_communicator = new OneCCLCommunicator(std::move(pp_world_comm), ctx, dev, 1);

    // EP communicator
    int ep_size;
    MPI_Comm_size(ep_comm, &ep_size);

    ccl::shared_ptr_class<ccl::kvs> ep_kvs;
    if (ep_rank == 0) ep_kvs = ccl::create_main_kvs();

    ccl::kvs::address_type ep_addr;
    if (ep_rank == 0) ep_addr = ep_kvs->get_address();

    MPI_Bcast(ep_addr.data(), ep_addr.size(), MPI_BYTE, 0, ep_comm);    

    if (ep_rank != 0) ep_kvs = ccl::create_kvs(ep_addr);

    auto ep_world_comm = ccl::create_communicator(ep_size, ep_rank, ccl_dev, ccl_ctx, ep_kvs);
    OneCCLCommunicator* ep_communicator = new OneCCLCommunicator(std::move(ep_world_comm), ctx, dev, 1);

#else
    MPICommunicator* dp_communicator = new MPICommunicator(dp_comm, MPI_FLOAT, 1);
    MPICommunicator* pp_communicator = new MPICommunicator(pp_comm, MPI_FLOAT, 1);
    MPICommunicator* ep_communicator = new MPICommunicator(ep_comm, MPI_FLOAT, 1);
#endif
    
    // Allocate buffers
    Tensor<_FLOAT, device>* grad_ptr = new Tensor<_FLOAT, device>(dp_allreduce_size);
    Tensor<_FLOAT, device>* sum_grad_ptr = new Tensor<_FLOAT, device>(dp_allreduce_size);
    
    Tensor<_FLOAT, device>* fwd_send_buff = new Tensor<_FLOAT, device>(pipe_msg_size);
    Tensor<_FLOAT, device>* fwd_recv_buff = new Tensor<_FLOAT, device>(pipe_msg_size);
    Tensor<_FLOAT, device>* bwd_send_buff = new Tensor<_FLOAT, device>(pipe_msg_size);
    Tensor<_FLOAT, device>* bwd_recv_buff = new Tensor<_FLOAT, device>(pipe_msg_size);
    
    Tensor<_FLOAT, device>* ep_send_buffer = new Tensor<_FLOAT, device>(ep_alltoall_size * num_expert_shards);
    Tensor<_FLOAT, device>* ep_recv_buffer = new Tensor<_FLOAT, device>(ep_alltoall_size * num_expert_shards);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Warmup
    std::vector<float> warmup_times;
    for(int wmp = 0; wmp < warmup; wmp++){
        float start_time = MPI_Wtime();
        run_data_pipe_expert_parallel(num_microbatches, stage_id, num_stage, layers_per_stage, pipe_msg_size,
                              fwd_rt_per_microbatch, bwd_rt_per_microbatch,
                              grad_ptr, sum_grad_ptr, dp_allreduce_size, non_expert_size,
                              fwd_send_buff, fwd_recv_buff, bwd_send_buff, bwd_recv_buff,
                              ep_send_buffer, ep_recv_buffer, ep_alltoall_size, num_experts,
                              dp_communicator, pp_communicator, ep_communicator);
        float end_time = MPI_Wtime();
        warmup_times.push_back(end_time - start_time);
    }

    if(args.min_exectime > 0) {
        runs = estimate_runs(warmup_times, args.min_exectime);
        CCUTILS_MPI_PRINT_ONCE(std::cout << "Estimated runs based on warm-up times to meet minimum execution time: " << runs << std::endl;)
    }

    #ifdef PROXY_LOOP
    while(true){
        run_data_pipe_expert_parallel(num_microbatches, stage_id, num_stage, layers_per_stage, pipe_msg_size,
                              fwd_rt_per_microbatch, bwd_rt_per_microbatch,
                              grad_ptr, sum_grad_ptr, dp_allreduce_size, non_expert_size,
                              fwd_send_buff, fwd_recv_buff, bwd_send_buff, bwd_recv_buff,
                              ep_send_buffer, ep_recv_buffer, ep_alltoall_size, num_experts,
                              dp_communicator, pp_communicator, ep_communicator);
    }
    #else
    // clear vectors before timed runs
    __timer_vals_pp_comm.clear();
    __timer_vals_dp_comm.clear();
    __timer_vals_ep_comm.clear();
    __timer_vals_dp_ep_comm.clear();
    for(int iter = 0; iter < runs; iter++){
        CCUTILS_MPI_TIMER_START(runtime)
        
        run_data_pipe_expert_parallel(num_microbatches, stage_id, num_stage, layers_per_stage, pipe_msg_size,
                              fwd_rt_per_microbatch, bwd_rt_per_microbatch,
                              grad_ptr, sum_grad_ptr, dp_allreduce_size, non_expert_size,
                              fwd_send_buff, fwd_recv_buff, bwd_send_buff, bwd_recv_buff,
                              ep_send_buffer, ep_recv_buffer, ep_alltoall_size, num_experts,
                              dp_communicator, pp_communicator, ep_communicator);
        
        CCUTILS_MPI_TIMER_STOP(runtime)
        if(stage_id > 0 && stage_id < num_stage-1){
            std::vector<float> merged_pp;
            // We want 2 entries per microbatch: [Fwd_Comm, Bwd_Comm]
            merged_pp.reserve(num_microbatches * 2);

            int fwd_offset = 0;
            int bwd_offset = num_microbatches * 2; // Bwd starts after all Fwd timers

            for(int i = 0; i < num_microbatches; i++){
                // Merge Forward: (Recv + Send)
                float fwd_comm = __timer_vals_pp_comm[fwd_offset + i*2] + 
                                __timer_vals_pp_comm[fwd_offset + i*2 + 1];
                merged_pp.push_back(fwd_comm);
            }
            
            for(int i = 0; i < num_microbatches; i++){
                // Merge Backward: (Recv + Send)
                float bwd_comm = __timer_vals_pp_comm[bwd_offset + i*2] + 
                                __timer_vals_pp_comm[bwd_offset + i*2 + 1];
                merged_pp.push_back(bwd_comm);
            }

            __timer_vals_pp_comm = std::move(merged_pp);
        }
    }
    
    char host_name[MPI_MAX_PROCESSOR_NAME];
    int namelen;
    MPI_Get_processor_name(host_name, &namelen);
    
    CCUTILS_MPI_SECTION_DEF(dp_pp_ep, "Data + Pipeline + Expert Parallelism")
    
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "model_name", model_name)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "num_stages", num_stage)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "num_microbatches", num_microbatches)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "num_expert_shards", num_expert_shards)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "num_experts", num_experts)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "sequence_length", sequence_length)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "embedded_dim", embedded_dim)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "samples_per_microbatch", samples_per_microbatch)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "local_batch_size", local_batch_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "global_batch_size", dp_size * num_expert_shards * local_batch_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "world_size", world_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "dp_size", dp_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "fwd_rt_per_microbatch", fwd_rt_per_microbatch)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "bwd_rt_per_microbatch", bwd_rt_per_microbatch)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "total_model_size_params", total_model_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "pipe_msg_size_bytes", pipe_msg_size * sizeof(_FLOAT))
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "ep_alltoall_size_bytes", ep_alltoall_size * sizeof(_FLOAT))
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "dp_allreduce_size_bytes", dp_allreduce_size * sizeof(_FLOAT))
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "device", (device == Device::CPU) ? "CPU" : "GPU")
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_ep, "backend", dp_communicator->get_name())
    
    CCUTILS_SECTION_JSON_PUT(dp_pp_ep, "runtimes", __timer_vals_runtime);
    //compute trhoughputs based on runtimes and batch size
    std::vector<float> throughputs;
    for(float rt : __timer_vals_runtime){
        float throughput = (local_batch_size * dp_size * num_expert_shards) / rt; // samples per second
        throughputs.push_back(throughput);
    }
    CCUTILS_SECTION_JSON_PUT(dp_pp_ep, "throughputs", throughputs);
    CCUTILS_SECTION_JSON_PUT(dp_pp_ep, "pp_comm_time", __timer_vals_pp_comm);
    CCUTILS_SECTION_JSON_PUT(dp_pp_ep, "dp_comm_time", __timer_vals_dp_comm);
    CCUTILS_SECTION_JSON_PUT(dp_pp_ep, "ep_comm_time", __timer_vals_ep_comm);
    CCUTILS_SECTION_JSON_PUT(dp_pp_ep, "dp_ep_comm_time", __timer_vals_dp_ep_comm);
    CCUTILS_SECTION_JSON_PUT(dp_pp_ep, "hostname", host_name);
    CCUTILS_SECTION_JSON_PUT(dp_pp_ep, "stage_id", stage_id);
    CCUTILS_SECTION_JSON_PUT(dp_pp_ep, "ep_id", ep_id);
    CCUTILS_SECTION_JSON_PUT(dp_pp_ep, "dp_id", dp_id);
    CCUTILS_MPI_SECTION_END(dp_pp_ep);
    #endif
    
#ifdef PROXY_ENABLE_CCL
    ncclCommDestroy(dp_world_comm);
    ncclCommDestroy(pp_world_comm);
    ncclCommDestroy(ep_world_comm);
#endif
    
    MPI_Finalize();
    
    return 0;
}
