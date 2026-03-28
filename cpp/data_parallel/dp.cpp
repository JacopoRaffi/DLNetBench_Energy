/*********************************************************************
 *
 * Description: C++/MPI proxy for Transformer-based models distributed training 
 *              with data parallelism
 * Author: Jacopo Raffi
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

#include <power_profiler.hpp>
#include <data_types.hpp>

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
#if defined(PROXY_ENABLE_CUDA) || defined(PROXY_ENABLE_HIP) || defined(PROXY_ENABLE_ONECCL)
    constexpr Device device = Device::GPU;
#else
    constexpr Device device = Device::CPU;
#endif

// Default values
#define NUM_B 10
#define WARM_UP 3
#define RUNS 5
#define POWER_SAMPLING_RATE_MS 5

CCUTILS_MPI_TIMER_DEF(runtime)
CCUTILS_MPI_TIMER_DEF(barrier)

/**
 * @brief Simulates one iteration of data-parallel (using bucketing approach) training for a Transformer model.
 *
 * This function performs a mock forward pass and a backward pass with asynchronous
 * all-reduce operations for gradients over multiple parameter buckets.
 *
 * @param grad_ptrs Array of pointers to gradient buffers for each bucket.
 * @param sum_grad_ptrs Array of pointers to buffers storing the reduced gradients.
 * @param num_buckets Number of parameter buckets.
 * @param params_per_bucket Array containing the number of parameters in each bucket.
 * @param fwd_rt_whole_model Forward pass runtime in microseconds.
 * @param bwd_rt_per_B Backward pass runtime per bucket in microseconds.
 * @param comm Pointer to the Communicator object for Collective operations.
 * @return int Always returns 0.
 */
int run_data_parallel(Tensor<_FLOAT, device>** grad_ptrs, Tensor<_FLOAT, device>** sum_grad_ptrs, 
                    int num_buckets, uint64_t* params_per_bucket,
                    uint64_t fwd_rt_whole_model, float bwd_rt_per_B, ProxyCommunicator* communicator) {
    

    //forward compute
    usleep(fwd_rt_whole_model);
    //backward (idea is to overlap all-reduce with backward compute)

    int index, flag;
    for(int i=0; i<num_buckets; i++){
        usleep(bwd_rt_per_B); //compute backward of a bucket 
        communicator->Iallreduce(grad_ptrs[i]->data, sum_grad_ptrs[i]->data, params_per_bucket[i], i); //start all-reduce for the bucket
    }

    CCUTILS_MPI_TIMER_START(barrier)
    communicator->WaitAll(num_buckets); //wait for all all-reduce to complete
    CCUTILS_MPI_TIMER_STOP(barrier) 
    return 0;
}

#define REQUIRED_ARGS                                                            \
    REQUIRED_STRING_ARG(model_name, "model", "Name of the model to use")         \
    REQUIRED_INT_ARG(num_buckets, "num_buckets", "Number of parameter buckets")  \
    REQUIRED_STRING_ARG(base_path, "base_path", "Base path for the repository")  

#define OPTIONAL_ARGS                                                                           \
    OPTIONAL_INT_ARG(warmup, WARM_UP, "-w", "warmups", "Number of warm-up iterations")          \
    OPTIONAL_INT_ARG(runs, RUNS, "-r", "runs", "Number of iterations to run")                   \
    OPTIONAL_STRING_ARG(devices, default_devices, "-d", "devices", "Comma-separated list of devices")  \
    OPTIONAL_INT_ARG(min_exectime, 0, "-m", "min_exectime", "Minimum total execution time in seconds (overrides runs)") \
    OPTIONAL_INT_ARG(batch_size, 16, "-b", "batch_size", "Batch size to use for the model (overrides batch size in model stats file)") \
    OPTIONAL_STRING_ARG(gpu, default_gpu, "-g", "gpu", "GPU to use") \
    OPTIONAL_STRING_ARG(dtype, default_dtype, "-t", "dtype", "Data type to use")

#define BOOLEAN_ARGS \
    BOOLEAN_ARG(help, "-h", "Show help")

#include <ccutils/easyargs.hpp>


int main(int argc, char* argv[]) {
#ifdef PROXY_ENABLE_ONECCL
   int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#else 
    MPI_Init(&argc,&argv);
#endif
    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    CCUTILS_MPI_INIT
    
    install_signal_handlers();

    args_t args = make_default_args();

    if (!parse_args(argc, argv, &args) || args.help) {
        print_help(argv[0]);
        return 1;
    }

    int num_buckets = NUM_B;

    std::string model_name = args.model_name;
    num_buckets = args.num_buckets;

    // --- Construct model stats file path ---
    fs::path repo_path = get_dnnproxy_base_path(args.base_path);
    fs::path file_path = repo_path / "model_stats" / (model_name + ".json");
    if (!fs::exists(file_path)) {
        std::cerr << "Error: model stats file does not exist: " << file_path << "\n";
        return -1;
    }

    uint64_t runs = args.runs;
    uint64_t warmup = args.warmup;

    std::map<std::string, uint64_t> model_stats = get_model_stats(file_path, args.gpu, args.dtype, (uint64_t)args.batch_size); // get model stats from file
    uint64_t fwd_rt_whole_model = model_stats["avgForwardTime"]; // in us
    float bwd_rt_per_B = (model_stats["avgBackwardTime"]) / num_buckets; // in us
    uint local_batch_size = model_stats["batchSize"];
    uint64_t total_model_size = model_stats["modelSize"]; // number of parameters
    
    uint64_t base_params_per_bucket = total_model_size / num_buckets;
    uint64_t remainder = total_model_size % num_buckets;
    uint64_t params_per_bucket[num_buckets];
    for (int i = 0; i < num_buckets; i++) {
        params_per_bucket[i] = base_params_per_bucket + (i < remainder ? 1 : 0); // distribute remainder across the buckets
    }

    print_topology_graph(MPI_COMM_WORLD);
    
    int my_device = set_local_device(MPI_COMM_WORLD, args.devices);

    CCUTILS_MPI_ALL_PRINT(fprintf(fp, "Using device %d\n", my_device);)
    
    #ifdef PROXY_ENABLE_CCL
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&world_comm, world_size, id, rank);
    CCLCommunicator* communicator = new CCLCommunicator(world_comm, num_buckets);
#elif defined(PROXY_ENABLE_ONECCL)
    // Select GPU devices
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    int num_gpus = gpus.size();
    sycl::device dev = gpus[rank % num_gpus];
 
    DeviceManager::init(dev);
    sycl::queue& queue = DeviceManager::get_queue();
  
    sycl::context ctx = queue.get_context();

    // Initialize oneCCL
    ccl::init();

    // Create KVS (acts like ncclUniqueId)
    ccl::shared_ptr_class<ccl::kvs> kvs;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
    }

    // Serialize and broadcast KVS address via MPI
    ccl::kvs::address_type kvs_addr;
    if (rank == 0) kvs_addr = kvs->get_address();

    size_t addr_size = kvs_addr.size();
    MPI_Bcast(kvs_addr.data(), kvs_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank != 0) kvs = ccl::create_kvs(kvs_addr);
    ccl::device ccl_dev = ccl::create_device(dev);
    ccl::context ccl_ctx = ccl::create_context(ctx);
    // Create communicator
    auto world_comm_ccl = ccl::create_communicator(world_size, rank, ccl_dev, ccl_ctx, kvs);
    OneCCLCommunicator* communicator = new OneCCLCommunicator(std::move(world_comm_ccl), ctx, dev, num_buckets); 
#else
    MPICommunicator* communicator = new MPICommunicator(MPI_COMM_WORLD, MPI_FLOAT, num_buckets);
#endif

    Tensor<_FLOAT, device>* grad_ptrs[num_buckets];
    Tensor<_FLOAT, device>* sum_grad_ptrs[num_buckets];
    for(int i=0; i<num_buckets; i++){
        grad_ptrs[i] = new Tensor<_FLOAT, device>(params_per_bucket[i]);
        sum_grad_ptrs[i] = new Tensor<_FLOAT, device>(params_per_bucket[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //warmup
    std::vector<float> warmup_times;
    for(int wmp = 0; wmp < warmup; wmp++){
        if(end){
            CCUTILS_MPI_PRINT_ONCE(printf("Interrupted during warm-up\n");)
            break;
        }
        float start_time = MPI_Wtime();
        run_data_parallel(grad_ptrs, sum_grad_ptrs, num_buckets, params_per_bucket,
                         fwd_rt_whole_model, bwd_rt_per_B, communicator);
        float end_time = MPI_Wtime();
        warmup_times.push_back(end_time - start_time);
    }

    if (args.min_exectime > 0) {
        runs = estimate_runs(warmup_times, args.min_exectime);
        CCUTILS_MPI_PRINT_ONCE(std::cout << "Estimated runs based on warm-up times to meet minimum execution time: " << runs << std::endl;)
    }

    #ifdef PROXY_LOOP
    while(true){
        run_data_parallel(grad_ptrs, sum_grad_ptrs, num_buckets, params_per_bucket,
                         fwd_rt_whole_model, bwd_rt_per_B, communicator);
    }
    #else
    // clear barrier times
    __timer_vals_barrier.clear();

    int node_id = 0;
    if (const char* slurm_nodeid = std::getenv("SLURM_NODEID")) {
    	node_id = std::atoi(slurm_nodeid);
    }

    profiler::PowerProfiler power_profiler(my_device, node_id, POWER_SAMPLING_RATE_MS);
    std::vector<float> dev_energies_mj;
    std::vector<profiler::data_types::power_trace_t> all_traces_per_iter;
    for(int iter = 0; iter < runs; iter++){
        if(end){
            CCUTILS_MPI_PRINT_ONCE(printf("Interrupted at iteration %d. Total iteration completed: %d \n", iter, __timer_vals_runtime.size());)
            break;
        }
        power_profiler.trace_clear();
        power_profiler.start();
        CCUTILS_MPI_TIMER_START(runtime)
        run_data_parallel(grad_ptrs, sum_grad_ptrs, num_buckets, params_per_bucket,
                         fwd_rt_whole_model, bwd_rt_per_B, communicator);
        CCUTILS_MPI_TIMER_STOP(runtime)
        power_profiler.stop();

        dev_energies_mj.push_back(power_profiler.get_device_energy() / 1000.0f); // convert uJ to mJ
        all_traces_per_iter.push_back(power_profiler.get_power_execution_data());
    }

    int executed_runs = __timer_vals_runtime.size();
    if (__timer_vals_barrier.size() > executed_runs)
        __timer_vals_barrier.resize(executed_runs);
    

    char host_name[MPI_MAX_PROCESSOR_NAME];
	char (*host_names)[MPI_MAX_PROCESSOR_NAME];
	int namelen,bytes,n,color;
	MPI_Get_processor_name(host_name,&namelen);

    std::vector<uint64_t> bucket_sizes(params_per_bucket,
                                   params_per_bucket + num_buckets);
    std::pair<float, float> msg_stats = compute_msg_stats(bucket_sizes, 1);

    CCUTILS_MPI_SECTION_DEF(dp, "Data Parallelism")
    float msg_size_avg = msg_stats.first;
    float msg_size_std = msg_stats.second;
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "model_name", model_name)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "num_buckets", num_buckets)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "local_batch_size", local_batch_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "world_size", world_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "fwd_rt_whole_model", fwd_rt_whole_model)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "bwd_rt_per_bucket", bwd_rt_per_B)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "total_model_size_params", total_model_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "msg_size_avg_bytes", msg_size_avg*sizeof(_FLOAT))
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "msg_size_std_bytes", msg_size_std*sizeof(_FLOAT))
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "GPU model", args.gpu)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "data_type", args.dtype)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "device", (device == Device::CPU) ? "CPU" : "GPU")
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "backend", communicator->get_name())

    //erase warm-up elemements
    CCUTILS_SECTION_JSON_PUT(dp, "runtimes", __timer_vals_runtime);
    CCUTILS_SECTION_JSON_PUT(dp, "barrier_time", __timer_vals_barrier);
    CCUTILS_SECTION_JSON_PUT(dp, "device_energies_mj",     dev_energies_mj);
    CCUTILS_SECTION_JSON_PUT(dp, "power_traces", all_traces_per_iter);
    // compute throughput per runtime (samples/s)
    std::vector<float> throughputs;
    for(float rt : __timer_vals_runtime){
        float throughput = (local_batch_size * world_size) / rt; // convert rt to seconds
        throughputs.push_back(throughput);
    }

    CCUTILS_SECTION_JSON_PUT(dp, "throughputs", throughputs);
    CCUTILS_SECTION_JSON_PUT(dp, "hostname", host_name);
    CCUTILS_SECTION_JSON_PUT(dp, "device_id", my_device);

    CCUTILS_MPI_SECTION_END(dp);
    #endif
	
    #ifdef PROXY_ENABLE_CLL
    ncclCommDestroy(world_comm);
    #endif

    MPI_Finalize();
}
