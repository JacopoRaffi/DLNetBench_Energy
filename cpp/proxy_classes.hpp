#ifndef PROXY_CLASSES_HPP
#define PROXY_CLASSES_HPP

#include <mpi.h>

#ifdef PROXY_ENABLE_CUDA   
    #include <cuda_runtime.h>
    #include <ccutils/cuda/cuda_macros.hpp>
#endif

#ifdef PROXY_ENABLE_HIP
    #include "tmp_hip_ccutils.hpp" 
    #include <hip/hip_runtime.h>
#endif

#ifdef PROXY_ENABLE_NCCL
    #include <nccl.h>
#endif

#ifdef PROXY_ENABLE_RCCL
    #include <rccl.h>
#endif

#ifdef PROXY_ENABLE_ONECCL
    #include <oneapi/ccl.hpp>
    #include <sycl/sycl.hpp>  // SYCL header for queues and device memory
#endif

class ProxyCommunicator {
public:
    virtual void Iallreduce(const void* sendbuf, void* recvbuf, int count, int index) = 0;
    virtual void Allreduce(const void* sendbuf, void* recvbuf, int count) = 0;
    virtual void Iallgather(const void* sendbuf, int sendcount,
                            void* recvbuf, int recvcount, int index) = 0;
    virtual void Allgather(const void* sendbuf, int sendcount,
                           void* recvbuf, int recvcount) = 0;
    virtual void Reduce_Scatter_block(const void* sendbuf, void* recvbuf, int recvcount) = 0;
    virtual void Alltoall(const void* sendbuf, int sendcount,
                            void* recvbuf, int recvcount) = 0;
    virtual void Barrier() = 0;
    virtual void WaitAll(int num_waits) = 0;
    virtual void Wait(int index) = 0;
    virtual void finalize() = 0;
    virtual void send(const void* buf, int count, int dest) = 0;
    virtual void recv(void* buf, int count, int source) = 0;
    virtual void Isend(const void* buf, int count, int dest, int index) = 0;
    virtual void Irecv(void* buf, int count, int source, int index) = 0;
    virtual std::string get_name() = 0;
    virtual ~ProxyCommunicator() {}
};

class MPICommunicator : public ProxyCommunicator {
public:
    MPICommunicator(MPI_Comm comm, MPI_Datatype datatype, int num_requests=0) {
        this->comm = comm;
        this->datatype = datatype;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &comm_size);

        if(num_requests > 0){
            requests = new MPI_Request[num_requests];
        }
    };

    void Iallreduce(const void* sendbuf, void* recvbuf, int count, int index) override {
        MPI_Iallreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, comm, &(requests[index]));
    };

    void Allreduce(const void* sendbuf, void* recvbuf, int count) override {
        MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, comm);
    };

    void Alltoall(const void* sendbuf, int sendcount,
                    void* recvbuf, int recvcount) override {
        MPI_Alltoall(sendbuf, sendcount, datatype, recvbuf, recvcount, datatype, comm);
    };

    void Barrier() override {
        MPI_Barrier(comm);
    };

    void WaitAll(int num_waits) override {
        MPI_Waitall(num_waits, requests, MPI_STATUSES_IGNORE);
    };

    void Wait(int index) override {
        MPI_Wait(&requests[index], MPI_STATUS_IGNORE);
    };

    void Iallgather(const void* sendbuf, int sendcount, void* recvbuf, int recvcount, int index) override {
        MPI_Iallgather(sendbuf, sendcount, datatype, recvbuf, recvcount, datatype, comm, &requests[index]);
    }

    void Allgather(const void* sendbuf, int sendcount, void* recvbuf, int recvcount) override {
        MPI_Allgather(sendbuf, sendcount, datatype, recvbuf, recvcount, datatype, comm);
    };

    void Reduce_Scatter_block(const void* sendbuf, void* recvbuf, int recvcount) override {
        MPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, MPI_FLOAT, MPI_SUM, comm);
    };

    void send(const void* buf, int count, int dest) override {
        MPI_Send(buf, count, datatype, dest, 0, comm);
    };

    void recv(void* buf, int count, int source) override {
        MPI_Recv(buf, count, datatype, source, 0, comm, MPI_STATUS_IGNORE);
    };

    void Isend(const void* buf, int count, int dest, int index) override {
        MPI_Isend(buf, count, datatype, dest, 0, comm, &requests[index]);
    };

    void Irecv(void* buf, int count, int source, int index) override {
        MPI_Irecv(buf, count, datatype, source, 0, comm, &requests[index]);
    };

    std::string get_name() override {
        return std::string("MPI");
    };

    void finalize() override {
        MPI_Finalize();
    };
private:
    MPI_Comm comm;
    int comm_size;
    int rank;
    MPI_Datatype datatype = MPI_FLOAT;
    MPI_Request* requests = nullptr;

};

#ifdef PROXY_ENABLE_CCL //NCCL or RCCL
class CCLCommunicator : public ProxyCommunicator {
public:
    CCLCommunicator(ncclComm_t comm, int num_streams=1) {
        this->comm = comm;
        this->num_streams = num_streams;
        ncclCommUserRank(comm, &rank);
        ncclCommCount(comm, &comm_size);
        this->streams = new _Stream[num_streams];
        for(int i = 0; i < num_streams; i++) {
            CREATE_STREAM(this->streams[i]);
        }
    };

    void Iallreduce(const void* sendbuf, void* recvbuf, int count, int index) override{
        ncclAllReduce(sendbuf, recvbuf, count, NCCL_FLOAT_TYPE, ncclSum,
                      comm, streams[index]);
    }

    void Allreduce(const void* sendbuf, void* recvbuf, int count) override {
        ncclAllReduce(sendbuf, recvbuf, count, NCCL_FLOAT_TYPE, ncclSum,
                      comm, streams[0]);
        Wait(0);
    }

    void Alltoall(const void* sendbuf, int sendcount,
                    void* recvbuf, int recvcount) override {
        
        // 1. Begin the Group: Batches all operations into a single kernel launch
        ncclGroupStart();

        // 2. Iterate over all ranks to post Send/Recv
        for (int r = 0; r < comm_size; r++) {
            // Calculate offsets: Move pointer by (rank * count) elements
            const _FLOAT* send_ptr = static_cast<const _FLOAT*>(sendbuf) + (r * sendcount);
            _FLOAT* recv_ptr = static_cast<_FLOAT*>(recvbuf) + (r * recvcount);

            // Standard NCCL Send/Recv calls
            ncclSend(send_ptr, sendcount, NCCL_FLOAT_TYPE, r, comm, streams[0]);
            ncclRecv(recv_ptr, recvcount, NCCL_FLOAT_TYPE, r, comm, streams[0]);
        }

        // 3. End the Group: Submits the optimized AllToAll pattern to the GPU
        ncclGroupEnd();
        
        // 4. Synchronize
        Wait(0);
    }

    void WaitAll(int num_waits) override {
        for(int i = 0; i < num_waits; i++) 
            SYNC_STREAM(streams[i]);
    }

    void Barrier() override {
        WaitAll(num_streams);
    };

    void Wait(int index) override {
        SYNC_STREAM(streams[index]);
    };

    void Allgather(const void* sendbuf, int sendcount, void* recvbuf, int recvcount) override {
        ncclAllGather(sendbuf, recvbuf, sendcount, NCCL_FLOAT_TYPE, comm, streams[0]);
        Wait(0);
    };

    void Iallgather(const void* sendbuf, int sendcount, void* recvbuf, int recvcount, int index) override {
        ncclAllGather(sendbuf, recvbuf, sendcount, NCCL_FLOAT_TYPE, comm, streams[index]);
    };

    void Reduce_Scatter_block(const void* sendbuf, void* recvbuf, int recvcount) override {
        ncclReduceScatter(sendbuf, recvbuf, recvcount, NCCL_FLOAT_TYPE, ncclSum, comm, streams[0]);
        Wait(0);
    };

    void send(const void* buf, int count, int peer) override {
        ncclSend(buf, count, NCCL_FLOAT_TYPE, peer, comm, streams[0]);
        Wait(0);
    }

    void recv(void* buf, int count, int peer) override {
        ncclRecv(buf, count, NCCL_FLOAT_TYPE, peer, comm, streams[0]);
        Wait(0);
    }

    void Isend(const void* buf, int count, int peer, int index) override {
        ncclSend(buf, count, NCCL_FLOAT_TYPE, peer, comm, streams[index]);
    }

    void Irecv(void* buf, int count, int peer, int index) override {
        ncclRecv(buf, count, NCCL_FLOAT_TYPE, peer, comm, streams[index]);
    }

    void finalize() override {
        for(int i = 0; i < num_streams; i++) {
            DESTROY_STREAM(streams[i]);
        }
        delete[] streams;
    };

    std::string get_name() override {
        std::string name;
        #ifdef PROXY_ENABLE_NCCL
            name = "NCCL";
        #elif defined(PROXY_ENABLE_RCCL)
            name = "RCCL";
        #endif
        return name;
    };
private:
    ncclComm_t comm;
    int comm_size;
    int rank;
    int num_streams;
    _Stream* streams = nullptr;
    
};
#endif

#ifdef PROXY_ENABLE_ONECCL
class OneCCLCommunicator : public ProxyCommunicator {
public:
    OneCCLCommunicator(ccl::communicator&& comm_in, sycl::context ctx, sycl::device dev, sycl::queue& queue, int num_streams = 1)
        : comm(std::move(comm_in)), num_streams(num_streams)
    {
        // create one queue per stream
        events.resize(num_streams);
        ccl_streams.reserve(num_streams);
        for (int i = 0; i < num_streams; i++) {
            ccl_streams.push_back(ccl::create_stream(queue));
        }
    }

    // Non-blocking allreduce
    void Iallreduce(const void* sendbuf, void* recvbuf, int count, int index) override {
        events[index] = ccl::allreduce(sendbuf, recvbuf, count, ONECCL_FLOAT_TYPE, ccl::reduction::sum, comm, ccl_streams[index]);
    }

    // Blocking allreduce
    void Allreduce(const void* sendbuf, void* recvbuf, int count) override {
        ccl::allreduce(sendbuf, recvbuf, count, ONECCL_FLOAT_TYPE, ccl::reduction::sum, comm, ccl_streams[0]).wait();
    }

    void Alltoall(const void* sendbuf, int sendcount,
                    void* recvbuf, int recvcount) override {
        ccl::alltoall(sendbuf, recvbuf, sendcount, ONECCL_FLOAT_TYPE, comm, ccl_streams[0]).wait();
    }

    void WaitAll(int num_waits) override {
        for (auto& e : events) e.wait();
    }

    void Wait(int index) override {
        if (index < events.size()) events[index].wait();
    }

    void Barrier() override {
        // oneCCL barrier via allreduce of zero bytes
        ccl::barrier(comm, ccl_streams[0]).wait();
    }

    void Allgather(const void* sendbuf, int sendcount, void* recvbuf, int recvcount) override {
	    std::vector<size_t> recvcounts(comm.size(), sendcount);
	ccl::allgatherv(sendbuf, sendcount, recvbuf, recvcounts, ONECCL_FLOAT_TYPE, comm, ccl_streams[0]).wait();
    }

    void Iallgather(const void* sendbuf, int sendcount, void* recvbuf, int recvcount, int index) override { 
        std::vector<size_t> recvcounts(comm.size(), sendcount);
        events[index] = ccl::allgatherv(sendbuf, sendcount, recvbuf, recvcounts, ONECCL_FLOAT_TYPE, comm, ccl_streams[index]);
    }
	

    void Reduce_Scatter_block(const void* sendbuf, void* recvbuf, int recvcount) override {
        ccl::reduce_scatter(sendbuf, recvbuf, recvcount, ONECCL_FLOAT_TYPE, ccl::reduction::sum, comm, ccl_streams[0]).wait();
    }

    void send(const void* buf, int count, int dest) override {
        ccl::send(const_cast<void*>(buf), count, ONECCL_FLOAT_TYPE, dest, comm, ccl_streams[0]).wait();
    }

    void recv(void* buf, int count, int source) override {
        ccl::recv(const_cast<void*>(buf), count, ONECCL_FLOAT_TYPE, source, comm, ccl_streams[0]).wait();
    }

    void Isend(const void* buf, int count, int dest, int index) override {
        events[index] = ccl::send(const_cast<void*>(buf), count, ONECCL_FLOAT_TYPE, dest, comm, ccl_streams[index]);
    }

    void Irecv(void* buf, int count, int source, int index) override {
        events[index] = ccl::recv(const_cast<void*>(buf), count, ONECCL_FLOAT_TYPE, source, comm, ccl_streams[index]);
    }

    void finalize() override {
        events.clear();
        ccl_streams.clear();
    }

    std::string get_name() override { return "oneCCL"; }

private:
    ccl::communicator comm;
    int num_streams;
    std::vector<ccl::stream> ccl_streams;
    std::vector<ccl::event> events;
};
#endif


/**
* @enum Device
* @brief Enum to specify the device type for a tensor.
*/
enum class Device { CPU, GPU };

#if defined(PROXY_ENABLE_ONECCL)
#include <sycl/sycl.hpp>
namespace DeviceManager {
    static std::unique_ptr<sycl::queue> q_ptr;

    // Call this in main() to lock in the correct GPU for this rank
    static void init(const sycl::device& dev) {
        if (!q_ptr) {
            q_ptr = std::make_unique<sycl::queue>(dev);
        }
    }

    static sycl::queue& get_queue() {
        if (!q_ptr) {
            throw std::runtime_error("SYCL queue not initialized! Call DeviceManager::init() first.");
        }
        return *q_ptr;
    }
}
#endif

/**
* @class Tensor
* @brief A lightweight wrapper for a contiguous buffer of data that can reside on CPU or GPU.
*
* This class manages memory allocation and deallocation automatically depending on the device.
* It supports both CPU (host) memory using calloc and GPU (device) memory using cudaMalloc.
*
* @tparam T The data type of the tensor elements (e.g., float, double, half).
*/
template<typename T, Device device = Device::CPU>
class Tensor {
public:
    T* data = nullptr;
    uint64_t size = 0;

    /**
    * @brief Constructs a tensor of given size on a specified device.
    *
    * Allocates memory on the CPU using calloc or on the GPU using cudaMalloc.
    *
    * @param size_ Number of elements in the tensor
    * @param dev Device type (CPU by default)
    */
    explicit Tensor(uint64_t size_) : size(size_) {
        if constexpr (device == Device::CPU) {
            data = static_cast<T*>(calloc(size, sizeof(T)));
            if (!data)
                throw std::runtime_error("Failed to allocate CPU memory");
        }
        else if constexpr (device == Device::GPU) {
    #if defined(PROXY_ENABLE_CUDA)
            CCUTILS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data),
                        size * sizeof(T)));
            CCUTILS_CUDA_CHECK(cudaMemset(data, 0, size * sizeof(T)));
    #elif defined(PROXY_ENABLE_HIP)
            CCUTILS_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data),
                        size * sizeof(T)));
            CCUTILS_HIP_CHECK(hipMemset(data, 0, size * sizeof(T)););
    #elif defined(PROXY_ENABLE_ONECCL)
        auto& q = DeviceManager::get_queue();
        data = sycl::malloc_device<T>(size, q);    // allocate device memory
        if (!data) 
            throw std::runtime_error("SYCL device allocation failed");
        q.memset(data, 0, size * sizeof(T)).wait(); // optional zero
    #endif
        }
        else {
            static_assert(device == Device::CPU || device == Device::GPU,
                        "Unsupported device type");
        }
    }

    /**
    * @brief Destructor that frees the allocated memory depending on the device.
    */
    ~Tensor() {
        if (data) {
            if constexpr (device == Device::CPU) {
                free(data);
            }
            else if constexpr (device == Device::GPU) {
    #if defined(PROXY_ENABLE_CUDA)
                CCUTILS_CUDA_FREE_SAFE(data);
    #elif defined(PROXY_ENABLE_HIP)
                CCUTILS_HIP_FREE_SAFE(data);
    #elif defined(PROXY_ENABLE_ONECCL)
                sycl::free(data, DeviceManager::get_queue());
    #endif

            }
        }
    }
};

#endif // PROXY_CLASSES_HPP
