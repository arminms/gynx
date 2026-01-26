#include <benchmark/benchmark.h>

#include <gynx/sq.hpp>
#include <gynx/io/fastaqz.hpp>
#include <gynx/algorithms/valid.hpp>
#include <gynx/algorithms/random.hpp>

using namespace gynx::execution;

const uint64_t seed_pi{3141592654};
const std::string fasta_filename{"perf_data.fa"};

//----------------------------------------------------------------------------//
// io_write_fasta()

template <typename T>
void io_write_fasta(benchmark::State& st)
{   size_t n = size_t(st.range());
    auto s = gynx::random::dna<gynx::sq_gen<T>>(n);

    for (auto _ : st)
    {   s.save(fasta_filename, gynx::out::fasta());
    }
    std::remove(fasta_filename.c_str());

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(typename T::value_type)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(io_write_fasta, std::vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(io_write_fasta, std::vector<char, gynx::default_init_allocator<char>>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);

#if defined(__CUDACC__) || defined(__HIPCC__)
BENCHMARK_TEMPLATE(io_write_fasta, thrust::host_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(io_write_fasta, thrust::device_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
#endif // __CUDACC__ || __HIPCC__

#if defined(__HIPCC__)
BENCHMARK_TEMPLATE(io_write_fasta, gynx::unified_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
#endif // __HIPCC__

//----------------------------------------------------------------------------//
// io_read_fasta()

template <typename T>
void io_read_fasta(benchmark::State& st)
{   size_t n = size_t(st.range());
    gynx::sq_gen<T> sr;
    auto sw = gynx::random::dna<gynx::sq_gen<T>>(n);
    sw.save(fasta_filename, gynx::out::fasta());

    for (auto _ : st)
    {   sr.load(fasta_filename);
        benchmark::ClobberMemory();
    }
    std::remove(fasta_filename.c_str());

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(typename T::value_type)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(io_read_fasta, std::vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(io_read_fasta, std::vector<char, gynx::default_init_allocator<char>>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);

#if defined(__CUDACC__) || defined(__HIPCC__)
BENCHMARK_TEMPLATE(io_read_fasta, thrust::host_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(io_read_fasta, thrust::device_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
#endif // __CUDACC__ || __HIPCC__

#if defined(__HIPCC__)
BENCHMARK_TEMPLATE(io_read_fasta, gynx::unified_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
#endif // __HIPCC__

//----------------------------------------------------------------------------//
// rand() algorithm

template <typename T, typename ExecPolicy>
void random(benchmark::State& st)
{   size_t n = size_t(st.range());
    gynx::sq_gen<T> s(n);
    ExecPolicy policy;

    for (auto _ : st)
    {   gynx::rand
        (   policy
        ,   s.begin()
        ,   n
        ,   "ACGT"
        ,   seed_pi
        );
        benchmark::ClobberMemory();
    }

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(typename T::value_type)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE2(random, std::vector<char>, sequenced_policy)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE2(random, std::vector<char>, unsequenced_policy)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE2(random, std::vector<char>, parallel_policy)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseRealTime()
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE2(random, std::vector<char>, parallel_unsequenced_policy)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseRealTime()
->  Unit(benchmark::kMillisecond);

#if defined(__CUDACC__)
template <class T>
void random_cuda(benchmark::State& st)
{   size_t n = size_t(st.range());
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    gynx::sq_gen<T> s(n);

    for (auto _ : st)
    {   cudaEventRecord(start);
        gynx::rand(thrust::cuda::par, s.begin(), n, "ACGT", seed_pi);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        st.SetIterationTime(milliseconds * 0.001f);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(typename T::value_type)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(random_cuda, thrust::device_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(random_cuda, thrust::universal_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);

#endif //__CUDACC__

#if defined(__HIPCC__)
template <class T>
void random_rocm(benchmark::State& st)
{   size_t n = size_t(st.range());
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    gynx::sq_gen<T> s(n);

    for (auto _ : st)
    {   hipEventRecord(start);
        gynx::rand(thrust::hip::par, s.begin(), n, "ACGT", seed_pi);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        st.SetIterationTime(milliseconds * 0.001f);
    }
    hipEventDestroy(start); hipEventDestroy(stop);

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(typename T::value_type)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(random_rocm, thrust::device_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(random_rocm, thrust::universal_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(random_rocm, gynx::unified_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);
#endif //__HIPCC__

//----------------------------------------------------------------------------//
// valid() algorithm

template <typename T, typename ExecPolicy>
void valid(benchmark::State& st)
{   size_t n = size_t(st.range());
    auto s = gynx::random::dna<gynx::sq_gen<T>>(n, seed_pi);
    ExecPolicy policy;

    for (auto _ : st)
        benchmark::DoNotOptimize(gynx::valid(policy, s));

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(typename T::value_type)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE2(valid, std::vector<char>, sequenced_policy)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE2(valid, std::vector<char>, unsequenced_policy)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE2(valid, std::vector<char>, parallel_policy)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseRealTime()
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE2(valid, std::vector<char>, parallel_unsequenced_policy)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseRealTime()
->  Unit(benchmark::kMillisecond);

#if defined(__CUDACC__)

template <class T>
void valid_cuda(benchmark::State& st)
{   size_t n = size_t(st.range());
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    auto s = gynx::random::dna<gynx::sq_gen<T>>(n, seed_pi);

    for (auto _ : st)
    {   cudaEventRecord(start);
        // cudaEventRecord(start, stream);
        benchmark::DoNotOptimize(gynx::valid(s));
        // benchmark::DoNotOptimize(gynx::valid(thrust::cuda::par.on(stream), s));
        cudaEventRecord(stop);
        // cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        st.SetIterationTime(milliseconds * 0.001f);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(typename T::value_type)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(valid_cuda, thrust::device_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(valid_cuda, thrust::universal_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);

#endif //__CUDACC__

#if defined(__HIPCC__)

template <class T>
void valid_rocm(benchmark::State& st)
{   size_t n = size_t(st.range());
    // hipStream_t stream;
    // hipStreamCreate(&stream);
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    auto s = gynx::random::dna<gynx::sq_gen<T>>(n, seed_pi);

    for (auto _ : st)
    {   hipEventRecord(start);
        // hipEventRecord(start, stream);
        benchmark::DoNotOptimize(gynx::valid(s));
        // benchmark::DoNotOptimize(gynx::valid(thrust::hip::par.on(stream), s));
        hipEventRecord(stop);
        // hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        st.SetIterationTime(milliseconds * 0.001f);
    }
    hipEventDestroy(start); hipEventDestroy(stop);

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(typename T::value_type)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(valid_rocm, thrust::device_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(valid_rocm, thrust::universal_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(valid_rocm, gynx::unified_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<25, 1<<28)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);

#endif //__HIPCC__

//----------------------------------------------------------------------------//
// test_unified_memory()

template <typename T>
void unified_physical_memory(benchmark::State& st)
{   size_t n = size_t(st.range());

    for (auto _ : st)
    {   auto sw = gynx::random::dna<gynx::sq_gen<T>>(n);
        sw.save(fasta_filename, gynx::out::fasta());
        gynx::sq_gen<T> sr;
        sr.load(fasta_filename);
        benchmark::DoNotOptimize(gynx::valid(sr));
    }
    std::remove(fasta_filename.c_str());

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(typename T::value_type)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}
BENCHMARK_TEMPLATE(unified_physical_memory, std::vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
#if defined(__CUDACC__) || defined(__HIPCC__)
BENCHMARK_TEMPLATE(unified_physical_memory, thrust::host_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(unified_physical_memory, thrust::device_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(unified_physical_memory, thrust::universal_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
#if defined(__HIPCC__)
BENCHMARK_TEMPLATE(unified_physical_memory, gynx::unified_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<28, 1<<29)
->  Unit(benchmark::kMillisecond);
#endif //__HIPCC__
#endif // __CUDACC__ || __HIPCC__

//-- get_runtime_version ------------------------------------------------------//

#if defined(__CUDACC__)
std::string get_runtime_version()
{   int version{0};
    cudaError_t err = cudaRuntimeGetVersion(&version);
    std::stringstream os;
    os << "\n  CUDA Runtime Version: ";
    if (err == cudaSuccess)
    {   int major = version / 1000;
        int minor = (version % 1000) / 10;
        int patch = version % 10;
        os << major << '.' << minor << '.' << patch;
    }
    else os << "failed to get CUDA version";
    return os.str();

}
#endif //__CUDACC__

#if defined(__HIPCC__)
std::string get_runtime_version()
{   int version{0};
    hipError_t err = hipRuntimeGetVersion(&version);
    std::stringstream os;
    os << "\n  ROCm Runtime Version: ";
    if (err == hipSuccess)
    {   int major = version / 10000000;
        int minor = (version / 100000) % 100;
        os << major << '.' << minor;
    }
    else os << "failed to get HIP version";
    return os.str();
}
#endif //__HIPCC__

//----------------------------------------------------------------------------//
// main()

int main(int argc, char** argv)
{   benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;

#if defined(__CUDACC__) || defined(__HIPCC__)
#if defined(__CUDACC__)
    cudaDeviceProp prop;
    auto status = cudaGetDeviceProperties(&prop, 0);
    if (status == cudaSuccess) {
#else
    int runtimeVersion;
    hipRuntimeGetVersion(&runtimeVersion);
    hipDeviceProp_t prop;
    auto status = hipGetDeviceProperties(&prop, 0);
    if (status == hipSuccess) {
#endif
    std::stringstream os;
    os << "\n  " << prop.name
       << "\n  (" << prop.multiProcessorCount << " X " << prop.clockRate / 1e6
                  << " MHz SM s)"
       << "\n  L2 Cache: " << prop.l2CacheSize / 1024 << " KiB (x"
                           << prop.multiProcessorCount << ")"
       << "\n  Peak Memory Bandwidth: "
       << std::fixed << std::setprecision(0)
       // based on https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-c
       << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6
       << " (GB/s)"
       << get_runtime_version();
    benchmark::AddCustomContext("GPU", os.str());
#if defined(__CUDACC__)
    } else
        benchmark::AddCustomContext("GPU", "No CUDA device found");
#else
    } else
        benchmark::AddCustomContext("GPU", "No ROCm device found");
#endif // __CUDACC__
#endif //__CUDACC__ || __HIPCC__

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}