#include <benchmark/benchmark.h>

#include <gynx/sq.hpp>
#include <gynx/algorithms/valid.hpp>
#include <gynx/algorithms/random.hpp>

const uint64_t seed_pi{3141592654};

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
        // benchmark::ClobberMemory();
    }

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(typename T::value_type)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE2(random, std::vector<char>, gynx::execution::sequenced_policy)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE2(random, std::vector<char>, gynx::execution::unsequenced_policy)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE2(random, std::vector<char>, gynx::execution::parallel_policy)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseRealTime()
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE2(random, std::vector<char>, gynx::execution::parallel_unsequenced_policy)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
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
->  Range(1<<20, 1<<24)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(random_cuda, thrust::universal_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);

#endif //__CUDACC__

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

BENCHMARK_TEMPLATE2(valid, std::vector<char>, gynx::execution::sequenced_policy)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE2(valid, std::vector<char>, gynx::execution::unsequenced_policy)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE2(valid, std::vector<char>, gynx::execution::parallel_policy)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseRealTime()
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE2(valid, std::vector<char>, gynx::execution::parallel_unsequenced_policy)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
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
->  Range(1<<20, 1<<24)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(valid_cuda, thrust::universal_vector<char>)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);

#endif //__CUDACC__

//----------------------------------------------------------------------------//
// main()

int main(int argc, char** argv)
{   benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;

#if defined(__CUDACC__)
    // adding GPU context
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
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
       << " (GB/s)";
    benchmark::AddCustomContext("GPU", os.str());
#endif //__CUDACC__

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}