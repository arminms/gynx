// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <string_view>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <utility>
#include <cstddef>
#include <concepts>
#include <span>
#include <random>

#include <omp.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
    #include <thrust/device_vector.h>
#endif

#include <ranx/bind.hpp>
#include <ranx/pcg/pcg_random.hpp>
#include <ranx/trng/uniform_int_dist.hpp>
#include <ranx/trng/fast_discrete_dist.hpp>
#include <ranx/trng/discrete_dist.hpp>

#include <gnx/concepts.hpp>
#include <gnx/execution.hpp>

namespace gnx {

// -- uniform random sequence generation ---------------------------------------

namespace detail {

#if defined(__CUDACC__) || defined(__HIPCC__)
namespace kernel {

template<typename T, typename Size, typename Generator>
__global__ void leapfrogging
(   T* out
,   Size n
,   const T* alphabet
,   Size alphabet_size
,   Generator g
)
{   extern __shared__ T shared_alphabet[];
    if (threadIdx.x < alphabet_size)
        shared_alphabet[threadIdx.x] = alphabet[threadIdx.x];
    __syncthreads();

    auto idx{blockIdx.x * blockDim.x + threadIdx.x};
    auto stride{blockDim.x * gridDim.x};
    g.discard(idx);
    for (auto i{idx}; i < n; i += stride, g.discard(stride - 1))
        out[i] = shared_alphabet[g()];
}

} // end kernel namespace

template
<   typename ExecPolicy
,   device_resident_iterator OutputIterator
,   typename Size
>
inline void rand_device
(   const ExecPolicy& policy
,   OutputIterator out
,   Size n
,   std::string_view alphabet
,   std::uint64_t seed = 0
)
{   typedef typename std::iterator_traits<OutputIterator>::value_type value_type;

    if (0 == seed)
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto gen = ranx::bind
    (   trng::uniform_int_dist(0, static_cast<Size>(alphabet.size()))
    ,   pcg32(seed)
    );

    const Size block_size{256};
    int sm_count;
#if defined(__HIPCC__)
    hipStream_t stream = 0;
#else
    cudaStream_t stream = 0;
#endif
    if constexpr (has_stream_member<ExecPolicy>)
        stream = policy.stream();

    thrust::device_vector<char> d_alphabet(alphabet.begin(), alphabet.end());

    int device;
#if defined(__HIPCC__)
    hipGetDevice(&device);
    hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, device);
#else
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
#endif

    // launch leapfrogging kernel
    kernel::leapfrogging<<<sm_count, block_size, alphabet.size() * sizeof(value_type), stream>>>
    (   thrust::raw_pointer_cast(&out[0])
    ,   n
    ,   thrust::raw_pointer_cast(d_alphabet.data())
    ,   static_cast<Size>(alphabet.size())
    ,   gen
    );
}
#endif // __CUDACC__ || __HIPCC__

} // end detail namespace

/// @brief Generate a random sequence from a given alphabet.
/// @tparam OutputIterator Output iterator type
/// @tparam Size Size type for the number of characters to generate
/// @param out Output iterator to write the random sequence
/// @param n Number of characters to generate
/// @param alphabet String view of the alphabet to sample from
/// @param seed Optional seed for the random number generator (default: current time)
#if defined(__CUDACC__)
template
<   device_resident_iterator OutputIterator
,   typename Size
>
inline void rand
(   OutputIterator out
,   Size n
,   std::string_view alphabet
,   std::uint64_t seed = 0
)
{   detail::rand_device(thrust::cuda::par, out, n, alphabet, seed);
}
template
<   host_resident_iterator OutputIterator
#elif defined(__HIPCC__)
template
<   device_resident_iterator OutputIterator
,   typename Size
>
constexpr void rand
(   OutputIterator out
,   Size n
,   std::string_view alphabet
,   std::uint64_t seed = 0
)
{   detail::rand_device(thrust::hip::par, out, n, alphabet, seed);
}
template
<   host_resident_iterator OutputIterator
#else
template
<   std::contiguous_iterator OutputIterator
#endif // __CUDACC__ || __HIPCC__
,   typename Size
>
inline void rand
(   OutputIterator out
,   Size n
,   std::string_view alphabet
,   std::uint64_t seed = 0
)
{   typedef typename std::iterator_traits<OutputIterator>::value_type value_type;

    if (0 == seed)
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    pcg32 r(seed);
    // std::uniform_int_distribution<> ndx(0, static_cast<int>(alphabet.size() - 1));
    trng::uniform_int_dist ndx(0, static_cast<Size>(alphabet.size()));
    for (Size i = 0; i < n; ++i)
        out[i] = static_cast<value_type>(alphabet[ndx(r)]);
}

/// @brief Generate a random sequence from a given alphabet using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., gnx::execution::seq)
/// @tparam OutputIterator Output iterator type
/// @tparam Size Size type for the number of characters to generate
/// @param policy Execution policy controlling algorithm execution
/// @param out Output iterator to write the random sequence
/// @param n Number of characters to generate
/// @param alphabet String view of the alphabet to sample from
/// @param seed Optional seed for the random number generator (default: current time)
#if defined(__CUDACC__) || defined(__HIPCC__)
template
<   typename ExecPolicy
,   device_resident_iterator OutputIterator
,   typename Size
>
inline void rand
(   ExecPolicy&& policy
,   OutputIterator out
,   Size n
,   std::string_view alphabet
,   std::uint64_t seed = 0
)
{   detail::rand_device(std::forward<ExecPolicy>(policy), out, n, alphabet, seed);
}
template
<   typename ExecPolicy
,   host_resident_iterator OutputIterator
,   typename Size
>
#else
template
<   typename ExecPolicy
,   std::contiguous_iterator OutputIterator
,   typename Size
>
#endif // __CUDACC__ || __HIPCC__
requires gnx::is_execution_policy_v<std::decay_t<ExecPolicy>>
inline void rand
(   ExecPolicy&& policy
,   OutputIterator out
,   Size n
,   std::string_view alphabet
,   std::uint64_t seed = 0
)
{   typedef typename std::iterator_traits<OutputIterator>::value_type value_type;

    if (0 == seed)
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto ndx = ranx::bind
    (   trng::uniform_int_dist(0, static_cast<Size>(alphabet.size()))
    ,   pcg32(seed)
    );

    // compile-time dispatch based on execution policy
    if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gnx::execution::unsequenced_policy>)
    {
        #pragma omp simd
        for (Size i = 0; i < n; ++i)
            out[i] = static_cast<value_type>(alphabet[ndx()]);
    }
    else if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gnx::execution::parallel_policy>)
    {   // block splitting algorithm
        #pragma omp parallel
        {   auto tidx{omp_get_thread_num()};
            auto size{omp_get_num_threads()};
            Size first{tidx * n / size};
            Size last{(tidx + 1) * n / size};
            auto tl_ndx = ndx;   // make a thread local copy
            auto tl_alphabet = alphabet;
            tl_ndx.discard(first);
            for (auto i{first}; i < last; ++i)
                out[i] = static_cast<value_type>(tl_alphabet[tl_ndx()]);
        }
    }
    else if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gnx::execution::parallel_unsequenced_policy>)
    {   // block splitting algorithm
        #pragma omp parallel
        {   auto tidx{omp_get_thread_num()};
            auto size{omp_get_num_threads()};
            Size first{tidx * n / size};
            Size last{(tidx + 1) * n / size};
            auto tl_ndx = ndx;   // make a thread local copy
            auto tl_alphabet = alphabet;
            tl_ndx.discard(first);
            #pragma omp simd
            for (Size i = first; i < last; ++i)
                out[i] = static_cast<value_type>(tl_alphabet[tl_ndx()]);
        }
    }
    else
        rand(out, n, alphabet, seed);
}

// -- weighted random sequence generation ---------------------------------------

/// @brief Generate a random sequence from a given alphabet with weights.
/// @tparam OutputIterator Output iterator type
/// @tparam Size Size type for the number of characters to generate
/// @param out Output iterator to write the random sequence
/// @param n Number of characters to generate
/// @param alphabet String view of the alphabet to sample from
/// @param weights Weights corresponding to each character in the alphabet
/// @param seed Optional seed for the random number generator (default: current time)
template
#if defined(__CUDACC__) || defined(__HIPCC__)
<   host_resident_iterator OutputIterator
#else
<   std::contiguous_iterator OutputIterator
#endif
,   typename Size
>
inline void rand
(   OutputIterator out
,   Size n
,   std::string_view alphabet
,   std::initializer_list<double> weights
,   std::uint64_t seed = 0
)
{   typedef typename std::iterator_traits<OutputIterator>::value_type value_type;

    if (0 == seed)
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    pcg32 r(seed);
    // std::discrete_distribution<> ndx
    // (   std::ranges::begin(weights),
    //     std::ranges::end(weights)
    // );
    trng::discrete_dist ndx
    (   std::ranges::begin(weights),
        std::ranges::end(weights)
    );

    for (Size i = 0; i < n; ++i)
        out[i] = static_cast<value_type>(alphabet[ndx(r)]);
}

namespace random {
///
/// @brief Generate a random DNA sequence.
/// @tparam Sequence Sequence container type
/// @param length Length of the sequence to generate
/// @param seed Optional seed for the random number generator (default: current time)
/// @return Randomly generated DNA sequence
template <sequence_container Sequence>
inline Sequence dna
(   std::size_t length
,   std::uint64_t seed = 0
)
{   const std::string_view alphabet = "ACGT";
    Sequence seq(length);
    rand(seq.begin(), length, alphabet, seed);
    return seq;
}
/// @brief Generate a random RNA sequence.
/// @tparam Sequence Sequence container type
/// @param length Length of the sequence to generate
/// @param seed Optional seed for the random number generator (default: current time)
/// @return Randomly generated RNA sequence
template <sequence_container Sequence>
inline Sequence rna
(   std::size_t length
,   std::uint64_t seed = 0
)
{   const std::string_view alphabet = "ACGU";
    Sequence seq(length);
    rand(seq.begin(), length, alphabet, seed);
    return seq;
}
/// @brief Generate a random peptide sequence.
/// @tparam Sequence Sequence container type
/// @param length Length of the sequence to generate
/// @param seed Optional seed for the random number generator (default: current time)
/// @return Randomly generated peptide sequence
template <sequence_container Sequence>
inline Sequence peptide
(   std::size_t length
,   std::uint64_t seed = 0
)
{   const std::string_view alphabet = "ACDEFGHIKLMNPQRSTVWY";
    Sequence seq(length);
    rand(seq.begin(), length, alphabet, seed);
    return seq;
}
/// @brief Generate a random DNA sequence with specified GC content.
/// @tparam Sequence Sequence container type
/// @param length Length of the sequence to generate
/// @param gc_content Desired GC content percentage (0-100)
/// @param seed Optional seed for the random number generator (default: current time)
/// @return Randomly generated DNA sequence with specified GC content
template <sequence_container Sequence>
inline Sequence dna
(   std::size_t length
,   double gc_content
,   std::uint64_t seed = 0
)
{   const std::string_view alphabet = "ACGT";
    const std::initializer_list<double> weights =
    {   100 - gc_content * 0.5
    ,   gc_content * 0.5
    ,   gc_content * 0.5
    ,   100 - gc_content * 0.5
    };

    Sequence seq(length);
    rand(seq.begin(), length, alphabet, weights, seed);
    return seq;
}

} // namespace random

} // namespace gnx
