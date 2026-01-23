//
// Copyright (c) 2025 Armin Sobhani
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
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

#include <gynx/concepts.hpp>
#include <gynx/execution.hpp>

namespace gynx {

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
    // thrust::copy
    // (   thrust::cuda::par.on(stream)
    // ,   alphabet.begin()
    // ,   alphabet.end()
    // ,   d_alphabet.begin()
    // );

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
requires gynx::is_execution_policy_v<std::decay_t<ExecPolicy>>
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
    if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gynx::execution::unsequenced_policy>)
    {
        #pragma omp simd
        for (Size i = 0; i < n; ++i)
            out[i] = static_cast<value_type>(alphabet[ndx()]);
    }
    else if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gynx::execution::parallel_policy>)
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
    else if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gynx::execution::parallel_unsequenced_policy>)
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

template
<   std::contiguous_iterator OutputIterator
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

// #if defined(__CUDACC__) || defined(__HIPCC__)
// template <device_resident Sequence, typename ExecPolicy>
// #else
// template <sequence_container Sequence, typename ExecPolicy>
// requires gynx::is_execution_policy_v<std::decay_t<ExecPolicy>>
// #endif
// inline Sequence dna
// (   ExecPolicy&& policy
// ,   std::size_t length
// ,   std::uint64_t seed = 0
// )
// {   const std::string_view alphabet = "ACGT";
//     Sequence seq(length);
//     rand(std::forward<ExecPolicy>(policy), seq.begin(), length, alphabet, seed);
//     return seq;
// }

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

} // namespace gynx
