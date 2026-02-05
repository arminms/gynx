// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <utility>
#include <cstddef>

#if defined(__CUDACC__)
#include <cub/cub.cuh>
#elif defined(__HIPCC__)
#include <hipcub/hipcub.hpp>
#endif // __CUDACC__

#include <gnx/concepts.hpp>
#include <gnx/execution.hpp>

namespace gnx {

namespace detail {

#if !defined(_WIN32)
#pragma omp declare simd uniform(v1, v2) linear(i:1)
#endif
template<typename T1, typename T2, typename SizeT>
inline int compare_func(const T1* v1, const T2* v2, SizeT i)
{   auto to_upper = [](auto c) { return (c >= 'a' && c <= 'z') ? c - 32 : c; };
    return to_upper(v1[i]) != to_upper(v2[i]);
}

#if defined(__CUDACC__) || defined(__HIPCC__)
namespace kernel {

#define BLOCK_THREADS 256
#define ITEMS_PER_THREAD 4

template<typename T1, typename T2, typename ResultT, typename SizeT>
__global__ void compare_kernel
(   T1* d_in1
,   T2* d_in2
,   ResultT* d_out
,   SizeT n
)
{   // allocate shared memory for the lookup table
    // (No LUT needed for compare, but keeping structure similar if needed later)

    int tid = threadIdx.x;

#if defined(__HIPCC__)
    typedef hipcub::BlockReduce<ResultT, BLOCK_THREADS> BlockReduceT;
#else
    typedef cub::BlockReduce<ResultT, BLOCK_THREADS> BlockReduceT;
#endif
    // allocate shared memory for CUB
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    ResultT local_result = 0;
    auto to_upper = [](auto c) { return (c >= 'a' && c <= 'z') ? c - 32 : c; };

    for (SizeT i = 0; i < ITEMS_PER_THREAD; ++i)
    {   // thread Coarsening loop
        SizeT idx
        =   static_cast<SizeT>(blockIdx.x)
        *   (BLOCK_THREADS * ITEMS_PER_THREAD)
        +   tid
        *   ITEMS_PER_THREAD
        +   i
        ;
        if (idx < n)
            local_result |= (to_upper(d_in1[idx]) != to_upper(d_in2[idx]));
    }

    // block reduction (only thread 0 returns the valid aggregate)
    // Reduce logical OR of mismatches. If 0, then all match.
    ResultT block_result = BlockReduceT(temp_storage).Reduce(local_result, thrust::logical_or<ResultT>());
    if (tid == 0)
        d_out[blockIdx.x] = block_result;
}

} // end kernel namespace

template<typename ExecPolicy, device_resident_iterator Iterator1, device_resident_iterator Iterator2>
inline bool compare_device
(   const ExecPolicy& policy
,   Iterator1 first1
,   Iterator1 last1
,   Iterator2 first2
,   Iterator2 last2
)
{   typedef typename std::iterator_traits<Iterator1>::value_type value_type1;
    typedef typename std::iterator_traits<Iterator2>::value_type value_type2;
    // assume both iterators use same difference_type
    typedef typename std::iterator_traits<Iterator1>::difference_type difference_type;
    typedef int result_type; 

    difference_type n1 = last1 - first1;
    difference_type n2 = last2 - first2;
    if (n1 != n2)
        return false;
    if (n1 <= 0)
        return true;

    difference_type elements_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    unsigned int grid_size = (n1 + elements_per_block - 1) / elements_per_block;

#if defined(__HIPCC__)
    hipStream_t stream = 0;
#else
    cudaStream_t stream = 0;
#endif
    if constexpr (has_stream_member<ExecPolicy>)
       stream = policy.stream();

    thrust::device_vector<result_type> d_partial_results(grid_size);

    kernel::compare_kernel<<<grid_size, BLOCK_THREADS, 0, stream>>>
    (   thrust::raw_pointer_cast(&first1[0])
    ,   thrust::raw_pointer_cast(&first2[0])
    ,   thrust::raw_pointer_cast(d_partial_results.data())
    ,   n1
    );

    result_type result = thrust::reduce
    (   d_partial_results.begin()
    ,   d_partial_results.end()
    ,   0
    ,   thrust::logical_or<result_type>()
    );

    return result == 0;
}
#endif // __CUDACC__

} // end detail namespace

/// @brief Compare if two sequences match (case insensitive).
/// @tparam Iterator1 Forward iterator type for first sequence
/// @tparam Iterator2 Forward iterator type for second sequence
/// @param first1 Iterator to the beginning of the first sequence
/// @param last1 Iterator to the end of the first sequence
/// @param first2 Iterator to the beginning of the second sequence
/// @param last2 Iterator to the end of the second sequence
/// @return true if sequences match, false otherwise
#if defined(__CUDACC__)
template<device_resident_iterator Iterator1, device_resident_iterator Iterator2>
constexpr bool compare
(   Iterator1 first1
,   Iterator1 last1
,   Iterator2 first2
,   Iterator2 last2
)
{   return detail::compare_device(thrust::cuda::par, first1, last1, first2, last2);
}
template<host_resident_iterator Iterator1, typename Iterator2>
#elif defined(__HIPCC__)
template<device_resident_iterator Iterator1, device_resident_iterator Iterator2>
constexpr bool compare
(   Iterator1 first1
,   Iterator1 last1
,   Iterator2 first2
,   Iterator2 last2
)
{   return detail::compare_device(thrust::hip::par, first1, last1, first2, last2);
}
template<host_resident_iterator Iterator1, typename Iterator2>
#else
template<typename Iterator1, typename Iterator2>
#endif
constexpr bool compare
(   Iterator1 first1
,   Iterator1 last1
,   Iterator2 first2
,   Iterator2 last2
)
{   auto to_upper = [](auto c) { return (c >= 'a' && c <= 'z') ? c - 32 : c; };
    while (first1 != last1 && first2 != last2)
    {
        if (to_upper(*first1++) != to_upper(*first2++))
            return false;
    }
    return first1 == last1 && first2 == last2;
}

/// @brief Parallel-enabled comparison using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., gnx::execution::seq)
/// @tparam Iterator1 Random access iterator type for first sequence
/// @tparam Iterator2 Random access iterator type for second sequence
/// @param policy Execution policy controlling algorithm execution
/// @param first1 Iterator to the beginning of the first sequence
/// @param last1 Iterator to the end of the first sequence
/// @param first2 Iterator to the beginning of the second sequence
/// @param last2 Iterator to the end of the second sequence
/// @return true if sequences match, false otherwise
#if defined(__CUDACC__) || defined(__HIPCC__)
template<typename ExecPolicy, device_resident_iterator Iterator1, device_resident_iterator Iterator2>
inline bool compare
(   ExecPolicy&& policy
,   Iterator1 first1
,   Iterator1 last1
,   Iterator2 first2
,   Iterator2 last2
)
{   return detail::compare_device
    (   std::forward<ExecPolicy>(policy)
    ,   first1
    ,   last1
    ,   first2
    ,   last2
    );
}
template<typename ExecPolicy, host_resident_iterator Iterator1, typename Iterator2>
#else
template<typename ExecPolicy, std::random_access_iterator Iterator1, std::random_access_iterator Iterator2>
#endif
requires gnx::is_execution_policy_v<std::decay_t<ExecPolicy>>
inline bool compare
(   ExecPolicy&& policy
,   Iterator1 first1
,   Iterator1 last1
,   Iterator2 first2
,   Iterator2 last2
)
{   typedef typename std::iterator_traits<Iterator1>::difference_type difference_type;
    int result = 0;
    
    difference_type n1 = last1 - first1;
    difference_type n2 = last2 - first2;
    if (n1 != n2) return false;
    difference_type n = n1;

    // compile-time dispatch based on execution policy
    if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gnx::execution::unsequenced_policy>)
    {   const auto vptr1 = &first1[0];
        const auto vptr2 = &first2[0];
        #pragma omp simd reduction(|:result)
        for (int i = 0; i < n; ++i)
            result |= detail::compare_func(vptr1, vptr2, i);
    }
    else if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gnx::execution::parallel_policy>)
    {   // firstprivate(table) must be used once the reference is avoided
        #pragma omp parallel for default(none) reduction(|:result) shared(first1, first2, n)
        for (int i = 0; i < n; ++i)
            result |= detail::compare_func(&first1[0], &first2[0], i);
    }
    else if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gnx::execution::parallel_unsequenced_policy>)
    {   const auto vptr1 = &first1[0];
        const auto vptr2 = &first2[0];
#if defined(_WIN32)
        #pragma omp parallel for default(none) reduction(|:result) shared(vptr1, vptr2, n)
#else
        #pragma omp parallel for simd default(none) reduction(|:result) shared(vptr1, vptr2, n)
#endif // _WIN32
        for (int i = 0; i < n; ++i)
            result |= detail::compare_func(vptr1, vptr2, i);
    }
    else
        return compare(first1, last1, first2, last2);

    return result == 0;
}

/// @brief Compare if two sequence ranges match (case insensitive).
/// @tparam Range1 Range type for first sequence
/// @tparam Range2 Range type for second sequence
/// @param seq1 The first sequence range
/// @param seq2 The second sequence range
/// @return true if sequences match, false otherwise
template<std::ranges::input_range Range1, std::ranges::input_range Range2>
constexpr bool compare
(   const Range1& seq1
,   const Range2& seq2
)
{   return compare(std::begin(seq1), std::end(seq1), std::begin(seq2), std::end(seq2));
}

/// @brief Parallel-enabled comparison of two sequence ranges.
/// @tparam ExecPolicy Execution policy type (e.g., std::execution::seq)
/// @tparam Range1 Range type for first sequence
/// @tparam Range2 Range type for second sequence
/// @param policy Execution policy controlling algorithm execution
/// @param seq1 The first sequence range
/// @param seq2 The second sequence range
/// @return true if sequences match, false otherwise
template<typename ExecPolicy, std::ranges::input_range Range1, std::ranges::input_range Range2>
inline bool compare
(   ExecPolicy&& policy
,   const Range1& seq1
,   const Range2& seq2
)
{   return compare
    (    std::forward<ExecPolicy>(policy)
    ,    std::begin(seq1)
    ,    std::end(seq1)
    ,    std::begin(seq2)
    ,    std::end(seq2)
    );
}

} // namespace gnx
