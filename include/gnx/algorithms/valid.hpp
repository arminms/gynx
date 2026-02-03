// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
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
#include <gnx/lut/valid.hpp>

namespace gnx {

namespace detail {

#if defined(_WIN32)
#pragma omp simd uniform(v, table) linear(i:1)
#else
#pragma omp declare simd uniform(v, table) linear(i:1)
#endif
template<typename T, typename SizeT, typename LutT>
inline LutT valid_func(const T* v, SizeT i, const LutT* table)
{   return table[static_cast<LutT>(v[i])];
}

#if defined(__CUDACC__) || defined(__HIPCC__)
namespace kernel {

#define BLOCK_THREADS 256
#define ITEMS_PER_THREAD 4

template<typename T, typename ResultT, typename SizeT, typename TableT>
__global__ void valid_kernel
(   T* d_in
,   ResultT* d_out
,   SizeT n
,   TableT* lut
)
{   // allocate shared memory for the lookup table
    __shared__ TableT shared_lut[256];
    int tid = threadIdx.x;
    shared_lut[tid] = lut[tid];
    __syncthreads();

#if defined(__HIPCC__)
    typedef hipcub::BlockReduce<ResultT, BLOCK_THREADS> BlockReduceT;
#else
    typedef cub::BlockReduce<ResultT, BLOCK_THREADS> BlockReduceT;
#endif
    // allocate shared memory for CUB
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    ResultT local_result = 0;

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
            local_result |= shared_lut[static_cast<TableT>(d_in[idx])];
    }

    // block reduction (only thread 0 returns the valid aggregate)
    ResultT block_result = BlockReduceT(temp_storage).Reduce(local_result, thrust::logical_or<ResultT>());
    if (tid == 0)
        d_out[blockIdx.x] = block_result;
}

} // end kernel namespace

template<typename ExecPolicy, device_resident_iterator Iterator>
inline bool valid_device
(   const ExecPolicy& policy
,   Iterator first
,   Iterator last
,   bool nucleotide = false
)
{   typedef typename std::iterator_traits<Iterator>::value_type value_type;
    typedef typename std::iterator_traits<Iterator>::difference_type difference_type;
    typedef decltype(lut::valid_nucleotide)::value_type result_type;

    difference_type n = last - first;
    if (n <= 0)
        return true;

    const auto& table = nucleotide ? lut::valid_nucleotide : lut::valid_peptide;
    result_type result = 0;
    difference_type elements_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    unsigned int grid_size = (n + elements_per_block - 1) / elements_per_block;

#if defined(__HIPCC__)
    hipStream_t stream = 0;
#else
    cudaStream_t stream = 0;
#endif
    if constexpr (has_stream_member<ExecPolicy>)
       stream = policy.stream();

    thrust::device_vector<result_type> d_partial_results(grid_size);
    thrust::device_vector<uint8_t> d_lut(table.begin(), table.end());

    kernel::valid_kernel<<<grid_size, BLOCK_THREADS, 0, stream>>>
    (   thrust::raw_pointer_cast(&first[0])
    ,   thrust::raw_pointer_cast(d_partial_results.data())
    ,   n
    ,   thrust::raw_pointer_cast(d_lut.data())
    );

    result = thrust::reduce
    (   d_partial_results.begin()
    ,   d_partial_results.end()
    ,   0
    ,   thrust::logical_or<result_type>()
    );

    return result == 0;
}
#endif // __CUDACC__

} // end detail namespace

/// @brief Check if all characters in a sequence are valid.
/// @tparam Iterator Forward iterator type
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @param nucleotide Type of sequence to validate (true for nucleotide,
/// false for peptide/nucleotide)
/// @return true if all characters are valid, false otherwise
#if defined(__CUDACC__)
template<device_resident_iterator Iterator>
constexpr bool valid
(   Iterator first
,   Iterator last
,   bool nucleotide = false
)
{   return detail::valid_device(thrust::cuda::par, first, last, nucleotide);
}
template<host_resident_iterator Iterator>
#elif defined(__HIPCC__)
template<device_resident_iterator Iterator>
constexpr bool valid
(   Iterator first
,   Iterator last
,   bool nucleotide = false
)
{   return detail::valid_device(thrust::hip::par, first, last, nucleotide);
}
template<host_resident_iterator Iterator>
#else
template<typename Iterator>
#endif
constexpr bool valid
(   Iterator first
,   Iterator last
,   bool nucleotide = false
)
{   decltype(lut::valid_peptide)::value_type result = 0;
    const auto& table = nucleotide ? lut::valid_nucleotide : lut::valid_peptide;
    while (first != last)
        result |= table[static_cast<uint8_t>(*first++)];
    return result == 0;
}

/// @brief Parallel-enabled validation using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., gnx::execution::seq)
/// @tparam Iterator Forward iterator type
/// @param policy Execution policy controlling algorithm execution
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @param nucleotide Type of sequence to validate (true for nucleotide,
/// false for peptide/nucleotide)
/// @return true if all characters are valid, false otherwise
#if defined(__CUDACC__) || defined(__HIPCC__)
template<typename ExecPolicy, device_resident_iterator Iterator>
inline bool valid
(   ExecPolicy&& policy
,   Iterator first
,   Iterator last
,   bool nucleotide = false
)
{   return detail::valid_device
    (   std::forward<ExecPolicy>(policy)
    ,   first
    ,   last
    ,   nucleotide
    );
}
template<typename ExecPolicy, host_resident_iterator Iterator>
#else
template<typename ExecPolicy, std::random_access_iterator Iterator>
#endif
requires gnx::is_execution_policy_v<std::decay_t<ExecPolicy>>
inline bool valid
(   ExecPolicy&& policy
,   Iterator first
,   Iterator last
,   bool nucleotide = false
)
{   typedef typename std::iterator_traits<Iterator>::difference_type difference_type;
    decltype(lut::valid_nucleotide)::value_type result = 0;
    const auto& table = nucleotide ? lut::valid_nucleotide : lut::valid_peptide;
    difference_type n = last - first;

    // compile-time dispatch based on execution policy
    if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gnx::execution::unsequenced_policy>)
    {   const auto vptr = &first[0];
        #pragma omp simd reduction(|:result)
        for (int i = 0; i < n; ++i)
            result |= detail::valid_func(vptr, i, table.data());
    }
    else if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gnx::execution::parallel_policy>)
    {   // firstprivate(table) must be used once the reference is avoided
        #pragma omp parallel for default(none) reduction(|:result) shared(first,table,n)
        for (int i = 0; i < n; ++i)
            result |= detail::valid_func(&first[0], i, table.data());
    }
    else if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gnx::execution::parallel_unsequenced_policy>)
    {   const auto vptr = &first[0];
#if defined(_WIN32)
        #pragma omp parallel for default(none) reduction(|:result) shared(vptr,table,n)
#else
        #pragma omp parallel for simd default(none) reduction(|:result) shared(vptr,table,n)
#endif // _WIN32
        for (int i = 0; i < n; ++i)
            result |= detail::valid_func(vptr, i, table.data());
    }
    else
        return valid(first, last, nucleotide);

    return result == 0;
}

/// @brief Check if all characters in a sequence range are valid.
/// @tparam Range Range type with begin() and end() methods
/// @param seq The sequence range to validate
/// @param nucleotide Type of sequence to validate (true for nucleotide,
/// false for peptide/nucleotide)
/// @return true if all characters are valid, false otherwise
template<std::ranges::input_range Range>
constexpr bool valid
(   const Range& seq
,   bool nucleotide = false
)
{   return valid(std::begin(seq), std::end(seq), nucleotide);
}

/// @brief Check if all characters in a sequence container are valid using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., std::execution::seq)
/// @tparam Range Range type with begin() and end() methods
/// @param policy Execution policy controlling algorithm execution
/// @param seq The sequence range to validate
/// @param nucleotide Type of sequence to validate (true for nucleotide,
/// false for peptide/nucleotide)
/// @return true if all characters are valid, false otherwise
template<typename ExecPolicy, std::ranges::input_range Range>
inline bool valid
(   ExecPolicy&& policy
,   const Range& seq
,   bool nucleotide = false
)
{   return valid
    (    std::forward<ExecPolicy>(policy)
    ,    std::begin(seq)
    ,    std::end(seq)
    ,    nucleotide
    );
}

/// @brief Check if all characters in a sequence are valid nucleotides.
/// @tparam Iterator Forward iterator type
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @return true if all characters are valid nucleotides, false otherwise
template<typename Iterator>
constexpr bool valid_nucleotide(Iterator first, Iterator last)
{   return valid(first, last, true);
}

/// @brief Check if all characters in a sequence are valid nucleotides using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., std::execution::seq)
/// @tparam Iterator Forward iterator type
/// @param policy Execution policy controlling algorithm execution
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @return true if all characters are valid nucleotides, false otherwise
template<typename ExecPolicy, typename Iterator>
inline bool valid_nucleotide
(   ExecPolicy&& policy
,   Iterator first
,   Iterator last
)
{   return valid
    (   std::forward<ExecPolicy>(policy)
    ,   first
    ,   last
    ,   true
    );
}

/// @brief Check if all characters in a sequence container are valid nucleotides.
/// @tparam Container Container type with begin() and end() methods
/// @param seq The sequence container to validate
/// @return true if all characters are valid nucleotides, false otherwise
template<std::ranges::input_range Container>
constexpr bool valid_nucleotide(const Container& seq)
{   return valid(seq, true);
}

/// @brief Check if all characters in a sequence container are valid nucleotides using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., std::execution::seq)
/// @tparam Container Container type with begin() and end() methods
/// @param policy Execution policy controlling algorithm execution
/// @param seq The sequence container to validate
/// @return true if all characters are valid nucleotides, false otherwise
template<typename ExecPolicy, std::ranges::input_range Container>
inline bool valid_nucleotide
(   ExecPolicy&& policy
,   const Container& seq
)
{   return valid
    (   std::forward<ExecPolicy>(policy)
    ,   std::begin(seq)
    ,   std::end(seq)
    ,   true
    );
}

/// @brief Check if all characters in a sequence are valid peptides (amino acids).
/// @tparam Iterator Forward iterator type
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @return true if all characters are valid amino acids/nucleotides, false otherwise
template<typename Iterator>
constexpr bool valid_peptide(Iterator first, Iterator last)
{   return valid(first, last, false);
}

/// @brief Check if all characters in a sequence are valid peptides using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., std::execution::seq)
/// @tparam Iterator Forward iterator type
/// @param policy Execution policy controlling algorithm execution
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @return true if all characters are valid amino acids/nucleotides, false otherwise
template<typename ExecPolicy, typename Iterator>
inline bool valid_peptide
(   ExecPolicy&& policy
,   Iterator first
,   Iterator last
)
{   return valid
    (   std::forward<ExecPolicy>(policy)
    ,   first
    ,   last
    ,   false
    );
}

/// @brief Check if all characters in a sequence container are valid peptides (amino acids).
/// @tparam Container Container type with begin() and end() methods
/// @param seq The sequence container to validate
/// @return true if all characters are valid amino acids/nucleotides, false otherwise
template<std::ranges::input_range Container>
constexpr bool valid_peptide(const Container& seq)
{   return valid(seq, false);
}

/// @brief Check if all characters in a sequence container are valid peptides using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., std::execution::seq)
/// @tparam Container Container type with begin() and end() methods
/// @param policy Execution policy controlling algorithm execution
/// @param seq The sequence container to validate
/// @return true if all characters are valid amino acids/nucleotides, false otherwise
template<typename ExecPolicy, std::ranges::input_range Container>
inline bool valid_peptide
(   ExecPolicy&& policy
,   const Container& seq
)
{   return valid
    (   std::forward<ExecPolicy>(policy)
    ,   std::begin(seq)
    ,   std::end(seq)
    ,   false
    );
}

} // namespace gnx
