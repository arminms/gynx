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
#ifndef _GYNX_ALGORITHMS_VALID_HPP_
#define _GYNX_ALGORITHMS_VALID_HPP_

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <utility>
#include <cstddef>

#if defined(__CUDACC__)
#include <cub/cub.cuh>
#endif // __CUDACC__

#include <gynx/concepts.hpp>
#include <gynx/execution.hpp>
#include <gynx/lut/valid.hpp>

namespace gynx {

namespace detail {

#if defined(__CUDACC__)
namespace kernel {

#define BLOCK_THREADS 256
#define ITEMS_PER_THREAD 4

template<typename T, typename SumT, typename SizeT, typename TableT>
__global__ void valid_kernel
(   T* d_in
,   SumT* d_out
,   SizeT n
,   TableT* lut
)
{   // allocate shared memory for the lookup table
    __shared__ TableT shared_lut[256];
    int tid = threadIdx.x;
    shared_lut[tid] = lut[tid];
    __syncthreads();

    typedef cub::BlockReduce<SumT, BLOCK_THREADS> BlockReduceT;
    // allocate shared memory for CUB
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    SumT local_sum = 0;

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
            local_sum += shared_lut[static_cast<TableT>(d_in[idx])];
    }

    // block reduction (only thread 0 returns the valid aggregate)
    T block_sum = BlockReduceT(temp_storage).Sum(local_sum);

    if (tid == 0)
        d_out[blockIdx.x] = block_sum;
}

} // end kernel namespace

template<typename ExecPolicy, device_resident_iterator Iterator>
bool valid_device
(   const ExecPolicy& policy
,   Iterator first
,   Iterator last
,   bool nucleotide = false
)
{   typedef typename std::iterator_traits<Iterator>::value_type value_type;
    typedef typename std::iterator_traits<Iterator>::difference_type difference_type;

    const auto& table = nucleotide ? lut::valid_nucleotide : lut::valid_peptide;

    cudaStream_t stream = 0;
    if constexpr (has_stream_member<ExecPolicy>)
       stream = policy.stream();

    difference_type n = last - first;
    difference_type sum = 0;
    difference_type elements_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;

    unsigned int grid_size = (n + elements_per_block - 1) / elements_per_block;

    thrust::device_vector<difference_type> d_partial_sums(grid_size);
    thrust::device_vector<uint8_t> d_lut(table.begin(), table.end());

    kernel::valid_kernel<<<grid_size, BLOCK_THREADS, 0, stream>>>
    (   thrust::raw_pointer_cast(&first[0])
    ,   thrust::raw_pointer_cast(d_partial_sums.data())
    ,   n
    ,   thrust::raw_pointer_cast(d_lut.data())
    );

    sum = thrust::reduce
    (   d_partial_sums.begin()
    ,   d_partial_sums.end()
    ,   0
    ,   thrust::plus<difference_type>()
    );

    return sum == 0;
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
#else
template<typename Iterator>
#endif
constexpr bool valid
(   Iterator first
,   Iterator last
,   bool nucleotide = false
)
{   const auto& table = nucleotide ? lut::valid_nucleotide : lut::valid_peptide;
    std::size_t sum = 0;
    while (first != last)
        sum += table[static_cast<uint8_t>(*first++)];
    return sum == 0;
}

/// @brief Parallel-enabled validation using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., gynx::execution::seq)
/// @tparam Iterator Forward iterator type
/// @param policy Execution policy controlling algorithm execution
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @param nucleotide Type of sequence to validate (true for nucleotide,
/// false for peptide/nucleotide)
/// @return true if all characters are valid, false otherwise
#if defined(__CUDACC__)
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
requires gynx::is_execution_policy_v<std::decay_t<ExecPolicy>>
inline bool valid
(   ExecPolicy&& policy
,   Iterator first
,   Iterator last
,   bool nucleotide = false
)
{   typedef typename std::iterator_traits<Iterator>::value_type value_type;
    typedef typename std::iterator_traits<Iterator>::difference_type difference_type;

    const auto& table = nucleotide ? lut::valid_nucleotide : lut::valid_peptide;

    difference_type n = last - first;
    difference_type sum = 0;

    // compile-time dispatch based on execution policy
    if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gynx::execution::unsequenced_policy>)
    {
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < n; ++i)
            sum += table[static_cast<uint8_t>(first[i])];
    }
    else if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gynx::execution::parallel_policy>)
    {   // firstprivate(table) must be used once the reference is avoided
        #pragma omp parallel for default(none) reduction(+:sum) shared(first,table,n)
        for (int i = 0; i < n; ++i)
            sum += table[static_cast<uint8_t>(first[i])];
    }
    else if constexpr (std::is_same_v<std::decay_t<ExecPolicy>, gynx::execution::parallel_unsequenced_policy>)
    {
#if defined(_WIN32)
        #pragma omp parallel for default(none) reduction(+:sum) shared(first,table,n)
#else
        #pragma omp parallel for simd default(none) reduction(+:sum) shared(first,table,n)
#endif // _WIN32
        for (int i = 0; i < n; ++i)
            sum += table[static_cast<uint8_t>(first[i])];
    }
    else
        return valid(first, last, nucleotide);

    return sum == 0;
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

} // namespace gynx

#endif  // _GYNX_ALGORITHMS_VALID_HPP_
