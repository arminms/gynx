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

#include <gynx/execution.hpp>
#include <gynx/lut/valid.hpp>

namespace gynx {

/// @brief Sequence type for validation
enum class sequence_type
{   nucleotide   ///< DNA/RNA sequence
,   peptide      ///< Protein/amino acid sequence
};

/// @brief Check if all characters in a sequence are valid.
/// @tparam Iterator Forward iterator type
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @param type Type of sequence to validate (nucleotide or peptide)
/// @return true if all characters are valid, false otherwise
template<typename Iterator>
constexpr bool valid
(   Iterator first
,   Iterator last
,   sequence_type type = sequence_type::nucleotide
)
{   const auto& table = (type == sequence_type::nucleotide) 
    ?   lut::valid_nucleotide 
    :   lut::valid_peptide;

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
/// @param type Type of sequence to validate (nucleotide or peptide)
/// @return true if all characters are valid, false otherwise
template<typename ExecPolicy, std::random_access_iterator Iterator>
requires gynx::is_execution_policy_v<std::decay_t<ExecPolicy>>
inline bool valid
(   ExecPolicy&& policy
,   Iterator first
,   Iterator last
//  must be changed to:
//  std::span<const uint8_t> lutv = std::span<const uint8_t>{lut::valid_nucleotide}
,   sequence_type type = sequence_type::nucleotide
)
{   const auto& table = (type == sequence_type::nucleotide) 
    ?   lut::valid_nucleotide 
    :   lut::valid_peptide;

    auto n = last - first;
    std::size_t sum = 0;

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
        #pragma omp parallel for simd default(none) reduction(+:sum) shared(first,table,n)
        for (int i = 0; i < n; ++i)
            sum += table[static_cast<uint8_t>(first[i])];
    }
    else
        return valid(first, last, type);

    return sum == 0;
}

/// @brief Check if all characters in a sequence container are valid.
/// @tparam Container Container type with begin() and end() methods
/// @param seq The sequence container to validate
/// @param type Type of sequence to validate (nucleotide or peptide)
/// @return true if all characters are valid, false otherwise
template<std::ranges::input_range Container>
constexpr bool valid
(   const Container& seq
,   sequence_type type = sequence_type::nucleotide
)
{   return valid(std::begin(seq), std::end(seq), type);
}

/// @brief Check if all characters in a sequence container are valid using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., std::execution::seq)
/// @tparam Container Container type with begin() and end() methods
/// @param policy Execution policy controlling algorithm execution
/// @param seq The sequence container to validate
/// @param type Type of sequence to validate (nucleotide or peptide)
/// @return true if all characters are valid, false otherwise
template<typename ExecPolicy, std::ranges::input_range Container>
inline bool valid
(   ExecPolicy&& policy
,   const Container& seq
,   sequence_type type = sequence_type::nucleotide
)
{   return valid
    (    std::forward<ExecPolicy>(policy)
    ,    std::begin(seq)
    ,    std::end(seq)
    ,    type
    );
}

/// @brief Check if all characters in a sequence are valid nucleotides.
/// @tparam Iterator Forward iterator type
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @return true if all characters are valid nucleotides, false otherwise
template<typename Iterator>
constexpr bool valid_nucleotide(Iterator first, Iterator last)
{   return valid(first, last, sequence_type::nucleotide);
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
    ,   sequence_type::nucleotide
    );
}

/// @brief Check if all characters in a sequence container are valid nucleotides.
/// @tparam Container Container type with begin() and end() methods
/// @param seq The sequence container to validate
/// @return true if all characters are valid nucleotides, false otherwise
template<std::ranges::input_range Container>
constexpr bool valid_nucleotide(const Container& seq)
{   return valid(seq, sequence_type::nucleotide);
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
    ,   sequence_type::nucleotide
    );
}

/// @brief Check if all characters in a sequence are valid peptides (amino acids).
/// @tparam Iterator Forward iterator type
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @return true if all characters are valid peptides, false otherwise
template<typename Iterator>
constexpr bool valid_peptide(Iterator first, Iterator last)
{   return valid(first, last, sequence_type::peptide);
}

/// @brief Check if all characters in a sequence are valid peptides using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., std::execution::seq)
/// @tparam Iterator Forward iterator type
/// @param policy Execution policy controlling algorithm execution
/// @param first Iterator to the beginning of the sequence
/// @param last Iterator to the end of the sequence
/// @return true if all characters are valid peptides, false otherwise
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
    ,   sequence_type::peptide
    );
}

/// @brief Check if all characters in a sequence container are valid peptides (amino acids).
/// @tparam Container Container type with begin() and end() methods
/// @param seq The sequence container to validate
/// @return true if all characters are valid peptides, false otherwise
template<std::ranges::input_range Container>
constexpr bool valid_peptide(const Container& seq)
{   return valid(seq, sequence_type::peptide);
}

/// @brief Check if all characters in a sequence container are valid peptides using an execution policy.
/// @tparam ExecPolicy Execution policy type (e.g., std::execution::seq)
/// @tparam Container Container type with begin() and end() methods
/// @param policy Execution policy controlling algorithm execution
/// @param seq The sequence container to validate
/// @return true if all characters are valid peptides, false otherwise
template<typename ExecPolicy, std::ranges::input_range Container>
inline bool valid_peptide
(   ExecPolicy&& policy
,   const Container& seq
)
{   return valid
    (   std::forward<ExecPolicy>(policy)
    ,   std::begin(seq)
    ,   std::end(seq)
    ,   sequence_type::peptide
    );
}

} // namespace gynx

#endif  // _GYNX_ALGORITHMS_VALID_HPP_
