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
#include <vector>
#include <tuple>
#include <array>

#include <gnx/concepts.hpp>
#include <gnx/lut/blosum.hpp>
#include <gnx/lut/pam.hpp>

namespace gnx {

/// @brief Direction for traceback in Smith-Waterman alignment
enum class alignment_direction : uint8_t
{   none = 0
,   diagonal = 1
,   up = 2
,   left = 3
};

/// @brief Result structure for local alignment
struct alignment_result
{   int score;                                      ///< Alignment score
    std::size_t max_i;                              ///< Row index of maximum score
    std::size_t max_j;                              ///< Column index of maximum score
    std::vector<alignment_direction> traceback;     ///< Traceback path
    std::string aligned_seq1;                       ///< First aligned sequence (with gaps)
    std::string aligned_seq2;                       ///< Second aligned sequence (with gaps)
};

namespace detail {

/// @brief Default scoring function for nucleotide sequences
/// @param a First character
/// @param b Second character
/// @param match Match score (default 2)
/// @param mismatch Mismatch penalty (default -1)
/// @return Score for aligning characters a and b
inline constexpr int default_score
(   char a
,   char b
,   int match = 2
,   int mismatch = -1
)
{   // Convert to uppercase for comparison
    // char a_upper = (a >= 'a' && a <= 'z') ? a - 32 : a;
    // char b_upper = (b >= 'a' && b <= 'z') ? b - 32 : b;
    // return (a_upper == b_upper) ? match : mismatch;
    return ((a ^ b) & 0xDF) == 0 ? match : mismatch;
}

/// @brief Perform traceback from maximum score position
/// @tparam Iterator1 Iterator type for first sequence
/// @tparam Iterator2 Iterator type for second sequence
/// @param traceback_matrix Matrix storing traceback directions
/// @param seq1_first Iterator to start of first sequence
/// @param seq2_first Iterator to start of second sequence
/// @param max_i Row position of maximum score
/// @param max_j Column position of maximum score
/// @param rows Number of rows in matrix
/// @param cols Number of columns in matrix
/// @return Tuple containing (traceback path, aligned seq1, aligned seq2)
template<typename Iterator1, typename Iterator2>
inline auto perform_traceback
(   const std::vector<alignment_direction>& traceback_matrix
,   Iterator1 seq1_first
,   Iterator2 seq2_first
,   std::size_t max_i
,   std::size_t max_j
,   std::size_t rows
,   std::size_t cols
)
{   std::vector<alignment_direction> path;
    std::string aligned1, aligned2;
    
    std::size_t i = max_i;
    std::size_t j = max_j;

    while (i > 0 && j > 0)
    {   alignment_direction dir = traceback_matrix[i * cols + j];
        if (dir == alignment_direction::none)
            break;

        path.push_back(dir);

        if (dir == alignment_direction::diagonal)
        {   aligned1 += seq1_first[i - 1];
            aligned2 += seq2_first[j - 1];
            --i;
            --j;
        }
        else if (dir == alignment_direction::up)
        {   aligned1 += seq1_first[i - 1];
            aligned2 += '-';
            --i;
        }
        else // left
        {   aligned1 += '-';
            aligned2 += seq2_first[j - 1];
            --j;
        }
    }

    // Reverse the aligned sequences since we built them backwards
    std::reverse(aligned1.begin(), aligned1.end());
    std::reverse(aligned2.begin(), aligned2.end());
    std::reverse(path.begin(), path.end());

    return std::make_tuple(path, aligned1, aligned2);
}

} // end detail namespace

/// @brief Perform Smith-Waterman local alignment between two sequences (iterator version)
/// @tparam Iterator1 Iterator type for first sequence
/// @tparam Iterator2 Iterator type for second sequence
/// @param seq1_first Iterator to the beginning of the first sequence
/// @param seq1_last Iterator to the end of the first sequence
/// @param seq2_first Iterator to the beginning of the second sequence
/// @param seq2_last Iterator to the end of the second sequence
/// @param match Score for matching characters (default 2)
/// @param mismatch Penalty for mismatching characters (default -1)
/// @param gap_penalty Penalty for gap insertion (default -1)
/// @return alignment_result containing score, positions, and aligned sequences
template<std::input_iterator Iterator1, std::input_iterator Iterator2>
alignment_result local_align
(   Iterator1 seq1_first
,   Iterator1 seq1_last
,   Iterator2 seq2_first
,   Iterator2 seq2_last
,   int match = 2
,   int mismatch = -1
,   int gap_penalty = -1
)
{   typedef typename std::iterator_traits<Iterator1>::difference_type difference_type1;
    typedef typename std::iterator_traits<Iterator2>::difference_type difference_type2;

    difference_type1 len1 = seq1_last - seq1_first;
    difference_type2 len2 = seq2_last - seq2_first;

    if (len1 <= 0 || len2 <= 0)
        return {0, 0, 0, {}, "", ""};

    std::size_t rows = len1 + 1;
    std::size_t cols = len2 + 1;

    // Initialize score matrix (flattened 2D array)
    std::vector<int> score_matrix(rows * cols, 0);

    // Initialize traceback matrix
    std::vector<alignment_direction> traceback_matrix(rows * cols, alignment_direction::none);

    int max_score = 0;
    std::size_t max_i = 0;
    std::size_t max_j = 0;

    // Fill the score matrix using Smith-Waterman algorithm
    for (std::size_t i = 1; i < rows; ++i)
    {   for (std::size_t j = 1; j < cols; ++j)
        {   char c1 = seq1_first[i - 1];
            char c2 = seq2_first[j - 1];

            int score_diag = score_matrix[(i - 1) * cols + (j - 1)]
                           + detail::default_score(c1, c2, match, mismatch);
            int score_up   = score_matrix[(i - 1) * cols + j] + gap_penalty;
            int score_left = score_matrix[i * cols + (j - 1)] + gap_penalty;

            // Smith-Waterman: max of (0, diagonal, up, left)
            int current_score = std::max({0, score_diag, score_up, score_left});

            score_matrix[i * cols + j] = current_score;

            // Track traceback direction
            if (current_score == 0)
                traceback_matrix[i * cols + j] = alignment_direction::none;
            else if (current_score == score_diag)
                traceback_matrix[i * cols + j] = alignment_direction::diagonal;
            else if (current_score == score_up)
                traceback_matrix[i * cols + j] = alignment_direction::up;
            else
                traceback_matrix[i * cols + j] = alignment_direction::left;

            // Track maximum score
            if (current_score > max_score)
            {   max_score = current_score;
                max_i = i;
                max_j = j;
            }
        }
    }

    // Perform traceback
    auto [path, aligned1, aligned2] = detail::perform_traceback
    (   traceback_matrix
    ,   seq1_first
    ,   seq2_first
    ,   max_i
    ,   max_j
    ,   rows
    ,   cols
    );

    return {max_score, max_i, max_j, path, aligned1, aligned2};
}

/// @brief Perform Smith-Waterman local alignment with substitution matrix
/// @tparam Iterator1 Iterator type for first sequence
/// @tparam Iterator2 Iterator type for second sequence
/// @tparam SubMatrix Substitution matrix type (24x24 array)
/// @param seq1_first Iterator to the beginning of the first sequence
/// @param seq1_last Iterator to the end of the first sequence
/// @param seq2_first Iterator to the beginning of the second sequence
/// @param seq2_last Iterator to the end of the second sequence
/// @param subst_matrix Substitution matrix (e.g., BLOSUM62, PAM250)
/// @param gap_penalty Penalty for gap insertion (default -8)
/// @return alignment_result containing score, positions, and aligned sequences
template<std::input_iterator Iterator1, std::input_iterator Iterator2, typename SubMatrix>
alignment_result local_align
(   Iterator1 seq1_first
,   Iterator1 seq1_last
,   Iterator2 seq2_first
,   Iterator2 seq2_last
,   const SubMatrix& subst_matrix
,   int gap_penalty = -8
)
{   typedef typename std::iterator_traits<Iterator1>::difference_type difference_type1;
    typedef typename std::iterator_traits<Iterator2>::difference_type difference_type2;

    difference_type1 len1 = seq1_last - seq1_first;
    difference_type2 len2 = seq2_last - seq2_first;

    if (len1 <= 0 || len2 <= 0)
        return {0, 0, 0, {}, "", ""};

    std::size_t rows = len1 + 1;
    std::size_t cols = len2 + 1;

    // Initialize score matrix (flattened 2D array)
    std::vector<int> score_matrix(rows * cols, 0);

    // Initialize traceback matrix
    std::vector<alignment_direction> traceback_matrix(rows * cols, alignment_direction::none);

    int max_score = 0;
    std::size_t max_i = 0;
    std::size_t max_j = 0;

    // Fill the score matrix using Smith-Waterman algorithm with substitution matrix
    for (std::size_t i = 1; i < rows; ++i)
    {   for (std::size_t j = 1; j < cols; ++j)
        {   char c1 = seq1_first[i - 1];
            char c2 = seq2_first[j - 1];

            // Use substitution matrix for scoring
            int idx1 = lut::aa_to_index(c1);
            int idx2 = lut::aa_to_index(c2);
            int match_score = subst_matrix[idx1][idx2];

            int score_diag = score_matrix[(i - 1) * cols + (j - 1)] + match_score;
            int score_up   = score_matrix[(i - 1) * cols + j] + gap_penalty;
            int score_left = score_matrix[i * cols + (j - 1)] + gap_penalty;

            // Smith-Waterman: max of (0, diagonal, up, left)
            int current_score = std::max({0, score_diag, score_up, score_left});

            score_matrix[i * cols + j] = current_score;

            // Track traceback direction
            if (current_score == 0)
                traceback_matrix[i * cols + j] = alignment_direction::none;
            else if (current_score == score_diag)
                traceback_matrix[i * cols + j] = alignment_direction::diagonal;
            else if (current_score == score_up)
                traceback_matrix[i * cols + j] = alignment_direction::up;
            else
                traceback_matrix[i * cols + j] = alignment_direction::left;

            // Track maximum score
            if (current_score > max_score)
            {   max_score = current_score;
                max_i = i;
                max_j = j;
            }
        }
    }

    // Perform traceback
    auto [path, aligned1, aligned2] = detail::perform_traceback
    (   traceback_matrix
    ,   seq1_first
    ,   seq2_first
    ,   max_i
    ,   max_j
    ,   rows
    ,   cols
    );

    return {max_score, max_i, max_j, path, aligned1, aligned2};
}

/// @brief Perform Smith-Waterman local alignment between two sequence ranges
/// (convenience wrapper)
/// @tparam Range1 Range type for first sequence
/// @tparam Range2 Range type for second sequence
/// @param seq1 The first sequence range
/// @param seq2 The second sequence range
/// @return alignment_result containing score, positions, and aligned sequences
template<std::ranges::input_range Range1, std::ranges::input_range Range2>
inline alignment_result local_align
(   const Range1& seq1
,   const Range2& seq2
)
{   return local_align
    (   std::begin(seq1)
    ,   std::end(seq1)
    ,   std::begin(seq2)
    ,   std::end(seq2)
    );
}

/// @brief Perform Smith-Waterman local alignment with substitution matrix
/// (convenience wrapper for ranges)
/// @tparam Range1 Range type for first sequence
/// @tparam Range2 Range type for second sequence
/// @tparam SubMatrix Substitution matrix type
/// @param seq1 The first sequence range
/// @param seq2 The second sequence range
/// @param subst_matrix Substitution matrix (e.g., BLOSUM62, PAM250)
/// @param gap_penalty Penalty for gap insertion (default -8)
/// @return alignment_result containing score, positions, and aligned sequences
template<std::ranges::input_range Range1, std::ranges::input_range Range2, typename SubMatrix>
inline alignment_result local_align
(   const Range1& seq1
,   const Range2& seq2
,   const SubMatrix& subst_matrix
,   int gap_penalty = -8
)
{   return local_align
    (   std::begin(seq1)
    ,   std::end(seq1)
    ,   std::begin(seq2)
    ,   std::end(seq2)
    ,   subst_matrix
    ,   gap_penalty
    );
}

} // namespace gnx
