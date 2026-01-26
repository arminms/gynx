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
#ifndef _GNX_LUT_VALID_HPP_
#define _GNX_LUT_VALID_HPP_

#include <array>
#include <cstdint>

namespace gnx::lut {

/// @brief Compile-time generated lookup table for valid nucleotide characters.
/// Valid characters include: A, C, G, T, U, N (both uppercase and lowercase)
constexpr std::array<uint8_t, 256> create_valid_nucleotide_table()
{   std::array<uint8_t, 256> table{};
    table.fill(1); // 1 = invalid, 0 = valid

    // Valid nucleotide characters (uppercase)
    table['A'] = 0;
    table['C'] = 0;
    table['G'] = 0;
    table['T'] = 0;
    table['U'] = 0;
    table['N'] = 0; // ambiguous/unknown base

    // Valid nucleotide characters (lowercase)
    table['a'] = 0;
    table['c'] = 0;
    table['g'] = 0;
    table['t'] = 0;
    table['u'] = 0;
    table['n'] = 0;

    // IUPAC ambiguity codes (uppercase)
    table['R'] = 0; // A or G (puRine)
    table['Y'] = 0; // C or T (pYrimidine)
    table['S'] = 0; // G or C (Strong)
    table['W'] = 0; // A or T (Weak)
    table['K'] = 0; // G or T (Keto)
    table['M'] = 0; // A or C (aMino)
    table['B'] = 0; // C, G or T (not A)
    table['D'] = 0; // A, G or T (not C)
    table['H'] = 0; // A, C or T (not G)
    table['V'] = 0; // A, C or G (not T)

    // IUPAC ambiguity codes (lowercase)
    table['r'] = 0;
    table['y'] = 0;
    table['s'] = 0;
    table['w'] = 0;
    table['k'] = 0;
    table['m'] = 0;
    table['b'] = 0;
    table['d'] = 0;
    table['h'] = 0;
    table['v'] = 0;

    return table;
}

/// @brief Compile-time generated lookup table for valid peptide (amino acid) characters.
/// Valid characters include: 20 standard amino acids + ambiguous codes (both uppercase and lowercase)
constexpr std::array<uint8_t, 256> create_valid_peptide_table()
{   std::array<uint8_t, 256> table{};
    table.fill(1); // 1 = invalid, 0 = valid

    // 20 standard amino acids (uppercase)
    table['A'] = 0; // Alanine
    table['C'] = 0; // Cysteine
    table['D'] = 0; // Aspartic acid
    table['E'] = 0; // Glutamic acid
    table['F'] = 0; // Phenylalanine
    table['G'] = 0; // Glycine
    table['H'] = 0; // Histidine
    table['I'] = 0; // Isoleucine
    table['K'] = 0; // Lysine
    table['L'] = 0; // Leucine
    table['M'] = 0; // Methionine
    table['N'] = 0; // Asparagine
    table['P'] = 0; // Proline
    table['Q'] = 0; // Glutamine
    table['R'] = 0; // Arginine
    table['S'] = 0; // Serine
    table['T'] = 0; // Threonine
    table['V'] = 0; // Valine
    table['W'] = 0; // Tryptophan
    table['Y'] = 0; // Tyrosine

    // 20 standard amino acids (lowercase)
    table['a'] = 0;
    table['c'] = 0;
    table['d'] = 0;
    table['e'] = 0;
    table['f'] = 0;
    table['g'] = 0;
    table['h'] = 0;
    table['i'] = 0;
    table['k'] = 0;
    table['l'] = 0;
    table['m'] = 0;
    table['n'] = 0;
    table['p'] = 0;
    table['q'] = 0;
    table['r'] = 0;
    table['s'] = 0;
    table['t'] = 0;
    table['v'] = 0;
    table['w'] = 0;
    table['y'] = 0;

    // Ambiguous/special codes (uppercase)
    table['B'] = 0; // Aspartic acid or Asparagine
    table['Z'] = 0; // Glutamic acid or Glutamine
    table['X'] = 0; // Unknown or any amino acid
    table['*'] = 0; // Stop codon
    table['U'] = 0; // Selenocysteine
    table['O'] = 0; // Pyrrolysine
    table['J'] = 0; // Leucine or Isoleucine

    // Ambiguous/special codes (lowercase)
    table['b'] = 0;
    table['z'] = 0;
    table['x'] = 0;
    table['u'] = 0;
    table['o'] = 0;
    table['j'] = 0;

    return table;
}

/// @brief Instantiate the nucleotide validation table in static memory.
/// Example: bool is_valid = gnx::lut::valid_nucleotide[static_cast<uint8_t>(ch)];
inline constexpr auto valid_nucleotide = create_valid_nucleotide_table();

/// @brief Instantiate the peptide validation table in static memory.
/// Example: bool is_valid = gnx::lut::valid_peptide[static_cast<uint8_t>(ch)];
inline constexpr auto valid_peptide = create_valid_peptide_table();

} // namespace gnx::lut

#endif  // _GNX_LUT_VALID_HPP_
