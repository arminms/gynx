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
#ifndef _GNX_LUT_PHRED64_HPP_
#define _GNX_LUT_PHRED64_HPP_

#include <array>
#include <cmath>

namespace gnx::lut {

// Generate the Lookup Table at Compile Time
// This maps every ASCII character to Error Probability equivalent.
constexpr std::array<double, 256> create_phred64_table()
{   std::array<double, 256> table{};
    table.fill(1.0); // C++20 feature
    // for (size_t i = 0; i < 256; ++i)
    //     table[i] = 1.0; 
    // initialize ASCII 64 (@) to 126 (~)
    for (int c = 64; c < 127; ++c)
    {   int q_score = c - 64;
        // P = 10 ^ (-Q/10)
        table[c] = std::pow(10.0, -q_score / 10.0);
    }
    return table;
}

// Instantiate the table in static memory (read-only, hot cache).
// Always cast your input char to uint8_t when indexing into this table
// to avoid negative indices due to sign extension.
// Example: double p = gnx::lut::phred64[static_cast<uint8_t>(ch)];
thread_local static constexpr auto phred64 = create_phred64_table();

} // namespace gnx::lut

#endif  // _GNX_LUT_PHRED64_HPP_