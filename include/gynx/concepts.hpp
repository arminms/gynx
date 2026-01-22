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
#ifndef _GYNX_CONCEPTS_HPP_
#define _GYNX_CONCEPTS_HPP_

#include <type_traits>
#include <concepts>
#include <vector>
#include <iterator>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/iterator_traits.h>
#endif // __CUDACC__

namespace gynx {

template<typename T>
concept has_value_type = requires
{   typename T::value_type;
};

template<typename T>
concept has_size_type = requires
{   typename T::size_type;
};

template<typename T>
concept has_container_type = requires
{   typename T::container_type;
};

template<typename T>
concept has_map_type = requires
{   typename T::map_type;
};

template<typename T>
concept sequence_container
=   has_container_type<T>
&&  has_map_type<T>
&&  has_value_type<T>
&&  has_size_type<T>
;

#if defined(__CUDACC__) || defined(__HIPCC__)
// helper alias to get the system tag of a container
template <typename Container>
using system_tag_t = typename thrust::iterator_system<typename Container::const_iterator>::type;

// checks if data is on the Device (GPU)
template <typename Container>
concept device_resident = std::convertible_to<system_tag_t<Container>, thrust::device_system_tag>;

// checks if data is on the Host (CPU)
// this covers both thrust::host_vector and std::vector
template <typename Container>
concept host_resident = std::convertible_to<system_tag_t<Container>, thrust::host_system_tag>;

// helper alias to extract the system tag from an Iterator
template <typename Iter>
using iterator_system_t = typename thrust::iterator_system<Iter>::type;

// iterator points to Device memory
template <typename Iter>
concept device_resident_iterator = std::convertible_to<iterator_system_t<Iter>, thrust::device_system_tag>;

// iterator points to Host memory
template <typename Iter>
concept host_resident_iterator = std::convertible_to<iterator_system_t<Iter>, thrust::host_system_tag>;

// helper to extract CUDA stream from execution policies
template <typename T>
concept has_stream_member = requires(T a)
#if defined(__HIPCC__)
{   { a.stream() } -> std::same_as<hipStream_t>;
#else
{   { a.stream() } -> std::same_as<cudaStream_t>;
#endif // __HIPCC__
};
#endif // __CUDACC__

} // namespace gynx

#endif // _GYNX_CONCEPTS_HPP_
