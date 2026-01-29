// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace gnx {

// forward declaration of sq_gen
template<typename Container, typename Map>
class sq_gen;

/// @brief A non-owning view over a sequence, similar to std::basic_string_view.
/// @tparam Container The container type used by the owning sq_gen.
template<typename Container>
class sq_view_gen : public std::ranges::view_interface<sq_view_gen<Container>>
{
public:
    using value_type = typename Container::value_type;
    using size_type = typename Container::size_type;
    using difference_type = typename Container::difference_type;
    using const_pointer = const value_type*;
    using const_reference = const value_type&;
    using const_iterator = const_pointer;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using container_type = Container;

    static constexpr size_type npos = static_cast<size_type>(-1);

// -- constructors -------------------------------------------------------------
    constexpr sq_view_gen() noexcept = default;

    constexpr sq_view_gen(const value_type* data, size_type count) noexcept
    :   _data(data)
    ,   _size(count)
    {}

    template<typename Map>
    constexpr sq_view_gen(const sq_gen<Container, Map>& seq) noexcept
    :   _data(seq.data())
    ,   _size(seq.size())
    {}

// -- iterators ----------------------------------------------------------------
    constexpr const_iterator begin() const noexcept
    {   return _data;
    }
    constexpr const_iterator end() const noexcept
    {   return _data + _size;
    }
    constexpr const_iterator cbegin() const noexcept
    {   return begin();
    }
    constexpr const_iterator cend() const noexcept
    {   return end();
    }
    constexpr const_reverse_iterator rbegin() const noexcept
    {   return const_reverse_iterator(end());
    }
    constexpr const_reverse_iterator rend() const noexcept
    {   return const_reverse_iterator(begin());
    }
    constexpr const_reverse_iterator crbegin() const noexcept
    {   return rbegin();
    }
    constexpr const_reverse_iterator crend() const noexcept
    {   return rend();
    }

// -- capacity -----------------------------------------------------------------
    constexpr size_type size() const noexcept
    {   return _size;
    }
    [[nodiscard]] constexpr bool empty() const noexcept
    {   return _size == 0;
    }

// -- element access -----------------------------------------------------------
    constexpr const_reference operator[] (size_type pos) const
    {   return _data[pos];
    }
    constexpr const_reference at(size_type pos) const
    {   if (pos >= _size)
            throw std::out_of_range("gnx::sq_view: pos >= size()");
        return _data[pos];
    }
    constexpr const_reference front() const
    {   return _data[0];
    }
    constexpr const_reference back() const
    {   return _data[_size - 1];
    }

    constexpr const_pointer data() const noexcept
    {   return _data;
    }

// -- modifiers ----------------------------------------------------------------

    constexpr void remove_prefix(size_type n)
    {   if (n > _size)
            throw std::out_of_range("gnx::sq_view: remove_prefix overflow");
        _data += n;
        _size -= n;
    }
    constexpr void remove_suffix(size_type n)
    {   if (n > _size)
            throw std::out_of_range("gnx::sq_view: remove_suffix overflow");
        _size -= n;
    }

// -- operations ---------------------------------------------------------------
    constexpr sq_view_gen substr(size_type pos, size_type count = npos) const
    {   if (pos > _size)
            throw std::out_of_range("gnx::sq_view: pos > size()");
        const size_type rlen = std::min(count, static_cast<size_type>(_size - pos));
        return sq_view_gen(_data + pos, rlen);
    }

private:
    const_pointer _data = nullptr;
    size_type _size = 0;
};

// -- comparison operators -----------------------------------------------------
    template<typename Container>
    constexpr bool operator==
    (   const sq_view_gen<Container>& lhs
    ,   const sq_view_gen<Container>& rhs
    )
    {   return lhs.size() == rhs.size()
        && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    // Note: comparisons with sq_gen are provided generically in sq.hpp.
    // Avoid defining overlapping overloads here to prevent ambiguity.

    template<typename Container>
    constexpr bool operator!= (const sq_view_gen<Container>& lhs, const sq_view_gen<Container>& rhs)
    {   return ! (lhs == rhs);
    }

    // Inequality with sq_gen is covered by the generic operators in sq.hpp.

    template<typename Container>
    constexpr std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator== (const sq_view_gen<Container>& lhs, std::string_view rhs)
    {   return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template<typename Container>
    constexpr std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator== (std::string_view lhs, const sq_view_gen<Container>& rhs)
    {   return rhs == lhs;
    }

    template<typename Container>
    constexpr std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator!= (const sq_view_gen<Container>& lhs, std::string_view rhs)
    {   return ! (lhs == rhs);
    }

    template<typename Container>
    constexpr std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator!= (std::string_view lhs, const sq_view_gen<Container>& rhs)
    {   return ! (lhs == rhs);
    }

    // Convenience overloads for C-string literals
    template<typename Container>
    constexpr std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator== (const sq_view_gen<Container>& lhs, const char* rhs)
    {   return lhs == std::string_view(rhs);
    }

    template<typename Container>
    constexpr std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator== (const char* lhs, const sq_view_gen<Container>& rhs)
    {   return std::string_view(lhs) == rhs;
    }

    template<typename Container>
    constexpr std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator!= (const sq_view_gen<Container>& lhs, const char* rhs)
    {   return ! (lhs == rhs);
    }

    template<typename Container>
    constexpr std::enable_if_t<std::is_same_v<typename Container::value_type, char>, bool>
    operator!= (const char* lhs, const sq_view_gen<Container>& rhs)
    {   return ! (lhs == rhs);
    }

// -- aliases ------------------------------------------------------------------
    using sq_view = sq_view_gen<std::vector<char>>;

}   // end gnx namespace

// Enable std::ranges::view concept for sq_view_gen
namespace std::ranges {
    template<typename Container>
    inline constexpr bool enable_view<gnx::sq_view_gen<Container>> = true;
}
