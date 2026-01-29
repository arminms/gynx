// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <concepts>
#include <sstream>
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <any>
#include <memory>
#include <typeindex>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
    #include <thrust/host_vector.h>
    #include <thrust/device_vector.h>
    #include <thrust/universal_vector.h>
    #include <thrust/memory.h>
#endif // __CUDACC__

#include <gnx/sq_view.hpp>
#include <gnx/visitor.hpp>
#include <gnx/io/fastaqz.hpp>
#include <gnx/memory.hpp>

namespace gnx {
//
/// @brief A generic sequence class template with tagged data support.
/// @tparam Container The underlying container type to hold the sequence.
/// @tparam Map The type of the map used for tagged data storage.
template
<   typename Container
,   typename Map = std::unordered_map<std::string, std::any>
>
class sq_gen
{   Container                _sq;  // sequence
    std::unique_ptr<Map> _ptr_td;  // pointer to tagged data

public:
    using value_type = typename Container::value_type;
    using size_type = typename Container::size_type;
    using difference_type = typename Container::difference_type;
    using reference = typename Container::reference;
    using const_reference = typename Container::const_reference;
    using iterator = typename Container::iterator;
    using const_iterator = typename Container::const_iterator;
    using reverse_iterator = typename Container::reverse_iterator;
    using const_reverse_iterator = typename Container::const_reverse_iterator;
    using container_type = Container;
    using map_type = Map;

    static constexpr size_type npos = static_cast<size_type>(-1);

// -- constructors -------------------------------------------------------------
    ///
    /// Default constructor. Constructs an empty sequence.
    sq_gen() noexcept
    :   _sq()
    ,   _ptr_td()
    {}
    ///
    /// @brief Constructs a sequence from a string view.
    /// @param sq The string view representing the sequence.
    explicit sq_gen(std::string_view sq)
    :   _sq(std::begin(sq), std::end(sq))
    ,   _ptr_td()
    {}
    ///
    /// Constructs a sequence with @a count residues.
    /// @param count The number of residues in the sequence.
    sq_gen(size_type count)
    :   _sq(count)
    ,   _ptr_td()
    {}
    ///
    /// Constructs a sequence with @a count residues, each initialized to
    /// @a value (default is 'A' (ASCII 65)).
    /// @param count The number of residues in the sequence.
    /// @param value The residue value to initialize each position with.
#if defined(__CUDACC__) || defined(__HIPCC__)
    sq_gen(size_type count, const_reference value)
    requires (!std::is_same_v<Container, thrust::device_vector<value_type>>)
#if defined(__HIPCC__)
    || (!std::is_same_v<Container, gnx::unified_vector<value_type>>)
#endif //__HIPCC__
    :   _sq(count, value)
    ,   _ptr_td()
    {}
#else
    sq_gen(size_type count, const_reference value)
    :   _sq(count, value)
    ,   _ptr_td()
    {}
#endif //__CUDACC__
    ///
    /// @brief Constructs a sequence from a sequence view.
    /// @param sv The sequence view to construct the sequence from.
#if defined(__CUDACC__) || defined(__HIPCC__)
    explicit sq_gen(sq_view_gen<Container> sv)
    requires (!std::is_same_v<Container, thrust::device_vector<value_type>>)
    :   _sq(std::begin(sv), std::end(sv))
    ,   _ptr_td()
    {}
#else
    explicit sq_gen(sq_view_gen<Container> sv)
    :   _sq(std::begin(sv), std::end(sv))
    ,   _ptr_td()
    {}
#endif //__CUDACC__
    ///
    template<typename InputIt>
    requires std::input_iterator<InputIt> &&
        std::is_same_v<typename std::iterator_traits<InputIt>::value_type, value_type>
    /// @brief Constructs a sequence from a pair of iterators.
    /// @tparam InputIt The type of the input iterators.
    /// @param first The beginning iterator of the sequence.
    /// @param last The ending iterator of the sequence.
    sq_gen(InputIt first, InputIt last)
    :   _sq(first, last)
    ,   _ptr_td()
    {}
    ///
    /// Copy constructor.
    sq_gen(const sq_gen& other)
    :   _sq(other._sq)
    ,   _ptr_td(other._ptr_td ? std::make_unique<Map>(*other._ptr_td) : nullptr)
    {}
    ///
    /// Move constructor.
    sq_gen(sq_gen&& other) noexcept
    :   _sq(std::move(other._sq))
    ,   _ptr_td(std::move(other._ptr_td))
    {}
    ///
    /// @brief Constructs a sequence from an initializer list.
    /// @param init The initializer list containing the residues.
    sq_gen(std::initializer_list<value_type> init)
    :   _sq(init)
    ,   _ptr_td()
    {}

// -- copy assignment operators ------------------------------------------------
    ///
    /// Copy assignment operator.
    sq_gen& operator= (const sq_gen& other)
    {   _sq = other._sq;
        _ptr_td = other._ptr_td ? std::make_unique<Map>(*other._ptr_td) : nullptr;
        return *this;
    }
    ///
    /// Move assignment operator.
    sq_gen& operator= (sq_gen&& other)
    {   _sq = std::move(other._sq);
        _ptr_td = std::move(other._ptr_td);
        return *this;
    }
    ///
    /// Assignment operator from an initializer list.
    sq_gen& operator= (std::initializer_list<value_type> init)
    {   _sq = init;
        return *this;
    }

// -- iterators ----------------------------------------------------------------
    ///
    /// Returns a read/write iterator that points to the first residue in the
    /// @a sq. Iteration is done in ordinary residue order.
    iterator begin() noexcept
    {   return _sq.begin();   }
    ///
    /// Returns a read-only (constant) iterator that points to the first
    /// residue in the @a sq. Iteration is done in ordinary residue order.
    const_iterator begin() const noexcept
    {   return _sq.begin();   }
    ///
    /// Returns a read-only (constant) iterator that points to the first
    /// residue in the @a sq. Iteration is done in ordinary residue order.
    const_iterator cbegin() const noexcept
    {   return _sq.cbegin();   }
    ///
    /// Returns a read/write iterator that points one past the last residue in
    /// the @a sq. Iteration is done in ordinary residue order.
    iterator end() noexcept
    {   return _sq.end();   }
    ///
    /// Returns a read-only (constant) iterator that points one past the last
    /// residue in the @a sq. Iteration is done in ordinary residue order.
    const_iterator end() const noexcept
    {   return _sq.end();   }
    ///
    /// Returns a read-only (constant) iterator that points one past the last
    /// residue in the @a sq. Iteration is done in ordinary residue order.
    const_iterator cend() const noexcept
    {   return _sq.cend();   }
    ///
    /// Returns a read/write iterator that points to the last residue in the
    /// @a sq. Iteration is done in reverse residue order.
    reverse_iterator rbegin() noexcept
    {   return _sq.rbegin();   }
    ///
    /// Returns a read-only (constant) iterator that points to the last residue
    /// in the @a sq. Iteration is done in reverse residue order.
    const_reverse_iterator rbegin() const noexcept
    {   return _sq.rbegin();   }
    ///
    /// Returns a read-only (constant) iterator that points to the last residue
    /// in the @a sq. Iteration is done in reverse residue order.
    const_reverse_iterator crbegin() const noexcept
    {   return _sq.crbegin();   }
    ///
    /// Returns a read/write iterator that points to one before the first
    /// residue in the @a sq. Iteration is done in reverse residue order.
    reverse_iterator rend() noexcept
    {   return _sq.rend();   }
    ///
    /// Returns a read-only (constant) iterator that points to one before the
    /// first residue in the @a sq. Iteration is done in reverse residue order.
    const_reverse_iterator rend() const noexcept
    {   return _sq.rend();   }
    ///
    /// Returns a read-only (constant) iterator that points to one before the
    /// first residue in the @a sq. Iteration is done in reverse residue order.
    const_reverse_iterator crend() const noexcept
    {   return _sq.crend();   }

// -- capacity -----------------------------------------------------------------
    ///
    /// Returns true if the @a sq is empty. (Thus begin() would equal end().)
    bool empty() const noexcept
    {   return (_sq.empty() && (!_ptr_td || _ptr_td->empty()));   }
    ///
    /// Returns the number of residues in the @a sq.
    size_type size() const noexcept
    {   return _sq.size();   }
    ///
    /// Returns the size in memory (in bytes) used by the @a sq including its
    /// tagged data.
    size_type size_in_memory() const noexcept
    {   size_type mem = sizeof(Container) + (_sq.capacity() * sizeof(value_type));
        if (_ptr_td)
        {   mem += sizeof(Map);
            for (const auto& [tag, data] : *_ptr_td)
            {   mem += tag.capacity() * sizeof(char);
            // Note: estimating size of std::any content is not straightforward.
            // Here we just add sizeof of the contained type as a rough estimate.
                mem += data.has_value() ? sizeof(data.type()) : 0;
            }
        }
        return mem;
    }

// -- subscript operator -------------------------------------------------------
    ///
    /// Returns a reference to the residue at position @a pos in the @a sq.
    reference operator[] (size_type pos)
    {   return _sq[pos];   }
    ///
    /// Returns a const reference to the residue at position @a pos in the @a sq.
    const_reference operator[] (size_type pos) const
    {   return _sq[pos];   }

// -- subseq operator ----------------------------------------------------------
    ///
    /// Returns a subsequence starting at position @a pos with length @a count.
    /// If @a count is gnx::sq::npos or exceeds the sequence length from
    /// @a pos, the subsequence extends to the end of the sequence.
#if defined(__CUDACC__) || defined(__HIPCC__)
    sq_gen<Container> operator()
    (   size_type pos
    ,   size_type count = npos
    )   const
    requires
    (   std::is_same_v<Container, thrust::device_vector<value_type>>
    ||  std::is_same_v<Container, thrust::universal_vector<value_type>>
    )
    {   if (pos > _sq.size())
            throw std::out_of_range("gnx::sq: pos > this->size()");
        return sq_gen
        (   _sq.begin() + pos
        ,   (count > _sq.size() - pos) ? _sq.end() : _sq.begin() + pos + count
        );
    }
#endif
    sq_view_gen<Container> operator()
    (   size_type pos
    ,   size_type count = npos
    )   const
    {   sq_view_gen<Container> sv(*this);
        return sv.substr(pos, count);
    }

// -- managing tagged data -----------------------------------------------------
    ///
    /// Returns true if the tagged data with the specified @a tag exists.
    bool has(std::string_view tag) const
    {   return
        (   _ptr_td
        ?   _ptr_td->find(std::string(tag)) == _ptr_td->end()
        ?   false
        :   true
        :   false
        );
    }
    ///
    /// Returns a reference to the tagged data associated with the specified
    /// @a tag. If the tagged data does not exist, a new entry is created.
    std::any& operator[] (const std::string& tag)
    {   if (!_ptr_td) _ptr_td = std::make_unique<Map>();
        return (*_ptr_td)[tag];
    }
    ///
    /// Returns a reference to the tagged data associated with the specified
    /// @a tag. If the tagged data does not exist, a new entry is created.
    std::any& operator[] (std::string&& tag)
    {   if (!_ptr_td) _ptr_td = std::make_unique<Map>();
        return (*_ptr_td)[std::move(tag)];
    }
    ///
    /// Returns a const reference to the tagged data associated with the specified
    /// @a tag. Throws std::out_of_range if the tag does not exist.
    const std::any& operator[] (const std::string& tag) const
    {   if (!_ptr_td || _ptr_td->find(tag) == _ptr_td->end())
            throw std::out_of_range("gnx::sq: tag not found -> " + tag);
        return _ptr_td->at(tag);
    }
#if defined(__CUDACC__) || defined(__HIPCC__)
    value_type* data()
    {   return thrust::raw_pointer_cast(_sq.data());
    }
    const value_type* data() const
    {   return thrust::raw_pointer_cast(_sq.data());
    }
#else
    ///
    /// Returns a reference to the underlying container's data.
    value_type* data() noexcept
    {   return _sq.data();
    }
    ///
    /// Returns a const reference to the underlying container's data.
    const value_type* data() const noexcept
    {   return _sq.data();
    }
#endif

    // -- comparison operators -----------------------------------------------------
    ///
    /// Equality operator with another sq_gen of possibly different Container/Map.
    template<typename Container2, typename Map2>
    bool operator==(const sq_gen<Container2, Map2>& rhs) const
    {   return _sq == rhs._sq;
    }
    ///
    /// Equality operator with std::string_view.
    bool operator==(std::string_view sv) const
    {   if (size() != sv.size())
            return false;
        return std::equal(_sq.begin(), _sq.end(), sv.begin());
    }
    ///
    /// Inequality operator with another sq_gen of possibly different Container/Map.
    template<typename Container2, typename Map2>
    bool operator!=(const sq_gen<Container2, Map2>& rhs) const
    {   return _sq != rhs._sq;
    }

// -- file i/o -----------------------------------------------------------------
    ///
    /// Loads a sequence from a file by its index using the provided
    /// @a read function object
    void load
    (   std::string_view filename
    ,   size_type ndx = 0
    ,   in::fast_aqz<sq_gen> read = in::fast_aqz<sq_gen>()
    )
    {   *this = read(filename, ndx);
    }
    ///
    /// Loads a sequence from a file by its identifier using the provided
    /// @a read function object
    void load
    (   std::string_view filename
    ,   std::string_view id
    ,   in::fast_aqz<sq_gen> read = in::fast_aqz<sq_gen>()
    )
    {   *this = read(filename, id);
    }
    ///
    /// Saves the sequence to a file using the provided write function object
    template<typename WriteFunc>
    void save
    (   std::string_view filename
    ,   WriteFunc write
    )
    {   write(filename, *this);
    }
    ///
    /// Prints the sequence and its tagged data to the output stream @a os.
    void print(std::ostream& os) const
    {   os << std::boolalpha << _sq.size();
#if defined(__CUDACC__) || defined(__HIPCC__)
        if constexpr
        (   std::is_same_v<Container, thrust::device_vector<value_type>>
        )
        {   universal_host_pinned_vector<value_type> uhpv(_sq);
            os.write(thrust::raw_pointer_cast(uhpv.data()), uhpv.size());
        }
        else if constexpr
        (   std::is_same_v<Container, thrust::universal_vector<value_type>>
        )
        {   os.write(thrust::raw_pointer_cast(_sq.data()), _sq.size());
        }
#if defined(__HIPCC__)
        else if constexpr
        (   std::is_same_v<Container, gnx::unified_vector<value_type>>
        )
        {   os.write(thrust::raw_pointer_cast(_sq.data()), _sq.size());
        }
#endif //__HIPCC__
        else
#endif  //__CUDACC__
            os.write(_sq.data(), _sq.size());
        if (_ptr_td)
            for (const auto& [tag, data] : *_ptr_td)
            {   os << std::quoted(tag, '#');
                if
                (   const auto it = td_print_visitor.find(std::type_index(data.type()))
                ;    it != td_print_visitor.cend()
                )
                    it->second(os, data);
                else
                    os << std::quoted("UNREGISTERED TYPE", '|')
                    //    << std::quoted(data.type().name())
                    << "{}";
            }
    }
    ///
    /// Scans the sequence and its tagged data from the input stream @a is.
    void scan(std::istream& is)
    {   size_type n;
        is >> std::boolalpha >> n;
        _sq.resize(n);
#if defined(__CUDACC__) || defined(__HIPCC__)
        if constexpr
        (   std::is_same_v<Container, thrust::device_vector<value_type>>
        )
        {   universal_host_pinned_vector<value_type> uhpv(n);
            is.read(thrust::raw_pointer_cast(uhpv.data()), n);
            thrust::copy(uhpv.begin(), uhpv.end(), _sq.begin());
        }
        else if constexpr
        (   std::is_same_v<Container, thrust::universal_vector<value_type>>
        )
        {   is.read(thrust::raw_pointer_cast(_sq.data()), n);
        }
#if defined(__HIPCC__)
        else if constexpr
        (   std::is_same_v<Container, gnx::unified_vector<value_type>>
        )
        {   is.read(thrust::raw_pointer_cast(_sq.data()), n);
        }
#endif //__HIPCC__
        else
#endif  //__CUDACC__
            is.read(_sq.data(), n);
        if (is.peek() == '#' && !_ptr_td)
            _ptr_td = std::make_unique<Map>();
        while (is.peek() == '#')
        {   std::string tag, type;
            std::any a;
            is >> std::quoted(tag, '#')
               >> std::quoted(type, '|');
            if
            (   const auto it = td_scan_visitor.find(type)
            ;    it != td_scan_visitor.cend()
            )
                it->second(is, a);
            else
                throw std::runtime_error
                (   "gnx::sq: unregistered type -> "
                +   type
                );
            (*_ptr_td)[tag] = a;
        }
    }
};

// -- comparison operators (external) -----------------------------------------
    ///
    /// Symmetric operator for "literal" == sq_gen
    template<typename Container>
    bool operator==(std::string_view lhs, const sq_gen<Container>& rhs)
    {   return rhs == lhs; 
    }
    ///
    /// Compare sq_gen with sq_view_gen
    template<typename Container1, typename Map1, typename Container2>
    bool operator==(const sq_gen<Container1, Map1>& lhs, const sq_view_gen<Container2>& rhs)
    {   if (lhs.size() != rhs.size()) return false;
        return std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }
    template<typename Container1, typename Map1, typename Container2>
    bool operator==(const sq_view_gen<Container2>& lhs, const sq_gen<Container1, Map1>& rhs)
    {   return rhs == lhs;
    }
    template<typename Container1, typename Map1, typename Container2>
    bool operator!=(const sq_gen<Container1, Map1>& lhs, const sq_view_gen<Container2>& rhs)
    {   return ! (lhs == rhs);
    }
    template<typename Container1, typename Map1, typename Container2>
    bool operator!=(const sq_view_gen<Container2>& lhs, const sq_gen<Container1, Map1>& rhs)
    {   return ! (lhs == rhs);
    }
    ///
    /// Specialization for C-string literal comparisons (const char*)
    template<typename Container>
    bool operator==(const char* lhs, const sq_gen<Container>& rhs)
    {   return std::string_view(lhs) == rhs;
    }
    template<typename Container>
    bool operator==(const sq_gen<Container>& lhs, const char* rhs)
    {   return lhs == std::string_view(rhs);
    }

// -- i/o stream operators -----------------------------------------------------
    ///
    /// Output stream operator for sq_gen.
    template<typename T>
    std::ostream& operator<< (std::ostream& os, const sq_gen<T>& s)
    {   s.print(os);
        return os;
    }
    ///
    /// Input stream operator for sq_gen.
    template<typename T>
    std::istream& operator>> (std::istream& is, sq_gen<T>& s)
    {   s.scan(is);
        return is;
    }
    ///
    /// A sequence of @a char
    using sq = sq_gen<std::vector<char>>;

}   // end gnx namespace

// -- string literal operator --------------------------------------------------

    gnx::sq operator""_sq (const char* str, std::size_t len)
    {   return gnx::sq(str);   }
