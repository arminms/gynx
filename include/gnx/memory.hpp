// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Armin Sobhani
//
#pragma once

#include <cstdlib>
#include <memory>
#include <type_traits>

#if defined(__CUDACC__)
    #include <thrust/universal_vector.h>
    #include <thrust/system/cuda/memory_resource.h>
    #include <thrust/device_allocator.h>
    #include <thrust/mr/allocator.h>
    #include <thrust/system/cpp/memory.h>
#elif defined(__HIPCC__)
    #include <thrust/universal_vector.h>
    #include <thrust/device_vector.h>
    #include <thrust/device_malloc_allocator.h>
    #include <hip/hip_runtime.h>
#endif

namespace gnx
{

#if defined(__HIPCC__)
    // custom allocator for host pinned memory on ROCm platform
    template<class T>
    struct host_pinned_allocator : thrust::device_malloc_allocator<T>
    {   typedef thrust::device_ptr<T> pointer;

        inline pointer allocate(size_t n)
        {   T* value = 0;
            // hipHostMalloc allocates pinned host memory accessible the GPU device
            hipError_t err = hipHostMalloc(&value, n * sizeof(T));

            if (err != hipSuccess) throw thrust::system_error(err, thrust::hip_category());
            return pointer(value);
        }

        inline void deallocate(pointer ptr, size_t n)
        {   hipError_t err = hipFree(ptr.get());
            if (err != hipSuccess) throw thrust::system_error(err, thrust::hip_category());
        }
    };

// Define a host pinned vector type
template<class T>
    using host_pinned_vector = thrust::host_vector<T, host_pinned_allocator<T>>;

    // custom allocator for unified physical memory (e.g. MI300A) on ROCm platform
    template<class T>
    struct unified_allocator : thrust::device_malloc_allocator<T>
    {   typedef thrust::device_ptr<T> pointer;

        inline pointer allocate(size_t n)
        {   T* value = 0;
            // hipMalloc creates a unified buffer accessible by CPU & GPU
            hipError_t err = hipMalloc(&value, n * sizeof(T));
            // hipError_t err = hipMallocManaged(&value, n * sizeof(T));
            // hipError_t err = hipHostMalloc(&value, n * sizeof(T));
            // hipError_t err = hipHostAlloc(&value, n * sizeof(T));

            if (err != hipSuccess) throw thrust::system_error(err, thrust::hip_category());
            return pointer(value);
        }

        inline void deallocate(pointer ptr, size_t n)
        {   hipError_t err = hipFree(ptr.get());
            if (err != hipSuccess) throw thrust::system_error(err, thrust::hip_category());
        }
    };

// Define a unified vector type
template<class T>
    using unified_vector = thrust::host_vector<T, unified_allocator<T>>;
#endif

#if defined(__CUDACC__)
    template <typename T>
    using pinned_host_allocator = thrust::mr::stateless_resource_allocator
    <   T
    ,   thrust::cuda::universal_host_pinned_memory_resource
    >;

    template <typename T>
    using universal_host_pinned_vector = thrust::universal_vector<T, pinned_host_allocator<T>>;
#elif defined(__HIPCC__)
    template <typename T>
    using universal_host_pinned_vector = host_pinned_vector<T>;
#endif // __CUDACC__


// -- default_init_allocator ---------------------------------------------------

    // allocator adaptor that interposes 'construct' calls
    // to convert value initialization into default initialization
    // by Casey Carter (@codercasey)
    // This is useful when you want to use a fundamental type with std::vector,
    // but you don't want the elements to be initialized by default.
    // usage: std::vector<int, default_init_allocator<int>> v(100);
    // The elements of v will be default-initialized, not value-initialized.
    template<typename T, typename Alloc = std::allocator<T>>
    class default_init_allocator : public Alloc
    {   using a_t = std::allocator_traits<Alloc>;
    public:
        // obtain alloc<U> where U ≠ T
        template<typename U>
        struct rebind
        {   using other = default_init_allocator<U,
            typename a_t::template rebind_alloc<U> >;
        };
        // make inherited ctors visible
        using Alloc::Alloc;
        // default-construct objects
        template<typename U>
        void construct(U* ptr)
            noexcept(std::is_nothrow_default_constructible<U>::value)
        {   // 'placement new':
            ::new(static_cast<void*>(ptr)) U;
        }
        // construct with ctor arguments
        template<typename U, typename... Args>
        void construct(U* ptr, Args&&... args)
        {   a_t::construct
            (   static_cast<Alloc&>(*this)
            ,   ptr
            ,   std::forward<Args>(args)...
            );
        }
    };

// -- no_init ------------------------------------------------------------------

    // no_init<T> is a wrapper around T that prevents automatic initialization
    // of the wrapped object. This is useful when you want to use a fundamental
    // type with std::vector, but you don't want the elements to be initialized
    // by default.
    // usage: vector<no_init<int>> v(100);
    template<typename T>
    class no_init
    {   static_assert(
        std::is_fundamental<T>::value,
        "should be a fundamental type");
    public: 
        // constructor without initialization
        // GNX_DEVICE_CODE
        no_init () noexcept {}
        // implicit conversion T → no_init<T>
        // GNX_DEVICE_CODE
        constexpr  no_init (T value) noexcept: v_{value} {}
        // implicit conversion no_init<T> → T
        // GNX_DEVICE_CODE
        constexpr  operator T () const noexcept { return v_; }
        private:
        T v_;
    };

// -- aligned_allocator ------------------------------------------------------------------

    enum class Alignment : size_t
    {   Normal = sizeof(void*),
        SSE    = 16,
        AVX    = 32,
        AVX512 = 64
    };

    namespace detail
    {   void* allocate_aligned_memory(size_t align, size_t size);
        void deallocate_aligned_memory(void* ptr) noexcept;
    }

    template <typename T, Alignment Align = Alignment::AVX>
    class aligned_allocator;

    // partial specialization for void
    // this is required for std::allocator_traits
    // to work correctly
    // see: https://en.cppreference.com/w/cpp/memory/allocator_traits
    template <Alignment Align>
    struct aligned_allocator<void, Align>
    {   typedef void*             pointer;
        typedef const void*       const_pointer;
        typedef void              value_type;

        template <class U> struct rebind
        {   typedef aligned_allocator<U, Align> other;   };
    };

    // primary template
    // this is the default allocator
    // it is used for all types except void
    // and it is used when the alignment is not specified
    // in the template arguments
    template <typename T, Alignment Align>
    struct aligned_allocator
    {   typedef T         value_type;
        typedef T*        pointer;
        typedef const T*  const_pointer;
        typedef T&        reference;
        typedef const T&  const_reference;
        typedef size_t    size_type;
        typedef ptrdiff_t difference_type;

        typedef std::true_type propagate_on_container_move_assignment;

        template <class U>
        struct rebind { typedef aligned_allocator<U, Align> other; };

        aligned_allocator() noexcept
        {}

        template <class U>
        aligned_allocator(const aligned_allocator<U, Align>&) noexcept
        {}

        size_type
        max_size() const noexcept
        { return (size_type(~0) - size_type(Align)) / sizeof(T); }

        pointer
        address(reference x) const noexcept
        { return std::addressof(x); }

        const_pointer
        address(const_reference x) const noexcept
        { return std::addressof(x); }

        pointer
        allocate(size_type n, typename aligned_allocator<void, Align>::const_pointer = 0)
        {   const size_type alignment = static_cast<size_type>( Align );
            void* ptr = detail::allocate_aligned_memory(alignment , n * sizeof(T));
            if (ptr == nullptr)
                throw std::bad_alloc();
            return reinterpret_cast<pointer>(ptr);
        }

        void
        deallocate(pointer p, size_type) noexcept
        {   return detail::deallocate_aligned_memory(p);   }

        template <class U, class ...Args>
        void
        construct(U* p, Args&&... args)
        {  ::new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);  }

        void
        destroy(pointer p)
        {   p->~T();   }
    };

    // partial specialization for const T
    // this is required for std::allocator_traits
    // to work correctly
    template <typename T, Alignment Align>
    struct aligned_allocator<const T, Align>
    {   typedef T         value_type;
        typedef const T*  pointer;
        typedef const T*  const_pointer;
        typedef const T&  reference;
        typedef const T&  const_reference;
        typedef size_t    size_type;
        typedef ptrdiff_t difference_type;

        typedef std::true_type propagate_on_container_move_assignment;

        template <class U>
        struct rebind { typedef aligned_allocator<U, Align> other; };

        aligned_allocator() noexcept
        {}

        template <class U>
        aligned_allocator(const aligned_allocator<U, Align>&) noexcept
        {}

        size_type
        max_size() const noexcept
        {   return (size_type(~0) - size_type(Align)) / sizeof(T);   }

        const_pointer
        address(const_reference x) const noexcept
        {   return std::addressof(x);   }

        pointer
        allocate(size_type n, typename aligned_allocator<void, Align>::const_pointer = 0)
        {   const size_type alignment = static_cast<size_type>( Align );
            void* ptr = detail::allocate_aligned_memory(alignment , n * sizeof(T));
            if (ptr == nullptr)
                throw std::bad_alloc();
            return reinterpret_cast<pointer>(ptr);
        }

        void
        deallocate(pointer p, size_type) noexcept
        {   return detail::deallocate_aligned_memory(p);   }

        template <class U, class ...Args>
        void
        construct(U* p, Args&&... args)
        {  ::new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);  }

        void
        destroy(pointer p)
        {   p->~T();   }
    };

    // comparison operators
    // these are required for std::allocator_traits
    // to work correctly
    template <typename T, Alignment TAlign, typename U, Alignment UAlign>
    inline
    bool
    operator== (const aligned_allocator<T,TAlign>&, const aligned_allocator<U, UAlign>&) noexcept
    {   return TAlign == UAlign;   }

    // comparison operators
    // these are required for std::allocator_traits
    // to work correctly
    template <typename T, Alignment TAlign, typename U, Alignment UAlign>
    inline
    bool
    operator!= (const aligned_allocator<T,TAlign>&, const aligned_allocator<U, UAlign>&) noexcept
    {   return TAlign != UAlign;   }

    // allocate aligned memory
    // this function is used by the aligned_allocator
    // to allocate memory with a specified alignment
    // the memory is deallocated with detail::deallocate_aligned_memory
    //
    // must be changed to use the C++17 std::aligned_alloc or std::align_val_t
    // return new (std::align_val_t(16)) int[40]; // 128-bit alignment
    void*
    detail::allocate_aligned_memory(size_t align, size_t size)
    {   // assert(align >= sizeof(void*));
        // assert(nail::is_power_of_two(align));
        if (0 == size)
            return nullptr;
#if defined(_WIN32)
        void* ptr = _aligned_malloc(size, align);
#else
        void* ptr = std::aligned_alloc(align, size);
#endif // _WIN32
        return ptr;
    }

    // deallocate aligned memory
    // this function is used by the aligned_allocator
    // to deallocate memory that was allocated with
    // detail::allocate_aligned_memory
    void
    detail::deallocate_aligned_memory(void *ptr) noexcept
#if defined(_WIN32)
    {   return _aligned_free(ptr);   }
#else
    {   return free(ptr);   }
#endif // _WIN32

}  // namespace gnx
