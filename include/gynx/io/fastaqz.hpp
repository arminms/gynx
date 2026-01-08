//
// Copyright (c) 2023 Armin Sobhani
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
#ifndef _GYNX_IO_FASTAQZ_HPP_
#define _GYNX_IO_FASTAQZ_HPP_

#include <string>
#include <string_view>
#include <stdexcept>

#include <zlib.h>
#include <gynx/io/kseq.h>
#include <gynx/memory.hpp>

namespace gynx {

/// Input file formats
namespace in {

KSEQ_INIT(gzFile, gzread)

/// @brief A function object for reading FASTA/FASTQ files (possibly compressed
/// with gzip) and returning a @a Sequence type.
/// @tparam Sequence
template <class Sequence>
struct fast_aqz
{   Sequence operator() (std::string_view filename, size_t ndx)
    {   gzFile fp = filename == "-"
        ?   gzdopen(fileno(stdin), "r")
        :   gzopen(std::string(filename).c_str(), "r");
        if (nullptr == fp)
            throw std::runtime_error
                ("gynx::fast_aqz: could not open file -> " + std::string(filename));
        kseq_t* seq = kseq_init(fp);
        size_t count = 0;
        int r{};
        while ((r = kseq_read(seq)) >= 0)
            if (ndx == count++)
                break;
        Sequence s = (r > 0) ? Sequence(seq->seq.s) : Sequence();
        if (r > 0)
        {   s["_id"] = std::string(seq->name.s);
            if (seq->qual.l)
                s["_qs"] = std::string(seq->qual.s);
            if (seq->comment.l)
                s["_desc"] = std::string(seq->comment.s);
        }
        kseq_destroy(seq);
        gzclose(fp);
        if (-2 == r)
            throw std::runtime_error
            (   "gynx::fast_aqz: truncated quality string in file -> "
            +   std::string(filename)
            );
        if (-3 == r)
            throw std::runtime_error
            (   "gynx::fast_aqz: error reading file -> "
            +   std::string(filename)
            ); 
        return s;
    }
    Sequence operator() (std::string_view filename, std::string_view id)
    {   gzFile fp = filename == "-"
        ?   gzdopen(fileno(stdin), "r")
        :   gzopen(std::string(filename).c_str(), "r");
        if (nullptr == fp)
            throw std::runtime_error
            (   "gynx::fast_aqz: could not open file -> "
            +   std::string(filename)
            );
        kseq_t* seq = kseq_init(fp);
        int r{};
        while ((r = kseq_read(seq)) >= 0)
        {   std::string_view name(seq->name.s);
            if (name == id)
                break;
        }
        Sequence s = (r > 0) ? Sequence(seq->seq.s) : Sequence();
        if (r > 0)
        {   s["_id"] = std::string(seq->name.s);
            if (seq->qual.l)
                s["_qs"] = std::string(seq->qual.s);
            if (seq->comment.l)
                s["_desc"] = std::string(seq->comment.s);
        }
        kseq_destroy(seq);
        gzclose(fp);
        if (-2 == r)
            throw std::runtime_error
            (   "gynx::fast_aqz: truncated quality string in file -> "
            +   std::string(filename)
            );
        if (-3 == r)
            throw std::runtime_error
            (   "gynx::fast_aqz: error reading file -> "
            +   std::string(filename)
            );
        return s;
    }
};

}   // end gynx::in namespace

/// Output file formats

namespace out {

/// @brief A function object for writing sequences to FASTA files.
struct fasta
{   fasta(std::size_t line_width = 80)
    :   _line_width(line_width)
    {}
    template <class Sequence>
    int operator()
    (   std::string_view filename
    ,   const Sequence& seq
    )
    {   FILE* fp = filename == "-"
        ?   stdout
        :   fopen(std::string(filename).c_str(), "w");
        if (nullptr == fp)
            throw std::runtime_error
            (   "gynx::fasta: could not open file -> "
            +   std::string(filename)
            );
        std::string id = seq.has("_id")
        ?   std::any_cast<std::string>(seq["_id"])
        :   "seq";
        std::string desc = seq.has("_desc")
        ?   " " + std::any_cast<std::string>(seq["_desc"])
        :   " generated by Gynx";
        fwrite
        (   (">" + id + desc + "\n").c_str()
        ,   sizeof(typename Sequence::value_type)
        ,   id.size() + desc.size() + 2
        ,   fp
        );
        const typename Sequence::value_type* data = nullptr;
#if defined(__CUDACC__) // handle device_vector
        universal_host_pinned_vector<typename Sequence::value_type> h_seq;
        if constexpr
        (   std::is_same_v<typename Sequence::container_type, thrust::device_vector<typename Sequence::value_type>>
        )
        {   h_seq.resize(std::size(seq));
            thrust::copy(seq.begin(), seq.end(), h_seq.begin());
            data = thrust::raw_pointer_cast(h_seq.data());
        }
        else
        {   data = seq.data();
        }
#else
        data = seq.data();
#endif //__CUDACC__
        if (_line_width)
        {   for
            (   typename Sequence::size_type i = 0
            ;   i < std::size(seq)
            ;   i += _line_width
            )
            {   std::string_view line
                (   data + i
                ,   std::min(_line_width, std::size(seq) - i)
                );
                fwrite
                (   line.data()
                ,   sizeof(typename Sequence::value_type)
                ,   line.size()
                ,   fp
                );
                fwrite("\n", 1, 1, fp);
            }
        }
        else
        {   fwrite
            (   data
            ,   sizeof(typename Sequence::value_type)
            ,   std::size(seq)
            ,   fp
            );
            fwrite("\n", 1, 1, fp);
        }
        fclose(fp);
        return 0;
    }

private:
    std::size_t _line_width;
};

/// @brief A function object for writing sequences to FASTA files compressed
/// with gzip.
struct fasta_gz
{   fasta_gz(std::size_t line_width = 80)
    :   _line_width(line_width)
    {}
    template <class Sequence>
    int operator()
    (   std::string_view filename
    ,   const Sequence& seq
    ,   typename Sequence::size_type line_width = 80
    )
    {   gzFile fp = filename == "-"
        ?   gzdopen(fileno(stdout), "w")
        :   gzopen(std::string(filename).c_str(), "w");
        if (nullptr == fp)
            throw std::runtime_error
            (   "gynx::fasta_gz: could not open file -> "
            +   std::string(filename)
            );
        std::string id = seq.has("_id")
        ?   std::any_cast<std::string>(seq["_id"])
        :   "seq";
        std::string desc = seq.has("_desc")
        ?   " " + std::any_cast<std::string>(seq["_desc"])
        :   " generated by Gynx";
        std::string header = ">" + id + desc + "\n";
        gzwrite(fp, header.c_str(), header.size());
        const typename Sequence::value_type* data = nullptr;
#if defined(__CUDACC__) // handle device_vector
        universal_host_pinned_vector<typename Sequence::value_type> h_seq;
        if constexpr
        (   std::is_same_v<typename Sequence::container_type, thrust::device_vector<typename Sequence::value_type>>
        )
        {   h_seq.resize(std::size(seq));
            thrust::copy(seq.begin(), seq.end(), h_seq.begin());
            data = thrust::raw_pointer_cast(h_seq.data());
        }
        else
        {   data = seq.data();
        }
#else
        data = seq.data();
#endif //__CUDACC__
        if (_line_width)
        {   for
            (   typename Sequence::size_type i = 0
            ;   i < std::size(seq)
            ;   i += _line_width
            )
            {   std::string_view line
                (   data + i
                ,   std::min(_line_width, std::size(seq) - i)
                );
                gzwrite(fp,  line.data(), line.size());
                gzwrite(fp, "\n", 1);
            }
        }
        else
        {   gzwrite(fp, data, std::size(seq));
            gzwrite(fp, "\n", 1);
        }   
        gzclose(fp);
        return 0;
    }

private:
    std::size_t _line_width;
};

/// @brief A function object for writing sequences to FASTQ files.
struct fastq
{   fastq(std::size_t line_width = 0)
    :   _line_width(line_width)
    {}
    template <class Sequence>
    int operator()
    (   std::string_view filename
    ,   const Sequence& seq
    ,   typename Sequence::size_type line_width = 80
    )
    {   FILE* fp = filename == "-"
        ?   stdout
        :   fopen(std::string(filename).c_str(), "w");
        if (nullptr == fp)
            throw std::runtime_error
            (   "gynx::fastq: could not open file -> "
            +   std::string(filename)
            );
        std::string id = seq.has("_id")
        ?   std::any_cast<std::string>(seq["_id"])
        :   "seq";
        std::string desc = seq.has("_desc")
        ?   " " + std::any_cast<std::string>(seq["_desc"])
        :   " generated by Gynx";
        fwrite
        (   ("@" + id + desc + "\n").c_str()
        ,   sizeof(typename Sequence::value_type)
        ,   id.size() + desc.size() + 2
        ,   fp
        );
        const typename Sequence::value_type* data = nullptr;
#if defined(__CUDACC__) // handle device_vector
        universal_host_pinned_vector<typename Sequence::value_type> h_seq;
        if constexpr
        (   std::is_same_v<typename Sequence::container_type, thrust::device_vector<typename Sequence::value_type>>
        )
        {   h_seq.resize(std::size(seq));
            thrust::copy(seq.begin(), seq.end(), h_seq.begin());
            data = thrust::raw_pointer_cast(h_seq.data());
        }
        else
        {   data = seq.data();
        }
#else
        data = seq.data();
#endif //__CUDACC__
        if (_line_width)
        {   for
            (   typename Sequence::size_type i = 0
            ;   i < std::size(seq)
            ;   i += _line_width
            )
            {   std::string_view line
            (   data + i
            ,   std::min(_line_width, std::size(seq) - i)
            );
            fwrite
            (   line.data()
            ,   sizeof(typename Sequence::value_type)
            ,   line.size()
            ,   fp
            );
            fwrite("\n", 1, 1, fp);
        }
        }
        else
        {   fwrite
            (   data
            ,   sizeof(typename Sequence::value_type)
            ,   std::size(seq)
            ,   fp
            );
            fwrite("\n", 1, 1, fp);
        }
        fwrite("+\n", 1, 2, fp);
        std::string qs = seq.has("_qs")
        ?   std::any_cast<std::string>(seq["_qs"])
        :   std::string(std::size(seq), 'I'); // dummy quality string
        if (_line_width)
        {   for
            (   typename Sequence::size_type i = 0
            ;   i < qs.size()
            ;   i += _line_width
            )
            {   std::string_view line
                (   qs.data() + i
                ,   std::min(_line_width, qs.size() - i)
                );
                fwrite
                (   line.data()
                ,   sizeof(std::string_view::value_type)
                ,   line.size()
                ,   fp
                );
                fwrite("\n", 1, 1, fp);
            }
        }
        else
        {   fwrite
            (   qs.data()
            ,   sizeof(std::string_view::value_type)
            ,   qs.size()
            ,   fp
            );
            fwrite("\n", 1, 1, fp);
        }
        fclose(fp);
        return 0;
    }

private:
    std::size_t _line_width;
};

/// @brief A function object for writing sequences to FASTQ files compressed
/// with gzip.
struct fastq_gz
{   fastq_gz(std::size_t line_width = 0)
    :   _line_width(line_width)
    {}
    template <class Sequence>
    int operator()
    (   std::string_view filename
    ,   const Sequence& seq
    ,   typename Sequence::size_type line_width = 80
    )
    {   gzFile fp = filename == "-"
        ?   gzdopen(fileno(stdout), "w")
        :   gzopen(std::string(filename).c_str(), "w");
        if (nullptr == fp)
            throw std::runtime_error
            (   "gynx::fastq_gz: could not open file -> "
            +   std::string(filename)
            );
        std::string id = seq.has("_id")
        ?   std::any_cast<std::string>(seq["_id"])
        :   "seq";
        std::string desc = seq.has("_desc")
        ?   " " + std::any_cast<std::string>(seq["_desc"])
        :   " generated by Gynx";
        std::string header = "@" + id + desc + "\n";
        gzwrite(fp, header.c_str(), header.size());
        const typename Sequence::value_type* data = nullptr;
#if defined(__CUDACC__) // handle device_vector
        universal_host_pinned_vector<typename Sequence::value_type> h_seq;
        if constexpr
        (   std::is_same_v<typename Sequence::container_type, thrust::device_vector<typename Sequence::value_type>>
        )
        {   h_seq.resize(std::size(seq));
            thrust::copy(seq.begin(), seq.end(), h_seq.begin());
            data = thrust::raw_pointer_cast(h_seq.data());
        }
        else
        {   data = seq.data();
        }
#else
        data = seq.data();
#endif //__CUDACC__
        if (_line_width)
        {   for
            (   typename Sequence::size_type i = 0
            ;   i < std::size(seq)
            ;   i += _line_width
            )
            {   std::string_view line
                (   data + i
                ,   std::min(_line_width, std::size(seq) - i)
                );
                gzwrite(fp, line.data(), line.size());
                gzwrite(fp, "\n", 1);
            }
        }
        else
        {   gzwrite(fp, data,  std::size(seq));
            gzwrite(fp, "\n", 1);
        }
        gzwrite(fp, "+\n", 2);
        std::string qs = seq.has("_qs")
        ?   std::any_cast<std::string>(seq["_qs"])
        :   std::string(std::size(seq), 'I'); // dummy quality string
        if (_line_width)
        {   for
            (   typename Sequence::size_type i = 0
            ;   i < qs.size()
            ;   i += _line_width
            )
            {   std::string_view line
                (   qs.data() + i
                ,   std::min(_line_width, qs.size() - i)
                );
                gzwrite(fp, line.data(), line.size());
                gzwrite(fp, "\n", 1);
            }
        }
        else
        {   gzwrite(fp, qs.data(), qs.size());
            gzwrite(fp, "\n", 1);
        }
        gzclose(fp);
        return 0;
    }

private:
    std::size_t _line_width;
};

}   // end gynx::out namespace
}   // end gynx namespace

#endif  //_GYNX_IO_FASTAQZ_HPP_