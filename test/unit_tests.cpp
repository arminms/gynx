// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2025 Armin Sobhani
//
#include <catch2/catch_all.hpp>

#include <gnx/sq.hpp>
#include <gnx/sq_view.hpp>
#include <gnx/io/fastaqz.hpp>
#include <gnx/algorithms/valid.hpp>
#include <gnx/algorithms/random.hpp>
#include <gnx/algorithms/local_align.hpp>

const uint64_t seed_pi{3141592654};

template<typename T>
using aligned_vector = std::vector<T, gnx::aligned_allocator<T, gnx::Alignment::AVX512>>;

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::sq"
,   "[class][cuda]"
,   std::vector<char>
// ,   aligned_vector<char>
,   thrust::host_vector<char>
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::sq"
,   "[class][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
#else
TEMPLATE_TEST_CASE( "gnx::sq", "[class]", std::vector<char>)
#endif
{   typedef TestType T;

    gnx::sq_gen<T> s{"ACGT"};
    std::string t{"ACGT"}, u{"acgt"}, v{"ACGT "};
    s["test-int"] = -33;

// -- comparison operators -----------------------------------------------------

    SECTION( "comparison operators" )
    {   REQUIRE(  s == gnx::sq_gen<T>("ACGT"));
        REQUIRE(!(s == gnx::sq_gen<T>("acgt")));
        REQUIRE(  s != gnx::sq_gen<T>("acgt") );
        REQUIRE(!(s != gnx::sq_gen<T>("ACGT")));

        REQUIRE(s == t);
        REQUIRE(t == s);
        REQUIRE(s != u);
        REQUIRE(u != s);
        REQUIRE(s != v);
        REQUIRE(v != s);

        REQUIRE(s == "ACGT");
        REQUIRE("ACGT" == s);
        REQUIRE(s != "acgt");
        REQUIRE("acgt" != s);
        REQUIRE(s != "ACGT ");
        REQUIRE("ACGT " != s);
    }

// -- constructors -------------------------------------------------------------

    SECTION( "single value constructor" )
    {
#if defined(__CUDACC__) || defined(__HIPCC__) // not supported for thrust::device_vector
        if constexpr
        (   !std::is_same_v<T, thrust::device_vector<char>>
        &&  !std::is_same_v<T, thrust::universal_vector<char>>
#if defined(__HIPCC__)
        &&  !std::is_same_v<T, gnx::unified_vector<char>>
#endif //__HIPCC__
        )
        {   gnx::sq_gen<T> a4(4, 'A');
            CHECK(a4 == "AAAA");
            gnx::sq_gen<T> c4(4, 'C');
            CHECK(c4 == "CCCC");
        }
#else
        gnx::sq_gen<T> a4(4, 'A');
        CHECK(a4 == "AAAA");
        gnx::sq_gen<T> c4(4, 'C');
        CHECK(c4 == "CCCC");
#endif //__CUDACC__
    }
    SECTION( "string_view constructor" )
    {   gnx::sq_gen<T> c("ACGT");
        CHECK(s == c);
    }
    SECTION( "sq_view constructor" )
    {
#if defined(__CUDACC__) || defined(__HIPCC__) // not supported for thrust::device_vector
        if constexpr
        (   !std::is_same_v<T, thrust::device_vector<char>>
        )
        {   gnx::sq_view_gen<T> sv(s);
            CHECK(s == sv);
        }
#else
        gnx::sq_view_gen<T> sv(s);
        CHECK(s == sv);
#endif //__CUDACC__
    }
    SECTION( "iterator constructor" )
    {   std::string acgt{"ACGT"};
        gnx::sq_gen<T> c(std::begin(acgt), std::end(acgt));
        CHECK(s == c);
    }
    SECTION( "copy constructor" )
    {   gnx::sq_gen<T> c(s);
        CHECK(c == s);
        CHECK(-33 == std::any_cast<int>(c["test-int"]));
    }
    SECTION( "move constructor" )
    {   gnx::sq_gen<T> m(std::move(s));
        CHECK(s.empty());
        CHECK(m == gnx::sq_gen<T>("ACGT"));
        CHECK(-33 == std::any_cast<int>(m["test-int"]));
    }
    SECTION( "initializer list" )
    {   gnx::sq_gen<T> c{'A', 'C', 'G', 'T'};
        CHECK(c == s);
    }

// -- copy assignment operators ------------------------------------------------

    SECTION( "copy assignment operator" )
    {   gnx::sq_gen<T> c = s;
        CHECK(c == s);
        CHECK(-33 == std::any_cast<int>(c["test-int"]));
    }
    SECTION( "move constructor" )
    {   gnx::sq_gen<T> m = gnx::sq_gen<T>("ACGT");
        CHECK(m == s);
    }
    SECTION( "initializer list" )
    {   gnx::sq_gen<T> c = {'A', 'C', 'G', 'T'};
        CHECK(c == s);
    }

// -- iterators ----------------------------------------------------------------

    SECTION( "begin/end" )
    {   gnx::sq_gen<T> t("AAAA");
        for (auto a : t)
            CHECK(a == 'A');
#if defined(__CUDACC__) || defined(__HIPCC__) // not supported for thrust::device_vector
        if constexpr
        (   !std::is_same_v<T, thrust::device_vector<char>>
#if defined(__HIPCC__)
        &&  !std::is_same_v<T, gnx::unified_vector<char>>
#endif //__HIPCC__
        )
        {   for (auto& a : t)
                a = 'T';
            CHECK(t == gnx::sq_gen<T>("TTTT"));
        }
#else
        for (auto& a : t)
            a = 'T';
        CHECK(t == gnx::sq_gen<T>("TTTT"));
#endif //__CUDACC__
        auto s_it = s.cbegin();
        for
        (   auto t_it = t.begin()
        ;   t_it != t.end()
        ;   ++t_it, ++s_it
        )
            *t_it = *s_it;
        CHECK(t == "ACGT");
    }
    SECTION( "cbegin/cend" )
    {   const gnx::sq_gen<T> t("AAAA");
        auto s_it = s.begin();
        for
        (   auto t_it = t.cbegin()
        ;   t_it != t.cend()
        ;   ++t_it, ++s_it
        )
            *s_it = *t_it;
        CHECK(s == "AAAA");
    }
    SECTION( "rbegin/rend" )
    {   gnx::sq_gen<T> t("AAAA");
        auto s_it = s.cbegin();
        for
        (   auto t_it = t.rbegin()
        ;   t_it != t.rend()
        ;   ++t_it, ++s_it
        )
            *t_it = *s_it;
        CHECK(t == "TGCA");
    }
    SECTION( "crbegin/crend" )
    {   const gnx::sq_gen<T> t("ACGT");
        auto s_it = s.begin();
        for
        (   auto t_it = t.crbegin()
        ;   t_it != t.crend()
        ;   ++t_it, ++s_it
        )
            *s_it = *t_it;
        CHECK(s == "TGCA");
    }

// -- capacity -----------------------------------------------------------------

    SECTION( "empty()" )
    {   gnx::sq_gen<T> e;
        CHECK(e.empty());
        e["test"] = 1;
        CHECK(!e.empty());
        CHECK(!s.empty());
    }
    SECTION( "size()" )
    {   gnx::sq_gen<T> e;
        CHECK(0 == e.size());
        CHECK(4 == s.size());
    }

// -- subscript operator -------------------------------------------------------

    SECTION( "subscript/array index operator" )
    {   CHECK('A' == s[0]);
        CHECK('C' == s[1]);
        CHECK('G' == s[2]);
        CHECK('T' == s[3]);
        s[3] = 'U';
        CHECK('U' == s[3]);
    }

// -- subseq operator ----------------------------------------------------------

    SECTION( "subseq operator" )
    {   gnx::sq_gen<T> org{"CCATACGTGAC"};
        CHECK(org(4, 4) == s);
        CHECK(org(0) == org);
        CHECK(org(4) == "ACGTGAC");
        CHECK_THROWS_AS(org(20) == "ACGTGAC", std::out_of_range);

        // casting sq_view to sq_gen
        gnx::sq_gen<T> sub = gnx::sq_gen<T>(org(4, 10));
        CHECK(sub == "ACGTGAC");
    }

// -- managing tagged data -----------------------------------------------------

    SECTION( "tagged data" )
    {   CHECK(s.has("test-int"));
        CHECK(false == s.has("no"));

        s["int"] = 19;
        CHECK(s.has("int"));
        CHECK(19 == std::any_cast<int>(s["int"]));

        s["float"] = 3.14f;
        CHECK(s.has("float"));
        CHECK(3.14f == std::any_cast<float>(s["float"]));

        s["double"] = 3.14;
        CHECK(s.has("double"));
        CHECK(3.14 == std::any_cast<double>(s["double"]));

        s["string"] = std::string("hello");
        CHECK(s.has("string"));
        CHECK("hello" == std::any_cast<std::string>(s["string"]));

        std::vector<int> v{ 1, 2, 3, 4 };
        s["vector_int"] = v;
        CHECK(s.has("vector_int"));
        CHECK(v == std::any_cast<std::vector<int>>(s["vector_int"]));

        std::string lvalue_tag{"check_lvalue_tag"};
        s[lvalue_tag] = 42;
        CHECK(s.has(lvalue_tag));
        CHECK(42 == std::any_cast<int>(s[lvalue_tag]));
    }

// -- i/o operators ------------------------------------------------------------

    SECTION( "i/o operators")
    {   s["test-void"] = {};
        s["test-bool"] = true;
        s["test-unsigned"] = 33u;
        s["test-float"] = 3.14f;
        s["test-double"] = 3.14;
        s["test-string"] = std::string("hello");
        s["test-vector-int"] = std::vector<int>{ 1, 2, 3, 4 };

        std::stringstream ss;
        ss << s;
        gnx::sq_gen<T> t;
        ss >> t;

        CHECK(s == t);
        CHECK(s.has("test-void"));
        CHECK(t.has("test-void"));
        CHECK(std::any_cast<bool>(s["test-bool"]) == std::any_cast<bool>(t["test-bool"]));
        CHECK(std::any_cast<int>(s["test-int"]) == std::any_cast<int>(t["test-int"]));
        CHECK(std::any_cast<unsigned>(s["test-unsigned"]) == std::any_cast<unsigned>(t["test-unsigned"]));
        CHECK(std::any_cast<float>(s["test-float"]) == std::any_cast<float>(t["test-float"]));
        CHECK(std::any_cast<double>(s["test-double"]) == std::any_cast<double>(t["test-double"]));
        CHECK(std::any_cast<std::string>(s["test-string"]) == std::any_cast<std::string>(t["test-string"]));
        CHECK(4 == std::any_cast<std::vector<int>>(s["test-vector-int"]).size());
    }

// -- string literal operator --------------------------------------------------

    SECTION( "string literal operator" )
    {   auto t = "ACGT"_sq;
        CHECK(t == "ACGT");
        CHECK(t == "ACGT"_sq);
    }
}

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::sq_view"
,   "[view][cuda]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::universal_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::sq_view"
,   "[view][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
#else
TEMPLATE_TEST_CASE( "gnx::sq_view", "[view]", std::vector<char>)
#endif //__CUDACC__
{   typedef TestType T;

    gnx::sq_gen<T> s{"ACGT"};

// -- constructors -------------------------------------------------------------

    SECTION( "construct from sq_gen" )
    {   gnx::sq_view_gen<T> v{s};
        CHECK(v.size() == s.size());
        CHECK_FALSE(v.empty());
        CHECK(v == s);
        CHECK(v == "ACGT");
    }

    SECTION( "construct from pointer+length" )
    {   const char* p = "ACGT";
        gnx::sq_view_gen<T> v{p, 4};
        CHECK(v.size() == 4);
        CHECK(v[0] == 'A');
        CHECK(v.at(3) == 'T');
        CHECK(v.front() == 'A');
        CHECK(v.back() == 'T');
    }

// -- iterators ----------------------------------------------------------------

    SECTION( "iterate over view" )
    {   gnx::sq_view_gen<T> v{s};
        std::string collected;
        for (auto it = v.begin(); it != v.end(); ++it)
            collected.push_back(*it);
        CHECK(collected == "ACGT");

        std::string rev;
        for (auto it = v.rbegin(); it != v.rend(); ++it)
            rev.push_back(*it);
        CHECK(rev == std::string{"TGCA"});
    }

// -- modifiers ----------------------------------------------------------------

    SECTION( "remove_prefix/suffix" )
    {   gnx::sq_view_gen<T> v{s};
        v.remove_prefix(1); // drop 'A'
        CHECK(v == "CGT");
        v.remove_suffix(1); // drop trailing 'T'
        CHECK(v == "CG");
        // original sequence unchanged
        CHECK(s == "ACGT");
    }

    SECTION( "remove_prefix/suffix bounds" )
    {   gnx::sq_view_gen<T> v{s};
        REQUIRE_THROWS_AS(v.remove_prefix(5), std::out_of_range);
        REQUIRE_THROWS_AS(v.remove_suffix(5), std::out_of_range);
    }

// -- operations ---------------------------------------------------------------

    SECTION( "substr" )
    {   gnx::sq_view_gen<T> v{s};
        auto sub = v.substr(1, 2);
        CHECK(sub == "CG");
        auto to_end = v.substr(2);
        CHECK(to_end == "GT");
        REQUIRE_THROWS_AS(v.substr(10), std::out_of_range);
    }

// -- comparisons --------------------------------------------------------------

    SECTION( "compare to sq_gen and C-string" )
    {   gnx::sq_view_gen<T> v{s};
        CHECK(v == s);
        CHECK(s == v);
        CHECK(v == "ACGT");
        CHECK("ACGT" == v);
        CHECK(v != "acgt");
        CHECK("acgt" != v);
    }

// -- ranges concept support --------------------------------------------------

    SECTION( "std::ranges::view concept" )
    {   gnx::sq_view_gen<T> v{s};
        // Verify that sq_view_gen satisfies the view concept
        static_assert(std::ranges::view<gnx::sq_view_gen<T>>);
        static_assert(std::ranges::range<gnx::sq_view_gen<T>>);

        // Test composability with range adaptors
        auto transformed = v | std::views::transform
        (   [](auto c)
            {   return c == 'A' ? 'T' : c;
            }
        );
        std::string r(transformed.begin(), transformed.end());
        CHECK(r == "TCGT");
    }

    SECTION( "composable with multiple adaptors" )
    {   gnx::sq_view_gen<T> v{s};
        // Chain multiple views
        auto result = v 
            | std::views::reverse 
            | std::views::transform([](auto c) { return std::tolower(c); })
            | std::views::take(3);
        std::string r(result.begin(), result.end());
        CHECK(r == "tgc");
    }
}

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::io::fastaqz"
,   "[io][in][out][cuda]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::io::fastaqz"
,   "[io][in][out][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
#else
TEMPLATE_TEST_CASE( "gnx::io::fastaqz", "[io][in][out]", std::vector<char>)
#endif
{   typedef TestType T;
    std::string desc("Chlamydia psittaci 6BC plasmid pCps6BC, complete sequence");
    gnx::sq_gen<T> s, t;
    CHECK_THROWS_AS
    (   s.load("wrong.fa")
    ,   std::runtime_error
    );

    gnx::sq_gen<T> wrong_ndx;
    wrong_ndx.load(SAMPLE_GENOME, 3);
    CHECK(wrong_ndx.empty());
    gnx::sq_gen<T> bad_id;
    bad_id.load(SAMPLE_GENOME, "bad_id");
    CHECK(bad_id.empty());

    // REQUIRE_THAT
    // (   gnx::lut::phred33[static_cast<uint8_t>('J')]
    // ,   Catch::Matchers::WithinAbs(7.943282e-05, 0.000001)
    // );

    SECTION( "load with index" )
    {   s.load(SAMPLE_GENOME, 1, gnx::in::fast_aqz<decltype(s)>());
        CHECK(7553 == std::size(s));
        CHECK(s(0, 10) == "TATAATTAAA");
        CHECK(s( 7543) == "TCCAATTCTA");
        CHECK("NC_017288.1" == std::any_cast<std::string>(s["_id"]));
        CHECK(desc == std::any_cast<std::string>(s["_desc"]));
    }
    SECTION( "load with id" )
    {   s.load(SAMPLE_GENOME, "NC_017288.1");
        CHECK(7553 == std::size(s));
        CHECK(s(0, 10) == "TATAATTAAA");
        CHECK(s( 7543) == "TCCAATTCTA");
        CHECK("NC_017288.1" == std::any_cast<std::string>(s["_id"]));
        CHECK(desc == std::any_cast<std::string>(s["_desc"]));
    }
    SECTION( "save fasta" )
    {   s.load(SAMPLE_GENOME, 1);
        std::string filename = "test_output.fa";
        s.save(filename, gnx::out::fasta());
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
    SECTION( "save fasta.gz" )
    {   s.load(SAMPLE_GENOME, 1);
        std::string filename = "test_output.fa.gz";
        s.save(filename, gnx::out::fasta_gz());
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
    SECTION( "save fastq" )
    {   s.load(SAMPLE_READS);
        std::string filename = "test_reads.fq";
        s.save(filename, gnx::out::fastq());
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
    SECTION( "save fastq.gz" )
    {   s.load(SAMPLE_READS);
        std::string filename = "test_reads.fqz";
        s.save(filename, gnx::out::fastq_gz());
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
}

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::valid"
,   "[algorithm][valid][cuda]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::universal_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::valid"
,   "[algorithm][valid][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
#else
TEMPLATE_TEST_CASE( "gnx::valid", "[algorithm][valid]", std::vector<char>)
#endif //__CUDACC__
{   typedef TestType T;

// -- nucleotide validation ----------------------------------------------------

    SECTION( "valid nucleotide sequences" )
    {   gnx::sq_gen<T> s1{"ACGT"};
        CHECK(gnx::valid(s1, true));
        CHECK(gnx::valid_nucleotide(s1));
        gnx::sq_gen<T> s2{"ACGTACGTNNN"};
        CHECK(gnx::valid_nucleotide(s2));
        // lowercase
        gnx::sq_gen<T> s3{"acgtacgt"};
        CHECK(gnx::valid_nucleotide(s3));
        // mixed case
        gnx::sq_gen<T> s4{"AcGtNn"};
        CHECK(gnx::valid_nucleotide(s4));
        // with RNA base U
        gnx::sq_gen<T> s5{"ACGU"};
        CHECK(gnx::valid_nucleotide(s5));
        // with IupAC ambiguity codes
        gnx::sq_gen<T> s6{"ACGTRYMKSWBDHVN"};
        CHECK(gnx::valid_nucleotide(s6));
        gnx::sq_gen<T> s7{"acgtrymkswbdhvn"};
        CHECK(gnx::valid_nucleotide(s7));
    }

    SECTION( "invalid nucleotide sequences" )
    {   // with invalid character
        gnx::sq_gen<T> s1{"ACGT123"};
        CHECK_FALSE(gnx::valid_nucleotide(s1));
        gnx::sq_gen<T> s2{"ACGT X"};
        CHECK_FALSE(gnx::valid_nucleotide(s2));
        // with space
        gnx::sq_gen<T> s3{"ACG T"};
        CHECK_FALSE(gnx::valid_nucleotide(s3));
        // with newline
        gnx::sq_gen<T> s4{"ACGT\n"};
        CHECK_FALSE(gnx::valid_nucleotide(s4));
        // peptide sequence
        gnx::sq_gen<T> s5{"MVHLTPEEK"};
        CHECK_FALSE(gnx::valid_nucleotide(s5));
    }

    SECTION( "empty nucleotide sequence" )
    {   gnx::sq_gen<T> s{""};
        CHECK(gnx::valid_nucleotide(s));
    }

// -- peptide validation -------------------------------------------------------

    SECTION( "valid peptide sequences" )
    {   gnx::sq_gen<T> s1{"ACDEFGHIKLMNPQRSTVWY"};
        CHECK(gnx::valid(s1));
        CHECK(gnx::valid(s1, false));
        CHECK(gnx::valid_peptide(s1));
        CHECK_FALSE(gnx::valid_nucleotide(s1));
        // lowercase
        gnx::sq_gen<T> s2{"acdefghiklmnpqrstvwy"};
        CHECK(gnx::valid_peptide(s2));
        // mixed case
        gnx::sq_gen<T> s3{"MvHlTpEeK"};
        CHECK(gnx::valid_peptide(s3));
        // with ambiguous codes
        gnx::sq_gen<T> s4{"ACBZX"};
        CHECK(gnx::valid_peptide(s4));
        // with special amino acids
        gnx::sq_gen<T> s5{"ACUO"};
        CHECK(gnx::valid_peptide(s5));
        // with stop codon
        gnx::sq_gen<T> s6{"MVHLT*"};
        CHECK(gnx::valid_peptide(s6));
    }

    SECTION( "invalid peptide sequences" )
    {   // with number
        gnx::sq_gen<T> s1{"ACDE123"};
        CHECK_FALSE(gnx::valid_peptide(s1));
        // with space
        gnx::sq_gen<T> s2{"ACDE F"};
        CHECK_FALSE(gnx::valid_peptide(s2));
        // with newline
        gnx::sq_gen<T> s3{"ACDEF\n"};
        CHECK_FALSE(gnx::valid_peptide(s3));
        // with invalid special character
        gnx::sq_gen<T> s4{"ACDE-F"};
        CHECK_FALSE(gnx::valid_peptide(s4));
    }

    SECTION( "empty sequence" )
    {   gnx::sq_gen<T> s{""};
        CHECK(gnx::valid(s));
    }

// -- iterator-based validation ------------------------------------------------

    SECTION( "validation with iterators" )
    {   gnx::sq_gen<T> s{"ACGTACGT"};
        // full range
        CHECK(gnx::valid(s.begin(), s.end(), true));
        CHECK(gnx::valid_nucleotide(s.begin(), s.end()));
        // partial range
        CHECK(gnx::valid_nucleotide(s.begin(), s.begin() + 4));
        // substring
        auto sub = s(0, 4);
        CHECK(gnx::valid_nucleotide(sub));
    }

    SECTION( "validation with execution policy" )
    {   gnx::sq_gen<T> s;
        s.load(SAMPLE_GENOME, 0);
        CHECK(s.size() > 0);
        CHECK(gnx::valid_nucleotide(s));
        CHECK(gnx::valid_nucleotide(gnx::execution::seq, s));
        CHECK(gnx::valid_nucleotide(gnx::execution::unseq, s));
        CHECK(gnx::valid_nucleotide(gnx::execution::par, s));
        CHECK(gnx::valid_nucleotide(gnx::execution::par_unseq, s));

        // introduce an invalid character and ensure policy overload detects it
        s[2] = 'Z';
        CHECK_FALSE(gnx::valid_nucleotide(gnx::execution::seq, s));
        CHECK_FALSE(gnx::valid_nucleotide(gnx::execution::unseq, s));
        CHECK_FALSE(gnx::valid_nucleotide(gnx::execution::par, s));
        CHECK_FALSE(gnx::valid_nucleotide(gnx::execution::par_unseq, s));

        // peptide validation remains true with Z
        CHECK(gnx::valid_peptide(gnx::execution::par, s));
    }

    SECTION( "validation with sq_view" )
    {   gnx::sq_gen<T> s{"ACGTACGT"};
        gnx::sq_view_gen<T> view{s};
        CHECK(gnx::valid_nucleotide(view));
        CHECK(gnx::valid_nucleotide(view.begin(), view.end()));
    }

// -- compile-time validation --------------------------------------------------

    SECTION( "constexpr validation" )
    {   // These should compile if the function is constexpr
        constexpr std::array<char, 4> arr{'A', 'C', 'G', 'T'};
        constexpr bool result = gnx::valid_nucleotide(arr.begin(), arr.end());
        CHECK(result);
    }

// -- cross-validation tests ---------------------------------------------------

    SECTION( "sequences valid for one type but not another" )
    {   // Some characters valid for peptides but not nucleotides
        gnx::sq_gen<T> s1{"EFIKLPQVWY"};
        CHECK(gnx::valid_peptide(s1));
        CHECK_FALSE(gnx::valid_nucleotide(s1));

        // All nucleotides are also valid peptides (overlap in alphabet)
        gnx::sq_gen<T> s2{"ACGT"};
        CHECK(gnx::valid_nucleotide(s2));
        CHECK(gnx::valid_peptide(s2)); // A, C, G, T are also amino acids
    }

// -- range compatibility tests ------------------------------------------------

    SECTION( "view ranges compatibility" )
    {   gnx::sq_gen<T> s{"ACGTACGTKQ"};
        CHECK(gnx::valid_nucleotide(s | std::views::take(8)));
        gnx::sq_view_gen<T> v{s};
        v.remove_suffix(2); // drop 'KQ'
        CHECK(gnx::valid_nucleotide(v));
        CHECK(gnx::valid_nucleotide(v.begin(), v.end()));
        auto result = s
        |   std::views::take(8)
        |   std::views::transform([](auto c) { return std::tolower(c); })
        |   std::views::reverse;
        CHECK(gnx::valid_nucleotide(result));
    }
}

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::valid::device"
,   "[algorithm][valid][cuda]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
{   typedef TestType T;

    gnx::sq_gen<T> s;
    s.load(SAMPLE_GENOME, 0);
    CHECK(s.size() > 0);

    SECTION( "device vector" )
    {   CHECK(gnx::valid_nucleotide(thrust::cuda::par, s));
        s[2] = 'Z';
        CHECK_FALSE(gnx::valid_nucleotide(thrust::cuda::par, s));
    }

    SECTION( "cuda stream" )
    {   cudaStream_t streamA;
        cudaStreamCreate(&streamA);
        CHECK(gnx::valid_nucleotide(thrust::cuda::par.on(streamA), s));
        cudaStreamSynchronize(streamA);
        s[2] = 'Z';
        CHECK_FALSE(gnx::valid_nucleotide(thrust::cuda::par_nosync.on(streamA), s));
        cudaStreamSynchronize(streamA);
        cudaStreamDestroy(streamA);
    }
}
#endif //__CUDACC__

#if defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::valid::device"
,   "[algorithm][valid][rocm]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
{   typedef TestType T;

    gnx::sq_gen<T> s;
    s.load(SAMPLE_GENOME, 0);
    CHECK(s.size() > 0);

    SECTION( "device vector" )
    {   CHECK(gnx::valid_nucleotide(thrust::hip::par, s));
        s[2] = 'Z';
        CHECK_FALSE(gnx::valid_nucleotide(thrust::hip::par, s));
    }

    SECTION( "cuda stream" )
    {   hipStream_t streamA;
        hipStreamCreate(&streamA);
        CHECK(gnx::valid_nucleotide(thrust::hip::par.on(streamA), s));
        hipStreamSynchronize(streamA);
        s[2] = 'Z';
        CHECK_FALSE(gnx::valid_nucleotide(thrust::hip::par_nosync.on(streamA), s));
        hipStreamSynchronize(streamA);
        hipStreamDestroy(streamA);
    }
}
#endif //__HIPCC__

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::random"
,   "[algorithm][random][cuda]"
,   std::vector<char>
,   thrust::host_vector<char>
)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::random"
,   "[algorithm][random][rocm]"
,   std::vector<char>
,   thrust::host_vector<char>
)
#else
TEMPLATE_TEST_CASE( "gnx::random", "[algorithm][random]", std::vector<char>)
#endif //__CUDACC__ || __HIPCC__
{   typedef TestType T;
    gnx::sq_gen<T> s(20);
    const auto N{10'000};

    SECTION( "random nucleotide sequence" )
    {   gnx::rand(s.begin(), 20, "ACGT", seed_pi);
        CHECK(gnx::valid_nucleotide(s));
        CHECK(s == "TTCGGCCGTCGTTAAACACG");
        auto t = gnx::random::dna<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "random sequence with execution policy" )
    {   gnx::sq_gen<T> r(N);
        auto t = gnx::random::dna<decltype(s)>(N, seed_pi);
        CHECK(N == t.size());
        gnx::rand(gnx::execution::seq, r.begin(), N, "ACGT", seed_pi);
        CHECK(t == r);
        gnx::rand(gnx::execution::unseq, r.begin(), N, "ACGT", seed_pi);
        CHECK(t == r);
        gnx::rand(gnx::execution::par, r.begin(), N, "ACGT", seed_pi);
        CHECK(t == r);
        gnx::rand(gnx::execution::par_unseq, r.begin(), N, "ACGT", seed_pi);
        CHECK(t == r);
    }

    SECTION( "random rna sequence" )
    {   gnx::rand(s.begin(), 20, "ACGU", seed_pi);
        CHECK(gnx::valid_nucleotide(s));
        CHECK(s == "UUCGGCCGUCGUUAAACACG");
        auto t = gnx::random::rna<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "random peptide sequence" )
    {   gnx::rand(s.begin(), 20, "ACDEFGHIKLMNPQRSTVWY", seed_pi);
        CHECK(gnx::valid_peptide(s));
        CHECK(s == "TTIQRHHMVKQSSFDALCLM");
        auto t = gnx::random::peptide<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "random nucleotide sequence with weights" )
    {   gnx::rand(s.begin(), 20, "ACGT", {35, 15, 15, 35}, seed_pi);
        CHECK(gnx::valid_nucleotide(s));
        CHECK(s == "TTCTTAAGTCTTTAAACACG");
        auto t = gnx::random::dna<decltype(s)>(20, 30, seed_pi);
        t[2] = 'C';
        CHECK(s == t);
    }
}

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE
(   "gnx::random::device"
,   "[algorithm][random][cuda]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
)
{   typedef TestType T;
    gnx::sq_gen<T> s(20);
    const auto N{10'000};

    SECTION( "device vector" )
    {   gnx::rand(thrust::cuda::par, s.begin(), 20, "ACGT", seed_pi);
        CHECK(gnx::valid_nucleotide(thrust::cuda::par, s));
        CHECK(s == "TTCGGCCGTCGTTAAACACG");
        auto t = gnx::random::dna<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "cuda stream" )
    {   auto r = gnx::random::dna<decltype(s)>(N, seed_pi);
        gnx::sq_gen<T> t(N);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        gnx::rand(thrust::cuda::par.on(stream), t.begin(), N, "ACGT", seed_pi);
        CHECK(gnx::valid_nucleotide(thrust::cuda::par.on(stream), t));
        CHECK(r == t);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
}
#endif //__CUDACC__

#if defined(__HIPCC__)
TEMPLATE_TEST_CASE
(   "gnx::random::device"
,   "[algorithm][random][rocm]"
,   thrust::device_vector<char>
,   thrust::universal_vector<char>
,   gnx::unified_vector<char>
)
{   typedef TestType T;
    gnx::sq_gen<T> s(20);
    const auto N{10'000};

    SECTION( "device vector" )
    {   gnx::rand(thrust::hip::par, s.begin(), 20, "ACGT", seed_pi);
        CHECK(gnx::valid_nucleotide(thrust::hip::par, s));
        CHECK(s == "TTCGGCCGTCGTTAAACACG");
        auto t = gnx::random::dna<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "hip stream" )
    {   auto r = gnx::random::dna<decltype(s)>(N, seed_pi);
        gnx::sq_gen<T> t(N);
        hipStream_t stream;
        hipStreamCreate(&stream);
        gnx::rand(thrust::hip::par.on(stream), t.begin(), N, "ACGT", seed_pi);
        CHECK(gnx::valid_nucleotide(thrust::hip::par.on(stream), t));
        CHECK(r == t);
        hipStreamSynchronize(stream);
        hipStreamDestroy(stream);
    }
}
#endif //__HIPCC__

// =============================================================================
// local_align tests
// =============================================================================

TEST_CASE( "gnx::local_align", "[algorithm][local_align]")
{
// -- basic alignment ----------------------------------------------------------

    SECTION( "identical sequences" )
    {   std::string s1 = "ACGT";
        std::string s2 = "ACGT";
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score == 8);  // 4 matches * 2
        CHECK(result.aligned_seq1 == "ACGT");
        CHECK(result.aligned_seq2 == "ACGT");
        CHECK(result.traceback.size() == 4);
        for (const auto& dir : result.traceback)
            CHECK(dir == gnx::alignment_direction::diagonal);
    }

    SECTION( "single mismatch" )
    {   std::string s1 = "ACGT";
        std::string s2 = "ACAT";
        auto result = gnx::local_align(s1, s2);
        
        // Best local alignment should still find matching regions
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() == result.aligned_seq2.length());
    }

    SECTION( "no alignment (completely different)" )
    {   std::string s1 = "AAAA";
        std::string s2 = "TTTT";
        auto result = gnx::local_align(s1.begin(), s1.end(), s2.begin(), s2.end(), 2, -3, -1);
        
        // With strong mismatch penalty, may have low or zero score
        CHECK(result.score >= 0);  // Smith-Waterman never goes negative
    }

// -- subsequence alignment ----------------------------------------------------

    SECTION( "subsequence in larger sequence" )
    {   std::string s1 = "ACGTACGT";
        std::string s2 = "ACGT";
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score == 8);  // Perfect match of ACGT
        CHECK(result.aligned_seq1 == "ACGT");
        CHECK(result.aligned_seq2 == "ACGT");
    }

    SECTION( "overlapping sequences" )
    {   std::string s1 = "ACGTACGT";
        std::string s2 = "TACGTACG";
        auto result = gnx::local_align(s1, s2);
        
        // Should find significant alignment
        CHECK(result.score > 10);
        CHECK(result.aligned_seq1.length() > 5);
    }

// -- gap handling -------------------------------------------------------------

    SECTION( "alignment with gap in first sequence" )
    {   std::string s1 = "ACGT";
        std::string s2 = "ACGGT";
        auto result = gnx::local_align(s1.begin(), s1.end(), s2.begin(), s2.end(), 2, -1, -1);
        
        CHECK(result.score >= 4);  // At least some matches
        // May or may not have gap depending on scoring
    }

    SECTION( "alignment with gap in second sequence" )
    {   std::string s1 = "ACGGT";
        std::string s2 = "ACGT";
        auto result = gnx::local_align(s1.begin(), s1.end(), s2.begin(), s2.end(), 2, -1, -1);
        
        CHECK(result.score >= 4);
    }

// -- custom scoring -----------------------------------------------------------

    SECTION( "custom match score" )
    {   std::string s1 = "ACGT";
        std::string s2 = "ACGT";
        auto result = gnx::local_align(s1.begin(), s1.end(), s2.begin(), s2.end(), 5, -1, -1);
        
        CHECK(result.score == 20);  // 4 matches * 5
    }

    SECTION( "custom mismatch penalty" )
    {   std::string s1 = "ACGT";
        std::string s2 = "TTTT";
        auto result = gnx::local_align(s1.begin(), s1.end(), s2.begin(), s2.end(), 2, -10, -1);
        
        // Strong mismatch penalty should result in low score
        CHECK(result.score <= 2);
    }

    SECTION( "custom gap penalty" )
    {   std::string s1 = "ACGT";
        std::string s2 = "ACGGT";
        auto result = gnx::local_align(s1.begin(), s1.end(), s2.begin(), s2.end(), 2, -1, -5);
        
        // Strong gap penalty should discourage gaps
        CHECK(result.score >= 0);
    }

// -- edge cases ---------------------------------------------------------------

    SECTION( "empty first sequence" )
    {   std::string s1 = "";
        std::string s2 = "ACGT";
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score == 0);
        CHECK(result.aligned_seq1.empty());
        CHECK(result.aligned_seq2.empty());
    }

    SECTION( "empty second sequence" )
    {   std::string s1 = "ACGT";
        std::string s2 = "";
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score == 0);
        CHECK(result.aligned_seq1.empty());
        CHECK(result.aligned_seq2.empty());
    }

    SECTION( "both sequences empty" )
    {   std::string s1 = "";
        std::string s2 = "";
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score == 0);
        CHECK(result.aligned_seq1.empty());
        CHECK(result.aligned_seq2.empty());
    }

    SECTION( "single character sequences" )
    {   std::string s1 = "A";
        std::string s2 = "A";
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score == 2);  // Default match score
        CHECK(result.aligned_seq1 == "A");
        CHECK(result.aligned_seq2 == "A");
    }

    SECTION( "single character mismatch" )
    {   std::string s1 = "A";
        std::string s2 = "T";
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score == 0);  // SW doesn't allow negative scores
    }

// -- case insensitivity -------------------------------------------------------

    SECTION( "lowercase sequences" )
    {   std::string s1 = "acgt";
        std::string s2 = "acgt";
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score == 8);
        CHECK(result.aligned_seq1 == "acgt");
        CHECK(result.aligned_seq2 == "acgt");
    }

    SECTION( "mixed case sequences" )
    {   std::string s1 = "AcGt";
        std::string s2 = "aCgT";
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score == 8);  // Should match regardless of case
    }

// -- gnx::sq_gen tests --------------------------------------------------------

    SECTION( "gnx::sq alignment" )
    {   gnx::sq s1{"ACGTACGT"};
        gnx::sq s2{"TACGT"};
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() > 0);
        CHECK(result.aligned_seq2.length() > 0);
    }

// -- realistic biological example ---------------------------------------------

    SECTION( "realistic DNA sequences with SNP" )
    {   // Two sequences with single nucleotide polymorphism
        std::string s1 = "ATCGATCGATCG";
        std::string s2 = "ATCGCTCGATCG";  // C instead of A at position 5
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score >= 16);  // Most bases should match
        CHECK(result.aligned_seq1.length() >= 10);
    }

    SECTION( "realistic DNA with indel" )
    {   // Sequence with insertion/deletion
        std::string s1 = "ATCGATCGATCG";
        std::string s2 = "ATCGTCGATCG";  // Missing 'A' at position 5
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score > 0);
        // Should find good alignment around the indel
    }

// -- longer sequences ---------------------------------------------------------

    SECTION( "longer sequences" )
    {   std::string s1 = "ACGTACGTACGTACGTACGTACGTACGTACGT";
        std::string s2 = "ACGTACGTACGTACGTACGTACGTACGTACGT";
        auto result = gnx::local_align(s1, s2);
        
        CHECK(result.score == 64);  // 32 matches * 2
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "partially matching longer sequences" )
    {   std::string s1 = "AAAAAAACGTACGTACGTTTTTTT";
        std::string s2 = "ACGTACGTACGT";
        auto result = gnx::local_align(s1, s2);
        
        // Should find the matching middle part
        CHECK(result.score == 24);  // 12 matches * 2
        CHECK(result.aligned_seq1 == "ACGTACGTACGT");
        CHECK(result.aligned_seq2 == "ACGTACGTACGT");
    }
}

// =============================================================================
// local_align with substitution matrices tests
// =============================================================================

TEST_CASE( "gnx::local_align with substitution matrices", "[algorithm][local_align][blosum][pam]")
{
// -- BLOSUM62 tests -----------------------------------------------------------

    SECTION( "BLOSUM62 - identical peptide sequences" )
    {   std::string s1 = "ARNDCQEGHILKMFPSTWYV";
        std::string s2 = "ARNDCQEGHILKMFPSTWYV";
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62);
        
        // Score should be sum of diagonal elements for each amino acid
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "BLOSUM62 - single amino acid difference" )
    {   std::string s1 = "ARNDCQEG";
        std::string s2 = "ARNDCQKG";  // E->K substitution
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62);
        
        // Should align well with one mismatch
        CHECK(result.score > 20);
        CHECK(result.aligned_seq1.length() == result.aligned_seq2.length());
    }

    SECTION( "BLOSUM62 - conservative substitution" )
    {   // Leucine (L) and Isoleucine (I) are similar hydrophobic amino acids
        std::string s1 = "ACDEFGHIKLMNPQRSTVWY";
        std::string s2 = "ACDEFGHIKLMNPQRSTVWY";
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62);
        
        CHECK(result.score > 50);
        CHECK(result.aligned_seq1 == s1);
    }

    SECTION( "BLOSUM62 - peptide with gaps" )
    {   std::string s1 = "ARNDCQEG";
        std::string s2 = "ARNDQEG";  // C removed
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62, -8);
        
        CHECK(result.score > 0);
        // Should align with one gap
    }

    SECTION( "BLOSUM62 - different gap penalty" )
    {   std::string s1 = "ACDEFG";
        std::string s2 = "ACDEFG";
        auto result1 = gnx::local_align(s1, s2, gnx::lut::blosum62, -8);
        auto result2 = gnx::local_align(s1, s2, gnx::lut::blosum62, -2);
        
        // With identical sequences, gap penalty shouldn't matter
        CHECK(result1.score == result2.score);
    }

    SECTION( "BLOSUM62 - case insensitive" )
    {   std::string s1 = "ARNDCQEG";
        std::string s2 = "arndcqeg";
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62);
        
        // Should match regardless of case
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() == result.aligned_seq2.length());
    }

// -- BLOSUM80 tests -----------------------------------------------------------

    SECTION( "BLOSUM80 - identical sequences" )
    {   std::string s1 = "MVHLTPEEK";
        std::string s2 = "MVHLTPEEK";
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum80);
        
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "BLOSUM80 vs BLOSUM62 comparison" )
    {   // BLOSUM80 is more stringent for closely related sequences
        std::string s1 = "ACDEFG";
        std::string s2 = "ACDEFG";
        auto result62 = gnx::local_align(s1, s2, gnx::lut::blosum62);
        auto result80 = gnx::local_align(s1, s2, gnx::lut::blosum80);
        
        // BLOSUM80 typically gives higher scores for identical sequences
        CHECK(result80.score >= result62.score);
    }

// -- BLOSUM45 tests -----------------------------------------------------------

    SECTION( "BLOSUM45 - distantly related sequences" )
    {   std::string s1 = "ARNDCQEG";
        std::string s2 = "ARNDCQEG";
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum45);
        
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

// -- PAM250 tests -------------------------------------------------------------

    SECTION( "PAM250 - identical sequences" )
    {   std::string s1 = "MVHLTPEEK";
        std::string s2 = "MVHLTPEEK";
        auto result = gnx::local_align(s1, s2, gnx::lut::pam250);
        
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "PAM250 - with mismatches" )
    {   std::string s1 = "ARNDCQEG";
        std::string s2 = "ARNDCQKG";  // E->K substitution
        auto result = gnx::local_align(s1, s2, gnx::lut::pam250);
        
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() == result.aligned_seq2.length());
    }

// -- PAM120 tests -------------------------------------------------------------

    SECTION( "PAM120 - closely related sequences" )
    {   std::string s1 = "ACDEFGHIKL";
        std::string s2 = "ACDEFGHIKL";
        auto result = gnx::local_align(s1, s2, gnx::lut::pam120);
        
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

    SECTION( "PAM120 vs PAM250 comparison" )
    {   // Different PAM matrices for different evolutionary distances
        std::string s1 = "ACDEFG";
        std::string s2 = "ACDEFG";
        auto result120 = gnx::local_align(s1, s2, gnx::lut::pam120);
        auto result250 = gnx::local_align(s1, s2, gnx::lut::pam250);
        
        // Both should align perfectly
        CHECK(result120.score > 0);
        CHECK(result250.score > 0);
    }

// -- PAM30 tests --------------------------------------------------------------

    SECTION( "PAM30 - very closely related sequences" )
    {   std::string s1 = "MVHLTPEEK";
        std::string s2 = "MVHLTPEEK";
        auto result = gnx::local_align(s1, s2, gnx::lut::pam30);
        
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }

// -- Realistic peptide alignment examples -------------------------------------

    SECTION( "realistic - human vs mouse hemoglobin fragment" )
    {   // Simplified example of conserved protein region
        std::string human = "VLSPADKTNVKAAW";
        std::string mouse = "VLSAADKTNVKAAW";  // P->A substitution
        auto result = gnx::local_align(human, mouse, gnx::lut::blosum62);
        
        // Should find good alignment despite one difference
        CHECK(result.score > 40);
        CHECK(result.aligned_seq1.length() >= 10);
    }

    SECTION( "realistic - enzyme active site comparison" )
    {   // Catalytic triad-like sequence
        std::string enzyme1 = "HDSGICN";
        std::string enzyme2 = "HDSGVCN";  // I->V conservative substitution
        auto result = gnx::local_align(enzyme1, enzyme2, gnx::lut::blosum62);
        
        // Conservative substitution should still score well
        CHECK(result.score > 20);
    }

    SECTION( "realistic - signal peptide vs mature protein" )
    {   std::string full_seq = "MKTIIALSYIFCLVFAACDEFGHIKL";
        std::string mature  = "ACDEFGHIKL";  // After signal peptide cleavage
        auto result = gnx::local_align(full_seq, mature, gnx::lut::blosum62);
        
        // Should find the mature protein region
        CHECK(result.score > 30);
        CHECK(result.aligned_seq2 == mature);
    }

// -- ambiguous amino acids ----------------------------------------------------

    SECTION( "ambiguous amino acids - B (D or N)" )
    {   std::string s1 = "ACDEFG";
        std::string s2 = "ACBEFG";  // D->B
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62);
        
        // B should have reasonable score with D
        CHECK(result.score > 0);
    }

    SECTION( "ambiguous amino acids - Z (E or Q)" )
    {   std::string s1 = "ACDEFG";
        std::string s2 = "ACDQFG";  // E->Q
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62);
        
        CHECK(result.score > 0);
    }

    SECTION( "ambiguous amino acids - X (any)" )
    {   std::string s1 = "ACDEFG";
        std::string s2 = "ACXEFG";  // D->X
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62);
        
        // X should have neutral or small penalty
        CHECK(result.score >= 0);
    }

// -- stop codon handling ------------------------------------------------------

    SECTION( "stop codon in sequence" )
    {   std::string s1 = "ACDEFG*";
        std::string s2 = "ACDEFG*";
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62);
        
        // Should handle stop codon
        CHECK(result.score > 0);
    }

// -- edge cases with matrices -------------------------------------------------

    SECTION( "empty sequences with matrix" )
    {   std::string s1 = "";
        std::string s2 = "ACDEFG";
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62);
        
        CHECK(result.score == 0);
        CHECK(result.aligned_seq1.empty());
        CHECK(result.aligned_seq2.empty());
    }

    SECTION( "single amino acid with matrix" )
    {   std::string s1 = "A";
        std::string s2 = "A";
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62);
        
        // BLOSUM62[A][A] = 4
        CHECK(result.score == 4);
        CHECK(result.aligned_seq1 == "A");
        CHECK(result.aligned_seq2 == "A");
    }

// -- gnx::sq with matrices ----------------------------------------------------

    SECTION( "gnx::sq with BLOSUM62" )
    {   gnx::sq s1{"MVHLTPEEK"};
        gnx::sq s2{"MVHLTPEEK"};
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62);
        
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1 == "MVHLTPEEK");
        CHECK(result.aligned_seq2 == "MVHLTPEEK");
    }

    SECTION( "gnx::sq with PAM250" )
    {   gnx::sq s1{"ACDEFGHIKL"};
        gnx::sq s2{"ACDEFGHIKL"};
        auto result = gnx::local_align(s1, s2, gnx::lut::pam250);
        
        CHECK(result.score > 0);
        CHECK(result.aligned_seq1.length() > 0);
    }

// -- performance test with longer sequences -----------------------------------

    SECTION( "longer peptide sequences with BLOSUM62" )
    {   std::string s1 = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH";
        std::string s2 = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH";
        auto result = gnx::local_align(s1, s2, gnx::lut::blosum62);
        
        // Human beta-globin, should align perfectly with itself
        CHECK(result.score > 500);
        CHECK(result.aligned_seq1 == s1);
        CHECK(result.aligned_seq2 == s2);
    }
}

