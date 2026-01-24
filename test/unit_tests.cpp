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
#include <catch2/catch_all.hpp>

#include <gynx/sq.hpp>
#include <gynx/sq_view.hpp>
#include <gynx/io/fastaqz.hpp>
#include <gynx/algorithms/valid.hpp>
#include <gynx/algorithms/random.hpp>

const uint64_t seed_pi{3141592654};

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE( "gynx::sq", "[class][cuda]", std::vector<char>, thrust::host_vector<char>, thrust::device_vector<char>, thrust::universal_vector<char>)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE( "gynx::sq", "[class][rocm]", std::vector<char>, thrust::host_vector<char>, thrust::device_vector<char>, thrust::universal_vector<char>, gynx::unified_vector<char>)
#else
TEMPLATE_TEST_CASE( "gynx::sq", "[class]", std::vector<char>)
#endif
{   typedef TestType T;

    gynx::sq_gen<T> s{"ACGT"};
    std::string t{"ACGT"}, u{"acgt"}, v{"ACGT "};
    s["test-int"] = -33;

    // thrust::device_vector<char> dv{s.begin(), s.end()};

// -- comparison operators -----------------------------------------------------

    SECTION( "comparison operators" )
    {   REQUIRE(  s == gynx::sq_gen<T>("ACGT"));
        REQUIRE(!(s == gynx::sq_gen<T>("acgt")));
        REQUIRE(  s != gynx::sq_gen<T>("acgt") );
        REQUIRE(!(s != gynx::sq_gen<T>("ACGT")));

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
        &&  !std::is_same_v<T, gynx::unified_vector<char>>
#endif //__HIPCC__
        )
        {   gynx::sq_gen<T> a4(4, 'A');
            CHECK(a4 == "AAAA");
            gynx::sq_gen<T> c4(4, 'C');
            CHECK(c4 == "CCCC");
        }
#else
        gynx::sq_gen<T> a4(4, 'A');
        CHECK(a4 == "AAAA");
        gynx::sq_gen<T> c4(4, 'C');
        CHECK(c4 == "CCCC");
#endif //__CUDACC__
    }
    SECTION( "string_view constructor" )
    {   gynx::sq_gen<T> c("ACGT");
        CHECK(s == c);
    }
    SECTION( "sq_view constructor" )
    {
#if defined(__CUDACC__) || defined(__HIPCC__) // not supported for thrust::device_vector
        if constexpr
        (   !std::is_same_v<T, thrust::device_vector<char>>
        )
        {   gynx::sq_view_gen<T> sv(s);
            CHECK(s == sv);
        }
#else
        gynx::sq_view_gen<T> sv(s);
        CHECK(s == sv);
#endif //__CUDACC__
    }
    SECTION( "iterator constructor" )
    {   std::string acgt{"ACGT"};
        gynx::sq_gen<T> c(std::begin(acgt), std::end(acgt));
        CHECK(s == c);
    }
    SECTION( "copy constructor" )
    {   gynx::sq_gen<T> c(s);
        CHECK(c == s);
        CHECK(-33 == std::any_cast<int>(c["test-int"]));
    }
    SECTION( "move constructor" )
    {   gynx::sq_gen<T> m(std::move(s));
        CHECK(s.empty());
        CHECK(m == gynx::sq_gen<T>("ACGT"));
        CHECK(-33 == std::any_cast<int>(m["test-int"]));
    }
    SECTION( "initializer list" )
    {   gynx::sq_gen<T> c{'A', 'C', 'G', 'T'};
        CHECK(c == s);
    }

// -- copy assignment operators ------------------------------------------------

    SECTION( "copy assignment operator" )
    {   gynx::sq_gen<T> c = s;
        CHECK(c == s);
        CHECK(-33 == std::any_cast<int>(c["test-int"]));
    }
    SECTION( "move constructor" )
    {   gynx::sq_gen<T> m = gynx::sq_gen<T>("ACGT");
        CHECK(m == s);
    }
    SECTION( "initializer list" )
    {   gynx::sq_gen<T> c = {'A', 'C', 'G', 'T'};
        CHECK(c == s);
    }

// -- iterators ----------------------------------------------------------------

    SECTION( "begin/end" )
    {   gynx::sq_gen<T> t("AAAA");
        for (auto a : t)
            CHECK(a == 'A');
#if defined(__CUDACC__) || defined(__HIPCC__) // not supported for thrust::device_vector
        if constexpr
        (   !std::is_same_v<T, thrust::device_vector<char>>
#if defined(__HIPCC__)
        &&  !std::is_same_v<T, gynx::unified_vector<char>>
#endif //__HIPCC__
        )
        {   for (auto& a : t)
                a = 'T';
            CHECK(t == gynx::sq_gen<T>("TTTT"));
        }
#else
        for (auto& a : t)
            a = 'T';
        CHECK(t == gynx::sq_gen<T>("TTTT"));
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
    {   const gynx::sq_gen<T> t("AAAA");
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
    {   gynx::sq_gen<T> t("AAAA");
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
    {   const gynx::sq_gen<T> t("ACGT");
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
    {   gynx::sq_gen<T> e;
        CHECK(e.empty());
        e["test"] = 1;
        CHECK(!e.empty());
        CHECK(!s.empty());
    }
    SECTION( "size()" )
    {   gynx::sq_gen<T> e;
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
    {   gynx::sq_gen<T> org{"CCATACGTGAC"};
        CHECK(org(4, 4) == s);
        CHECK(org(0) == org);
        CHECK(org(4) == "ACGTGAC");
        CHECK_THROWS_AS(org(20) == "ACGTGAC", std::out_of_range);

        // casting sq_view to sq_gen
        gynx::sq_gen<T> sub = gynx::sq_gen<T>(org(4, 10));
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
        gynx::sq_gen<T> t;
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

#if defined(__CUDACC__) || defined(__HIPCC__)
TEMPLATE_TEST_CASE( "gynx::sq_view", "[view]", std::vector<char>, thrust::host_vector<char>, thrust::universal_vector<char>)
#else
TEMPLATE_TEST_CASE( "gynx::sq_view", "[view]", std::vector<char>)
#endif //__CUDACC__
{   typedef TestType T;

    gynx::sq_gen<T> s{"ACGT"};

// -- constructors -------------------------------------------------------------

    SECTION( "construct from sq_gen" )
    {   gynx::sq_view_gen<T> v{s};
        CHECK(v.size() == s.size());
        CHECK_FALSE(v.empty());
        CHECK(v == s);
        CHECK(v == "ACGT");
    }

    SECTION( "construct from pointer+length" )
    {   const char* p = "ACGT";
        gynx::sq_view_gen<T> v{p, 4};
        CHECK(v.size() == 4);
        CHECK(v[0] == 'A');
        CHECK(v.at(3) == 'T');
        CHECK(v.front() == 'A');
        CHECK(v.back() == 'T');
    }

// -- iterators ----------------------------------------------------------------

    SECTION( "iterate over view" )
    {   gynx::sq_view_gen<T> v{s};
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
    {   gynx::sq_view_gen<T> v{s};
        v.remove_prefix(1); // drop 'A'
        CHECK(v == "CGT");
        v.remove_suffix(1); // drop trailing 'T'
        CHECK(v == "CG");
        // original sequence unchanged
        CHECK(s == "ACGT");
    }

    SECTION( "remove_prefix/suffix bounds" )
    {   gynx::sq_view_gen<T> v{s};
        REQUIRE_THROWS_AS(v.remove_prefix(5), std::out_of_range);
        REQUIRE_THROWS_AS(v.remove_suffix(5), std::out_of_range);
    }

// -- operations ---------------------------------------------------------------

    SECTION( "substr" )
    {   gynx::sq_view_gen<T> v{s};
        auto sub = v.substr(1, 2);
        CHECK(sub == "CG");
        auto to_end = v.substr(2);
        CHECK(to_end == "GT");
        REQUIRE_THROWS_AS(v.substr(10), std::out_of_range);
    }

// -- comparisons --------------------------------------------------------------

    SECTION( "compare to sq_gen and C-string" )
    {   gynx::sq_view_gen<T> v{s};
        CHECK(v == s);
        CHECK(s == v);
        CHECK(v == "ACGT");
        CHECK("ACGT" == v);
        CHECK(v != "acgt");
        CHECK("acgt" != v);
    }

// -- ranges concept support --------------------------------------------------

    SECTION( "std::ranges::view concept" )
    {   gynx::sq_view_gen<T> v{s};
        // Verify that sq_view_gen satisfies the view concept
        static_assert(std::ranges::view<gynx::sq_view_gen<T>>);
        static_assert(std::ranges::range<gynx::sq_view_gen<T>>);

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
    {   gynx::sq_view_gen<T> v{s};
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
TEMPLATE_TEST_CASE( "gynx::io::fastaqz", "[io][in][out][cuda]", std::vector<char>, thrust::host_vector<char>, thrust::device_vector<char>, thrust::universal_vector<char>)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE( "gynx::io::fastaqz", "[io][in][out][rocm]", std::vector<char>, thrust::host_vector<char>, thrust::device_vector<char>, thrust::universal_vector<char>, gynx::unified_vector<char>)
#else
TEMPLATE_TEST_CASE( "gynx::io::fastaqz", "[io][in][out]", std::vector<char>)
#endif
{   typedef TestType T;
    std::string desc("Chlamydia psittaci 6BC plasmid pCps6BC, complete sequence");
    gynx::sq_gen<T> s, t;
    CHECK_THROWS_AS
    (   s.load("wrong.fa")
    ,   std::runtime_error
    );

    gynx::sq_gen<T> wrong_ndx;
    wrong_ndx.load(SAMPLE_GENOME, 3);
    CHECK(wrong_ndx.empty());
    gynx::sq_gen<T> bad_id;
    bad_id.load(SAMPLE_GENOME, "bad_id");
    CHECK(bad_id.empty());

    // REQUIRE_THAT
    // (   gynx::lut::phred33[static_cast<uint8_t>('J')]
    // ,   Catch::Matchers::WithinAbs(7.943282e-05, 0.000001)
    // );

    SECTION( "load with index" )
    {   s.load(SAMPLE_GENOME, 1, gynx::in::fast_aqz<decltype(s)>());
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
        s.save(filename, gynx::out::fasta());
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
    SECTION( "save fasta.gz" )
    {   s.load(SAMPLE_GENOME, 1);
        std::string filename = "test_output.fa.gz";
        s.save(filename, gynx::out::fasta_gz());
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
    SECTION( "save fastq" )
    {   s.load(SAMPLE_READS);
        std::string filename = "test_reads.fq";
        s.save(filename, gynx::out::fastq());
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
    SECTION( "save fastq.gz" )
    {   s.load(SAMPLE_READS);
        std::string filename = "test_reads.fqz";
        s.save(filename, gynx::out::fastq_gz());
        t.load(filename);
        CHECK(s == t);
        std::remove(filename.c_str());
    }
}

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE( "gynx::valid", "[algorithm][valid][cuda]", std::vector<char>, thrust::host_vector<char>, thrust::universal_vector<char>)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE( "gynx::valid", "[algorithm][valid][rocm]", std::vector<char>, thrust::host_vector<char>, thrust::universal_vector<char>, gynx::unified_vector<char>)
#else
TEMPLATE_TEST_CASE( "gynx::valid", "[algorithm][valid]", std::vector<char>)
#endif //__CUDACC__
{   typedef TestType T;

// -- nucleotide validation ----------------------------------------------------

    SECTION( "valid nucleotide sequences" )
    {   gynx::sq_gen<T> s1{"ACGT"};
        CHECK(gynx::valid(s1, true));
        CHECK(gynx::valid_nucleotide(s1));
        gynx::sq_gen<T> s2{"ACGTACGTNNN"};
        CHECK(gynx::valid_nucleotide(s2));
        // lowercase
        gynx::sq_gen<T> s3{"acgtacgt"};
        CHECK(gynx::valid_nucleotide(s3));
        // mixed case
        gynx::sq_gen<T> s4{"AcGtNn"};
        CHECK(gynx::valid_nucleotide(s4));
        // with RNA base U
        gynx::sq_gen<T> s5{"ACGU"};
        CHECK(gynx::valid_nucleotide(s5));
        // with IUPAC ambiguity codes
        gynx::sq_gen<T> s6{"ACGTRYMKSWBDHVN"};
        CHECK(gynx::valid_nucleotide(s6));
        gynx::sq_gen<T> s7{"acgtrymkswbdhvn"};
        CHECK(gynx::valid_nucleotide(s7));
    }

    SECTION( "invalid nucleotide sequences" )
    {   // with invalid character
        gynx::sq_gen<T> s1{"ACGT123"};
        CHECK_FALSE(gynx::valid_nucleotide(s1));
        gynx::sq_gen<T> s2{"ACGT X"};
        CHECK_FALSE(gynx::valid_nucleotide(s2));
        // with space
        gynx::sq_gen<T> s3{"ACG T"};
        CHECK_FALSE(gynx::valid_nucleotide(s3));
        // with newline
        gynx::sq_gen<T> s4{"ACGT\n"};
        CHECK_FALSE(gynx::valid_nucleotide(s4));
        // peptide sequence
        gynx::sq_gen<T> s5{"MVHLTPEEK"};
        CHECK_FALSE(gynx::valid_nucleotide(s5));
    }

    SECTION( "empty nucleotide sequence" )
    {   gynx::sq_gen<T> s{""};
        CHECK(gynx::valid_nucleotide(s));
    }

// -- peptide validation -------------------------------------------------------

    SECTION( "valid peptide sequences" )
    {   gynx::sq_gen<T> s1{"ACDEFGHIKLMNPQRSTVWY"};
        CHECK(gynx::valid(s1));
        CHECK(gynx::valid(s1, false));
        CHECK(gynx::valid_peptide(s1));
        CHECK_FALSE(gynx::valid_nucleotide(s1));
        // lowercase
        gynx::sq_gen<T> s2{"acdefghiklmnpqrstvwy"};
        CHECK(gynx::valid_peptide(s2));
        // mixed case
        gynx::sq_gen<T> s3{"MvHlTpEeK"};
        CHECK(gynx::valid_peptide(s3));
        // with ambiguous codes
        gynx::sq_gen<T> s4{"ACBZX"};
        CHECK(gynx::valid_peptide(s4));
        // with special amino acids
        gynx::sq_gen<T> s5{"ACUO"};
        CHECK(gynx::valid_peptide(s5));
        // with stop codon
        gynx::sq_gen<T> s6{"MVHLT*"};
        CHECK(gynx::valid_peptide(s6));
    }

    SECTION( "invalid peptide sequences" )
    {   // with number
        gynx::sq_gen<T> s1{"ACDE123"};
        CHECK_FALSE(gynx::valid_peptide(s1));
        // with space
        gynx::sq_gen<T> s2{"ACDE F"};
        CHECK_FALSE(gynx::valid_peptide(s2));
        // with newline
        gynx::sq_gen<T> s3{"ACDEF\n"};
        CHECK_FALSE(gynx::valid_peptide(s3));
        // with invalid special character
        gynx::sq_gen<T> s4{"ACDE-F"};
        CHECK_FALSE(gynx::valid_peptide(s4));
    }

    SECTION( "empty sequence" )
    {   gynx::sq_gen<T> s{""};
        CHECK(gynx::valid(s));
    }

// -- iterator-based validation ------------------------------------------------

    SECTION( "validation with iterators" )
    {   gynx::sq_gen<T> s{"ACGTACGT"};
        // full range
        CHECK(gynx::valid(s.begin(), s.end(), true));
        CHECK(gynx::valid_nucleotide(s.begin(), s.end()));
        // partial range
        CHECK(gynx::valid_nucleotide(s.begin(), s.begin() + 4));
        // substring
        auto sub = s(0, 4);
        CHECK(gynx::valid_nucleotide(sub));
    }

    SECTION( "validation with execution policy" )
    {   gynx::sq_gen<T> s;
        s.load(SAMPLE_GENOME, 0);
        CHECK(s.size() > 0);
        CHECK(gynx::valid_nucleotide(s));
        CHECK(gynx::valid_nucleotide(gynx::execution::seq, s));
        CHECK(gynx::valid_nucleotide(gynx::execution::unseq, s));
        CHECK(gynx::valid_nucleotide(gynx::execution::par, s));
        CHECK(gynx::valid_nucleotide(gynx::execution::par_unseq, s));

        // introduce an invalid character and ensure policy overload detects it
        s[2] = 'Z';
        CHECK_FALSE(gynx::valid_nucleotide(gynx::execution::seq, s));
        CHECK_FALSE(gynx::valid_nucleotide(gynx::execution::unseq, s));
        CHECK_FALSE(gynx::valid_nucleotide(gynx::execution::par, s));
        CHECK_FALSE(gynx::valid_nucleotide(gynx::execution::par_unseq, s));

        // peptide validation remains true with Z
        CHECK(gynx::valid_peptide(gynx::execution::par, s));
    }

    SECTION( "validation with sq_view" )
    {   gynx::sq_gen<T> s{"ACGTACGT"};
        gynx::sq_view_gen<T> view{s};
        CHECK(gynx::valid_nucleotide(view));
        CHECK(gynx::valid_nucleotide(view.begin(), view.end()));
    }

// -- compile-time validation --------------------------------------------------

    SECTION( "constexpr validation" )
    {   // These should compile if the function is constexpr
        constexpr std::array<char, 4> arr{'A', 'C', 'G', 'T'};
        constexpr bool result = gynx::valid_nucleotide(arr.begin(), arr.end());
        CHECK(result);
    }

// -- cross-validation tests ---------------------------------------------------

    SECTION( "sequences valid for one type but not another" )
    {   // Some characters valid for peptides but not nucleotides
        gynx::sq_gen<T> s1{"EFIKLPQVWY"};
        CHECK(gynx::valid_peptide(s1));
        CHECK_FALSE(gynx::valid_nucleotide(s1));

        // All nucleotides are also valid peptides (overlap in alphabet)
        gynx::sq_gen<T> s2{"ACGT"};
        CHECK(gynx::valid_nucleotide(s2));
        CHECK(gynx::valid_peptide(s2)); // A, C, G, T are also amino acids
    }

// -- range compatibility tests ------------------------------------------------

    SECTION( "view ranges compatibility" )
    {   gynx::sq_gen<T> s{"ACGTACGTKQ"};
        CHECK(gynx::valid_nucleotide(s | std::views::take(8)));
        gynx::sq_view_gen<T> v{s};
        v.remove_suffix(2); // drop 'KQ'
        CHECK(gynx::valid_nucleotide(v));
        CHECK(gynx::valid_nucleotide(v.begin(), v.end()));
        auto result = s
        |   std::views::take(8)
        |   std::views::transform([](auto c) { return std::tolower(c); })
        |   std::views::reverse;
        CHECK(gynx::valid_nucleotide(result));
    }
}

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE( "gynx::valid::device", "[algorithm][valid][cuda]", thrust::device_vector<char>, thrust::universal_vector<char>)
{   typedef TestType T;

    gynx::sq_gen<T> s;
    s.load(SAMPLE_GENOME, 0);
    CHECK(s.size() > 0);

    SECTION( "device vector" )
    {   CHECK(gynx::valid_nucleotide(thrust::cuda::par, s));
        s[2] = 'Z';
        CHECK_FALSE(gynx::valid_nucleotide(thrust::cuda::par, s));
    }

    SECTION( "cuda stream" )
    {   cudaStream_t streamA;
        cudaStreamCreate(&streamA);
        CHECK(gynx::valid_nucleotide(thrust::cuda::par.on(streamA), s));
        cudaStreamSynchronize(streamA);
        s[2] = 'Z';
        CHECK_FALSE(gynx::valid_nucleotide(thrust::cuda::par_nosync.on(streamA), s));
        cudaStreamSynchronize(streamA);
        cudaStreamDestroy(streamA);
    }
}
#endif //__CUDACC__

#if defined(__HIPCC__)
TEMPLATE_TEST_CASE( "gynx::valid::device", "[algorithm][valid][rocm]", thrust::device_vector<char>, thrust::universal_vector<char>, gynx::unified_vector<char>)
{   typedef TestType T;

    gynx::sq_gen<T> s;
    s.load(SAMPLE_GENOME, 0);
    CHECK(s.size() > 0);

    SECTION( "device vector" )
    {   CHECK(gynx::valid_nucleotide(thrust::hip::par, s));
        s[2] = 'Z';
        CHECK_FALSE(gynx::valid_nucleotide(thrust::hip::par, s));
    }

    SECTION( "cuda stream" )
    {   hipStream_t streamA;
        hipStreamCreate(&streamA);
        CHECK(gynx::valid_nucleotide(thrust::hip::par.on(streamA), s));
        hipStreamSynchronize(streamA);
        s[2] = 'Z';
        CHECK_FALSE(gynx::valid_nucleotide(thrust::hip::par_nosync.on(streamA), s));
        hipStreamSynchronize(streamA);
        hipStreamDestroy(streamA);
    }
}
#endif //__HIPCC__

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE( "gynx::random", "[algorithm][random][cuda]", std::vector<char>, thrust::host_vector<char>)
#elif defined(__HIPCC__)
TEMPLATE_TEST_CASE( "gynx::random", "[algorithm][random][rocm]", std::vector<char>, thrust::host_vector<char>)
#else
TEMPLATE_TEST_CASE( "gynx::random", "[algorithm][random]", std::vector<char>)
#endif //__CUDACC__ || __HIPCC__
{   typedef TestType T;
    gynx::sq_gen<T> s(20);
    const auto N{10'000};

    SECTION( "random nucleotide sequence" )
    {   gynx::rand(s.begin(), 20, "ACGT", seed_pi);
        CHECK(gynx::valid_nucleotide(s));
        CHECK(s == "TTCGGCCGTCGTTAAACACG");
        auto t = gynx::random::dna<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "random sequence with execution policy" )
    {   gynx::sq_gen<T> r(N);
        auto t = gynx::random::dna<decltype(s)>(N, seed_pi);
        CHECK(N == t.size());
        gynx::rand(gynx::execution::seq, r.begin(), N, "ACGT", seed_pi);
        CHECK(t == r);
        gynx::rand(gynx::execution::unseq, r.begin(), N, "ACGT", seed_pi);
        CHECK(t == r);
        gynx::rand(gynx::execution::par, r.begin(), N, "ACGT", seed_pi);
        CHECK(t == r);
        gynx::rand(gynx::execution::par_unseq, r.begin(), N, "ACGT", seed_pi);
        CHECK(t == r);
    }

    // SECTION( "random nucleotide sequence with weights" )
    // {   gynx::rand(s.begin(), 20, "ACGT", {35, 15, 15, 35}, seed_pi);
    //     CHECK(gynx::valid_nucleotide(s));
    //     // CAPTURE(s);
    //     CHECK(s == "TTCTTAAGTCTTTAAACACG");
    //     auto t = gynx::random::dna<decltype(s)>(20, 30, seed_pi);
    //     t[2] = 'C';
    //     CHECK(s == t);
    // }
}

#if defined(__CUDACC__)
TEMPLATE_TEST_CASE( "gynx::random::device", "[algorithm][random][cuda]", thrust::device_vector<char>, thrust::universal_vector<char> )
{   typedef TestType T;
    gynx::sq_gen<T> s(20);
    const auto N{10'000};

    SECTION( "device vector" )
    {   gynx::rand(thrust::cuda::par, s.begin(), 20, "ACGT", seed_pi);
        CHECK(gynx::valid_nucleotide(thrust::cuda::par, s));
        CHECK(s == "TTCGGCCGTCGTTAAACACG");
        auto t = gynx::random::dna<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "cuda stream" )
    {   auto r = gynx::random::dna<decltype(s)>(N, seed_pi);
        gynx::sq_gen<T> t(N);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        gynx::rand(thrust::cuda::par.on(stream), t.begin(), N, "ACGT", seed_pi);
        CHECK(gynx::valid_nucleotide(thrust::cuda::par.on(stream), t));
        CHECK(r == t);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
}
#endif //__CUDACC__

#if defined(__HIPCC__)
TEMPLATE_TEST_CASE( "gynx::random::device", "[algorithm][random][rocm]", thrust::device_vector<char>, thrust::universal_vector<char>, gynx::unified_vector<char> )
{   typedef TestType T;
    gynx::sq_gen<T> s(20);
    const auto N{10'000};

    SECTION( "device vector" )
    {   gynx::rand(thrust::hip::par, s.begin(), 20, "ACGT", seed_pi);
        CHECK(gynx::valid_nucleotide(thrust::hip::par, s));
        CHECK(s == "TTCGGCCGTCGTTAAACACG");
        auto t = gynx::random::dna<decltype(s)>(20, seed_pi);
        CHECK(s == t);
    }

    SECTION( "hip stream" )
    {   auto r = gynx::random::dna<decltype(s)>(N, seed_pi);
        gynx::sq_gen<T> t(N);
        hipStream_t stream;
        hipStreamCreate(&stream);
        gynx::rand(thrust::hip::par.on(stream), t.begin(), N, "ACGT", seed_pi);
        CHECK(gynx::valid_nucleotide(thrust::hip::par.on(stream), t));
        CHECK(r == t);
        hipStreamSynchronize(stream);
        hipStreamDestroy(stream);
    }
}
#endif //__HIPCC__