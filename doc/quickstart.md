---
title: Start using Gnx
subject: Gnx Quickstart
short_title: Get Started
description: Using Gnx in a nutshell.
kernelspec:
  name: xcpp20-openmp
  display_name: C++20-OpenMP
---

# Start using Gnx

---

```{code-cell} cpp
#pragma cling add_include_path("/home/armin/src/gnx/include")
#pragma cling add_include_path("/home/armin/src/g3p/include")

#include <fstream>

#include <g3p/gnuplot>
#include <gnx/sq.hpp>
#include <gnx/sq_view.hpp>
#include <gnx/io/fastaqz.hpp>
#include <gnx/algorithms/valid.hpp>
```

Making a biological sequence in Gnx is easy:

```{code-cell} cpp
gnx::sq s{"ACGT"};
```
Or even easier like this:
```{code-cell} cpp
auto t = "ACGT"_sq;
(s == t)
```
It's also easy to load from compressed/uncompressed fasta/fastq files. First let's download a sample file:
```{code-cell} cpp
!wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/204/255/GCF_000204255.1_ASM20425v1/GCF_000204255.1_ASM20425v1_genomic.fna.gz
```
+++
```{code-cell} cpp
gnx::sq plasmid;
plasmid.load("GCF_000204255.1_ASM20425v1_genomic.fna.gz", "NC_017288.1");
(7553 == std::size(plasmid))
```
+++
```{code-cell} cpp
gnx::sq plot = plasmid(0, 500);
// g3p::gnuplot gp("test.log");
g3p::gnuplot gp;

gp
( "set pm3d at b" )
( "set view map" )
( "set palette gray negative" )
( "set size square" )
// ( "unset xtics" )
// ( "unset ytics" )
// ( "unset colorbox" )
;
gp("splot '-' using 1:2:3 with pm3d");
for (size_t i = 0; i < std::size(plot); i += 10)
{
    for (size_t j = 0; j < std::size(plot); j += 10)
        gp << i << j << (plot[i] == plot[j] ? 1 : 0) << "\n";
    gp << "\n";
}
gp << g3p::end << g3p::endl
```
+++
```{code-cell} cpp
gp  ( "set cbrange [0:10]" )
    ( "splot '-' using 1:2:3 with pm3d" )
;
for (size_t i = 0; i < std::size(plot); ++i)
{
    for (size_t j = 0; j < std::size(plot); ++j)
        if (plot[i] == plot[j])
            gp << i << j << 10 << "\n";
    gp << "\n";
}
gp << g3p::end << g3p::endl
```
+++
```{code-cell} cpp
auto plot = "AGCCAACCACCCCCG"_sq;
auto gnx_x = "CCAATATTATC"_sq;
auto gnx_y = "CCTATGTTACT"_sq;

gp
( "set size square" )
( "unset key")
( "set xrange [0:14]" )
( "set yrange [0:14]" )
;
// gp("plot '-' u 1:2 w dot lc \"black\"");
gp("plot '-' u 1:2:3:xticlabels(4):yticlabels(5) w circles lc \"black\" fill solid noborder");
for (size_t i = 0; i < std::size(plot); ++i)
{
    for (size_t j = 0; j < std::size(plot); ++j)
        if (plot[i] == plot[j])
            gp("%d %d %f %c %c", i, j, 0.5, plot[i], plot[j]);
    gp << "\n";
}
gp << g3p::end << g3p::endl;
gp
```
+++
```{code-cell} cpp
plasmid["int_test"] = 19;
plasmid["float_test"] = 3.14f;
// plasmid["wrong"] = std::vector<int>{1, 2, 3, 4};
std::cout << plasmid;
```
+++
```{code-cell} cpp
gnx::sq s{"ACGT"};
s["test"] = 33;
// std::cout << s;
s
```
+++
```{code-cell} cpp
std::stringstream ss;
ss << s;
gnx::sq t;
ss >> t;
(s == t)
```
+++
```{code-cell} cpp
std::cout << t;
```
++
## Non-owning sequence views

`gnx::sq_view` is a lightweight, non-owning view over a `gnx::sq` (or any compatible container), similar to `std::string_view`. It avoids copying while letting you slice and compare.

```{code-cell} cpp
gnx::sq s{"ACGT"};
gnx::sq_view v{s};         // view over s, no copy

// iteration
for (auto it = v.begin(); it != v.end(); ++it) { /* use *it */ }

// slicing without allocation
auto mid = v.substr(1, 2);  // "CG"

// adjust view window
v.remove_prefix(1);          // now "CGT"
v.remove_suffix(1);          // now "CG"

// comparisons
(v == "CG");               // true
(v != s);                    // true, because s is "ACGT"
```

Use `gnx::sq_view_gen<Container>` for custom containers whose `value_type` and layout match `sq_gen`’s expectations.
+++
```{code-cell} cpp
std::stringstream ss;
ss << plasmid;
gnx::sq p;
ss >> p;
(p == plasmid)
```
+++
```{code-cell} cpp
gnx::sq s{"ACGT"};
s["test"] = 33;
s["test2"] = 1.33f; 
std::ofstream os("test.gnx");
os << s;
os.close();
```
+++
```{code-cell} cpp
gnx::sq s{"ACGT"};
s["test"] = 33;
s["test-void"] = {};
s["test-bool"] = true;
s["test-float"] = 3.14f;
s["test-vector-int"] = std::vector<int>{ 1, 2, 3, 4 };
std::cout << s;
```
+++
```{code-cell} cpp
std::stringstream ss;
ss << s;
gnx::sq t;
ss >> t;
std::cout << t;
```
### valid() algorithm
+++
```{code-cell} cpp
gnx::sq s;
s.load("/home/armin/src/gnx/build/_deps/sample_genome-src/GCF_000204255.1_ASM20425v1_genomic.fna.gz");
```
+++
```{code-cell} cpp
%%timeit
gnx::valid_nucleotide(s);
```
+++
```{code-cell} cpp
%%timeit
gnx::valid_nucleotide(gnx::execution::seq, s);
```
+++
```{code-cell} cpp
%%timeit
gnx::valid_nucleotide(gnx::execution::unseq, s);
```
+++
```{code-cell} cpp
%%timeit
gnx::valid_nucleotide(gnx::execution::par, s);
```
+++
```{code-cell} cpp
%%timeit
gnx::valid_nucleotide(gnx::execution::par_unseq, s);
```