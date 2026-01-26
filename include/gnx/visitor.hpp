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
#ifndef _GNX_VISITOR_HPP_
#define _GNX_VISITOR_HPP_

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <typeindex>
#include <functional>
#include <any>

namespace gnx {

// -- make_td_print_visitor() --------------------------------------------------

template<class T, class F>
inline
std::pair<const std::type_index, std::function<void(std::ostream&, const std::any&)>>
    make_td_print_visitor(const F& f)
{   return
    {   std::type_index(typeid(T)),
        [g = f](std::ostream& os, const std::any& a)
        {   if constexpr (std::is_void_v<T>) g(os);
            else g(os, std::any_cast<const T&>(a));
        }
    };
}

// -- td_print_visitor ---------------------------------------------------------

static std::unordered_map<std::type_index, std::function<void(std::ostream&, const std::any&)>>
    td_print_visitor
{   make_td_print_visitor<void>
    (   [] (std::ostream& os)
        { os << std::quoted("void", '|') << "{}"; }
    )
,   make_td_print_visitor<bool>
    (   [](std::ostream& os, bool x)
        { os << std::quoted("bool", '|') << x; }
    )
,   make_td_print_visitor<int>
    (   [](std::ostream& os, int x)
        { os << std::quoted("int", '|') << x; }
    )
,   make_td_print_visitor<unsigned>
    (   [](std::ostream& os, unsigned x)
        { os << std::quoted("unsigned", '|') << x; }
    )
,   make_td_print_visitor<float>
    (   [](std::ostream& os, float x)
        { os << std::quoted("float", '|') << x; }
    )
,   make_td_print_visitor<double>
    (   [](std::ostream& os, double x)
        { os << std::quoted("double", '|') << x; }
    )
,   make_td_print_visitor<std::string>
    (   [] (std::ostream& os, std::string s)
        { os << std::quoted("string", '|') << std::quoted(s); }
    )
,   make_td_print_visitor<std::vector<int>>
    (   [] (std::ostream& os, std::vector<int> v)
        {   os << std::quoted("std::vector<int>", '|') << '{';
            for (auto i : v)
                os << i << ',';
            os << '}';
        }
    )
    // ... add more handlers here ...
};

// -- register_td_print_visitor ------------------------------------------------

template<class T, class F>
inline void register_td_print_visitor(const F& f)
{   td_print_visitor.insert(make_td_print_visitor<T>(f));   }

// -- td_scan_visitor ----------------------------------------------------------

static std::unordered_map<std::string, std::function<void(std::istream&, std::any&)>>
    td_scan_visitor
{   {   "void"
    ,   [] (std::istream& is, std::any& a)
        { is.ignore(2) ; a = {}; }
    }
    ,
    {   "bool"
    ,   [] (std::istream& is, std::any& a)
        { bool x; is >> x; a = x; }
    }
    ,
    {   "int"
    ,   [] (std::istream& is, std::any& a)
        { int x; is >> x; a = x; }
    }
    ,
    {   "unsigned"
    ,   [] (std::istream& is, std::any& a)
        { unsigned x; is >> x; a = x; }
    }
    ,
    {   "float"
    ,   [] (std::istream& is, std::any& a)
        { float x; is >> x; a = x; }
    }
    ,
    {   "double"
    ,   [] (std::istream& is, std::any& a)
        { double x; is >> x; a = x; }
    }
    ,
    {   "string"
    ,   [] (std::istream& is, std::any& a)
        { std::string s; is >> std::quoted(s); a = s; }
    }
    ,
    {   "std::vector<int>"
    ,   [] (std::istream& is, std::any& a)
        {   std::vector<int> v;
            int i;
            is.ignore();
            while (is.peek() != '}')
            {   is >> i;
                is.ignore();
                v.push_back(i);
            }
            is.ignore();
            a = v;
        }
    }
    ,
    {   "UNREGISTERED TYPE"
    ,   [] (std::istream& is, std::any& a)
        { is.ignore(2) ; a = {}; }
    }
    // ... add more handlers here ...
};

// -- register_td_scan_visitor ------------------------------------------------

template<class F>
inline void register_td_scan_visitor(std::string type, const F& f)
{   td_scan_visitor.insert(std::make_pair(type, f));   }


}   // end gnx namespace

#endif  //_GNX_VISITOR_HPP_