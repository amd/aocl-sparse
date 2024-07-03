/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#ifndef AOCLSPARSE_ITSOL_LIST_OPTIONS_HPP_
#define AOCLSPARSE_ITSOL_LIST_OPTIONS_HPP_

#include "aoclsparse_itsol_data.hpp"
#include "aoclsparse_itsol_options.hpp"
#include "aoclsparse_utils.hpp"

#include <cmath>
#include <vector>

/*
 * Option registration
 * ===================
 *
 * Options are first defined using four option classes (Int, Real<T>(float|double), string or Boolean),
 * next are registered using the Register() method.
 *
 * Integer options
 * using OptionInt
 *  const  string          name,     //  Option name
 *  const  aoclsparse_int  id,       //  Reserved for futureuse
 *  const  string          desc,     //  Short derscription of option
 *  const  bool            hidden,   //  Reserved for future use
 *  const  aoclsparse_int  pgrp,     //  Reserved for future use
 *  const  aoclsparse_int  lower,    //  lower bound for the value of the option
 *  const  lbound_t        lbound,   //  lower bound type, either `<` or `<=` (greaterthan or greaterequal)
 *  const  aoclsparse_int  upper,    //  upper bound for the value of the option
 *  const  ubound_t        ubound,   //  upper bound type, either `>` or `>=` (lessthan or lessequal)
 *  const  aoclsparse_int  vdefault  //  default value
 *  const  aoclsparse_int  setby     //  how last modified this option /defauld/user/solver/
 *
 * Real options
 * use OptionReal<T>(float|double)
 *  const  string          name,     //  Option name
 *  const  aoclsparse_int  id,       //  Reserved for futureuse
 *  const  string          desc,     //  Short derscription of option
 *  const  bool            hidden,   //  Reserved for future use
 *  const  aoclsparse_int  pgrp,     //  Reserved for future use
 *  const  T               lower,    //  lower bound for the value of the option
 *  const  lbound_t        lbound,   //  lower bound type, either `<` or `<=` (greaterthan or greaterequal)
 *  const  T               upper,    //  upper bound for the value of the option
 *  const  ubound_t        ubound,   //  upper bound type, either `>` or `>=` (lessthan or lessequal)
 *  const  T               vdefault  //  default value
 *  const  aoclsparse_int  setby     //  how last modified this option /defauld/user/solver/
 *
 * Bool options
 * use OptionBool
 *  const  string          name,     //  Option name
 *  const  aoclsparse_int  id,       //  Reserved for futureuse
 *  const  string          desc,     //  Short derscription of option
 *  const  bool            hidden,   //  Reserved for future use
 *  const  aoclsparse_int  pgrp,     //  Reserved for future use
 *  const  T               vdefault  //  default value
 *  const  aoclsparse_int  setby     //  how last modified this option /defauld/user/solver/
 *
 * String options
 * use OptionString
 *  const  string          name,     //  Option name
 *  const  aoclsparse_int  id,       //  Reserved for future use
 *  const  string          desc,     //  Short derscription of option
 *  const  bool            hidden,   //  Reserved for future use
 *  const  aoclsparse_int  pgrp,     //  Reserved for future use
 *  const  map<string, aoclsparse_int> labels,  // map {"OptionName", OptionValue}
 *  const  string          vdefault  //  default value
 *  const  aoclsparse_int  setby     //  how last modified this option /defauld/user/solver/
 *
 *  Doxygen documentation for all the options is generated using the
 *  OptionRegistry.PrintDetails() method (see aoclsparse_itsol_options.hpp).
 */

template <typename T>
int register_options(aoclsparse_options::OptionRegistry<T> &opts)
{
    using namespace aoclsparse_options;

    {
        OptionString o("iterative method",
                       3,
                       "Choose solver to use",
                       false,
                       0,
                       {{"CG", solver_cg},
                        {"PCG", solver_cg},
                        {"GMRES", solver_gmres},
                        {"GM RES", solver_gmres}},
                       "CG");
        if(opts.Register(o))
            return 2;
    }
    {
        OptionInt o("cg iteration limit",
                    2,
                    "Set CG iteration limit",
                    false,
                    1,
                    1,
                    greaterequal,
                    1,
                    p_inf,
                    500);
        if(opts.Register(o))
            return 2;
    }
    {
        OptionReal<tolerance_t<T>> o("cg rel tolerance",
                                     4,
                                     "Set relative convergence tolerance for cg method",
                                     false,
                                     1,
                                     0.0,
                                     greaterequal,
                                     1.0,
                                     p_inf,
                                     expected_precision((tolerance_t<T>)2.0));
        if(opts.Register(o))
            return 2;
    }
    {
        OptionReal<tolerance_t<T>> o("cg abs tolerance",
                                     1,
                                     "Set absolute convergence tolerance for cg method",
                                     false,
                                     1,
                                     0.0,
                                     greaterequal,
                                     1.0,
                                     p_inf,
                                     expected_precision<tolerance_t<T>>());
        if(opts.Register(o))
            return 2;
    }
    {
        OptionString o("cg preconditioner",
                       3,
                       "Choose preconditioner to use with cg method",
                       false,
                       0,
                       {{"None", 0},
                        {"User", 1},
                        // {"Jacobi", 2}, not yet implemented
                        {"GS", 3},
                        {"SymGS", 3},
                        {"SGS", 3}},
                       "None");
        if(opts.Register(o))
            return 2;
    }
    /*************GMRES OPTIONS******************/
    {
        OptionInt o("gmres iteration limit",
                    2,
                    "Set GMRES iteration limit",
                    false,
                    1,
                    1,
                    greaterequal,
                    1,
                    p_inf,
                    150);
        if(opts.Register(o))
            return 2;
    }
    {
        OptionReal<tolerance_t<T>> o("gmres rel tolerance",
                                     4,
                                     "Set relative convergence tolerance for gmres method",
                                     false,
                                     1,
                                     0.0,
                                     greaterequal,
                                     1.0,
                                     p_inf,
                                     expected_precision((tolerance_t<T>)2.0));
        if(opts.Register(o))
            return 2;
    }
    {
        OptionReal<tolerance_t<T>> o("gmres abs tolerance",
                                     1,
                                     "Set absolute convergence tolerance for gmres method",
                                     false,
                                     1,
                                     0.0,
                                     greaterequal,
                                     1.0,
                                     p_inf,
                                     expected_precision<tolerance_t<T>>());
        if(opts.Register(o))
            return 2;
    }
    {
        OptionString o("gmres preconditioner",
                       3,
                       "Choose preconditioner to use with gmres method",
                       false,
                       0,
                       {{"None", 0}, {"User", 1}, {"ILU0", 2}},
                       "None");
        if(opts.Register(o))
            return 2;
    }
    {
        OptionInt o("gmres restart iterations",
                    2,
                    "Set GMRES restart iterations",
                    false,
                    1,
                    1,
                    greaterequal,
                    1,
                    p_inf,
                    20);
        if(opts.Register(o))
            return 2;
    }
    return 0;
}

#endif
