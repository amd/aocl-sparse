/* ************************************************************************
 * Copyright (c) 2021-2025 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef AOCLSPARSE_PTHREAD_H
#define AOCLSPARSE_PTHREAD_H

#include "aoclsparse.h"

#include <algorithm>
#include <memory>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include "Au/Cpuid/X86Cpu.hh"
#pragma GCC diagnostic pop

namespace aoclsparse
{
    // ISA context preference
    enum class context_isa_t
    {
        UNSET            = 0, // Not set (default)
        GENERIC          = 1,
        AVX2             = 2,
        AVX512F          = 3,
        AVX512DQ         = 4,
        AVX512VL         = 5,
        AVX512IFMA       = 6,
        AVX512CD         = 7,
        AVX512BW         = 8,
        AVX512_BF16      = 9,
        AVX512_VBMI      = 10,
        AVX512_VNNI      = 11,
        AVX512_VPOPCNTDQ = 12,
        LENGTH           = 13
    };

    // Dispatch::archs (Zen architecture and group)
    // MUST BE unsigned int for __builtin_popcount(unsigned int) to work
    enum class archs : unsigned int
    {
        ALL     = ~0U,
        UNKNOWN = 0U,
        ZEN     = 1U << 0U,
        ZEN2    = 1U << 1U,
        ZEN3    = 1U << 2U,
        ZEN4    = 1U << 3U,
        ZEN5    = 1U << 4U,
        ZEN_NEW = ZEN5, // Replace with next generation
        ZEN123  = ZEN | ZEN2 | ZEN3,
        ZENS    = ZEN | ZEN2 | ZEN3 | ZEN4 | ZEN5 // Update
    };

    inline constexpr unsigned int operator|(archs a, archs b)
    {
        return static_cast<unsigned int>(a) | static_cast<unsigned int>(b);
    }
    inline constexpr unsigned int operator|(unsigned int a, archs b)
    {
        return a | static_cast<unsigned int>(b);
    }
    inline constexpr unsigned int operator&(archs a, unsigned int b)
    {
        return static_cast<unsigned int>(a) & b;
    }

    /********************************************************************************
    * \brief aoclsparse_env_get_var<T> is a function used to query the environment
    * variable and return the same. In case of int, it converts the string into
    * an integer and return the same. This function can only be used for int and string
    * type inputs.
    ********************************************************************************/
    template <typename T>
    T env_get_var(const char *env, const T fallback)
    {
        T     r_val;
        char *str;

        // Query the environment variable and store the result in str.
        str = getenv(env);
        // Set the return value based on the string obtained from getenv().
        if(str != NULL)
        {
            if constexpr(std::is_same_v<T, aoclsparse_int>)
            {
                // If there was no error, convert the char[] to an integer and
                // return that integer.
                r_val = static_cast<aoclsparse_int>(strtol(str, NULL, 10));
                return r_val;
            }
            else if constexpr(std::is_same_v<T, std::string>)
            {
                // If there was no error, convert the char[] to an std::string
                return std::string(str);
            }
        }

        // If there was an error, use the "fallback" as the return value.
        return fallback;
    }

    /******************************************************************************************
    * \brief aoclsparse_context is a class holding the number of threads, ISA information
    * It gets initialised by aoclsparse_init_once().
    *****************************************************************************************/
    class context
    {
    private:
        static context   *global_obj;
        static std::mutex global_lock;

        //AOCLSPARSE local arch info container
        archs lib_local_arch;

        bool cpuflags[static_cast<int>(context_isa_t::LENGTH)];

        // ISA path preference, set by AOCL_ENABLE_INSTRUCTIONS
        context_isa_t global_isa_hint = context_isa_t::UNSET;

        // Ensure direct calls to constructor is not possible
        context()
        {
            Au::X86Cpu Cpu   = {0};
            Au::EUarch uarch = Cpu.getUarch();

            for(int f = 0; f < static_cast<int>(context_isa_t::LENGTH); ++f)
                cpuflags[f] = false;

            // Check for the list of flags supported
            // Note: Utils does not support BF16 flag lookup
            this->cpuflags[static_cast<int>(context_isa_t::AVX2)]
                = Cpu.hasFlag(Au::ECpuidFlag::avx2);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512F)]
                = Cpu.hasFlag(Au::ECpuidFlag::avx512f);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512DQ)]
                = Cpu.hasFlag(Au::ECpuidFlag::avx512dq);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512VL)]
                = Cpu.hasFlag(Au::ECpuidFlag::avx512vl);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512IFMA)]
                = Cpu.hasFlag(Au::ECpuidFlag::avx512ifma);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512CD)]
                = Cpu.hasFlag(Au::ECpuidFlag::avx512cd);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512BW)]
                = Cpu.hasFlag(Au::ECpuidFlag::avx512bw);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512_VBMI)]
                = Cpu.hasFlag(Au::ECpuidFlag::avx512vbmi);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512_VNNI)]
                = Cpu.hasFlag(Au::ECpuidFlag::avx512_4vnniw);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512_VPOPCNTDQ)]
                = Cpu.hasFlag(Au::ECpuidFlag::avx512_vpopcntdq);

            // Check for the enviromental variable "AOCL_ENABLE_INSTRUCTIONS"
            // global_context.isa is already initialized to UNSET (default)
            this->global_isa_hint = context_isa_t::UNSET;

            std::string str;
            std::string isa_env = env_get_var("AOCL_ENABLE_INSTRUCTIONS", str);

            if(isa_env != "")
            {
                using namespace std::string_literals;
                std::string next_isa;
                transform(isa_env.begin(), isa_env.end(), isa_env.begin(), ::toupper);

                if(isa_env == "AVX512"s)
                {
                    if(this->cpuflags[static_cast<int>(context_isa_t::AVX512F)])
                        this->global_isa_hint = context_isa_t::AVX512F;
                    else
                        next_isa = "AVX2";
                }
                if(isa_env == "AVX2"s || next_isa == "AVX2"s)
                {
                    if(this->cpuflags[static_cast<int>(context_isa_t::AVX2)])
                        this->global_isa_hint = context_isa_t::AVX2;
                    else
                        next_isa = "GENERIC";
                }
                if(isa_env == "GENERIC"s || next_isa == "GENERIC"s)
                {
                    this->global_isa_hint = context_isa_t::GENERIC;
                }
            }

            switch(uarch)
            {
            case Au::EUarch::Zen:
            case Au::EUarch::ZenPlus:
                lib_local_arch = archs::ZEN;
                break;
            case Au::EUarch::Zen2:
                lib_local_arch = archs::ZEN2;
                break;
            case Au::EUarch::Zen3:
                lib_local_arch = archs::ZEN3;
                break;
            case Au::EUarch::Zen4:
                lib_local_arch = archs::ZEN4;
                break;
            case Au::EUarch::Zen5:
                lib_local_arch = archs::ZEN5;
                break;
            default:
                // Check to see if it is a new Zen model
                if(Cpu.isAMD()) // Assume new model
                {
                    lib_local_arch = archs::ZEN_NEW; // Fall-back to latest known model
                }
                else if(this->cpuflags[static_cast<int>(context_isa_t::AVX512F)]
                        && this->cpuflags[static_cast<int>(context_isa_t::AVX512DQ)]
                        && this->cpuflags[static_cast<int>(context_isa_t::AVX512VL)])
                {
                    // Map to an equivalent Zen model...
                    lib_local_arch = archs::ZEN_NEW;
                }
                else if(this->cpuflags[static_cast<int>(context_isa_t::AVX2)])
                {
                    lib_local_arch = archs::ZEN3;
                }
                else
                {
                    lib_local_arch = archs::UNKNOWN;
                }
            }
        }

        aoclsparse_int get_thread_from_env()
        {
            aoclsparse_int nt = 1;

#ifdef _OPENMP
            // Try to read AOCLSPARSE_NUM_THREADS with fallback value
            // to indicate if it has been set or not.
            nt = env_get_var("AOCLSPARSE_NUM_THREADS",
                             static_cast<aoclsparse_int>(omp_get_num_procs()));

            // When NT is set to an invalid value number (negative numbers of zero) of threads will be 1.
            if(nt <= 0)
                nt = 1;

#endif
            return nt;
        }

    protected:
        // Ensure direct calls to destructor is avoided with delete
        ~context() {}

    public:
        // Delete the copy constructor of the context class
        context(context &t) = delete;

        // Delete the assignment operator of the context class
        void operator=(const context &) = delete;

        // Function to check if an ISA is supported
        // ----------------------------------------
        //
        // Usage   - Pass the isa as a template parameter
        // Example - bool check = supports<context_isa_t::AVX512F>();
        // Note: This is for compile time checks
        template <context_isa_t... isa>
        bool supports()
        {
            return (... && this->cpuflags[static_cast<aoclsparse_int>(isa)]);
        }

        // Function to check if an ISA is supported
        // ----------------------------------------
        //
        // Usage   - Pass the isa as a function parameter
        // Example - bool check = supports(context_isa_t::AVX512F);
        // Note: This is for run time checks
        bool supports(context_isa_t isa)
        {
            // Return true for generic
            if(isa == context_isa_t::GENERIC)
                return true;

            return this->cpuflags[static_cast<aoclsparse_int>(isa)];
        }

        // Returns the number of threads that can be launched by the calling thread
        aoclsparse_int get_num_threads(void)
        {
            aoclsparse_int num_threads = 1;

            /*
            * Read from OpenMP params and sparse ENVs for threading, only if OMP is enabled.
            * If both AOCLSPARSE_NUM_THREADS and OMP_NUM_THREADS are set, the number of threads
            * used by the library will be the least of both the values.
            */
#ifdef _OPENMP
            /*
            * Read the sparse specific thread-count environment
            * variable to initialize the global object.
            * The expectation is that 'get_thread_from_env()' cannot
            * return invalid values.
            */
            aoclsparse_int env_num_threads = this->get_thread_from_env();

            aoclsparse_int max_threads  = omp_get_max_threads();
            aoclsparse_int thread_limit = omp_get_thread_limit();

            // Get the threads upper bound from OMP.
            aoclsparse_int omp_min = (max_threads > thread_limit) ? thread_limit : max_threads;

            /*
            * If the AOCLSPARSE_NUM_THREADS is set, set the number of threads to the
            * minimum of OMP variable(s) and AOCLSPARSE_NUM_THREADS.
            *
            * In certain scenarios, the value set in "AOCLSPARSE_NUM_THREADS" will be overridden
            * if any of (OMP_NUM_THREADS, OMP_THREAD_LIMIT, OMP_NESTED) are set. When the OMP variables
            * are set "AOCLSPARSE_NUM_THREADS" will only act as a thread capping/limiting mechanism
            * .i.e. num_threads will never be greater than omp_min.
            */
            num_threads = (omp_min > env_num_threads) ? env_num_threads : omp_min;

            aoclsparse_int max_levels    = omp_get_max_active_levels();
            aoclsparse_int current_level = omp_get_active_level();

            /*
            * If the number of maximum active levels variable is less than or equal
            * to the current active level then number of threads is set to one.
            */
            if(max_levels <= current_level)
            {
                num_threads = 1;
            }
#endif

            return num_threads;
        }

        // Returns the ISA hint set
        context_isa_t get_isa_hint()
        {
            return this->global_isa_hint;
        }

        // Returns the architecture of the system
        archs get_archs(void)
        {
            return this->lib_local_arch;
        }

        // Returns a reference to the global context
        static context *get_context();
    };

    /******************************************************************************************
    * \brief aoclsparse_isa_hint keep record of the ISA path preference set be the user
    * either by using the environmental variable AOCL_ENABLE_INSTRUCTIONS or by using the
    * API aoclsparse_enable_instructions()
    *****************************************************************************************/
    class isa_hint
    {
        aoclsparse::context_isa_t old_hint;
        aoclsparse::context_isa_t current_hint;

    public:
        // Constructor for the class
        isa_hint()
        {
            // Initialize isa hint with the global isa hint from the context
            current_hint = old_hint = context::get_context()->get_isa_hint();
        };

        // Get the hint
        context_isa_t get_isa_hint()
        {
            return this->current_hint;
        };

        // Set the hint
        void set_isa_hint(context_isa_t isa)
        {
            // Save the current hint
            this->old_hint = this->current_hint;

            // Rewrite the current hint
            this->current_hint = isa;
        };

        bool is_isa_updated()
        {
            return (this->current_hint != this->old_hint);
        }

        // Delete the copy constructor of the context class
        isa_hint(isa_hint &t) = delete;

        // Delete the assignment operator of the context class
        void operator=(const isa_hint &) = delete;
    };
}

extern thread_local aoclsparse::isa_hint tl_isa_hint;

#endif // AOCLSPARSE_THREAD_H
