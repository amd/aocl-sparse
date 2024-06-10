/* ************************************************************************
 * Copyright (c) 2021-2024 Advanced Micro Devices, Inc.
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

#include "alci/cxx/cpu.hh"

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
    // unsigned int -> __builtin_popcount(unsigned int)
    enum class archs : unsigned int
    {
        ALL     = ~0U,
        UNKNOWN = 0U,
        ZEN     = 1U << 0U,
        ZEN2    = 1U << 1U,
        ZEN3    = 1U << 2U,
        ZEN4    = 1U << 3U,
        ZEN123  = ZEN | ZEN2 | ZEN3,
        ZENS    = ZEN | ZEN2 | ZEN3 | ZEN4
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
                r_val = (aoclsparse_int)strtol(str, NULL, 10);
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

        // num of threads
        aoclsparse_int num_threads = 1;

        // ALCI CPU object
        std::unique_ptr<alci::Cpu> Cpu = nullptr;

        // Architecture reported by ALCI
        alci::Uarch Uarch;

        //AOCLSPARSE local arch info container
        archs lib_local_arch;

        bool cpuflags[static_cast<int>(context_isa_t::LENGTH)];

        // ISA path preference, set by AOCL_ENABLE_INSTRUCTIONS
        context_isa_t global_isa_hint = context_isa_t::UNSET;

        // Ensure direct calls to constructor is not possible
        context()
        {
            /*
            * Read from OpenMP params and sparse ENVs for threading, only if OMP is enabled.
            * Since the library relies on OpenMP for multithreading OpenMP variables get the
            * maximum priority.
            */
#ifdef _OPENMP
            /*
            * Read the sparse specific thread-count environment
            * variable to initialize the global object.
            */
            aoclsparse_int env_num_threads = this->get_thread_from_env();

            // Set the num threads value set in aoclsparse_num_threads
            this->num_threads = env_num_threads;

            // TODO - add code sections to handle nested parallelism scenarios and OMP ICVs
#endif
            this->Cpu   = std::make_unique<alci::Cpu>();
            this->Uarch = this->Cpu->getUarch();

            // Check for the list of flags supported
            // Note: Utils does not support BF16 flag lookup
            this->cpuflags[static_cast<int>(context_isa_t::AVX2)]
                = this->Cpu->isAvailable(alci::ALC_E_FLAG_AVX2);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512F)]
                = this->Cpu->isAvailable(alci::ALC_E_FLAG_AVX512F);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512DQ)]
                = this->Cpu->isAvailable(alci::ALC_E_FLAG_AVX512DQ);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512VL)]
                = this->Cpu->isAvailable(alci::ALC_E_FLAG_AVX512VL);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512IFMA)]
                = this->Cpu->isAvailable(alci::ALC_E_FLAG_AVX512_IFMA);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512CD)]
                = this->Cpu->isAvailable(alci::ALC_E_FLAG_AVX512CD);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512BW)]
                = this->Cpu->isAvailable(alci::ALC_E_FLAG_AVX512BW);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512_VBMI)]
                = this->Cpu->isAvailable(alci::ALC_E_FLAG_AVX512_VBMI);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512_VNNI)]
                = this->Cpu->isAvailable(alci::ALC_E_FLAG_AVX512_VNNI);

            this->cpuflags[static_cast<int>(context_isa_t::AVX512_VPOPCNTDQ)]
                = this->Cpu->isAvailable(alci::ALC_E_FLAG_AVX512_VPOPCNTDQ);

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

            switch(this->Uarch)
            {
            case alci::Uarch::eZen:
                lib_local_arch = archs::ZEN;
            case alci::Uarch::eZen2:
                lib_local_arch = archs::ZEN2;
            case alci::Uarch::eZen3:
                lib_local_arch = archs::ZEN3;
            case alci::Uarch::eZen4:
                lib_local_arch = archs::ZEN4;
            // Todo: Add support for newer and older AMD architectures
            default:
                lib_local_arch = archs::UNKNOWN;
            }
        }

        aoclsparse_int get_thread_from_env()
        {
            aoclsparse_int nt = 1;
#ifdef _OPENMP
            // Try to read AOCLSPARSE_NUM_THREADS with fallback value as number of processors.
            nt = env_get_var("AOCLSPARSE_NUM_THREADS", (aoclsparse_int)1);
#endif
            // If AOCLSPARSE_NUM_THREADS was not set, return number of processors.
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
        // Usage   - Pass the isa as part of the template parameter
        // Example - bool check = supports<context_isa_t::AVX512F>();
        template <context_isa_t... isa>
        bool supports()
        {
            return (... && this->cpuflags[static_cast<int>(isa)]);
        }

        // Returns the number of threads set
        aoclsparse_int get_num_threads(void)
        {
            return this->num_threads;
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
            return (this->current_hint == this->old_hint);
        }

        // Delete the copy constructor of the context class
        isa_hint(isa_hint &t) = delete;

        // Delete the assignment operator of the context class
        void operator=(const isa_hint &) = delete;
    };
}

extern thread_local aoclsparse::isa_hint tl_isa_hint;

#endif // AOCLSPARSE_THREAD_H
