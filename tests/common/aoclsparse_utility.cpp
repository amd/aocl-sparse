/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse_random.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"

#include <cstdlib>
#include <cstring>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <time.h>
#endif

// Random number generator
// Note: We do not use random_device to initialize the RNG, because we want
// repeatability in case of test failure. TODO: Add seed as an optional CLI
// argument, and print the seed on output, to ensure repeatability.
aoclsparse_rng_t aoclsparse_rng(69069);
aoclsparse_rng_t aoclsparse_seed(aoclsparse_rng);
/* ============================================================================================ */
// Return path of this executable
std::string aoclsparse_exepath()
{
    std::string pathstr;
#if defined(_WIN32) || defined(_WIN64)
    char *path = (char *)malloc(MAX_PATH * sizeof(char));
    char *pgmptr;
    if(_get_pgmptr(&pgmptr) == 0)
    {
        strcpy(path, pgmptr);
        pgmptr = NULL;
    }
    else
    {
        free(path);
    }

    if(path)
    {
        char *p = strrchr(path, '\\');
        if(p)
        {
            p[1]    = 0;
            pathstr = path;
        }
        free(path);
    }
#else
    char *path = realpath("/proc/self/exe", 0);
    if(path)
    {
        char *p = strrchr(path, '/');
        if(p)
        {
            p[1]    = 0;
            pathstr = path;
        }
        free(path);
    }
#endif
    return pathstr;
}

/* ============================================================================================ */
/*  timing:*/

/*! \brief  CPU Timer(in second) return wall time */
static double gtod_ref_time_sec = 0.0;

double aoclsparse_clock(void)
{
    return aoclsparse_clock_helper();
}

double aoclsparse_clock_min_diff(double time_min, double time_start)
{
    double time_min_prev;
    double time_diff;

    // Save the old value.
    time_min_prev = time_min;

    time_diff = aoclsparse_clock() - time_start;

    time_min = (std::min)(time_min, time_diff);

    // Assume that anything:
    // - under or equal to zero,
    // - under a nanosecond
    // is actually garbled due to the clocks being taken too closely together.
    if(time_min <= 0.0)
        time_min = time_min_prev;
    else if(time_min < 1.0e-9)
        time_min = time_min_prev;

    return time_min;
}

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) \
    || defined(__WIN64) && !defined(__CYGWIN__)
// --- Begin Windows build definitions -----------------------------------------

double aoclsparse_clock_helper()
{
    LARGE_INTEGER clock_freq = {0};
    LARGE_INTEGER clock_val;
    BOOL          r_val;

    r_val = QueryPerformanceFrequency(&clock_freq);

    if(r_val == 0)
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
    }

    r_val = QueryPerformanceCounter(&clock_val);

    if(r_val == 0)
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
    }

    return ((double)clock_val.QuadPart / (double)clock_freq.QuadPart);
}

// --- End Windows build definitions -------------------------------------------
#else
// --- Begin Linux build definitions -------------------------------------------

double aoclsparse_clock_helper()
{
    double the_time, norm_sec;
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    if(gtod_ref_time_sec == 0.0)
        gtod_ref_time_sec = (double)ts.tv_sec;

    norm_sec = (double)ts.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + ts.tv_nsec * 1.0e-9;

    return the_time;
}

// --- End Linux build definitions ---------------------------------------------
#endif
