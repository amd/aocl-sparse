/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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
// -- Type and macro definitions -----------------------------------------------

#if defined(AOCLSPARSE_DISABLE_SYSTEM)

// This branch defines a pthread-like API, aoclsparse_pthread_*(), and implements it
// in terms of "dummy" code that doesn't depend on POSIX threads or any other
// threading mechanism.
// NOTE: THIS CODE DOES NOT IMPLEMENT THREADING AND IS NOT THREAD-SAFE!

// -- pthread types --

typedef aoclsparse_int aoclsparse_pthread_mutex_t;
typedef aoclsparse_int aoclsparse_pthread_once_t;

// -- pthreads macros --

#define AOCLSPARSE_PTHREAD_MUTEX_INITIALIZER 0
#define AOCLSPARSE_PTHREAD_ONCE_INIT 0

#elif defined(_MSC_VER) // !defined(AOCLSPARSE_DISABLE_SYSTEM)

#include <windows.h>
// This branch defines a pthread-like API, aoclsparse_pthread_*(), and implements it
// in terms of Windows API calls.

// -- pthread types --
typedef SRWLOCK   aoclsparse_pthread_mutex_t;
typedef INIT_ONCE aoclsparse_pthread_once_t;

// -- pthreads macros --

#define AOCLSPARSE_PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define AOCLSPARSE_PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

#else // !defined(AOCLSPARSE_DISABLE_SYSTEM) && !defined(_MSC_VER)

#include <pthread.h>

// This branch defines a pthreads-like API, aoclsparse_pthreads_*(), and implements it
// in terms of the corresponding pthreads_*() types, macros, and function calls.

// -- pthread types --

typedef pthread_mutex_t aoclsparse_pthread_mutex_t;
typedef pthread_once_t  aoclsparse_pthread_once_t;

// -- pthreads macros --

#define AOCLSPARSE_PTHREAD_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#define AOCLSPARSE_PTHREAD_ONCE_INIT PTHREAD_ONCE_INIT

#endif

// -- Function definitions -----------------------------------------------------

// -- pthread_mutex_*() --

aoclsparse_int aoclsparse_pthread_mutex_lock(aoclsparse_pthread_mutex_t *mutex);

aoclsparse_int aoclsparse_pthread_mutex_unlock(aoclsparse_pthread_mutex_t *mutex);

// -- pthread_once() --

void aoclsparse_pthread_once(aoclsparse_pthread_once_t *once, void (*init)(void));

/******************************************************************************************
 * \brief aoclsparse_context is a structure holding the number of threads, ISA information
 * It gets initialised by aoclsparse_init_once().
 *****************************************************************************************/
typedef struct _aoclsparse_context
{
    // num of threads
    aoclsparse_int num_threads = 0;
    bool           is_avx512   = false;
} aoclsparse_context;

extern aoclsparse_context global_context;

/*! \ingroup aux_module
 *  \brief Initialise number of threads from environment variables
 *
 *  \retval none.
 */
void aoclsparse_init_once();

#endif // AOCLSPARSE_THREAD_H
