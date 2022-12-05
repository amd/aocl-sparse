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

#include "aoclsparse.h"
#include "aoclsparse_context.h"

#include <alci/cpu_features.h>
#include <cstdlib>
#if defined(AOCLSPARSE_DISABLE_SYSTEM)

// This branch defines a pthread-like API, aoclsparse_pthread_*(), and implements it
// in terms of "dummy" code that doesn't depend on POSIX threads or any other
// threading mechanism.
// NOTE: THIS CODE DOES NOT IMPLEMENT THREADING AND IS NOT THREAD-SAFE!

aoclsparse_int aoclsparse_pthread_mutex_lock(aoclsparse_pthread_mutex_t *mutex)
{
    //return pthread_mutex_lock( mutex );
    return 0;
}

aoclsparse_int aoclsparse_pthread_mutex_unlock(aoclsparse_pthread_mutex_t *mutex)
{
    //return pthread_mutex_unlock( mutex );
    return 0;
}

// -- pthread_once() --

void aoclsparse_pthread_once(aoclsparse_pthread_once_t *once, void (*init)(void))
{
    //pthread_once( once, init );
    init();
}

#elif defined(_MSC_VER) // !defined(AOCLSPARSE_DISABLE_SYSTEM)

#include <errno.h>

// This branch defines a pthread-like API, aoclsparse_pthread_*(), and implements it
// in terms of Windows API calls.

// -- pthread_mutex_*() --

aoclsparse_int aoclsparse_pthread_mutex_lock(aoclsparse_pthread_mutex_t *mutex)
{
    AcquireSRWLockExclusive(mutex);
    return 0;
}

aoclsparse_int aoclsparse_pthread_mutex_unlock(aoclsparse_pthread_mutex_t *mutex)
{
    ReleaseSRWLockExclusive(mutex);
    return 0;
}

// -- pthread_once() --

static BOOL
    aoclsparse_init_once_wrapper(aoclsparse_pthread_once_t *once, void *param, void **context)
{
    (void)once;
    (void)context;
    typedef void (*callback)(void);
    ((callback)param)();
    return TRUE;
}

void aoclsparse_pthread_once(aoclsparse_pthread_once_t *once, void (*init)(void))
{
    InitOnceExecuteOnce(once, aoclsparse_init_once_wrapper, init, NULL);
}

#else // !defined(AOCLSPARSE_DISABLE_SYSTEM) && !defined(_MSC_VER)

// This branch defines a pthreads-like API, aoclsparse_pthreads_*(), and implements it
// in terms of the corresponding pthreads_*() types, macros, and function calls.
// This branch is compiled for Linux and other non-Windows environments where
// we assume that *some* implementation of pthreads is provided (although it
// may lack barriers--see below).

// -- pthread_mutex_*() --

aoclsparse_int aoclsparse_pthread_mutex_lock(aoclsparse_pthread_mutex_t *mutex)
{
    return pthread_mutex_lock(mutex);
}

aoclsparse_int aoclsparse_pthread_mutex_unlock(aoclsparse_pthread_mutex_t *mutex)
{
    return pthread_mutex_unlock(mutex);
}

// -- pthread_once() --

void aoclsparse_pthread_once(aoclsparse_pthread_once_t *once, void (*init)(void))
{
    pthread_once(once, init);
}

#endif // !defined(AOCLSPARSE_DISABLE_SYSTEM) && !defined(_MSC_VER)

// The global aoclsparse_context structure, which holds the global thread,ISA settings
aoclsparse_context sparse_global_context;

// A mutex to allow synchronous access to global_thread.
aoclsparse_pthread_mutex_t global_thread_mutex = AOCLSPARSE_PTHREAD_MUTEX_INITIALIZER;

/********************************************************************************
 * \brief aoclsparse_env_get_var is a function used to query the environment
 * variable and convert the string into integer and return the same
 ********************************************************************************/
aoclsparse_int aoclsparse_env_get_var(const char *env, aoclsparse_int fallback)
{
    aoclsparse_int r_val;
    char          *str;

    // Query the environment variable and store the result in str.
    str = getenv(env);

    // Set the return value based on the string obtained from getenv().
    if(str != NULL)
    {
        // If there was no error, convert the string to an integer and
        // prepare to return that integer.
        r_val = (aoclsparse_int)strtol(str, NULL, 10);
    }
    else
    {
        // If there was an error, use the "fallback" as the return value.
        r_val = fallback;
    }

    return r_val;
}

void aoclsparse_thread_init_rntm_from_env(aoclsparse_context *context)
{
    aoclsparse_int nt;

    // Try to read AOCLSPARSE_NUM_THREADS first.
    nt = aoclsparse_env_get_var("AOCLSPARSE_NUM_THREADS", -1);

    // If AOCLSPARSE_NUM_THREADS was not set, try to read OMP_NUM_THREADS.
    if(nt == -1)
        nt = aoclsparse_env_get_var("OMP_NUM_THREADS", -1);

    // If AOCLSPARSE_NUM_THREADS and OMP_NUM_THREADS was not set, set it to 1 for single threaded run.
    if(nt == -1)
        nt = 1;
    context->num_threads = nt;
}

void aoclsparse_isa_init(aoclsparse_context *context)
{
    // Check if the target supports AVX512
    if(alc_cpu_has_avx512f())
    {
        context->is_avx512 = true;
    }
}

// -----------------------------------------------------------------------------
void aoclsparse_context_init(void)
{
    // Read the environment variables and use them to initialize the
    // global runtime object.
    aoclsparse_thread_init_rntm_from_env(&sparse_global_context);

    // Read target ISA if it supports avx512
    aoclsparse_isa_init(&sparse_global_context);
}

// -----------------------------------------------------------------------------

void aoclsparse_thread_finalize(void) {}

// -----------------------------------------------------------------------------

// A pthread_once_t variable is a pthread structure used in pthread_once().
// pthread_once() is guaranteed to execute exactly once among all threads that
// pass in this control object. Thus, we need one for initialization and a
// separate one for finalization.
static aoclsparse_pthread_once_t once_init     = AOCLSPARSE_PTHREAD_ONCE_INIT;
static aoclsparse_pthread_once_t once_finalize = AOCLSPARSE_PTHREAD_ONCE_INIT;

void aoclsparse_init_once(void)
{
    aoclsparse_pthread_once(&once_init, aoclsparse_context_init);
}

void aoclsparse_finalize_once(void)
{
    aoclsparse_pthread_once(&once_finalize, aoclsparse_thread_finalize);
}

aoclsparse_int aoclsparse_thread_get_num_threads(void)
{
    // We must ensure that global_rntm has been initialized.
    aoclsparse_init_once();

    return sparse_global_context.num_threads;
}

void aoclsparse_thread_set_num_threads(aoclsparse_int n_threads)
{
    // We must ensure that global_thread has been initialized.
    aoclsparse_init_once();

    // Acquire the mutex protecting global_thread.
    aoclsparse_pthread_mutex_lock(&global_thread_mutex);

    sparse_global_context.num_threads = n_threads;

    // Release the mutex protecting global_thread.
    aoclsparse_pthread_mutex_unlock(&global_thread_mutex);
}
