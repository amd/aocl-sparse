#include "aoclsparse.h"
#include "functionality_tests_utils.h"
#include "aoclsparse_itsol_functions.hpp"

#include <iostream>

#define VERBOSE 1

typedef struct
{
    std::string option;
    std::string value;
} itsol_opts;

typedef struct
{
    std::string    testid = "";
    aoclsparse_int itest;
} user_data;

/* Define itsol callbacks */
template <typename T>
aoclsparse_int precond(aoclsparse_int flag, aoclsparse_int n, const T *u, T *v, void *udata)
{
    user_data *ud = (user_data *)udata;

    if(ud->testid == "positive")
    {
        // test_cg_positive was called
        if(ud->itest == 2)
        {
            // identity preconditioner
            for(aoclsparse_int i = 0; i < 8; i++)
                v[i] = u[i];
        }
    }

    return 0;
}

template <typename T>
aoclsparse_int monit(aoclsparse_int n, const T *x, const T *r, T rinfo[100], void *udata)
{
    aoclsparse_int itest;
    user_data     *ud = (user_data *)udata;

    itest = ud->itest;
    if(ud->testid == "errors")
    {
        // test_cg_errors was called
        if(rinfo[30] > 1 && itest == 1)
            // Test user stop
            return 1;
    }

    return 0;
}

template <typename T>
aoclsparse_int test_cg_error(aoclsparse_int itest, bool &pass)
{
    std::string                 testid;
    aoclsparse_status           status, status_exp, ret;
    aoclsparse_matrix           A;
    std::vector<aoclsparse_int> icrow, icol;
    std::vector<T>              aval, b, x;
    std::vector<itsol_opts>     opts = {};
    aoclsparse_int              n, m, nnz;
    aoclsparse_mat_descr        descr;
    T                           rinfo[100];
    user_data                   udata;

    pass = true;

    switch(itest)
    {
    case 0:
        testid = "maximum iterations";
        ret    = create_matrix(sample_cg_mat, m, n, nnz, icrow, icol, aval, A, descr, VERBOSE);
        if(ret != aoclsparse_status_success)
            pass = false;
        status_exp = aoclsparse_status_maxit;
        opts       = {{"CG Iteration Limit", "2"}};
        break;

    case 1:
        testid = "user stop";
        ret    = create_matrix(sample_cg_mat, m, n, nnz, icrow, icol, aval, A, descr, VERBOSE);
        if(ret != aoclsparse_status_success)
            pass = false;
        status_exp = aoclsparse_status_user_stop;
        break;

    case 2:
        testid = "invalid user matrix";
        ret    = create_matrix(invalid_mat, m, n, nnz, icrow, icol, aval, A, descr, VERBOSE);
        if(ret != aoclsparse_status_success)
            pass = false;
        status_exp = aoclsparse_status_invalid_value;
        break;

    case 3:
        testid = "non symmetric matrix";
        ret    = create_matrix(N5_full_sorted, m, n, nnz, icrow, icol, aval, A, descr, VERBOSE);
        if(ret != aoclsparse_status_success)
            pass = false;
        status_exp = aoclsparse_status_invalid_value;
        break;

    default:
        return 0;
    }
    std::cout << "Testing " << testid << "... " << std::endl;

    // Create the iteartive solver handle
    aoclsparse_itsol_handle handle = nullptr;
    itsol_init<T>(&handle);

    // Initialize rhs and initial point
    b.resize(n);
    x.resize(n);
    for(aoclsparse_int i = 0; i < n; i++)
    {
        x[i] = b[i] = 1.0;
    }

    // add options
    for(auto op : opts)
    {
        ret = aoclsparse_itsol_option_set(handle, op.option.c_str(), op.value.c_str());
        if(ret != aoclsparse_status_success)
        {
            if(VERBOSE)
                std::cout << "Options " << op.option << "could not be set to " << op.value
                          << std::endl;
            pass = false;
        }
    }
    // Set up user data
    udata.itest  = itest;
    udata.testid = "errors";

    // Call the CG solver
    status = itsol_solve(handle, n, A, descr, &b[0], &x[0], rinfo, precond, monit, &udata);

    pass = pass && (status == status_exp);
    if(pass)
        std::cout << "OK" << std::endl;
    else
    {
        std::cout << "FAILED, expected status " << status_exp << std::endl
                  << "             Got status " << status << std::endl;
    }

    aoclsparse_destroy(A);
    aoclsparse_itsol_destroy(&handle);
    aoclsparse_destroy_mat_descr(descr);
    return 1;
}

template <typename T>
aoclsparse_int test_cg_positive(aoclsparse_int itest, bool &pass)
{
    std::string                 testid;
    aoclsparse_status           status, status_exp, ret;
    aoclsparse_matrix           A;
    std::vector<aoclsparse_int> icrow, icol;
    std::vector<T>              aval, b, x, x_exp;
    std::vector<itsol_opts>     opts = {};
    aoclsparse_int              n, m, nnz;
    aoclsparse_mat_descr        descr;
    T                           rinfo[100];
    user_data                   udata;

    pass       = true;
    status_exp = aoclsparse_status_success;

    // Create the iteartive solver handle
    aoclsparse_itsol_handle handle = nullptr;
    itsol_init<T>(&handle);

    switch(itest)
    {
    case 0:
        testid = "small symmetric mat, no precond";
        ret    = create_matrix(sample_cg_mat, m, n, nnz, icrow, icol, aval, A, descr, VERBOSE);
        if(ret != aoclsparse_status_success)
            pass = false;
        break;

    case 1:
        testid = "small symmetric mat, symgs precond";
        ret    = create_matrix(sample_cg_mat, m, n, nnz, icrow, icol, aval, A, descr, VERBOSE);
        if(ret != aoclsparse_status_success)
            pass = false;
        opts = {{"CG Preconditioner", "SGS"}, {"CG Iteration Limit", "6"}};
        break;

    case 2:
        testid = "small symmetric mat, user precond";
        ret    = create_matrix(sample_cg_mat, m, n, nnz, icrow, icol, aval, A, descr, VERBOSE);
        if(ret != aoclsparse_status_success)
            pass = false;
        opts = {{"CG Preconditioner", "User"}, {"CG Iteration Limit", "8"}};
        break;

    case 3:
        testid = "double call";
        ret    = create_matrix(sample_cg_mat, m, n, nnz, icrow, icol, aval, A, descr, VERBOSE);
        if(ret != aoclsparse_status_success)
            pass = false;
        b      = {1, 1, 1, 1, 1, 1, 1, 1};
        x      = {1, 1, 1, 1, 1, 1, 1, 1};
        status = itsol_solve(handle, n, A, descr, &b[0], &x[0], rinfo, precond, monit, &udata);
        if(status != aoclsparse_status_success)
        {
            if(VERBOSE)
                std::cout << "Error in firt call of itsolve_solve" << std::endl;
            pass = false;
        }
        break;

    default:
        aoclsparse_itsol_destroy(&handle);
        return 0;
    }
    std::cout << "Testing " << testid << "... " << std::endl;

    // Initialize rhs and initial point
    T alpha = 1.0, beta = 0.0;
    b.resize(n);
    x.resize(n);
    x_exp.resize(n);
    for(aoclsparse_int i = 0; i < n; i++)
    {
        x_exp[i] = i;
        x[i]     = 1.0;
    }
    status = aoclsparse_dmv(aoclsparse_operation_none, &alpha, A, descr, &x_exp[0], &beta, &b[0]);
    if(status != aoclsparse_status_success)
    {
        if(VERBOSE)
            std::cout << "unexpected error in matrix vector product" << std::endl;
        pass = false;
    }

    // add options
    for(auto op : opts)
    {
        ret = aoclsparse_itsol_option_set(handle, op.option.c_str(), op.value.c_str());
        if(ret != aoclsparse_status_success)
        {
            if(VERBOSE)
                std::cout << "Options " << op.option << "could not be set to " << op.value
                          << std::endl;
            pass = false;
        }
    }
    udata.itest  = itest;
    udata.testid = "positive";

    // Call the CG solver
    status = itsol_solve(handle, n, A, descr, &b[0], &x[0], rinfo, precond, monit, &udata);

    bool sol_ok;
    pass = pass && (status == status_exp);
    comp_tol_vec("X solution", 1.0e-05, n, &x_exp[0], &x[0], sol_ok, VERBOSE);
    pass = pass && sol_ok;
    if(pass)
        std::cout << "OK" << std::endl;
    else
    {
        std::cout << "FAILED" << std::endl;
    }

    aoclsparse_destroy(A);
    aoclsparse_itsol_destroy(&handle);
    aoclsparse_destroy_mat_descr(descr);
    return 1;
}

bool run_tests(std::string testid, aoclsparse_int test(aoclsparse_int itest, bool &pass))
{
    aoclsparse_int more_tests = 1;
    aoclsparse_int itest      = 0;
    bool           pass;
    bool           all_pass  = true;
    std::string    separator = "------------------------------------------";

    std::cout << "Running " << testid << " tests" << std::endl << separator << std::endl;
    while(more_tests)
    {
        more_tests = test(itest, pass);
        all_pass   = all_pass && pass;
        itest++;
    }
    std::cout << separator << std::endl << "  " << testid << ": ";
    if(all_pass)
        std::cout << "OK" << std::endl;
    else
        std::cout << "FAILED" << std::endl;
    std::cout << std::endl;
    return all_pass;
}

int main()
{
    std::cout << "++++++++++++++++++++++++" << std::endl
              << "Testing CG functionality" << std::endl
              << "++++++++++++++++++++++++" << std::endl
              << std::endl;

    bool all_pass = true;

    all_pass = all_pass && run_tests("CG errors (double)", test_cg_error<double>);
    all_pass = all_pass && run_tests("CG errors (float)", test_cg_error<float>);
    all_pass = all_pass && run_tests("CG normal (double)", test_cg_positive<double>);
    all_pass = all_pass && run_tests("CG normal (float)", test_cg_positive<double>);

    std::cout << "Overall CG tests: ";
    if(all_pass)
        std::cout << "OK" << std::endl;
    else
        std::cout << "FAILED" << std::endl;
    return (all_pass ? 0 : 1);
}