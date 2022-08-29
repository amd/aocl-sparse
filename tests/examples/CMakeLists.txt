# Add executables for each of the sample examples

if(WIN32)
    # CSR2M Sample
    add_executable(sample_csr2m sample_csr2m.cpp)
    target_link_libraries(sample_csr2m aoclsparse)

    # SPMV Sample
    add_executable(sample_spmv sample_spmv.cpp)
    target_link_libraries(sample_spmv aoclsparse)
	
	# TRSV Sample
	add_executable(sample_dtrsv sample_dtrsv.cpp)
	target_link_libraries(sample_dtrsv aoclsparse)		

    # CG Samples 
    add_executable(sample_itsol_d_cg_rci sample_itsol_d_cg_rci.cpp)
    target_link_libraries(sample_itsol_d_cg_rci aoclsparse)

    add_executable(sample_itsol_d_cg sample_itsol_d_cg.cpp)
    target_link_libraries(sample_itsol_d_cg aoclsparse)

    add_executable(sample_itsol_s_cg_rci sample_itsol_s_cg_rci.cpp)
    target_link_libraries(sample_itsol_s_cg_rci aoclsparse)

    add_executable(sample_itsol_s_cg sample_itsol_s_cg.cpp)
    target_link_libraries(sample_itsol_s_cg aoclsparse)

    # GMRES Samples
    add_executable(sample_itsol_d_gmres_rci sample_itsol_d_gmres_rci.cpp)
    target_link_libraries(sample_itsol_d_gmres_rci aoclsparse)

    add_executable(sample_itsol_d_gmres sample_itsol_d_gmres.cpp)
    target_link_libraries(sample_itsol_d_gmres aoclsparse)

    add_executable(sample_itsol_s_gmres_rci sample_itsol_s_gmres_rci.cpp)
    target_link_libraries(sample_itsol_s_gmres_rci aoclsparse)

    add_executable(sample_itsol_s_gmres sample_itsol_s_gmres.cpp)
    target_link_libraries(sample_itsol_s_gmres aoclsparse)
else()
    # CSR2M Sample
    add_executable(sample_csr2m sample_csr2m.cpp)
    target_link_libraries(sample_csr2m aoclsparse pthread)

    # SPMV Sample
    add_executable(sample_spmv sample_spmv.cpp)
    target_link_libraries(sample_spmv aoclsparse pthread)
	
	# TRSV Sample
	add_executable(sample_dtrsv sample_dtrsv.cpp)
	target_link_libraries(sample_dtrsv aoclsparse pthread)	

    # CG Samples 
    add_executable(sample_itsol_d_cg_rci sample_itsol_d_cg_rci.cpp)
    target_link_libraries(sample_itsol_d_cg_rci aoclsparse pthread)

    add_executable(sample_itsol_d_cg sample_itsol_d_cg.cpp)
    target_link_libraries(sample_itsol_d_cg aoclsparse pthread)

    add_executable(sample_itsol_s_cg_rci sample_itsol_s_cg_rci.cpp)
    target_link_libraries(sample_itsol_s_cg_rci aoclsparse pthread)

    add_executable(sample_itsol_s_cg sample_itsol_s_cg.cpp)
    target_link_libraries(sample_itsol_s_cg aoclsparse pthread)

    # GMRES Samples
    add_executable(sample_itsol_d_gmres_rci sample_itsol_d_gmres_rci.cpp)
    target_link_libraries(sample_itsol_d_gmres_rci aoclsparse pthread)

    add_executable(sample_itsol_d_gmres sample_itsol_d_gmres.cpp)
    target_link_libraries(sample_itsol_d_gmres aoclsparse pthread)

    add_executable(sample_itsol_s_gmres_rci sample_itsol_s_gmres_rci.cpp)
    target_link_libraries(sample_itsol_s_gmres_rci aoclsparse pthread)

    add_executable(sample_itsol_s_gmres sample_itsol_s_gmres.cpp)
    target_link_libraries(sample_itsol_s_gmres aoclsparse pthread)
endif()