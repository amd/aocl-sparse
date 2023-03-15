set(utils_source functionality_tests_utils.cpp)
include_directories(${CMAKE_SOURCE_DIR}/library/src/include
                    ${CMAKE_SOURCE_DIR}/library/src/solvers
                    ${CMAKE_SOURCE_DIR}/library/src/analysis
                    ${CMAKE_SOURCE_DIR}/library/src/level2
                    ${CMAKE_SOURCE_DIR}/library/src/extra)
if(WIN32)
else()
    # TRSV TESTS
    add_executable(trsv_tests trsv_tests.cpp ${CMAKE_SOURCE_DIR}/library/src/extra/aoclsparse_context.cpp)
    target_link_libraries(trsv_tests aoclsparse pthread)

    # CG TESTS ! FIXME check diff with unit-test counterpart
    # add_executable(cg_ftests cg_tests.cpp ${utils_source})
    # target_link_libraries(cg_ftests aoclsparse pthread)
endif()

set(FUNCTIONALITY_TEST_EXECUTABLES
    trsv_tests
)
install(TARGETS ${FUNCTIONALITY_TEST_EXECUTABLES} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/functionality)
