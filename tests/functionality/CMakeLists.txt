set(utils_source functionality_tests_utils.cpp)
include_directories(${CMAKE_SOURCE_DIR}/library/src/include
                    ${CMAKE_SOURCE_DIR}/library/src/solvers
                    ${CMAKE_SOURCE_DIR}/library/src/analysis
                    ${CMAKE_SOURCE_DIR}/library/src/level2
                    ${CMAKE_SOURCE_DIR}/library/src/extra)
if(WIN32)
    # HINT TESTS
    add_executable(hint_tests hint_tests.cpp ${utils_source})
    target_link_libraries(hint_tests aoclsparse)

    # TRSV TESTS
    add_executable(trsv_tests trsv_tests.cpp ${CMAKE_SOURCE_DIR}/library/src/extra/aoclsparse_context.cpp)
    target_link_libraries(trsv_tests aoclsparse)

    # CG TESTS
    # add_executable(cg_ftests cg_tests.cpp ${utils_source})
    # target_link_libraries(cg_ftests aoclsparse)

else()
    # HINT TESTS
    add_executable(hint_tests hint_tests.cpp ${utils_source})
    target_link_libraries(hint_tests aoclsparse pthread)

    # TRSV TESTS
    add_executable(trsv_tests trsv_tests.cpp ${CMAKE_SOURCE_DIR}/library/src/extra/aoclsparse_context.cpp)
    target_link_libraries(trsv_tests aoclsparse pthread)

    # CG TESTS
    # add_executable(cg_ftests cg_tests.cpp ${utils_source})
    # target_link_libraries(cg_ftests aoclsparse pthread)
endif()

set(FUNCTIONALITY_TEST_EXECUTABLES
    hint_tests
    trsv_tests
    # cg_tests
)
install(TARGETS ${FUNCTIONALITY_TEST_EXECUTABLES} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/functionality)
