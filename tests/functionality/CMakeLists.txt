add_executable(hint_tests hint_tests.cpp)
target_include_directories(hint_tests PRIVATE ${CMAKE_SOURCE_DIR}/library/src/include)
target_link_libraries(hint_tests aoclsparse)

add_executable(trsv_tests trsv_tests.cpp)
target_include_directories(trsv_tests PRIVATE 
    ${CMAKE_SOURCE_DIR}/library/src/include 
    ${CMAKE_SOURCE_DIR}/library/src/level2
    ${CMAKE_SOURCE_DIR}/library/src/analysis)
target_link_libraries(trsv_tests aoclsparse)