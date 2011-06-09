file(COPY matrix-market DESTINATION ${EXECUTABLE_OUTPUT_PATH}/config)
file(COPY florida-matrix-market DESTINATION ${EXECUTABLE_OUTPUT_PATH}/config)
file(COPY tnl-sparse-matrix-check.cfg.desc DESTINATION ${EXECUTABLE_OUTPUT_PATH}/config)

set (tnl_sparse_matrix_check_SOURCES tnl-sparse-matrix-check.cpp tnl-sparse-matrix-check.h)
set (matrix_formats_test_SOURCES matrix-formats-test.cpp matrix-formats-test.h)
set (sparse_matrix_benchmark_SOURCES sparse-matrix-benchmark.cpp sparse-matrix-benchmark.h)
#sparse_matrix_benchmark_CXXFLAGS = $(OPTIMISECXXFLAGS)

add_executable (tnl_sparse_matrix_check${mpiExt}${debugExt} ${tnl_sparse_matrix_check_SOURCES})
target_link_libraries (tnl_sparse_matrix_check${mpiExt}${debugExt} libtnl${mpiExt}${debugExt}-0.1 
    libtnlcore${mpiExt}${debugExt}-0.1 libtnlmatrix${mpiExt}${debugExt}-0.1)

add_executable (matrix_formats_test${mpiExt}${debugExt} ${matrix_formats_test_SOURCES})
target_link_libraries (matrix_formats_test${mpiExt}${debugExt} libtnl${mpiExt}${debugExt}-0.1 
    libtnlcore${mpiExt}${debugExt}-0.1 libtnlmatrix${mpiExt}${debugExt}-0.1)

add_executable (sparse_matrix_benchmark${mpiExt}${debugExt} ${sparse_matrix_benchmark_SOURCES})
target_link_libraries (sparse_matrix_benchmark${mpiExt}${debugExt} libtnl${mpiExt}${debugExt}-0.1 
    libtnlcore${mpiExt}${debugExt}-0.1 libtnlmatrix${mpiExt}${debugExt}-0.1)