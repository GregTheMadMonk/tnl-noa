#include "../../otherGPUsorts/cederman/cederman_qsort.cu"
#include <vector>

void sorter(std::vector<int> & vec)
{
    gpuqsort((unsigned int *)vec.data(), vec.size());
}

//------------------------------------

#include "../../GPUSort/benchmark/benchmarker.cpp"
#include "../../GPUSort/benchmark/measure.cpp"
