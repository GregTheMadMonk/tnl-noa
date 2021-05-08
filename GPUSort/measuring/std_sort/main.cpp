#include <algorithm>
#include <vector>

#define TRIES 5

void sorter(std::vector<int>&vec)
{
    std::sort(vec.begin(), vec.end());
}
//---------------------------
#include "../../GPUSort/benchmark/benchmarker.cpp"
#include "../../GPUSort/benchmark/measure.cpp"
