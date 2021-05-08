#include "../../../otherGPUsorts/cudaExamples/sortingNetworks/bitonicSort.cu"
#include "../../../GPUSort/src/util/timer.h"
#include "../../../GPUSort/src/util/algorithm.h"
#include <TNL/Containers/Array.h>
#include <vector>
#include <numeric>
using namespace std;
using namespace TNL;
using namespace TNL::Containers;
//---------------------

double measure(const std::vector<int>&vec, int tries, int & wrongAnsCnt)
{
    vector<double> resAcc;

    Array<int, Devices::Cuda> arr(vec.size());
    Array<int, Devices::Cuda> arr2(vec.size());
    for(int i = 0; i < tries; i++)
    {
        arr = vec;
        arr2 = vec;
        {
            TIMER t([&](double res){resAcc.push_back(res);});
            bitonicSort((unsigned *)arr.getData(), (unsigned *)arr2.getData(),
                        (unsigned *)arr.getData(), (unsigned *)arr2.getData(),
                        1, arr.getSize(), 1);
            cudaDeviceSynchronize();
        }

        if(!is_sorted(arr.getView()))
            wrongAnsCnt++;
    }

    return accumulate(resAcc.begin(), resAcc.end(), 0.0) / resAcc.size();
}

#include "../../../GPUSort/benchmark/benchmarker.cpp"