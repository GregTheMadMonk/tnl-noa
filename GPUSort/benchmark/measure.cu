#pragma once

#include <vector>

#include "measure.h"
#include "../src/util/timer.h"

#include <TNL/Containers/Array.h>
#include "../src/util/algorithm.h"
using namespace TNL;
using namespace TNL::Containers;

//--------------------------------------------------------

template<typename Value>
void sorter(ArrayView<Value, Devices::Cuda> arr);

//--------------------------------------------------------

template<typename Value>
double measure(const std::vector<Value>&vec, int tries, int & wrongAnsCnt)
{
    vector<double> resAcc;

    for(int i = 0; i < tries; i++)
    {
        Array<Value, Devices::Cuda> arr(vec);
        auto view = arr.getView();
        {
            TIMER t([&](double res){resAcc.push_back(res);});
            sorter(view);
        }

        if(!is_sorted(view))
            wrongAnsCnt++;
    }

    return accumulate(resAcc.begin(), resAcc.end(), 0.0) / resAcc.size();
}