#pragma once

#include "measure.h"
#include "../src/util/timer.h"

//--------------------------------------------------------

template<typename Value>
void sorter(std::vector<Value>&vec);

//--------------------------------------------------------

template<typename Value>
double measure(const std::vector<Value>&vec, int tries, int & wrongAnsCnt)
{
    vector<double> resAcc;

    for(int i = 0; i < tries; i++)
    {
        vector<Value> tmp = vec;
        {
            TIMER t([&](double res){resAcc.push_back(res);});
            sorter(tmp);
        }

        if(!std::is_sorted(tmp.begin(), tmp.end()))
            wrongAnsCnt++;
    }

    return accumulate(resAcc.begin(), resAcc.end(), 0.0) / resAcc.size();
}