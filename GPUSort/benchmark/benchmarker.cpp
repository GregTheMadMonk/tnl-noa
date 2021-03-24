#include <iostream>
#include <numeric>
#include <iomanip>

#include <TNL/Containers/Array.h>
#include "../src/util/timer.h"

//---------------------------
/**
 * important! to make use of this benchmarker, it is needed to define SORTERFUNCTION
 * then include this file
 * */
//---------------------------

using namespace TNL;
using namespace TNL::Containers;
using namespace std;

const int lowPow = 15, highLow = 25;
const int tries = 50;

double measure(const vector<int>&vec)
{
    Array<int, Devices::Cuda> arr(vec.size());
    vector<double> resAcc;

    for(int i = 0; i < tries; i++)
    {
        arr = vec;
        auto view = arr.getView();

        {
            TIMER t([&](double res){resAcc.push_back(res);});
            SORTERFUNCTION(view);
        }
    }

    return accumulate(resAcc.begin(), resAcc.end(), 0.0) / resAcc.size();
}

double sorted(int size)
{
    vector<int> vec(size);
    iota(vec.begin(), vec.end(), 0);
    
    return measure(vec);
}

double random(int size)
{
    srand(size);

    vector<int> vec(size);
    iota(vec.begin(), vec.end(), 0);
    random_shuffle(vec.begin(), vec.end());

    return measure(vec);
}

double almostSorted(int size)
{
    vector<int> vec(size);
    iota(vec.begin(), vec.end(), 0);
    for(int i = 0; i < 3; i++) //swaps 3 times in array
    {
        int s = rand() % (size - 3);
        std::swap(vec[s], vec[s + 1]);
    }

    return measure(vec);
}

double decreasing(int size)
{
    vector<int> vec(size);
    for(size_t i = 0; i < size; i++)
        vec[i] = -i;
                
    return measure(vec);
}

void start(ostream & out, string delim)
{
    out << "size" << delim;
    out << "random" << delim;
    out << "sorted" << delim;
    out << "almost" << delim;
    out << "decreasing";
    out << endl;
    
    for(int pow = lowPow; pow <= highLow; pow++)
    {
        int size =(1<< pow);
        vector<int> vec(size);

        out << "2^" << pow << delim;
        out << fixed << setprecision(3);
        out << random(size) << delim;
        out << sorted(size) << delim;
        out << almostSorted(size) << delim;
        out << decreasing(size);
        out << endl;
    }
}

int main(int argc, char *argv[])
{
    if(argc == 1)
    {
        start(cout, "\t");
    }
    else
    {
        ofstream out(argv[1]);
        start(out, ",");
    }
    return 0;
}