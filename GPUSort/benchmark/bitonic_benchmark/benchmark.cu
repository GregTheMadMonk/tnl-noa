#include <iostream>
#include <numeric>
#include <iomanip>

#include <TNL/Containers/Array.h>
#include "../../src/util/timer.h"

//---------------------------
#include "../../src/bitonicSort/bitonicSort.h"
#define SORTERFUNCTION bitonicSort
//---------------------------

using namespace TNL;
using namespace TNL::Containers;
using namespace std;

const int lowPow = 15, highLow = 22;
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

int main()
{
    string delim = "\t";
    cout << "size" << delim;
    cout << "random" << delim;
    cout << "sorted" << delim;
    cout << "almost" << delim;
    cout << "decreasing" << delim;
    cout << endl;
    
    for(int pow = lowPow; pow <= highLow; pow++)
    {
        int size =(1<< pow);
        vector<int> vec(size);

        cout << "2^" << pow << delim;
        cout << fixed << setprecision(3);
        cout << random(size) << delim;
        cout << sorted(size) << delim;
        cout << almostSorted(size) << delim;
        cout << decreasing(size) << delim;
        cout << endl;
    }
    return 0;
}