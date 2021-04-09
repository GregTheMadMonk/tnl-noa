#include <iostream>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

#include "../src/util/timer.h"
#include "generators.cpp"

//---------------------------
/**
 * important! to make use of this benchmarker, it is needed to define SORTERFUNCTION
 * then include this file
 * */
//---------------------------

#ifdef HAVE_CUDA

#include <TNL/Containers/Array.h>
#include "../src/util/algorithm.h"
using namespace TNL;
using namespace TNL::Containers;

#endif

static int notCorrectCounters = 0;

#ifndef LOW_POW
    #define LOW_POW 10
#endif

#ifndef HIGH_POW
    #define HIGH_POW 25
#endif

#ifndef TRIES
    #define TRIES 20
#endif

double measure(const vector<int>&vec);

#ifndef MY_OWN_MEASURE
double measure(const vector<int>&vec)
{
    vector<double> resAcc;


    for(int i = 0; i < TRIES; i++)
    {
    #ifdef HAVE_CUDA
        Array<int, Devices::Cuda> arr(vec);
        auto view = arr.getView();

        {
            TIMER t([&](double res){resAcc.push_back(res);});
            SORTERFUNCTION(view);
        }

        if(!is_sorted(view))
            notCorrectCounters++;
    #else
        vector<int> tmp = vec;

        {
            TIMER t([&](double res){resAcc.push_back(res);});
            SORTERFUNCTION(tmp);
        }

        if(!std::is_sorted(tmp.begin(), tmp.end()))
            notCorrectCounters++;
    #endif
    }

    return accumulate(resAcc.begin(), resAcc.end(), 0.0) / resAcc.size();
}
#endif

double sorted(int size)
{    
    return measure(generateSorted(size));
}

double random(int size)
{
    return measure(generateRandom(size));
}

double shuffle(int size)
{
    return measure(generateShuffle(size));
}

double almostSorted(int size)
{
    return measure(generateAlmostSorted(size));
}

double decreasing(int size)
{
    return measure(generateDecreasing(size));
}

double zero_entropy(int size)
{               
    return measure(generateZero_entropy(size));
}

double gaussian(int size)
{
    return measure(generateZero_entropy(size));  
}

double bucket(int size)
{
    return measure(generateBucket(size));
}

double staggared(int size)
{
    return measure(generateStaggered(size));
}

void start(ostream & out, string delim)
{
    out << "size" << delim;
    out << "random" << delim;
    out << "shuffle" << delim;
    out << "sorted" << delim;
    out << "almost" << delim;
    out << "decreas" << delim;
    out << "gauss" << delim;
    out << "bucket" << delim;
    out << "stagger" << delim;
    out << "zero_entropy";
    out << endl;
    
    for(int pow = LOW_POW; pow <= HIGH_POW; pow++)
    {
        int size =(1<< pow);
        vector<int> vec(size);

        out << "2^" << pow << delim;
        out << fixed << setprecision(3);
        
        out << random(size) << delim;
        out.flush();
        
        out << shuffle(size) << delim;
        out.flush();
        
        out << sorted(size) << delim;
        out.flush();
        
        out << almostSorted(size) << delim;
        out.flush();
        
        out << decreasing(size) << delim;
        out.flush();
        
        out << gaussian(size) << delim;
        out.flush();
        
        out << bucket(size) << delim;
        out.flush();
        
        out << staggared(size) << delim;
        out.flush();
        
        out << zero_entropy(size);
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
        std::ofstream out(argv[1]);
        start(out, ",");
    }
    if(notCorrectCounters > 0)
    {
        std::cerr << notCorrectCounters << "tries were sorted incorrectly" << std::endl;
    }
    return 0;
}