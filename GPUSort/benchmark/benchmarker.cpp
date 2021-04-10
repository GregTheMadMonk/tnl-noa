#include <iostream>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

#include "generators.cpp"
#include "measure.h"

#ifndef LOW_POW
    #define LOW_POW 10
#endif

#ifndef HIGH_POW
    #define HIGH_POW 25
#endif

#ifndef TRIES
    #define TRIES 20
#endif

//------------------------------------------------------------

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
    
    int wrongAnsCnt = 0;
    
    for(int pow = LOW_POW; pow <= HIGH_POW; pow++)
    {
        int size =(1<< pow);
        vector<int> vec(size);

        out << "2^" << pow << delim;
        out << fixed << setprecision(3);
        
        out << measure(generateRandom(size), TRIES, wrongAnsCnt);
        out << delim;

        out << measure(generateShuffle(size), TRIES, wrongAnsCnt);
        out << delim;

        out << measure(generateSorted(size), TRIES, wrongAnsCnt);
        out << delim;

        out << measure(generateAlmostSorted(size), TRIES, wrongAnsCnt);
        out << delim;

        out << measure(generateDecreasing(size), TRIES, wrongAnsCnt);
        out << delim;

        out << measure(generateGaussian(size), TRIES, wrongAnsCnt) ;
        out << delim;

        out << measure(generateBucket(size), TRIES, wrongAnsCnt);
        out << delim;

        out << measure(generateStaggered(size), TRIES, wrongAnsCnt);
        out << delim;

        out << measure(generateZero_entropy(size), TRIES, wrongAnsCnt);
        out << endl;
    }

    if(wrongAnsCnt > 0)
        std::cerr << wrongAnsCnt << "tries were sorted incorrectly" << std::endl;
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
    return 0;
}