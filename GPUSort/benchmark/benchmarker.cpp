#include <iostream>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

#include "../src/util/timer.h"

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
    vector<int> vec(size);
    iota(vec.begin(), vec.end(), 0);
    
    return measure(vec);
}

double random(int size)
{
    srand(size + 2021);

    vector<int> vec(size);
    generate(vec.begin(), vec.end(), [=](){return std::rand() % (2*size);});

    return measure(vec);
}

double shuffle(int size)
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
    for(int i = 0; i < size; i++)
        vec[i] = size - i;
                
    return measure(vec);
}

double zero_entropy(int size)
{
    vector<int> vec(size);
    for(auto & x : vec)
        x = size;
                
    return measure(vec);
}

double gaussian(int size)
{
	srand(size + 2000);

    vector<int> vec(size);
	for (int i = 0; i < size; ++i)
    {
		int value = 0;
		for (int j = 0; j < 4; ++j) 
			value += rand()%16384;

		vec[i] = value /4;
	}
    return measure(vec);  
}

double bucket(int size)
{
	srand (size + 94215);
    vector<int> vec(size);
    
	double tmp = ((double)size)*3000000; //(RAND_MAX)/p; --> ((double)N)*30000;
	double tmp2 = sqrt(tmp);

	int p= (size+tmp2-1)/tmp2;

	const int VALUE = 8192/p; //(RAND_MAX)/p;

	int i=0; int x=0;
	//the array of size N is split into 'p' buckets
	while(i < p)
	{
		for (int z = 0; z < p; ++z)
			for (int j = 0; j < size/(p*p); ++j)
			{
				//every bucket has N/(p*p) items and the range is [min : VALUE-1 ]
				int min = VALUE*z;

				vec[x]= min + ( rand() %  (VALUE-1) ) ;
				x++;
			}
		i++;
	}

    return measure(vec);
}

double staggared(int size)
{
	srand (size + 815618);
    vector<int> vec(size);

	int tmp=4096; //(RAND_MAX)/p; --> size=2048
	int p= (size+tmp-1)/tmp;

	const int VALUE = (1<<31)/p; //(RAND_MAX)/p;

	int i=1; int x=0;
	//the array of size N is split into 'p' buckets
	while(i <= p)
	{
		//every bucket has N/(p) items
		for (int j = 0; j < size/(p); ++j)
		{
			int min;

			if(i<=(p/2))
				min = (2*i -1)*VALUE;

			else
				min = (2*i-p-1)*VALUE;

			vec[x++]= min + ( rand() % (VALUE - 1) );
		}
		i++;
	}

    return measure(vec);
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
        out << shuffle(size) << delim;
        out << sorted(size) << delim;
        out << almostSorted(size) << delim;
        out << decreasing(size) << delim;
        out << gaussian(size) << delim;
        out << bucket(size) << delim;
        out << staggared(size) << delim;
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