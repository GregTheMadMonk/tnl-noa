#pragma once
#include <numeric>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

vector<int> generateSorted(int size)
{
    vector<int> vec(size);

    iota(vec.begin(), vec.end(), 0);

    return vec;
}

vector<int> generateRandom(int size)
{
    vector<int> vec(size);

    srand(size + 2021);
    generate(vec.begin(), vec.end(), [=](){return std::rand() % (2*size);});

    return vec;
}

vector<int> generateShuffle(int size)
{
    vector<int> vec(size);

    iota(vec.begin(), vec.end(), 0);
    srand(size);
    random_shuffle(vec.begin(), vec.end());

    return vec;
}

vector<int> generateAlmostSorted(int size)
{
    vector<int> vec(size);

    iota(vec.begin(), vec.end(), 0);
    srand(9451);
    for(int i = 0; i < 3; i++) //swaps 3 times in array
    {
        int s = rand() % (size - 3);
        std::swap(vec[s], vec[s + 1]);
    }

    return vec;
}

vector<int> generateDecreasing(int size)
{
    vector<int> vec(size);

    for(int i = 0; i < size; i++)
        vec[i] = size - i;
            
    return vec;
}

vector<int> generateZero_entropy(int size)
{
    vector<int> vec(size, 515);
    return vec;
}

vector<int> generateGaussian(int size)
{
    vector<int> vec(size);
	srand(size + 2000);

	for (int i = 0; i < size; ++i)
    {
		int value = 0;
		for (int j = 0; j < 4; ++j) 
			value += rand()%16384;

		vec[i] = value /4;
	}

    return vec;
}

vector<int> generateBucket(int size)
{
    vector<int> vec(size);

	srand (size + 94215);
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

    return vec;
}

vector<int> generateStaggered(int size)
{
    vector<int> vec(size);

	srand (size + 815618);
	int tmp=4096; //(RAND_MAX)/p; --> size=2048
	int p= (size+tmp-1)/tmp;

	const int VALUE = (1<<30)/p; //(RAND_MAX)/p;

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

    return vec;
}