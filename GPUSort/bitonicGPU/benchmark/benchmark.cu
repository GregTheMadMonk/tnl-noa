#include <string>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>

#include <TNL/Containers/Array.h>

#include "../bitonicSort.h"

using namespace TNL;
using namespace TNL::Containers;

typedef Devices::Cuda Device;

template <class T>
std::ostream& operator<< (std::ostream&out, std::vector<T> &arr)
{
    for (auto x : arr)
        std::cout << x << " ";
    return out;
}


struct TIMER
{
    std::function<void(double)> f;
    std::chrono::high_resolution_clock::time_point begin;

    TIMER(std::function<void(double)> func/* = [](double res){std::cout << res << std::endl;}*/)
        : f(func), begin(std::chrono::high_resolution_clock::now()) {}

    ~TIMER()
    {
        auto end = std::chrono::high_resolution_clock::now();
        double result = (std::chrono::duration_cast<std::chrono::microseconds >(end - begin).count() / 1000.);
        f(result);
    }
};


void test1()
{
    int size = 1<<10;
    TNL::Containers::Array<int, Device> cudaArr(size);
    cudaArr.evaluate([=] __cuda_callable__ (int i) {return i;});
    auto view = cudaArr.getView();

    {
        TIMER t([](double res){std::cout << res << std::endl;});
        bitonicSort(view);
    }
}

void randomShuffles()
{
    int iterations = 100;
    std::cout << iterations << " random permutations" << std::endl;
    for(int p = 13; p <= 19; ++p)
    {
        int size = 1<<p;
        std::vector<int> orig(size);
        std::iota(orig.begin(), orig.end(), 0);
        std::vector<double> results;

        for (int i = 0; i < iterations; i++)
        {
            std::random_shuffle(orig.begin(), orig.end());

            TNL::Containers::Array<int, Device> cudaArr(orig);
            auto view = cudaArr.getView();
            {
                TIMER t([&](double res){results.push_back(res);});
                bitonicSort(view);
            }

        }
        std::cout << "average time for arrSize = 2^" << p << ": " << std::accumulate(results.begin(), results.end(), 0.)/results.size() << " ms" << std::endl;

    }
}

void allPermutations(std::vector<int> orig)
{
    std::vector<double> results;
    while (std::next_permutation(orig.begin(), orig.end()))
    {
        TNL::Containers::Array<int, Device> cudaArr(orig);
        auto view = cudaArr.getView();

        {
            TIMER t([&](double res){results.push_back(res);});
            bitonicSort(view);
        }
    }
    std::cout << "average time: " << std::accumulate(results.begin(), results.end(), 0.)/results.size() << " ms" << std::endl;
}


int main()
{
    randomShuffles();

    return 0;
}