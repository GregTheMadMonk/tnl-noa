#include <string>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric>

#include <TNL/Containers/Array.h>

#include "../bitonicSort.h"

template <class T>
std::ostream& operator<< (std::ostream&out, std::vector<T> &arr)
{
    for (auto x : arr)
        std::cout << x << " ";
    return out;
}

struct TIMER
{
    std::string s;
    std::chrono::steady_clock::time_point begin;
    double result = 0;
    bool stopped = false;

    TIMER(const std::string &name = "")
        : s(name), begin(std::chrono::steady_clock::now()) {}

    double stop()
    {
        auto end = std::chrono::steady_clock::now();
        result = (std::chrono::duration_cast<std::chrono::microseconds >(end - begin).count() / 1000.);
        stopped = true;
        return result;
    }

    void printTime()
    {
        if(!stopped)
            stop();
        std::cout << ("Measured " + s + ": ") << result << " ms" << std::endl;
    }
    
    ~TIMER()
    {
        if(!stopped)
        {
            stop();
            printTime();
        }
    }
};


void test1()
{
    int size = 1<<10;
    TNL::Containers::Array<int, Device> cudaArr(size);
    cudaArr.evaluate([=] __cuda_callable__ (int i) {return i;});
    auto view = cudaArr.getView();

    {
        TIMER t("sorted sequences");
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
            std::vector<int> tmp(orig.begin(), orig.end());

            {
                TIMER t("random permutation");

                //std::sort(tmp.begin(), tmp.end());
                bitonicSort(view);
                
                results.push_back(t.stop());
                //t.printTime();
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
            TIMER t("random permutation");
            bitonicSort(view);
            results.push_back(t.stop());
            //t.printTime();
        }
    }
    std::cout << "average time: " << std::accumulate(results.begin(), results.end(), 0.)/results.size() << " ms" << std::endl;
}


int main()
{
    randomShuffles();

    return 0;
}