#include "gtest/gtest.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/MemoryOperations.h>
#include "../../src/bitonicSort/bitonicSort.h"
#include "../../src/util/algorithm.h"

//----------------------------------------------------------------------------------

TEST(permutations, allPermutationSize_1_to_8)
{
    for(int i = 2; i<=8; i++ )
    {
        int size = i;
        std::vector<int> orig(size);
        std::iota(orig.begin(), orig.end(), 0);

        do
        {
            TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(orig);
            auto view = cudaArr.getView();

            bitonicSort(view);

            ASSERT_TRUE(is_sorted(view)) << "failed " << i << std::endl;
        } 
        while (std::next_permutation(orig.begin(), orig.end()));
    }
}

TEST(permutations, somePermutationSize9)
{
    int size = 9;
    const int stride = 227;
    int i = 0;

    std::vector<int> orig(size);
    std::iota(orig.begin(), orig.end(), 0);

    do
    {
        if ((i++) % stride != 0)
            continue;

        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(orig);
        auto view = cudaArr.getView();

        bitonicSort(view);

        ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
    }
    while (std::next_permutation(orig.begin(), orig.end()));
}

//-----------------------------------------------------------------------

TEST(selectedSize, size15)
{
    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr{5, 9, 4, 8, 6, 1, 2, 3, 4, 8, 1, 6, 9, 4, 9};
    auto view = cudaArr.getView();
    ASSERT_EQ(15, view.getSize()) << "size not 15" << std::endl;
    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
}

TEST(multiblock, 32768_decreasingNegative)
{
    std::vector<int> arr(1<<15);
    for (size_t i = 0; i < arr.size(); i++)
        arr[i] = -i;
    
    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);
    auto view = cudaArr.getView();

    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
}

TEST(randomGenerated, smallArray_randomVal)
{
    std::srand(2006);
    for(int i = 0; i < 100; i++)
    {
        std::vector<int> arr(std::rand()%(1<<10));
        for(auto & x : arr)
            x = std::rand();

        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);

        auto view = cudaArr.getView();
        bitonicSort(view);
        ASSERT_TRUE(is_sorted(view));
    }
}

TEST(randomGenerated, bigArray_all0)
{
    std::srand(304);
    for(int i = 0; i < 50; i++)
    {
        int size = (1<<20) + (std::rand()% (1<<19));

        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(size);

        auto view = cudaArr.getView();
        bitonicSort(view);
        ASSERT_TRUE(is_sorted(view));
    }
}

TEST(nonIntegerType, float_notPow2)
{
    TNL::Containers::Array<float, TNL::Devices::Cuda> cudaArr{5.0, 9.4, 4.6, 8.9, 6.2, 1.15184, 2.23};
    auto view = cudaArr.getView();
    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
}

TEST(nonIntegerType, double_notPow2)
{
    TNL::Containers::Array<double, TNL::Devices::Cuda> cudaArr{5.0, 9.4, 4.6, 8.9, 6.2, 1.15184, 2.23};
    auto view = cudaArr.getView();
    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
}


struct TMPSTRUCT{
    uint8_t m_data[6];
    TMPSTRUCT(){m_data[0] = 0;}
    TMPSTRUCT(int first){m_data[0] = first;};
    bool operator <(const TMPSTRUCT& other) const { return m_data[0] < other.m_data[0];}
    bool operator <=(const TMPSTRUCT& other) const { return m_data[0] <= other.m_data[0];}
};

TEST(nonIntegerType, struct)
{
    TNL::Containers::Array<TMPSTRUCT, TNL::Devices::Cuda> cudaArr{TMPSTRUCT(5), TMPSTRUCT(6), TMPSTRUCT(9), TMPSTRUCT(1)};
    auto view = cudaArr.getView();
    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view));
}


//error bypassing
//https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/blob/fbc34f6a97c13ec865ef7969b9704533222ed408/src/UnitTests/Containers/VectorTest-8.h
void descendingSort(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> view)
{
    auto cmpDescending = [] __cuda_callable__ (int a, int b) {return a > b;};
    bitonicSort(view, cmpDescending);
}

TEST(sortWithFunction, descending)
{
    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr{6, 9, 4, 2, 3};
    auto view = cudaArr.getView();
    descendingSort(view);

    ASSERT_FALSE(is_sorted(view)) << "result " << view << std::endl;
    
    ASSERT_TRUE(view.getElement(0) == 9);
    ASSERT_TRUE(view.getElement(1) == 6);
    ASSERT_TRUE(view.getElement(2) == 4);
    ASSERT_TRUE(view.getElement(3) == 3);
    ASSERT_TRUE(view.getElement(4) == 2);
}

TEST(sortstdVector, stdvector)
{
    std::vector<int> arr(84561);
    for(size_t i = 0; i < arr.size(); i++)
        arr[i] = -i;

    bitonicSort(arr);

    ASSERT_TRUE(std::is_sorted(arr.begin(), arr.end()));
}

TEST(sortRange, secondHalf)
{
    std::vector<int> arr(19);
    int s = 19/2;
    for(size_t i = 0; i < s; i++) arr[i] = -1;
    for(size_t i = s; i < 19; i++) arr[i] = -i;

    bitonicSort(arr, s, 19);

    ASSERT_TRUE(std::is_sorted(arr.begin() + s, arr.end()));
    ASSERT_TRUE(arr[0] == -1); 
    ASSERT_TRUE(arr[s-1] == -1); 
}

TEST(sortRange, middle)
{
    std::srand(8705);

    std::vector<int> arr(20);
    int s = 5, e = 15;
    for(size_t i = 0; i < s; i++) arr[i] = -1;
    for(size_t i = e; i < 20; i++) arr[i] = -1;

    for(size_t i = s; i < e; i++) arr[i] = std::rand();

    bitonicSort(arr, s, e);

    ASSERT_TRUE(std::is_sorted(arr.begin() + s, arr.begin() + e));
    ASSERT_TRUE(arr[0] == -1); 
    ASSERT_TRUE(arr.back() == -1); 
    ASSERT_TRUE(arr[s-1] == -1); 
    ASSERT_TRUE(arr[e] == -1); 
}

TEST(sortRange, middleMultiBlock)
{
    std::srand(4513);
    int size = 1<<20;
    int s = 2000, e = size - 1512;

    std::vector<int> arr(size);
    for(size_t i = 0; i < s; i++) arr[i] = -1;
    for(size_t i = e; i < size; i++) arr[i] = -1;

    for(size_t i = s; i < e; i++) arr[i] = std::rand();

    bitonicSort(arr, s, e);

    ASSERT_TRUE(std::is_sorted(arr.begin() + s, arr.begin() + e));

    ASSERT_TRUE(arr[0] == -1); 
    ASSERT_TRUE(arr[std::rand() % s] == -1); 
    ASSERT_TRUE(arr[s-1] == -1); 

    ASSERT_TRUE(arr[e] == -1);
    ASSERT_TRUE(arr[e + (std::rand() % (size - e))] == -1);
    ASSERT_TRUE(arr.back() == -1); 
}
/*
void fetchAndSwapSorter(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> view)
{
    
    //auto Fetch = [=]__cuda_callable__(int i){return view[i];};
    //auto Cmp = [=]__cuda_callable__(const int & a, const int & b){return a < b;};
    //auto Swap = [=] __device__ (int i, int j){TNL::swap(view[i], view[j]);};
    //bitonicSort(0, view.getSize(), Fetch, Cmp, Swap);
    
}

TEST(fetchAndSwap, oneBlockSort)
{
    int size = 9;
    const int stride = 227;
    int i = 0;

    std::vector<int> orig(size);
    std::iota(orig.begin(), orig.end(), 0);

    do
    {
        if ((i++) % stride != 0)
            continue;

        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(orig);
        auto view = cudaArr.getView();
        fetchAndSwapSorter(view);
        ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
    }
    while (std::next_permutation(orig.begin(), orig.end()));
}
*/

//----------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}