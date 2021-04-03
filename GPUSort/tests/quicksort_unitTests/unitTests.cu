#include "gtest/gtest.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/MemoryOperations.h>
#include "../../src/quicksort/quicksort.cuh"
#include "../../src/util/algorithm.h"

//----------------------------------------------------------------------------------

TEST(selectedSize, size15)
{
    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr{5, 9, 4, 8, 6, 1, 2, 3, 4, 8, 1, 6, 9, 4, 9};
    auto view = cudaArr.getView();
    ASSERT_EQ(15, view.getSize()) << "size not 15" << std::endl;
    quicksort(view);
    ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
}

TEST(multiblock, 32768_decreasingNegative)
{
    std::vector<int> arr(1<<15);
    for (size_t i = 0; i < arr.size(); i++)
        arr[i] = -i;
    
    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);
    auto view = cudaArr.getView();

    quicksort(view);
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
        quicksort(view);
        ASSERT_TRUE(is_sorted(view));
    }
}


TEST(randomGenerated, bigArray_randomVal)
{
    std::srand(304);
    for(int i = 0; i < 50; i++)
    {
        int size = (1<<20) + (std::rand()% (1<<19));
        std::vector<int> arr(size);
        for(auto & x : arr) x = std::rand();
        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);

        auto view = cudaArr.getView();
        quicksort(view);
        ASSERT_TRUE(is_sorted(view));
    }
}

TEST(noLostElement, smallArray)
{
    std::srand(9151);

    int size = (1<<7);
    std::vector<int> arr(size);
    for(auto & x : arr) x = std::rand();

    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);
    auto view = cudaArr.getView();
    quicksort(view);

    std::sort(arr.begin(), arr.end());
    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr2(arr);
    ASSERT_TRUE(view == cudaArr2.getView());
}

TEST(noLostElement, midSizedArray)
{
    std::srand(91503);

    int size = (1<<15);
    std::vector<int> arr(size);
    for(auto & x : arr) x = std::rand();

    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);
    auto view = cudaArr.getView();
    quicksort(view);

    std::sort(arr.begin(), arr.end());
    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr2(arr);
    ASSERT_TRUE(view == cudaArr2.getView());
}

TEST(noLostElement, bigSizedArray)
{
    std::srand(15611);

    int size = (1<<22);
    std::vector<int> arr(size);
    for(auto & x : arr) x = std::rand();
    for(int i = 0; i < 10000; i++)
        arr[std::rand() % arr.size()] = (1<<10);

    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);
    auto view = cudaArr.getView();
    quicksort(view);

    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr2(arr);
    thrust::sort(thrust::device, cudaArr2.getData(), cudaArr2.getData() + cudaArr2.getSize());
    ASSERT_TRUE(view == cudaArr2.getView());
}

//----------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}