#include "gtest/gtest.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include <TNL/Containers/Array.h>

#include "../bitonicSort.h"

//----------------------------------------------------------------------------------

bool is_sorted(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr)
{
    for (int i = 1; i < arr.getSize(); i++)
        if (arr.getElement(i - 1) > arr.getElement(i))
            return false;

    return true;
}

//----------------------------------------------------------------------------------

TEST(permutations, allPermutationSize_3_to_7)
{
    for(int i = 3; i<=7; i++ )
    {
        int size = i;
        std::vector<int> orig(size);
        std::iota(orig.begin(), orig.end(), 0);

        while (std::next_permutation(orig.begin(), orig.end()))
        {
            TNL::Containers::Array<int, Device> cudaArr(orig);
            auto view = cudaArr.getView();

            bitonicSort(view);

            ASSERT_TRUE(is_sorted(view)) << "failed " << i << std::endl;
        } 
    }
}

TEST(permutations, somePermutationSize8)
{
    int size = 8;
    const int stride = 23;
    int i = 0;

    std::vector<int> orig(size);
    std::iota(orig.begin(), orig.end(), 0);

    while (std::next_permutation(orig.begin(), orig.end()))
    {
        if ((i++) % stride != 0)
            continue;

        TNL::Containers::Array<int, Device> cudaArr(orig);
        auto view = cudaArr.getView();

        bitonicSort(view);

        ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
    }
}

TEST(permutations, somePermutationSize9)
{
    int size = 9;
    const int stride = 227;
    int i = 0;

    std::vector<int> orig(size);
    std::iota(orig.begin(), orig.end(), 0);

    while (std::next_permutation(orig.begin(), orig.end()))
    {
        if ((i++) % stride != 0)
            continue;

        TNL::Containers::Array<int, Device> cudaArr(orig);
        auto view = cudaArr.getView();

        bitonicSort(view);

        ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
    }
}

TEST(selectedSize, size15)
{
    TNL::Containers::Array<int, Device> cudaArr{5, 9, 4, 8, 6, 1, 2, 3, 4, 8, 1, 6, 9, 4, 9};
    auto view = cudaArr.getView();
    ASSERT_EQ(15, view.getSize());
    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
}

TEST(multiblock, 32768_decreasingNegative)
{
    TNL::Containers::Array<int, Device> cudaArr(1 << 15);
    for (int i = 0; i < cudaArr.getSize(); i++)
        cudaArr.setElement(i, -i);

    auto view = cudaArr.getView();
    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
}

TEST(randomGenerated, smallArray_randomVal)
{
    for(int i = 0; i < 100; i++)
    {
        TNL::Containers::Array<int, Device> cudaArr(std::rand()%(1<<10));
        for (int j = 0; j < cudaArr.getSize(); j++)
            cudaArr.setElement(j, std::rand());

        auto view = cudaArr.getView();
        bitonicSort(view);
        ASSERT_TRUE(is_sorted(view));
    }
}

TEST(randomGenerated, bigArray_all0)
{
    for(int i = 0; i < 50; i++)
    {
        int size = (1<<20) + (std::rand()% (1<<19));

        TNL::Containers::Array<int, Device> cudaArr(size);

        auto view = cudaArr.getView();
        bitonicSort(view);
        ASSERT_TRUE(true);
    }
}

//----------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}