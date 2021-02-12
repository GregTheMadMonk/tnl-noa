#include "gtest/gtest.h"
#include <vector>
#include <algorithm>
#include <numeric>

#include <TNL/Containers/Array.h>

#include "../bitonicSort.h"

//----------------------------------------------------------------------------------

bool is_sorted(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> arr)
{
    for(int i = 1; i < arr.getSize(); i++)
        if(arr.getElement(i-1) > arr.getElement(i))
            return false;

    return true;
}

//----------------------------------------------------------------------------------

TEST(sortPow2, allPermutationSize4)
{
    int size = 4;
    std::vector<int> orig(size);
    std::iota(orig.begin(), orig.end(), 0);

    while (std::next_permutation(orig.begin(), orig.end()))
    {
        TNL::Containers::Array<int, Device> cudaArr(orig);
        auto view = cudaArr.getView();

        bitonicSort(view);

        ASSERT_TRUE(is_sorted(view));
    }
}

TEST(sortPow2, somePermutationSize8)
{
    int size = 8;
    const int stride = 5;
    int i = 0;

    std::vector<int> orig(size);
    std::iota(orig.begin(), orig.end(), 0);

    while (std::next_permutation(orig.begin(), orig.end()))
    {
        if((i++)%stride != 0)
            continue;

        TNL::Containers::Array<int, Device> cudaArr(orig);
        auto view = cudaArr.getView();

        bitonicSort(view);

        ASSERT_TRUE(is_sorted(view));
    }
}


TEST(selectedSize, size5)
{
    TNL::Containers::Array<int, Device> cudaArr{8, 1, 45, 9, -5};
    auto view = cudaArr.getView();
    ASSERT_EQ(5, view.getSize());
    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view));
}

TEST(selectedSize, size6)
{
    TNL::Containers::Array<int, Device> cudaArr{5, 9, 4, 3, 4, 0};
    auto view = cudaArr.getView();
    ASSERT_EQ(6, view.getSize());
    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view));
}


TEST(selectedSize, size7)
{
    TNL::Containers::Array<int, Device> cudaArr{5, 8, 1, 6, 9, 7, 1};
    auto view = cudaArr.getView();
    ASSERT_EQ(7, view.getSize());
    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view));
}

TEST(selectedSize, size9)
{
    TNL::Containers::Array<int, Device> cudaArr{5, 8, 1, 6, 9, 7, 1, 6, 0};
    auto view = cudaArr.getView();
    ASSERT_EQ(9, view.getSize());
    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view));
}



//----------------------------------------------------------------------------------

int main(int argc, char ** argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}