#include "gtest/gtest.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include <TNL/Containers/Array.h>

#include "../bitonicSort.h"

//----------------------------------------------------------------------------------
template <typename Value>
bool is_sorted(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr)
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
            TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(orig);
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

        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(orig);
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

        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(orig);
        auto view = cudaArr.getView();

        bitonicSort(view);

        ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
    }
}

TEST(selectedSize, size15)
{
    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr{5, 9, 4, 8, 6, 1, 2, 3, 4, 8, 1, 6, 9, 4, 9};
    auto view = cudaArr.getView();
    ASSERT_EQ(15, view.getSize());
    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
}

TEST(multiblock, 32768_decreasingNegative)
{
    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(1 << 15);
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
        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(std::rand()%(1<<10));
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

        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(size);

        auto view = cudaArr.getView();
        bitonicSort(view);
        ASSERT_TRUE(true);
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

/*
struct TMPSTRUCT{
    uint8_t m_data[6];
    TMPSTRUCT(){m_data[0] = 0;}
    TMPSTRUCT(int first){m_data[0] = first;};
    bool operator <(const TMPSTRUCT& other) const { return m_data[0] < other.m_data[0];}

    bool operator ==(const TMPSTRUCT& other) const {return !(*this < other) && !(other < *this); }

    bool operator >=(const TMPSTRUCT& other) const {return !(*this < other); }
    bool operator >(const TMPSTRUCT& other) const {return !(*this <= other); }
    bool operator <=(const TMPSTRUCT& other) const {return (*this < other) || (other == *this); }

    std::ostream& operator << (std::ostream & out) { return out << "{ " << m_data[0] << " }";}
};

TEST(nonIntegerType, struct)
{

    TNL::Containers::Array<TMPSTRUCT, TNL::Devices::Cuda> cudaArr{TMPSTRUCT(5), TMPSTRUCT(6), TMPSTRUCT(9), TMPSTRUCT(1)};
    auto view = cudaArr.getView();
    bitonicSort(view);
    ASSERT_TRUE(is_sorted(view)) << "result " << view << std::endl;
}
*/

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



//----------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}