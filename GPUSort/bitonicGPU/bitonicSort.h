#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/ParallelFor.h>

using namespace TNL;
using namespace TNL::Containers;

typedef Devices::Cuda Device;

//---------------------------------------------

__host__ __device__ int closestPow2(int x)
{
    if(x ==0)
        return 0;

    int ret = 1;
    while (ret < x)
        ret <<= 1;

    return ret;
}

//---------------------------------------------

void bitonicSort(ArrayView<int, Device> arr, int begin, int end, bool sortAscending)
{
    int arrSize = end - begin;
    int paddedSize = closestPow2(arrSize);

    auto CmpSwap = [=] __cuda_callable__ (int i, int monotonicSeqLen, int len, int partsInSeq) mutable {

        int part = i / (len / 2);

        int s = begin + part * len + (i % (len / 2));
        int e = s + len / 2;
        if (e >= end)
            return;

        //calculate the direction of swapping
        int monotonicSeqIdx = part / partsInSeq;
        bool ascending = (monotonicSeqIdx % 2) == 0 ? !sortAscending : sortAscending;

        //special case for parts with no "partner"
        if ((monotonicSeqIdx + 1) * monotonicSeqLen >= end)
            ascending = sortAscending;

        //templated size of block

        auto &a = arr[s];
        auto &b = arr[e];
        if ((ascending && a > b) || (!ascending && a < b))
            TNL::swap(a, b);
    };

    for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int len = monotonicSeqLen, partsInSeq = 1; len > 1; len /= 2, partsInSeq *= 2)
        {
            Algorithms::ParallelFor< Device>::exec(0, arrSize/2, CmpSwap, monotonicSeqLen, len, partsInSeq);
        }
    }
}


void bitonicSort(ArrayView<int, Device> arr, bool sortAscending = true)
{
    bitonicSort(arr, 0, arr.getSize(), sortAscending);
}