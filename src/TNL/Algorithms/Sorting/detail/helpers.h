/***************************************************************************
                          helpers.h  -  description
                             -------------------
    begin                : Jul 13, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Xuan Thang Nguyen

#pragma once
#include <TNL/Math.h>

namespace TNL {
    namespace Algorithms {
        namespace Sorting {

#ifdef HAVE_CUDA

// Inline PTX call to return index of highest non-zero bit in a word
static __device__ __forceinline__ unsigned int __btflo(unsigned int word)
{
    unsigned int ret;
    asm volatile("bfind.u32 %0, %1;"
                 : "=r"(ret)
                 : "r"(word));
    return ret;
}

__device__ int closestPow2_ptx(int bitonicLen)
{
    return 1 << (__btflo((unsigned)bitonicLen - 1U) + 1);
}

__host__ __device__ int closestPow2(int x)
{
    if (x == 0)
        return 0;

    int ret = 1;
    while (ret < x)
        ret <<= 1;

    return ret;
}

template <typename Value, typename CMP>
__cuda_callable__ void cmpSwap(Value &a, Value &b, bool ascending, const CMP &Cmp)
{
    if (ascending == Cmp(b, a))
        TNL::swap(a, b);
}

#endif
        } //namespace Sorting
    } //namespace Algorithms
} // namespace TNL