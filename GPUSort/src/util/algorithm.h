#pragma once
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/Reduction.h>

template <typename Value, typename Function>
bool is_sorted(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, const Function &Cmp)
{
    if(arr.getSize() <= 1) return true;

    auto fetch = [=] __cuda_callable__(int i) { return Cmp(arr[i - 1], arr[i]); };
    auto reduction = [] __cuda_callable__(bool a, bool b) { return a && b; };
    return TNL::Algorithms::Reduction<TNL::Devices::Cuda>::reduce(1, arr.getSize(), fetch, reduction, true);
}

template <typename Value>
bool is_sorted(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr)
{
    return is_sorted(arr, [] __cuda_callable__(const Value &a, const Value &b) { return a <= b; });
}
