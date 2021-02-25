#pragma once
#include <TNL/Containers/Array.h>

template <typename Value>
bool is_sorted(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr)
{
    std::vector<Value> tmp(arr.getSize());
    TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Host, TNL::Devices::Cuda >::copy(tmp.data(), arr.getData(), arr.getSize());

    return std::is_sorted(tmp.begin(), tmp.end());
}
