#pragma once

#include <TNL/Containers/Array.h>
#include "task.h"

using CudaArrayView = TNL::Containers::ArrayView<int, TNL::Devices::Cuda>;
template<typename Function>
void quicksort(CudaArrayView arr, const Function & Cmp);

void quicksort(TNL::Containers::ArrayView<int, TNL::Devices::Cuda>arr);