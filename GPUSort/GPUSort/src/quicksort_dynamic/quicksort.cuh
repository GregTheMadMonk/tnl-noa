#pragma once

#include <TNL/Containers/Array.h>
#include "task.h"

using namespace TNL;
using namespace TNL::Containers;

template <typename Function>
__global__ void cudaQuickSort(ArrayView<int, Devices::Cuda> arr, ArrayView<int, Devices::Cuda> aux,
                              const Function &Cmp, int availBlocks, int depth);

template<typename Function>
void quicksort(ArrayView<int, Devices::Cuda> arr, const Function & Cmp);

void quicksort(ArrayView<int, Devices::Cuda>arr);