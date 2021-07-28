/***************************************************************************
                          task.h  -  description
                             -------------------
    begin                : Jul 13, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Xuan Thang Nguyen

#pragma once

#include <iostream>

#include <TNL/Cuda/CudaCallable.h>

namespace TNL {
    namespace Algorithms {
        namespace Sorting {

struct TASK
{
    //start and end position of array to read and write from
    int partitionBegin, partitionEnd;
    //-----------------------------------------------
    //helper variables for blocks working on this task

    int iteration;
    int pivotIdx;
    int dstBegin, dstEnd;
    int firstBlock, blockCount;//for workers read only values

    __cuda_callable__
    TASK(int begin, int end, int iteration)
        : partitionBegin(begin), partitionEnd(end),
        iteration(iteration), pivotIdx(-1),
        dstBegin(-151561), dstEnd(-151561),
        firstBlock(-100), blockCount(-100)
        {}

    __cuda_callable__
    void initTask(int firstBlock, int blocks, int pivotIdx)
    {
        dstBegin= 0; dstEnd = partitionEnd - partitionBegin;
        this->firstBlock = firstBlock;
        blockCount = blocks;
        this->pivotIdx = pivotIdx;
    }

    __cuda_callable__
    int getSize() const
    {
        return partitionEnd - partitionBegin;
    }

    TASK() = default;
};

inline std::ostream& operator<<(std::ostream & out, const TASK & task)
{
    out << "[ ";
    out << task.partitionBegin << " - " << task.partitionEnd;
    out << " | " << "iteration: " << task.iteration;
    out << " | " << "pivotIdx: " << task.pivotIdx;
    return out << " ] ";
}

        } // namespace Sorting
    } // namespace Algorithms
} // namespace TNL
