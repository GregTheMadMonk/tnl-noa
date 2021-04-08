#pragma once

struct TASK
{
    //start and end position of array to read and write from
    int partitionBegin, partitionEnd;
    //-----------------------------------------------
    //helper variables for blocks working on this task

    int depth;
    int pivotIdx;
    int dstBegin, dstEnd;
    int firstBlock, blockCount;//for workers read only values

    __cuda_callable__
    TASK(int begin, int end, int depth)
        : partitionBegin(begin), partitionEnd(end),
        depth(depth), pivotIdx(-1),
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

std::ostream& operator<<(std::ostream & out, const TASK & task)
{
    out << "[ ";
    out << task.partitionBegin << " - " << task.partitionEnd;
    out << " | " << "depth: " << task.depth;
    out << " | " << "pivotIdx: " << task.pivotIdx;
    return out << " ] ";
}