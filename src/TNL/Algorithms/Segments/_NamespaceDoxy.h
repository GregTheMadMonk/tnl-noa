/***************************************************************************
                          _NamespaceDoxy.h -  description
                             -------------------
    begin                : Apr 1, 2021
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Algorithms {
/**
 * \brief Namespace holding segments data structures.

 *Segments* represent data structure for manipulation with several local arrays (denoted also as segments)
 having different size in general. All the local arrays are supposed to be allocated in one continuos global array.
 The data structure segments offers mapping between indexes of particular local arrays and indexes
 of the global array. In addition,one can perform parallel operations like for or flexible reduction on partical
 local arrays.

 A typical example for use of *segments* is implementation of sparse matrices. Sparse matrix like the following
 \f[
  \left(
  \begin{array}{ccccc}
   1  &  0  &  2  &  0  &  0 \\
    0  &  0  &  5  &  0  &  0 \\
    3  &  4  &  7  &  9  &  0 \\
    0  &  0  &  0  &  0  & 12 \\
   0  &  0  & 15  & 17  & 20
  \end{array}
  \right)
 \f]
 is usually first compressed which means that the zero elements are omitted to get the following "matrix":

 \f[
 \begin{array}{ccccc}
    1  &   2  \\
    5   \\
    3  &   4  &  7 &  9   \\
    12 \\
    15 & 17  & 20
 \end{array}
 \f]
 We have to store column index of each matrix elements as well in a "matrix" like this:
 \f[
 \begin{array}{ccccc}
    0  &   2  \\
    2   \\
    0  &   1  &  2 &  3   \\
    4 \\
    2 & 3  & 4
 \end{array}
 \f]

 Such "matrices" can be stored in memory in a row-wise manner in one contiguous array because of the performance reasons. The first "matrix" (i.e. values of the matrix elements)
 would be stored as follows

 \f[
    \begin{array}{|cc|c|cccc|c|cc|} 1 & 2 &  5 & 3 & 4 & 7 & 9 & 12 & 15 & 17 & 20 \end{array}
 \f]

and the second one (i.e. column indexes of the matrix values) as follows

\f[
    \begin{array}{|cc|c|cccc|c|cc|} 0 & 2 & 2 & 0 & 1 & 2 & 3 & 4 & 2 & 3 & 4 \end{array}
 \f]

What we see above is so called [CSR sparse matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)).
It is the most popular format for storage of sparse matrices designed for high performance. However, it may not be the most efficient format for storage
of sparse matrices on GPUs. Therefore many other formats have been developed to get better performance. These formats often have different layout
of the matrix elements in the memory. They have to deal especially with two difficulties:

1. Efficient storage of matrix elements in the memory to fulfill the requirements of coalesced memory accesses on GPUs or good spatial locality
 for efficient use of caches on CPUs.
2. Efficient mapping of GPU threads to different matrix rows.

Necessity of working with this kind of data structure is not limited only to sparse matrices. We could name at least few others:

1. Efficient storage of [graphs](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)) - one segment represents one graph node,
   the elements in one segments are indexes of its neighbors.
2. [Unstructured numerical meshes](https://en.wikipedia.org/wiki/Types_of_mesh) - unstructured numerical mesh is a graph in fact.
3. [Particle in cell method](https://en.wikipedia.org/wiki/Particle-in-cell) - one segment represents one cell, the elements in one segment
   are indexes of the particles.
4. [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) - segments represent one cluster, the elements represent vectors
   belonging to given cluster.
5. [Hashing](https://arxiv.org/abs/1907.02900) - segments are particular rows of the hash table, elements in segments corresponds with coliding
   hashed elements.

In general, segments can be used for problems that somehow corresponds wit 2D data structure where each row can have different size and we need
to perform miscellaneous operations within the rows. The name *segments* comes from segmented parallel reduction or
[segmented scan (prefix-sum)](https://en.wikipedia.org/wiki/Segmented_scan).

The following example demonstrates the essence of *segments* in TNL:

\includelineno Algorithms/Segments/SegmentsExample_General.cpp

The result looks as follows:

\include SegmentsExample_General.out

*/



      namespace Segments {

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
