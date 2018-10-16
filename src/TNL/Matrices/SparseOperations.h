/***************************************************************************
                          SparseOperations.h  -  description
                             -------------------
    begin                : Oct 4, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

// Note that these functions cannot be methods of the Sparse class, because
// their implementation uses methods (marked with __cuda_callable__) which are
// defined only on the subclasses, but are not virtual methods of Sparse.

#pragma once

namespace TNL {
namespace Matrices {

template< typename Matrix1, typename Matrix2 >
void copySparseMatrix( Matrix1& A, const Matrix2& B );

// NOTE: if `has_symmetric_pattern`, the sparsity pattern of `A` is assumed
// to be symmetric and it is just copied to `B`. Otherwise, the sparsity
// pattern of `A^T + A` is copied to `B`.
template< typename Matrix, typename AdjacencyMatrix >
void
copyAdjacencyStructure( const Matrix& A, AdjacencyMatrix& B,
                        bool has_symmetric_pattern = false,
                        bool ignore_diagonal = true );

// Applies a permutation to the rows of a sparse matrix and its inverse
// permutation to the columns of the matrix, i.e. A_perm = P*A*P^{-1}, where
// P is the permutation matrix represented by the perm vector and P^{-1} is the
// inverse permutation represented by the iperm vector.
template< typename Matrix1, typename Matrix2, typename PermutationVector >
void
reorderSparseMatrix( const Matrix1& A, Matrix2& A_perm,
                     const PermutationVector& perm, const PermutationVector& iperm );

// TODO: the method does not belong here, but there is no better place...
template< typename Vector, typename PermutationVector >
void
reorderVector( const Vector& src, Vector& dest, const PermutationVector& perm );

} // namespace Matrices
} // namespace TNL

#include "SparseOperations_impl.h"
