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

} // namespace Matrices
} // namespace TNL

#include "SparseOperations_impl.h"
