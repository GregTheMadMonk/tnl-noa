/***************************************************************************
                          Diagonal_impl.h  -  description
                             -------------------
    begin                : Dec 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include "Diagonal.h"

#include <TNL/Algorithms/ParallelFor.h>

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

template< typename Matrix >
void
Diagonal< Matrix >::
update( const MatrixPointer& matrixPointer )
{
   TNL_ASSERT_GT( matrixPointer->getRows(), 0, "empty matrix" );
   TNL_ASSERT_EQ( matrixPointer->getRows(), matrixPointer->getColumns(), "matrix must be square" );

   diagonal.setSize( matrixPointer->getRows() );

   VectorViewType diag_view( diagonal );

   const auto kernel_matrix = matrixPointer->getView();

   // TODO: Rewrite this with SparseMatrix::forAllRows
   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      diag_view[ i ] = kernel_matrix.getElement( i, i );
   };

   Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, diagonal.getSize(), kernel );
}

template< typename Matrix >
void
Diagonal< Matrix >::
solve( ConstVectorViewType b, VectorViewType x ) const
{
   ConstVectorViewType diag_view( diagonal );

   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      x[ i ] = b[ i ] / diag_view[ i ];
   };

   Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, diagonal.getSize(), kernel );
}


template< typename Matrix, typename Communicator >
void
Diagonal< Matrices::DistributedMatrix< Matrix, Communicator > >::
update( const MatrixPointer& matrixPointer )
{
   TNL_ASSERT_GT( matrixPointer->getRows(), 0, "empty matrix" );
   TNL_ASSERT_EQ( matrixPointer->getRows(), matrixPointer->getColumns(), "matrix must be square" );

   diagonal.setSize( matrixPointer->getLocalMatrix().getRows() );

   LocalViewType diag_view( diagonal );
   // FIXME: SparseMatrix::getConstView is broken
//   const auto matrix_view = matrixPointer->getLocalMatrix().getConstView();
   const auto matrix_view = matrixPointer->getLocalMatrix().getView();
   const auto row_range = matrixPointer->getLocalRowRange();

   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      const IndexType gi = row_range.getGlobalIndex( i );
      diag_view[ i ] = matrix_view.getElement( i, gi );
   };

   Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, diagonal.getSize(), kernel );
}

template< typename Matrix, typename Communicator >
void
Diagonal< Matrices::DistributedMatrix< Matrix, Communicator > >::
solve( ConstVectorViewType b, VectorViewType x ) const
{
   ConstLocalViewType diag_view( diagonal );
   const auto b_view = b.getConstLocalView();
   auto x_view = x.getLocalView();

   TNL_ASSERT_EQ( b_view.getSize(), diagonal.getSize(), "The size of the vector b does not match the size of the extracted diagonal." );
   TNL_ASSERT_EQ( x_view.getSize(), diagonal.getSize(), "The size of the vector x does not match the size of the extracted diagonal." );

   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      x_view[ i ] = b_view[ i ] / diag_view[ i ];
   };

   Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, diagonal.getSize(), kernel );
}

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL
