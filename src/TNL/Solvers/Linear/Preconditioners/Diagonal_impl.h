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

#include <TNL/ParallelFor.h>

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
   const Matrix* kernel_matrix = &matrixPointer.template getData< DeviceType >();

   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      diag_view[ i ] = kernel_matrix->getElementFast( i, i );
   };

   ParallelFor< DeviceType >::exec( (IndexType) 0, diagonal.getSize(), kernel );
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

   ParallelFor< DeviceType >::exec( (IndexType) 0, diagonal.getSize(), kernel );
}


template< typename Matrix, typename Communicator >
void
Diagonal< Matrices::DistributedMatrix< Matrix, Communicator > >::
update( const MatrixPointer& matrixPointer )
{
   TNL_ASSERT_GT( matrixPointer->getRows(), 0, "empty matrix" );
   TNL_ASSERT_EQ( matrixPointer->getRows(), matrixPointer->getColumns(), "matrix must be square" );

   diagonal.setSize( matrixPointer->getLocalMatrix().getRows() );

   LocalVectorViewType diag_view( diagonal );
   const MatrixType* kernel_matrix = &matrixPointer.template getData< DeviceType >();

   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      const IndexType gi = kernel_matrix->getLocalRowRange().getGlobalIndex( i );
      diag_view[ i ] = kernel_matrix->getLocalMatrix().getElementFast( i, gi );
   };

   ParallelFor< DeviceType >::exec( (IndexType) 0, diagonal.getSize(), kernel );
}

template< typename Matrix, typename Communicator >
void
Diagonal< Matrices::DistributedMatrix< Matrix, Communicator > >::
solve( ConstVectorViewType b, VectorViewType x ) const
{
   ConstLocalVectorViewType diag_view( diagonal );
   const auto b_view = b.getLocalVectorView();
   auto x_view = x.getLocalVectorView();

   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      x_view[ i ] = b_view[ i ] / diag_view[ i ];
   };

   ParallelFor< DeviceType >::exec( (IndexType) 0, diagonal.getSize(), kernel );
}

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL
