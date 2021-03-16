/***************************************************************************
                          DistributedMatrix.h  -  description
                             -------------------
    begin                : Sep 10, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <type_traits>

#include <TNL/Containers/Subrange.h>
#include <TNL/Containers/DistributedVector.h>
#include <TNL/Containers/DistributedVectorView.h>
#include <TNL/Matrices/DistributedSpMV.h>

namespace TNL {
namespace Matrices {

// TODO: 2D distribution for dense matrices (maybe it should be in different template,
//       because e.g. setRowFast doesn't make sense for dense matrices)
template< typename Matrix >
class DistributedMatrix
{
public:
   using MatrixType = Matrix;
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using LocalRangeType = Containers::Subrange< typename Matrix::IndexType >;

   using RowsCapacitiesType = Containers::DistributedVector< IndexType, DeviceType, IndexType >;

   using MatrixRow = typename Matrix::RowViewType;
   using ConstMatrixRow = typename Matrix::ConstRowViewType;

   template< typename _Real = RealType,
             typename _Device = DeviceType,
             typename _Index = IndexType >
   using Self = DistributedMatrix< typename MatrixType::template Self< _Real, _Device, _Index > >;

   DistributedMatrix() = default;

   DistributedMatrix( DistributedMatrix& ) = default;

   DistributedMatrix( LocalRangeType localRowRange, IndexType rows, IndexType columns, MPI_Comm group = MPI::AllGroup() );

   void setDistribution( LocalRangeType localRowRange, IndexType rows, IndexType columns, MPI_Comm group = MPI::AllGroup() );

   const LocalRangeType& getLocalRowRange() const;

   MPI_Comm getCommunicationGroup() const;

   const Matrix& getLocalMatrix() const;

   Matrix& getLocalMatrix();


   /*
    * Some common Matrix methods follow below.
    */

   DistributedMatrix& operator=( const DistributedMatrix& matrix );

   template< typename MatrixT >
   DistributedMatrix& operator=( const MatrixT& matrix );

   template< typename MatrixT >
   void setLike( const MatrixT& matrix );

   void reset();

   IndexType getRows() const;

   IndexType getColumns() const;

   template< typename RowCapacitiesVector >
   void setRowCapacities( const RowCapacitiesVector& rowCapacities );

   template< typename Vector >
   void getCompressedRowLengths( Vector& rowLengths ) const;

   IndexType getRowCapacity( IndexType row ) const;

   void setElement( IndexType row,
                    IndexType column,
                    RealType value );

   RealType getElement( IndexType row,
                        IndexType column ) const;

   RealType getElementFast( IndexType row,
                            IndexType column ) const;

   MatrixRow getRow( IndexType row );

   ConstMatrixRow getRow( IndexType row ) const;

   // multiplication with a global vector
   template< typename InVector,
             typename OutVector >
   typename std::enable_if< ! HasGetCommunicationGroupMethod< InVector >::value >::type
   vectorProduct( const InVector& inVector,
                  OutVector& outVector ) const;

   // Optimization for distributed matrix-vector multiplication
   void updateVectorProductCommunicationPattern();

   // multiplication with a distributed vector
   // (not const because it modifies internal bufers)
   template< typename InVector,
             typename OutVector >
   typename std::enable_if< HasGetCommunicationGroupMethod< InVector >::value >::type
   vectorProduct( const InVector& inVector,
                  OutVector& outVector ) const;

   // FIXME: does not work for distributed matrices, here only due to common interface
   template< typename Vector1, typename Vector2 >
   bool performSORIteration( const Vector1& b,
                             const IndexType row,
                             Vector2& x,
                             const RealType& omega = 1.0 ) const;

protected:
   LocalRangeType localRowRange;
   IndexType rows = 0;  // global rows count
   MPI_Comm group = MPI::NullGroup();
   Matrix localMatrix;

   DistributedSpMV< Matrix > spmv;
};

} // namespace Matrices
} // namespace TNL

#include "DistributedMatrix_impl.h"
