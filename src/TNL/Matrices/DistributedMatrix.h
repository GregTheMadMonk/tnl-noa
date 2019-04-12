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

#include <TNL/Matrices/SparseRow.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Containers/Subrange.h>
#include <TNL/Containers/DistributedVector.h>
#include <TNL/Containers/DistributedVectorView.h>
#include <TNL/Matrices/DistributedSpMV.h>

namespace TNL {
namespace Matrices {

template< typename T, typename R = void >
struct enable_if_type
{
   using type = R;
};

template< typename T, typename Enable = void >
struct has_communicator : std::false_type {};

template< typename T >
struct has_communicator< T, typename enable_if_type< typename T::CommunicatorType >::type >
: std::true_type
{};


// TODO: 2D distribution for dense matrices (maybe it should be in different template,
//       because e.g. setRowFast doesn't make sense for dense matrices)
template< typename Matrix,
          typename Communicator = Communicators::MpiCommunicator >
class DistributedMatrix
: public Object
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
public:
   using MatrixType = Matrix;
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using CommunicatorType = Communicator;
   using LocalRangeType = Containers::Subrange< typename Matrix::IndexType >;

   using HostType = DistributedMatrix< typename Matrix::HostType, Communicator >;
   using CudaType = DistributedMatrix< typename Matrix::CudaType, Communicator >;

   using CompressedRowLengthsVector = Containers::DistributedVector< IndexType, DeviceType, IndexType, CommunicatorType >;

   using MatrixRow = Matrices::SparseRow< RealType, IndexType >;
   using ConstMatrixRow = Matrices::SparseRow< std::add_const_t< RealType >, std::add_const_t< IndexType > >;

   DistributedMatrix() = default;

   DistributedMatrix( DistributedMatrix& ) = default;

   DistributedMatrix( LocalRangeType localRowRange, IndexType rows, IndexType columns, CommunicationGroup group = Communicator::AllGroup );

   void setDistribution( LocalRangeType localRowRange, IndexType rows, IndexType columns, CommunicationGroup group = Communicator::AllGroup );

   __cuda_callable__
   const LocalRangeType& getLocalRowRange() const;

   __cuda_callable__
   CommunicationGroup getCommunicationGroup() const;

   __cuda_callable__
   const Matrix& getLocalMatrix() const;


   static String getType();

   virtual String getTypeVirtual() const;

   // TODO: no getSerializationType method until there is support for serialization


   /*
    * Some common Matrix methods follow below.
    */

   DistributedMatrix& operator=( const DistributedMatrix& matrix );

   template< typename MatrixT >
   DistributedMatrix& operator=( const MatrixT& matrix );

   template< typename MatrixT >
   void setLike( const MatrixT& matrix );

   void reset();

   __cuda_callable__
   IndexType getRows() const;

   __cuda_callable__
   IndexType getColumns() const;

   void setCompressedRowLengths( const CompressedRowLengthsVector& rowLengths );

   void getCompressedRowLengths( CompressedRowLengthsVector& rowLengths ) const;

   IndexType getRowLength( IndexType row ) const;

   bool setElement( IndexType row,
                    IndexType column,
                    RealType value );

   __cuda_callable__
   bool setElementFast( IndexType row,
                        IndexType column,
                        RealType value );

   RealType getElement( IndexType row,
                        IndexType column ) const;

   __cuda_callable__
   RealType getElementFast( IndexType row,
                            IndexType column ) const;

   __cuda_callable__
   bool setRowFast( IndexType row,
                    const IndexType* columnIndexes,
                    const RealType* values,
                    IndexType elements );

   __cuda_callable__
   void getRowFast( IndexType row,
                    IndexType* columns,
                    RealType* values ) const;

   __cuda_callable__
   MatrixRow getRow( IndexType row );

   __cuda_callable__
   ConstMatrixRow getRow( IndexType row ) const;

   // multiplication with a global vector
   template< typename InVector,
             typename OutVector >
   typename std::enable_if< ! has_communicator< InVector >::value >::type
   vectorProduct( const InVector& inVector,
                  OutVector& outVector ) const;

   // Optimization for distributed matrix-vector multiplication
   void updateVectorProductCommunicationPattern();

   // multiplication with a distributed vector
   // (not const because it modifies internal bufers)
   template< typename InVector,
             typename OutVector >
   typename std::enable_if< has_communicator< InVector >::value >::type
   vectorProduct( const InVector& inVector,
                  OutVector& outVector ) const;

protected:
   LocalRangeType localRowRange;
   IndexType rows = 0;  // global rows count
   CommunicationGroup group = Communicator::NullGroup;
   Matrix localMatrix;

   DistributedSpMV< Matrix, Communicator > spmv;

private:
   // TODO: disabled until they are implemented
   using Object::save;
   using Object::load;
   using Object::boundLoad;
};

} // namespace Matrices
} // namespace TNL

#include "DistributedMatrix_impl.h"
