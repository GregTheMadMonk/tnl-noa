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

#include <type_traits>  // std::add_const

#include <TNL/Matrices/SparseRow.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/DistributedContainers/IndexMap.h>
#include <TNL/DistributedContainers/DistributedVector.h>

// buffers for vectorProduct
#include <vector>
#include <utility>  // std::pair
#include <TNL/Matrices/Dense.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
namespace DistributedContainers {

// TODO: 2D distribution for dense matrices (maybe it should be in different template,
//       because e.g. setRowFast doesn't make sense for dense matrices)
template< typename Matrix,
          typename Communicator = Communicators::MpiCommunicator,
          typename IndexMap = Subrange< typename Matrix::IndexType > >
class DistributedMatrix
: public Object
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;

   template< typename Real >
   using DistVector = DistributedVector< Real, typename Matrix::DeviceType, Communicator, typename Matrix::IndexType, IndexMap >;

public:
   using MatrixType = Matrix;
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using CommunicatorType = Communicator;
   using IndexMapType = IndexMap;

   using HostType = DistributedMatrix< typename Matrix::HostType, Communicator, IndexMap >;
   using CudaType = DistributedMatrix< typename Matrix::CudaType, Communicator, IndexMap >;

   using CompressedRowLengthsVector = DistributedVector< IndexType, DeviceType, CommunicatorType, IndexType, IndexMapType >;

   using MatrixRow = Matrices::SparseRow< RealType, IndexType >;
   using ConstMatrixRow = Matrices::SparseRow< typename std::add_const< RealType >::type, typename std::add_const< IndexType >::type >;

   DistributedMatrix() = default;

   DistributedMatrix( DistributedMatrix& ) = default;

   DistributedMatrix( IndexMap rowIndexMap, IndexType columns, CommunicationGroup group = Communicator::AllGroup );

   void setDistribution( IndexMap rowIndexMap, IndexType columns, CommunicationGroup group = Communicator::AllGroup );

   const IndexMap& getRowIndexMap() const;

   CommunicationGroup getCommunicationGroup() const;

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
   template< typename Vector,
             typename RealOut >
   void vectorProduct( const Vector& inVector,
                       DistVector< RealOut >& outVector ) const;

   // Optimization for matrix-vector multiplication:
   // - communication pattern matrix is an nproc x nproc binary matrix C, where
   //   C_ij = 1 iff the i-th process needs data from the j-th process
   // - assembly of the i-th row involves traversal of the local matrix stored
   //   in the i-th process
   // - assembly the full matrix needs all-to-all communication
   template< typename Partitioner >
   void updateVectorProductCommunicationPattern();

   // multiplication with a distributed vector
   // (not const because it modifies internal bufers)
   template< typename Partitioner,
             typename RealIn,
             typename RealOut >
   void vectorProduct( const DistVector< RealIn >& inVector,
                       DistVector< RealOut >& outVector );

protected:
   IndexMap rowIndexMap;
   CommunicationGroup group = Communicator::NullGroup;
   Matrix localMatrix;

   void resetBuffers()
   {
      commPattern.reset();
      globalBuffer.reset();
      commRequests.clear();
   }

   // communication pattern for matrix-vector product
   // TODO: probably should be stored elsewhere
   Matrices::Dense< bool, Devices::Host, int > commPattern;

   // span of rows with only block-diagonal entries
   std::pair< IndexType, IndexType > localOnlySpan;

   // global buffer for operations such as distributed matrix-vector multiplication
   // TODO: probably should be stored elsewhere
   Containers::Vector< RealType, DeviceType, IndexType > globalBuffer;

   // buffer for asynchronous communication requests
   // TODO: probably should be stored elsewhere
   std::vector< typename CommunicatorType::Request > commRequests;

private:
   // TODO: disabled until they are implemented
   using Object::save;
   using Object::load;
   using Object::boundLoad;
};

} // namespace DistributedContainers
} // namespace TNL

#include "DistributedMatrix_impl.h"
