/***************************************************************************
                          LambdaMatrix.hpp -  description
                             -------------------
    begin                : Mar 17, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/LambdaMatrix.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/LambdaMatrix.h>

namespace TNL {
namespace Matrices {

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
LambdaMatrix( MatrixElementsLambda& matrixElements,
              CompressedRowLengthsLambda& compressedRowLengths )
: rows( 0 ), columns( 0 ), matrixElementsLambda( matrixElements ), compressedRowLengthsLambda( compressedRowLengths )
{
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
LambdaMatrix( const IndexType& rows,
              const IndexType& columns,
              MatrixElementsLambda& matrixElements,
              CompressedRowLengthsLambda& compressedRowLengths )
: rows( rows ), columns( columns ), matrixElementsLambda( matrixElements ), compressedRowLengthsLambda( compressedRowLengths )
{
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
setDimensions( const IndexType& rows,
               const IndexType& columns )
{
   this->rows = rows;
   this->columns = columns;
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
getRows() const
{
   return this->rows;
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
getColumns() const
{
   return this->columns;
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   using Device_ = typename Devices::PickDevice< DeviceType >::DeviceType;
   
   rowLengths.setSize( this->getRows() );
   const IndexType rows = this->getRows();
   const IndexType columns = this->getColumns();
   auto rowLengthsView = rowLengths.getView();
   auto compressedRowLengths = this->compressedRowLengthsLambda;

   if( std::is_same< typename Vector::DeviceType, Device_ >::value )
      Algorithms::ParallelFor< Device_ >::exec(
         ( IndexType ) 0,
         this->getRows(),
         [=] __cuda_callable__ ( const IndexType row ) mutable {
            rowLengthsView[ row ] = compressedRowLengths( rows, columns, row );
         } );
   else
   {
      Containers::Vector< IndexType, Device_, IndexType > aux( this->getRows() );
      auto auxView = aux.getView();
      Algorithms::ParallelFor< Device_ >::exec(
         ( IndexType ) 0,
         this->getRows(),
         [=] __cuda_callable__ ( const IndexType row ) mutable {
            auxView[ row ] = compressedRowLengths( rows, columns, row );
         } );
      rowLengths = aux;
   }
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
Index
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
getNumberOfNonzeroMatrixElements() const
{
   Containers::Vector< IndexType, typename Devices::PickDevice< DeviceType >::DeviceType, IndexType > rowLengthsVector;
   this->getCompressedRowLengths( rowLengthsVector );
   return sum( rowLengthsVector );
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
Real
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
getElement( const IndexType row,
            const IndexType column ) const
{
   using Device_ = typename Devices::PickDevice< Devices::Host >::DeviceType;
   Containers::Array< RealType, Device_ > value( 1 );
   auto valueView = value.getView();
   auto rowLengths = this->compressedRowLengthsLambda;
   auto matrixElements = this->matrixElementsLambda;
   const IndexType rows = this->getRows();
   const IndexType columns = this->getColumns();
   auto getValue = [=] __cuda_callable__ (  IndexType rowIdx ) mutable {
      const IndexType rowSize = rowLengths( rows, columns, row );
      valueView[ 0 ] = 0.0;
      for( IndexType localIdx = 0; localIdx < rowSize; localIdx++ )
      {
         RealType elementValue;
         IndexType elementColumn;
         matrixElements( rows, columns, row, localIdx, elementColumn, elementValue );
         if( elementColumn == column )
         {
            valueView[ 0 ] = elementValue;
            break;
         }
      }
   };
   Algorithms::ParallelFor< Device_ >::exec( row, row + 1, getValue );
   return valueView.getElement( 0 );
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
      template< typename Vector >
__cuda_callable__
typename Vector::RealType
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
rowVectorProduct( const IndexType row,
                  const Vector& vector ) const
{
   
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
vectorProduct( const InVector& inVector,
               OutVector& outVector,
               const RealType& matrixMultiplicator,
               const RealType& outVectorMultiplicator ) const
{
   TNL_ASSERT_EQ( this->getColumns(), inVector.getSize(), "Matrix columns do not fit with input vector." );
   TNL_ASSERT_EQ( this->getRows(), outVector.getSize(), "Matrix rows do not fit with output vector." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   auto fetch = [=] __cuda_callable__ ( IndexType row, IndexType localIdx, IndexType columnIdx, const RealType& value ) mutable -> RealType {
      if( value == 0.0 )
         return 0.0;
      return value * inVectorView[ columnIdx ];
   };
   auto reduce = [] __cuda_callable__ ( RealType& sum, const RealType& value ) {
      sum += value;
   };
   auto keep = [=] __cuda_callable__ ( IndexType row, const RealType& value ) mutable {
      if( outVectorMultiplicator == 0.0 )
         outVectorView[ row ] = matrixMultiplicator * value;
      else
         outVectorView[ row ] = outVectorMultiplicator * outVectorView[ row ] + matrixMultiplicator * value;
   };
   this->allRowsReduction( fetch, reduce, keep, 0.0 );
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   using FetchType = decltype( fetch( IndexType(), IndexType(), IndexType(), RealType() ) );
   using Device_ = typename Devices::PickDevice< DeviceType >::DeviceType;

   const IndexType rows = this->getRows();
   const IndexType columns = this->getColumns();
   auto rowLengths = this->compressedRowLengthsLambda;
   auto matrixElements = this->matrixElementsLambda;
   auto processRow = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      const IndexType rowLength = rowLengths( rows, columns, rowIdx );
      FetchType result( zero );
      for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ )
      {
        IndexType elementColumn( 0 );
        RealType elementValue( 0.0 );
        matrixElements( rows, columns, rowIdx, localIdx, elementColumn, elementValue );
        FetchType fetchValue( zero );
        if( elementValue != 0.0 )
            fetchValue = fetch( rowIdx, localIdx, elementColumn, elementValue );
        reduce( result, fetchValue );
      }
      keep( rowIdx, result );
   };
   Algorithms::ParallelFor< Device_ >::exec( first, last, processRow );
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
   template< typename Function >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
forRows( IndexType first, IndexType last, Function& function ) const
{
   using FetchType = decltype( fetch( IndexType(), IndexType(), RealType(), IndexType() ) );
   using Device_ = typename Devices::PickDevice< DeviceType >::DeviceType;

   const IndexType rows = this->getRows();
   const IndexType columns = this->getColumns();
   auto rowLengths = this->compressedRowLengthsLambda;
   auto matrixElements = this->matrixElementsLambda;
   auto processRow = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      const IndexType rowLength = rowLengths( rows, columns, rowIdx );
      bool compute( true );
      for( IndexType localIdx = 0; localIdx < rowLength && compute; localIdx++ )
      {
        IndexType elementColumn( 0 );
        RealType elementValue( 0.0 );
        matrixElements( rows, columns, rowIdx, localIdx, elementColumn, elementValue );
        if( elementValue != 0.0 )
            function( rowIdx, localIdx, elementColumn, elementValue, compute );
      }
   };
   Algorithms::ParallelFor< Device_ >::exec( first, last, processRow );
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
   template< typename Function >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
forRows( IndexType first, IndexType last, Function& function )
{
   this->forRows( 0, this->getRows(), function );
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
   template< typename Vector1, typename Vector2 >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
performSORIteration( const Vector1& b,
                          const IndexType row,
                          Vector2& x,
                          const RealType& omega ) const
{
   
}

template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::
print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      for( IndexType column = 0; column < this->getColumns(); column++ )
      {
         auto value = this->getElement( row, column );
         if( value != ( RealType ) 0 )
            str << " Col:" << column << "->" << value << "\t";
      }
      str << std::endl;
   }
}

} //namespace Matrices
} //namespace TNL
