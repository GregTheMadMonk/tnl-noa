/***************************************************************************
                          LambdaMatrix.h -  description
                             -------------------
    begin                : Mar 17, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/Devices/AnyDevice.h>

namespace TNL {
namespace Matrices {

/**
 * \brief "Matrix-free" matrix based on lambda functions.
 * 
 * \tparam MatrixElementsLambda is a lambda function returning matrix elements
 *    values and positions.
 * \tparam CompressedRowLengthsLambda is a lambda function returning a number
 *    of non-zero elements in each row.
 * \tparam Real is a type of matrix elements values.
 * \tparam Device is a device on which the lambda functions can evaluated. 
 *    Devices::AnyDevice can be used for lambdas with no restriction.
 * \ẗparam Index is a type used for indexing.
 */
template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real = double,
          typename Device = Devices::AnyDevice,
          typename Index = int >
class LambdaMatrix
{
   public:
      static constexpr bool isSymmetric() { return false; };
      static constexpr bool isBinary() { return false; };

      using RealType = Real;
      using DeviceType = Device;
      using IndexType = Index;

      LambdaMatrix( MatrixElementsLambda& matrixElements,
                    CompressedRowLengthsLambda& compressedRowLentghs );

      LambdaMatrix( const IndexType& rows,
                    const IndexType& columns,
                    MatrixElementsLambda& matrixElements,
                    CompressedRowLengthsLambda& compressedRowLentghs );

      void setDimensions( const IndexType& rows,
                         const IndexType& columns );

      __cuda_callable__
      IndexType getRows() const;

      __cuda_callable__
      IndexType getColumns() const;

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      IndexType getNumberOfNonzeroMatrixElements() const;

      RealType getElement( const IndexType row,
                           const IndexType column ) const;

            template< typename Vector >
      __cuda_callable__
      typename Vector::RealType rowVectorProduct( const IndexType row,
                                                  const Vector& vector ) const;

      /***
       * \brief This method computes outVector = matrixMultiplicator * ( *this ) * inVector + inVectorAddition * inVector
       */
      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector,
                          const RealType& matrixMultiplicator = 1.0,
                          const RealType& outVectorMultiplicator = 0.0 ) const;

      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function ) const;

      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function );

      template< typename Function >
      void forAllRows( Function& function ) const;

      template< typename Function >
      void forAllRows( Function& function );

      template< typename Vector1, typename Vector2 >
      void performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      void print( std::ostream& str ) const;

   protected:

      IndexType rows, columns;

      MatrixElementsLambda matrixElementsLambda;

      CompressedRowLengthsLambda compressedRowLengthsLambda;
};


/**
 * \brief Helper class for creating instances of LambdaMatrix.
 * @param matrixElementsLambda
 * @param compressedRowLengthsLambda
 * @return 
 */
template< typename Real = double,
          typename Device = Devices::AnyDevice,
          typename Index = int >
struct LambdaMatrixFactory
{
   using RealType = Real;
   using IndexType = Index;
   
   template< typename MatrixElementsLambda,
             typename CompressedRowLengthsLambda >
   static auto create( MatrixElementsLambda& matrixElementsLambda,
                       CompressedRowLengthsLambda& compressedRowLengthsLambda )
   -> LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >
   {
      return LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >(
         matrixElementsLambda,
         compressedRowLengthsLambda );
   };
   
   template< typename MatrixElementsLambda,
             typename CompressedRowLengthsLambda >
   static auto create( const IndexType& rows,
                       const IndexType& columns,
                       MatrixElementsLambda& matrixElementsLambda,
                       CompressedRowLengthsLambda& compressedRowLengthsLambda )
   -> LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >
   {
      return LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >(
         matrixElementsLambda,
         compressedRowLengthsLambda );
   };
};

} //namespace Matrices
} //namespace TNL

#include <TNL/Matrices/LambdaMatrix.hpp>
