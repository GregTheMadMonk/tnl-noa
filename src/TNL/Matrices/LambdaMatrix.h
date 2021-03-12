/***************************************************************************
                          LambdaMatrix.h -  description
                             -------------------
    begin                : Mar 17, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <TNL/String.h>
#include <TNL/Devices/Host.h>

namespace TNL {
namespace Matrices {

/**
 * \brief "Matrix-free matrix" based on lambda functions.
 *
 * The elements of this matrix are not stored explicitly in memory but
 * implicitly on a form of lambda functions.
 *
 * \tparam MatrixElementsLambda is a lambda function returning matrix elements values and positions.
 *
 *    It has the following form:
 *
 *   `matrixElements( Index rows, Index columns, Index rowIdx, Index localIdx, Index& columnIdx, Real& value )`
 *
 *    where \e rows is the number of matrix rows, \e columns is the number of matrix columns, \e rowIdx is the index of matrix row being queried,
 *    \e localIdx is the rank of the non-zero element in given row, \e columnIdx is a column index of the matrix element computed by
 *    this lambda and \e value is a value of the matrix element computed by this lambda.
 * \tparam CompressedRowLengthsLambda is a lambda function returning a number of non-zero elements in each row.
 *
 *    It has the following form:
 *
 *    `rowLengths( Index rows, Index columns, Index rowIdx ) -> IndexType`
 *
 *    where \e rows is the number of matrix rows, \e columns is the number of matrix columns and \e rowIdx is an index of the row being queried.
 *
 * \tparam Real is a type of matrix elements values.
 * \tparam Device is a device on which the lambda functions will be evaluated.
 * \ẗparam Index is a type to be used for indexing.
 */
template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class LambdaMatrix
{
   public:

      /**
       * \brief The type of matrix elements.
       */
      using RealType = Real;

      /**
       * \brief The device where the matrix is allocated.
       */
      using DeviceType = Device;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = Index;

      static constexpr bool isSymmetric() { return false; };
      static constexpr bool isBinary() { return false; };

      /**
       * \brief Constructor with lambda functions defining the matrix elements.
       *
       * Note: It might be difficult to express the types of the lambdas. For easier creation of
       * \e LambdaMatrix you may use \ref LambdaMatrixFactory.
       *
       * \param matrixElements is a lambda function giving matrix elements position and value.
       * \param compressedRowLentghs is a lambda function returning how many non-zero matrix elements are in given row.
       *
       * \par Example
       * \include Matrices/LambdaMatrix/LambdaMatrixExample_Constructor.cpp
       * \par Output
       * \include LambdaMatrixExample_Constructor.out
       */
      LambdaMatrix( MatrixElementsLambda& matrixElements,
                    CompressedRowLengthsLambda& compressedRowLentghs );

      /**
       * \brief Constructor with matrix dimensions and lambda functions defining the matrix elements.
       *
       * Note: It might be difficult to express the types of the lambdas. For easier creation of
       * \e LambdaMatrix you may use \ref LambdaMatrixFactory.
       *
       * \param rows is a number of the matrix rows.
       * \param columns is a number of the matrix columns.
       * \param matrixElements is a lambda function giving matrix elements position and value.
       * \param compressedRowLentghs is a lambda function returning how many non-zero matrix elements are in given row.
       *
       * \par Example
       * \include Matrices/LambdaMatrix/LambdaMatrixExample_Constructor.cpp
       * \par Output
       * \include LambdaMatrixExample_Constructor.out
       */
      LambdaMatrix( const IndexType& rows,
                    const IndexType& columns,
                    MatrixElementsLambda& matrixElements,
                    CompressedRowLengthsLambda& compressedRowLentghs );

      /**
       * \brief Copy constructor.
       *
       * \param matrix is input matrix.
       */
      LambdaMatrix( const LambdaMatrix& matrix ) = default;

      /**
       * \brief Move constructor.
       *
       * \param matrix is input matrix.
       */
      LambdaMatrix( LambdaMatrix&& matrix ) = default;

      /**
       * \brief Set number of rows and columns of this matrix.
       *
       * \param rows is the number of matrix rows.
       * \param columns is the number of matrix columns.
       */
      void setDimensions( const IndexType& rows,
                         const IndexType& columns );

      /**
       * \brief Returns a number of matrix rows.
       *
       * \return number of matrix rows.
       */
      __cuda_callable__
      IndexType getRows() const;

      /**
       * \brief Returns a number of matrix columns.
       *
       * \return number of matrix columns.
       */
      __cuda_callable__
      IndexType getColumns() const;

      /**
       * \brief Compute capacities of all rows.
       *
       * The row capacities are not stored explicitly and must be computed.
       *
       * \param rowCapacities is a vector where the row capacities will be stored.
       */
      template< typename Vector >
      void getRowCapacities( Vector& rowCapacities ) const;

      /**
       * \brief Computes number of non-zeros in each row.
       *
       * \param rowLengths is a vector into which the number of non-zeros in each row
       * will be stored.
       *
       * \par Example
       * \include Matrices/LambdaMatrix/LambdaMatrixExample_getCompressedRowLengths.cpp
       * \par Output
       * \include LambdaMatrixExample_getCompressedRowLengths.out
       */
      template< typename RowLentghsVector >
      void getCompressedRowLengths( RowLentghsVector& rowLengths ) const;

      /**
       * \brief Returns number of non-zero matrix elements.
       *
       * \return number of all non-zero matrix elements.
       *
       * \par Example
       * \include Matrices/LambdaMatrix/LambdaMatrixExample_getElementsCount.cpp
       * \par Output
       * \include LambdaMatrixExample_getElementsCount.out
       */
      IndexType getNonzeroElementsCount() const;

      /**
       * \brief Returns value of matrix element at position given by its row and column index.
       *
       * \param row is a row index of the matrix element.
       * \param column i a column index of the matrix element.
       *
       * \return value of given matrix element.
       */
      RealType getElement( const IndexType row,
                           const IndexType column ) const;

      /**
       * \brief Method for performing general reduction on matrix rows.
       *
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *          `fetch( IndexType rowIdx, IndexType columnIdx, RealType elementValue ) -> FetchValue`.
       *          The return type of this lambda can be any non void.
       * \tparam Reduce is a type of lambda function for reduction declared as
       *          `reduce( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue`.
       * \tparam Keep is a type of lambda function for storing results of reduction in each row.
       *          It is declared as `keep( const IndexType rowIdx, const double& value )`.
       * \tparam FetchValue is type returned by the Fetch lambda function.
       *
       * \param begin defines beginning of the range [begin,end) of rows to be processed.
       * \param end defines ending of the range [begin,end) of rows to be processed.
       * \param fetch is an instance of lambda function for data fetch.
       * \param reduce is an instance of lambda function for reduction.
       * \param keep in an instance of lambda function for storing results.
       * \param zero is zero of given reduction operation also known as idempotent element.
       *
       * \par Example
       * \include Matrices/LambdaMatrix/LambdaMatrixExample_rowsReduction.cpp
       * \par Output
       * \include LambdaMatrixExample_rowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void rowsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      /**
       * \brief Method for performing general reduction on ALL matrix rows.
       *
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *          `fetch( IndexType rowIdx, IndexType columnIdx, RealType elementValue ) -> FetchValue`.
       *          The return type of this lambda can be any non void.
       * \tparam Reduce is a type of lambda function for reduction declared as
       *          `reduce( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue`.
       * \tparam Keep is a type of lambda function for storing results of reduction in each row.
       *          It is declared as `keep( const IndexType rowIdx, const double& value )`.
       * \tparam FetchValue is type returned by the Fetch lambda function.
       *
       * \param fetch is an instance of lambda function for data fetch.
       * \param reduce is an instance of lambda function for reduction.
       * \param keep in an instance of lambda function for storing results.
       * \param zero is zero of given reduction operation also known as idempotent element.
       *
       * \par Example
       * \include Matrices/LambdaMatrix/LambdaMatrixExample_allRowsReduction.cpp
       * \par Output
       * \include LambdaMatrixExample_allRowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      /**
       * \brief Method for iteration over all matrix rows for constant instances.
       *
       * \tparam Function is type of lambda function that will operate on matrix elements.
       *    It is should have form like
       *  `function( IndexType rowIdx, IndexType columnIdx, IndexType columnIdx_, const RealType& value, bool& compute )`.
       *  The column index repeats twice only for compatibility with sparse matrices.
       *  If the 'compute' variable is set to false the iteration over the row can
       *  be interrupted.
       *
       * \param begin defines beginning of the range [begin,end) of rows to be processed.
       * \param end defines ending of the range [begin,end) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/LambdaMatrix/LambdaMatrixExample_forRows.cpp
       * \par Output
       * \include LambdaMatrixExample_forRows.out
       */
      template< typename Function >
      void forElements( IndexType first, IndexType last, Function& function ) const;

      /**
       * \brief This method calls \e forElements for all matrix rows (for constant instances).
       *
       * See \ref LambdaMatrix::forElements.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/LambdaMatrix/LambdaMatrixExample_forAllRows.cpp
       * \par Output
       * \include LambdaMatrixExample_forAllRows.out
       */
      template< typename Function >
      void forEachElement( Function& function ) const;

      /**
       * \brief Method for sequential iteration over all matrix rows for constant instances.
       *
       * \tparam Function is type of lambda function that will operate on matrix elements.
       *    It is should have form like
       *  `function( IndexType rowIdx, IndexType columnIdx, IndexType columnIdx_, const RealType& value, bool& compute )`.
       *  The column index repeats twice only for compatibility with sparse matrices.
       *  If the 'compute' variable is set to false the iteration over the row can
       *  be interrupted.
       *
       * \param begin defines beginning of the range [begin,end) of rows to be processed.
       * \param end defines ending of the range [begin,end) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void sequentialForRows( IndexType begin, IndexType end, Function& function ) const;

      /**
       * \brief This method calls \e sequentialForRows for all matrix rows (for constant instances).
       *
       * See \ref LambdaMatrix::sequentialForRows.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void sequentialForAllRows( Function& function ) const;

      /**
       * \brief Computes product of matrix and vector.
       *
       * More precisely, it computes:
       *
       * `outVector = matrixMultiplicator * ( *this ) * inVector + outVectorMultiplicator * outVector`
       *
       * \tparam InVector is type of input vector.  It can be \ref Vector,
       *     \ref VectorView, \ref Array, \ref ArraView or similar container.
       * \tparam OutVector is type of output vector. It can be \ref Vector,
       *     \ref VectorView, \ref Array, \ref ArraView or similar container.
       *
       * \param inVector is input vector.
       * \param outVector is output vector.
       * \param matrixMultiplicator is a factor by which the matrix is multiplied. It is one by default.
       * \param outVectorMultiplicator is a factor by which the outVector is multiplied before added
       *    to the result of matrix-vector product. It is zero by default.
       * \param begin is the beginning of the rows range for which the vector product
       *    is computed. It is zero by default.
       * \param end is the end of the rows range for which the vector product
       *    is computed. It is number if the matrix rows by default.
       */
      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector,
                          const RealType& matrixMultiplicator = 1.0,
                          const RealType& outVectorMultiplicator = 0.0,
                          const IndexType begin = 0,
                          IndexType end = 0 ) const;


      template< typename Vector1, typename Vector2 >
      void performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      /**
       * \brief Method for printing the matrix to output stream.
       *
       * \param str is the output stream.
       */
      void print( std::ostream& str ) const;

   protected:

      IndexType rows, columns;

      MatrixElementsLambda matrixElementsLambda;

      CompressedRowLengthsLambda compressedRowLengthsLambda;
};

/**
 * \brief Insertion operator for dense matrix and output stream.
 *
 * \param str is the output stream.
 * \param matrix is the lambda matrix.
 * \return reference to the stream.
 */
template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Device,
          typename Index >
std::ostream& operator<< ( std::ostream& str, const LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >& matrix );

/**
 * \brief Helper class for creating instances of LambdaMatrix.
 *
 * See \ref LambdaMatrix.
 *
 * \param matrixElementsLambda
 * \param compressedRowLengthsLambda
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
struct LambdaMatrixFactory
{
   using RealType = Real;
   using IndexType = Index;

   /**
    * \brief Creates lambda matrix with given lambda functions.
    *
    * \param matrixElementsLambda is a lambda function evaluating matrix elements.
    * \param compressedRowLengthsLambda is a lambda function returning number of
    *    non-zero matrix elements in given \e row.
    * \return instance of LambdaMatrix.
    *
    * \par Example
    * \include Matrices/LambdaMatrix/LambdaMatrixExample_Constructor.cpp
    * \par Output
    * \include LambdaMatrixExample_Constructor.out
    */
   template< typename MatrixElementsLambda,
             typename CompressedRowLengthsLambda >
   static auto create( MatrixElementsLambda& matrixElementsLambda,
                       CompressedRowLengthsLambda& compressedRowLengthsLambda )
   -> LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >
   {
      // TODO: fix the following asserts, they do not work in fact
      static_assert( std::is_same<
            std::enable_if_t< true, decltype(matrixElementsLambda( Index(), Index(), Index(), Index(), std::declval< Index& >(), std::declval< Real& >() ) ) >,
            void >::value,
         "Wong type of MatrixElementsLambda, it should be - matrixElementsLambda( Index rows, Index columns, Index rowIdx, Index localIdx, Index& columnIdx, Real& value )" );
      static_assert( std::is_integral<
         std::enable_if_t< true, decltype(compressedRowLengthsLambda( Index(), Index(), Index() ) ) >
          >::value ,
         "Wong type of CompressedRowLengthsLambda, it should be - matrixElementsLambda( Index rows, Index columns, Index rowIdx )" );
      return LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >(
         matrixElementsLambda,
         compressedRowLengthsLambda );
   };

   /**
    * \brief Creates lambda matrix with given dimensions and lambda functions.
    *
    * \param rows is number of matrix rows.
    * \param columns is number of matrix columns.
    * \param matrixElementsLambda is a lambda function evaluating matrix elements.
    * \param compressedRowLengthsLambda is a lambda function returning number of
    *    non-zero matrix elements in given \e row.
    * \return instance of LambdaMatrix.
    *
    * \par Example
    * \include Matrices/LambdaMatrix/LambdaMatrixExample_Constructor.cpp
    * \par Output
    * \include LambdaMatrixExample_Constructor.out
    */
   template< typename MatrixElementsLambda,
             typename CompressedRowLengthsLambda >
   static auto create( const IndexType& rows,
                       const IndexType& columns,
                       MatrixElementsLambda& matrixElementsLambda,
                       CompressedRowLengthsLambda& compressedRowLengthsLambda )
   -> LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >
   {
      // TODO: fix the following asserts, they do not work in fact
      static_assert( std::is_same<
            std::enable_if_t< true, decltype(matrixElementsLambda( Index(), Index(), Index(), Index(), std::declval< Index& >(), std::declval< Real& >() ) ) >,
            void >::value,
         "Wong type of MatrixElementsLambda, it should be - matrixElementsLambda( Index rows, Index columns, Index rowIdx, Index localIdx, Index& columnIdx, Real& value )" );
      static_assert( std::is_integral<
         std::enable_if_t< true, decltype(compressedRowLengthsLambda( Index(), Index(), Index() ) ) >
          >::value ,
         "Wong type of CompressedRowLengthsLambda, it should be - matrixElementsLambda( Index rows, Index columns, Index rowIdx )" );

      return LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >(
         rows, columns,
         matrixElementsLambda,
         compressedRowLengthsLambda );
   };
};

} //namespace Matrices
} //namespace TNL

#include <TNL/Matrices/LambdaMatrix.hpp>
