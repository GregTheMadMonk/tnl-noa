/***************************************************************************
                          TridiagonalMatrixView.h  -  description
                             -------------------
    begin                : Jan 9, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/MatrixView.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/TridiagonalMatrixRowView.h>
#include <TNL/Containers/Segments/Ellpack.h>
#include <TNL/Matrices/details/TridiagonalMatrixIndexer.h>

namespace TNL {
namespace Matrices {

/**
 * \brief Implementation of sparse tridiagonal matrix.
 *
 * It serves as an accessor to \ref SparseMatrix for example when passing the
 * matrix to lambda functions. SparseMatrix view can be also created in CUDA kernels.
 *
 * See \ref TridiagonalMatrix for more details.
 * 
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam Organization tells the ordering of matrix elements. It is either RowMajorOrder
 *         or ColumnMajorOrder.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          ElementsOrganization Organization = Containers::Segments::DefaultElementsOrganization< Device >::getOrganization() >
class TridiagonalMatrixView : public MatrixView< Real, Device, Index >
{
   public:


      // Supporting types - they are not important for the user
      using BaseType = MatrixView< Real, Device, Index >;
      using ValuesViewType = typename BaseType::ValuesView;
      using IndexerType = details::TridiagonalMatrixIndexer< Index, Organization >;

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

      /**
       * \brief Type of related matrix view. 
       */
      using ViewType = TridiagonalMatrixView< Real, Device, Index, Organization >;

      /**
       * \brief Matrix view type for constant instances.
       */
      using ConstViewType = TridiagonalMatrixView< typename std::add_const< Real >::type, Device, Index, Organization >;

      /**
       * \brief Type for accessing matrix rows.
       */
      using RowView = TridiagonalMatrixRowView< ValuesViewType, IndexerType >;

      /**
       * \brief Helper type for getting self type or its modifications.
       */
      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index,
                ElementsOrganization Organization_ = Containers::Segments::DefaultElementsOrganization< Device >::getOrganization() >
      using Self = TridiagonalMatrixView< _Real, _Device, _Index, Organization_ >;

      /**
       * \brief Constructor with no parameters.
       */
      __cuda_callable__
      TridiagonalMatrixView();

      /**
       * \brief Constructor with all necessary data and views.
       * 
       * \param values is a vector view with matrix elements values
       * \param indexer is an indexer of matrix elements
       */
      __cuda_callable__
      TridiagonalMatrixView( const ValuesViewType& values, const IndexerType& indexer );

      /**
       * \brief Copy constructor.
       * 
       * \param matrix is an input tridiagonal matrix view.
       */
      __cuda_callable__
      TridiagonalMatrixView( const TridiagonalMatrixView& view ) = default;

      /**
       * \brief Move constructor.
       * 
       * \param matrix is an input tridiagonal matrix view.
       */
      __cuda_callable__
      TridiagonalMatrixView( TridiagonalMatrixView&& view ) = default;

      /**
       * \brief Returns a modifiable view of the tridiagonal matrix.
       * 
       * \return tridiagonal matrix view.
       */
      ViewType getView();

      /**
       * \brief Returns a non-modifiable view of the tridiagonal matrix.
       * 
       * \return tridiagonal matrix view.
       */
      ConstViewType getConstView() const;

      /**
       * \brief Returns string with serialization type.
       * 
       * The string has a form `Matrices::TridiagonalMatrix< RealType,  [any_device], IndexType, Organization, [any_allocator] >`.
       * 
       * See \ref TridiagonalMatrix::getSerializationType.
       * 
       * \return \ref String with the serialization type.
       */
      static String getSerializationType();

      /**
       * \brief Returns string with serialization type.
       * 
       * See \ref TridiagonalMatrix::getSerializationType.
       * 
       * \return \ref String with the serialization type.
       */
      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Computes number of non-zeros in each row.
       * 
       * \param rowLengths is a vector into which the number of non-zeros in each row
       * will be stored.
       * 
       * \par Example
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_getCompressedRowLengths.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_getCompressedRowLengths.out
       */
      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      //[[deprecated]]
      //IndexType getRowLength( const IndexType row ) const;

      //IndexType getMaxRowLength() const;

      /**
       * \brief Returns number of non-zero matrix elements.
       *
       * This method really counts the non-zero matrix elements and so
       * it returns zero for matrix having all allocated elements set to zero.
       *
       * \return number of non-zero matrix elements.
       */
      IndexType getNonzeroElementsCount() const;

      /**
       * \brief Comparison operator with another tridiagonal matrix.
       * 
       * \tparam Real_ is \e Real type of the source matrix.
       * \tparam Device_ is \e Device type of the source matrix.
       * \tparam Index_ is \e Index type of the source matrix.
       * \tparam Organization_ is \e Organization of the source matrix.
       * 
       * \return \e true if both matrices are identical and \e false otherwise.
       */
      template< typename Real_,
                typename Device_,
                typename Index_,
                ElementsOrganization Organization_ >
      bool operator == ( const TridiagonalMatrixView< Real_, Device_, Index_, Organization_ >& matrix ) const;

      /**
       * \brief Comparison operator with another multidiagonal matrix.
       * 
       * \tparam Real_ is \e Real type of the source matrix.
       * \tparam Device_ is \e Device type of the source matrix.
       * \tparam Index_ is \e Index type of the source matrix.
       * \tparam Organization_ is \e Organization of the source matrix.
       * 
       * \param matrix is the source matrix.
       * 
       * \return \e true if both matrices are NOT identical and \e false otherwise.
       */
      template< typename Real_,
                typename Device_,
                typename Index_,
                ElementsOrganization Organization_ >
      bool operator != ( const TridiagonalMatrixView< Real_, Device_, Index_, Organization_ >& matrix ) const;

      /**
       * \brief Non-constant getter of simple structure for accessing given matrix row.
       * 
       * \param rowIdx is matrix row index.
       * 
       * \return RowView for accessing given matrix row.
       * 
       * \par Example
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_getRow.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_getRow.out
       * 
       * See \ref TridiagonalMatrixRowView.
       */
      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );

      /**
       * \brief Constant getter of simple structure for accessing given matrix row.
       * 
       * \param rowIdx is matrix row index.
       * 
       * \return RowView for accessing given matrix row.
       *
       * \par Example
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_getConstRow.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_getConstRow.out
       * 
       * See \ref TridiagonalMatrixRowView.
       */
      __cuda_callable__
      const RowView getRow( const IndexType& rowIdx ) const;

      /**
       * \brief Set all matrix elements to given value.
       * 
       * \param value is the new value of all matrix elements.
       */
      void setValue( const RealType& v );

      /**
       * \brief Sets element at given \e row and \e column to given \e value.
       * 
       * This method can be called from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated on GPU this method
       * can be called even from device kernels. If the matrix is allocated in GPU device
       * this method is called from CPU, it transfers values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref TridiagonalMatrix::getRow
       * or \ref TridiagonalMatrix::forRows and \ref TridiagonalMatrix::forAllRows.
       * The call may fail if the matrix row capacity is exhausted.
       * 
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       * 
       * \par Example
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_setElement.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_setElement.out
       */
      void setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      /**
       * \brief Add element at given \e row and \e column to given \e value.
       * 
       * This method can be called from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated on GPU this method
       * can be called even from device kernels. If the matrix is allocated in GPU device
       * this method is called from CPU, it transfers values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref TridiagonalMatrix::getRow
       * or \ref TridiagonalMatrix::forRows and \ref TridiagonalMatrix::forAllRows.
       * The call may fail if the matrix row capacity is exhausted.
       * 
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       * \param thisElementMultiplicator is multiplicator the original matrix element
       *   value is multiplied by before addition of given \e value.
       * 
       * \par Example
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_addElement.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_addElement.out
       * 
       */
      void addElement( const IndexType row,
                       const IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator = 1.0 );

      /**
       * \brief Returns value of matrix element at position given by its row and column index.
       * 
       * This method can be called from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated on GPU this method
       * can be called even from device kernels. If the matrix is allocated in GPU device
       * this method is called from CPU, it transfers values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref TridiagonalMatrix::getRow
       * or \ref TridiagonalMatrix::forRows and \ref TridiagonalMatrix::forAllRows.
       * 
       * \param row is a row index of the matrix element.
       * \param column i a column index of the matrix element.
       * 
       * \return value of given matrix element.
       * 
       * \par Example
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_getElement.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_getElement.out
       * 
       */
      RealType getElement( const IndexType row,
                           const IndexType column ) const;

      /**
       * \brief Method for performing general reduction on matrix rows for constant instances.
       * 
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *          `fetch( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue`.
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
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_rowsReduction.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_rowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      /**
       * \brief Method for performing general reduction on matrix rows.
       * 
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *          `fetch( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue`.
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
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_rowsReduction.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_rowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero );

      /**
       * \brief Method for performing general reduction on all matrix rows for constant instances.
       * 
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *          `fetch( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue`.
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
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_allRowsReduction.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_allRowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      /**
       * \brief Method for performing general reduction on all matrix rows.
       * 
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *          `fetch( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue`.
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
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_allRowsReduction.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_allRowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero );

      /**
       * \brief Method for iteration over all matrix rows for constant instances.
       * 
       * \tparam Function is type of lambda function that will operate on matrix elements.
       *    It is should have form like
       *  `function( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value, bool& compute )`.
       *  The \e localIdx parameter is a rank of the non-zero element in given row. 
       *  If the 'compute' variable is set to false the iteration over the row can 
       *  be interrupted.
       * 
       * \param begin defines beginning of the range [begin,end) of rows to be processed.
       * \param end defines ending of the range [begin,end) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       * 
       * \par Example
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_forRows.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function ) const;

      /**
       * \brief Method for iteration over all matrix rows for non-constant instances.
       * 
       * \tparam Function is type of lambda function that will operate on matrix elements.
       *    It is should have form like
       *  `function( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value, bool& compute )`.
       *  The \e localIdx parameter is a rank of the non-zero element in given row. 
       *  If the 'compute' variable is set to false the iteration over the row can 
       *  be interrupted.
       * 
       * \param begin defines beginning of the range [begin,end) of rows to be processed.
       * \param end defines ending of the range [begin,end) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       * 
       * \par Example
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_forRows.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function );

      /**
       * \brief This method calls \e forRows for all matrix rows (for constant instances).
       * 
       * See \ref TridiagonalMatrix::forRows.
       * 
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       * 
       * \par Example
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_forAllRows.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_forAllRows.out
       */
      template< typename Function >
      void forAllRows( Function& function ) const;

      /**
       * \brief This method calls \e forRows for all matrix rows.
       * 
       * See \ref TridiagonalMatrix::forRows.
       * 
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       * 
       * \par Example
       * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_forAllRows.cpp
       * \par Output
       * \include TridiagonalMatrixViewExample_forAllRows.out
       */
      template< typename Function >
      void forAllRows( Function& function );

      /**
       * \brief Computes product of matrix and vector.
       * 
       * More precisely, it computes:
       * 
       * `outVector = matrixMultiplicator * ( * this ) * inVector + outVectorMultiplicator * outVector`
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
                          const RealType matrixMultiplicator = 1.0,
                          const RealType outVectorMultiplicator = 0.0,
                          const IndexType begin = 0,
                          IndexType end = 0 ) const;

      template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
      void addMatrix( const TridiagonalMatrixView< Real_, Device_, Index_, Organization_ >& matrix,
                      const RealType& matrixMultiplicator = 1.0,
                      const RealType& thisMatrixMultiplicator = 1.0 );

      template< typename Real2, typename Index2 >
      void getTransposition( const TridiagonalMatrixView< Real2, Device, Index2 >& matrix,
                             const RealType& matrixMultiplicator = 1.0 );

      template< typename Vector1, typename Vector2 >
      __cuda_callable__
      void performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      /**
       * \brief Assignment of exactly the same matrix type.
       * 
       * \param matrix is input matrix for the assignment.
       * \return reference to this matrix.
       */
      TridiagonalMatrixView& operator=( const TridiagonalMatrixView& view );

      /**
       * \brief Method for saving the matrix to a file.
       * 
       * \param file is the output file.
       */
      void save( File& file ) const;

      /**
       * \brief Method for saving the matrix to the file with given filename.
       * 
       * \param fileName is name of the file.
       */
      void save( const String& fileName ) const;

      /**
       * \brief Method for printing the matrix to output stream.
       * 
       * \param str is the output stream.
       */
      void print( std::ostream& str ) const;

      /**
       * \brief This method returns matrix elements indexer used by this matrix.
       * 
       * \return constant reference to the indexer.
       */
      __cuda_callable__
      const IndexerType& getIndexer() const;

      /**
       * \brief This method returns matrix elements indexer used by this matrix.
       * 
       * \return non-constant reference to the indexer.
       */
      __cuda_callable__
      IndexerType& getIndexer();

      /**
       * \brief Returns padding index denoting padding zero elements.
       * 
       * These elements are used for efficient data alignment in memory.
       * 
       * \return value of the padding index.
       */
      __cuda_callable__
      IndexType getPaddingIndex() const;

   protected:

      __cuda_callable__
      IndexType getElementIndex( const IndexType row,
                                 const IndexType localIdx ) const;

      IndexerType indexer;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/TridiagonalMatrixView.hpp>
