/***************************************************************************
                          DenseMatrix.h  -  description
                             -------------------
    begin                : Nov 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Allocators/Default.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/DenseMatrixRowView.h>
#include <TNL/Matrices/Matrix.h>
#include <TNL/Matrices/DenseMatrixView.h>
#include <TNL/Containers/Segments/Ellpack.h>

namespace TNL {
namespace Matrices {

/**
 * \brief Implementation of dense matrix, i.e. matrix storing explicitly all of its elements including zeros.
 * 
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam RowMajorOrder tells the ordering of matrix elements. If it is \e true the matrix elements
 *         are stored in row major order. If it is \e false, the matrix elements are stored in column major order.
 * \tparam RealAllocator is allocator for the matrix elements.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value,
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real > >
class DenseMatrix : public Matrix< Real, Device, Index >
{
   protected:
      using BaseType = Matrix< Real, Device, Index, RealAllocator >;
      using ValuesVectorType = typename BaseType::ValuesVectorType;
      using ValuesViewType = typename ValuesVectorType::ViewType;
      using SegmentsType = Containers::Segments::Ellpack< Device, Index, typename Allocators::Default< Device >::template Allocator< Index >, RowMajorOrder, 1 >;
      using SegmentViewType = typename SegmentsType::SegmentViewType;


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

      /**
       * \brief The allocator for matrix elements.
       */
      using RealAllocatorType = RealAllocator;

      /**
       * \brief Type of related matrix view. 
       * 
       * See \ref DenseMatrixView.
       */
      using ViewType = DenseMatrixView< Real, Device, Index, RowMajorOrder >;

      /**
       * \brief Matrix view type for constant instances.
       * 
       * See \ref DenseMatrixView.
       */
      using ConstViewType = DenseMatrixView< typename std::add_const< Real >::type, Device, Index, RowMajorOrder >;

      /**
       * \brief Type for accessing matrix row.
       */
      using RowView = DenseMatrixRowView< SegmentViewType, ValuesViewType >;

      /**
       * \brief Helper type for getting self type or its modifications.
       */
      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index,
                bool _RowMajorOrder = RowMajorOrder,
                typename _RealAllocator = RealAllocator >
      using Self = DenseMatrix< _Real, _Device, _Index, _RowMajorOrder, _RealAllocator >;

      /**
       * \brief Constructor without parameters.
       */
      DenseMatrix();

      /**
       * \brief Constructor with matrix dimensions.
       * 
       * \param rows is number of matrix rows.
       * \param columns is number of matrix columns.
       */
      DenseMatrix( const IndexType rows, const IndexType columns );

      /**
       * \brief Constructor with 2D initializer list.
       * 
       * The number of matrix rows is set to the outer list size and the number
       * of matrix columns is set to maximum size of inner lists. Missing elements
       * are filled in with zeros.
       * 
       * \param data is a initializer list of initializer lists representing
       * list of matrix rows.
       * 
       * \par Example
       * \include Matrices/DenseMatrixExample_Constructor_init_list.cpp
       * \par Output
       * \include DenseMatrixExample_Constructor_init_list.out
       */
      DenseMatrix( std::initializer_list< std::initializer_list< RealType > > data );

      /**
       * \brief Returns a modifiable view of the dense matrix.
       * 
       * See \ref DenseMatrixView.
       * 
       * \return dense matrix view.
       */
      ViewType getView();

      /**
       * \brief Returns a non-modifiable view of the dense matrix.
       * 
       * See \ref DenseMatrixView.
       * 
       * \return dense matrix view.
       */
      ConstViewType getConstView() const;

      /**
       * \brief Returns string with serialization type.
       * 
       * The string has a form \e `Matrices::DenseMatrix< RealType,  [any_device], IndexType, [any_allocator], true/false >`.
       * 
       * \return \e String with the serialization type.
       */
      static String getSerializationType();

      /**
       * \brief Returns string with serialization type.
       * 
       * See \ref DenseMatrix::getSerializationType.
       * 
       * \return \e String with the serialization type.
       */
      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Set number of rows and columns of this matrix.
       * 
       * \param rows is the number of matrix rows.
       * \param columns is the number of matrix columns.
       */
      void setDimensions( const IndexType rows,
                          const IndexType columns );

      /**
       * \brief Set the number of matrix rows and columns by the given matrix.
       * 
       * \tparam Matrix is matrix type. This can be any matrix having methods 
       *  \ref getRows and \ref getColumns.
       * 
       * \param matrix in the input matrix dimensions of which are to be adopted.
       */
      template< typename Matrix >
      void setLike( const Matrix& matrix );

      /**
       * \brief This method recreates the dense matrix from 2D initializer list.
       * 
       * The number of matrix rows is set to the outer list size and the number
       * of matrix columns is set to maximum size of inner lists. Missing elements
       * are filled in with zeros.
       * 
       * \param data is a initializer list of initializer lists representing
       * list of matrix rows.
       * 
       * \par Example
       * \include Matrices/DenseMatrixExample_setElements.cpp
       * \par Output
       * \include DenseMatrixExample_setElements.out
       */
      void setElements( std::initializer_list< std::initializer_list< RealType > > data );

      /**
       * \brief This method is only for the compatibility with the sparse matrices.
       * 
       * This method does nothing. In debug mode it contains assertions checking
       * that given rowCapacities are compatible with the current matrix dimensions.
       */
      template< typename RowCapacitiesVector >
      void setRowCapacities( const RowCapacitiesVector& rowCapacities );

      /**
       * \brief Computes number of non-zeros in each row.
       * 
       * \param rowLengths is a vector into which the number of non-zeros in each row
       * will be stored.
       * 
       * \par Example
       * \include Matrices/DenseMatrixExample_getCompressedRowLengths.cpp
       * \par Output
       * \include DenseMatrixExample_getCompressedRowLengths.out
       */
      template< typename RowLengthsVector >
      void getCompressedRowLengths( RowLengthsVector& rowLengths ) const;

      /**
       * \brief Returns number of all matrix elements.
       * 
       * This method is here mainly for compatibility with sparse matrices since
       * the number of all matrix elements is just number of rows times number of
       * columns.
       * 
       * \return number of all matrix elements.
       * 
       * \par Example
       * \include Matrices/DenseMatrixExample_getElementsCount.cpp
       * \par Output
       * \include DenseMatrixExample_getElementsCount.out
       */
      IndexType getElementsCount() const;

      /**
       * \brief Returns number of non-zero matrix elements.
       * 
       * \return number of all non-zero matrix elements.
       * 
       * \par Example
       * \include Matrices/DenseMatrixExample_getElementsCount.cpp
       * \par Output
       * \include DenseMatrixExample_getElementsCount.out
       */
      IndexType getNonzeroElementsCount() const;

      /**
       * \brief Resets the matrix to zero dimensions.
       */
      void reset();

      /**
       * \brief Constant getter of simple structure for accessing given matrix row.
       * 
       * \param rowIdx is matrix row index.
       * 
       * \return RowView for accessing given matrix row.
       *
       * \par Example
       * \include Matrices/DenseMatrixExample_getConstRow.cpp
       * \par Output
       * \include DenseMatrixExample_getConstRow.out
       * 
       * See \ref DenseMatrixRowView.
       */
      __cuda_callable__
      const RowView getRow( const IndexType& rowIdx ) const;

      /**
       * \brief Non-constant getter of simple structure for accessing given matrix row.
       * 
       * \param rowIdx is matrix row index.
       * 
       * \return RowView for accessing given matrix row.
       * 
       * \par Example
       * \include Matrices/DenseMatrixExample_getRow.cpp
       * \par Output
       * \include DenseMatrixExample_getRow.out
       * 
       * See \ref DenseMatrixRowView.
       */
      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );

      /**
       * \brief Sets all matrix elements to value \e v.
       * 
       * \param v is value all matrix elements will be set to.
       */
      void setValue( const RealType& v );

      /**
       * \brief Returns non-constant reference to element at row \e row and column column.
       * 
       * Since this method returns reference to the element, it cannot be called across
       * different address spaces. It means that it can be called only form CPU if the matrix
       * is allocated on CPU or only from GPU kernels if the matrix is allocated on GPU.
       * 
       * \param row is a row index of the element.
       * \param column is a columns index of the element. 
       * \return reference to given matrix element.
       */
      __cuda_callable__
      Real& operator()( const IndexType row,
                        const IndexType column );

      /**
       * \brief Returns constant reference to element at row \e row and column column.
       * 
       * Since this method returns reference to the element, it cannot be called across
       * different address spaces. It means that it can be called only form CPU if the matrix
       * is allocated on CPU or only from GPU kernels if the matrix is allocated on GPU.
       * 
       * \param row is a row index of the element.
       * \param column is a columns index of the element. 
       * \return reference to given matrix element.
       */
      __cuda_callable__
      const Real& operator()( const IndexType row,
                              const IndexType column ) const;

      /**
       * \brief Sets element at given \e row and \e column to given \e value.
       * 
       * This method can be called only from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated in GPU device
       * this methods transfer values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref DenseMatrix::getRow
       * or \ref DenseMatrix::forRows and \ref DenseMatrix::forAllRows.
       * 
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       * 
       * \par Example
       * \include Matrices/DenseMatrixExample_setElement.cpp
       * \par Output
       * \include DenseMatrixExample_setElement.out
       */
      void setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      /**
       * \brief Add element at given \e row and \e column to given \e value.
       * 
       * This method can be called only from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated in GPU device
       * this methods transfer values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref DenseMatrix::getRow
       * or \ref DenseMatrix::forRows and \ref DenseMatrix::forAllRows.
       * 
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       * \param thisElementMultiplicator is multiplicator the original matrix element
       *   value is multiplied by before addition of given e value.
       */
      void addElement( const IndexType row,
                       const IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator = 1.0 );

      /**
       * \brief Returns value of matrix element at position given by its row and column index.
       * 
       * This method can be called only from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated in GPU device
       * this methods transfer values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref DenseMatrix::getRow
       * or \ref DenseMatrix::forRows and \ref DenseMatrix::forAllRows.
       * 
       * \param row is a row index of the matrix element.
       * \param column i a column index of the matrix element.
       * 
       * \return value of given matrix element.
       */
      Real getElement( const IndexType row,
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
       * \param first is an index of the first row the reduction will be performed on.
       * \param last is an index of the row  after the last row the reduction will be performed on.
       * \param fetch is an instance of lambda function for data fetch.
       * \param reduce is an instance of lambda function for reduction.
       * \param keep in an instance of lambda function for storing results.
       * \param zero is zero of given reduction operation also known as idempotent element.
       * 
       * \par Example
       * \include Matrices/DenseMatrixExample_rowsReduction.cpp
       * \par Output
       * \include DenseMatrixExample_rowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
      void rowsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchValue& zero ) const;

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
       * \include Matrices/DenseMatrixExample_allRowsReduction.cpp
       * \par Output
       * \include DenseMatrixExample_allRowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      /**
       * \brief Method for iteration over all matrix rows for constant instances.
       * 
       * \tparam Function is type of lambda function that will operate on matrix elements.
       *    It is should have form like
       *  `function( IndexType rowIdx, IndexType columnIdx, IndexType columnIdx, const RealType& value, bool& compute )`.
       *  The column index repeats twice only for compatibility with sparse matrices. 
       *  If the 'compute' variable is set to false the iteration over the row can 
       *  be interrupted.
       * 
       * \param first is index is the first row to be processed.
       * \param last is index of the row after the last row to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       * 
       * \par Example
       * \include Matrices/DenseMatrixExample_forRows.cpp
       * \par Output
       * \include DenseMatrixExample_forRows.out
       */
      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function ) const;

      /**
       * \brief Method for iteration over all matrix rows for non-constant instances.
       * 
       * \tparam Function is type of lambda function that will operate on matrix elements.
       *    It is should have form like
       *  `function( IndexType rowIdx, IndexType columnIdx, IndexType columnIdx, RealType& value, bool& compute )`.
       *  The column index repeats twice only for compatibility with sparse matrices. 
       *  If the 'compute' variable is set to false the iteration over the row can 
       *  be interrupted.
       * 
       * \param first is index is the first row to be processed.
       * \param last is index of the row after the last row to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       * 
       * \par Example
       * \include Matrices/DenseMatrixExample_forRows.cpp
       * \par Output
       * \include DenseMatrixExample_forRows.out
       */
      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function );

      /**
       * \brief This method calls \e forRows for all matrix rows.
       * 
       * See \ref DenseMatrix::forRows.
       * 
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void forAllRows( Function& function ) const;

      /**
       * \brief This method calls \e forRows for all matrix rows.
       * 
       * See \ref DenseMatrix::forRows.
       * 
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void forAllRows( Function& function );

      /**
       * \brief This method computes scalar product of given vector and one 
       *  row of the matrix.
       * 
       * \tparam Vector is type of input vector. It can be \ref Vector,
       *     \ref VectorView, \ref Array, \ref ArraView or similar container.
       * \param row is index of the row used for the scalar product.
       * \param vector is the input vector.
       * \return 
       */
      template< typename Vector >
      __cuda_callable__
      typename Vector::RealType rowVectorProduct( const IndexType row,
                                                  const Vector& vector ) const;

      /**
       * \brief Computes product of matrix and vector.
       * 
       * \tparam InVector is type of input vector.  It can be \ref Vector,
       *     \ref VectorView, \ref Array, \ref ArraView or similar container.
       * \tparam OutVector is type of output vector. It can be \ref Vector,
       *     \ref VectorView, \ref Array, \ref ArraView or similar container.
       * 
       * \param inVector is input vector.
       * \param outVector is output vector.
       */
      template< typename InVector, typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const;

      template< typename Matrix >
      void addMatrix( const Matrix& matrix,
                      const RealType& matrixMultiplicator = 1.0,
                      const RealType& thisMatrixMultiplicator = 1.0 );

      template< typename Matrix1, typename Matrix2, int tileDim = 32 >
      void getMatrixProduct( const Matrix1& matrix1,
                          const Matrix2& matrix2,
                          const RealType& matrix1Multiplicator = 1.0,
                          const RealType& matrix2Multiplicator = 1.0 );

      template< typename Matrix, int tileDim = 32 >
      void getTransposition( const Matrix& matrix,
                             const RealType& matrixMultiplicator = 1.0 );

      template< typename Vector1, typename Vector2 >
      void performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      /**
       * \brief Assignment operator for exactly the same type of the dense matrix.
       * 
       * \param matrix is the right-hand side matrix.
       * \return reference to this matrix.
       */
      DenseMatrix& operator=( const DenseMatrix& matrix );

      /**
       * \brief Assignment operator for other dense matrices.
       * 
       * \param matrix is the right-hand side matrix.
       * \return reference to this matrix.
       */
      template< typename RHSReal, typename RHSDevice, typename RHSIndex,
                 bool RHSRowMajorOrder, typename RHSRealAllocator >
      DenseMatrix& operator=( const DenseMatrix< RHSReal, RHSDevice, RHSIndex, RHSRowMajorOrder, RHSRealAllocator >& matrix );

      /**
       * \brief Assignment operator for other (sparse) types of matrices.
       * 
       * \param matrix is the right-hand side matrix.
       * \return reference to this matrix.
       */
      template< typename RHSMatrix >
      DenseMatrix& operator=( const RHSMatrix& matrix );

      /**
       * \brief Comparison operator with another dense matrix.
       * 
       * \param matrix is the right-hand side matrix.
       * \return \e true if the RHS matrix is equal, \e false otherwise.
       */
      template< typename Real_, typename Device_, typename Index_, typename RealAllocator_ >
      bool operator==( const DenseMatrix< Real_, Device_, Index_, RowMajorOrder >& matrix ) const;

      /**
       * \brief Comparison operator with another dense matrix.
       * 
       * \param matrix is the right-hand side matrix.
       * \return \e false if the RHS matrix is equal, \e true otherwise.
       */
      template< typename Real_, typename Device_, typename Index_, typename RealAllocator_ >
      bool operator!=( const DenseMatrix< Real_, Device_, Index_, RowMajorOrder >& matrix ) const;

      /**
       * \brief Method for saving the matrix to the file with given filename.
       * 
       * \param fileName is name of the file.
       */
      void save( const String& fileName ) const;

      /**
       * \brief Method for loading the matrix from the file with given filename.
       * 
       * \param fileName is name of the file.
       */
      void load( const String& fileName );

      /**
       * \brief Method for saving the matrix to a file.
       * 
       * \param fileName is name of the file.
       */
      void save( File& file ) const;

      /**
       * \brief Method for loading the matrix from a file.
       * 
       * \param fileName is name of the file.
       */
      void load( File& file );

      /**
       * \brief Method for printing the matrix to output stream.
       * 
       * \param str is the output stream.
       */
      void print( std::ostream& str ) const;

   protected:

      __cuda_callable__
      IndexType getElementIndex( const IndexType row,
                                 const IndexType column ) const;

      SegmentsType segments;

      ViewType view;
};

/**
 * \brief Insertion operator for dense matrix and output stream.
 * 
 * \param str is the output stream.
 * \param matrix is the dense matrix.
 * \return  reference to the stream.
 */
template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
std::ostream& operator<< ( std::ostream& str, const DenseMatrix< Real, Device, Index, RowMajorOrder, RealAllocator >& matrix );

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/DenseMatrix.hpp>
