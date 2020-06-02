/***************************************************************************
                          Multidiagonal.h  -  description
                             -------------------
    begin                : Oct 13, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/Matrix.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/MultidiagonalMatrixRowView.h>
#include <TNL/Containers/Segments/Ellpack.h>
#include <TNL/Matrices/details/MultidiagonalMatrixIndexer.h>
#include <TNL/Matrices/MultidiagonalMatrixView.h>

namespace TNL {
namespace Matrices {

/**
 * \brief Implementation of sparse multi-diagonal matrix.
 * 
 * Use this matrix type for storing of matrices where the offsets of non-zero elements
 * from the diagonal are the same in each row. Typically such matrices arise from
 * discretization of partial differential equations on regular numerical grids.
 * 
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam Organization tells the ordering of matrix elements. It is either RowMajorOrder
 *         or ColumnMajorOrder.
 * \tparam RealAllocator is allocator for the matrix elements.
 * \tparam IndexAllocator is allocator for the matrix elements offsets.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          ElementsOrganization Organization = Containers::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real >,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class MultidiagonalMatrix : public Matrix< Real, Device, Index, RealAllocator >
{
   public:

      // Supporting types - they are not important for the user
      using BaseType = Matrix< Real, Device, Index, RealAllocator >;
      using ValuesVectorType = typename BaseType::ValuesVectorType;
      using ValuesViewType = typename ValuesVectorType::ViewType;
      using IndexerType = details::MultidiagonalMatrixIndexer< Index, Organization >;
      using DiagonalsOffsetsType = Containers::Vector< Index, Device, Index, IndexAllocator >;
      using DiagonalsOffsetsView = typename DiagonalsOffsetsType::ViewType;
      using HostDiagonalsOffsetsType = Containers::Vector< Index, Devices::Host, Index >;
      using HostDiagonalsOffsetsView = typename HostDiagonalsOffsetsType::ViewType;

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
       * \brief The allocator for matrix elements values.
       */
      using RealAllocatorType = RealAllocator;

      /**
       * \brief The allocator for matrix elements offsets from the diagonal.
       */
      using IndexAllocatorType = IndexAllocator;

      /**
       * \brief Type of related matrix view. 
       * 
       * See \ref MultidiagonalMatrixView.
       */
      using ViewType = MultidiagonalMatrixView< Real, Device, Index, Organization >;

      /**
       * \brief Matrix view type for constant instances.
       * 
       * See \ref MultidiagonalMatrixView.
       */
      using ConstViewType = MultidiagonalMatrixView< typename std::add_const< Real >::type, Device, Index, Organization >;

      /**
       * \brief Type for accessing matrix rows.
       */
      using RowView = MultidiagonalMatrixRowView< ValuesViewType, IndexerType, DiagonalsOffsetsView >;

      /**
       * \brief Type for accessing constant matrix rows.
       */
      using ConstRowView = typename RowView::ConstViewType;

      /**
       * \brief Helper type for getting self type or its modifications.
       */
      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index,
                ElementsOrganization _Organization = Organization,
                typename _RealAllocator = RealAllocator,
                typename _IndexAllocator = IndexAllocator >
      using Self = MultidiagonalMatrix< _Real, _Device, _Index, _Organization, _RealAllocator, _IndexAllocator >;

      /**
       * \brief Elements organization getter.
       */
      static constexpr ElementsOrganization getOrganization() { return Organization; };

      /**
       * \brief Constructor with no parameters.
       */
      MultidiagonalMatrix();

      /**
       * \brief Constructor with matrix dimensions.
       * 
       * \param rows is number of matrix rows.
       * \param columns is number of matrix columns.
       */
      MultidiagonalMatrix( const IndexType rows,
                           const IndexType columns );

      /**
       * \brief Constructor with matrix dimensions and matrix elements offsets.
       * 
       * \param rows is number of matrix rows.
       * \param columns is number of matrix columns.
       * \param diagonalsOffsets are offsets of subdiagonals from the main diagonal.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_Constructor.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_Constructor.out
       */
      template< typename Vector >
      MultidiagonalMatrix( const IndexType rows,
                           const IndexType columns,
                           const Vector& diagonalsOffsets );

      /**
       * \brief Constructor with matrix dimensions and diagonals offsets.
       * 
       * \param rows is number of matrix rows.
       * \param columns is number of matrix columns.
       * \param diagonalsOffsets are offsets of sub-diagonals from the main diagonal.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_Constructor_init_list_1.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_Constructor_init_list_1.out
       */
      template< typename ListIndex >
      MultidiagonalMatrix( const IndexType rows,
                           const IndexType columns,
                           const std::initializer_list< ListIndex > diagonalsOffsets );

      /**
       * \brief Constructor with matrix dimensions, diagonals offsets and matrix elements.
       * 
       * The number of matrix rows is given by the size of the initializer list \e data.
       * 
       * \param columns is number of matrix columns.
       * \param diagonalOffsets are offsets of sub-diagonals from the main diagonal.
       * \param data is initializer list holding matrix elements. The size of the outer list
       *    defines the number of matrix rows. Each inner list defines values of each sub-diagonal
       *    and so its size must be lower or equal to the size of \e diagonalsOffsets. Values
       *    of sub-diagonals which do not fit to given row are omitted.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_Constructor_init_list_2.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_Constructor_init_list_2.out
       */
      template< typename ListIndex, typename ListReal >
      MultidiagonalMatrix( const IndexType columns,
                           const std::initializer_list< ListIndex > diagonalsOffsets,
                           const std::initializer_list< std::initializer_list< ListReal > >& data );

      /**
       * \brief Copy constructor.
       * 
       * \param matrix is an input matrix.
       */
      MultidiagonalMatrix( const MultidiagonalMatrix& matrix ) = default;

      /**
       * \brief Move constructor.
       * 
       * \param matrix is an input matrix.
       */
      MultidiagonalMatrix( MultidiagonalMatrix&& matrix ) = default;

      /**
       * \brief Returns a modifiable view of the mutlidiagonal matrix.
       * 
       * See \ref MultidiagonalMatrixView.
       * 
       * \return multidiagonal matrix view.
       */
      ViewType getView() const; // TODO: remove const

      /**
       * \brief Returns a non-modifiable view of the multidiagonal matrix.
       * 
       * See \ref MultidiagonalMatrixView.
       * 
       * \return multidiagonal matrix view.
       */
      ConstViewType getConstView() const;

      /**
       * \brief Returns string with serialization type.
       * 
       * The string has a form `Matrices::MultidiagonalMatrix< RealType,  [any_device], IndexType, ElementsOrganization, [any_allocator], [any_allocator] >`.
       * 
       * \return \ref String with the serialization type.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_getSerializationType.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_getSerializationType.out
       */
      static String getSerializationType();

      /**
       * \brief Returns string with serialization type.
       * 
       * See \ref MultidiagonalMatrix::getSerializationType.
       * 
       * \return \e String with the serialization type.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_getSerializationType.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_getSerializationType.out
       */
      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Set matrix dimensions and diagonals offsets.
       * 
       * \tparam Vector is type of vector like container holding the diagonals offsets.
       * 
       * \param rows is number of matrix rows.
       * \param columns is number of matrix columns.
       * \param diagonalsOffsets is vector with diagonals offsets.
       */
      template< typename Vector >
      void setDimensions( const IndexType rows,
                          const IndexType columns,
                          const Vector& diagonalsOffsets );

      /**
       * \brief This method is for compatibility with \ref SparseMatrix.
       * 
       * It checks if the number of matrix diagonals is compatible with
       * required number of non-zero matrix elements in each row. If not
       * exception is thrown.
       * 
       * \tparam RowCapacitiesVector is vector-like container type for holding required 
       *    row capacities.
       * 
       * \param rowCapacities is vector-like container holding required row capacities.
       */
      template< typename RowCapacitiesVector >
      void setRowCapacities( const RowCapacitiesVector& rowCapacities );

      /**
       * \brief Set matrix elements from an initializer list.
       * 
       * \tparam ListReal is data type of the initializer list.
       * 
       * \param data is initializer list holding matrix elements. The size of the outer list
       *    defines the number of matrix rows. Each inner list defines values of each sub-diagonal
       *    and so its size must be lower or equal to the size of \e diagonalsOffsets. Values
       *    of sub-diagonals which do not fit to given row are omitted.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_setElements.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_setElements.out
       */
      template< typename ListReal >
      void setElements( const std::initializer_list< std::initializer_list< ListReal > >& data );

      /**
       * \brief Returns number of diagonals.
       * 
       * \return Number of diagonals.
       */
      const IndexType& getDiagonalsCount() const;

      /**
       * \brief Returns vector with diagonals offsets.
       * 
       * \return vector with diagonals offsets.
       */
      const DiagonalsOffsetsType& getDiagonalsOffsets() const;

      /**
       * \brief Computes number of non-zeros in each row.
       * 
       * \param rowLengths is a vector into which the number of non-zeros in each row
       * will be stored.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_getCompressedRowLengths.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_getCompressedRowLengths.out
       */
      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      [[deprecated]]
      IndexType getRowLength( const IndexType row ) const;

      /**
       * \brief Setup the matrix dimensions and diagonals offsets based on another multidiagonal matrix.
       * 
       * \tparam Real_ is \e Real type of the source matrix.
       * \tparam Device_ is \e Device type of the source matrix.
       * \tparam Index_ is \e Index type of the source matrix.
       * \tparam Organization_ is \e Organization of the source matrix.
       * \tparam RealAllocator_ is \e RealAllocator of the source matrix.
       * \tparam IndexAllocator_ is \e IndexAllocator of the source matrix.
       * 
       * \param matrix is the source matrix.
       */
      template< typename Real_,
                typename Device_,
                typename Index_,
                ElementsOrganization Organization_,
                typename RealAllocator_,
                typename IndexAllocator_ >
      void setLike( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix );

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
       * \brief Resets the matrix to zero dimensions.
       */
      void reset();

      /**
       * \brief Comparison operator with another multidiagonal matrix.
       * 
       * \tparam Real_ is \e Real type of the source matrix.
       * \tparam Device_ is \e Device type of the source matrix.
       * \tparam Index_ is \e Index type of the source matrix.
       * \tparam Organization_ is \e Organization of the source matrix.
       * \tparam RealAllocator_ is \e RealAllocator of the source matrix.
       * \tparam IndexAllocator_ is \e IndexAllocator of the source matrix.
       * 
       * \return \e true if both matrices are identical and \e false otherwise.
       */
      template< typename Real_,
                typename Device_,
                typename Index_,
                ElementsOrganization Organization_,
                typename RealAllocator_,
                typename IndexAllocator_ >
      bool operator == ( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix ) const;

      /**
       * \brief Comparison operator with another multidiagonal matrix.
       * 
       * \tparam Real_ is \e Real type of the source matrix.
       * \tparam Device_ is \e Device type of the source matrix.
       * \tparam Index_ is \e Index type of the source matrix.
       * \tparam Organization_ is \e Organization of the source matrix.
       * \tparam RealAllocator_ is \e RealAllocator of the source matrix.
       * \tparam IndexAllocator_ is \e IndexAllocator of the source matrix.
       * 
       * \param matrix is the source matrix.
       * 
       * \return \e true if both matrices are NOT identical and \e false otherwise.
       */
      template< typename Real_,
                typename Device_,
                typename Index_,
                ElementsOrganization Organization_,
                typename RealAllocator_,
                typename IndexAllocator_ >
      bool operator != ( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix ) const;

      /**
       * \brief Non-constant getter of simple structure for accessing given matrix row.
       * 
       * \param rowIdx is matrix row index.
       * 
       * \return RowView for accessing given matrix row.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_getRow.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_getRow.out
       * 
       * See \ref MultidiagonalMatrixRowView.
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
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_getConstRow.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_getConstRow.out
       * 
       * See \ref MultidiagonalMatrixRowView.
       */
      __cuda_callable__
      const RowView getRow( const IndexType& rowIdx ) const;

      /**
       * \brief Set all matrix elements to given value.
       * 
       * \param value is the new value of all matrix elements.
       */
      void setValue( const RealType& value );

      /**
       * \brief Sets element at given \e row and \e column to given \e value.
       * 
       * This method can be called from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated on GPU this method
       * can be called even from device kernels. If the matrix is allocated in GPU device
       * this method is called from CPU, it transfers values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref MultidiagonalMatrix::getRow
       * or \ref MultidiagonalMatrix::forRows and \ref MultidiagonalMatrix::forAllRows.
       * The call may fail if the matrix row capacity is exhausted.
       * 
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_setElement.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_setElement.out
       */
      __cuda_callable__
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
       * performance is very low. For higher performance see. \ref MultidiagonalMatrix::getRow
       * or \ref MultidiagonalMatrix::forRows and \ref MultidiagonalMatrix::forAllRows.
       * The call may fail if the matrix row capacity is exhausted.
       * 
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       * \param thisElementMultiplicator is multiplicator the original matrix element
       *   value is multiplied by before addition of given \e value.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_addElement.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_addElement.out
       * 
       */
      __cuda_callable__
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
       * performance is very low. For higher performance see. \ref MultidiagonalMatrix::getRow
       * or \ref MultidiagonalMatrix::forRows and \ref MultidiagonalMatrix::forAllRows.
       * 
       * \param row is a row index of the matrix element.
       * \param column i a column index of the matrix element.
       * 
       * \return value of given matrix element.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_getElement.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_getElement.out
       * 
       */
      __cuda_callable__
      RealType getElement( const IndexType row,
                           const IndexType column ) const;

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
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_rowsReduction.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_rowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero );

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
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_rowsReduction.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_rowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

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
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_allRowsReduction.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_allRowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero );

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
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_allRowsReduction.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_allRowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

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
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_forRows.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_forRows.out
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
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_forRows.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_forRows.out
       */
      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function );

      /**
       * \brief This method calls \e forRows for all matrix rows (for constant instances).
       * 
       * See \ref MultidiagonalMatrix::forRows.
       * 
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_forAllRows.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_forAllRows.out
       */
      template< typename Function >
      void forAllRows( Function& function ) const;

      /**
       * \brief This method calls \e forRows for all matrix rows.
       * 
       * See \ref MultidiagonalMatrix::forRows.
       * 
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_forAllRows.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_forAllRows.out
       */
      template< typename Function >
      void forAllRows( Function& function );

      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const;

      template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
      void addMatrix( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix,
                      const RealType& matrixMultiplicator = 1.0,
                      const RealType& thisMatrixMultiplicator = 1.0 );

      template< typename Real2, typename Index2 >
      void getTransposition( const MultidiagonalMatrix< Real2, Device, Index2 >& matrix,
                             const RealType& matrixMultiplicator = 1.0 );

      template< typename Vector1, typename Vector2 >
      __cuda_callable__
      void performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      // copy assignment
      MultidiagonalMatrix& operator=( const MultidiagonalMatrix& matrix );

      // cross-device copy assignment
      template< typename Real_,
                typename Device_,
                typename Index_,
                ElementsOrganization Organization_,
                typename RealAllocator_,
                typename IndexAllocator_ >
      MultidiagonalMatrix& operator=( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix );

      void save( File& file ) const;

      void load( File& file );

      void save( const String& fileName ) const;

      void load( const String& fileName );

      void print( std::ostream& str ) const;

      const IndexerType& getIndexer() const;

      IndexerType& getIndexer();

      __cuda_callable__
      IndexType getPaddingIndex() const;

   protected:

      __cuda_callable__
      IndexType getElementIndex( const IndexType row,
                                 const IndexType localIdx ) const;

      DiagonalsOffsetsType diagonalsOffsets;

      HostDiagonalsOffsetsType hostDiagonalsOffsets;

      IndexerType indexer;

      ViewType view;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MultidiagonalMatrix.hpp>
