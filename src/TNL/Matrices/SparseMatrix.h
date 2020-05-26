/***************************************************************************
                          SparseMatrix.h -  description
                             -------------------
    begin                : Nov 29, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <map>
#include <TNL/Matrices/Matrix.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Allocators/Default.h>
#include <TNL/Containers/Segments/CSR.h>
#include <TNL/Matrices/SparseMatrixRowView.h>
#include <TNL/Matrices/SparseMatrixView.h>
#include <TNL/Matrices/DenseMatrix.h>

namespace TNL {
namespace Matrices {

/**
 * \brief Implementation of sparse matrix, i.e. matrix storing only non-zero elements.
 * 
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam MatrixType specifies the type of matrix - its symmetry or binarity. See \ref MatrixType.
 *    Both symmetric and binary matrix types reduces memory consumption. Binary matrix does not store
 *    the matrix values explicitly since the non-zero elements can have only value equal to one. Symmetric
 *    matrices store only lower part of the matrix and its diagonal. The upper part is reconstructed on the fly.
 *    GeneralMatrix with no symmetry is used by default.
 * \tparam Segments is a structure representing the sparse matrix format. Depending on the pattern of the non-zero elements
 *    different matrix formats can perform differently especially on GPUs. By default \ref CSR format is used. See also
 *    \ref Ellpack, \ref SlicedEllpack, \ref ChunkedEllpack or \ref BiEllpack.
 * \tparam RealAllocator is allocator for the matrix elements values.
 * \tparam IndexAllocator is allocator for the matrix elements column indexes.
 */
template< typename Real,
          typename Device = Devices::Host,
          typename Index = int,
          typename MatrixType = GeneralMatrix,
          template< typename Device_, typename Index_, typename IndexAllocator_ > class Segments = Containers::Segments::CSR,
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real >,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class SparseMatrix : public Matrix< Real, Device, Index, RealAllocator >
{
   static_assert(
         ! MatrixType::isSymmetric() ||
         ! std::is_same< Device, Devices::Cuda >::value ||
         ( std::is_same< Real, float >::value || std::is_same< Real, double >::value || std::is_same< Real, int >::value || std::is_same< Real, long long int >::value ),
         "Given Real type is not supported by atomic operations on GPU which are necessary for symmetric operations." );

   public:

      // Supporting types - they are not important for the user
      using BaseType = Matrix< Real, Device, Index, RealAllocator >;
      using ValuesVectorType = typename Matrix< Real, Device, Index, RealAllocator >::ValuesVectorType;
      using ValuesViewType = typename ValuesVectorType::ViewType;
      using ConstValuesViewType = typename ValuesViewType::ConstViewType;
      using ColumnsIndexesVectorType = Containers::Vector< Index, Device, Index, IndexAllocator >;
      using ColumnsIndexesViewType = typename ColumnsIndexesVectorType::ViewType;
      using ConstColumnsIndexesViewType = typename ColumnsIndexesViewType::ConstViewType;
      using RowsCapacitiesType = Containers::Vector< Index, Device, Index, IndexAllocator >;
      using RowsCapacitiesView = Containers::VectorView< Index, Device, Index >;
      using ConstRowsCapacitiesView = typename RowsCapacitiesView::ConstViewType;

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
       * \brief Templated type of segments, i.e. sparse matrix format.
       */
      template< typename Device_, typename Index_, typename IndexAllocator_ >
      using SegmentsTemplate = Segments< Device_, Index_, IndexAllocator_ >;

      /**
       * \brief Type of segments used by this matrix. It represents the sparse matrix format.
       */
      using SegmentsType = Segments< Device, Index, IndexAllocator >;

      /**
       * \brief Templated view type of segments, i.e. sparse matrix format.
       */
      template< typename Device_, typename Index_ >
      using SegmentsViewTemplate = typename SegmentsType::template ViewTemplate< Device_, Index >;

      /**
       * \brief Type of segments view used by the related matrix view. It represents the sparse matrix format.
       */
      using SegmentsViewType = typename SegmentsType::ViewType;

      /**
       * \brief The allocator for matrix elements values.
       */
      using RealAllocatorType = RealAllocator;

      /**
       * \brief The allocator for matrix elements column indexes.
       */
      using IndexAllocatorType = IndexAllocator;

      /**
       * \brief Type of related matrix view. 
       * 
       * See \ref SparseMatrixView.
       */
      using ViewType = SparseMatrixView< Real, Device, Index, MatrixType, SegmentsViewTemplate >;

      /**
       * \brief Matrix view type for constant instances.
       * 
       * See \ref SparseMatrixView.
       */
      using ConstViewType = SparseMatrixView< std::add_const_t< Real >, Device, Index, MatrixType, SegmentsViewTemplate >;

      //using SegmentViewType = typename SegmentsType::SegmentViewType;

      /**
       * \brief Type for accessing matrix rows.
       */
      using RowView = SparseMatrixRowView< typename SegmentsType::SegmentViewType, ValuesViewType, ColumnsIndexesViewType, MatrixType::isBinary() >;

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
                typename _MatrixType = MatrixType,
                template< typename, typename, typename > class _Segments = Segments,
                typename _RealAllocator = typename Allocators::Default< _Device >::template Allocator< _Real >,
                typename _IndexAllocator = typename Allocators::Default< _Device >::template Allocator< _Index > >
      using Self = SparseMatrix< _Real, _Device, _Index, _MatrixType, _Segments, _RealAllocator, _IndexAllocator >;

      /**
       * \brief Test of symmetric matrix type.
       * 
       * \return \e true if the matrix is stored as symmetric and \e false otherwise.
       */
      static constexpr bool isSymmetric() { return MatrixType::isSymmetric(); };

      /**
       * \brief Test of binary matrix type.
       * 
       * \return \e true if the matrix is stored as binary and \e false otherwise.
       */
      static constexpr bool isBinary() { return MatrixType::isBinary(); };

      /**
       * \brief Constructor only with values and column indexes allocators.
       * 
       * \param realAllocator is used for allocation of matrix elements values.
       * \param indexAllocator is used for allocation of matrix elements column indexes.
       */
      SparseMatrix( const RealAllocatorType& realAllocator = RealAllocatorType(),
                    const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

      /**
       * \brief Copy constructor.
       * 
       * \param matrix is the source matrix
       */
      SparseMatrix( const SparseMatrix& matrix1 ) = default;

      /**
       * \brief Move constructor.
       * 
       * \param matrix is the source matrix
       */
      SparseMatrix( SparseMatrix&& matrix ) = default;

      /**
       * \brief Constructor with matrix dimensions.
       * 
       * \param rows is number of matrix rows.
       * \param columns is number of matrix columns.
       * \param realAllocator is used for allocation of matrix elements values.
       * \param indexAllocator is used for allocation of matrix elements column indexes.
       */
      SparseMatrix( const IndexType rows,
                    const IndexType columns,
                    const RealAllocatorType& realAllocator = RealAllocatorType(),
                    const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

      /**
       * \brief Constructor with matrix rows capacities and number of columns.
       * 
       * The number of matrix rows is given by the size of \e rowCapacities list.
       * 
       * \tparam ListIndex is the initializer list values type.
       * \param rowCapacities is a list telling how many matrix elements must be
       *    allocated in each row.
       * \param columns is the number of matrix columns.
       * \param realAllocator is used for allocation of matrix elements values.
       * \param indexAllocator is used for allocation of matrix elements column indexes.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_Constructor_init_list_1.cpp
       * \par Output
       * \include SparseMatrixExample_Constructor_init_list_1.out
       */
      template< typename ListIndex >
      explicit SparseMatrix( const std::initializer_list< ListIndex >& rowCapacities,
                             const IndexType columns,
                             const RealAllocatorType& realAllocator = RealAllocatorType(),
                             const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

      /**
       * \brief Constructor with matrix dimensions and data in initializer list.
       * 
       * The matrix elements values are given as a list \e data of triples:
       * { { row1, column1, value1 },
       *   { row2, column2, value2 },
       * ... }.
       * 
       * \param rows is number of matrix rows.
       * \param columns is number of matrix columns.
       * \param data is a list of matrix elements values.
       * \param realAllocator is used for allocation of matrix elements values.
       * \param indexAllocator is used for allocation of matrix elements column indexes.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_Constructor_init_list_2.cpp
       * \par Output
       * \include SparseMatrixExample_Constructor_init_list_2.out
       */
      explicit SparseMatrix( const IndexType rows,
                             const IndexType columns,
                             const std::initializer_list< std::tuple< IndexType, IndexType, RealType > >& data,
                             const RealAllocatorType& realAllocator = RealAllocatorType(),
                             const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

      /**
       * \brief Constructor with matrix dimensions and data in std::map.
       * 
       * The matrix elements values are given as a map \e data where keys are
       * std::pair of matrix coordinates ( {row, column} ) and value is the
       * matrix element value.
       * 
       * \tparam MapIndex is a type for indexing rows and columns.
       * \tparam MapValue is a type for matrix elements values in the map.
       * 
       * \param rows is number of matrix rows.
       * \param columns is number of matrix columns.
       * \param map is std::map containing matrix elements.
       * \param realAllocator is used for allocation of matrix elements values.
       * \param indexAllocator is used for allocation of matrix elements column indexes.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_Constructor_std_map.cpp
       * \par Output
       * \include SparseMatrixExample_Constructor_std_map.out
       */
      template< typename MapIndex,
                typename MapValue >
      explicit SparseMatrix( const IndexType rows,
                             const IndexType columns,
                             const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
                             const RealAllocatorType& realAllocator = RealAllocatorType(),
                             const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

      /**
       * \brief Returns a modifiable view of the sparse matrix.
       * 
       * See \ref SparseMatrixView.
       * 
       * \return sparse matrix view.
       */
      ViewType getView() const; // TODO: remove const

      /**
       * \brief Returns a non-modifiable view of the sparse matrix.
       * 
       * See \ref SparseMatrixView.
       * 
       * \return sparse matrix view.
       */
      ConstViewType getConstView() const;

      /**
       * \brief Returns string with serialization type.
       * 
       * The string has a form \e `Matrices::SparseMatrix< RealType,  [any_device], IndexType, General/Symmetric, Format, [any_allocator] >`.
       * 
       * \return \e String with the serialization type.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_getSerializationType.cpp
       * \par Output
       * \include SparseMatrixExample_getSerializationType.out
       */
      static String getSerializationType();

      /**
       * \brief Returns string with serialization type.
       * 
       * See \ref SparseMatrix::getSerializationType.
       * 
       * \return \e String with the serialization type.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_getSerializationType.cpp
       * \par Output
       * \include SparseMatrixExample_getSerializationType.out
       */
      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Set number of rows and columns of this matrix.
       * 
       * \param rows is the number of matrix rows.
       * \param columns is the number of matrix columns.
       */
      virtual void setDimensions( const IndexType rows,
                                  const IndexType columns ) override;

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
       * \brief Allocates memory for non-zero matrix elements.
       * 
       * The size of the input vector must be equal to the number of matrix rows.
       * The number of allocated matrix elements for each matrix row depends on
       * the sparse matrix format. Some formats may allocate more elements than
       * required.
       * 
       * \tparam RowsCapacitiesVector is a type of vector/array used for row
       *    capacities setting.
       * 
       * \param rowCapacities is a vector telling the number of required non-zero
       *    matrix elements in each row.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_setRowCapacities.cpp
       * \par Output
       * \include SparseMatrixExample_setRowCapacities.out
       */
      template< typename RowsCapacitiesVector >
      void setRowCapacities( const RowsCapacitiesVector& rowCapacities );

      // TODO: Remove this when possible
      template< typename RowsCapacitiesVector >
      [[deprecated]]
      void setCompressedRowLengths( const RowsCapacitiesVector& rowLengths ) {
         this->setRowCapacities( rowLengths );
      };

      /**
       * \brief This method sets the sparse matrix elements from initializer list.
       * 
       * The number of matrix rows and columns must be set already.
       * The matrix elements values are given as a list \e data of triples:
       * { { row1, column1, value1 },
       *   { row2, column2, value2 },
       * ... }.
       * 
       * \param data is a initializer list of initializer lists representing
       * list of matrix rows.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_setElements.cpp
       * \par Output
       * \include SparseMatrixExample_setElements.out
       */
      void setElements( const std::initializer_list< std::tuple< IndexType, IndexType, RealType > >& data );

      /**
       * \brief This method sets the sparse matrix elements from std::map.
       * 
       * The matrix elements values are given as a map \e data where keys are
       * std::pair of matrix coordinates ( {row, column} ) and value is the
       * matrix element value.
       * 
       * \tparam MapIndex is a type for indexing rows and columns.
       * \tparam MapValue is a type for matrix elements values in the map.
       * 
       * \param map is std::map containing matrix elements.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_setElements_map.cpp
       * \par Output
       * \include SparseMatrixExample_setElements_map.out
       */
      template< typename MapIndex,
                typename MapValue >
      void setElements( const std::map< std::pair< MapIndex, MapIndex > , MapValue >& map );

      /**
       * \brief Computes number of non-zeros in each row.
       * 
       * \param rowLengths is a vector into which the number of non-zeros in each row
       * will be stored.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_getCompressedRowLengths.cpp
       * \par Output
       * \include SparseMatrixExample_getCompressedRowLengths.out
       */
      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      /**
       * \brief Returns capacity of given matrix row.
       * 
       * \param row index of matrix row.
       * \return number of matrix elements allocated for the row.
       */
      __cuda_callable__
      IndexType getRowCapacity( const IndexType row ) const;

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
       * \brief Constant getter of simple structure for accessing given matrix row.
       * 
       * \param rowIdx is matrix row index.
       * 
       * \return RowView for accessing given matrix row.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_getConstRow.cpp
       * \par Output
       * \include SparseMatrixExample_getConstRow.out
       * 
       * See \ref SparseMatrixRowView.
       */
      __cuda_callable__
      const ConstRowView getRow( const IndexType& rowIdx ) const;

      /**
       * \brief Non-constant getter of simple structure for accessing given matrix row.
       * 
       * \param rowIdx is matrix row index.
       * 
       * \return RowView for accessing given matrix row.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_getRow.cpp
       * \par Output
       * \include SparseMatrixExample_getRow.out
       * 
       * See \ref SparseMatrixRowView.
       */
      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );

      /**
       * \brief Sets element at given \e row and \e column to given \e value.
       * 
       * This method can be called from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated on GPU this method
       * can be called even from device kernels. If the matrix is allocated in GPU device
       * this method is called from CPU, it transfers values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref SparseMatrix::getRow
       * or \ref SparseMatrix::forRows and \ref SparseMatrix::forAllRows.
       * The call may fail if the matrix row capacity is exhausted.
       * 
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_setElement.cpp
       * \par Output
       * \include SparseMatrixExample_setElement.out
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
       * performance is very low. For higher performance see. \ref SparseMatrix::getRow
       * or \ref SparseMatrix::forRows and \ref SparseMatrix::forAllRows.
       * The call may fail if the matrix row capacity is exhausted.
       * 
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       * \param thisElementMultiplicator is multiplicator the original matrix element
       *   value is multiplied by before addition of given \e value.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_addElement.cpp
       * \par Output
       * \include SparseMatrixExample_addElement.out
       * 
       */
      __cuda_callable__
      void addElement( const IndexType row,
                       const IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator );

      /**
       * \brief Returns value of matrix element at position given by its row and column index.
       * 
       * This method can be called from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated on GPU this method
       * can be called even from device kernels. If the matrix is allocated in GPU device
       * this method is called from CPU, it transfers values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref SparseMatrix::getRow
       * or \ref SparseMatrix::forRows and \ref SparseMatrix::forAllRows.
       * 
       * \param row is a row index of the matrix element.
       * \param column i a column index of the matrix element.
       * 
       * \return value of given matrix element.
       * 
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_getElement.cpp
       * \par Output
       * \include SparseMatrixExample_getElement.out
       * 
       */
      __cuda_callable__
      RealType getElement( const IndexType row,
                           const IndexType column ) const;

      /**
       * \brief Method for performing general reduction on matrix rows.
       * 
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *          `fetch( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, RealType elementValue ) -> FetchValue`.
       *          Parameter \e globalIdx is position of the matrix element in arrays \e values and \e columnIdexes
       *          of this matrix. The return type of this lambda can be any non void.
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
       * \include Matrices/SparseMatrix/SparseMatrixExample_rowsReduction.cpp
       * \par Output
       * \include SparseMatrixExample_rowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void rowsReduction( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      /**
       * \brief Method for performing general reduction on all matrix rows.
       * 
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *          `fetch( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, RealType elementValue ) -> FetchValue`.
       *          Parameter \e globalIdx is position of the matrix element in arrays \e values and \e columnIdexes
       *          of this matrix. The return type of this lambda can be any non void.
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
       * \include Matrices/SparseMatrix/SparseMatrixExample_rowsReduction.cpp
       * \par Output
       * \include SparseMatrixExample_rowsReduction.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      template< typename Function >
      void forRows( IndexType begin, IndexType end, Function& function ) const;

      template< typename Function >
      void forRows( IndexType begin, IndexType end, Function& function );

      template< typename Function >
      void forAllRows( Function& function ) const;

      template< typename Function >
      void forAllRows( Function& function );

      /***
       * \brief This method computes outVector = matrixMultiplicator * ( *this ) * inVector + inVectorAddition * inVector
       */
      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector,
                          const RealType& matrixMultiplicator = 1.0,
                          const RealType& outVectorMultiplicator = 0.0,
                          const IndexType firstRow = 0,
                          const IndexType lastRow = 0 ) const;

      /*template< typename Real2, typename Index2 >
      void addMatrix( const SparseMatrix< Real2, Segments, Device, Index2 >& matrix,
                      const RealType& matrixMultiplicator = 1.0,
                      const RealType& thisMatrixMultiplicator = 1.0 );

      template< typename Real2, typename Index2 >
      void getTransposition( const SparseMatrix< Real2, Segments, Device, Index2 >& matrix,
                             const RealType& matrixMultiplicator = 1.0 );
       */

      template< typename Vector1, typename Vector2 >
      bool performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      /**
       * \brief Assignment of exactly the same matrix type.
       * @param matrix
       * @return
       */
      SparseMatrix& operator=( const SparseMatrix& matrix );

      /**
       * \brief Assignment of dense matrix
       */
      template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization, typename RealAllocator_ >
      SparseMatrix& operator=( const DenseMatrix< Real_, Device_, Index_, Organization, RealAllocator_ >& matrix );


      /**
       * \brief Assignment of any other matrix type.
       * @param matrix
       * @return
       */
      template< typename RHSMatrix >
      SparseMatrix& operator=( const RHSMatrix& matrix );

      template< typename Matrix >
      bool operator==( const Matrix& m ) const;

      template< typename Matrix >
      bool operator!=( const Matrix& m ) const;

      void save( File& file ) const;

      void load( File& file );

      void save( const String& fileName ) const;

      void load( const String& fileName );

      void print( std::ostream& str ) const;

      __cuda_callable__
      IndexType getPaddingIndex() const;

      SegmentsType& getSegments();

      const SegmentsType& getSegments() const;


// TODO: restore it and also in Matrix
//   protected:

      ColumnsIndexesVectorType columnIndexes;

      SegmentsType segments;

      IndexAllocator indexAllocator;

      ViewType view;
};

   } // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/SparseMatrix.hpp>
