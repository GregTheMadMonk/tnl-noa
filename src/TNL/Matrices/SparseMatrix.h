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
       * \param rows is number of matrix rows.
       * \param columns is number of matrix columns.
       * \param data is std::map containing matrix elements values.
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


      ViewType getView(); // TODO: remove const

      ConstViewType getConstView() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      virtual void setDimensions( const IndexType rows,
                                  const IndexType columns ) override;

      template< typename RowsCapacitiesVector >
      void setRowCapacities( const RowsCapacitiesVector& rowCapacities );

      // TODO: Remove this when possible
      template< typename RowsCapacitiesVector >
      void setCompressedRowLengths( const RowsCapacitiesVector& rowLengths ) {
         this->setRowCapacities( rowLengths );
      };

      void setElements( const std::initializer_list< std::tuple< IndexType, IndexType, RealType > >& data );

      template< typename MapIndex,
                typename MapValue >
      void setElements( const std::map< std::pair< MapIndex, MapIndex > , MapValue >& map );

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      __cuda_callable__
      IndexType getRowCapacity( const IndexType row ) const;

      template< typename Matrix >
      void setLike( const Matrix& matrix );

      IndexType getNumberOfNonzeroMatrixElements() const;

      void reset();

      __cuda_callable__
      const ConstRowView getRow( const IndexType& rowIdx ) const;

      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );

      __cuda_callable__
      void setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      __cuda_callable__
      void addElement( const IndexType row,
                       const IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator );

      __cuda_callable__
      RealType getElement( const IndexType row,
                           const IndexType column ) const;

      /*template< typename Vector >
      __cuda_callable__
      typename Vector::RealType rowVectorProduct( const IndexType row,
                                                  const Vector& vector ) const;*/

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

      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void rowsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function ) const;

      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function );

      template< typename Function >
      void forAllRows( Function& function ) const;

      template< typename Function >
      void forAllRows( Function& function );

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
