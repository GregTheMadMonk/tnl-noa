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
      using DiagonalsShiftsType = Containers::Vector< Index, Device, Index, IndexAllocator >;
      using DiagonalsShiftsView = typename DiagonalsShiftsType::ViewType;
      using HostDiagonalsShiftsType = Containers::Vector< Index, Devices::Host, Index >;
      using HostDiagonalsShiftsView = typename HostDiagonalsShiftsType::ViewType;

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
       * See \ref MutlidiagonlMatrixView.
       */
      using ConstViewType = MultidiagonalMatrixView< typename std::add_const< Real >::type, Device, Index, Organization >;

      /**
       * \brief Type for accessing matrix rows.
       */
      using RowView = MultidiagonalMatrixRowView< ValuesViewType, IndexerType, DiagonalsShiftsView >;

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
       * \param diagonalsShifts are shifts of subdiagonals from the main diagonal.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_Constructor.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_Constructor.out
       */
      template< typename Vector >
      MultidiagonalMatrix( const IndexType rows,
                           const IndexType columns,
                           const Vector& diagonalsShifts );

      /**
       * \brief Constructor with matrix dimensions and diagonals shifts.
       * 
       * \param rows is number of matrix rows.
       * \param columns is number of matrix columns.
       * \param diagonalsShifts are shifts of sub-diagonals from the main diagonal.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_Constructor_init_list_1.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_Constructor_init_list_1.out
       */
      template< typename ListIndex >
      MultidiagonalMatrix( const IndexType rows,
                           const IndexType columns,
                           const std::initializer_list< ListIndex > diagonalsShifts );

      /**
       * \brief Constructor with matrix dimensions, diagonals shifts and matrix elements.
       * 
       * The number of matrix rows is given by the size of the initializer list \e data.
       * 
       * \param columns is number of matrix columns.
       * \param diagonalShifts are shifts of sub-diagonals from the main diagonal.
       * \param data is initializer list holding matrix elements. The size of the outer list
       *    defines the number of matrix rows. Each inner list defines non-zero elements in each row
       *    and so its size must be lower or equal to the size of \e diagonalsShifts.
       * 
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_Constructor_init_list_2.cpp
       * \par Output
       * \include MultidiagonalMatrixExample_Constructor_init_list_2.out
       */
      template< typename ListIndex, typename ListReal >
      MultidiagonalMatrix( const IndexType columns,
                           const std::initializer_list< ListIndex > diagonalsShifts,
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

      template< typename Vector >
      void setDimensions( const IndexType rows,
                          const IndexType columns,
                          const Vector& diagonalsShifts );

      template< typename RowLengthsVector >
      void setCompressedRowLengths( const RowLengthsVector& rowCapacities );

      template< typename ListReal >
      void setElements( const std::initializer_list< std::initializer_list< ListReal > >& data );

      const IndexType& getDiagonalsCount() const;

      const DiagonalsShiftsType& getDiagonalsShifts() const;

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      IndexType getNonemptyRowsCount() const;

      [[deprecated]]
      IndexType getRowLength( const IndexType row ) const;

      IndexType getMaxRowLength() const;

      template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
      void setLike( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& m );

      IndexType getNumberOfNonzeroMatrixElements() const;

      void reset();

      template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
      bool operator == ( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix ) const;

      template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
      bool operator != ( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix ) const;

      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );

      __cuda_callable__
      const RowView getRow( const IndexType& rowIdx ) const;

      void setValue( const RealType& v );

      void setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      void addElement( const IndexType row,
                       const IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator = 1.0 );

      RealType getElement( const IndexType row,
                           const IndexType column ) const;

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

      template< typename Vector >
      __cuda_callable__
      typename Vector::RealType rowVectorProduct( const IndexType row,
                                                  const Vector& vector ) const;

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

      DiagonalsShiftsType diagonalsShifts;

      HostDiagonalsShiftsType hostDiagonalsShifts;

      IndexerType indexer;

      ViewType view;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MultidiagonalMatrix.hpp>
