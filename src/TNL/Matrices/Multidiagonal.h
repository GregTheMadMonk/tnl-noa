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

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          ElementsOrganization Organization = Containers::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real >,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class Multidiagonal : public Matrix< Real, Device, Index, RealAllocator >
{
   public:
      using RealType = Real;
      using DeviceType = Device;
      using IndexType = Index;
      using RealAllocatorType = RealAllocator;
      using IndexAllocatorType = IndexAllocator;
      using BaseType = Matrix< Real, Device, Index, RealAllocator >;
      using ValuesVectorType = typename BaseType::ValuesVectorType;
      using ValuesViewType = typename ValuesVectorType::ViewType;
      using IndexerType = details::MultidiagonalMatrixIndexer< IndexType, Organization >;
      using DiagonalsShiftsType = Containers::Vector< IndexType, DeviceType, IndexType, IndexAllocatorType >;
      using DiagonalsShiftsView = typename DiagonalsShiftsType::ViewType;
      using RowView = MultidiagonalMatrixRowView< ValuesViewType, IndexerType, DiagonalsShiftsView >;
      using ViewType = MultidiagonalMatrixView< Real, Device, Index, Organization >;
      using ConstViewType = MultidiagonalMatrixView< typename std::add_const< Real >::type, Device, Index, Organization >;

      using HostDiagonalsShiftsType = Containers::Vector< IndexType, Devices::Host, IndexType >;
      using HostDiagonalsShiftsView = typename HostDiagonalsShiftsType::ViewType;


      // TODO: remove this - it is here only for compatibility with original matrix implementation
      typedef Containers::Vector< IndexType, DeviceType, IndexType > CompressedRowLengthsVector;
      typedef Containers::VectorView< IndexType, DeviceType, IndexType > CompressedRowLengthsVectorView;
      typedef typename CompressedRowLengthsVectorView::ConstViewType ConstCompressedRowLengthsVectorView;

      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index >
      using Self = Multidiagonal< _Real, _Device, _Index >;

      static constexpr ElementsOrganization getOrganization() { return Organization; };

      Multidiagonal();

      Multidiagonal( const IndexType rows,
                     const IndexType columns );

      template< typename Vector >
      Multidiagonal( const IndexType rows,
                     const IndexType columns,
                     const Vector& diagonalsShifts );

      ViewType getView() const; // TODO: remove const

      //ConstViewType getConstView() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      template< typename Vector >
      void setDimensions( const IndexType rows,
                          const IndexType columns,
                          const Vector&  diagonalsShifts );

      //template< typename Vector >
      void setCompressedRowLengths( const ConstCompressedRowLengthsVectorView rowCapacities );

      const IndexType& getDiagonalsCount() const;

      const DiagonalsShiftsType& getDiagonalsShifts() const;

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      IndexType getNonemptyRowsCount() const;

      [[deprecated]]
      IndexType getRowLength( const IndexType row ) const;

      IndexType getMaxRowLength() const;

      template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
      void setLike( const Multidiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& m );

      IndexType getNumberOfNonzeroMatrixElements() const;

      void reset();

      template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
      bool operator == ( const Multidiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix ) const;

      template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
      bool operator != ( const Multidiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix ) const;

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
      void addMatrix( const Multidiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix,
                      const RealType& matrixMultiplicator = 1.0,
                      const RealType& thisMatrixMultiplicator = 1.0 );

      template< typename Real2, typename Index2 >
      void getTransposition( const Multidiagonal< Real2, Device, Index2 >& matrix,
                             const RealType& matrixMultiplicator = 1.0 );

      template< typename Vector1, typename Vector2 >
      __cuda_callable__
      void performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      // copy assignment
      Multidiagonal& operator=( const Multidiagonal& matrix );

      // cross-device copy assignment
      template< typename Real_,
                typename Device_,
                typename Index_,
                ElementsOrganization Organization_,
                typename RealAllocator_,
                typename IndexAllocator_ >
      Multidiagonal& operator=( const Multidiagonal< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix );

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

#include <TNL/Matrices/Multidiagonal.hpp>
