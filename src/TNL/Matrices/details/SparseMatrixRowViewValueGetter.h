

/***************************************************************************
                          SparseMatrixRowViewValueGetter.h  -  description
                             -------------------
    begin                : May 4, 2021
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Matrices {
      namespace details {


template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
struct SparseMatrixRowViewValueGetter {};

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
struct SparseMatrixRowViewValueGetter< SegmentView, ValuesView, ColumnsIndexesView, true >
{
   using RealType = typename ValuesView::RealType;

   using IndexType = typename ColumnsIndexesView::IndexType;

   using ResultType = bool;

   using ConstResultType = bool;

   __cuda_callable__
   static bool getValue( const IndexType& globalIdx, const ValuesView& values, const ColumnsIndexesView& columnIndexes, const IndexType& paddingIndex )
   {
      if( columnIndexes[ globalIdx ] != paddingIndex )
         return true;
      return false;
   };
};

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
struct SparseMatrixRowViewValueGetter< SegmentView, ValuesView, ColumnsIndexesView, false >
{
   using RealType = typename ValuesView::RealType;

   using IndexType = typename ColumnsIndexesView::IndexType;

   using ResultType = RealType&;

   using ConstResultType = const RealType&;

   __cuda_callable__
   static const RealType& getValue( const IndexType& globalIdx, const ValuesView& values, const ColumnsIndexesView& columnIndexes, const IndexType& paddingIndex )
   {
      return values[ globalIdx ];
   };

   __cuda_callable__
   static RealType& getValue( const IndexType& globalIdx, ValuesView& values, ColumnsIndexesView& columnIndexes, const IndexType& paddingIndex )
   {
      return values[ globalIdx ];
   };
};

      } //namespace details
   } //namepsace Matrices
} //namespace TNL
