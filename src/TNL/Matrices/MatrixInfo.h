/***************************************************************************
                          Matrix.h  -  description
                             -------------------
    begin                : Dec 18, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/Matrices/Dense.h>
#include <TNL/Matrices/DenseView.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/SparseMatrixView.h>

namespace TNL {
/**
 * \brief Namespace for matrix formats.
 */
namespace Matrices {

template< typename Matrix >
struct MatrixInfo
{};

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
struct MatrixInfo< DenseView< Real, Device, RowMajorOrder > >
{
   static String getDensity() { return String( "dense" ); };
};

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
struct MatrixInfo< Dense< Real, Device, RowMajorOrder, RealAllocator > >
: public MatrixInfo< typename Dense< Real, Device, RowMajorOrder, RealAllocator >::ViewType >
{
};


template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename Device_, typename Index_ > class SegmentsView >
struct MatrixInfo< SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() {
      if( std::is_same< SegementsView ........ >)
   };
};

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename Device_, typename Index_, typename IndexAllocator_ > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
struct MatrixInfo< SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator > >
:public MatrixInfo< typename SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::ViewType >
{
}

} //namespace Matrices
} //namespace TNL