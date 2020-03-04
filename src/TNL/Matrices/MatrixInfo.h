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
#include <TNL/Matrices/DenseMatrixView.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/SparseMatrixView.h>
#include <TNL/Containers/Segments/CSRView.h>
#include <TNL/Containers/Segments/EllpackView.h>
#include <TNL/Containers/Segments/SlicedEllpackView.h>
#include <TNL/Matrices/Legacy/CSR.h>
#include <TNL/Matrices/Legacy/Ellpack.h>
#include <TNL/Matrices/Legacy/SlicedEllpack.h>
#include <TNL/Matrices/Legacy/ChunkedEllpack.h>
#include <TNL/Matrices/Legacy/BiEllpack.h>

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
struct MatrixInfo< DenseMatrixView< Real, Device, Index, RowMajorOrder > >
{
   static String getDensity() { return String( "dense" ); };
};

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
struct MatrixInfo< Dense< Real, Device, Index, RowMajorOrder, RealAllocator > >
: public MatrixInfo< typename Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::ViewType >
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

   static String getFormat() { return SegmentsView< Device, Index >::getSegmentsType(); };
};

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename Device_, typename Index_, typename IndexAllocator_ > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
struct MatrixInfo< SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator > >
: public MatrixInfo< typename SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::ViewType >
{
};

/////
// Legacy matrices
template< typename Real, typename Device, typename Index >
struct MatrixInfo< Legacy::BiEllpack< Real, Device, Index > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "BiEllpack Legacy"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Legacy::CSR< Real, Device, Index > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Legacy::ChunkedEllpack< Real, Device, Index > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "ChunkedEllpack Legacy"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Legacy::Ellpack< Real, Device, Index > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "Ellpack Legacy"; };
};

template< typename Real, typename Device, typename Index, int SliceSize >
struct MatrixInfo< Legacy::SlicedEllpack< Real, Device, Index, SliceSize> >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "SlicedEllpack Legacy"; };
};

} //namespace Matrices
} //namespace TNL
