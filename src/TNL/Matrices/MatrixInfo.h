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
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/DenseMatrixView.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/SparseMatrixView.h>
#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/EllpackView.h>
#include <TNL/Algorithms/Segments/SlicedEllpackView.h>
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

/// This is to prevent from appearing in Doxygen documentation.
/// \cond HIDDEN_CLASS
template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
struct MatrixInfo< DenseMatrixView< Real, Device, Index, Organization > >
{
   static String getDensity() { return String( "dense" ); };
};

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
struct MatrixInfo< DenseMatrix< Real, Device, Index, Organization, RealAllocator > >
: public MatrixInfo< typename DenseMatrix< Real, Device, Index, Organization, RealAllocator >::ViewType >
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
struct MatrixInfo< Legacy::CSR< Real, Device, Index, Legacy::CSRScalar > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Scalar"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Legacy::CSR< Real, Device, Index, Legacy::CSRVector> >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Vector"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Legacy::CSR< Real, Device, Index, Legacy::CSRLight > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Light"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Legacy::CSR< Real, Device, Index, Legacy::CSRAdaptive > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Adaptive"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Legacy::CSR< Real, Device, Index, Legacy::CSRStream > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Stream"; };
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

/// \endcond
} //namespace Matrices
} //namespace TNL
