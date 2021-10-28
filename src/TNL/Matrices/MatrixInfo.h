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
#include <TNL/Matrices/Sandbox/SparseSandboxMatrix.h>
#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/EllpackView.h>
#include <TNL/Algorithms/Segments/SlicedEllpackView.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/CSR.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/Ellpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/SlicedEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/ChunkedEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/BiEllpack.h>

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

   static String getFormat()
   {
      String prefix;
      if( MatrixType::isSymmetric() )
      {
         if( std::is_same< Real, bool >::value )
            prefix = "Symmetric binary ";
         else
            prefix = "Symmetric ";
      }
      else if( std::is_same< Real, bool >::value )
         prefix = "Binary ";
      return prefix + SegmentsView< Device, Index >::getSegmentsType();
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
: public MatrixInfo< typename SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::ViewType >
{
};

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType >
struct MatrixInfo< Sandbox::SparseSandboxMatrixView< Real, Device, Index, MatrixType > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat()
   {
      if( MatrixType::isSymmetric() )
         return TNL::String( "Symmetric Sandbox" );
      else
         return TNL::String( "Sandbox" );
   };
};

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          typename RealAllocator,
          typename IndexAllocator >
struct MatrixInfo< Sandbox::SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator > >
: public MatrixInfo< typename Sandbox::SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::ViewType >
{
};

/////
// Legacy matrices
template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::BiEllpack< Real, Device, Index > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "BiEllpack Legacy"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRScalar > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Scalar"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRVector> >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Vector"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Light"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight2 > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Light2"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight3 > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Light3"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight4 > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Light4"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight5 > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Light5"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight6 > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Light6"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRAdaptive > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy Adaptive"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRMultiVector > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy MultiVector"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLightWithoutAtomic > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "CSR Legacy LightWithoutAtomic"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::ChunkedEllpack< Real, Device, Index > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "ChunkedEllpack Legacy"; };
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::Ellpack< Real, Device, Index > >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "Ellpack Legacy"; };
};

template< typename Real, typename Device, typename Index, int SliceSize >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::SlicedEllpack< Real, Device, Index, SliceSize> >
{
   static String getDensity() { return String( "sparse" ); };

   static String getFormat() { return "SlicedEllpack Legacy"; };
};

/// \endcond
} //namespace Matrices
} //namespace TNL
