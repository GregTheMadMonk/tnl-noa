/***************************************************************************
                          DenseMatrix.h  -  description
                             -------------------
    begin                : Jan 5, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Matrices {
      namespace details {

template< typename Device >
class DenseDeviceDependentCode;
template<>
class DenseDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index,
                bool RowMajorOrder,
                typename RealAllocator,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const DenseMatrixView< Real, Device, Index, RowMajorOrder >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
         for( Index row = 0; row < matrix.getRows(); row ++ )
            outVector[ row ] = matrix.rowVectorProduct( row, inVector );
      }
};

template<>
class DenseDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index,
                bool RowMajorOrder,
                typename RealAllocator,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const DenseMatrixView< Real, Device, Index, RowMajorOrder >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         MatrixVectorProductCuda( matrix, inVector, outVector );
      }
};

      } //namespace details
   } //namepsace Matrices
} //namespace TNL