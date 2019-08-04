/***************************************************************************
                          VectorOperationsCuda_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Containers/Algorithms/VectorOperations.h>
#include <TNL/Containers/Algorithms/CudaPrefixSumKernel.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Vector >
void
VectorOperations< Devices::Cuda >::
addElement( Vector& v,
            const typename Vector::IndexType i,
            const typename Vector::RealType& value )
{
   v[ i ] += value;
}

template< typename Vector, typename Scalar >
void
VectorOperations< Devices::Cuda >::
addElement( Vector& v,
            const typename Vector::IndexType i,
            const typename Vector::RealType& value,
            const Scalar thisElementMultiplicator )
{
   v[ i ] = thisElementMultiplicator * v[ i ] + value;
}

template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2 >
void
VectorOperations< Devices::Cuda >::
addVector( Vector1& _y,
           const Vector2& _x,
           const Scalar1 alpha,
           const Scalar2 thisMultiplicator )
{
   TNL_ASSERT_GT( _x.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( _x.getSize(), _y.getSize(), "The vector sizes must be the same." );

#ifdef HAVE_CUDA
   using IndexType = typename Vector1::IndexType;
   using RealType = typename Vector1::RealType;

   RealType* y = _y.getData();
   const RealType* x = _x.getData();
   auto add1 = [=] __cuda_callable__ ( IndexType i ) { y[ i ] += alpha * x[ i ]; };
   auto add2 = [=] __cuda_callable__ ( IndexType i ) { y[ i ] = thisMultiplicator * y[ i ] + alpha * x[ i ]; };

   if( thisMultiplicator == 1.0 )
      ParallelFor< Devices::Cuda >::exec( ( IndexType ) 0, _y.getSize(), add1 );
   else
      ParallelFor< Devices::Cuda >::exec( ( IndexType ) 0, _y.getSize(), add2 );
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

#ifdef HAVE_CUDA
template< typename Real1, typename Real2, typename Real3, typename Index,
          typename Scalar1, typename Scalar2, typename Scalar3 >
__global__ void
vectorAddVectorsCudaKernel( Real1* v,
                            const Real2* v1,
                            const Real3* v2,
                            const Index size,
                            const Scalar1 multiplicator1,
                            const Scalar2 multiplicator2,
                            const Scalar3 thisMultiplicator )
{
   Index elementIdx = blockDim.x * blockIdx.x + threadIdx.x;
   const Index maxGridSize = blockDim.x * gridDim.x;
   if( thisMultiplicator == 1.0 )
      while( elementIdx < size )
      {
         v[ elementIdx ] += multiplicator1 * v1[ elementIdx ] +
                            multiplicator2 * v2[ elementIdx ];
         elementIdx += maxGridSize;
      }
   else
      while( elementIdx < size )
      {
         v[ elementIdx ] = thisMultiplicator * v[ elementIdx ] +
                           multiplicator1 * v1[ elementIdx ] +
                           multiplicator2 * v2[ elementIdx ];
         elementIdx += maxGridSize;
      }
}
#endif

template< typename Vector1, typename Vector2, typename Vector3,
          typename Scalar1, typename Scalar2, typename Scalar3 >
void
VectorOperations< Devices::Cuda >::
addVectors( Vector1& v,
            const Vector2& v1,
            const Scalar1 multiplicator1,
            const Vector3& v2,
            const Scalar2 multiplicator2,
            const Scalar3 thisMultiplicator )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v.getSize(), v1.getSize(), "The vector sizes must be the same." );
   TNL_ASSERT_EQ( v.getSize(), v2.getSize(), "The vector sizes must be the same." );

#ifdef HAVE_CUDA
   typedef typename Vector1::IndexType Index;

   const Index& size = v.getSize();
   dim3 cudaBlockSize( 256 );
   dim3 cudaBlocks;
   cudaBlocks.x = TNL::min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( size, cudaBlockSize.x ) );

   vectorAddVectorsCudaKernel<<< cudaBlocks, cudaBlockSize >>>( v.getData(),
                                                                v1.getData(),
                                                                v2.getData(),
                                                                size,
                                                                multiplicator1,
                                                                multiplicator2,
                                                                thisMultiplicator);
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename Vector, typename ResultType >
ResultType
VectorOperations< Devices::Cuda >::
getVectorSum( const Vector& v )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   if( std::is_same< ResultType, bool >::value )
      abort();

   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   const auto* data = v.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i )  -> ResultType { return  data[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
   return Reduction< Devices::Cuda >::reduce( v.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0 );
}

template< Algorithms::PrefixSumType Type,
          typename Vector >
void
VectorOperations< Devices::Cuda >::
prefixSum( Vector& v,
           typename Vector::IndexType begin,
           typename Vector::IndexType end )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   auto reduction = [=] __cuda_callable__ ( RealType& a, const RealType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile RealType& a, volatile RealType& b ) { a += b; };

   PrefixSum< Devices::Cuda, Type >::perform( v, begin, end, reduction, volatileReduction, ( RealType ) 0.0 );
}

template< Algorithms::PrefixSumType Type, typename Vector, typename Flags >
void
VectorOperations< Devices::Cuda >::
segmentedPrefixSum( Vector& v,
                    Flags& f,
                    typename Vector::IndexType begin,
                    typename Vector::IndexType end )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   auto reduction = [=] __cuda_callable__ ( RealType& a, const RealType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile RealType& a, volatile RealType& b ) { a += b; };

   SegmentedPrefixSum< Devices::Cuda, Type >::perform( v, f, begin, end, reduction, volatileReduction, ( RealType ) 0.0 );
}


} // namespace Algorithms
} // namespace Containers
} // namespace TNL
