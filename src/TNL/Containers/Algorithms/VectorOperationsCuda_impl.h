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

#ifdef HAVE_CUDA
template< typename Real, typename Index, typename Scalar >
__global__ void
vectorScalarMultiplicationCudaKernel( Real* data,
                                      Index size,
                                      Scalar alpha )
{
   Index elementIdx = blockDim.x * blockIdx.x + threadIdx.x;
   const Index maxGridSize = blockDim.x * gridDim.x;
   while( elementIdx < size )
   {
      data[ elementIdx ] *= alpha;
      elementIdx += maxGridSize;
   }
}
#endif

template< typename Vector, typename Scalar >
void
VectorOperations< Devices::Cuda >::
vectorScalarMultiplication( Vector& v,
                            const Scalar alpha )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

#ifdef HAVE_CUDA
   typedef typename Vector::IndexType Index;
   dim3 blockSize( 0 ), gridSize( 0 );
   const Index& size = v.getSize();
   blockSize.x = 256;
   Index blocksNumber = TNL::ceil( ( double ) size / ( double ) blockSize.x );
   gridSize.x = min( blocksNumber, Devices::Cuda::getMaxGridSize() );
   vectorScalarMultiplicationCudaKernel<<< gridSize, blockSize >>>( v.getData(),
                                                                    size,
                                                                    alpha );
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
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

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
