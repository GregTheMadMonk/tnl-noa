/***************************************************************************
                          tnlVectorOperationsCuda_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLVECTOROPERATIONSCUDA_IMPL_H_
#define TNLVECTOROPERATIONSCUDA_IMPL_H_

template< typename Real, typename Index >
Real tnlVectorOperations< tnlCuda > :: getVectorMax( const Real* v,
                                                     const Index size )
{
   Real result( 0 );
   tnlParallelReductionMax< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          size,
                          v,
                          ( Real* ) 0,
                          result );
   return result;
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlCuda > :: getVectorMin( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   tnlAssert( v. getSize() > 0,
              cerr << "Vector name is " << v. getName() );

   Real result( 0 );
   tnlParallelReductionMin< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return result;
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlCuda > :: getVectorAbsMax( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   tnlAssert( v. getSize() > 0,
              cerr << "Vector name is " << v. getName() );

   Real result( 0 );
   tnlParallelReductionAbsMax< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return result;
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlCuda > :: getVectorAbsMin( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   tnlAssert( v. getSize() > 0,
              cerr << "Vector name is " << v. getName() );

   Real result( 0 );
   tnlParallelReductionAbsMin< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return result;
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlCuda > :: getVectorLpNorm( const Vector& v,
                                                    const typename Vector :: RealType& p )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   tnlAssert( v. getSize() > 0,
              cerr << "Vector name is " << v. getName() );
   tnlAssert( p > 0.0,
              cerr << " p = " << p );

   Real result( 0 );
   tnlParallelReductionLpNorm< Real, Index > operation;
   operation. setPower( p );
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return pow( result, 1.0 / p );
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlCuda > :: getVectorSum( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   tnlAssert( v. getSize() > 0,
              cerr << "Vector name is " << v. getName() );

   Real result( 0 );
   tnlParallelReductionSum< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
                          ( Real* ) 0,
                          result );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getVectorDifferenceMax( const Vector1& v1,
                                                            const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0,
              cerr << "Vector name is " << v1. getName() );
   tnlAssert( v1. getSize() == v2. getSize(),
              cerr << "Vector names are " << v1. getName() << " and " << v2. getName() );

   Real result( 0 );
   tnlParallelReductionDiffMax< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getVectorDifferenceMin( const Vector1& v1,
                                                            const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0,
              cerr << "Vector name is " << v1. getName() );
   tnlAssert( v1. getSize() == v2. getSize(),
              cerr << "Vector names are " << v1. getName() << " and " << v2. getName() );

   Real result( 0 );
   tnlParallelReductionDiffMin< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}


template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getVectorDifferenceAbsMax( const Vector1& v1,
                                                               const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0,
              cerr << "Vector name is " << v1. getName() );
   tnlAssert( v1. getSize() == v2. getSize(),
              cerr << "Vector names are " << v1. getName() << " and " << v2. getName() );

   Real result( 0 );
   tnlParallelReductionDiffAbsMax< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getVectorDifferenceAbsMin( const Vector1& v1,
                                                            const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0,
              cerr << "Vector name is " << v1. getName() );
   tnlAssert( v1. getSize() == v2. getSize(),
              cerr << "Vector names are " << v1. getName() << " and " << v2. getName() );

   Real result( 0 );
   tnlParallelReductionDiffAbsMin< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getVectorDifferenceLpNorm( const Vector1& v1,
                                                               const Vector2& v2,
                                                               const typename Vector1 :: RealType& p )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( p > 0.0,
              cerr << " p = " << p );
   tnlAssert( v1. getSize() > 0,
              cerr << "Vector name is " << v1. getName() );
   tnlAssert( v1. getSize() == v2. getSize(),
              cerr << "Vector names are " << v1. getName() << " and " << v2. getName() );

   Real result( 0 );
   tnlParallelReductionDiffLpNorm< Real, Index > operation;
   operation.setPower( p );
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return pow( result, 1.0 / p );
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getVectorDifferenceSum( const Vector1& v1,
                                                         const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0,
              cerr << "Vector name is " << v1. getName() );
   tnlAssert( v1. getSize() == v2. getSize(),
              cerr << "Vector names are " << v1. getName() << " and " << v2. getName() );

   Real result( 0 );
   tnlParallelReductionDiffSum< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}

#ifdef HAVE_CUDA
template< typename Real, typename Index >
__global__ void vectorScalarMultiplicationCudaKernel( Real* data,
                                                      Index size,
                                                      Real alpha )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      data[ elementIdx ] *= alpha;
      elementIdx += maxGridSize;
   }
}
#endif

template< typename Vector >
void tnlVectorOperations< tnlCuda > :: vectorScalarMultiplication( Vector& v,
                                                                   const typename Vector::RealType& alpha )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   tnlAssert( v. getSize() > 0,
              cerr << "Vector name is " << v. getName() );

   #ifdef HAVE_CUDA
      dim3 blockSize( 0 ), gridSize( 0 );
      const Index& size = v.getSize();
      blockSize. x = 256;
      Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
      gridSize. x = Min( blocksNumber, tnlCuda::getMaxGridSize() );
      vectorScalarMultiplicationCudaKernel<<< gridSize, blockSize >>>( v.getData(),
                                                                       size,
                                                                       alpha );
      checkCudaDevice;
   #else
      tnlCudaSupportMissingMessage;;
   #endif
}


template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getScalarProduct( const Vector1& v1,
                                                                                 const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0,
              cerr << "Vector name is " << v1. getName() );
   tnlAssert( v1. getSize() == v2. getSize(),
              cerr << "Vector names are " << v1. getName() << " and " << v2. getName() );

   Real result( 0 );
   tnlParallelReductionScalarProduct< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index >
__global__ void vectorAlphaXPlusYCudaKernel( Real* y,
                                             const Real* x,
                                             const Index size,
                                             const Real alpha )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      y[ elementIdx ] += alpha * x[ elementIdx ];
      elementIdx += maxGridSize;
   }
}
#endif

template< typename Vector1, typename Vector2 >
void tnlVectorOperations< tnlCuda > :: alphaXPlusY( Vector1& y,
                                                    const Vector2& x,
                                                    const typename Vector1 :: RealType& alpha )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( y. getSize() > 0,
              cerr << "Vector name is " << y. getName() );
   tnlAssert( y. getSize() == x. getSize(),
              cerr << "Vector names are " << x. getName() << " and " << y. getName() );

   #ifdef HAVE_CUDA
      dim3 blockSize( 0 ), gridSize( 0 );
      const Index& size = x.getSize();
      blockSize. x = 256;
      Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
      gridSize. x = Min( blocksNumber, tnlCuda::getMaxGridSize() );
      vectorAlphaXPlusYCudaKernel<<< gridSize, blockSize >>>( y.getData(),
                                                              x.getData(), 
                                                              size,
                                                              alpha );
      checkCudaDevice;
   #else
      tnlCudaSupportMissingMessage;;
   #endif
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index >
__global__ void vectorAlphaXPlusBetaYCudaKernel( Real* y,
                                                 const Real* x,
                                                 const Index size,
                                                 const Real alpha,
                                                 const Real beta )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      y[ elementIdx ] = alpha * x[ elementIdx ] + beta * y[ elementIdx ];
      elementIdx += maxGridSize;
   }
}
#endif

template< typename Vector1, typename Vector2 >
void tnlVectorOperations< tnlCuda > :: alphaXPlusBetaY( Vector1& y,
                                                        const Vector2& x,
                                                        const typename Vector1::RealType& alpha,
                                                        const typename Vector1::RealType& beta )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( y. getSize() > 0,
              cerr << "Vector name is " << y. getName() );
   tnlAssert( y. getSize() == x. getSize(),
              cerr << "Vector names are " << x. getName() << " and " << y. getName() );

   #ifdef HAVE_CUDA
      dim3 blockSize( 0 ), gridSize( 0 );
      const Index& size = x.getSize();
      blockSize. x = 256;
      Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
      gridSize. x = Min( blocksNumber, tnlCuda::getMaxGridSize() );
      vectorAlphaXPlusBetaYCudaKernel<<< gridSize, blockSize >>>( y.getData(),
                                                                  x.getData(),
                                                                  size,
                                                                  alpha,
                                                                  beta );
      checkCudaDevice;
   #else
      tnlCudaSupportMissingMessage;;
   #endif
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index >
__global__ void vectorAlphaXPlusBetaZCudaKernel( Real* y,
                                                 const Real* x,
                                                 const Real* z,
                                                 const Index size,
                                                 const Real alpha,
                                                 const Real beta )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      y[ elementIdx ] = alpha * x[ elementIdx ] + beta * z[ elementIdx ];
      elementIdx += maxGridSize;
   }
}
#endif


template< typename Vector1, typename Vector2 >
void tnlVectorOperations< tnlCuda > :: alphaXPlusBetaZ( Vector1& y,
                                                        const Vector2& x,
                                                        const typename Vector1 :: RealType& alpha,
                                                        const Vector2& z,
                                                        const typename Vector1 :: RealType& beta )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( y.getSize() > 0,
              cerr << "Vector name is " << y.getName() );
   tnlAssert( y.getSize() == x.getSize() && x.getSize() == z.getSize(),
              cerr << "Vector names are " << x.getName() << ", " << y.getName() << " and " << z.getName() );

   #ifdef HAVE_CUDA
      dim3 blockSize( 0 ), gridSize( 0 );
      const Index& size = x.getSize();
      blockSize. x = 256;
      Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
      gridSize. x = Min( blocksNumber, tnlCuda::getMaxGridSize() );
      vectorAlphaXPlusBetaZCudaKernel<<< gridSize, blockSize >>>( y.getData(),
                                                                  x.getData(),
                                                                  z.getData(),
                                                                  size,
                                                                  alpha,
                                                                  beta );
      checkCudaDevice;
   #else
      tnlCudaSupportMissingMessage;;
   #endif
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index >
__global__ void vectorAlphaXPlusBetaZPlusYCudaKernel( Real* y,
                                                     const Real* x,
                                                     const Real* z,
                                                     const Index size,
                                                     const Real alpha,
                                                     const Real beta )
{
   Index elementIdx = blockDim. x * blockIdx. x + threadIdx. x;
   const Index maxGridSize = blockDim. x * gridDim. x;
   while( elementIdx < size )
   {
      y[ elementIdx ] += alpha * x[ elementIdx ] + beta * z[ elementIdx ];
      elementIdx += maxGridSize;
   }
}
#endif

template< typename Vector1, typename Vector2 >
void tnlVectorOperations< tnlCuda > :: alphaXPlusBetaZPlusY( Vector1& y,
                                                             const Vector2& x,
                                                             const typename Vector1 :: RealType& alpha,
                                                             const Vector2& z,
                                                             const typename Vector1 :: RealType& beta )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( y.getSize() > 0,
              cerr << "Vector name is " << y.getName() );
   tnlAssert( y.getSize() == x.getSize() && x.getSize() == z.getSize(),
              cerr << "Vector names are " << x.getName() << ", " << y.getName() << " and " << z.getName() );

   #ifdef HAVE_CUDA
      dim3 blockSize( 0 ), gridSize( 0 );
      const Index& size = x.getSize();
      blockSize. x = 256;
      Index blocksNumber = ceil( ( double ) size / ( double ) blockSize. x );
      gridSize. x = Min( blocksNumber, tnlCuda::getMaxGridSize() );
      vectorAlphaXPlusBetaZPlusYCudaKernel<<< gridSize, blockSize >>>( y.getData(),
                                                                       x.getData(),
                                                                       z.getData(),
                                                                       size,
                                                                       alpha,
                                                                       beta );
      checkCudaDevice;
   #else
      tnlCudaSupportMissingMessage;;
   #endif
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template float       tnlVectorOperations< tnlCuda >::getVectorMax( const float* v,       const int size );
extern template double      tnlVectorOperations< tnlCuda >::getVectorMax( const double* v,      const int size );
extern template long double tnlVectorOperations< tnlCuda >::getVectorMax( const long double* v, const int size );
extern template float       tnlVectorOperations< tnlCuda >::getVectorMax( const float* v,       const long int size );
extern template double      tnlVectorOperations< tnlCuda >::getVectorMax( const double* v,      const long int size );
extern template long double tnlVectorOperations< tnlCuda >::getVectorMax( const long double* v, const long int size );

#endif

#endif /* TNLVECTOROPERATIONSCUDA_IMPL_H_ */
