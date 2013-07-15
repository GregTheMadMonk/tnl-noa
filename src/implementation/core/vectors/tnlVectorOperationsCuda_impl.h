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

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlCuda > :: getVectorMax( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   tnlAssert( v. getSize() > 0,
              cerr << "Vector name is " << v. getName() );
   Real result( 0 );
   tnlParallelReductionMax< Real, Index > operation;
   reductionOnCudaDevice( operation,
                          v. getSize(),
                          v. getData(),
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
   reductionOnCudaDevice( operation,
                          v1. getSize(),
                          v1. getData(),
                          v2. getData(),
                          result );
   return result;
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


template< typename Vector >
void tnlVectorOperations< tnlCuda > :: vectorScalarMultiplication( Vector& v,
                                     const typename Vector :: RealType& alpha )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   tnlAssert( v. getSize() > 0,
              cerr << "Vector name is " << v. getName() );

#ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = v. getSize() / 512 + 1;
   // TODO: Fix this - the grid size might be limiting for large vectors.

   /*
   tnlVectorCUDAScalaMultiplicationKernel<<< gridSize, blockSize >>>( v. getSize(),
                                                                      alpha,
                                                                      v. getData() );*/
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif

}


template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlCuda > :: getVectorSdot( const Vector1& v1,
                                                const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0,
              cerr << "Vector name is " << v1. getName() );
   tnlAssert( v1. getSize() == v2. getSize(),
              cerr << "Vector names are " << v1. getName() << " and " << v2. getName() );

   Real result( 0 );
   /*reductionOnCudaDevice< Real,
                               Real,
                               Index,
                               tnlParallelReductionSdot >
                             ( v1. getSize(),
                               v1. getData(),
                               v2. getData(),
                               result,
                               ( Real ) 0 );*/
   return result;
}

template< typename Vector1, typename Vector2 >
void tnlVectorOperations< tnlCuda > :: vectorSaxpy( Vector1& y,
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
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = x. getSize() / 512 + 1;

   // TODO: fix this
   /*tnlVectorCUDASaxpyKernel<<< gridSize, blockSize >>>( y. getSize(),
                                                        alpha,
                                                        x. getData(),
                                                        y. getData() );*/
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
}


template< typename Vector1, typename Vector2 >
void tnlVectorOperations< tnlCuda > :: vectorSaxmy( Vector1& y,
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
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = x. getSize() / 512 + 1;

   // TODO: fix this
   /*tnlVectorCUDASaxmyKernel<<< gridSize, blockSize >>>( y. getSize(),
                                                        alpha,
                                                        x. getData(),
                                                        y. getData() );*/
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
}

template< typename Vector1, typename Vector2 >
void tnlVectorOperations< tnlCuda > :: vectorSaxpsby( Vector1& y,
                        const Vector2& x,
                        const typename Vector1 :: RealType& alpha,
                        const typename Vector1 :: RealType& beta )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( y. getSize() > 0,
              cerr << "Vector name is " << y. getName() );
   tnlAssert( y. getSize() == x. getSize(),
              cerr << "Vector names are " << x. getName() << " and " << y. getName() );

#ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = x. getSize() / 512 + 1;

   // TODO: fix this
   /*tnlVectorCUDASaxpsbzKernel<<< gridSize, blockSize >>>( y. getSize(),
                                                          alpha,
                                                          x. getData(),
                                                          beta );*/
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
}

template< typename Vector1, typename Vector2 >
void tnlVectorOperations< tnlCuda > :: vectorSaxpsbz( Vector1& y,
                        const Vector2& x,
                        const typename Vector1 :: RealType& alpha,
                        const Vector2& z,
                        const typename Vector1 :: RealType& beta )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( y. getSize() > 0,
              cerr << "Vector name is " << y. getName() );
   tnlAssert( y. getSize() == x. getSize(),
              cerr << "Vector names are " << x. getName() << " and " << y. getName() );

#ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = x. getSize() / 512 + 1;

   // TODO: fix this
   /*tnlVectorCUDASaxpsbzKernel<<< gridSize, blockSize >>>( y. getSize(),
                                                          alpha,
                                                          x. getData(),
                                                          beta,
                                                          z. getData() );*/
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
}

template< typename Vector1, typename Vector2 >
void tnlVectorOperations< tnlCuda > :: vectorSaxpsbzpy( Vector1& y,
                          const Vector2& x,
                          const typename Vector1 :: RealType& alpha,
                          const Vector2& z,
                          const typename Vector1 :: RealType& beta )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( y. getSize() > 0,
              cerr << "Vector name is " << y. getName() );
   tnlAssert( y. getSize() == x. getSize(),
              cerr << "Vector names are " << x. getName() << " and " << y. getName() );

#ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = x. getSize() / 512 + 1;

   // TODO: fix this
   /*tnlVectorCUDASaxpsbzpyKernel<<< gridSize, blockSize >>>( y. getSize(),
                                                          alpha,
                                                          x. getData(),
                                                          beta,
                                                          z. getData() );*/
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
}


#endif /* TNLVECTOROPERATIONSCUDA_IMPL_H_ */
