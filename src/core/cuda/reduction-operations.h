/***************************************************************************
                          reduction-operations.h  -  description
                             -------------------
    begin                : Mar 22, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef REDUCTION_OPERATIONS_H_
#define REDUCTION_OPERATIONS_H_

#ifdef HAVE_CUDA
#include <cuda.h>
#include <core/mfuncs.h>

enum tnlTupleOperation {  tnlParallelReductionLpNorm,
                          tnlParallelReductionSdot };


/***
 * This function returns minimum of two numbers stored on the device.
 */
template< class T > __device__ T tnlCudaMin( const T& a,
                                             const T& b )
{
   return a < b ? a : b;
}

__device__ int tnlCudaMin( const int& a,
                           const int& b )
{
   return min( a, b );
}

__device__ float tnlCudaMin( const float& a,
                             const float& b )
{
   return fminf( a, b );
}

__device__ double tnlCudaMin( const double& a,
                              const double& b )
{
   return fmin( a, b );
}

/***
 * This function returns maximum of two numbers stored on the device.
 */
template< class T > __device__ T tnlCudaMax( const T& a,
                                             const T& b )
{
   return a > b ? a : b;
}

__device__ int tnlCudaMax( const int& a,
                           const int& b )
{
   return max( a, b );
}

__device__ float tnlCudaMax( const float& a,
                             const float& b )
{
   return fmaxf( a, b );
}

__device__ double tnlCudaMax( const double& a,
                              const double& b )
{
   return fmax( a, b );
}

/***
 * This function returns absolute value of given number on the device.
 */
__device__ int tnlCudaAbs( const int& a )
{
   return abs( a );
}

__device__ float tnlCudaAbs( const float& a )
{
   return fabs( a );
}

__device__ double tnlCudaAbs( const double& a )
{
   return fabs( a );
}

template< typename Real, typename Index >
class tnlParallelReductionSum
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return data1[ idx ];
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return current + data1[ idx ];
   };

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return data1[ idx1 ] + data1[ idx2 ];
   };

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return data1[ idx1 ];
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const RealType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + data2[ idx2 ] + data2[ idx3 ];
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const RealType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + data2[ idx2 ];
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
};

template< typename Real, typename Index >
class tnlParallelReductionMin
{
   public:

   typedef Real RealType;
   typedef Index IndexType;

   RealType initialValueOnHost( const IndexType idx,
                                const RealType* data1,
                                const RealType* data2 ) const
   {
      return data1[ idx ];
   };

   RealType reduceOnHost( const IndexType idx,
                          const RealType& current,
                          const RealType* data1,
                          const RealType* data2 ) const
   {
      return Min( current, data1[ idx ] );
   };

   __device__ RealType initialValueOnDevice( const IndexType idx1,
                                             const IndexType idx2,
                                             const RealType* data1,
                                             const RealType* data2 ) const
   {
      return tnlCudaMin( data1[ idx1 ], data1[ idx2 ] );
   }

   __device__ RealType initialValueOnDevice( const IndexType idx1,
                                             const RealType* data1,
                                             const RealType* data2 ) const
   {
      return data1[ idx1 ];
   };

   __device__ RealType firstReductionOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const IndexType idx3,
                                               const RealType* data1,
                                               const RealType* data2,
                                               const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], tnlCudaMin(  data2[ idx2 ],  data2[ idx3 ] ) );
   };

   __device__ RealType firstReductionOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2,
                                               const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], data2[ idx2 ] );
   };

   __device__ RealType commonReductionOnDevice( const IndexType idx1,
                                                const IndexType idx2,
                                                const RealType* data ) const
   {
      return tnlCudaMin( data[ idx1 ], data[ idx2 ] );
   };
};

template< typename Real, typename Index >
class tnlParallelReductionMax
{
   public:

   typedef Real RealType;
   typedef Index IndexType;

   RealType initialValueOnHost( const IndexType idx,
                                const RealType* data1,
                                const RealType* data2 ) const
   {
      return data1[ idx ];
   };

   RealType reduceOnHost( const IndexType idx,
                          const RealType& current,
                          const RealType* data1,
                          const RealType* data2 ) const
   {
      return Max( current, data1[ idx ] );
   };

   __device__ RealType initialValueOnDevice( const IndexType idx1,
                                             const IndexType idx2,
                                             const RealType* data1,
                                             const RealType* data2 ) const
   {
      return tnlCudaMax( data1[ idx1 ], data1[ idx2 ] );
   }

   __device__ RealType initialValueOnDevice( const IndexType idx1,
                                             const RealType* data1,
                                             const RealType* data2 ) const
   {
      return data1[ idx1 ];
   };

   __device__ RealType firstReductionOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const IndexType idx3,
                                               const RealType* data1,
                                               const RealType* data2,
                                               const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], tnlCudaMax( data2[ idx2 ], data2[ idx3 ] ) );
   };

   __device__ RealType firstReductionOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2,
                                               const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], data2[ idx2 ] );
   };

   __device__ RealType commonReductionOnDevice( const IndexType idx1,
                                                const IndexType idx2,
                                                const RealType* data ) const
   {
      return tnlCudaMax( data[ idx1 ], data[ idx2 ] );
   };
};

template< typename Real, typename Index >
class tnlParallelReductionAbsSum
{
   public:

   typedef Real RealType;
   typedef Index IndexType;

   RealType initialValueOnHost( const IndexType idx,
                                const RealType* data1,
                                const RealType* data2 ) const
   {
      return tnlAbs( data1[ idx ] );
   };

   RealType reduceOnHost( const IndexType idx,
                          const RealType& current,
                          const RealType* data1,
                          const RealType* data2 ) const
   {
      return current + tnlAbs( data1[ idx ] );
   };

   __device__ RealType initialValueOnDevice( const IndexType idx1,
                                             const IndexType idx2,
                                             const RealType* data1,
                                             const RealType* data2 ) const
   {
      return tnlCudaAbs( data1[ idx1 ] ) + tnlCudaAbs( data1[ idx2 ] );
   };

   __device__ RealType initialValueOnDevice( const IndexType idx1,
                                             const RealType* data1,
                                             const RealType* data2 ) const
   {
      return tnlCudaAbs( data1[ idx1 ] );
   };

   __device__ RealType firstReductionOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const IndexType idx3,
                                               const RealType* data1,
                                               const RealType* data2,
                                               const RealType* data3 ) const
   {
      return data1[ idx1 ] + tnlCudaAbs( data2[ idx2 ] ) + tnlCudaAbs( data2[ idx3 ] );
   };

   __device__ RealType firstReductionOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2,
                                               const RealType* data3 ) const
   {
      return data1[ idx1 ] + tnlCudaAbs( data2[ idx2 ] );
   };

   __device__ RealType commonReductionOnDevice( const IndexType idx1,
                                                const IndexType idx2,
                                                const RealType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
};

template< typename Real, typename Index >
class tnlParallelReductionAbsMin
{
   public:

   typedef Real RealType;
   typedef Index IndexType;

   RealType initialValueOnHost( const IndexType idx,
                                const RealType* data1,
                                const RealType* data2 ) const
   {
      return tnlAbs( data1[ idx ] );
   };

   RealType reduceOnHost( const IndexType idx,
                          const RealType& current,
                          const RealType* data1,
                          const RealType* data2 ) const
   {
      return Min( current, tnlAbs( data1[ idx ] ) );
   };

   __device__ RealType initialValueOnDevice( const IndexType idx1,
                                             const IndexType idx2,
                                             const RealType* data1,
                                             const RealType* data2 ) const
   {
      return tnlCudaMin( tnlCudaAbs( data1[ idx1 ] ), tnlCudaAbs( data1[ idx2 ] ) );
   }

   __device__ RealType initialValueOnDevice( const IndexType idx1,
                                             const RealType* data1,
                                             const RealType* data2 ) const
   {
      return tnlCudaAbs( data1[ idx1 ] );
   };

   __device__ RealType firstReductionOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const IndexType idx3,
                                               const RealType* data1,
                                               const RealType* data2,
                                               const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], tnlCudaMin(  tnlCudaAbs( data2[ idx2 ] ),  tnlCudaAbs( data2[ idx3 ] ) ) );
   };

   __device__ RealType firstReductionOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2,
                                               const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], tnlCudaAbs( data2[ idx2 ] ) );
   };

   __device__ RealType commonReductionOnDevice( const IndexType idx1,
                                                const IndexType idx2,
                                                const RealType* data ) const
   {
      return tnlCudaMin( data[ idx1 ], tnlCudaAbs( data[ idx2 ] ) );
   };
};

template< typename Real, typename Index >
class tnlParallelReductionAbsMax
{
   public:

   typedef Real RealType;
   typedef Index IndexType;

   RealType initialValueOnHost( const IndexType idx,
                                const RealType* data1,
                                const RealType* data2 ) const
   {
      return tnlAbs( data1[ idx ] );
   };

   RealType reduceOnHost( const IndexType idx,
                          const RealType& current,
                          const RealType* data1,
                          const RealType* data2 ) const
   {
      return Max( current, tnlAbs( data1[ idx ] ) );
   };

   __device__ RealType initialValueOnDevice( const IndexType idx1,
                                             const IndexType idx2,
                                             const RealType* data1,
                                             const RealType* data2 ) const
   {
      return tnlCudaMax( tnlCudaAbs( data1[ idx1 ] ), tnlCudaAbs( data1[ idx2 ] ) );
   }

   __device__ RealType initialValueOnDevice( const IndexType idx1,
                                             const RealType* data1,
                                             const RealType* data2 ) const
   {
      return tnlCudaAbs( data1[ idx1 ] );
   };

   __device__ RealType firstReductionOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const IndexType idx3,
                                               const RealType* data1,
                                               const RealType* data2,
                                               const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], tnlCudaMax( tnlCudaAbs( data2[ idx2 ] ), tnlCudaAbs( data2[ idx3 ] ) ) );
   };

   __device__ RealType firstReductionOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2,
                                               const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], tnlCudaAbs( data2[ idx2 ] ) );
   };

   __device__ RealType commonReductionOnDevice( const IndexType idx1,
                                                const IndexType idx2,
                                                const RealType* data ) const
   {
      return tnlCudaMax( data[ idx1 ], tnlCudaAbs( data[ idx2 ] ) );
   };
};


#include <implementation/core/cuda/reduction-operations_impl.h>

#endif

#endif /* REDUCTION_OPERATIONS_H_ */
