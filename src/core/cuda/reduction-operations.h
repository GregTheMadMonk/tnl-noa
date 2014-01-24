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

/***
 * This function returns minimum of two numbers stored on the device.
 */
template< class T > __device__ T tnlCudaMin( const T& a,
                                             const T& b )
{
   return a < b ? a : b;
}

__device__ inline int tnlCudaMin( const int& a,
                                  const int& b )
{
   return min( a, b );
}

__device__ inline  float tnlCudaMin( const float& a,
                                     const float& b )
{
   return fminf( a, b );
}

__device__ inline  double tnlCudaMin( const double& a,
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

__device__  inline int tnlCudaMax( const int& a,
                                   const int& b )
{
   return max( a, b );
}

__device__  inline float tnlCudaMax( const float& a,
                                     const float& b )
{
   return fmaxf( a, b );
}

__device__  inline double tnlCudaMax( const double& a,
                                      const double& b )
{
   return fmax( a, b );
}

/***
 * This function returns absolute value of given number on the device.
 */
__device__  inline int tnlCudaAbs( const int& a )
{
   return abs( a );
}

__device__  inline long int tnlCudaAbs( const long int& a )
{
   return abs( a );
}

__device__  inline float tnlCudaAbs( const float& a )
{
   return fabs( a );
}

__device__  inline double tnlCudaAbs( const double& a )
{
   return fabs( a );
}

__device__  inline long double tnlCudaAbs( const long double& a )
{
   return fabs( ( double ) a );
}

template< typename Type1, typename Type2 >
__device__ Type1 tnlCudaPow( const Type1& x, const Type2& power )
{
   return ( Type1 ) pow( ( double ) x, ( double ) power );
}
#endif

template< typename Real, typename Index >
class tnlParallelReductionSum
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

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
#ifdef HAVE_CUDA
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
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + data2[ idx2 ] + data2[ idx3 ];
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
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

   __device__ ResultType commonReductionOnDevice( const ResultType& a,
                                                  const ResultType& b ) const
   {
      return a + b;
   };


   __device__ RealType identity() const
   {
      return 0;
   };

   __device__ void performInPlace( ResultType& a,
                                   const ResultType& b ) const
   {
      a += b;
   }

#endif
};

template< typename Real, typename Index >
class tnlParallelReductionMin
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMin< Real, Index > LaterReductionOperation;

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
      return Min( current, data1[ idx ] );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaMin( data1[ idx1 ], data1[ idx2 ] );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return data1[ idx1 ];
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], tnlCudaMin(  data2[ idx2 ],  data2[ idx3 ] ) );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], data2[ idx2 ] );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return tnlCudaMin( data[ idx1 ], data[ idx2 ] );
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionMax
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMax< Real, Index > LaterReductionOperation;

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
      return Max( current, data1[ idx ] );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaMax( data1[ idx1 ], data1[ idx2 ] );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return data1[ idx1 ];
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], tnlCudaMax( data2[ idx2 ], data2[ idx3 ] ) );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], data2[ idx2 ] );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return tnlCudaMax( data[ idx1 ], data[ idx2 ] );
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionAbsSum
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return tnlAbs( data1[ idx ] );
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return current + tnlAbs( data1[ idx ] );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaAbs( data1[ idx1 ] ) + tnlCudaAbs( data1[ idx2 ] );
   };

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaAbs( data1[ idx1 ] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + tnlCudaAbs( data2[ idx2 ] ) + tnlCudaAbs( data2[ idx3 ] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + tnlCudaAbs( data2[ idx2 ] );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionAbsMin
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMin< Real, Index > LaterReductionOperation;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return tnlAbs( data1[ idx ] );
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return Min( current, tnlAbs( data1[ idx ] ) );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaMin( tnlCudaAbs( data1[ idx1 ] ), tnlCudaAbs( data1[ idx2 ] ) );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaAbs( data1[ idx1 ] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], tnlCudaMin(  tnlCudaAbs( data2[ idx2 ] ),  tnlCudaAbs( data2[ idx3 ] ) ) );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], tnlCudaAbs( data2[ idx2 ] ) );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return tnlCudaMin( data[ idx1 ], tnlCudaAbs( data[ idx2 ] ) );
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionAbsMax
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMax< Real, Index > LaterReductionOperation;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return tnlAbs( data1[ idx ] );
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return Max( current, tnlAbs( data1[ idx ] ) );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaMax( tnlCudaAbs( data1[ idx1 ] ), tnlCudaAbs( data1[ idx2 ] ) );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaAbs( data1[ idx1 ] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], tnlCudaMax( tnlCudaAbs( data2[ idx2 ] ), tnlCudaAbs( data2[ idx3 ] ) ) );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], tnlCudaAbs( data2[ idx2 ] ) );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return tnlCudaMax( data[ idx1 ], tnlCudaAbs( data[ idx2 ] ) );
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionLogicalAnd
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionLogicalAnd< Real, Index > LaterReductionOperation;

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
      return current && data1[ idx ];
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return data1[ idx1 ] && data1[ idx2 ];
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return data1[ idx1 ];
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] && data2[ idx2 ] && data2[ idx3 ];
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] && data2[ idx2 ];
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return data[ idx1 ] && data[ idx2 ];
   };
#endif
};


template< typename Real, typename Index >
class tnlParallelReductionLogicalOr
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionLogicalOr< Real, Index > LaterReductionOperation;

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
      return current || data1[ idx ];
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return data1[ idx1 ] || data1[ idx2 ];
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return data1[ idx1 ];
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] || data2[ idx2 ] || data2[ idx3 ];
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] || data2[ idx2 ];
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return data[ idx1 ] || data[ idx2 ];
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionLpNorm
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   void setPower( const RealType& p )
   {
      this -> p = p;
   };

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return pow( tnlAbs( data1[ idx ] ), p );
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return current + pow( tnlAbs( data1[ idx ] ), p );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaPow( tnlCudaAbs( data1[ idx1 ] ), p ) + tnlCudaPow( tnlCudaAbs( data1[ idx2 ] ), p );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaPow( tnlCudaAbs( data1[ idx1 ] ), p );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] +
             tnlCudaPow( tnlCudaAbs( data2[ idx2 ] ), p ) +
             tnlCudaPow( tnlCudaAbs( data2[ idx3 ] ), p );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + tnlCudaPow( tnlCudaAbs( data2[ idx2 ] ), p );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
#endif

   protected:

   RealType p;
};

template< typename Real, typename Index >
class tnlParallelReductionEqualities
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef bool ResultType;
   typedef tnlParallelReductionLogicalAnd< bool, Index > LaterReductionOperation;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return  ( data1[ idx ] == data2[ idx ] );
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return current && ( data1[ idx ] == data2[ idx ] );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return ( data1[ idx1 ] == data2[ idx1 ] ) && ( data1[ idx2 ] == data2[ idx2] );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return ( data1[ idx1 ]== data2[ idx1 ] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] &&
             ( data2[ idx2 ] == data2[ idx2] ) &&
             ( data2[ idx3 ] == data3[ idx3] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] && ( data2[ idx2 ] == data3[ idx2 ] );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return data[ idx1 ] && data[ idx2 ];
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionInequalities
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef bool ResultType;
   typedef tnlParallelReductionLogicalAnd< bool, Index > LaterReductionOperation;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return  ( data1[ idx ] != data2[ idx ] );
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return current && ( data1[ idx ] != data2[ idx ] );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return ( data1[ idx1 ] != data2[ idx1 ] ) && ( data1[ idx2 ] != data2[ idx2] );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return ( data1[ idx1 ] != data2[ idx1 ] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] &&
             ( data2[ idx2 ] != data2[ idx2] ) &&
             ( data2[ idx3 ] != data3[ idx3] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] && ( data2[ idx2 ] != data3[ idx2 ] );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return data[ idx1 ] && data[ idx2 ];
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionScalarProduct
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return  data1[ idx ] * data2[ idx ];
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return current + ( data1[ idx ] * data2[ idx ] );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return ( data1[ idx1 ] * data2[ idx1 ] ) + ( data1[ idx2 ] * data2[ idx2] );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return ( data1[ idx1 ] * data2[ idx1 ] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] +
             ( data2[ idx2 ] * data2[ idx2] ) +
             ( data2[ idx3 ] * data3[ idx3] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + ( data2[ idx2 ] * data3[ idx2 ] );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffSum
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return data1[ idx ] - data2[ idx ];
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return current + ( data1[ idx ] - data2[ idx ] );
   };
#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return ( data1[ idx1 ] - data2[ idx1 ] ) + ( data1[ idx2 ] - data2[ idx2 ] );
   };

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return data1[ idx1 ] - data2[ idx1 ];
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] +
             ( data2[ idx2 ] - data3[ idx2 ] ) +
             ( data2[ idx3 ] - data3[ idx3 ] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + ( data2[ idx2 ] - data3[ idx2 ] );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffMin
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMin< Real, Index > LaterReductionOperation;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return data1[ idx ] - data2[ idx ];
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return Min( current, data1[ idx ] - data2[ idx ] );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaMin( data1[ idx1 ] - data2[ idx1 ], data1[ idx2 ] - data2[ idx2 ] );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return data1[ idx1 ] - data2[ idx1 ];
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ],
                         tnlCudaMin(  data2[ idx2 ] - data3[ idx2 ],
                                      data2[ idx3 ] - data3[ idx3 ] ) );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], data2[ idx2 ] - data3[ idx2 ] );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return tnlCudaMin( data[ idx1 ], data[ idx2 ] );
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffMax
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMax< Real, Index > LaterReductionOperation;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return data1[ idx ] - data2[ idx ];
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return Max( current, data1[ idx ] - data2[ idx ] );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaMax( data1[ idx1 ] - data2[ idx1 ],
                         data1[ idx2 ] - data2[ idx2 ] );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return data1[ idx1 ] - data2[ idx1 ];
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ],
                         tnlCudaMax( data2[ idx2 ] - data3[ idx2 ],
                                     data2[ idx3 ] - data3[ idx3 ] ) );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], data2[ idx2 ] - data3[ idx2 ] );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return tnlCudaMax( data[ idx1 ], data[ idx2 ] );
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffAbsSum
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return tnlAbs( data1[ idx ] - data2[ idx ] );
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return current + tnlAbs( data1[ idx ] - data2[ idx ] );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaAbs( data1[ idx1 ] - data2[ idx1 ] ) + tnlCudaAbs( data1[ idx2 ] - data2[ idx2 ] );
   };

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaAbs( data1[ idx1 ] - data2[ idx1 ] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] +
             tnlCudaAbs( data2[ idx2 ] - data3[ idx2 ] ) +
             tnlCudaAbs( data2[ idx3 ] - data3[ idx3 ] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] +
             tnlCudaAbs( data2[ idx2 ] - data3[ idx2 ] );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffAbsMin
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMin< Real, Index > LaterReductionOperation;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return tnlAbs( data1[ idx ] - data2[ idx ] );
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return Min( current, tnlAbs( data1[ idx ] - data2[ idx ] ) );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaMin( tnlCudaAbs( data1[ idx1 ] - data2[ idx1 ] ),
                         tnlCudaAbs( data1[ idx2 ] - data2[ idx2 ] ) );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaAbs( data1[ idx1 ] - data2[ idx1 ] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ],
                         tnlCudaMin(  tnlCudaAbs( data2[ idx2 ] - data3[ idx2 ] ),
                                      tnlCudaAbs( data2[ idx3 ] - data3[ idx3 ] ) ) );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ],
                         tnlCudaAbs( data2[ idx2 ] - data3[ idx2 ] ) );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      //return tnlCudaMin( data[ idx1 ], tnlCudaAbs( data[ idx2 ] ) );
      return tnlCudaMin( data[ idx1 ], data[ idx2 ] );
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffAbsMax
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMax< Real, Index > LaterReductionOperation;

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return tnlAbs( data1[ idx ] -data2[ idx ] );
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return Max( current, tnlAbs( data1[ idx ] - data2[ idx ] ) );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaMax( tnlCudaAbs( data1[ idx1 ] - data2[ idx1 ] ),
                         tnlCudaAbs( data1[ idx2 ] - data2[ idx2 ] ) );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaAbs( data1[ idx1 ] - data2[ idx1 ] );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ],
                         tnlCudaMax( tnlCudaAbs( data2[ idx2 ] - data3[ idx2 ] ),
                                     tnlCudaAbs( data2[ idx3 ] - data3[ idx3 ] ) ) );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ],
                         tnlCudaAbs( data2[ idx2 ] - data3[ idx2 ] ) );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      //return tnlCudaMax( data[ idx1 ], tnlCudaAbs( data[ idx2 ] ) );
      return tnlCudaMax( data[ idx1 ], data[ idx2 ] );
   };
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffLpNorm
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   void setPower( const RealType& p )
   {
      this -> p = p;
   };

   ResultType initialValueOnHost( const IndexType idx,
                                  const RealType* data1,
                                  const RealType* data2 ) const
   {
      return pow( tnlAbs( data1[ idx ] - data2[ idx ] ), p );
   };

   ResultType reduceOnHost( const IndexType idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 ) const
   {
      return current + pow( tnlAbs( data1[ idx ] - data2[ idx ] ), p );
   };

#ifdef HAVE_CUDA
   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const IndexType idx2,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaPow( tnlCudaAbs( data1[ idx1 ] - data2[ idx1 ] ), p ) +
             tnlCudaPow( tnlCudaAbs( data1[ idx2 ] - data2[ idx2 ] ), p );
   }

   __device__ ResultType initialValueOnDevice( const IndexType idx1,
                                               const RealType* data1,
                                               const RealType* data2 ) const
   {
      return tnlCudaPow( tnlCudaAbs( data1[ idx1 ] - data2[ idx1 ] ), p );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] +
             tnlCudaPow( tnlCudaAbs( data2[ idx2 ] - data3[ idx2 ] ), p ) +
             tnlCudaPow( tnlCudaAbs( data2[ idx3 ] - data3[ idx3 ] ), p );
   };

   __device__ ResultType firstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + tnlCudaPow( tnlCudaAbs( data2[ idx2 ] - data3[ idx2 ] ), p );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
#endif

   protected:

   RealType p;
};


#endif /* REDUCTION_OPERATIONS_H_ */
