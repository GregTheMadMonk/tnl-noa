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

#include <core/tnlConstants.h>
#include <core/tnlCuda.h>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <core/mfuncs.h>

/***
 * This function returns minimum of two numbers stored on the device.
 * TODO: Make it tnlMin, tnlMax etc.
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

template< class T > __device__ T tnlCudaMin( volatile const T& a,
                                             volatile const T& b )
{
   return a < b ? a : b;
}

__device__ inline int tnlCudaMin( volatile const int& a,
                                  volatile const int& b )
{
   return min( a, b );
}

__device__ inline  float tnlCudaMin( volatile const float& a,
                                     volatile const float& b )
{
   return fminf( a, b );
}

__device__ inline  double tnlCudaMin( volatile const double& a,
                                      volatile const double& b )
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

template< class T > __device__ T tnlCudaMax( volatile const T& a,
                                             volatile const T& b )
{
   return a > b ? a : b;
}

__device__  inline int tnlCudaMax( volatile const int& a,
                                   volatile const int& b )
{
   return max( a, b );
}

__device__  inline float tnlCudaMax( volatile const float& a,
                                     volatile const float& b )
{
   return fmaxf( a, b );
}

__device__  inline double tnlCudaMax( volatile const double& a,
                                      volatile const double& b )
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

__device__  inline int tnlCudaAbs( volatile const int& a )
{
   return abs( a );
}

__device__  inline long int tnlCudaAbs( volatile const long int& a )
{
   return abs( a );
}

__device__  inline float tnlCudaAbs( volatile const float& a )
{
   return fabs( a );
}

__device__  inline double tnlCudaAbs( volatile const double& a )
{
   return fabs( a );
}

__device__  inline long double tnlCudaAbs( volatile const long double& a )
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
   
   __cuda_callable__ ResultType initialValue() const { return 0; };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                              const IndexType index,
                                              const RealType* data1,
                                              const RealType* data2 ) const
   {
      result += data1[ index ];
   }
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + data2[ idx2 ] + data2[ idx3 ];
   };

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + data2[ idx2 ];
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };

   __device__ ResultType commonReductionOnDevice( ResultType& result,
                                                  const ResultType& data ) const
   {
      result += data;
   };
   
   __device__ ResultType commonReductionOnDevice( volatile ResultType& result,
                                                  volatile const ResultType& data ) const
   {
      result += data;
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

   __cuda_callable__ ResultType initialValue() const { return tnlMaxValue< ResultType>(); };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                              const IndexType index,
                                              const RealType* data1,
                                              const RealType* data2 ) const
   {
      result = tnlCudaMin( result, data1[ index ] );
   }
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], tnlCudaMin(  data2[ idx2 ],  data2[ idx3 ] ) );
   };

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], data2[ idx2 ] );
   };
   
   __device__ ResultType commonReductionOnDevice( ResultType& result,
                                                  const ResultType& data ) const
   {
      result = tnlCudaMin( result, data );
   };
   
   __device__ ResultType commonReductionOnDevice( volatile ResultType& result,
                                                  volatile const ResultType& data ) const
   {
      result = tnlCudaMin( result, data );
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

   __cuda_callable__ ResultType initialValue() const { return tnlMinValue< ResultType>(); };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                              const IndexType index,
                                              const RealType* data1,
                                              const RealType* data2 ) const
   {
      result = tnlCudaMax( result, data1[ index ] );
   }   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], tnlCudaMax( data2[ idx2 ], data2[ idx3 ] ) );
   };

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], data2[ idx2 ] );
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return tnlCudaMax( data[ idx1 ], data[ idx2 ] );
   };
   
   __device__ ResultType commonReductionOnDevice( ResultType& result,
                                                  const ResultType& data ) const
   {
      result = tnlCudaMax( result, data );
   };

   __device__ ResultType commonReductionOnDevice( volatile ResultType& result,
                                                  volatile const ResultType& data ) const
   {
      result = tnlCudaMax( result, data );
   };   
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionAbsSum : public tnlParallelReductionSum< Real, Index >
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

   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) 0; };

   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                              const IndexType index,
                                              const RealType* data1,
                                              const RealType* data2 ) const
   {
      result += tnlCudaAbs( data1[ index ] );
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + tnlCudaAbs( data2[ idx2 ] ) + tnlCudaAbs( data2[ idx3 ] );
   };

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + tnlCudaAbs( data2[ idx2 ] );
   };

   /*__device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return data1 + data2;
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return data1 + data2;
   };*/
   
   
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionAbsMin : public tnlParallelReductionMin< Real, Index >
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

   __cuda_callable__ ResultType initialValue() const { return tnlMaxValue< ResultType>(); };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                              const IndexType index,
                                              const RealType* data1,
                                              const RealType* data2 ) const
   {
      result = tnlCudaMin( result, tnlCudaAbs( data1[ index ] ) );
   }   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], tnlCudaMin(  tnlCudaAbs( data2[ idx2 ] ),  tnlCudaAbs( data2[ idx3 ] ) ) );
   };

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], tnlCudaAbs( data2[ idx2 ] ) );
   };

   /*__device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      volatile ResultType aux = tnlCudaAbs( data[ idx2 ] );
      return tnlCudaMin( data[ idx1 ],  aux );
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return tnlCudaMin( data1, data2 );
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return tnlCudaMin( data1, data2 );
   };*/
   
   
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionAbsMax : public tnlParallelReductionMax< Real, Index >
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

   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) 0; };

   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                              const IndexType index,
                                              const RealType* data1,
                                              const RealType* data2 ) const
   {
      result = tnlCudaMax( result, tnlCudaAbs( data1[ index ] ) );
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], tnlCudaMax( tnlCudaAbs( data2[ idx2 ] ), tnlCudaAbs( data2[ idx3 ] ) ) );
   };

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], tnlCudaAbs( data2[ idx2 ] ) );
   };

   /*__device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      volatile ResultType aux = tnlCudaAbs( data[ idx2 ] );
      return tnlCudaMax( data[ idx1 ], aux );
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return tnlCudaMax( data1, data2 );
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return tnlCudaMax( data1, data2 );
   };*/
   
   
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

   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) true; };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                              const IndexType index,
                                              const RealType* data1,
                                              const RealType* data2 ) const
   {
      result = result && data1[ index ];
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] && data2[ idx2 ] && data2[ idx3 ];
   };

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] && data2[ idx2 ];
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return data[ idx1 ] && data[ idx2 ];
   };
   
   __device__ ResultType commonReductionOnDevice( ResultType& result,
                                                  const ResultType& data ) const
   {
      result = result && data;
   };
   
   __device__ ResultType commonReductionOnDevice( volatile ResultType& result,
                                                  volatile const ResultType& data ) const
   {
      result = result && data;
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
   
   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) false; };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                              const IndexType index,
                                              const RealType* data1,
                                              const RealType* data2 ) const
   {
      result = result || data1[ index ];
   }


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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] || data2[ idx2 ] || data2[ idx3 ];
   };

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] || data2[ idx2 ];
   };

   __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return data[ idx1 ] || data[ idx2 ];
   };
   
   __device__ ResultType commonReductionOnDevice( ResultType& result,
                                                  const ResultType& data ) const
   {
      result = result || data;
   };
   
   __device__ ResultType commonReductionOnDevice( volatile ResultType& result,
                                                  volatile const ResultType& data ) const
   {
      result = result || data;
   };
   
   
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionLpNorm : public tnlParallelReductionSum< Real, Index >
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

   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) 0; };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                              const IndexType index,
                                              const RealType* data1,
                                              const RealType* data2 ) const
   {
      result += tnlCudaPow( tnlCudaAbs( data1[ index ] ), p );
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + tnlCudaPow( tnlCudaAbs( data2[ idx2 ] ), p );
   };

   /*__device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return data1 + data2;
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return data1 + data2;
   };*/
   
   
#endif

   protected:

   RealType p;
};

template< typename Real, typename Index >
class tnlParallelReductionEqualities : public tnlParallelReductionLogicalAnd< bool, Index >
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

   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) true; }; 
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                              const IndexType index,
                                              const RealType* data1,
                                              const RealType* data2 ) const
   {
      result = result && ( data1[ index ] == data2[ index ] );
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] && ( data2[ idx2 ] == data3[ idx2 ] );
   };

  /* __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return data[ idx1 ] && data[ idx2 ];
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return data1 && data2;
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return data1 && data2;
   };*/

   
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionInequalities : public tnlParallelReductionLogicalAnd< bool, Index >
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

   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) false; };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                              const IndexType index,
                                              const RealType* data1,
                                              const RealType* data2 ) const
   {
      result = result && ( data1[ index ] != data2[ index ] );
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] && ( data2[ idx2 ] != data3[ idx2 ] );
   };

   /*__device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return data[ idx1 ] && data[ idx2 ];
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return data1 && data2;
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return data1 && data2;
   };*/
   
   
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionScalarProduct : public tnlParallelReductionSum< Real, Index >
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

   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) 0; };
   
   __cuda_callable__ inline void cudaFirstReduction( ResultType& result, 
                                                 const IndexType index,
                                                 const RealType* data1,
                                                 const RealType* data2 ) const
   {
      result += data1[ index ] * data2[ index ];
   }

   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const IndexType idx3,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] +
             ( data2[ idx2 ] * data3[ idx2] ) +
             ( data2[ idx3 ] * data3[ idx3] );
   };

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + ( data2[ idx2 ] * data3[ idx2 ] );
   };

   /*__device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return data1 + data2;
   };

   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return data1 + data2;
   };*/

#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffSum : public tnlParallelReductionSum< Real, Index >
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
   
   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) 0; };   
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                          const IndexType index,
                                          const RealType* data1,
                                          const RealType* data2 ) const
   {
      result += data1[ index ] - data2[ index ];
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + ( data2[ idx2 ] - data3[ idx2 ] );
   };

   /*__device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return data1 + data2;
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return data1 + data2;
   };*/
   

#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffMin : public tnlParallelReductionMin< Real, Index >
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

   __cuda_callable__ ResultType initialValue() const { return tnlMaxValue< ResultType>(); };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                          const IndexType index,
                                          const RealType* data1,
                                          const RealType* data2 ) const
   {
      result = tnlCudaMin( result, data1[ index ] - data2[ index ] );
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ], data2[ idx2 ] - data3[ idx2 ] );
   };

  /* __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return tnlCudaMin( data[ idx1 ], data[ idx2 ] );
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return tnlCudaMin( data1, data2 );
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return tnlCudaMin( data1, data2 );
   };*/
   

#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffMax : public tnlParallelReductionMax< Real, Index >
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

   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) 0; };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                              const IndexType index,
                                              const RealType* data1,
                                              const RealType* data2 ) const
   {
      result = tnlCudaMax( result, data1[ index ] - data2[ index ] );
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ], data2[ idx2 ] - data3[ idx2 ] );
   };

  /* __device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return tnlCudaMax( data[ idx1 ], data[ idx2 ] );
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return tnlCudaMax( data1, data2 );
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return tnlCudaMax( data1, data2 );
   };*/

   
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffAbsSum : public tnlParallelReductionMax< Real, Index >
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

   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) 0; };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                          const IndexType index,
                                          const RealType* data1,
                                          const RealType* data2 ) const
   {
      result += tnlCudaAbs( data1[ index ] - data2[ index ] );
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] +
             tnlCudaAbs( data2[ idx2 ] - data3[ idx2 ] );
   };

   /*__device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return data1 + data2;
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return data1 + data2;
   };*/
   
   
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffAbsMin : public tnlParallelReductionMin< Real, Index >
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

   __cuda_callable__ ResultType initialValue() const { return tnlMaxValue< ResultType>(); };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                          const IndexType index,
                                          const RealType* data1,
                                          const RealType* data2 ) const
   {
      result = tnlCudaMin( result, tnlCudaAbs( data1[ index ] - data2[ index ] ) );
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMin( data1[ idx1 ],
                         tnlCudaAbs( data2[ idx2 ] - data3[ idx2 ] ) );
   };

   /*__device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      //return tnlCudaMin( data[ idx1 ], tnlCudaAbs( data[ idx2 ] ) );
      return tnlCudaMin( data[ idx1 ], data[ idx2 ] );
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return tnlCudaMin( data1, data2 );
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return tnlCudaMin( data1, data2 );
   };*/

   
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffAbsMax : public tnlParallelReductionMax< Real, Index >
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

   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) 0; };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                          const IndexType index,
                                          const RealType* data1,
                                          const RealType* data2 ) const
   {
      result = tnlCudaMax( result, tnlCudaAbs( data1[ index ] - data2[ index ] ) );
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return tnlCudaMax( data1[ idx1 ],
                         tnlCudaAbs( data2[ idx2 ] - data3[ idx2 ] ) );
   };

   /*__device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      //return tnlCudaMax( data[ idx1 ], tnlCudaAbs( data[ idx2 ] ) );
      return tnlCudaMax( data[ idx1 ], data[ idx2 ] );
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return tnlCudaMax( data1, data2 );
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return tnlCudaMax( data1, data2 );
   };*/

   
#endif
};

template< typename Real, typename Index >
class tnlParallelReductionDiffLpNorm : public tnlParallelReductionSum< Real, Index >
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

   __cuda_callable__ ResultType initialValue() const { return ( ResultType ) 0; };
   
   __cuda_callable__ void cudaFirstReduction( ResultType& result, 
                                          const IndexType index,
                                          const RealType* data1,
                                          const RealType* data2 ) const
   {
      result += tnlCudaPow( tnlCudaAbs( data1[ index ] - data2[ index ] ), p );
   }
   
   
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
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

   __device__ ResultType cudaFirstReductionOnDevice( const IndexType idx1,
                                                 const IndexType idx2,
                                                 const ResultType* data1,
                                                 const RealType* data2,
                                                 const RealType* data3 ) const
   {
      return data1[ idx1 ] + tnlCudaPow( tnlCudaAbs( data2[ idx2 ] - data3[ idx2 ] ), p );
   };

   /*__device__ ResultType commonReductionOnDevice( const IndexType idx1,
                                                  const IndexType idx2,
                                                  volatile const ResultType* data ) const
   {
      return data[ idx1 ] + data[ idx2 ];
   };
   
   __device__ ResultType commonReductionOnDevice( const ResultType& data1,
                                                  const ResultType& data2 ) const
   {
      return data1 + data2;
   };
   
   __device__ ResultType commonReductionOnDevice( volatile const ResultType& data1,
                                                  volatile const ResultType& data2 ) const
   {
      return data1 + data2;
   };*/

   
#endif

   protected:

   RealType p;
};


#endif /* REDUCTION_OPERATIONS_H_ */
