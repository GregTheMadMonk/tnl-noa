/***************************************************************************
                      tnlVectorCUDA.h  -  description
 -------------------
 begin                : Dec 27, 2009
 copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef TNLVECTORCUDA_H_
#define TNLVECTORCUDA_H_

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#else
#include <iostream>
using namespace std;
#endif

#include <core/tnlAssert.h>
#include <core/tnlArrayManager.h>
#include <core/tnlObject.h>
#include <core/tnlVector.h>
#include <core/param-types.h>
#include <core/mfuncs.h>
#include <core/tnlCudaSupport.h>
#include <core/low-level/cuda-long-vector-kernels.h>
#include <debug/tnlDebug.h>


template< typename RealType, typename IndexType >
class tnlVector< RealType, tnlCuda, IndexType > : public tnlArrayManager< RealType, tnlCuda, IndexType >
{
   //! We do not allow constructor without parameters.
   tnlVector(){};

   /****
    * We do not allow copy constructors as well to avoid having two
    * vectors with the same name.
    */
   tnlVector( const tnlVector< RealType, tnlHost, IndexType >& v ){};

   tnlVector( const tnlVector< RealType, tnlCuda, IndexType >& v ){};

   public:

   //! Basic constructor
   tnlVector( const tnlString& name, IndexType _size = 0 );

   //! Constructor with another long vector as template
   tnlVector( const tnlString& name, const tnlVector< RealType, tnlHost, IndexType >& v );

   //! Constructor with another long vector as template
   tnlVector( const tnlString& name, const tnlVector< RealType, tnlCuda, IndexType >& v );

   tnlVector< RealType, tnlCuda, IndexType >& operator = ( const tnlVector< RealType, tnlCuda, IndexType >& cuda_vector );

   tnlVector< RealType, tnlCuda, IndexType >& operator = ( const tnlVector< RealType, tnlHost, IndexType >& long_vector );

   template< typename RealType2, typename IndexType2 >
   tnlVector< RealType, tnlCuda, IndexType >& operator = ( const tnlVector< RealType2, tnlCuda, IndexType2 >& long_vector );

   bool operator == ( const tnlVector< RealType, tnlCuda, IndexType >& long_vector ) const;

   bool operator != ( const tnlVector< RealType, tnlCuda, IndexType >& long_vector ) const;

   bool operator == ( const tnlVector< RealType, tnlHost, IndexType >& long_vector ) const;

   bool operator != ( const tnlVector< RealType, tnlHost, IndexType >& long_vector ) const;

   void setValue( const RealType& v );

    ~tnlVector();
};

template< typename RealType, typename IndexType >
bool operator == ( const tnlVector< RealType, tnlHost, IndexType >& host_vector,
                   const tnlVector< RealType, tnlCuda, IndexType >& cuda_vector );

template< typename RealType, typename IndexType >
bool operator != ( const tnlVector< RealType, tnlHost, IndexType >& host_vector,
                   const tnlVector< RealType, tnlCuda, IndexType >& cuda_vector );

template< typename RealType, typename IndexType >
ostream& operator << ( ostream& str, const tnlVector< RealType, tnlCuda, IndexType >& vec );

/****
 * Here are some Blas style functions. They are not methods
 * because it would put too many restrictions on type RealType.
 * In fact, we would like to use tnlVector to store objects
 * like edges or triangles in case of meshes. For these objects
 * operations like +, min or max are not defined.
 */

template< typename RealType, typename IndexType >
RealType tnlMax( const tnlVector< RealType, tnlCuda, IndexType >& v );

template< typename RealType, typename IndexType >
RealType tnlMin( const tnlVector< RealType, tnlCuda, IndexType >& v );

template< typename RealType, typename IndexType >
RealType tnlAbsMax( const tnlVector< RealType, tnlCuda, IndexType >& v );

template< typename RealType, typename IndexType >
RealType tnlAbsMin( const tnlVector< RealType, tnlCuda, IndexType >& v );

template< typename RealType, typename IndexType >
RealType tnlLpNorm( const tnlVector< RealType, tnlCuda, IndexType >& v, const RealType& p );

template< typename RealType, typename IndexType >
RealType tnlDifferenceMax( const tnlVector< RealType, tnlCuda, IndexType >& u,
                       const tnlVector< RealType, tnlCuda, IndexType >& v );

template< typename RealType, typename IndexType >
RealType tnlDifferenceMin( const tnlVector< RealType, tnlCuda, IndexType >& u,
                       const tnlVector< RealType, tnlCuda, IndexType >& v );

template< typename RealType, typename IndexType >
RealType tnlDifferenceAbsMax( const tnlVector< RealType, tnlCuda, IndexType >& u,
                          const tnlVector< RealType, tnlCuda, IndexType >& v );

template< typename RealType, typename IndexType >
RealType tnlDifferenceAbsMin( const tnlVector< RealType, tnlCuda, IndexType >& u,
                          const tnlVector< RealType, tnlCuda, IndexType >& v );

template< typename RealType, typename IndexType >
RealType tnlDifferenceLpNorm( const tnlVector< RealType, tnlCuda, IndexType >& u,
                          const tnlVector< RealType, tnlCuda, IndexType >& v, const RealType& p );

template< typename RealType, typename IndexType >
RealType tnlDifferenceSum( const tnlVector< RealType, tnlCuda, IndexType >& u,
                       const tnlVector< RealType, tnlCuda, IndexType >& v );

template< typename RealType, typename IndexType >
RealType tnlSum( const tnlVector< RealType, tnlCuda, IndexType >& v );

//! Computes u *= aplha
template< typename RealType, typename IndexType >
void tnlScalarMultiplication( const RealType& alpha,
                              tnlVector< RealType, tnlCuda, IndexType >& u );

//! Computes scalar dot product
template< typename RealType, typename IndexType >
RealType tnlSDOT( const tnlVector< RealType, tnlCuda, IndexType >& u ,
              const tnlVector< RealType, tnlCuda, IndexType >& v );

//! Computes SAXPY operation (Scalar Alpha X Pus Y ).
template< typename RealType, typename IndexType >
void tnlSAXPY( const RealType& alpha,
               const tnlVector< RealType, tnlCuda, IndexType >& x,
               tnlVector< RealType, tnlCuda, IndexType >& y );

//! Computes SAXMY operation (Scalar Alpha X Minus Y ).
/*!**
 * It is not a standard BLAS function but it is useful for GMRES solver.
 */
template< typename RealType, typename IndexType >
void tnlSAXMY( const RealType& alpha,
               const tnlVector< RealType, tnlCuda, IndexType >& x,
               tnlVector< RealType, tnlCuda, IndexType >& y );


#ifdef HAVE_CUDA
template< typename RealType, typename IndexType >
__global__ void tnlVectorCUDASetValueKernel( RealType* data,
                                                 const IndexType size,
                                                 const RealType v  );
#endif


template< typename RealType, typename IndexType >
tnlVector< RealType, tnlCuda, IndexType > :: tnlVector( const tnlString& name, IndexType _size )
: tnlArrayManager< RealType, tnlCuda, IndexType >( name )
{
  setSize( _size );
};

template< typename RealType, typename IndexType >
tnlVector< RealType, tnlCuda, IndexType > :: tnlVector( const tnlString& name,
                                                const tnlVector< RealType, tnlHost, IndexType >& v )
   : tnlArrayManager< RealType, tnlCuda, IndexType >( name )
{
   setSize( v. getSize() );
};

template< typename RealType, typename IndexType >
tnlVector< RealType, tnlCuda, IndexType > :: tnlVector( const tnlString& name,
                                                        const tnlVector< RealType, tnlCuda, IndexType >& v )
   : tnlArrayManager< RealType, tnlCuda, IndexType >( name )
{
   setSize( v. getSize() );
};

template< typename RealType, typename IndexType >
tnlVector< RealType, tnlCuda, IndexType >& tnlVector< RealType, tnlCuda, IndexType > :: operator = ( const tnlVector< RealType, tnlHost, IndexType >& vector )
{
   tnlArrayManager< RealType, tnlCuda, IndexType > :: operator = ( vector );
   return *this;
};

template< typename RealType, typename IndexType >
tnlVector< RealType, tnlCuda, IndexType >& tnlVector< RealType, tnlCuda, IndexType > :: operator = ( const tnlVector< RealType, tnlCuda, IndexType >& vector )
{
   tnlArrayManager< RealType, tnlCuda, IndexType > :: operator = ( vector );
   return *this;
};

template< typename RealType, typename IndexType >
bool tnlVector< RealType, tnlCuda, IndexType > :: operator == ( const tnlVector< RealType, tnlCuda, IndexType >& cuda_vector ) const
{
   tnlAssert( this -> getSize() == cuda_vector. getSize(),
              cerr << "You try to compare two long vectors with different sizes." << endl
                   << "The first one is " << this -> getName() << " with size " << this -> getSize()
                   << " while the second one is " << cuda_vector. getName() << " with size " << cuda_vector. getSize() << "." );
#ifdef HAVE_CUDA
   if( tnlCUDALongVectorComparison( this -> getSize(),
                                    this -> getData(),
                                    cuda_vector. getData() ) )
      return true;
   return false;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return;
#endif
};

template< typename RealType, typename IndexType >
bool tnlVector< RealType, tnlCuda, IndexType > :: operator != ( const tnlVector< RealType, tnlCuda, IndexType >& long_vector ) const
{
   return ! ( ( *this ) == long_vector );
};

template< typename RealType, typename IndexType >
bool tnlVector< RealType, tnlCuda, IndexType > :: operator == ( const tnlVector< RealType, tnlHost, IndexType >& host_vector ) const
{
   tnlAssert( this -> getSize() == host_vector. getSize(),
              cerr << "You try to compare two long vectors with different sizes." << endl
                   << "The first one is " << this -> getName() << " with size " << this -> getSize()
                   << " while the second one is " << host_vector. getName() << " with size " << host_vector. getSize() << "." );
#ifdef HAVE_CUDA
   IndexType host_buffer_size = :: Min( ( IndexType ) ( tnlGPUvsCPUTransferBufferSize / sizeof( RealType ) ),
                                    this -> getSize() );
   RealType* host_buffer = new RealType[ host_buffer_size ];
   if( ! host_buffer )
   {
      cerr << "I am sorry but I cannot allocate supporting buffer on the host for comparing data between CUDA GPU and CPU." << endl;
      return false;
   }
   IndexType compared( 0 );
   while( compared < this -> getSize() )
   {
      IndexType transfer = Min( this -> getSize() - compared, host_buffer_size );
      if( cudaMemcpy( ( void* ) host_buffer,
                      ( void* ) & ( this -> getData()[ compared ] ),
                      transfer * sizeof( RealType ),
                      cudaMemcpyDeviceToHost ) != cudaSuccess )
      {
         cerr << "Transfer of data to the element number of the CUDA long vector " << this -> getName()
              << " from the device failed." << endl;
         checkCUDAError( __FILE__, __LINE__ );
         delete[] host_buffer;
         return false;
      }
      IndexType buffer_IndexType( 0 );
      while( buffer_IndexType < transfer &&
             host_buffer[ buffer_IndexType ] == host_vector. getElement( compared ) )
      {
         buffer_IndexType ++;
         compared ++;
      }
      if( buffer_IndexType < transfer )
      {
         delete[] host_buffer;
         return false;
      }
   }
   delete[] host_buffer;
   return true;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return;
#endif
};

template< typename RealType, typename IndexType >
bool tnlVector< RealType, tnlCuda, IndexType > :: operator != ( const tnlVector< RealType, tnlHost, IndexType >& long_vector ) const
{
   return ! ( ( *this ) == long_vector );
};

#ifdef HAVE_CUDA
template< typename RealType, typename IndexType, typename RealType2, typename IndexType2 >
__global__ void tnlVectorCUDAAssignOperatorKernel( const RealType2* source,
                                                       const IndexType targetSize,
                                                       RealType* target )
{
   const IndexType i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < targetSize )
      target[ i ] = ( RealType ) source[ ( IndexType2 ) i ];
}
#endif



template< typename RealType, typename IndexType >
void tnlVector< RealType, tnlCuda, IndexType > :: setValue( const RealType& v )
{
#ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = this -> getSize() / 512 + 1;

   tnlVectorCUDASetValueKernel<<< gridSize, blockSize >>>( this -> getData(),
                                                               this -> getSize(),
                                                               v );
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
}

template< typename RealType, typename IndexType >
tnlVector< RealType, tnlCuda, IndexType > :: ~tnlVector()
{
};


#ifdef HAVE_CUDA
/****
 * TODO: !!! If the type RealType would be some larger type this might fail.
 * CUDA cannot pass more than 256 bytes. We may need to pass the
 * parameter v explicitly.
 */
template< typename RealType, typename IndexType >
__global__ void tnlVectorCUDASetValueKernel( RealType* data,
                                                 const IndexType size,
                                                 const RealType v )
{
   const IndexType i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      data[ i ] = v;
}
#endif

template< typename RealType, typename IndexType >
bool operator == ( const tnlVector< RealType, tnlHost, IndexType >& host_vector,
		             const tnlVector< RealType, tnlCuda, IndexType >& cuda_vector )
{
	return ( cuda_vector == host_vector );
};

template< typename RealType, typename IndexType >
bool operator != ( const tnlVector< RealType, tnlHost, IndexType >& host_vector,
		             const tnlVector< RealType, tnlCuda, IndexType >& cuda_vector )
{
	return !( cuda_vector == host_vector );
};

template< typename RealType, typename IndexType >
ostream& operator << ( ostream& str, const tnlVector< RealType, tnlCuda, IndexType >& vec )
{
#ifdef HAVE_CUDA
   IndexType size = vec. getSize();
   RealType buffer[ 1024 ];
   IndexType pos( 0 );
   while( pos < size )
   {
	   int transfer = Min( size - pos, 1024 );
	   if( cudaMemcpy( buffer,
			             &( vec. getData()[ pos ] ),
   		    		    transfer * sizeof( RealType ), cudaMemcpyDeviceToHost ) != cudaSuccess )
	   {
		   cerr << "Transfer of data from CUDA device ( " << vec. getName()
   		        << " ) failed." << endl;
		   checkCUDAError( __FILE__, __LINE__ );
		   return str;
	   }
	   for( IndexType i = 0; i < transfer; i ++ )
		   str << buffer[ i ] <<  " ";
	   pos += transfer;
   }
#else
   cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
   return str;

};

template< typename RealType, typename IndexType >
RealType tnlMax( const tnlVector< RealType, tnlCuda, IndexType >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   RealType result( 0 );
   tnlCUDALongVectorReduction< RealType, RealType, IndexType, tnlParallelReductionMax >( v. getSize(),
                                                                             v. getData(),
                                                                             ( RealType* ) NULL,
                                                                             result,
                                                                             ( RealType ) 0 );
   return result;
}

template< typename RealType, typename IndexType >
RealType tnlMin( const tnlVector< RealType, tnlCuda, IndexType >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   RealType result( 0 );
   tnlCUDALongVectorReduction< RealType, RealType, IndexType, tnlParallelReductionMin >( v. getSize(),
                                                                             v. getData(),
                                                                             ( RealType* ) NULL,
                                                                             result,
                                                                             ( RealType ) 0 );
   return result;
}

template< typename RealType, typename IndexType >
RealType tnlAbsMax( const tnlVector< RealType, tnlCuda, IndexType >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   RealType result( 0 );
   tnlCUDALongVectorReduction< RealType, RealType, IndexType, tnlParallelReductionAbsMax >( v. getSize(),
                                                                                v. getData(),
                                                                                ( RealType* ) NULL,
                                                                                result,
                                                                                ( RealType ) 0 );
   return result;
}

template< typename RealType, typename IndexType >
RealType tnlAbsMin( const tnlVector< RealType, tnlCuda, IndexType >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   RealType result( 0 );
   tnlCUDALongVectorReduction< RealType, RealType, IndexType, tnlParallelReductionAbsMin >( v. getSize(),
                                                                                v. getData(),
                                                                                ( RealType* ) NULL,
                                                                                result,
                                                                                ( RealType ) 0 );
   return result;
}

template< typename RealType, typename IndexType >
RealType tnlLpNorm( const tnlVector< RealType, tnlCuda, IndexType >& v, const RealType& p )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   RealType result( 0 );
   tnlCUDALongVectorReduction< RealType, RealType, IndexType, tnlParallelReductionLpNorm >( v. getSize(),
                                                                                v. getData(),
                                                                                ( RealType* ) NULL,
                                                                                result,
                                                                                p );
   return result;
}

template< typename RealType, typename IndexType >
RealType tnlSum( const tnlVector< RealType, tnlCuda, IndexType >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   RealType result( 0 );
   tnlCUDALongVectorReduction< RealType, RealType, IndexType, tnlParallelReductionSum >( v. getSize(),
                                                                             v. getData(),
                                                                             ( RealType* ) NULL,
                                                                             result,
                                                                             ( RealType ) 0 );
   return result;
}

#ifdef HAVE_CUDA
template< typename RealType, typename IndexType >
__global__ void tnlVectorCUDAScalaMultiplicationKernel( const IndexType size,
                                                            const RealType alpha,
                                                            RealType* x )
{
   IndexType tid = blockDim. x * blockIdx. x + threadIdx. x;
   if( tid < size )
      x[ tid ] *= alpha;
}
#endif

template< typename RealType, typename IndexType >
RealType tnlDifferenceMax( const tnlVector< RealType, tnlCuda, IndexType >& u,
                       const tnlVector< RealType, tnlCuda, IndexType >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   tnlAssert( false, ); // TODO: fix this
   /*RealType result = u[ 0 ] - v[ 0 ];
   const IndexType n = v. getSize();
   for( IndexType i = 1; i < n; i ++ )
      result = Max( result, u[ i ] - v[ i ] );
   return result;*/
}

template< typename RealType, typename IndexType >
RealType tnlDifferenceMin( const tnlVector< RealType, tnlCuda, IndexType >& u,
                       const tnlVector< RealType, tnlCuda, IndexType >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   tnlAssert( false, ); // TODO: fix this
   /*RealType result = u[ 0 ] - v[ 0 ];
   const IndexType n = v. getSize();
   for( IndexType i = 1; i < n; i ++ )
      result = Min( result, u[ i ] - v[ i ] );
   return result;*/
}

template< typename RealType, typename IndexType >
RealType tnlDifferenceAbsMax( const tnlVector< RealType, tnlCuda, IndexType >& u,
                          const tnlVector< RealType, tnlCuda, IndexType >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   tnlAssert( false, ); // TODO: fix this
   /*RealType result = u[ 0 ] - v[ 0 ];
   const IndexType n = v. getSize();
   for( IndexType i = 1; i < n; i ++ )
      result = Max( result, ( RealType ) fabs( u[ i ] - v[ i ] ) );
   return result;*/
}

template< typename RealType, typename IndexType >
RealType tnlDifferenceAbsMin( const tnlVector< RealType, tnlCuda, IndexType >& u,
                          const tnlVector< RealType, tnlCuda, IndexType >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   tnlAssert( false, ); // TODO: fix this
   /*RealType result = u[ 0 ] - v[ 0 ];
   const IndexType n = v. getSize();
   for( IndexType i = 1; i < n; i ++ )
      result = Min( result, ( RealType ) fabs(  u[ i ] - v[ i ] ) );
   return result;*/
}

template< typename RealType, typename IndexType >
RealType tnlDifferenceLpNorm( const tnlVector< RealType, tnlCuda, IndexType >& u,
                          const tnlVector< RealType, tnlCuda, IndexType >& v, const RealType& p )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   tnlAssert( false, ); // TODO: fix this
   /*const IndexType n = v. getSize();
   RealType result = pow( ( RealType ) fabs( u[ 0 ] - v[ 0 ] ), ( RealType ) p );
   for( IndexType i = 1; i < n; i ++ )
      result += pow( ( RealType ) fabs( u[ i ] - v[ i ] ), ( RealType ) p  );
   return pow( result, 1.0 / p );*/
}

template< typename RealType, typename IndexType >
RealType tnlDifferenceSum( const tnlVector< RealType, tnlCuda, IndexType >& u,
                       const tnlVector< RealType, tnlCuda, IndexType >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   tnlAssert( false, ); // TODO: fix this
   /*RealType result = u[ 0 ] - v[ 0 ];
   const IndexType n = u. getSize();
   for( IndexType i = 1; i < n; i ++ )
      result += u[ i ] - v[ i ];
   return result;*/
};


template< typename RealType, typename IndexType >
void tnlScalarMultiplication( const RealType& alpha,
                              tnlVector< RealType, tnlCuda, IndexType >& u )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
#ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = u. getSize() / 512 + 1;

   tnlVectorCUDAScalaMultiplicationKernel<<< gridSize, blockSize >>>( u. getSize(),
                                                                          alpha,
                                                                          u. getData() );
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
}

template< typename RealType, typename IndexType >
RealType tnlSDOT( const tnlVector< RealType, tnlCuda, IndexType >& u ,
              const tnlVector< RealType, tnlCuda, IndexType >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "You try to compute SDOT of two vectors with different sizes." << endl
                   << "The first one is " << u. getName() << " with size " << u. getSize() << endl
                   << "The second one is " << v. getName() << " with size " << v. getSize() << endl );
   RealType result( 0 );
   tnlCUDALongVectorReduction< RealType, RealType, IndexType, tnlParallelReductionSdot >( u. getSize(),
                                                                              u. getData(),
                                                                              v. getData(),
                                                                              result,
                                                                              ( RealType ) 0 );
   return result;
}

#ifdef HAVE_CUDA
template< typename RealType, typename IndexType >
__global__ void tnlVectorCUDASaxpyKernel( const IndexType size,
                                              const RealType alpha,
                                              const RealType* x,
                                              RealType* y )
{
   IndexType tid = blockDim. x * blockIdx. x + threadIdx. x;
   if( tid < size )
      y[ tid ] += alpha * x[ tid ];
}
#endif

//! Compute SAXPY operation (Scalar Alpha X Plus Y ).
template< typename RealType, typename IndexType >
void tnlSAXPY( const RealType& alpha,
               const tnlVector< RealType, tnlCuda, IndexType >& x,
               tnlVector< RealType, tnlCuda, IndexType >& y )
{
   tnlAssert( x. getSize() != 0,
              cerr << "Vector name is " << x. getName() );
   tnlAssert( x. getSize() == y. getSize(),
              cerr << "You try to compute SAXPY for two vectors with different sizes." << endl
                   << "The first one is " << x. getName() << " with size " << x. getSize() << endl
                   << "The second one is " << y. getName() << " with size " << y. getSize() << endl );
#ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = x. getSize() / 512 + 1;

   tnlVectorCUDASaxpyKernel<<< gridSize, blockSize >>>( x. getSize(),
                                                            alpha,
                                                            x. getData(),
                                                            y. getData() );
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif

}

#ifdef HAVE_CUDA
template< typename RealType, typename IndexType >
__global__ void tnlVectorCUDASaxmyKernel( const IndexType size,
                                              const RealType alpha,
                                              const RealType* x,
                                              RealType* y )
{
   IndexType tid = blockDim. x * blockIdx. x + threadIdx. x;
   if( tid < size )
      y[ tid ] = alpha * x[ tid ] - y[ tid ];
}
#endif

template< typename RealType, typename IndexType >
void tnlSAXMY( const RealType& alpha,
               const tnlVector< RealType, tnlCuda, IndexType >& x,
               tnlVector< RealType, tnlCuda, IndexType >& y )
{
   tnlAssert( x. getSize() != 0,
              cerr << "Vector name is " << x. getName() );
   tnlAssert( x. getSize() == y. getSize(),
              cerr << "You try to compute SAXPY for two vectors with different sizes." << endl
                   << "The first one is " << x. getName() << " with size " << x. getSize() << endl
                   << "The second one is " << y. getName() << " with size " << y. getSize() << endl );
#ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = x. getSize() / 512 + 1;

   tnlVectorCUDASaxmyKernel<<< gridSize, blockSize >>>( x. getSize(),
                                                            alpha,
                                                            x. getData(),
                                                            y. getData() );
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif

}



#endif /* TNLLONGVECTORCUDA_H_ */
