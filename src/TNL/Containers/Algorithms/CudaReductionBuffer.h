/***************************************************************************
                          CudaReductionBuffer.h  -  description
                             -------------------
    begin                : June 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <stdlib.h>

#include <TNL/Devices/Cuda.h>
#include <TNL/Exceptions/CudaBadAlloc.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

class CudaReductionBuffer
{
   public:
      inline static CudaReductionBuffer& getInstance( size_t size = 0 )
      {
         static CudaReductionBuffer instance( size );
         return instance;
      }

      inline bool setSize( size_t size )
      {
#ifdef HAVE_CUDA
         if( size > this->size )
         {
            if( data ) cudaFree( data );
            this->size = size;
            if( cudaMalloc( ( void** ) &this->data, size ) != cudaSuccess )
            {
               this->data = 0;
               throw Exceptions::CudaBadAlloc();
            }
            return checkCudaDevice;
         }
         else
            return true;
#else
         return false;
#endif
      }

      template< typename Type >
      Type* getData() { return ( Type* ) this->data; }

   private:
      // stop the compiler generating methods of copy the object
      CudaReductionBuffer( CudaReductionBuffer const& copy );            // Not Implemented
      CudaReductionBuffer& operator=( CudaReductionBuffer const& copy ); // Not Implemented

      // private constructor of the singleton
      inline CudaReductionBuffer( size_t size = 0 ): data( 0 ), size( 0 )
      {
#ifdef HAVE_CUDA
         if( size != 0 ) setSize( size );
         atexit( CudaReductionBuffer::free_atexit );
#endif
      }

      inline static void free_atexit( void )
      {
         CudaReductionBuffer::getInstance().free();
      }

   protected:
      inline void free( void )
      {
#ifdef HAVE_CUDA
         if( data )
         {
            cudaFree( data );
            data = 0;
         }
#endif
      }

      void* data;

      size_t size;
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

