/***************************************************************************
                          tnlCudaReductionBuffer.h  -  description
                             -------------------
    begin                : June 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef TNLCUDAREDUCTIONBUFFER_H
#define	TNLCUDAREDUCTIONBUFFER_H

#include <stdlib.h>

#include <core/tnlCuda.h>

class tnlCudaReductionBuffer
{
   public:
      inline static tnlCudaReductionBuffer& getInstance( size_t size = 0 )
      {
         static tnlCudaReductionBuffer instance( size );
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
               cerr << "I am not able to allocate reduction buffer on the GPU." << endl;
               this->data = 0;
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
      tnlCudaReductionBuffer( tnlCudaReductionBuffer const& copy );            // Not Implemented
      tnlCudaReductionBuffer& operator=( tnlCudaReductionBuffer const& copy ); // Not Implemented

      // private constructor of the singleton
      inline tnlCudaReductionBuffer( size_t size = 0 ): data( 0 ), size( 0 )
      {
#ifdef HAVE_CUDA
         if( size != 0 ) setSize( size );
         atexit( tnlCudaReductionBuffer::free_atexit );
#endif
      }

      inline static void free_atexit( void )
      {
         tnlCudaReductionBuffer::getInstance().free();
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


#endif	/* TNLCUDAREDUCTIONBUFFER_H */

