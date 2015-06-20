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

#include <core/tnlCuda.h>

class tnlCudaReductionBuffer
{
   public:
      inline tnlCudaReductionBuffer( size_t size = 0 ): data( 0 ), size( 0 )
      {
         if( size != 0 ) setSize( size );
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
      
      inline ~tnlCudaReductionBuffer()
      {
#ifdef HAVE_CUDA         
         if( data ) cudaFree( data );
#endif         
      }
      
   protected:
      
      void* data;
      
      size_t size;
};


#endif	/* TNLCUDAREDUCTIONBUFFER_H */

