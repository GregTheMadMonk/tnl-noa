/***************************************************************************
                          tnlCudaSupport.h  -  description
                             -------------------
    begin                : Feb 23, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLCUDASUPPORT_H_
#define TNLCUDASUPPORT_H_

template< class T >
T* passToCudaDevice( const T& data )
{
#ifdef HAVE_CUDA
   T* cuda_data;
   if( cudaMalloc( ( void** ) & cuda_data, sizeof( T ) ) != cudaSuccess )
   {
      cerr << "Unable to allocate CUDA device memory to pass a data structure there." << endl;
      return 0;
   }
   if( cudaMemcpy( cuda_data, &data, sizeof( T ), cudaMemcpyHostToDevice ) != cudaSuccess )
   {
      cerr << "Unable to pass data structure to CUDA device." << endl;
      return 0;
   }
   return cuda_data;
#else
   return 0;
#endif
}

#endif /* TNLCUDASUPPORT_H_ */
