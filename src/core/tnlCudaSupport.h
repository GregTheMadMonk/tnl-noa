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

#include <iostream>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

using namespace std;

#define CHECK_CUDA_ERROR checkCUDAError( __FILE__, __LINE__ )

inline bool checkCUDAError( const char* file_name, int line )
{
#ifdef HAVE_CUDA
   cudaError error = cudaGetLastError();
   if( error == cudaSuccess )
      return true;
   cerr << "CUDA ERROR: ";
   switch( error )
   {
      case cudaErrorMissingConfiguration:      cerr << "Missing configuration error"; break;
      case cudaErrorMemoryAllocation:          cerr << "Memory allocation error"; break;
      case cudaErrorInitializationError:       cerr << "Initialization error"; break;
      case cudaErrorLaunchFailure:             cerr << "Launch failure"; break;
      case cudaErrorPriorLaunchFailure:        cerr << "Prior launch failure"; break;
      case cudaErrorLaunchTimeout:             cerr << "Launch timeout error"; break;
      case cudaErrorLaunchOutOfResources:      cerr << "Launch out of resources error"; break;
      case cudaErrorInvalidDeviceFunction:     cerr << "Invalid device function"; break;
      case cudaErrorInvalidConfiguration:      cerr << "Invalid configuration"; break;
      case cudaErrorInvalidDevice:             cerr << "Invalid device"; break;
      case cudaErrorInvalidValue:              cerr << "Invalid value"; break;
      case cudaErrorInvalidPitchValue:         cerr << "Invalid pitch value"; break;
      case cudaErrorInvalidSymbol:             cerr << "Invalid symbol"; break;
      case cudaErrorMapBufferObjectFailed:     cerr << "Map buffer object failed"; break;
      case cudaErrorUnmapBufferObjectFailed:   cerr << "Unmap buffer object failed"; break;
      case cudaErrorInvalidHostPointer:        cerr << "Invalid host pointer"; break;
      case cudaErrorInvalidDevicePointer:      cerr << "Invalid device pointer"; break;
      case cudaErrorInvalidTexture:            cerr << "Invalid texture"; break;
      case cudaErrorInvalidTextureBinding:     cerr << "Invalid texture binding"; break;
      case cudaErrorInvalidChannelDescriptor:  cerr << "Invalid channel descriptor"; break;
      case cudaErrorInvalidMemcpyDirection:    cerr << "Invalid memcpy direction"; break;
      case cudaErrorAddressOfConstant:         cerr << "Address of constant error"; break;
      case cudaErrorTextureFetchFailed:        cerr << "Texture fetch failed"; break;
      case cudaErrorTextureNotBound:           cerr << "Texture not bound error"; break;
      case cudaErrorSynchronizationError:      cerr << "Synchronization error"; break;
      case cudaErrorInvalidFilterSetting:      cerr << "Invalid filter setting"; break;
      case cudaErrorInvalidNormSetting:        cerr << "Invalid norm setting"; break;
      case cudaErrorMixedDeviceExecution:      cerr << "Mixed device execution"; break;
      case cudaErrorCudartUnloading:           cerr << "CUDA runtime unloading"; break;
      case cudaErrorUnknown:                   cerr << "Unknown error condition"; break;
      case cudaErrorNotYetImplemented:         cerr << "Function not yet implemented"; break;
      case cudaErrorMemoryValueTooLarge:       cerr << "Memory value too large"; break;
      case cudaErrorInvalidResourceHandle:     cerr << "Invalid resource handle"; break;
      case cudaErrorNotReady:                  cerr << "Not ready error"; break;
      case cudaErrorInsufficientDriver:        cerr << "CUDA runtime is newer than driver"; break;
      case cudaErrorSetOnActiveProcess:        cerr << "Set on active process error"; break;
      case cudaErrorNoDevice:                  cerr << "No available CUDA device"; break;
      case cudaErrorStartupFailure:            cerr << "Startup failure"; break;
      case cudaErrorApiFailureBase:            cerr << "API failure base"; break;
   }
   cerr << " at line " << line << " in " << file_name << endl;
   return false;
#else
   return true;
#endif
}


//! This function allocates a variable or an array on the CUDA device.
/*! It checks for possible errors.
 */
template< typename Type, typename Index > // TODO: Index = int - does not work now with nvcc
bool tnlAllocateOnCudaDevice( Type*& pointer, Index elements = 1 )
{
#ifdef HAVE_CUDA
   if( cudaMalloc( ( void** ) & pointer,
                     sizeof( Type ) * elements ) != cudaSuccess )
   {
      cerr << "I am not able to allocate new variable(s) on the CUDA device." << endl;
      checkCUDAError( __FILE__, __LINE__ );
      return false;
   }
   return true;
#else
   return false;
#endif
};

//! This function allocates a variable or an array on the CUDA device.
/*! It checks for possible errors.
 */
template< typename Type > // TODO: remove after fixing default templaet parameter value - does not work now with nvcc
bool tnlAllocateOnCudaDevice( Type*& pointer, int elements = 1 )
{
#ifdef HAVE_CUDA
   if( cudaMalloc( ( void** ) & pointer,
                     sizeof( Type ) * elements ) != cudaSuccess )
   {
      cerr << "I am not able to allocate new variable(s) on the CUDA device." << endl;
      checkCUDAError( __FILE__, __LINE__ );
      return false;
   }
   return true;
#else
   return false;
#endif
};



//! This function can be used if you need to pass some data to the CUDA device.
/*! It allocates necessary memory on the device and then copies the data.
 */
template< typename Type >
Type* tnlPassToCudaDevice( const Type& data )
{
#ifdef HAVE_CUDA
   Type* cuda_data;
   if( cudaMalloc( ( void** ) & cuda_data, sizeof( Type ) ) != cudaSuccess )
   {
      cerr << "Unable to allocate CUDA device memory to pass a data structure there." << endl;
      checkCUDAError( __FILE__, __LINE__ );
      return 0;
   }
   if( cudaMemcpy( cuda_data, &data, sizeof( Type ), cudaMemcpyHostToDevice ) != cudaSuccess )
   {
      cerr << "Unable to pass data structure to CUDA device." << endl;
      checkCUDAError( __FILE__, __LINE__ );
      return 0;
   }
   return cuda_data;
#else
   return 0;
#endif
}

//! Reads data from the CUDA device.
template< typename Type >
Type tnlGetFromCudaDevice( const Type* device_data )
{
#ifdef HAVE_CUDA
   Type host_data;
   if( cudaMemcpy( &host_data,
                   &device_data,
                   sizeof( Type ),
                   cudaMemcpyHostToDevice ) != cudaSuccess )
   {
      cerr << "Unable to get data from the CUDA device." << endl;
      checkCUDAError( __FILE__, __LINE__ );
      return host_data;
   }
   return host_data;
#else
   return 0;
#endif
}

//! Reads data from the CUDA device.
/*! Moreover it frees memory on the CUDA device.
 */
template< typename Type >
Type tnlPassFromCudaDevice( const Type* device_data )
{
#ifdef HAVE_CUDA
   Type host_data;
   if( cudaMemcpy( &host_data,
                   &device_data,
                   sizeof( Type ),
                   cudaMemcpyHostToDevice ) != cudaSuccess )
   {
      cerr << "Unable to get data from the CUDA device." << endl;
      checkCUDAError( __FILE__, __LINE__ );
      return host_data;
   }
   if( ! cudaFree( device_data, sizeof( Type ) ) )
   {
      cerr << "Unable to free memory on the CUDA device." << endl;
      checkCUDAError( __FILE__, __LINE__ );
      return host_data;
   }
   return host_data;
#else
   return 0;
#endif
}


#endif /* TNLCUDASUPPORT_H_ */
