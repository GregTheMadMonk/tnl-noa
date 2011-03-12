/***************************************************************************
                          tnlCudaSupport.cpp  -  description
                             -------------------
    begin                : Jul 8, 2010
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

#include <iostream>
#include <tnlCudaSupport.h>

using namespace std;
/*
bool checkCUDAError( const char* file_name, int line )
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
*/
