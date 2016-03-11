/***************************************************************************
                          tnlCuda.cu  -  description
                             -------------------
    begin                : Dec 22, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#include <core/tnlCuda.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>


/*void tnlCuda::configSetup( tnlConfigDescription& config, const tnlString& prefix )
{
#ifdef HAVE_CUDA
   config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device.", 0 );
#else
   config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device (CUDA is not supported on this system).", 0 );   
#endif   
}
      
bool tnlCuda::setup( const tnlParameterContainer& parameters,
                    const tnlString& prefix )
{
   int cudaDevice = parameters.getParameter< int >( prefix + "cuda-device" );
#ifdef HAVE_CUDA
    cudaSetDevice( cudaDevice );
    checkCudaDevice;
#endif
   return true;
}
*/

bool tnlCuda::checkDevice( const char* file_name, int line )
{
   cudaError error = cudaGetLastError();
   if( error == cudaSuccess )
      return true;
   cerr << "CUDA ERROR(" << error << ") at line " << line << " in " << file_name << ":" << endl;
   switch( error )
   {
      // 1
      case cudaErrorMissingConfiguration:
         cerr
          << "The device function being invoked (usually via ::cudaLaunch()) was not " << endl
          << "previously configured via the ::cudaConfigureCall() function. " << endl;
       break;

      // 2
      case cudaErrorMemoryAllocation:   
         cerr
          << "The API call failed because it was unable to allocate enough memory to " << endl
          << "perform the requested operation. " << endl;
       break;

      // 3
      case cudaErrorInitializationError:
         cerr
          << "The API call failed because the CUDA driver and runtime could not be " << endl
          << "initialized. " << endl;
       break;
   
      // 4
      case cudaErrorLaunchFailure:
         cerr
          << "An exception occurred on the device while executing a kernel. Common " << endl
          << "causes include dereferencing an invalid device pointer and accessing " << endl
          << "out of bounds shared memory. The device cannot be used until " << endl
          << "::cudaThreadExit() is called. All existing device memory allocations " << endl
          << "are invalid and must be reconstructed if the program is to continue " << endl
          << "using CUDA. " << endl;
       break;

      // 5
      case cudaErrorPriorLaunchFailure:
         cerr
          << "This indicated that a previous kernel launch failed. This was previously " << endl
          << "used for device emulation of kernel launches. " << endl
          << "This error return is deprecated as of CUDA 3.1. Device emulation mode was " << endl
          << "removed with the CUDA 3.1 release. " << endl;
       break;

      // 6
      case cudaErrorLaunchTimeout:
         cerr
          << "This indicates that the device kernel took too long to execute. This can " << endl
          << "only occur if timeouts are enabled - see the device property " << endl
          << "ref ::cudaDeviceProp::kernelExecTimeoutEnabled \"kernelExecTimeoutEnabled\" " << endl
          << "for more information. The device cannot be used until ::cudaThreadExit() " << endl
          << "is called. All existing device memory allocations are invalid and must be " << endl
          << "reconstructed if the program is to continue using CUDA. " << endl;
       break;

      // 7
      case cudaErrorLaunchOutOfResources:
         cerr
          << "This indicates that a launch did not occur because it did not have " << endl
          << "appropriate resources. Although this error is similar to " << endl
          << "::cudaErrorInvalidConfiguration, this error usually indicates that the " << endl
          << "user has attempted to pass too many arguments to the device kernel, or the " << endl
          << "kernel launch specifies too many threads for the kernel's register count. " << endl;
       break;

      // 8
      case cudaErrorInvalidDeviceFunction:
         cerr
          << "The requested device function does not exist or is not compiled for the " << endl
          << "proper device architecture. " << endl;
       break;
 
      // 9 
      case cudaErrorInvalidConfiguration:
         cerr
          << "This indicates that a kernel launch is requesting resources that can " << endl
          << "never be satisfied by the current device. Requesting more shared memory " << endl
          << "per block than the device supports will trigger this error, as will " << endl
          << "requesting too many threads or blocks. See ::cudaDeviceProp for more " << endl
          << "device limitations. " << endl;
       break;

      // 10 
      case cudaErrorInvalidDevice:
         cerr
          << "This indicates that the device ordinal supplied by the user does not " << endl
          << "correspond to a valid CUDA device. " << endl;
       break;

      // 11
      case cudaErrorInvalidValue:
         cerr
          << "This indicates that one or more of the parameters passed to the API call " << endl
          << "is not within an acceptable range of values. " << endl;
       break;

      // 12
      case cudaErrorInvalidPitchValue:
         cerr
          << "This indicates that one or more of the pitch-related parameters passed " << endl
          << "to the API call is not within the acceptable range for pitch. " << endl;
       break;

      // 13
      case cudaErrorInvalidSymbol:
         cerr
          << "This indicates that the symbol name/identifier passed to the API call " << endl
          << "is not a valid name or identifier. " << endl;
       break;

      // 14
      case cudaErrorMapBufferObjectFailed:
      cerr
       << "This indicates that the buffer object could not be mapped. " << endl;
       break;

      // 15
      case cudaErrorUnmapBufferObjectFailed:
         cerr
          << "This indicates that the buffer object could not be unmapped. " << endl;
       break;

      // 16
      case cudaErrorInvalidHostPointer:
         cerr
          << "This indicates that at least one host pointer passed to the API call is " << endl
          << "not a valid host pointer. " << endl;
       break;

      // 17
      case cudaErrorInvalidDevicePointer:
         cerr
          << "This indicates that at least one device pointer passed to the API call is " << endl
          << "not a valid device pointer. " << endl;
       break;

      case cudaErrorInvalidTexture:
         cerr
          << "This indicates that the texture passed to the API call is not a valid " << endl
          << "texture. " << endl;
       break;

      case cudaErrorInvalidTextureBinding:
         cerr
          << "This indicates that the texture binding is not valid. This occurs if you " << endl
          << "call ::cudaGetTextureAlignmentOffset() with an unbound texture. " << endl;
       break;

      case cudaErrorInvalidChannelDescriptor:
         cerr
          << "This indicates that the channel descriptor passed to the API call is not " << endl
          << "valid. This occurs if the format is not one of the formats specified by " << endl
          << "::cudaChannelFormatKind, or if one of the dimensions is invalid. " << endl;
       break;

      case cudaErrorInvalidMemcpyDirection:
         cerr
          << "This indicates that the direction of the memcpy passed to the API call is " << endl
          << "not one of the types specified by ::cudaMemcpyKind. " << endl;
       break;

      case cudaErrorAddressOfConstant:
         cerr
          << "This indicated that the user has taken the address of a constant variable, " << endl
          << "which was forbidden up until the CUDA 3.1 release. " << endl
          << "This error return is deprecated as of CUDA 3.1. Variables in constant " << endl
          << "memory may now have their address taken by the runtime via " << endl
          << "::cudaGetSymbolAddress(). " << endl;
       break;

      case cudaErrorTextureFetchFailed:
         cerr
          << "This indicated that a texture fetch was not able to be performed. " << endl
          << "This was previously used for device emulation of texture operations. " << endl
          << "This error return is deprecated as of CUDA 3.1. Device emulation mode was " << endl
          << "removed with the CUDA 3.1 release. " << endl;
       break;

      case cudaErrorTextureNotBound:
         cerr
          << "This indicated that a texture was not bound for access. " << endl
          << "This was previously used for device emulation of texture operations. " << endl
          << "This error return is deprecated as of CUDA 3.1. Device emulation mode was " << endl
          << "removed with the CUDA 3.1 release. " << endl;
       break;

      case cudaErrorSynchronizationError:
         cerr
          << "This indicated that a synchronization operation had failed. " << endl
          << "This was previously used for some device emulation functions. " << endl
          << "This error return is deprecated as of CUDA 3.1. Device emulation mode was " << endl
          << "removed with the CUDA 3.1 release. " << endl;
       break;

      case cudaErrorInvalidFilterSetting:
         cerr
          << "This indicates that a non-float texture was being accessed with linear " << endl
          << "filtering. This is not supported by CUDA. " << endl;
       break;

      case cudaErrorInvalidNormSetting:
         cerr
          << "This indicates that an attempt was made to read a non-float texture as a " << endl
          << "normalized float. This is not supported by CUDA. " << endl;
       break;

      case cudaErrorMixedDeviceExecution:
         cerr
          << "Mixing of device and device emulation code was not allowed. " << endl
          << "This error return is deprecated as of CUDA 3.1. Device emulation mode was " << endl
          << "removed with the CUDA 3.1 release. " << endl;
       break;

      case cudaErrorCudartUnloading:
         cerr
          << "This indicated an issue with calling API functions during the unload " << endl
          << "process of the CUDA runtime in prior releases. " << endl
          << "This error return is deprecated as of CUDA 3.2. " << endl;
       break;

      case cudaErrorUnknown:
         cerr
          << "This indicates that an unknown internal error has occurred. " << endl;
       break;

      case cudaErrorNotYetImplemented:
         cerr
          << "This indicates that the API call is not yet implemented. Production " << endl
          << "releases of CUDA will never return this error. " << endl;
       break;

      case cudaErrorMemoryValueTooLarge:
         cerr
          << "This indicated that an emulated device pointer exceeded the 32-bit address " << endl
          << "range. " << endl
          << "This error return is deprecated as of CUDA 3.1. Device emulation mode was " << endl
          << "removed with the CUDA 3.1 release. " << endl;
       break;

      case cudaErrorInvalidResourceHandle:
         cerr
          << "This indicates that a resource handle passed to the API call was not " << endl
          << "valid. Resource handles are opaque types like ::cudaStream_t and " << endl
          << "::cudaEvent_t. " << endl;
       break;

      case cudaErrorNotReady:
         cerr
          << "This indicates that asynchronous operations issued previously have not " << endl
          << "completed yet. This result is not actually an error, but must be indicated " << endl
          << "differently than ::cudaSuccess (which indicates completion). Calls that " << endl
          << "may return this value include ::cudaEventQuery() and ::cudaStreamQuery(). " << endl;
       break;

      case cudaErrorInsufficientDriver:
         cerr
          << "This indicates that the installed NVIDIA CUDA driver is older than the " << endl
          << "CUDA runtime library. This is not a supported configuration. Users should " << endl
          << "install an updated NVIDIA display driver to allow the application to run. " << endl;
       break;

      case cudaErrorSetOnActiveProcess:
         cerr
          << "This indicates that the user has called ::cudaSetDevice(), " << endl
          << "::cudaSetValidDevices(), ::cudaSetDeviceFlags(), " << endl
          << "::cudaD3D9SetDirect3DDevice(), ::cudaD3D10SetDirect3DDevice, " << endl
          << "::cudaD3D11SetDirect3DDevice(), * or ::cudaVDPAUSetVDPAUDevice() after " << endl
          << "initializing the CUDA runtime by calling non-device management operations " << endl
          << "(allocating memory and launching kernels are examples of non-device " << endl
          << "management operations). This error can also be returned if using " << endl
          << "runtime/driver interoperability and there is an existing ::CUcontext " << endl
          << "active on the host thread. " << endl;
       break;

      case cudaErrorInvalidSurface:
         cerr
          << "This indicates that the surface passed to the API call is not a valid " << endl
          << "surface. " << endl;
       break;

      case cudaErrorNoDevice:
      cerr
       << "This indicates that no CUDA-capable devices were detected by the installed " << endl
       << "CUDA driver. " << endl;
       break;

      case cudaErrorECCUncorrectable:
      cerr
       << "This indicates that an uncorrectable ECC error was detected during " << endl
       << "execution. " << endl;
       break;

      case cudaErrorSharedObjectSymbolNotFound:
      cerr
       << "This indicates that a link to a shared object failed to resolve. " << endl;
       break;

      case cudaErrorSharedObjectInitFailed:
      cerr
       << "This indicates that initialization of a shared object failed. " << endl;
       break;

      case cudaErrorUnsupportedLimit:
      cerr
       << "This indicates that the ::cudaLimit passed to the API call is not " << endl
       << "supported by the active device. " << endl;
       break;

      case cudaErrorDuplicateVariableName:
      cerr
       << "This indicates that multiple global or constant variables (across separate " << endl
       << "CUDA source files in the application) share the same string name. " << endl;
       break;

      case cudaErrorDuplicateTextureName:
      cerr
       << "This indicates that multiple textures (across separate CUDA source " << endl
       << "files in the application) share the same string name. " << endl;
       break;

      case cudaErrorDuplicateSurfaceName:
      cerr
       << "This indicates that multiple surfaces (across separate CUDA source " << endl
       << "files in the application) share the same string name. " << endl;
       break;

      case cudaErrorDevicesUnavailable:
      cerr
       << "This indicates that all CUDA devices are busy or unavailable at the current " << endl
       << "time. Devices are often busy/unavailable due to use of " << endl
       << "::cudaComputeModeExclusive or ::cudaComputeModeProhibited. They can also " << endl
       << "be unavailable due to memory constraints on a device that already has " << endl
       << "active CUDA work being performed. " << endl;
       break;

      case cudaErrorInvalidKernelImage:
      cerr
       << "This indicates that the device kernel image is invalid. " << endl;
       break;

      case cudaErrorNoKernelImageForDevice:
      cerr
       << "This indicates that there is no kernel image available that is suitable " << endl
       << "for the device. This can occur when a user specifies code generation " << endl
       << "options for a particular CUDA source file that do not include the " << endl
       << "corresponding device configuration. " << endl;
       break;

      case cudaErrorIncompatibleDriverContext:
      cerr
       << "This indicates that the current context is not compatible with this " << endl
       << "version of the CUDA Runtime. This can only occur if you are using CUDA " << endl
       << "Runtime/Driver interoperability and have created an existing Driver " << endl
       << "context using an older API. Please see \ref CUDART_DRIVER " << endl
       << "\"Interactions with the CUDA Driver API\" for more information. " << endl;
       break;

      case cudaErrorStartupFailure:
      cerr
       << "This indicates an internal startup failure in the CUDA runtime. " << endl;
       break;

      case cudaErrorApiFailureBase:
      cerr
       << "Any unhandled CUDA driver error is added to this value and returned via " << endl
       << "the runtime. Production releases of CUDA should not return such errors. " << endl;
       break;

   }
   //throw EXIT_FAILURE;
   return false;
}
