/***************************************************************************
                          Cuda.cu  -  description
                             -------------------
    begin                : Dec 22, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Devices/Cuda.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Devices {

/*void Cuda::configSetup( tnlConfigDescription& config, const String& prefix )
{
#ifdef HAVE_CUDA
   config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device.", 0 );
#else
   config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device (CUDA is not supported on this system).", 0 );
#endif
}
 
bool Cuda::setup( const tnlParameterContainer& parameters,
                    const String& prefix )
{
   int cudaDevice = parameters.getParameter< int >( prefix + "cuda-device" );
#ifdef HAVE_CUDA
    cudaSetDevice( cudaDevice );
    checkCudaDevice;
#endif
   return true;
}
*/

bool Cuda::checkDevice( const char* file_name, int line )
{
   cudaError error = cudaGetLastError();
   if( error == cudaSuccess )
      return true;
   std::cerr << "CUDA ERROR(" << error << ") at line " << line << " in " << file_name << ":" << std::endl;
   switch( error )
   {
      // 1
      case cudaErrorMissingConfiguration:
         std::cerr
          << "The device function being invoked (usually via ::cudaLaunch()) was not " << std::endl
          << "previously configured via the ::cudaConfigureCall() function. " << std::endl;
       break;

      // 2
      case cudaErrorMemoryAllocation:
         std::cerr
          << "The API call failed because it was unable to allocate enough memory to " << std::endl
          << "perform the requested operation. " << std::endl;
       break;

      // 3
      case cudaErrorInitializationError:
         std::cerr
          << "The API call failed because the CUDA driver and runtime could not be " << std::endl
          << "initialized. " << std::endl;
       break;
 
      // 4
      case cudaErrorLaunchFailure:
         std::cerr
          << "An exception occurred on the device while executing a kernel. Common " << std::endl
          << "causes include dereferencing an invalid device pointer and accessing " << std::endl
          << "out of bounds shared memory. The device cannot be used until " << std::endl
          << "::cudaThreadExit() is called. All existing device memory allocations " << std::endl
          << "are invalid and must be reconstructed if the program is to continue " << std::endl
          << "using CUDA. " << std::endl;
       break;

      // 5
      case cudaErrorPriorLaunchFailure:
         std::cerr
          << "This indicated that a previous kernel launch failed. This was previously " << std::endl
          << "used for device emulation of kernel launches. " << std::endl
          << "This error return is deprecated as of CUDA 3.1. Device emulation mode was " << std::endl
          << "removed with the CUDA 3.1 release. " << std::endl;
       break;

      // 6
      case cudaErrorLaunchTimeout:
         std::cerr
          << "This indicates that the device kernel took too long to execute. This can " << std::endl
          << "only occur if timeouts are enabled - see the device property " << std::endl
          << "ref ::cudaDeviceProp::kernelExecTimeoutEnabled \"kernelExecTimeoutEnabled\" " << std::endl
          << "for more information. The device cannot be used until ::cudaThreadExit() " << std::endl
          << "is called. All existing device memory allocations are invalid and must be " << std::endl
          << "reconstructed if the program is to continue using CUDA. " << std::endl;
       break;

      // 7
      case cudaErrorLaunchOutOfResources:
         std::cerr
          << "This indicates that a launch did not occur because it did not have " << std::endl
          << "appropriate resources. Although this error is similar to " << std::endl
          << "::cudaErrorInvalidConfiguration, this error usually indicates that the " << std::endl
          << "user has attempted to pass too many arguments to the device kernel, or the " << std::endl
          << "kernel launch specifies too many threads for the kernel's register count. " << std::endl;
       break;

      // 8
      case cudaErrorInvalidDeviceFunction:
         std::cerr
          << "The requested device function does not exist or is not compiled for the " << std::endl
          << "proper device architecture. " << std::endl;
       break;
 
      // 9
      case cudaErrorInvalidConfiguration:
         std::cerr
          << "This indicates that a kernel launch is requesting resources that can " << std::endl
          << "never be satisfied by the current device. Requesting more shared memory " << std::endl
          << "per block than the device supports will trigger this error, as will " << std::endl
          << "requesting too many threads or blocks. See ::cudaDeviceProp for more " << std::endl
          << "device limitations. " << std::endl;
       break;

      // 10
      case cudaErrorInvalidDevice:
         std::cerr
          << "This indicates that the device ordinal supplied by the user does not " << std::endl
          << "correspond to a valid CUDA device. " << std::endl;
       break;

      // 11
      case cudaErrorInvalidValue:
         std::cerr
          << "This indicates that one or more of the parameters passed to the API call " << std::endl
          << "is not within an acceptable range of values. " << std::endl;
       break;

      // 12
      case cudaErrorInvalidPitchValue:
         std::cerr
          << "This indicates that one or more of the pitch-related parameters passed " << std::endl
          << "to the API call is not within the acceptable range for pitch. " << std::endl;
       break;

      // 13
      case cudaErrorInvalidSymbol:
         std::cerr
          << "This indicates that the symbol name/identifier passed to the API call " << std::endl
          << "is not a valid name or identifier. " << std::endl;
       break;

      // 14
      case cudaErrorMapBufferObjectFailed:
      std::cerr
       << "This indicates that the buffer object could not be mapped. " << std::endl;
       break;

      // 15
      case cudaErrorUnmapBufferObjectFailed:
         std::cerr
          << "This indicates that the buffer object could not be unmapped. " << std::endl;
       break;

      // 16
      case cudaErrorInvalidHostPointer:
         std::cerr
          << "This indicates that at least one host pointer passed to the API call is " << std::endl
          << "not a valid host pointer. " << std::endl;
       break;

      // 17
      case cudaErrorInvalidDevicePointer:
         std::cerr
          << "This indicates that at least one device pointer passed to the API call is " << std::endl
          << "not a valid device pointer. " << std::endl;
       break;

      case cudaErrorInvalidTexture:
         std::cerr
          << "This indicates that the texture passed to the API call is not a valid " << std::endl
          << "texture. " << std::endl;
       break;

      case cudaErrorInvalidTextureBinding:
         std::cerr
          << "This indicates that the texture binding is not valid. This occurs if you " << std::endl
          << "call ::cudaGetTextureAlignmentOffset() with an unbound texture. " << std::endl;
       break;

      case cudaErrorInvalidChannelDescriptor:
         std::cerr
          << "This indicates that the channel descriptor passed to the API call is not " << std::endl
          << "valid. This occurs if the format is not one of the formats specified by " << std::endl
          << "::cudaChannelFormatKind, or if one of the dimensions is invalid. " << std::endl;
       break;

      case cudaErrorInvalidMemcpyDirection:
         std::cerr
          << "This indicates that the direction of the memcpy passed to the API call is " << std::endl
          << "not one of the types specified by ::cudaMemcpyKind. " << std::endl;
       break;

      case cudaErrorAddressOfConstant:
         std::cerr
          << "This indicated that the user has taken the address of a constant variable, " << std::endl
          << "which was forbidden up until the CUDA 3.1 release. " << std::endl
          << "This error return is deprecated as of CUDA 3.1. Variables in constant " << std::endl
          << "memory may now have their address taken by the runtime via " << std::endl
          << "::cudaGetSymbolAddress(). " << std::endl;
       break;

      case cudaErrorTextureFetchFailed:
         std::cerr
          << "This indicated that a texture fetch was not able to be performed. " << std::endl
          << "This was previously used for device emulation of texture operations. " << std::endl
          << "This error return is deprecated as of CUDA 3.1. Device emulation mode was " << std::endl
          << "removed with the CUDA 3.1 release. " << std::endl;
       break;

      case cudaErrorTextureNotBound:
         std::cerr
          << "This indicated that a texture was not bound for access. " << std::endl
          << "This was previously used for device emulation of texture operations. " << std::endl
          << "This error return is deprecated as of CUDA 3.1. Device emulation mode was " << std::endl
          << "removed with the CUDA 3.1 release. " << std::endl;
       break;

      case cudaErrorSynchronizationError:
         std::cerr
          << "This indicated that a synchronization operation had failed. " << std::endl
          << "This was previously used for some device emulation functions. " << std::endl
          << "This error return is deprecated as of CUDA 3.1. Device emulation mode was " << std::endl
          << "removed with the CUDA 3.1 release. " << std::endl;
       break;

      case cudaErrorInvalidFilterSetting:
         std::cerr
          << "This indicates that a non-float texture was being accessed with linear " << std::endl
          << "filtering. This is not supported by CUDA. " << std::endl;
       break;

      case cudaErrorInvalidNormSetting:
         std::cerr
          << "This indicates that an attempt was made to read a non-float texture as a " << std::endl
          << "normalized float. This is not supported by CUDA. " << std::endl;
       break;

      case cudaErrorMixedDeviceExecution:
         std::cerr
          << "Mixing of device and device emulation code was not allowed. " << std::endl
          << "This error return is deprecated as of CUDA 3.1. Device emulation mode was " << std::endl
          << "removed with the CUDA 3.1 release. " << std::endl;
       break;

      case cudaErrorCudartUnloading:
         std::cerr
          << "This indicated an issue with calling API functions during the unload " << std::endl
          << "process of the CUDA runtime in prior releases. " << std::endl
          << "This error return is deprecated as of CUDA 3.2. " << std::endl;
       break;

      case cudaErrorUnknown:
         std::cerr
          << "This indicates that an unknown internal error has occurred. " << std::endl;
       break;

      case cudaErrorNotYetImplemented:
         std::cerr
          << "This indicates that the API call is not yet implemented. Production " << std::endl
          << "releases of CUDA will never return this error. " << std::endl;
       break;

      case cudaErrorMemoryValueTooLarge:
         std::cerr
          << "This indicated that an emulated device pointer exceeded the 32-bit address " << std::endl
          << "range. " << std::endl
          << "This error return is deprecated as of CUDA 3.1. Device emulation mode was " << std::endl
          << "removed with the CUDA 3.1 release. " << std::endl;
       break;

      case cudaErrorInvalidResourceHandle:
         std::cerr
          << "This indicates that a resource handle passed to the API call was not " << std::endl
          << "valid. Resource handles are opaque types like ::cudaStream_t and " << std::endl
          << "::cudaEvent_t. " << std::endl;
       break;

      case cudaErrorNotReady:
         std::cerr
          << "This indicates that asynchronous operations issued previously have not " << std::endl
          << "completed yet. This result is not actually an error, but must be indicated " << std::endl
          << "differently than ::cudaSuccess (which indicates completion). Calls that " << std::endl
          << "may return this value include ::cudaEventQuery() and ::cudaStreamQuery(). " << std::endl;
       break;

      case cudaErrorInsufficientDriver:
         std::cerr
          << "This indicates that the installed NVIDIA CUDA driver is older than the " << std::endl
          << "CUDA runtime library. This is not a supported configuration. Users should " << std::endl
          << "install an updated NVIDIA display driver to allow the application to run. " << std::endl;
       break;

      case cudaErrorSetOnActiveProcess:
         std::cerr
          << "This indicates that the user has called ::cudaSetDevice(), " << std::endl
          << "::cudaSetValidDevices(), ::cudaSetDeviceFlags(), " << std::endl
          << "::cudaD3D9SetDirect3DDevice(), ::cudaD3D10SetDirect3DDevice, " << std::endl
          << "::cudaD3D11SetDirect3DDevice(), * or ::cudaVDPAUSetVDPAUDevice() after " << std::endl
          << "initializing the CUDA runtime by calling non-device management operations " << std::endl
          << "(allocating memory and launching kernels are examples of non-device " << std::endl
          << "management operations). This error can also be returned if using " << std::endl
          << "runtime/driver interoperability and there is an existing ::CUcontext " << std::endl
          << "active on the host thread. " << std::endl;
       break;

      case cudaErrorInvalidSurface:
         std::cerr
          << "This indicates that the surface passed to the API call is not a valid " << std::endl
          << "surface. " << std::endl;
       break;

      case cudaErrorNoDevice:
      std::cerr
       << "This indicates that no CUDA-capable devices were detected by the installed " << std::endl
       << "CUDA driver. " << std::endl;
       break;

      case cudaErrorECCUncorrectable:
      std::cerr
       << "This indicates that an uncorrectable ECC error was detected during " << std::endl
       << "execution. " << std::endl;
       break;

      case cudaErrorSharedObjectSymbolNotFound:
      std::cerr
       << "This indicates that a link to a shared object failed to resolve. " << std::endl;
       break;

      case cudaErrorSharedObjectInitFailed:
      std::cerr
       << "This indicates that initialization of a shared object failed. " << std::endl;
       break;

      case cudaErrorUnsupportedLimit:
      std::cerr
       << "This indicates that the ::cudaLimit passed to the API call is not " << std::endl
       << "supported by the active device. " << std::endl;
       break;

      case cudaErrorDuplicateVariableName:
      std::cerr
       << "This indicates that multiple global or constant variables (across separate " << std::endl
       << "CUDA source files in the application) share the same string name. " << std::endl;
       break;

      case cudaErrorDuplicateTextureName:
      std::cerr
       << "This indicates that multiple textures (across separate CUDA source " << std::endl
       << "files in the application) share the same string name. " << std::endl;
       break;

      case cudaErrorDuplicateSurfaceName:
      std::cerr
       << "This indicates that multiple surfaces (across separate CUDA source " << std::endl
       << "files in the application) share the same string name. " << std::endl;
       break;

      case cudaErrorDevicesUnavailable:
      std::cerr
       << "This indicates that all CUDA devices are busy or unavailable at the current " << std::endl
       << "time. Devices are often busy/unavailable due to use of " << std::endl
       << "::cudaComputeModeExclusive or ::cudaComputeModeProhibited. They can also " << std::endl
       << "be unavailable due to memory constraints on a device that already has " << std::endl
       << "active CUDA work being performed. " << std::endl;
       break;

      case cudaErrorInvalidKernelImage:
      std::cerr
       << "This indicates that the device kernel image is invalid. " << std::endl;
       break;

      case cudaErrorNoKernelImageForDevice:
      std::cerr
       << "This indicates that there is no kernel image available that is suitable " << std::endl
       << "for the device. This can occur when a user specifies code generation " << std::endl
       << "options for a particular CUDA source file that do not include the " << std::endl
       << "corresponding device configuration. " << std::endl;
       break;

      case cudaErrorIncompatibleDriverContext:
      std::cerr
       << "This indicates that the current context is not compatible with this " << std::endl
       << "version of the CUDA Runtime. This can only occur if you are using CUDA " << std::endl
       << "Runtime/Driver interoperability and have created an existing Driver " << std::endl
       << "context using an older API. Please see \ref CUDART_DRIVER " << std::endl
       << "\"Interactions with the CUDA Driver API\" for more information. " << std::endl;
       break;

      case cudaErrorStartupFailure:
      std::cerr
       << "This indicates an internal startup failure in the CUDA runtime. " << std::endl;
       break;

      case cudaErrorApiFailureBase:
      std::cerr
       << "Any unhandled CUDA driver error is added to this value and returned via " << std::endl
       << "the runtime. Production releases of CUDA should not return such errors. " << std::endl;
       break;

   }
   //throw EXIT_FAILURE;
   return false;
}

} // namespace Devices
} // namespace TNL
