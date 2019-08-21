/***************************************************************************
                          Cuda_impl.h  -  description
                             -------------------
    begin                : Jan 21, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Math.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Cuda/DeviceInfo.h>
#include <TNL/Exceptions/CudaBadAlloc.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Exceptions/CudaRuntimeError.h>

namespace TNL {
namespace Devices {

inline void
Cuda::configSetup( Config::ConfigDescription& config,
                   const String& prefix )
{
#ifdef HAVE_CUDA
   config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device to run the computation.", 0 );
#else
   config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device to run the computation (not supported on this system).", 0 );
#endif
}

inline bool
Cuda::setup( const Config::ParameterContainer& parameters,
             const String& prefix )
{
#ifdef HAVE_CUDA
   int cudaDevice = parameters.getParameter< int >( prefix + "cuda-device" );
   if( cudaSetDevice( cudaDevice ) != cudaSuccess )
   {
      std::cerr << "I cannot activate CUDA device number " << cudaDevice << "." << std::endl;
      return false;
   }
   getSmartPointersSynchronizationTimer().reset();
   getSmartPointersSynchronizationTimer().stop();
#endif
   return true;
}

inline constexpr int Cuda::getGPUTransferBufferSize()
{
   return 1 << 20;
}

inline void Cuda::insertSmartPointer( Pointers::SmartPointer* pointer )
{
   getSmartPointersRegister().insert( pointer, TNL::Cuda::DeviceInfo::getActiveDevice() );
}

inline void Cuda::removeSmartPointer( Pointers::SmartPointer* pointer )
{
   getSmartPointersRegister().remove( pointer, TNL::Cuda::DeviceInfo::getActiveDevice() );
}

inline bool Cuda::synchronizeDevice( int deviceId )
{
#ifdef HAVE_CUDA
   if( deviceId < 0 )
      deviceId = TNL::Cuda::DeviceInfo::getActiveDevice();
   getSmartPointersSynchronizationTimer().start();
   bool b = getSmartPointersRegister().synchronizeDevice( deviceId );
   getSmartPointersSynchronizationTimer().stop();
   return b;
#else
   return true;
#endif
}

inline Timer& Cuda::getSmartPointersSynchronizationTimer()
{
   static Timer timer;
   return timer;
}

inline Pointers::SmartPointersRegister& Cuda::getSmartPointersRegister()
{
   static Pointers::SmartPointersRegister reg;
   return reg;
}

// double-precision atomicAdd function for Maxwell and older GPUs
// copied from: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#ifdef HAVE_CUDA
#if __CUDA_ARCH__ < 600
namespace {
   __device__ double atomicAdd(double* address, double val)
   {
       unsigned long long int* address_as_ull =
                                 (unsigned long long int*)address;
       unsigned long long int old = *address_as_ull, assumed;

       do {
           assumed = old;
           old = atomicCAS(address_as_ull, assumed,
                           __double_as_longlong(val +
                                  __longlong_as_double(assumed)));

       // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
       } while (assumed != old);

       return __longlong_as_double(old);
   }
} // namespace
#endif
#endif

} // namespace Devices
} // namespace TNL
