/***************************************************************************
                          Cuda_impl.h  -  description
                             -------------------
    begin                : Jan 21, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

#include <TNL/Math.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Cuda/DeviceInfo.h>
#include <TNL/Exceptions/CudaBadAlloc.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Exceptions/CudaRuntimeError.h>
#include <TNL/Pointers/SmartPointersRegister.h>

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
   Pointers::getSmartPointersSynchronizationTimer< Devices::Cuda >().reset();
   Pointers::getSmartPointersSynchronizationTimer< Devices::Cuda >().stop();
#endif
   return true;
}

inline constexpr int Cuda::getGPUTransferBufferSize()
{
   return 1 << 20;
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
