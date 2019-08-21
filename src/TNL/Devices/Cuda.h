/***************************************************************************
                          Cuda.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

#include <TNL/String.h>
#include <TNL/Assert.h>
#include <TNL/Pointers/SmartPointersRegister.h>
#include <TNL/Timer.h>
#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Devices {

class Cuda
{
public:
   static inline void configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   static inline bool setup( const Config::ParameterContainer& parameters,
                             const String& prefix = "" );

   static inline constexpr int getGPUTransferBufferSize();

   static inline void insertSmartPointer( Pointers::SmartPointer* pointer );

   static inline void removeSmartPointer( Pointers::SmartPointer* pointer );

   // Negative deviceId means that CudaDeviceInfo::getActiveDevice will be
   // called to get the device ID.
   static inline bool synchronizeDevice( int deviceId = -1 );

   static inline Timer& getSmartPointersSynchronizationTimer();

   ////
   // When we transfer data between the GPU and the CPU we use 5 MB buffer. This
   // size should ensure good performance -- see.
   // http://wiki.accelereyes.com/wiki/index.php/GPU_Memory_Transfer .
   // We use the same buffer size even for retyping data during IO operations.
   //
   static constexpr std::size_t TransferBufferSize = 5 * 2<<20;


   protected:

   static inline Pointers::SmartPointersRegister& getSmartPointersRegister();
};

#ifdef HAVE_CUDA
#if __CUDA_ARCH__ < 600
namespace {
   __device__ double atomicAdd(double* address, double val);
}
#endif
#endif

} // namespace Devices
} // namespace TNL

#include <TNL/Devices/Cuda_impl.h>
