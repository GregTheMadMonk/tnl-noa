/***************************************************************************
                          Cuda.cpp  -  description
                             -------------------
    begin                : Jul 11, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Devices/Cuda.h>
#include <TNL/Math.h>
#include <TNL/Devices/CudaDeviceInfo.h>

namespace TNL {
namespace Devices {

SmartPointersRegister Cuda::smartPointersRegister;
Timer Cuda::smartPointersSynchronizationTimer;

String Cuda::getDeviceType()
{
   return String( "Devices::Cuda" );
}

int Cuda::getNumberOfBlocks( const int threads,
                             const int blockSize )
{
   return roundUpDivision( threads, blockSize );
}

int Cuda::getNumberOfGrids( const int blocks,
                            const int gridSize )
{
   return roundUpDivision( blocks, gridSize );
}

void Cuda::insertSmartPointer( SmartPointer* pointer )
{
   smartPointersRegister.insert( pointer, Devices::CudaDeviceInfo::getActiveDevice() );
}

void Cuda::removeSmartPointer( SmartPointer* pointer )
{
   smartPointersRegister.remove( pointer, Devices::CudaDeviceInfo::getActiveDevice() );
}

bool Cuda::synchronizeDevice( int deviceId )
{
#ifdef HAVE_CUDA_UNIFIED_MEMORY
   return true;
#else
   if( deviceId < 0 )
      deviceId = Devices::CudaDeviceInfo::getActiveDevice();
   smartPointersSynchronizationTimer.start();
   bool b = smartPointersRegister.synchronizeDevice( deviceId );
   smartPointersSynchronizationTimer.stop();
   return b;
#endif
}

} // namespace Devices
} // namespace TNL

