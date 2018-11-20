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
#include <TNL/tnlConfig.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
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

void Cuda::configSetup( Config::ConfigDescription& config,
                        const String& prefix )
{
// FIXME: HAVE_CUDA is never defined in .cpp files
#ifdef HAVE_CUDA
   config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device to run the computation.", 0 );
#else
   config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device to run the computation (not supported on this system).", 0 );
#endif
}

bool Cuda::setup( const Config::ParameterContainer& parameters,
                  const String& prefix )
{
// FIXME: HAVE_CUDA is never defined in .cpp files
#ifdef HAVE_CUDA
   int cudaDevice = parameters.getParameter< int >( prefix + "cuda-device" );
   if( cudaSetDevice( cudaDevice ) != cudaSuccess )
   {
      std::cerr << "I cannot activate CUDA device number " << cudaDevice << "." << std::endl;
      return false;
   }
   smartPointersSynchronizationTimer.reset();
   smartPointersSynchronizationTimer.stop();
#endif
   return true;
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

