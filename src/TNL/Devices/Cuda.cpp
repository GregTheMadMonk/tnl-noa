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

namespace TNL {
namespace Devices {
 
SmartPointersRegister Cuda::smartPointersRegister;   
   
String Cuda::getDeviceType()
{
   return String( "Cuda" );
}

int Cuda::getGPUTransferBufferSize()
{
   return 1 << 20;
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

/*size_t Cuda::getFreeMemory()
{

}*/

void Cuda::configSetup( Config::ConfigDescription& config, const String& prefix )
{
#ifdef HAVE_CUDA
   config.addEntry<  int >( prefix + "cuda-device", "Choose CUDA device to run the computation.", 0 );
#else
   config.addEntry<  int >( prefix + "cuda-device", "Choose CUDA device to run the computation (not supported on this system).", 0 );
#endif
}
 
bool Cuda::setup( const Config::ParameterContainer& parameters,
                      const String& prefix )
{
#ifdef HAVE_CUDA
   int cudaDevice = parameters.getParameter< int >( "cuda-device" );
   if( cudaSetDevice( cudaDevice ) != cudaSuccess )
   {
      std::cerr << "I cannot activate CUDA device number " << cudaDevice << "." << std::endl;
      return false;
   }
#endif
   return true;
}

void Cuda::insertSmartPointer( SmartPointer* pointer )
{
    smartPointersRegister.insert( pointer, 0 );
}

void Cuda::removeSmartPointer( SmartPointer* pointer )
{
    smartPointersRegister.remove( pointer, 0 );
}
   
bool Cuda::synchronizeDevice( int deviceId )
{
    smartPointersRegister.synchronizeDevice( deviceId );
    return checkCudaDevice;
}

} // namespace Devices
} // namespace TNL

