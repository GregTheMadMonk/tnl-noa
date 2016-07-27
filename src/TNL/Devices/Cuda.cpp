/***************************************************************************
                          Cuda.cpp  -  description
                             -------------------
    begin                : Jul 11, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Devices/Cuda.h>
#include <TNL/core/mfuncs.h>
#include <TNL/tnlConfig.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Devices {
 
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
   config.addEntry<  int >( prefix + "cuda-device", "Choose CUDA device to run the computationon.", 0 );
#else
   config.addEntry<  int >( prefix + "cuda-device", "Choose CUDA device to run the computationon (not supported on this system).", 0 );
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

} // namespace Devices
} // namespace TNL

