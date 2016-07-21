/***************************************************************************
                          tnlCuda.cpp  -  description
                             -------------------
    begin                : Jul 11, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/core/tnlCuda.h>
#include <TNL/core/mfuncs.h>
#include <TNL/tnlConfig.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
 
String tnlCuda :: getDeviceType()
{
   return String( "tnlCuda" );
}

int tnlCuda::getGPUTransferBufferSize()
{
   return 1 << 20;
}

int tnlCuda::getNumberOfBlocks( const int threads,
                                const int blockSize )
{
   return roundUpDivision( threads, blockSize );
}

int tnlCuda::getNumberOfGrids( const int blocks,
                               const int gridSize )
{
   return roundUpDivision( blocks, gridSize );
}

/*size_t tnlCuda::getFreeMemory()
{

}*/

void tnlCuda::configSetup( Config::ConfigDescription& config, const String& prefix )
{
#ifdef HAVE_CUDA
   config.addEntry<  int >( prefix + "cuda-device", "Choose CUDA device to run the computationon.", 0 );
#else
   config.addEntry<  int >( prefix + "cuda-device", "Choose CUDA device to run the computationon (not supported on this system).", 0 );
#endif
}
 
bool tnlCuda::setup( const Config::ParameterContainer& parameters,
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

} // namespace TNL

