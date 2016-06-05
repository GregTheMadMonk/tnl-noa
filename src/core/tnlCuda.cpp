/***************************************************************************
                          tnlCuda.cpp  -  description
                             -------------------
    begin                : Jul 11, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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
#include <core/mfuncs.h>
#include <tnlConfig.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

tnlSmartPointersRegister tnlCuda::smartPointersRegister;

tnlString tnlCuda :: getDeviceType()
{
   return tnlString( "tnlCuda" );
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

void tnlCuda::configSetup( tnlConfigDescription& config, const tnlString& prefix )
{
#ifdef HAVE_CUDA
   config.addEntry<  int >( prefix + "cuda-device", "Choose CUDA device to run the computationon.", 0 );
#else
   config.addEntry<  int >( prefix + "cuda-device", "Choose CUDA device to run the computationon (not supported on this system).", 0 );
#endif
}
      
bool tnlCuda::setup( const tnlParameterContainer& parameters,
                      const tnlString& prefix )
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

void tnlCuda::insertSmartPointer( tnlSmartPointer* pointer )
{
    smartPointersRegister.insert( pointer, 0 );
}

void tnlCuda::removeSmartPointer( tnlSmartPointer* pointer )
{
    smartPointersRegister.remove( pointer, 0 );
}

   
bool tnlCuda::synchronizeDevice( int deviceId )
{
    smartPointersRegister.synchronizeDevice( deviceId );
    return checkCudaDevice;
}

