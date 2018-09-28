/***************************************************************************
                          Cuda.cu  -  description
                             -------------------
    begin                : Dec 22, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Devices/Cuda.h>
#include <TNL/Exceptions/CudaRuntimeError.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Devices {


void Cuda::setupThreads( const dim3& blockSize,
                         dim3& blocksCount,
                         dim3& gridsCount,
                         long long int xThreads,
                         long long int yThreads,
                         long long int zThreads )
{
   blocksCount.x = max( 1, xThreads / blockSize.x + ( xThreads % blockSize.x != 0 ) );
   blocksCount.y = max( 1, yThreads / blockSize.y + ( yThreads % blockSize.y != 0 ) );
   blocksCount.z = max( 1, zThreads / blockSize.z + ( zThreads % blockSize.z != 0 ) );
   
   /****
    * TODO: Fix the following:
    * I do not known how to get max grid size in kernels :(
    * 
    * Also, this is very slow. */
   /*int currentDevice( 0 );
   cudaGetDevice( currentDevice );
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, currentDevice );
   gridsCount.x = blocksCount.x / properties.maxGridSize[ 0 ] + ( blocksCount.x % properties.maxGridSize[ 0 ] != 0 );
   gridsCount.y = blocksCount.y / properties.maxGridSize[ 1 ] + ( blocksCount.y % properties.maxGridSize[ 1 ] != 0 );
   gridsCount.z = blocksCount.z / properties.maxGridSize[ 2 ] + ( blocksCount.z % properties.maxGridSize[ 2 ] != 0 );
   */
   gridsCount.x = blocksCount.x / getMaxGridSize() + ( blocksCount.x % getMaxGridSize() != 0 );
   gridsCount.y = blocksCount.y / getMaxGridSize() + ( blocksCount.y % getMaxGridSize() != 0 );
   gridsCount.z = blocksCount.z / getMaxGridSize() + ( blocksCount.z % getMaxGridSize() != 0 );
}

void Cuda::setupGrid( const dim3& blocksCount,
                      const dim3& gridsCount,
                      const dim3& gridIdx,
                      dim3& gridSize )
{
   /* TODO: this is extremely slow!!!!
   int currentDevice( 0 );
   cudaGetDevice( &currentDevice );
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, currentDevice );*/
 
   /****
    * TODO: fix the following
   if( gridIdx.x < gridsCount.x )
      gridSize.x = properties.maxGridSize[ 0 ];
   else
      gridSize.x = blocksCount.x % properties.maxGridSize[ 0 ];
   
   if( gridIdx.y < gridsCount.y )
      gridSize.y = properties.maxGridSize[ 1 ];
   else
      gridSize.y = blocksCount.y % properties.maxGridSize[ 1 ];

   if( gridIdx.z < gridsCount.z )
      gridSize.z = properties.maxGridSize[ 2 ];
   else
      gridSize.z = blocksCount.z % properties.maxGridSize[ 2 ];*/
   
   if( gridIdx.x < gridsCount.x - 1 )
      gridSize.x = getMaxGridSize();
   else
      gridSize.x = blocksCount.x % getMaxGridSize();
   
   if( gridIdx.y < gridsCount.y - 1 )
      gridSize.y = getMaxGridSize();
   else
      gridSize.y = blocksCount.y % getMaxGridSize();

   if( gridIdx.z < gridsCount.z - 1 )
      gridSize.z = getMaxGridSize();
   else
      gridSize.z = blocksCount.z % getMaxGridSize();
}

void Cuda::printThreadsSetup( const dim3& blockSize,
                              const dim3& blocksCount,
                              const dim3& gridSize,
                              const dim3& gridsCount,
                              std::ostream& str )
{
   str << "Block size: " << blockSize << std::endl
       << " Blocks count: " << blocksCount << std::endl
       << " Grid size: " << gridSize << std::endl
       << " Grids count: " << gridsCount << std::endl;
}


void Cuda::checkDevice( const char* file_name, int line, cudaError error )
{
   if( error != cudaSuccess )
      throw Exceptions::CudaRuntimeError( error, file_name, line );
}

std::ostream& operator << ( std::ostream& str, const dim3& d )
{
   str << "( " << d.x << ", " << d.y << ", " << d.z << " )";
   return str;
}

} // namespace Devices
} // namespace TNL
