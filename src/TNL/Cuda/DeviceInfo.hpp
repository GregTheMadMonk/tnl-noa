/***************************************************************************
                          DeviceInfo.hpp  -  description
                             -------------------
    begin                : Jun 21, 2015
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <unordered_map>

#include <TNL/Cuda/DeviceInfo.h>
#include <TNL/Exceptions/CudaSupportMissing.h>

namespace TNL {
namespace Cuda {

inline int
DeviceInfo::
getNumberOfDevices()
{
#ifdef HAVE_CUDA
   int devices;
   cudaGetDeviceCount( &devices );
   return devices;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline int
DeviceInfo::
getActiveDevice()
{
#ifdef HAVE_CUDA
   int device;
   cudaGetDevice( &device );
   return device;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline String
DeviceInfo::
getDeviceName( int deviceNum )
{
#ifdef HAVE_CUDA
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, deviceNum );
   return String( properties.name );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline int
DeviceInfo::
getArchitectureMajor( int deviceNum )
{
#ifdef HAVE_CUDA
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, deviceNum );
   return properties.major;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline int
DeviceInfo::
getArchitectureMinor( int deviceNum )
{
#ifdef HAVE_CUDA
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, deviceNum );
   return properties.minor;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline int
DeviceInfo::
getClockRate( int deviceNum )
{
#ifdef HAVE_CUDA
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, deviceNum );
   return properties.clockRate;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline std::size_t
DeviceInfo::
getGlobalMemory( int deviceNum )
{
#ifdef HAVE_CUDA
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, deviceNum );
   return properties.totalGlobalMem;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline std::size_t
DeviceInfo::
getFreeGlobalMemory()
{
#ifdef HAVE_CUDA
   std::size_t free = 0;
   std::size_t total = 0;
   cudaMemGetInfo( &free, &total );
   return free;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline int
DeviceInfo::
getMemoryClockRate( int deviceNum )
{
#ifdef HAVE_CUDA
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, deviceNum );
   return properties.memoryClockRate;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline bool
DeviceInfo::
getECCEnabled( int deviceNum )
{
#ifdef HAVE_CUDA
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, deviceNum );
   return properties.ECCEnabled;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline int
DeviceInfo::
getCudaMultiprocessors( int deviceNum )
{
#ifdef HAVE_CUDA
   // results are cached because they are used for configuration of some kernels
   static std::unordered_map< int, int > results;
   if( results.count( deviceNum ) == 0 ) {
      cudaDeviceProp properties;
      cudaGetDeviceProperties( &properties, deviceNum );
      results.emplace( deviceNum, properties.multiProcessorCount );
      return properties.multiProcessorCount;
   }
   return results[ deviceNum ];
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline int
DeviceInfo::
getCudaCoresPerMultiprocessors( int deviceNum )
{
#ifdef HAVE_CUDA
   int major = DeviceInfo::getArchitectureMajor( deviceNum );
   int minor = DeviceInfo::getArchitectureMinor( deviceNum );
   switch( major )
   {
      case 1:   // Tesla generation, G80, G8x, G9x classes
         return 8;
      case 2:   // Fermi generation
         switch( minor )
         {
            case 0:  // GF100 class
               return 32;
            case 1:  // GF10x class
               return 48;
            default:
               return -1;
         }
      case 3: // Kepler generation -- GK10x, GK11x classes
         return 192;
      case 5: // Maxwell generation -- GM10x, GM20x classes
         return 128;
      case 6: // Pascal generation
         switch( minor )
         {
            case 0:  // GP100 class
               return 64;
            case 1:  // GP10x classes
            case 2:
               return 128;
            default:
               return -1;
         }
      default:
         return -1;
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline int
DeviceInfo::
getCudaCores( int deviceNum )
{
#ifdef HAVE_CUDA
   return DeviceInfo::getCudaMultiprocessors( deviceNum ) *
          DeviceInfo::getCudaCoresPerMultiprocessors( deviceNum );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline int
DeviceInfo::
getRegistersPerMultiprocessor( int deviceNum )
{
#ifdef HAVE_CUDA
   // results are cached because they are used for configuration of some kernels
   static std::unordered_map< int, int > results;
   if( results.count( deviceNum ) == 0 ) {
      cudaDeviceProp properties;
      cudaGetDeviceProperties( &properties, deviceNum );
      results.emplace( deviceNum, properties.regsPerMultiprocessor );
      return properties.regsPerMultiprocessor;
   }
   return results[ deviceNum ];
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

} // namespace Cuda
} // namespace TNL
