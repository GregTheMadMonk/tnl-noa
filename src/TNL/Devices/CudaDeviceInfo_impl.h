/***************************************************************************
                          CudaDeviceInfo_impl.h  -  description
                             -------------------
    begin                : Jun 21, 2015
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <unordered_map>

#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Exceptions/CudaSupportMissing.h>

namespace TNL {
namespace Devices {

inline int
CudaDeviceInfo::
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
CudaDeviceInfo::
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
CudaDeviceInfo::
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
CudaDeviceInfo::
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
CudaDeviceInfo::
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
CudaDeviceInfo::
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

inline size_t
CudaDeviceInfo::
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

inline size_t
CudaDeviceInfo::
getFreeGlobalMemory()
{
#ifdef HAVE_CUDA
   size_t free = 0;
   size_t total = 0;
   cudaMemGetInfo( &free, &total );
   return free;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline int
CudaDeviceInfo::
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
CudaDeviceInfo::
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
CudaDeviceInfo::
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
CudaDeviceInfo::
getCudaCoresPerMultiprocessors( int deviceNum )
{
#ifdef HAVE_CUDA
   int major = CudaDeviceInfo::getArchitectureMajor( deviceNum );
   int minor = CudaDeviceInfo::getArchitectureMinor( deviceNum );
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
         }
      default:
         return -1;
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline int
CudaDeviceInfo::
getCudaCores( int deviceNum )
{
#ifdef HAVE_CUDA
   return CudaDeviceInfo::getCudaMultiprocessors( deviceNum ) *
          CudaDeviceInfo::getCudaCoresPerMultiprocessors( deviceNum );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

inline int
CudaDeviceInfo::
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

} // namespace Devices
} // namespace TNL
