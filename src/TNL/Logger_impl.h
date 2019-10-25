/***************************************************************************
                          Logger_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <sstream>
#include <iomanip>

#include <TNL/Logger.h>
#include <TNL/Cuda/DeviceInfo.h>
#include <TNL/Devices/SystemInfo.h>

namespace TNL {

inline void
Logger::writeHeader( const String& title )
{
   const int fill = stream.fill();
   const int titleLength = title.getLength();
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream << "|" << std::setfill( ' ' ) << std::setw( width ) << "|" << std::endl;
   stream << "|" << std::setw( width / 2 + titleLength / 2 )
          << title << std::setw( width / 2 - titleLength / 2  ) << "|" << std::endl;
   stream << "|" << std::setfill( ' ' ) << std::setw( width ) << "|" << std::endl;
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream.fill( fill );
}

inline void
Logger::writeSeparator()
{
   const int fill = stream.fill();
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream.fill( fill );
}

inline bool
Logger::writeSystemInformation( const Config::ParameterContainer& parameters )
{
// compiler detection macros:
// http://nadeausoftware.com/articles/2012/10/c_c_tip_how_detect_compiler_name_and_version_using_compiler_predefined_macros
// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#compilation-phases
#if defined(__NVCC__)
   #define TNL_STRINGIFY(x) #x
   const char* compiler_name = "Nvidia NVCC (" TNL_STRINGIFY(__CUDACC_VER_MAJOR__) "." TNL_STRINGIFY(__CUDACC_VER_MINOR__) "." TNL_STRINGIFY(__CUDACC_VER_BUILD__) ")";
   #undef TNL_STRINGIFY
#elif defined(__clang__)
   const char* compiler_name = "Clang/LLVM (" __VERSION__ ")";
#elif defined(__ICC) || defined(__INTEL_COMPILER)
   const char* compiler_name = "Intel ICPC (" __VERSION__ ")";
#elif defined(__GNUC__) || defined(__GNUG__)
   const char* compiler_name = "GNU G++ (" __VERSION__ ")";
#else
   const char* compiler_name = "(unknown)";
#endif

   writeParameter< String >( "Host name:", Devices::SystemInfo::getHostname() );
   writeParameter< String >( "System:", Devices::SystemInfo::getSystemName() );
   writeParameter< String >( "Release:", Devices::SystemInfo::getSystemRelease() );
   writeParameter< String >( "Architecture:", Devices::SystemInfo::getArchitecture() );
   writeParameter< String >( "TNL compiler:", compiler_name );
   // FIXME: generalize for multi-socket systems, here we consider only the first found CPU
   const int cpu_id = 0;
   const int threads = Devices::SystemInfo::getNumberOfThreads( cpu_id );
   const int cores = Devices::SystemInfo::getNumberOfCores( cpu_id );
   int threadsPerCore = 0;
   if( cores > 0 )
      threadsPerCore = threads / cores;
   writeParameter< String >( "CPU info", "" );
   writeParameter< String >( "Model name:", Devices::SystemInfo::getCPUModelName( cpu_id ), 1 );
   writeParameter< int >( "Cores:", cores, 1 );
   writeParameter< int >( "Threads per core:", threadsPerCore, 1 );
   writeParameter< double >( "Max clock rate (in MHz):", Devices::SystemInfo::getCPUMaxFrequency( cpu_id ) / 1000, 1 );
   const Devices::CacheSizes cacheSizes = Devices::SystemInfo::getCPUCacheSizes( cpu_id );
   const String cacheInfo = convertToString( cacheSizes.L1data ) + ", "
                          + convertToString( cacheSizes.L1instruction ) + ", "
                          + convertToString( cacheSizes.L2 ) + ", "
                          + convertToString( cacheSizes.L3 );
   writeParameter< String >( "Cache (L1d, L1i, L2, L3):", cacheInfo, 1 );

   if( parameters.getParameter< String >( "device" ) == "cuda" ) {
      writeParameter< String >( "CUDA GPU info", "" );
      // TODO: Printing all devices does not make sense until TNL can actually
      //       use more than one device for computations. Printing only the active
      //       device for now...
   //   int devices = getNumberOfDevices();
   //   writeParameter< int >( "Number of devices", devices, 1 );
   //   for( int i = 0; i < devices; i++ )
   //   {
   //      logger.writeParameter< int >( "Device no.", i, 1 );
         const int i = Cuda::DeviceInfo::getActiveDevice();
         writeParameter< String >( "Name", Cuda::DeviceInfo::getDeviceName( i ), 2 );
         const String deviceArch = convertToString( Cuda::DeviceInfo::getArchitectureMajor( i ) ) + "." +
                                   convertToString( Cuda::DeviceInfo::getArchitectureMinor( i ) );
         writeParameter< String >( "Architecture", deviceArch, 2 );
         writeParameter< int >( "CUDA cores", Cuda::DeviceInfo::getCudaCores( i ), 2 );
         const double clockRate = ( double ) Cuda::DeviceInfo::getClockRate( i ) / 1.0e3;
         writeParameter< double >( "Clock rate (in MHz)", clockRate, 2 );
         const double globalMemory = ( double ) Cuda::DeviceInfo::getGlobalMemory( i ) / 1.0e9;
         writeParameter< double >( "Global memory (in GB)", globalMemory, 2 );
         const double memoryClockRate = ( double ) Cuda::DeviceInfo::getMemoryClockRate( i ) / 1.0e3;
         writeParameter< double >( "Memory clock rate (in Mhz)", memoryClockRate, 2 );
         writeParameter< bool >( "ECC enabled", Cuda::DeviceInfo::getECCEnabled( i ), 2 );
   //   }
   }
   return true;
}

inline void
Logger::writeCurrentTime( const char* label )
{
   writeParameter< String >( label, Devices::SystemInfo::getCurrentTime() );
}

template< typename T >
void
Logger::writeParameter( const String& label,
                        const String& parameterName,
                        const Config::ParameterContainer& parameters,
                        int parameterLevel )
{
   stream << "| ";
   int i;
   for( i = 0; i < parameterLevel; i ++ )
      stream << " ";
   std::stringstream str;
   str << parameters.getParameter< T >( parameterName );
   stream << label
          << std::setw( width - label.getLength() - parameterLevel - 3 )
          << str.str() << " |" << std::endl;
}

template< typename T >
void
Logger::writeParameter( const String& label,
                        const T& value,
                        int parameterLevel )
{
   stream << "| ";
   int i;
   for( i = 0; i < parameterLevel; i ++ )
      stream << " ";
   std::stringstream str;
   str << value;
   stream << label
          << std::setw( width - label.getLength() - parameterLevel - 3 )
          << str.str() << " |" << std::endl;
}

} // namespace TNL
