/***************************************************************************
                          Logger.cpp  -  description
                             -------------------
    begin                : 2007/08/22
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <iomanip>
#include <TNL/Logger.h>
#include <TNL/tnlConfig.h>
#include <TNL/SystemInfo.h>
#include <TNL/core/tnlCudaDeviceInfo.h>

namespace TNL {

Logger :: Logger( int _width,
                        std::ostream& _stream )
: width( _width ),
  stream( _stream )
{
}

void Logger :: writeHeader( const String& title )
{
   int fill = stream. fill();
   int titleLength = title. getLength();
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream << "|" << std::setfill( ' ' ) << std::setw( width ) << "|" << std::endl;
   stream << "|" << std::setw( width / 2 + titleLength / 2 )
    << title << std::setw( width / 2 - titleLength / 2  ) << "|" << std::endl;
   stream << "|" << std::setfill( ' ' ) << std::setw( width ) << "|" << std::endl;
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream. fill( fill );
}

void Logger :: writeSeparator()
{
   int fill = stream. fill();
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream. fill( fill );
}

bool Logger :: writeSystemInformation( const Config::ParameterContainer& parameters )
{
   SystemInfo systemInfo;


   writeParameter< String >( "Host name:", systemInfo.getHostname() );
   writeParameter< String >( "Architecture:", systemInfo.getArchitecture() );
   // FIXME: generalize for multi-socket systems, here we consider only the first found CPU
   const int cpu_id = 0;
   const int threads = systemInfo.getNumberOfThreads( cpu_id );
   const int cores = systemInfo.getNumberOfCores( cpu_id );
   int threadsPerCore = threads / cores;
   writeParameter< String >( "CPU info", String("") );
   writeParameter< String >( "Model name:", systemInfo.getCPUModelName( cpu_id ), 1 );
   writeParameter< int >( "Cores:", cores, 1 );
   writeParameter< int >( "Threads per core:", threadsPerCore, 1 );
   writeParameter< String >( "Max clock rate (in MHz):", systemInfo.getCPUMaxFrequency( cpu_id ) / 1000, 1 );
   tnlCacheSizes cacheSizes = systemInfo.getCPUCacheSizes( cpu_id );
   String cacheInfo = String( cacheSizes.L1data ) + ", "
                       + String( cacheSizes.L1instruction ) + ", "
                       + String( cacheSizes.L2 ) + ", "
                       + String( cacheSizes.L3 );
   writeParameter< String >( "Cache (L1d, L1i, L2, L3):", cacheInfo, 1 );
   if( parameters.getParameter< String >( "device" ) == "cuda" )
   {
      writeParameter< String >( "CUDA GPU info", String("") );
      // TODO: Printing all devices does not make sense, but in the future TNL
      //       might use more than one device for computations. Printing only
      //       the active device for now...
//      int devices = tnlCudaDeviceInfo::getNumberOfDevices();
//      writeParameter< int >( "Number of devices", devices, 1 );
//      for( int i = 0; i < devices; i++ )
//      {
//        writeParameter< int >( "Device no.", i, 1 );
        int i = tnlCudaDeviceInfo::getActiveDevice();
        writeParameter< String >( "Name", tnlCudaDeviceInfo::getDeviceName( i ), 2 );
        String deviceArch = String( tnlCudaDeviceInfo::getArchitectureMajor( i ) ) + "." +
                                String( tnlCudaDeviceInfo::getArchitectureMinor( i ) );
        writeParameter< String >( "Architecture", deviceArch, 2 );
        writeParameter< int >( "CUDA cores", tnlCudaDeviceInfo::getCudaCores( i ), 2 );
        double clockRate = ( double ) tnlCudaDeviceInfo::getClockRate( i ) / 1.0e3;
        writeParameter< double >( "Clock rate (in MHz)", clockRate, 2 );
        double globalMemory = ( double ) tnlCudaDeviceInfo::getGlobalMemory( i ) / 1.0e9;
        writeParameter< double >( "Global memory (in GB)", globalMemory, 2 );
        double memoryClockRate = ( double ) tnlCudaDeviceInfo::getMemoryClockRate( i ) / 1.0e3;
        writeParameter< double >( "Memory clock rate (in Mhz)", memoryClockRate, 2 );
        writeParameter< bool >( "ECC enabled", tnlCudaDeviceInfo::getECCEnabled( i ), 2 );
//      }
   }
   writeParameter< String >( "System:", systemInfo.getSystemName() );
   writeParameter< String >( "Release:", systemInfo.getSystemRelease() );
   writeParameter< char* >( "TNL Compiler:", ( char* ) TNL_CPP_COMPILER_NAME );
   return true;
}

void Logger :: writeCurrentTime( const char* label )
{
   SystemInfo systemInfo;
   writeParameter< String >( label, systemInfo.getCurrentTime() );
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION
template void Logger::writeParameter< char* >( const String&,
                                                  const String&,
                                                  const Config::ParameterContainer&,
                                                  int );
template void Logger::writeParameter< double >( const String&,
                                                   const String&,
                                                   const Config::ParameterContainer&,
                                                   int );
template void Logger::writeParameter< int >( const String&,
                                                const String&,
                                                const Config::ParameterContainer&,
                                                int );

// TODO: fix this
//template void Logger :: WriteParameter< char* >( const char*,
//                                                    const char*&,
//                                                    int );
template void Logger::writeParameter< double >( const String&,
                                                   const double&,
                                                   int );
template void Logger::writeParameter< int >( const String&,
                                                const int&,
                                                int );

#endif

} // namespace TNL
