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
#include <TNL/Devices/Host.h>
#include <TNL/Devices/CudaDeviceInfo.h>

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
   writeParameter< String >( "Host name:", Devices::Host::getHostname() );
   writeParameter< String >( "Architecture:", Devices::Host::getArchitecture() );
   // FIXME: generalize for multi-socket systems, here we consider only the first found CPU
   const int cpu_id = 0;
   const int threads = Devices::Host::getNumberOfThreads( cpu_id );
   const int cores = Devices::Host::getNumberOfCores( cpu_id );
   int threadsPerCore = threads / cores;
   writeParameter< String >( "CPU info", String("") );
   writeParameter< String >( "Model name:", Devices::Host::getCPUModelName( cpu_id ), 1 );
   writeParameter< int >( "Cores:", cores, 1 );
   writeParameter< int >( "Threads per core:", threadsPerCore, 1 );
   writeParameter< String >( "Max clock rate (in MHz):", Devices::Host::getCPUMaxFrequency( cpu_id ) / 1000, 1 );
   Devices::CacheSizes cacheSizes = Devices::Host::getCPUCacheSizes( cpu_id );
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
//      int devices = Devices::CudaDeviceInfo::getNumberOfDevices();
//      writeParameter< int >( "Number of devices", devices, 1 );
//      for( int i = 0; i < devices; i++ )
//      {
//        writeParameter< int >( "Device no.", i, 1 );
        int i = Devices::CudaDeviceInfo::getActiveDevice();
        writeParameter< String >( "Name", Devices::CudaDeviceInfo::getDeviceName( i ), 2 );
        String deviceArch = String( Devices::CudaDeviceInfo::getArchitectureMajor( i ) ) + "." +
                                String( Devices::CudaDeviceInfo::getArchitectureMinor( i ) );
        writeParameter< String >( "Architecture", deviceArch, 2 );
        writeParameter< int >( "CUDA cores", Devices::CudaDeviceInfo::getCudaCores( i ), 2 );
        double clockRate = ( double ) Devices::CudaDeviceInfo::getClockRate( i ) / 1.0e3;
        writeParameter< double >( "Clock rate (in MHz)", clockRate, 2 );
        double globalMemory = ( double ) Devices::CudaDeviceInfo::getGlobalMemory( i ) / 1.0e9;
        writeParameter< double >( "Global memory (in GB)", globalMemory, 2 );
        double memoryClockRate = ( double ) Devices::CudaDeviceInfo::getMemoryClockRate( i ) / 1.0e3;
        writeParameter< double >( "Memory clock rate (in Mhz)", memoryClockRate, 2 );
        writeParameter< bool >( "ECC enabled", Devices::CudaDeviceInfo::getECCEnabled( i ), 2 );
//      }
   }
   writeParameter< String >( "System:", Devices::Host::getSystemName() );
   writeParameter< String >( "Release:", Devices::Host::getSystemRelease() );
   writeParameter< char* >( "TNL Compiler:", ( char* ) TNL_CPP_COMPILER_NAME );
   return true;
}

void Logger :: writeCurrentTime( const char* label )
{
   writeParameter< String >( label, Devices::Host::getCurrentTime() );
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
