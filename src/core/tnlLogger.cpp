/***************************************************************************
                          tnlLogger.cpp  -  description
                             -------------------
    begin                : 2007/08/22
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <iomanip>
#include <core/tnlLogger.h>
#include <tnlConfig.h>
#include <core/tnlSystemInfo.h>
#include <core/tnlCudaDeviceInfo.h>

namespace TNL {

tnlLogger :: tnlLogger( int _width,
                        ostream& _stream )
: width( _width ),
  stream( _stream )
{
}

void tnlLogger :: writeHeader( const tnlString& title )
{
   int fill = stream. fill();
   int titleLength = title. getLength();
   stream << "+" << setfill( '-' ) << setw( width ) << "+" << endl;
   stream << "|" << setfill( ' ' ) << setw( width ) << "|" << endl;
   stream << "|" << setw( width / 2 + titleLength / 2 )
    << title << setw( width / 2 - titleLength / 2  ) << "|" << endl;
   stream << "|" << setfill( ' ' ) << setw( width ) << "|" << endl;
   stream << "+" << setfill( '-' ) << setw( width ) << "+" << endl;
   stream. fill( fill );
}

void tnlLogger :: writeSeparator()
{
   int fill = stream. fill();
   stream << "+" << setfill( '-' ) << setw( width ) << "+" << endl;
   stream. fill( fill );
}

bool tnlLogger :: writeSystemInformation( const tnlParameterContainer& parameters )
{
   tnlSystemInfo systemInfo;


   writeParameter< tnlString >( "Host name:", systemInfo.getHostname() );
   writeParameter< tnlString >( "Architecture:", systemInfo.getArchitecture() );
   // FIXME: generalize for multi-socket systems, here we consider only the first found CPU
   const int cpu_id = 0;
   const int threads = systemInfo.getNumberOfThreads( cpu_id );
   const int cores = systemInfo.getNumberOfCores( cpu_id );
   int threadsPerCore = threads / cores;
   writeParameter< tnlString >( "CPU info", tnlString("") );
   writeParameter< tnlString >( "Model name:", systemInfo.getCPUModelName( cpu_id ), 1 );
   writeParameter< int >( "Cores:", cores, 1 );
   writeParameter< int >( "Threads per core:", threadsPerCore, 1 );
   writeParameter< tnlString >( "Max clock rate (in MHz):", systemInfo.getCPUMaxFrequency( cpu_id ) / 1000, 1 );
   tnlCacheSizes cacheSizes = systemInfo.getCPUCacheSizes( cpu_id );
   tnlString cacheInfo = tnlString( cacheSizes.L1data ) + ", "
                       + tnlString( cacheSizes.L1instruction ) + ", "
                       + tnlString( cacheSizes.L2 ) + ", "
                       + tnlString( cacheSizes.L3 );
   writeParameter< tnlString >( "Cache (L1d, L1i, L2, L3):", cacheInfo, 1 );
   if( parameters.getParameter< tnlString >( "device" ) == "cuda" )
   {
      writeParameter< tnlString >( "CUDA GPU info", tnlString("") );
      // TODO: Printing all devices does not make sense, but in the future TNL
      //       might use more than one device for computations. Printing only
      //       the active device for now...
//      int devices = tnlCudaDeviceInfo::getNumberOfDevices();
//      writeParameter< int >( "Number of devices", devices, 1 );
//      for( int i = 0; i < devices; i++ )
//      {
//        writeParameter< int >( "Device no.", i, 1 );
        int i = tnlCudaDeviceInfo::getActiveDevice();
        writeParameter< tnlString >( "Name", tnlCudaDeviceInfo::getDeviceName( i ), 2 );
        tnlString deviceArch = tnlString( tnlCudaDeviceInfo::getArchitectureMajor( i ) ) + "." +
                                tnlString( tnlCudaDeviceInfo::getArchitectureMinor( i ) );
        writeParameter< tnlString >( "Architecture", deviceArch, 2 );
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
   writeParameter< tnlString >( "System:", systemInfo.getSystemName() );
   writeParameter< tnlString >( "Release:", systemInfo.getSystemRelease() );
   writeParameter< char* >( "TNL Compiler:", ( char* ) TNL_CPP_COMPILER_NAME );
   return true;
}

void tnlLogger :: writeCurrentTime( const char* label )
{
   tnlSystemInfo systemInfo;
   writeParameter< tnlString >( label, systemInfo.getCurrentTime() );
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION
template void tnlLogger::writeParameter< char* >( const tnlString&,
                                                  const tnlString&,
                                                  const tnlParameterContainer&,
                                                  int );
template void tnlLogger::writeParameter< double >( const tnlString&,
                                                   const tnlString&,
                                                   const tnlParameterContainer&,
                                                   int );
template void tnlLogger::writeParameter< int >( const tnlString&,
                                                const tnlString&,
                                                const tnlParameterContainer&,
                                                int );

// TODO: fix this
//template void tnlLogger :: WriteParameter< char* >( const char*,
//                                                    const char*&,
//                                                    int );
template void tnlLogger::writeParameter< double >( const tnlString&,
                                                   const double&,
                                                   int );
template void tnlLogger::writeParameter< int >( const tnlString&,
                                                const int&,
                                                int );

#endif

} // namespace TNL
