/***************************************************************************
                          CudaDeviceInfo.cu  -  description
                             -------------------
    begin                : Jun 21, 2015
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifdef HAVE_CUDA

#include <unordered_map>

#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Logger.h>

namespace TNL {
namespace Devices {

int
CudaDeviceInfo::
getNumberOfDevices()
{
    int devices;
    cudaGetDeviceCount( &devices );
    return devices;
}

int
CudaDeviceInfo::
getActiveDevice()
{
    int device;
    cudaGetDevice( &device );
    return device;
}

String
CudaDeviceInfo::
getDeviceName( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return String( properties.name );
}

int
CudaDeviceInfo::
getArchitectureMajor( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.major;
}

int
CudaDeviceInfo::
getArchitectureMinor( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.minor;
}

int
CudaDeviceInfo::
getClockRate( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.clockRate;
}

size_t
CudaDeviceInfo::
getGlobalMemory( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.totalGlobalMem;
}

size_t
CudaDeviceInfo::
getFreeGlobalMemory()
{
   size_t free = 0;
   size_t total = 0;
   cudaMemGetInfo( &free, &total );
   return free;
}

int
CudaDeviceInfo::
getMemoryClockRate( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.memoryClockRate;
}

bool
CudaDeviceInfo::
getECCEnabled( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.ECCEnabled;
}

int
CudaDeviceInfo::
getCudaMultiprocessors( int deviceNum )
{
    // results are cached because they are used for configuration of some kernels
    static std::unordered_map< int, int > results;
    if( results.count( deviceNum ) == 0 ) {
        cudaDeviceProp properties;
        cudaGetDeviceProperties( &properties, deviceNum );
        results.emplace( deviceNum, properties.multiProcessorCount );
        return properties.multiProcessorCount;
    }
    return results[ deviceNum ];
}

int
CudaDeviceInfo::
getCudaCoresPerMultiprocessors( int deviceNum )
{
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
}

int
CudaDeviceInfo::
getCudaCores( int deviceNum )
{
    return CudaDeviceInfo::getCudaMultiprocessors( deviceNum ) *
           CudaDeviceInfo::getCudaCoresPerMultiprocessors( deviceNum );
}

void
CudaDeviceInfo::
writeDeviceInfo( Logger& logger )
{
   logger.writeParameter< String >( "CUDA GPU info", String("") );
   // TODO: Printing all devices does not make sense until TNL can actually
   //       use more than one device for computations. Printing only the active
   //       device for now...
//   int devices = getNumberOfDevices();
//   writeParameter< int >( "Number of devices", devices, 1 );
//   for( int i = 0; i < devices; i++ )
//   {
//      logger.writeParameter< int >( "Device no.", i, 1 );
      int i = getActiveDevice();
      logger.writeParameter< String >( "Name", getDeviceName( i ), 2 );
      String deviceArch = String( getArchitectureMajor( i ) ) + "." +
                              String( getArchitectureMinor( i ) );
      logger.writeParameter< String >( "Architecture", deviceArch, 2 );
      logger.writeParameter< int >( "CUDA cores", getCudaCores( i ), 2 );
      double clockRate = ( double ) getClockRate( i ) / 1.0e3;
      logger.writeParameter< double >( "Clock rate (in MHz)", clockRate, 2 );
      double globalMemory = ( double ) getGlobalMemory( i ) / 1.0e9;
      logger.writeParameter< double >( "Global memory (in GB)", globalMemory, 2 );
      double memoryClockRate = ( double ) getMemoryClockRate( i ) / 1.0e3;
      logger.writeParameter< double >( "Memory clock rate (in Mhz)", memoryClockRate, 2 );
      logger.writeParameter< bool >( "ECC enabled", getECCEnabled( i ), 2 );
//   }
}

} // namespace Devices
} // namespace TNL

#endif
