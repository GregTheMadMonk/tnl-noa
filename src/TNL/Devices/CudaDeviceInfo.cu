/***************************************************************************
                          CudaDeviceInfo.cu  -  description
                             -------------------
    begin                : Jun 21, 2015
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifdef HAVE_CUDA

#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Devices/Cuda.h>

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
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.multiProcessorCount;
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

} // namespace Devices
} // namespace TNL

#endif
