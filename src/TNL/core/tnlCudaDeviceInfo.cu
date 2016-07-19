/***************************************************************************
                          tnlCudaDeviceInfo.cu  -  description
                             -------------------
    begin                : Jun 21, 2015
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifdef HAVE_CUDA

#include <TNL/core/tnlCudaDeviceInfo.h>
#include <TNL/core/tnlCuda.h>

namespace TNL {

int
tnlCudaDeviceInfo::
getNumberOfDevices()
{
    int devices;
    cudaGetDeviceCount( &devices );
    return devices;
}

int
tnlCudaDeviceInfo::
getActiveDevice()
{
    int device;
    cudaGetDevice( &device );
    return device;
}

tnlString
tnlCudaDeviceInfo::
getDeviceName( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return tnlString( properties.name );
}

int
tnlCudaDeviceInfo::
getArchitectureMajor( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.major;
}

int
tnlCudaDeviceInfo::
getArchitectureMinor( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.minor;
}

int
tnlCudaDeviceInfo::
getClockRate( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.clockRate;
}

size_t
tnlCudaDeviceInfo::
getGlobalMemory( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.totalGlobalMem;
}

int
tnlCudaDeviceInfo::
getMemoryClockRate( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.memoryClockRate;
}

bool
tnlCudaDeviceInfo::
getECCEnabled( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.ECCEnabled;
}

int
tnlCudaDeviceInfo::
getCudaMultiprocessors( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.multiProcessorCount;
}

int
tnlCudaDeviceInfo::
getCudaCoresPerMultiprocessors( int deviceNum )
{
    int major = tnlCudaDeviceInfo::getArchitectureMajor( deviceNum );
    int minor = tnlCudaDeviceInfo::getArchitectureMinor( deviceNum );
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
tnlCudaDeviceInfo::
getCudaCores( int deviceNum )
{
    return tnlCudaDeviceInfo::getCudaMultiprocessors( deviceNum ) *
           tnlCudaDeviceInfo::getCudaCoresPerMultiprocessors( deviceNum );
}

} // namespace TNL

#endif
