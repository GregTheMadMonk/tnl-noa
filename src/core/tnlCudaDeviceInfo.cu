/***************************************************************************
                          tnlCudaDeviceInfo.cu  -  description
                             -------------------
    begin                : Jun 21, 2015
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifdef HAVE_CUDA

#include <core/tnlCudaDeviceInfo.h>
#include <core/tnlCuda.h>

int
tnlCudaDeviceInfo::
getNumberOfDevices()
{
    int devices;
    cudaGetDeviceCount( &devices );
    return devices;
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
getClockRate( int deviceNum )
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties( &properties, deviceNum );
    return properties.clockRate;
}
      
int
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

#endif