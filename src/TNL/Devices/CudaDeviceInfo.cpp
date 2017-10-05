/***************************************************************************
                          Devices::CudaDeviceInfo.cpp  -  description
                             -------------------
    begin                : Jun 21, 2015
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef HAVE_CUDA

#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Logger.h>

namespace TNL {
namespace Devices {   

int
CudaDeviceInfo::
getNumberOfDevices()
{
   return -1;
}

int
CudaDeviceInfo::
getActiveDevice()
{
   return -1;
}

String
CudaDeviceInfo::
getDeviceName( int deviceNum )
{
   return String( "" );
}

int
CudaDeviceInfo::
getArchitectureMajor( int deviceNum )
{
    return 0;
}

int
CudaDeviceInfo::
getArchitectureMinor( int deviceNum )
{
    return 0;
}

int
CudaDeviceInfo::
getClockRate( int deviceNum )
{
   return 0;
}

size_t
CudaDeviceInfo::
getGlobalMemory( int deviceNum )
{
   return 0;
}

size_t
CudaDeviceInfo::
getFreeGlobalMemory()
{
   return 0;
}

int
CudaDeviceInfo::
getMemoryClockRate( int deviceNum )
{
   return 0;
}

bool
CudaDeviceInfo::
getECCEnabled( int deviceNum )
{
   return 0;
}

int
CudaDeviceInfo::
getCudaMultiprocessors( int deviceNum )
{
   return 0;
}

int
CudaDeviceInfo::
getCudaCoresPerMultiprocessors( int deviceNum )
{
   return 0;
}

int
CudaDeviceInfo::
getCudaCores( int deviceNum )
{
   return 0;
}

int
CudaDeviceInfo::
getRegistersPerMultiprocessor( int deviceNum )
{
   return 0;
}

void
CudaDeviceInfo::
writeDeviceInfo( Logger& logger )
{
}

} // namespace Devices
} // namespace TNL

#endif
