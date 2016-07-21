/***************************************************************************
                          tnlCudaDeviceInfo.cpp  -  description
                             -------------------
    begin                : Jun 21, 2015
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef HAVE_CUDA

#include <TNL/core/tnlCudaDeviceInfo.h>

namespace TNL {

int
tnlCudaDeviceInfo::
getNumberOfDevices()
{
   return -1;
}

int
tnlCudaDeviceInfo::
getActiveDevice()
{
   return -1;
}

String
tnlCudaDeviceInfo::
getDeviceName( int deviceNum )
{
   return String( "" );
}

int
tnlCudaDeviceInfo::
getArchitectureMajor( int deviceNum )
{
    return 0;
}

int
tnlCudaDeviceInfo::
getArchitectureMinor( int deviceNum )
{
    return 0;
}

int
tnlCudaDeviceInfo::
getClockRate( int deviceNum )
{
   return 0;
}

size_t
tnlCudaDeviceInfo::
getGlobalMemory( int deviceNum )
{
   return 0;
}

int
tnlCudaDeviceInfo::
getMemoryClockRate( int deviceNum )
{
   return 0;
}

bool
tnlCudaDeviceInfo::
getECCEnabled( int deviceNum )
{
   return 0;
}

int
tnlCudaDeviceInfo::
getCudaMultiprocessors( int deviceNum )
{
   return 0;
}

int
tnlCudaDeviceInfo::
getCudaCoresPerMultiprocessors( int deviceNum )
{
   return 0;
}

int
tnlCudaDeviceInfo::
getCudaCores( int deviceNum )
{
   return 0;
}

} // namespace TNL

#endif
