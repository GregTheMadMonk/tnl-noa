/***************************************************************************
                          tnlCudaDeviceInfo.cpp  -  description
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

#ifndef HAVE_CUDA

#include <core/tnlCudaDeviceInfo.h>

int
tnlCudaDeviceInfo::
getNumberOfDevices()
{
   return -1;
}
      
tnlString
tnlCudaDeviceInfo::
getDeviceName( int deviceNum )
{
   return tnlString( "" );
}
      
int
tnlCudaDeviceInfo::
getClockRate( int deviceNum )
{
   return 0;
}
      
int
tnlCudaDeviceInfo::
getGlobalMemory( int deviceNum )
{
   return 0;
}

int
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

#endif