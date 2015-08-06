/***************************************************************************
                          tnlHost.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLHOST_H_
#define TNLHOST_H_

#include <unistd.h>
#include <core/tnlDevice.h>
#include <core/tnlString.h>

class tnlHost
{
   public:

   enum { DeviceType = tnlHostDevice };

   static tnlString getDeviceType();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   static inline tnlDeviceEnum getDevice() { return tnlHostDevice; };

   static size_t getFreeMemory();
};

#endif /* TNLHOST_H_ */
