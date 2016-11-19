/***************************************************************************
                          CudaDeviceInfo.h  -  description
                             -------------------
    begin                : Jun 21, 2015
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <stdlib.h>

#include <TNL/String.h>

namespace TNL {
namespace Devices {

class CudaDeviceInfo
{
   public:

      static int getNumberOfDevices();

      static int getActiveDevice();

      static String getDeviceName( int deviceNum );

      static int getArchitectureMajor( int deviceNum );

      static int getArchitectureMinor( int deviceNum );

      static int getClockRate( int deviceNum );

      static size_t getGlobalMemory( int deviceNum );

      static size_t getFreeGlobalMemory();

      static int getMemoryClockRate( int deviceNum );

      static bool getECCEnabled( int deviceNum );

      static int getCudaMultiprocessors( int deviceNum );

      static int getCudaCoresPerMultiprocessors( int deviceNum );

      static int getCudaCores( int deviceNum );

};

} // namespace Devices
} // namespace TNL

