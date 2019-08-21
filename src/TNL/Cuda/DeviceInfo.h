/***************************************************************************
                          CudaDeviceInfo.h  -  description
                             -------------------
    begin                : Jun 21, 2015
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>

namespace TNL {
namespace Cuda {

struct DeviceInfo
{
   static int getNumberOfDevices();

   static int getActiveDevice();

   static String getDeviceName( int deviceNum );

   static int getArchitectureMajor( int deviceNum );

   static int getArchitectureMinor( int deviceNum );

   static int getClockRate( int deviceNum );

   static std::size_t getGlobalMemory( int deviceNum );

   static std::size_t getFreeGlobalMemory();

   static int getMemoryClockRate( int deviceNum );

   static bool getECCEnabled( int deviceNum );

   static int getCudaMultiprocessors( int deviceNum );

   static int getCudaCoresPerMultiprocessors( int deviceNum );

   static int getCudaCores( int deviceNum );

   static int getRegistersPerMultiprocessor( int deviceNum );
};

} // namespace Cuda
} // namespace TNL

#include <TNL/Cuda/DeviceInfo.hpp>
