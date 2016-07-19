/***************************************************************************
                          tnlHost.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <unistd.h>
#include <core/tnlDevice.h>
#include <core/tnlString.h>

namespace TNL {

class tnlConfigDescription;
class tnlParameterContainer;

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
 
      static void disableOMP();
 
      static void enableOMP();
 
      static inline bool isOMPEnabled() { return ompEnabled; };
 
      static void setMaxThreadsCount( int maxThreadsCount );
 
      static int getMaxThreadsCount();
 
      static int getThreadIdx();
 
      static void configSetup( tnlConfigDescription& config, const tnlString& prefix = "" );
 
      static bool setup( const tnlParameterContainer& parameters,
                         const tnlString& prefix = "" );

   protected:
 
      static bool ompEnabled;
 
      static int maxThreadsCount;


};

} // namespace TNL
