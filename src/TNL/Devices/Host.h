/***************************************************************************
                          Host.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>

namespace TNL {

namespace Config {
   class ConfigDescription;
   class ParameterContainer;
}

namespace Devices {

class Host
{
   public:

      static String getDeviceType();

      static void disableOMP();

      static void enableOMP();

      static inline bool isOMPEnabled()
      {
         // This MUST stay in the header since we are interested in whether the
         // client was compiled with OpenMP support, not the libtnl.so file.
         // Also, keeping it in the header makes it inline-able.
#ifdef HAVE_OPENMP
         return ompEnabled;
#else
         return false;
#endif
      }

      static void setMaxThreadsCount( int maxThreadsCount );

      static int getMaxThreadsCount();

      static int getThreadIdx();

      static void configSetup( Config::ConfigDescription& config, const String& prefix = "" );

      static bool setup( const Config::ParameterContainer& parameters,
                         const String& prefix = "" );

   protected:

      static bool ompEnabled;

      static int maxThreadsCount;
};

} // namespace Devices
} // namespace TNL
