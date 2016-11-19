/***************************************************************************
                          Host.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <fstream>

#include <TNL/String.h>

namespace TNL {

namespace Config { 
   class ConfigDescription;
   class ParameterContainer;
}

namespace Devices {

struct CacheSizes {
   int L1instruction = 0;
   int L1data = 0;
   int L2 = 0;
   int L3 = 0;
};

class Host
{
   public:

      static String getDeviceType();

      static String getHostname( void );
      static String getArchitecture( void );
      static String getSystemName( void );
      static String getSystemRelease( void );
      static String getCurrentTime( const char* format = "%a %b %d %Y, %H:%M:%S" );

      static int    getNumberOfProcessors( void );
      static String getOnlineCPUs( void );
      static int    getNumberOfCores( int cpu_id );
      static int    getNumberOfThreads( int cpu_id );
      static String getCPUModelName( int cpu_id );
      static int    getCPUMaxFrequency( int cpu_id );
      static CacheSizes getCPUCacheSizes( int cpu_id );

      static size_t getFreeMemory();
 
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

      static int numberOfProcessors;
      static String CPUModelName;
      static int CPUThreads;
      static int CPUCores;
   
      static void parseCPUInfo( void );

      template< typename ResultType >
      static ResultType
      readFile( const String & fileName )
      {
         std::ifstream file( fileName.getString() );
         if( ! file ) {
            std::cerr << "Unable to read information from " << fileName << "." << std::endl;
            return 0;
         }
         ResultType result;
         file >> result;
         return result;
      }
 
      static bool ompEnabled;
 
      static int maxThreadsCount;
};

} // namespace Devices
} // namespace TNL
