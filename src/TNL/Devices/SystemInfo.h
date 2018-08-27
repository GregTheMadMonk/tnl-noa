/***************************************************************************
                          SystemInfo.h  -  description
                             -------------------
    begin                : Jul 8, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <fstream>

#include <TNL/String.h>

namespace TNL {

class Logger;

namespace Devices {

struct CacheSizes {
   int L1instruction = 0;
   int L1data = 0;
   int L2 = 0;
   int L3 = 0;
};

class SystemInfo
{
public:
   static String getHostname();
   static String getArchitecture();
   static String getSystemName();
   static String getSystemRelease();
   static String getCurrentTime( const char* format = "%a %b %d %Y, %H:%M:%S" );

   static int    getNumberOfProcessors();
   static String getOnlineCPUs();
   static int    getNumberOfCores( int cpu_id );
   static int    getNumberOfThreads( int cpu_id );
   static String getCPUModelName( int cpu_id );
   static int    getCPUMaxFrequency( int cpu_id );
   static CacheSizes getCPUCacheSizes( int cpu_id );
   static size_t getFreeMemory();

   static void writeDeviceInfo( Logger& logger );

protected:
   static int numberOfProcessors;
   static String CPUModelName;
   static int CPUThreads;
   static int CPUCores;

   static void parseCPUInfo();

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
};

} // namespace Devices
} // namespace TNL
