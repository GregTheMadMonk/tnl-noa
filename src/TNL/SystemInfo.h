#pragma once

#include <fstream>
#include <sstream>

#include <sys/utsname.h>

#include <TNL/String.h>

namespace TNL {

// TODO: Move this to Devices::Host
struct tnlCacheSizes {
   int L1instruction = 0;
   int L1data = 0;
   int L2 = 0;
   int L3 = 0;
};

class SystemInfo
{
public:
   SystemInfo();

   String getHostname( void ) const;
   String getArchitecture( void ) const;
   String getSystemName( void ) const;
   String getSystemRelease( void ) const;
   String getCurrentTime( const char* format = "%a %b %d %Y, %H:%M:%S" ) const;

   int getNumberOfProcessors( void ) const;
   String getOnlineCPUs( void ) const;
   int getNumberOfCores( int cpu_id ) const;
   int getNumberOfThreads( int cpu_id ) const;
   String getCPUModelName( int cpu_id ) const;
   int getCPUMaxFrequency( int cpu_id ) const;
   tnlCacheSizes getCPUCacheSizes( int cpu_id ) const;

protected:
   struct utsname uts;
   int numberOfProcessors = 0;
   String CPUModelName;
   int CPUThreads = 0;
   int CPUCores = 0;

   void parseCPUInfo( void );

   template< typename ResultType >
   ResultType
   readFile( const String & fileName ) const
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

} // namespace TNL