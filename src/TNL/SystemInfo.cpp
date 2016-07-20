#include <set>
#include <iomanip>
#include <cstring>
#include <ctime>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "SystemInfo.h"

namespace TNL {

SystemInfo::SystemInfo()
{
   uname( &uts );
   parseCPUInfo();
}

void
SystemInfo::parseCPUInfo( void )
{
   std::ifstream file( "/proc/cpuinfo" );
   if( ! file ) {
      std::cerr << "Unable to read information from /proc/cpuinfo." << std::endl;
      return;
   }

   char line[ 1024 ];
   std::set< int > processors;
   while( ! file. eof() )
   {
      int i;
      file.getline( line, 1024 );
      if( strncmp( line, "physical id", strlen( "physical id" ) ) == 0 )
      {
         i = strlen( "physical id" );
         while( line[ i ] != ':' && line[ i ] ) i ++;
         processors.insert( atoi( &line[ i + 1 ] ) );
         continue;
      }
      // FIXME: the rest does not work on heterogeneous multi-socket systems
      if( strncmp( line, "model name", strlen( "model name" ) ) == 0 )
      {
         i = strlen( "model name" );
         while( line[ i ] != ':' && line[ i ] ) i ++;
         CPUModelName.setString( &line[ i + 1 ] );
         continue;
      }
      if( strncmp( line, "cpu cores", strlen( "cpu cores" ) ) == 0 )
      {
         i = strlen( "cpu MHz" );
         while( line[ i ] != ':' && line[ i ] ) i ++;
         CPUCores = atoi( &line[ i + 1 ] );
         continue;
      }
      if( strncmp( line, "siblings", strlen( "siblings" ) ) == 0 )
      {
         i = strlen( "siblings" );
         while( line[ i ] != ':' && line[ i ] ) i ++;
         CPUThreads = atoi( &line[ i + 1 ] );
      }
   }
   numberOfProcessors = processors.size();
}

String
SystemInfo::getHostname( void ) const
{
   char host_name[ 256 ];
   gethostname( host_name, 255 );
   return String( host_name );
}

String
SystemInfo::getArchitecture( void ) const
{
   return String( uts.machine );
}

String
SystemInfo::getSystemName( void ) const
{
   return String( uts.sysname );
}

String
SystemInfo::getSystemRelease( void ) const
{
   return String( uts.release );
}

String
SystemInfo::getCurrentTime( const char* format ) const
{
   const std::time_t time_since_epoch = std::time( nullptr );
   std::tm* localtime = std::localtime( &time_since_epoch );
   // TODO: use std::put_time in the future (available since GCC 5)
//   std::stringstream ss;
//   ss << std::put_time( localtime, format );
//   return String( ss.str().c_str() );
   char buffer[1024];
   std::strftime( buffer, 1024, format, localtime );
   return String( buffer );
}

int
SystemInfo::getNumberOfProcessors( void ) const
{
   return numberOfProcessors;
}

String
SystemInfo::getOnlineCPUs( void ) const
{
   std::string online = readFile< std::string >( "/sys/devices/system/cpu/online" );
   return String( online.c_str() );
}

int
SystemInfo::getNumberOfCores( int cpu_id ) const
{
   return CPUCores;
}

int
SystemInfo::getNumberOfThreads( int cpu_id ) const
{
   return CPUThreads;
}

String
SystemInfo::getCPUModelName( int cpu_id ) const
{
   std::cout << "model name = " << CPUModelName << std::endl;
   return CPUModelName;
}

int
SystemInfo::getCPUMaxFrequency( int cpu_id ) const
{
   String fileName( "/sys/devices/system/cpu/cpu" );
   fileName += String( cpu_id ) + "/cpufreq/cpuinfo_max_freq";
   return readFile< int >( fileName );
}

tnlCacheSizes
SystemInfo::getCPUCacheSizes( int cpu_id ) const
{
   String directory( "/sys/devices/system/cpu/cpu" );
   directory += String( cpu_id ) + "/cache";

   tnlCacheSizes sizes;
   for( int i = 0; i <= 3; i++ ) {
      const String cache = directory + "/index" + String( i );

      // check if the directory exists
      struct stat st;
      if( stat( cache.getString(), &st ) != 0 || ! S_ISDIR( st.st_mode ) )
         break;

      const int level = readFile< int >( cache + "/level" );
      const std::string type = readFile< std::string >( cache + "/type" );
      const int size = readFile< int >( cache + "/size" );

      if( level == 1 && type == "Instruction" )
         sizes.L1instruction = size;
      else if( level == 1 && type == "Data" )
         sizes.L1data = size;
      else if( level == 2 )
         sizes.L2 = size;
      else if( level == 3 )
         sizes.L3 = size;
   }
   return sizes;
}

} // namespace TNL
