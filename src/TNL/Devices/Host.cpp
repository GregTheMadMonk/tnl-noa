/***************************************************************************
                          Host.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <set>
#include <iomanip>
#include <cstring>
#include <ctime>

#include <sys/utsname.h>
#include <sys/stat.h>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include <TNL/Devices/Host.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Devices {   

int Host::numberOfProcessors( 0 );
String Host::CPUModelName( "" );
int Host::CPUThreads( 0 );
int Host::CPUCores( 0 );
bool Host::ompEnabled( true );
int Host::maxThreadsCount( -1 );

String Host::getDeviceType()
{
   return String( "Devices::Host" );
};


String
Host::getHostname( void )
{
   char host_name[ 256 ];
   gethostname( host_name, 255 );
   return String( host_name );
}

String
Host::getArchitecture( void )
{
   utsname uts;
   uname( &uts );
   return String( uts.machine );
}

String
Host::getSystemName( void )
{
   utsname uts;
   uname( &uts );
   return String( uts.sysname );
}

String
Host::getSystemRelease( void )
{
   utsname uts;
   uname( &uts );
   return String( uts.release );
}

String
Host::getCurrentTime( const char* format )
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
Host::getNumberOfProcessors( void )
{
   return numberOfProcessors;
}

String
Host::getOnlineCPUs( void )
{
   std::string online = readFile< std::string >( "/sys/devices/system/cpu/online" );
   return String( online.c_str() );
}

int
Host::getNumberOfCores( int cpu_id )
{
   return CPUCores;
}

int
Host::getNumberOfThreads( int cpu_id )
{
   return CPUThreads;
}

String
Host::getCPUModelName( int cpu_id )
{
   return CPUModelName;
}

int
Host::getCPUMaxFrequency( int cpu_id )
{
   String fileName( "/sys/devices/system/cpu/cpu" );
   fileName += String( cpu_id ) + "/cpufreq/cpuinfo_max_freq";
   return readFile< int >( fileName );
}

CacheSizes
Host::getCPUCacheSizes( int cpu_id )
{
   String directory( "/sys/devices/system/cpu/cpu" );
   directory += String( cpu_id ) + "/cache";

   CacheSizes sizes;
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

void
Host::parseCPUInfo( void )
{
   std::ifstream file( "/proc/cpuinfo" );
   if( ! file ) {
      std::cerr << "Unable to read information from /proc/cpuinfo." << std::endl;
      return;
   }

   char line[ 1024 ];
   std::set< int > processors;
   while( ! file.eof() )
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


size_t Host::getFreeMemory()
{
   long pages = sysconf(_SC_PHYS_PAGES);
   long page_size = sysconf(_SC_PAGE_SIZE);
   return pages * page_size;
};

void Host::enableOMP()
{
   ompEnabled = true;
}

void Host::disableOMP()
{
   ompEnabled = false;
}

void Host::setMaxThreadsCount( int maxThreadsCount_ )
{
   maxThreadsCount = maxThreadsCount_;
#ifdef HAVE_OPENMP
   omp_set_num_threads( maxThreadsCount );
#endif
}

int Host::getMaxThreadsCount()
{
#ifdef HAVE_OPENMP
   if( maxThreadsCount == -1 )
      return omp_get_max_threads();
   return maxThreadsCount;
#else
   return 0;
#endif
}
 
int Host::getThreadIdx()
{
#ifdef HAVE_OPENMP
   return omp_get_thread_num();
#else
   return 0;
#endif
}

void Host::configSetup( Config::ConfigDescription& config, const String& prefix )
{
#ifdef HAVE_OPENMP
   config.addEntry< bool >( prefix + "openmp-enabled", "Enable support of OpenMP.", true );
   config.addEntry<  int >( prefix + "openmp-max-threads", "Set maximum number of OpenMP threads.", omp_get_max_threads() );
#else
   config.addEntry< bool >( prefix + "openmp-enabled", "Enable support of OpenMP (not supported on this system).", false );
   config.addEntry<  int >( prefix + "openmp-max-threads", "Set maximum number of OpenMP threads (not supported on this system).", 0 );
#endif
 
}
 
bool Host::setup( const Config::ParameterContainer& parameters,
                  const String& prefix )
{
   if( parameters.getParameter< bool >( prefix + "openmp-enabled" ) )
      enableOMP();
   else
      disableOMP();
   setMaxThreadsCount( parameters.getParameter< int >( prefix + "openmp-max-threads" ) );
   return true;
}

} // namespace Devices
} // namespace TNL

