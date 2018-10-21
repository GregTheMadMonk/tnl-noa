/***************************************************************************
                          SystemInfo.cpp  -  description
                             -------------------
    begin                : Jul 8, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <set>
#include <iomanip>
#include <cstring>
#include <ctime>

#include <unistd.h>
#include <sys/utsname.h>
#include <sys/stat.h>

#include <TNL/Devices/SystemInfo.h>
#include <TNL/Logger.h>

namespace TNL {
namespace Devices {

int SystemInfo::numberOfProcessors( 0 );
String SystemInfo::CPUModelName( "" );
int SystemInfo::CPUThreads( 0 );
int SystemInfo::CPUCores( 0 );

String
SystemInfo::getHostname( void )
{
   char host_name[ 256 ];
   gethostname( host_name, 255 );
   return String( host_name );
}

String
SystemInfo::getArchitecture( void )
{
   utsname uts;
   uname( &uts );
   return String( uts.machine );
}

String
SystemInfo::getSystemName( void )
{
   utsname uts;
   uname( &uts );
   return String( uts.sysname );
}

String
SystemInfo::getSystemRelease( void )
{
   utsname uts;
   uname( &uts );
   return String( uts.release );
}

String
SystemInfo::getCurrentTime( const char* format )
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
SystemInfo::getNumberOfProcessors( void )
{
   if( numberOfProcessors == 0 )
      parseCPUInfo();
   return numberOfProcessors;
}

String
SystemInfo::getOnlineCPUs( void )
{
   std::string online = readFile< std::string >( "/sys/devices/system/cpu/online" );
   return String( online.c_str() );
}

int
SystemInfo::getNumberOfCores( int cpu_id )
{
   if( CPUCores == 0 )
      parseCPUInfo();
   return CPUCores;
}

int
SystemInfo::getNumberOfThreads( int cpu_id )
{
   if( CPUThreads == 0 )
      parseCPUInfo();
   return CPUThreads;
}

String
SystemInfo::getCPUModelName( int cpu_id )
{
   if( CPUModelName == "" )
      parseCPUInfo();
   return CPUModelName;
}

int
SystemInfo::getCPUMaxFrequency( int cpu_id )
{
   String fileName( "/sys/devices/system/cpu/cpu" );
   fileName += convertToString( cpu_id ) + "/cpufreq/cpuinfo_max_freq";
   return readFile< int >( fileName );
}

CacheSizes
SystemInfo::getCPUCacheSizes( int cpu_id )
{
   String directory( "/sys/devices/system/cpu/cpu" );
   directory += convertToString( cpu_id ) + "/cache";

   CacheSizes sizes;
   for( int i = 0; i <= 3; i++ ) {
      const String cache = directory + "/index" + convertToString( i );

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
SystemInfo::
writeDeviceInfo( Logger& logger )
{
// compiler detection macros:
// http://nadeausoftware.com/articles/2012/10/c_c_tip_how_detect_compiler_name_and_version_using_compiler_predefined_macros
// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#compilation-phases
#if defined(__NVCC__)
   #define TNL_STRINGIFY(x) #x
   const char* compiler_name = "Nvidia NVCC (" TNL_STRINGIFY(__CUDACC_VER_MAJOR__) "." TNL_STRINGIFY(__CUDACC_VER_MINOR__) "." TNL_STRINGIFY(__CUDACC_VER_BUILD__) ")";
   #undef TNL_STRINGIFY
#elif defined(__clang__)
   const char* compiler_name = "Clang/LLVM (" __VERSION__ ")";
#elif defined(__ICC) || defined(__INTEL_COMPILER)
   const char* compiler_name = "Intel ICPC (" __VERSION__ ")";
#elif defined(__GNUC__) || defined(__GNUG__)
   const char* compiler_name = "GNU G++ (" __VERSION__ ")";
#else
   const char* compiler_name = "(unknown)";
#endif

   logger.writeParameter< String >( "Host name:", getHostname() );
   logger.writeParameter< String >( "System:", getSystemName() );
   logger.writeParameter< String >( "Release:", getSystemRelease() );
   logger.writeParameter< String >( "Architecture:", getArchitecture() );
   logger.writeParameter< String >( "TNL compiler:", compiler_name );
   // FIXME: generalize for multi-socket systems, here we consider only the first found CPU
   const int cpu_id = 0;
   const int threads = getNumberOfThreads( cpu_id );
   const int cores = getNumberOfCores( cpu_id );
   int threadsPerCore = 0;
   if( cores > 0 )
      threadsPerCore = threads / cores;
   logger.writeParameter< String >( "CPU info", String("") );
   logger.writeParameter< String >( "Model name:", getCPUModelName( cpu_id ), 1 );
   logger.writeParameter< int >( "Cores:", cores, 1 );
   logger.writeParameter< int >( "Threads per core:", threadsPerCore, 1 );
   logger.writeParameter< float >( "Max clock rate (in MHz):", getCPUMaxFrequency( cpu_id ) / 1000, 1 );
   CacheSizes cacheSizes = getCPUCacheSizes( cpu_id );
   String cacheInfo = convertToString( cacheSizes.L1data ) + ", "
                       + convertToString( cacheSizes.L1instruction ) + ", "
                       + convertToString( cacheSizes.L2 ) + ", "
                       + convertToString( cacheSizes.L3 );
   logger.writeParameter< String >( "Cache (L1d, L1i, L2, L3):", cacheInfo, 1 );
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


size_t SystemInfo::getFreeMemory()
{
   long pages = sysconf(_SC_PHYS_PAGES);
   long page_size = sysconf(_SC_PAGE_SIZE);
   return pages * page_size;
}

} // namespace Devices
} // namespace TNL
