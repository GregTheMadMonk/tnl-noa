/***************************************************************************
                          benchmarks.h  -  description
                             -------------------
    begin                : Dec 30, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky,
//                 Tomas Oberhuber

#pragma once

#include "FunctionTimer.h"
#include "Logging.h"

#include <iostream>
#include <exception>
#include <limits>

#include <TNL/String.h>

#include <TNL/Devices/Host.h>
#include <TNL/SystemInfo.h>
#include <TNL/Cuda/DeviceInfo.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/MPI/Wrappers.h>

namespace TNL {
namespace Benchmarks {

const double oneGB = 1024.0 * 1024.0 * 1024.0;


struct BenchmarkResult
{
   using HeaderElements = Logging::HeaderElements;
   using RowElements = Logging::RowElements;

   double time = std::numeric_limits<double>::quiet_NaN();
   double stddev = std::numeric_limits<double>::quiet_NaN();
   double bandwidth = std::numeric_limits<double>::quiet_NaN();
   double speedup = std::numeric_limits<double>::quiet_NaN();

   virtual HeaderElements getTableHeader() const
   {
      return HeaderElements({ "time", "stddev", "stddev/time", "bandwidth", "speedup" });
   }

   virtual RowElements getRowElements() const
   {
      RowElements elements;
      elements << time << stddev << stddev / time << bandwidth;
      if( speedup != 0 )
         elements << speedup;
      else
         elements << "N/A";
      return elements;
   }
};

template< typename Logger = Logging >
class Benchmark
: protected Logger
{
public:
   using typename Logger::MetadataElement;
   using typename Logger::MetadataMap;
   using typename Logger::MetadataColumns;
   using SolverMonitorType = Solvers::IterativeSolverMonitor< double, int >;

   Benchmark( int loops = 10,
              bool verbose = true )
   : Logger(verbose), loops(loops)
   {}

   static void configSetup( Config::ConfigDescription& config )
   {
      config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
      config.addEntry< bool >( "reset", "Call reset function between loops.", true );
      config.addEntry< double >( "min-time", "Minimal real time in seconds for every computation.", 0.0 );
      config.addEntry< int >( "verbose", "Verbose mode, the higher number the more verbosity.", 1 );
   }

   void setup( const Config::ParameterContainer& parameters )
   {
      this->loops = parameters.getParameter< int >( "loops" );
      this->reset = parameters.getParameter< bool >( "reset" );
      this->minTime = parameters.getParameter< double >( "min-time" );
      const int verbose = parameters.getParameter< int >( "verbose" );
      Logger::setVerbose( verbose );
   }
   // TODO: ensure that this is not called in the middle of the benchmark
   // (or just remove it completely?)
   void
   setLoops( int loops )
   {
      this->loops = loops;
   }

   void setMinTime( const double& minTime )
   {
      this->minTime = minTime;
   }

   // Marks the start of a new benchmark
   void
   newBenchmark( const String & title )
   {
      Logger::closeTable();
      Logger::writeTitle( title );
   }

   // Marks the start of a new benchmark (with custom metadata)
   void
   newBenchmark( const String & title,
                 MetadataMap metadata )
   {
      Logger::closeTable();
      Logger::writeTitle( title );
      // add loops and reset flag to metadata
      metadata["loops"] = convertToString(loops);
      metadata["reset"] = convertToString( reset );
      metadata["minimal test time"] = convertToString( minTime );
      Logger::writeMetadata( metadata );
   }

   // Sets metadata columns -- values used for all subsequent rows until
   // the next call to this function.
   void
   setMetadataColumns( const MetadataColumns & metadata )
   {
      if( Logger::metadataColumns != metadata )
         Logger::header_changed = true;
      Logger::metadataColumns = metadata;
   }

   // TODO: maybe should be renamed to createVerticalGroup and ensured that vertical and horizontal groups are not used within the same "Benchmark"
   // Sets current operation -- operations expand the table vertically
   //  - baseTime should be reset to 0.0 for most operations, but sometimes
   //    it is useful to override it
   //  - Order of operations inside a "Benchmark" does not matter, rows can be
   //    easily sorted while converting to HTML.)
   void
   setOperation( const String & operation,
                 const double datasetSize = 0.0, // in GB
                 const double baseTime = 0.0 )
   {
      monitor.setStage( operation.getString() );
      if( Logger::metadataColumns.size() > 0 && String(Logger::metadataColumns[ 0 ].first) == "operation" ) {
         Logger::metadataColumns[ 0 ].second = operation;
      }
      else {
         Logger::metadataColumns.insert( Logger::metadataColumns.begin(), {"operation", operation} );
      }
      setOperation( datasetSize, baseTime );
      Logger::header_changed = true;
   }

   void
   setOperation( const double datasetSize = 0.0,
                 const double baseTime = 0.0 )
   {
      this->datasetSize = datasetSize;
      this->baseTime = baseTime;
   }

   // Creates new horizontal groups inside a benchmark -- increases the number
   // of columns in the "Benchmark", implies column spanning.
   // (Useful e.g. for SpMV formats, different configurations etc.)
   void
   createHorizontalGroup( const String & name,
                          int subcolumns )
   {
      if( Logger::horizontalGroups.size() == 0 ) {
         Logger::horizontalGroups.push_back( {name, subcolumns} );
      }
      else {
         auto & last = Logger::horizontalGroups.back();
         if( last.first != name && last.second > 0 ) {
            Logger::horizontalGroups.push_back( {name, subcolumns} );
         }
         else {
            last.first = name;
            last.second = subcolumns;
         }
      }
   }

   // Times a single ComputeFunction. Subsequent calls implicitly split
   // the current "horizontal group" into sub-columns identified by
   // "performer", which are further split into "bandwidth", "time" and
   // "speedup" columns.
   // TODO: allow custom columns bound to lambda functions (e.g. for Gflops calculation)
   // Also terminates the recursion of the following variadic template.
   template< typename Device,
             typename ResetFunction,
             typename ComputeFunction >
   double
   time( ResetFunction reset,
         const String & performer,
         ComputeFunction & compute,
         BenchmarkResult & result )
   {
      result.time = std::numeric_limits<double>::quiet_NaN();
      result.stddev = std::numeric_limits<double>::quiet_NaN();
      FunctionTimer< Device > functionTimer;
      try {
         if( Logger::verbose > 1 ) {
            // run the monitor main loop
            Solvers::SolverMonitorThread monitor_thread( monitor );
            if( this->reset )
               std::tie( result.time, result.stddev ) = functionTimer.timeFunction( compute, reset, loops, minTime, Logger::verbose, monitor );
            else
               std::tie( result.time, result.stddev ) = functionTimer.timeFunction( compute, loops, minTime, Logger::verbose, monitor );
         }
         else {
            if( this->reset )
               std::tie( result.time, result.stddev ) = functionTimer.timeFunction( compute, reset, loops, minTime, Logger::verbose, monitor );
            else
               std::tie( result.time, result.stddev ) = functionTimer.timeFunction( compute, loops, minTime, Logger::verbose, monitor );
         }
         this->performedLoops = functionTimer.getPerformedLoops();
      }
      catch ( const std::exception& e ) {
         std::cerr << "timeFunction failed due to a C++ exception with description: " << e.what() << std::endl;
      }

      result.bandwidth = datasetSize / result.time;
      result.speedup = this->baseTime / result.time;
      if( this->baseTime == 0.0 )
         this->baseTime = result.time;

      Logger::writeTableHeader( performer, result.getTableHeader() );
      Logger::writeTableRow( performer, result.getRowElements() );

      return this->baseTime;
   }

   template< typename Device,
             typename ResetFunction,
             typename ComputeFunction >
   inline double
   time( ResetFunction reset,
         const String & performer,
         ComputeFunction & compute )
   {
      BenchmarkResult result;
      return time< Device, ResetFunction, ComputeFunction >( reset, performer, compute, result );
   }

   /****
    * The same methods as above but without reset function
    */
   template< typename Device,
             typename ComputeFunction >
   double
   time( const String & performer,
         ComputeFunction & compute,
         BenchmarkResult & result )
   {
      result.time = std::numeric_limits<double>::quiet_NaN();
      result.stddev = std::numeric_limits<double>::quiet_NaN();
      FunctionTimer< Device > functionTimer;
      try {
         if( Logger::verbose > 1 ) {
            // run the monitor main loop
            Solvers::SolverMonitorThread monitor_thread( monitor );
            std::tie( result.time, result.stddev ) = functionTimer.timeFunction( compute, loops, minTime, Logger::verbose, monitor );
         }
         else {
            std::tie( result.time, result.stddev ) = functionTimer.timeFunction( compute, loops, minTime, Logger::verbose, monitor );
         }
      }
      catch ( const std::exception& e ) {
         std::cerr << "Function timer failed due to a C++ exception with description: " << e.what() << std::endl;
      }

      result.bandwidth = datasetSize / result.time;
      result.speedup = this->baseTime / result.time;
      if( this->baseTime == 0.0 )
         this->baseTime = result.time;

      Logger::writeTableHeader( performer, result.getTableHeader() );
      Logger::writeTableRow( performer, result.getRowElements() );

      return this->baseTime;
   }

   template< typename Device,
             typename ComputeFunction >
   inline double
   time( const String & performer,
         ComputeFunction & compute )
   {
      BenchmarkResult result;
      return time< Device, ComputeFunction >( performer, compute, result );
   }

   // Adds an error message to the log. Should be called in places where the
   // "time" method could not be called (e.g. due to failed allocation).
   void
   addErrorMessage( const char* msg,
                    int numberOfComputations = 1 ) {
      // each computation has 3 subcolumns
      const int colspan = 3 * numberOfComputations;
      Logger::writeErrorMessage( msg, colspan );
      std::cerr << msg << std::endl;
   }

   using Logging::save;

   SolverMonitorType& getMonitor() {
      return monitor;
   }

   int getPerformedLoops() const {
      return this->performedLoops;
   }

   bool isResetingOn() const {
      return reset;
   }

protected:
   int loops = 1, performedLoops = 0;
   double minTime = 0.0;
   double datasetSize = 0.0;
   double baseTime = 0.0;
   bool reset = true;
   SolverMonitorType monitor;
};


template< typename Logger >
inline typename Benchmark< Logger >::MetadataMap getHardwareMetadata()
{
   const int cpu_id = 0;
   const CacheSizes cacheSizes = SystemInfo::getCPUCacheSizes( cpu_id );
   String cacheInfo = convertToString( cacheSizes.L1data ) + ", "
                       + convertToString( cacheSizes.L1instruction ) + ", "
                       + convertToString( cacheSizes.L2 ) + ", "
                       + convertToString( cacheSizes.L3 );
#ifdef HAVE_CUDA
   const int activeGPU = Cuda::DeviceInfo::getActiveDevice();
   const String deviceArch = convertToString( Cuda::DeviceInfo::getArchitectureMajor( activeGPU ) ) + "." +
                             convertToString( Cuda::DeviceInfo::getArchitectureMinor( activeGPU ) );
#endif

#ifdef HAVE_MPI
   int nproc = 1;
   // check if MPI was initialized (some benchmarks do not initialize MPI even when
   // they are built with HAVE_MPI and thus MPI::GetSize() cannot be used blindly)
   if( TNL::MPI::Initialized() )
      nproc = TNL::MPI::GetSize();
#endif

   typename Benchmark< Logger >::MetadataMap metadata {
       { "host name", SystemInfo::getHostname() },
       { "architecture", SystemInfo::getArchitecture() },
       { "system", SystemInfo::getSystemName() },
       { "system release", SystemInfo::getSystemRelease() },
       { "start time", SystemInfo::getCurrentTime() },
#ifdef HAVE_MPI
       { "number of MPI processes", convertToString( nproc ) },
#endif
       { "OpenMP enabled", convertToString( Devices::Host::isOMPEnabled() ) },
       { "OpenMP threads", convertToString( Devices::Host::getMaxThreadsCount() ) },
       { "CPU model name", SystemInfo::getCPUModelName( cpu_id ) },
       { "CPU cores", convertToString( SystemInfo::getNumberOfCores( cpu_id ) ) },
       { "CPU threads per core", convertToString( SystemInfo::getNumberOfThreads( cpu_id ) / SystemInfo::getNumberOfCores( cpu_id ) ) },
       { "CPU max frequency (MHz)", convertToString( SystemInfo::getCPUMaxFrequency( cpu_id ) / 1e3 ) },
       { "CPU cache sizes (L1d, L1i, L2, L3) (kiB)", cacheInfo },
#ifdef HAVE_CUDA
       { "GPU name", Cuda::DeviceInfo::getDeviceName( activeGPU ) },
       { "GPU architecture", deviceArch },
       { "GPU CUDA cores", convertToString( Cuda::DeviceInfo::getCudaCores( activeGPU ) ) },
       { "GPU clock rate (MHz)", convertToString( (double) Cuda::DeviceInfo::getClockRate( activeGPU ) / 1e3 ) },
       { "GPU global memory (GB)", convertToString( (double) Cuda::DeviceInfo::getGlobalMemory( activeGPU ) / 1e9 ) },
       { "GPU memory clock rate (MHz)", convertToString( (double) Cuda::DeviceInfo::getMemoryClockRate( activeGPU ) / 1e3 ) },
       { "GPU memory ECC enabled", convertToString( Cuda::DeviceInfo::getECCEnabled( activeGPU ) ) },
#endif
   };

   return metadata;
}

} // namespace Benchmarks
} // namespace TNL
