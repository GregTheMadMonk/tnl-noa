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
#include <iomanip>
#include <exception>
#include <limits>

#include <TNL/String.h>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/SystemInfo.h>
#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Communicators/MpiCommunicator.h>

namespace TNL {
namespace Benchmarks {

const double oneGB = 1024.0 * 1024.0 * 1024.0;



struct BenchmarkResult
{
   using HeaderElements = Logging::HeaderElements;
   using RowElements = Logging::RowElements;

   double bandwidth = std::numeric_limits<double>::quiet_NaN();
   double time = std::numeric_limits<double>::quiet_NaN();
   double speedup = std::numeric_limits<double>::quiet_NaN();

   virtual HeaderElements getTableHeader() const
   {
      return HeaderElements({"bandwidth", "time", "speedup"});
   }

   virtual RowElements getRowElements() const
   {
      return RowElements({ bandwidth, time, speedup });
   }
};


class Benchmark
: protected Logging
{
public:
   using Logging::MetadataElement;
   using Logging::MetadataMap;
   using Logging::MetadataColumns;
   
   Benchmark( int loops = 10,
              bool verbose = true )
   : Logging(verbose), loops(loops)
   {}
   
   static void configSetup( Config::ConfigDescription& config )
   {
      config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
      config.addEntry< double >( "min-time", "Minimal real time in seconds for every computation.", 1 );
      config.addEntry< int >( "verbose", "Verbose mode, the higher number the more verbosity.", 1 );
   }

   void setup( const Config::ParameterContainer& parameters )
   {
      this->loops = parameters.getParameter< unsigned >( "loops" );
      this->minTime = parameters.getParameter< double >( "min-time" );
      const int verbose = parameters.getParameter< unsigned >( "verbose" );
      Logging::setVerbose( verbose );
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
      closeTable();
      writeTitle( title );
   }

   // Marks the start of a new benchmark (with custom metadata)
   void
   newBenchmark( const String & title,
                 MetadataMap metadata )
   {
      closeTable();
      writeTitle( title );
      // add loops to metadata
      metadata["loops"] = convertToString(loops);
      writeMetadata( metadata );
   }

   // Sets metadata columns -- values used for all subsequent rows until
   // the next call to this function.
   void
   setMetadataColumns( const MetadataColumns & metadata )
   {
      if( metadataColumns != metadata )
         header_changed = true;
      metadataColumns = metadata;
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
      if( metadataColumns.size() > 0 && String(metadataColumns[ 0 ].first) == "operation" ) {
         metadataColumns[ 0 ].second = operation;
      }
      else {
         metadataColumns.insert( metadataColumns.begin(), {"operation", operation} );
      }
      setOperation( datasetSize, baseTime );
      header_changed = true;
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
      if( horizontalGroups.size() == 0 ) {
         horizontalGroups.push_back( {name, subcolumns} );
      }
      else {
         auto & last = horizontalGroups.back();
         if( last.first != name && last.second > 0 ) {
            horizontalGroups.push_back( {name, subcolumns} );
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
      try {
         if( verbose > 1 ) {
            // run the monitor main loop
            Solvers::SolverMonitorThread monitor_thread( monitor );
            result.time = FunctionTimer< Device >::timeFunction( compute, reset, loops, minTime, verbose, monitor );
         }
         else {
            result.time = FunctionTimer< Device >::timeFunction( compute, reset, loops, minTime, verbose, monitor );
         }
      }
      catch ( const std::exception& e ) {
         std::cerr << "timeFunction failed due to a C++ exception with description: " << e.what() << std::endl;
      }

      result.bandwidth = datasetSize / result.time;
      result.speedup = this->baseTime / result.time;
      if( this->baseTime == 0.0 )
         this->baseTime = result.time;

      writeTableHeader( performer, result.getTableHeader() );
      writeTableRow( performer, result.getRowElements() );

      return this->baseTime;
   }

   template< typename Device, 
             typename ResetFunction,
             typename ComputeFunction,
             typename... NextComputations >
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
      try {
         if( verbose > 1 ) {
            // run the monitor main loop
            Solvers::SolverMonitorThread monitor_thread( monitor );
            result.time = FunctionTimer< Device >::timeFunction( compute, loops, minTime, verbose, monitor );
         }
         else {
            result.time = FunctionTimer< Device >::timeFunction( compute, loops, minTime, verbose, monitor );
         }
      }
      catch ( const std::exception& e ) {
         std::cerr << "timeFunction failed due to a C++ exception with description: " << e.what() << std::endl;
      }

      result.bandwidth = datasetSize / result.time;
      result.speedup = this->baseTime / result.time;
      if( this->baseTime == 0.0 )
         this->baseTime = result.time;

      writeTableHeader( performer, result.getTableHeader() );
      writeTableRow( performer, result.getRowElements() );

      return this->baseTime;
   }

   template< typename Device, 
             typename ComputeFunction,
             typename... NextComputations >
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
                    int numberOfComputations = 1 )
   {
      // each computation has 3 subcolumns
      const int colspan = 3 * numberOfComputations;
      writeErrorMessage( msg, colspan );
   }

   using Logging::save;

   Solvers::IterativeSolverMonitor< double, int >&
   getMonitor()
   {
      return monitor;
   }

protected:
   int loops = 1;
   double minTime = 1;
   double datasetSize = 0.0;
   double baseTime = 0.0;
   Solvers::IterativeSolverMonitor< double, int > monitor;
};


Benchmark::MetadataMap getHardwareMetadata()
{
   const int cpu_id = 0;
   Devices::CacheSizes cacheSizes = Devices::SystemInfo::getCPUCacheSizes( cpu_id );
   String cacheInfo = convertToString( cacheSizes.L1data ) + ", "
                       + convertToString( cacheSizes.L1instruction ) + ", "
                       + convertToString( cacheSizes.L2 ) + ", "
                       + convertToString( cacheSizes.L3 );
#ifdef HAVE_CUDA
   const int activeGPU = Devices::CudaDeviceInfo::getActiveDevice();
   const String deviceArch = convertToString( Devices::CudaDeviceInfo::getArchitectureMajor( activeGPU ) ) + "." +
                             convertToString( Devices::CudaDeviceInfo::getArchitectureMinor( activeGPU ) );
#endif
   Benchmark::MetadataMap metadata {
       { "host name", Devices::SystemInfo::getHostname() },
       { "architecture", Devices::SystemInfo::getArchitecture() },
       { "system", Devices::SystemInfo::getSystemName() },
       { "system release", Devices::SystemInfo::getSystemRelease() },
       { "start time", Devices::SystemInfo::getCurrentTime() },
#ifdef HAVE_MPI
       { "number of MPI processes", convertToString( (Communicators::MpiCommunicator::IsInitialized())
                                       ? Communicators::MpiCommunicator::GetSize( Communicators::MpiCommunicator::AllGroup )
                                       : 1 ) },
#endif
       { "OpenMP enabled", convertToString( Devices::Host::isOMPEnabled() ) },
       { "OpenMP threads", convertToString( Devices::Host::getMaxThreadsCount() ) },
       { "CPU model name", Devices::SystemInfo::getCPUModelName( cpu_id ) },
       { "CPU cores", convertToString( Devices::SystemInfo::getNumberOfCores( cpu_id ) ) },
       { "CPU threads per core", convertToString( Devices::SystemInfo::getNumberOfThreads( cpu_id ) / Devices::SystemInfo::getNumberOfCores( cpu_id ) ) },
       { "CPU max frequency (MHz)", convertToString( Devices::SystemInfo::getCPUMaxFrequency( cpu_id ) / 1e3 ) },
       { "CPU cache sizes (L1d, L1i, L2, L3) (kiB)", cacheInfo },
#ifdef HAVE_CUDA
       { "GPU name", Devices::CudaDeviceInfo::getDeviceName( activeGPU ) },
       { "GPU architecture", deviceArch },
       { "GPU CUDA cores", convertToString( Devices::CudaDeviceInfo::getCudaCores( activeGPU ) ) },
       { "GPU clock rate (MHz)", convertToString( (double) Devices::CudaDeviceInfo::getClockRate( activeGPU ) / 1e3 ) },
       { "GPU global memory (GB)", convertToString( (double) Devices::CudaDeviceInfo::getGlobalMemory( activeGPU ) / 1e9 ) },
       { "GPU memory clock rate (MHz)", convertToString( (double) Devices::CudaDeviceInfo::getMemoryClockRate( activeGPU ) / 1e3 ) },
       { "GPU memory ECC enabled", convertToString( Devices::CudaDeviceInfo::getECCEnabled( activeGPU ) ) },
#endif
   };

   return metadata;
}

} // namespace Benchmarks
} // namespace TNL
