/***************************************************************************
                          Benchmarks.h  -  description
                             -------------------
    begin                : Dec 30, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky,
//                 Tomas Oberhuber

#pragma once

#include "CustomLogging.h"

#include <limits>

#include <TNL/String.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>

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
   using HeaderElements = typename Logging::HeaderElements;
   using RowElements = typename Logging::RowElements;

   double time = std::numeric_limits<double>::quiet_NaN();
   double stddev = std::numeric_limits<double>::quiet_NaN();
   double bandwidth = std::numeric_limits<double>::quiet_NaN();
   double speedup = std::numeric_limits<double>::quiet_NaN();

   virtual HeaderElements getTableHeader() const
   {
      return HeaderElements({ "time", "stddev", "stddev/time", "bandwidth", "speedup" });
   }

   virtual std::vector< int > getColumnWidthHints() const
   {
      return std::vector< int >({ 8, 8, 8, 8, 8 });
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

template< typename Logger = CustomLogging >
class Benchmark
{
   public:
      using MetadataElement = typename Logger::MetadataElement;
      using MetadataMap = typename Logger::MetadataMap;
      using MetadataColumns = typename Logger::MetadataColumns;
      using SolverMonitorType = Solvers::IterativeSolverMonitor< double, int >;

      Benchmark( int loops = 10,
               bool verbose = true,
               String outputMode = "",
               bool logFileAppend = false );

      static void configSetup( Config::ConfigDescription& config );

      void setup( const Config::ParameterContainer& parameters );

      // TODO: ensure that this is not called in the middle of the benchmark
      // (or just remove it completely?)
      void setLoops( int loops );

      void setMinTime( const double& minTime );

      // Marks the start of a new benchmark
      void newBenchmark( const String & title );

      // Marks the start of a new benchmark (with custom metadata)
      void newBenchmark( const String & title,
                        MetadataMap metadata );

      // Sets metadata columns -- values used for all subsequent rows until
      // the next call to this function.
      void setMetadataColumns( const MetadataColumns & metadata );

      // Sets the value of one metadata column -- useful for iteratively
      // changing MetadataColumns that were set using the previous method.
      void setMetadataElement( const typename MetadataColumns::value_type & element );

      // Sets the dataset size and base time for the calculations of bandwidth
      // and speedup in the benchmarks result.
      void setDatasetSize( const double datasetSize = 0.0, // in GB
                           const double baseTime = 0.0 );

      // Sets current operation -- operations expand the table vertically
      //  - baseTime should be reset to 0.0 for most operations, but sometimes
      //    it is useful to override it
      //  - Order of operations inside a "Benchmark" does not matter, rows can be
      //    easily sorted while converting to HTML.)
      void
      setOperation( const String & operation,
                    const double datasetSize = 0.0, // in GB
                    const double baseTime = 0.0 );

      // Times a single ComputeFunction. Subsequent calls implicitly split
      // the current operation into sub-columns identified by "performer",
      // which are further split into "bandwidth", "time" and "speedup" columns.
      // TODO: allow custom columns bound to lambda functions (e.g. for Gflops calculation)
      // Also terminates the recursion of the following variadic template.
      template< typename Device,
               typename ResetFunction,
               typename ComputeFunction >
      double time( ResetFunction reset,
                  const String & performer,
                  ComputeFunction & compute,
                  BenchmarkResult & result );

      template< typename Device,
               typename ResetFunction,
               typename ComputeFunction >
      inline double time( ResetFunction reset,
                        const String & performer,
                        ComputeFunction & compute );
      /*{
         BenchmarkResult result;
         return time< Device, ResetFunction, ComputeFunction >( reset, performer, compute, result );
      }*/

      /****
       * The same methods as above but without reset function
       */
      template< typename Device,
               typename ComputeFunction >
      double time( const String & performer,
                  ComputeFunction & compute,
                  BenchmarkResult & result );

      template< typename Device,
               typename ComputeFunction >
      inline double time( const String & performer,
                        ComputeFunction & compute );

      // Adds an error message to the log. Should be called in places where the
      // "time" method could not be called (e.g. due to failed allocation).
      void addErrorMessage( const char* msg );

      bool save( std::ostream& logFile );

      SolverMonitorType& getMonitor();

      int getPerformedLoops() const;

      bool isResetingOn() const;

   protected:
      Logger logger;

      int loops = 1, performedLoops = 0;

      double minTime = 0.0;

      double datasetSize = 0.0;

      double baseTime = 0.0;

      bool reset = true;

      SolverMonitorType monitor;
};


inline typename Logging::MetadataMap getHardwareMetadata()
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

   typename Logging::MetadataMap metadata {
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

#include "Benchmark.hpp"
