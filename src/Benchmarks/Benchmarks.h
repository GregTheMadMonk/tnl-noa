/***************************************************************************
                          benchmarks.h  -  description
                             -------------------
    begin                : Dec 30, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <exception>
#include <limits>

#include <TNL/Timer.h>
#include <TNL/String.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/SystemInfo.h>
#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Communicators/MpiCommunicator.h>

namespace TNL {
namespace Benchmarks {

const double oneGB = 1024.0 * 1024.0 * 1024.0;

template< typename ComputeFunction,
          typename ResetFunction,
          typename Monitor = TNL::Solvers::IterativeSolverMonitor< double, int > >
double
timeFunction( ComputeFunction compute,
              ResetFunction reset,
              int loops,
              Monitor && monitor = Monitor() )
{
   // the timer is constructed zero-initialized and stopped
   Timer timer;

   // set timer to the monitor
   monitor.setTimer( timer );

   // warm up
   reset();
   compute();

   for(int i = 0; i < loops; ++i) {
      // abuse the monitor's "time" for loops
      monitor.setTime( i + 1 );

      reset();

      // Explicit synchronization of the CUDA device
      // TODO: not necessary for host computations
#ifdef HAVE_CUDA
      cudaDeviceSynchronize();
#endif
      timer.start();
      compute();
#ifdef HAVE_CUDA
      cudaDeviceSynchronize();
#endif
      timer.stop();
   }

   return timer.getRealTime();
}


class Logging
{
public:
   using MetadataElement = std::pair< const char*, String >;
   using MetadataMap = std::map< const char*, String >;
   using MetadataColumns = std::vector<MetadataElement>;

   using HeaderElements = std::initializer_list< String >;
   using RowElements = std::initializer_list< double >;

   Logging( bool verbose = true )
   : verbose(verbose)
   {}

   void
   writeTitle( const String & title )
   {
      if( verbose )
         std::cout << std::endl << "== " << title << " ==" << std::endl << std::endl;
      log << ": title = " << title << std::endl;
   }

   void
   writeMetadata( const MetadataMap & metadata )
   {
      if( verbose )
         std::cout << "properties:" << std::endl;

      for( auto & it : metadata ) {
         if( verbose )
            std::cout << "   " << it.first << " = " << it.second << std::endl;
         log << ": " << it.first << " = " << it.second << std::endl;
      }
      if( verbose )
         std::cout << std::endl;
   }

   void
   writeTableHeader( const String & spanningElement,
                     const HeaderElements & subElements )
   {
      using namespace std;

      if( verbose && header_changed ) {
         for( auto & it : metadataColumns ) {
            std::cout << std::setw( 20 ) << it.first;
         }

         // spanning element is printed as usual column to stdout,
         // but is excluded from header
         std::cout << std::setw( 15 ) << "";

         for( auto & it : subElements ) {
            std::cout << std::setw( 15 ) << it;
         }
         std::cout << std::endl;

         header_changed = false;
      }

      // initial indent string
      header_indent = "!";
      log << std::endl;
      for( auto & it : metadataColumns ) {
         log << header_indent << " " << it.first << std::endl;
      }

      // dump stacked spanning columns
      if( horizontalGroups.size() > 0 )
         while( horizontalGroups.back().second <= 0 ) {
            horizontalGroups.pop_back();
            header_indent.pop_back();
         }
      for( size_t i = 0; i < horizontalGroups.size(); i++ ) {
         if( horizontalGroups[ i ].second > 0 ) {
            log << header_indent << " " << horizontalGroups[ i ].first << std::endl;
            header_indent += "!";
         }
      }

      log << header_indent << " " << spanningElement << std::endl;
      for( auto & it : subElements ) {
         log << header_indent << "! " << it << std::endl;
      }

      if( horizontalGroups.size() > 0 ) {
         horizontalGroups.back().second--;
         header_indent.pop_back();
      }
   }

   void
   writeTableRow( const String & spanningElement,
                  const RowElements & subElements )
   {
      using namespace std;

      if( verbose ) {
         for( auto & it : metadataColumns ) {
            std::cout << std::setw( 20 ) << it.second;
         }
         // spanning element is printed as usual column to stdout
         std::cout << std::setw( 15 ) << spanningElement;
         for( auto & it : subElements ) {
            std::cout << std::setw( 15 );
            if( it != 0.0 )std::cout << it;
            else std::cout << "N/A";
         }
         std::cout << std::endl;
      }

      // only when changed (the header has been already adjusted)
      // print each element on separate line
      for( auto & it : metadataColumns ) {
         log << it.second << std::endl;
      }

      // benchmark data are indented
      const String indent = "    ";
      for( auto & it : subElements ) {
         if( it != 0.0 ) log << indent << it << std::endl;
         else log << indent << "N/A" << std::endl;
      }
   }

   void
   writeErrorMessage( const char* msg,
                      int colspan = 1 )
   {
      // initial indent string
      header_indent = "!";
      log << std::endl;
      for( auto & it : metadataColumns ) {
         log << header_indent << " " << it.first << std::endl;
      }

      // make sure there is a header column for the message
      if( horizontalGroups.size() == 0 )
         horizontalGroups.push_back( {"", 1} );

      // dump stacked spanning columns
      while( horizontalGroups.back().second <= 0 ) {
         horizontalGroups.pop_back();
         header_indent.pop_back();
      }
      for( size_t i = 0; i < horizontalGroups.size(); i++ ) {
         if( horizontalGroups[ i ].second > 0 ) {
            log << header_indent << " " << horizontalGroups[ i ].first << std::endl;
            header_indent += "!";
         }
      }
      if( horizontalGroups.size() > 0 ) {
         horizontalGroups.back().second -= colspan;
         header_indent.pop_back();
      }

      // only when changed (the header has been already adjusted)
      // print each element on separate line
      for( auto & it : metadataColumns ) {
         log << it.second << std::endl;
      }
      log << msg << std::endl;
   }

   void
   closeTable()
   {
      log << std::endl;
      header_indent = body_indent = "";
      header_changed = true;
      horizontalGroups.clear();
   }

   bool save( std::ostream & logFile )
   {
      closeTable();
      logFile << log.str();
      if( logFile.good() ) {
         log.str() = "";
         return true;
      }
      return false;
   }

protected:

   // manual double -> String conversion with fixed precision
   static String
   _to_string( double num, int precision = 0, bool fixed = false )
   {
      std::stringstream str;
      if( fixed )
         str << std::fixed;
      if( precision )
         str << std::setprecision( precision );
      str << num;
      return String( str.str().data() );
   }

   std::stringstream log;
   std::string header_indent;
   std::string body_indent;

   bool verbose;
   MetadataColumns metadataColumns;
   bool header_changed = true;
   std::vector< std::pair< String, int > > horizontalGroups;
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

   // TODO: ensure that this is not called in the middle of the benchmark
   // (or just remove it completely?)
   void
   setLoops( int loops )
   {
      this->loops = loops;
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
      metadata["loops"] = String(loops);
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
   template< typename ResetFunction,
             typename ComputeFunction >
   double
   time( ResetFunction reset,
         const String & performer,
         ComputeFunction & compute )
   {
      double time;
      try {
         if( verbose ) {
            // run the monitor main loop
            Solvers::SolverMonitorThread monitor_thread( monitor );
            time = timeFunction( compute, reset, loops, monitor );
         }
         else {
            time = timeFunction( compute, reset, loops, monitor );
         }
      }
      catch ( const std::exception& e ) {
         std::cerr << "timeFunction failed due to a C++ exception with description: " << e.what() << std::endl;
         time = std::numeric_limits<double>::quiet_NaN();
      }

      const double bandwidth = datasetSize / time;
      const double speedup = this->baseTime / time;
      if( this->baseTime == 0.0 )
         this->baseTime = time;

      writeTableHeader( performer, HeaderElements({"bandwidth", "time", "speedup"}) );
      writeTableRow( performer, RowElements({ bandwidth, time, speedup }) );

      return this->baseTime;
   }

   // Recursive template function to deal with multiple computations with the
   // same reset function.
   template< typename ResetFunction,
             typename ComputeFunction,
             typename... NextComputations >
   inline double
   time( ResetFunction reset,
         const String & performer,
         ComputeFunction & compute,
         NextComputations & ... nextComputations )
   {
      time( reset, performer, compute );
      time( reset, nextComputations... );
      return this->baseTime;
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
   int loops;
   double datasetSize = 0.0;
   double baseTime = 0.0;
   Solvers::IterativeSolverMonitor< double, int > monitor;
};


Benchmark::MetadataMap getHardwareMetadata()
{
   const int cpu_id = 0;
   Devices::CacheSizes cacheSizes = Devices::SystemInfo::getCPUCacheSizes( cpu_id );
   String cacheInfo = String( cacheSizes.L1data ) + ", "
                       + String( cacheSizes.L1instruction ) + ", "
                       + String( cacheSizes.L2 ) + ", "
                       + String( cacheSizes.L3 );
#ifdef HAVE_CUDA
   const int activeGPU = Devices::CudaDeviceInfo::getActiveDevice();
   const String deviceArch = String( Devices::CudaDeviceInfo::getArchitectureMajor( activeGPU ) ) + "." +
                             String( Devices::CudaDeviceInfo::getArchitectureMinor( activeGPU ) );
#endif
   Benchmark::MetadataMap metadata {
       { "host name", Devices::SystemInfo::getHostname() },
       { "architecture", Devices::SystemInfo::getArchitecture() },
       { "system", Devices::SystemInfo::getSystemName() },
       { "system release", Devices::SystemInfo::getSystemRelease() },
       { "start time", Devices::SystemInfo::getCurrentTime() },
#ifdef HAVE_MPI
       { "number of MPI processes", Communicators::MpiCommunicator::GetSize( Communicators::MpiCommunicator::AllGroup ) },
#endif
       { "OpenMP enabled", Devices::Host::isOMPEnabled() },
       { "OpenMP threads", Devices::Host::getMaxThreadsCount() },
       { "CPU model name", Devices::SystemInfo::getCPUModelName( cpu_id ) },
       { "CPU cores", Devices::SystemInfo::getNumberOfCores( cpu_id ) },
       { "CPU threads per core", Devices::SystemInfo::getNumberOfThreads( cpu_id ) / Devices::SystemInfo::getNumberOfCores( cpu_id ) },
       { "CPU max frequency (MHz)", Devices::SystemInfo::getCPUMaxFrequency( cpu_id ) / 1e3 },
       { "CPU cache sizes (L1d, L1i, L2, L3) (kiB)", cacheInfo },
#ifdef HAVE_CUDA
       { "GPU name", Devices::CudaDeviceInfo::getDeviceName( activeGPU ) },
       { "GPU architecture", deviceArch },
       { "GPU CUDA cores", Devices::CudaDeviceInfo::getCudaCores( activeGPU ) },
       { "GPU clock rate (MHz)", (double) Devices::CudaDeviceInfo::getClockRate( activeGPU ) / 1e3 },
       { "GPU global memory (GB)", (double) Devices::CudaDeviceInfo::getGlobalMemory( activeGPU ) / 1e9 },
       { "GPU memory clock rate (MHz)", (double) Devices::CudaDeviceInfo::getMemoryClockRate( activeGPU ) / 1e3 },
       { "GPU memory ECC enabled", Devices::CudaDeviceInfo::getECCEnabled( activeGPU ) },
#endif
   };

   return metadata;
}

} // namespace Benchmarks
} // namespace TNL
