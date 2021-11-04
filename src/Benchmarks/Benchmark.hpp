/***************************************************************************
                          Benchmarks.hpp  -  description
                             -------------------
    begin                : Jun 7, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky,
//                 Tomas Oberhuber

#pragma once

#include "Benchmarks.h"
#include "FunctionTimer.h"

#include <iostream>
#include <exception>

namespace TNL {
namespace Benchmarks {


template< typename Logger >
Benchmark< Logger >::
Benchmark( int loops,
           bool verbose,
           String outputMode,
           bool logFileAppend )
: Logger(verbose, outputMode, logFileAppend), loops(loops)
{}

template< typename Logger >
void
Benchmark< Logger >::
configSetup( Config::ConfigDescription& config )
{
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< bool >( "reset", "Call reset function between loops.", true );
   config.addEntry< double >( "min-time", "Minimal real time in seconds for every computation.", 0.0 );
   config.addEntry< int >( "verbose", "Verbose mode, the higher number the more verbosity.", 1 );
}

template< typename Logger >
void
Benchmark< Logger >::
setup( const Config::ParameterContainer& parameters )
{
   this->loops = parameters.getParameter< int >( "loops" );
   this->reset = parameters.getParameter< bool >( "reset" );
   this->minTime = parameters.getParameter< double >( "min-time" );
   const int verbose = parameters.getParameter< int >( "verbose" );
   Logger::setVerbose( verbose );
}

template< typename Logger >
void
Benchmark< Logger >::
setLoops( int loops )
{
   this->loops = loops;
}

template< typename Logger >
void
Benchmark< Logger >::
setMinTime( const double& minTime )
{
   this->minTime = minTime;
}

template< typename Logger >
void
Benchmark< Logger >::
newBenchmark( const String & title )
{
   Logger::closeTable();
   Logger::writeTitle( title );
}

template< typename Logger >
void
Benchmark< Logger >::
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

template< typename Logger >
void
Benchmark< Logger >::
setMetadataColumns( const MetadataColumns & metadata )
{
   Logger::setMetadataColumns( metadata );
}

template< typename Logger >
void
Benchmark< Logger >::
setMetadataElement( const typename MetadataColumns::value_type & element )
{
   Logger::setMetadataElement( element );
}

template< typename Logger >
void
Benchmark< Logger >::
setDatasetSize( const double datasetSize,
                const double baseTime )
{
   this->datasetSize = datasetSize;
   this->baseTime = baseTime;
}

template< typename Logger >
void
Benchmark< Logger >::
setOperation( const String & operation,
              const double datasetSize,
              const double baseTime )
{
   monitor.setStage( operation.getString() );
   Logger::setMetadataElement( {"operation", operation}, 0 );
   setDatasetSize( datasetSize, baseTime );
}

template< typename Logger >
   template< typename Device,
             typename ResetFunction,
             typename ComputeFunction >
double
Benchmark< Logger >::
time( ResetFunction reset,
      const String & performer,
      ComputeFunction & compute,
      BenchmarkResult & result )
{
   result.time = std::numeric_limits<double>::quiet_NaN();
   result.stddev = std::numeric_limits<double>::quiet_NaN();
   FunctionTimer< Device > functionTimer;

   // run the monitor main loop
   Solvers::SolverMonitorThread monitor_thread( monitor );
   if( Logger::verbose <= 1 )
      // stop the main loop when not verbose
      monitor.stopMainLoop();

   try {
      if( this->reset )
         std::tie( result.time, result.stddev ) = functionTimer.timeFunction( compute, reset, loops, minTime, monitor );
      else
         std::tie( result.time, result.stddev ) = functionTimer.timeFunction( compute, loops, minTime, monitor );
      this->performedLoops = functionTimer.getPerformedLoops();
   }
   catch ( const std::exception& e ) {
      std::cerr << "timeFunction failed due to a C++ exception with description: " << e.what() << std::endl;
   }

   result.bandwidth = datasetSize / result.time;
   result.speedup = this->baseTime / result.time;
   if( this->baseTime == 0.0 )
      this->baseTime = result.time;

   Logger::logResult( performer, result.getTableHeader(), result.getRowElements(), result.getColumnWidthHints() );

   return this->baseTime;
}

template< typename Logger >
   template< typename Device,
             typename ResetFunction,
             typename ComputeFunction >
inline double
Benchmark< Logger >::
time( ResetFunction reset,
      const String& performer,
      ComputeFunction& compute )
{
   BenchmarkResult result;
   return time< Device >( reset, performer, compute, result );
}

template< typename Logger >
   template< typename Device,
             typename ComputeFunction >
double
Benchmark< Logger >::
time( const String & performer,
      ComputeFunction & compute,
      BenchmarkResult & result )
{
   auto noReset = [] () {};
   return time< Device >( noReset, performer, compute, result );
}

template< typename Logger >
   template< typename Device,
             typename ComputeFunction >
inline double
Benchmark< Logger >::
time( const String & performer,
      ComputeFunction & compute )
{
   BenchmarkResult result;
   return time< Device >( performer, compute, result );
}

template< typename Logger >
void
Benchmark< Logger >::
addErrorMessage( const char* msg )
{
   Logger::writeErrorMessage( msg );
   std::cerr << msg << std::endl;
}

template< typename Logger >
auto
Benchmark< Logger >::
getMonitor() -> SolverMonitorType&
{
   return monitor;
}

template< typename Logger >
int
Benchmark< Logger >::
getPerformedLoops() const
{
   return this->performedLoops;
}

template< typename Logger >
bool
Benchmark< Logger >::
isResetingOn() const
{
   return reset;
}

} // namespace Benchmarks
} // namespace TNL
