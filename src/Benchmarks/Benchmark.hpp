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
   if( Logger::metadataColumns != metadata )
      Logger::header_changed = true;
   Logger::metadataColumns = metadata;
}

template< typename Logger >
void
Benchmark< Logger >::
setOperation( const String & operation,
              const double datasetSize,
              const double baseTime )
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

template< typename Logger >
void
Benchmark< Logger >::
setOperation( const double datasetSize,
              const double baseTime )
{
   this->datasetSize = datasetSize;
   this->baseTime = baseTime;
}

template< typename Logger >
void
Benchmark< Logger >::
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
   return time< Device, ResetFunction, ComputeFunction >( reset, performer, compute, result );
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

template< typename Logger >
   template< typename Device,
             typename ComputeFunction >
inline double
Benchmark< Logger >::
time( const String & performer,
      ComputeFunction & compute )
{
   BenchmarkResult result;
   return time< Device, ComputeFunction >( performer, compute, result );
}

template< typename Logger >
void
Benchmark< Logger >::
addErrorMessage( const char* msg,
                 int numberOfComputations )
{
   // each computation has 3 subcolumns
   const int colspan = 3 * numberOfComputations;
   Logger::writeErrorMessage( msg, colspan );
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
