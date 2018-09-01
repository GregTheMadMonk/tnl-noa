/***************************************************************************
                          IterativeSolverMonitor_impl.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iomanip>
#include <limits>

// make sure to include the config before the check
#include <TNL/tnlConfig.h>
#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include <TNL/Solvers/IterativeSolver.h>

namespace TNL {
namespace Solvers {   

template< typename Real, typename Index>
IterativeSolverMonitor< Real, Index > :: IterativeSolverMonitor()
: SolverMonitor(),
  stage( "" ),
  saved_stage( "" ),
  saved( false ),
  time( 0.0 ),
  saved_time( 0.0 ),
  timeStep( 0.0 ),
  saved_timeStep( 0.0 ),
  residue( 0.0 ),
  saved_residue( 0.0 ),
  iterations( 0 ),
  saved_iterations( 0 ),
  verbose( 1 )
{
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setStage( const std::string& stage )
{
   // save the items after a complete stage
   if( iterations > 0 ) {
      saved_stage = this->stage;
      saved_time = time;
      saved_timeStep = timeStep;
      saved_iterations = iterations;
      saved_residue = residue;
   }

   // reset the current items
   iterations = 0;
   residue = 0.0;

   this->stage = stage;
   saved = true;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setTime( const RealType& time )
{
   this->time = time;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setTimeStep( const RealType& timeStep )
{
   this->timeStep = timeStep;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setIterations( const Index& iterations )
{
   this->iterations = iterations;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setResidue( const Real& residue )
{
   this->residue = residue;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setVerbose( const Index& verbose )
{
   this->verbose = verbose;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: refresh()
{
   if( this->verbose > 0 )
   {
      // Check if we should display the current values or the values saved after
      // the previous stage. If the iterations cycle much faster than the solver
      // monitor refreshes, we display only the values saved after the whole
      // cycle to hide the irrelevant partial progress.
      const bool saved = this->saved;
      this->saved = false;

      const int line_width = getLineWidth();
      int free = line_width ? line_width : std::numeric_limits<int>::max();

      auto real_to_string = []( Real value, int precision = 6 ) {
         std::stringstream stream;
         stream << std::setprecision( precision ) << value;
         return stream.str();
      };

      auto print_item = [&free]( const std::string& item, int width = 0 ) {
         width = min( free, (width) ? width : item.length() );
         std::cout << std::setw( width ) << item.substr( 0, width );
         free -= width;
      };

      // \33[2K erases the current line, see https://stackoverflow.com/a/35190285
      std::cout << "\33[2K\r";

      // FIXME: nvcc 8.0 ignores default parameter values for lambda functions in template functions, so we have to pass the defaults
//      print_item( " ELA:" );
      print_item( " ELA:", 0 );
      print_item( real_to_string( getElapsedTime(), 5 ), 8 );
//      print_item( " T:" );
      print_item( " T:", 0 );
      print_item( real_to_string( (saved) ? saved_time : time, 5 ), 8 );
      if( (saved) ? saved_timeStep : timeStep > 0 ) {
//         print_item( " TAU:" );
         print_item( " TAU:", 0 );
         print_item( real_to_string( (saved) ? saved_timeStep : timeStep, 5 ), 8 );
      }

      const std::string displayed_stage = (saved) ? saved_stage : stage;
      if( displayed_stage.length() && free > 5 ) {
         if( (int) displayed_stage.length() <= free - 2 ) {
            std::cout << "  " << displayed_stage;
            free -= ( 2 + displayed_stage.length() );
         }
         else {
            std::cout << "  " << displayed_stage.substr( 0, free - 5 ) << "...";
            free = 0;
         }
      }

      if( (saved) ? saved_iterations : iterations > 0 && free >= 14 ) {
//         print_item( " ITER:" );
         print_item( " ITER:", 0 );
         print_item( std::to_string( (saved) ? saved_iterations : iterations ), 8 );
      }
      if( (saved) ? saved_residue : residue && free >= 17 ) {
//         print_item( " RES:" );
         print_item( " RES:", 0 );
         print_item( real_to_string( (saved) ? saved_residue : residue, 5 ), 12 );
      }

      // return to the beginning of the line
      std::cout << "\r" << std::flush;
   }
}

template< typename Real, typename Index>
int IterativeSolverMonitor< Real, Index > :: getLineWidth()
{
#ifdef HAVE_SYS_IOCTL_H
   struct winsize w;
   ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
   return w.ws_col;
#else
   return 0;
#endif
}

} // namespace Solvers
} // namespace TNL
