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

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace TNL {
namespace Solvers {   

template< typename Real, typename Index>
IterativeSolverMonitor< Real, Index > :: IterativeSolverMonitor()
: SolverMonitor(),
  time( 0.0 ),
  timeStep( 0.0 ),
  stage( "" ),
  iterations( 0 ),
  residue( 0 ),
  verbose( 1 )
{
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
void IterativeSolverMonitor< Real, Index > :: setStage( const std::string& stage )
{
   this->stage = stage;
   // reset numerical items displayed after stage
   this->iterations = 0;
   this->residue = 0.0;
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
void IterativeSolverMonitor< Real, Index > :: refresh( bool force )
{
//   if( this->verbose > 0 && ( force || this->getIterations() % this->refreshRate == 0 ) )
   if( this->verbose > 0 && force )
   {
      const int line_width = this->getLineWidth();
      int free = line_width ? line_width : std::numeric_limits<int>::max();

      std::cout << " ELA:" << std::setw( 8 ) << this->getElapsedTime()
                << " T:"   << std::setw( 8 ) << this->time;
      free -= 24;
      if( this->timeStep > 0 ) {
         std::cout << " TAU:" << std::setw( 5 ) << this->timeStep;
         free -= 10;
      }

      if( this->stage.length() && free > 5 ) {
         if( this->stage.length() <= free - 2 ) {
            std::cout << "  " << this->stage;
            free -= ( 2 + this->stage.length() );
         }
         else {
            std::cout << "  " << this->stage.substr( 0, free - 5 ) << "...";
            free = 0;
         }
      }

      if( this->iterations > 0 && free >= 14 ) {
         std::cout << " ITER:" << std::setw( 8 ) << this->iterations;
         free -= 14;
      }
      if( this->residue && free >= 17 ) {
         std::cout << " RES:" << std::setprecision( 5 ) << std::setw( 12 ) << this->residue;
         free -= 17;
      }

      // fill the rest of the line with spaces to clear previous content
      while( line_width && free-- > 0 )
         std::cout << " ";
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
