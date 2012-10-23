/***************************************************************************
                          tnlExplicitSolver.h  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef tnlExplicitSolverH
#define tnlExplicitSolverH

#include <iomanip>
#include <core/tnlTimerCPU.h>
#include <core/tnlTimerRT.h>
#include <core/tnlFlopsCounter.h>

template< class Problem, class Mesh, typename Real = double, typename Device = tnlHost, typename Index = int >
class tnlExplicitSolver : public tnlObject
{
   public:
   
   tnlExplicitSolver( const tnlString& name );

   void setTime( const Real& t );

   const Real& getTime() const;
   
   void setStopTime( const Real& stopTime );

   Real getStopTime() const;

   void setTau( const Real& t );
   
   const Real& getTau() const;

   const Real& getResidue() const;

   void setMaxResidue( const Real& maxResidue );

   Real getMaxResidue() const;

   int getIterationsNumber() const;

   void setMaxIterationsNumber( int maxIterationsNumber );

   int getMaxIterationsNumber() const;

   void setMPIComm( MPI_Comm comm );
  
   void setVerbosity( int v );

   void setTimerCPU( tnlTimerCPU* timer );

   void setTimerRT( tnlTimerRT* timer );

   void printOut() const;
   
   virtual bool solve( Problem& scheme,
                       Mesh& u ) = 0;

   void setTestingMode( bool testingMode );

protected:
    
   //! Current time of the parabolic problem.
   Real time;

   //! The method solve will stop when reaching the stopTime.
   Real stopTime;

   //! Current time step.
   Real tau;

   Real residue;

   //! The method solve will stop when the residue drops below tolerance given by maxResidue.
   /****
    * This is useful for finding steady state solution.
    */
   Real maxResidue;

   //! Number of iterations.
   int iteration;

   //! The method solve will stop when reaching the maximal number of iterations.
   int maxIterationsNumber;

   MPI_Comm solver_comm;

   int verbosity;

   tnlTimerCPU* cpu_timer;
   
   tnlTimerRT* rt_timer;

   bool testingMode;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: tnlExplicitSolver( const tnlString&  name )
:  tnlObject( name ),
   iteration( 0 ),
   time( 0.0 ),
   tau( 0.0 ),
   residue( 0.0 ),
   stopTime( 0.0 ),
   maxResidue( 0.0 ),
   maxIterationsNumber( -1 ),
   solver_comm( MPI_COMM_WORLD ),
   verbosity( 0 ),
   cpu_timer( &default_mcore_cpu_timer ),
   rt_timer( &default_mcore_rt_timer ),
   testingMode( false )
   {
   };

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: setTime( const Real& t )
{
   time = t;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
const Real& tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: getTime() const
{
   return time;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
int tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: getIterationsNumber() const
{
   return iteration;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: setTau( const Real& t )
{
   tau = t;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
const Real& tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: getTau() const
{
   return tau;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
const Real& tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: getResidue() const
{
   return residue;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
int tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: getMaxIterationsNumber() const
{
    return maxIterationsNumber;
}

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
Real tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: getMaxResidue() const
{
    return maxResidue;
}

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
Real tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: getStopTime() const
{
    return stopTime;
}

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: setMaxIterationsNumber( int maxIterationsNumber )
{
    this -> maxIterationsNumber = maxIterationsNumber;
}

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: setMaxResidue( const Real& maxResidue )
{
    this -> maxResidue = maxResidue;
}

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: setStopTime( const Real& stopTime )
{
    this -> stopTime = stopTime;
}

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: setMPIComm( MPI_Comm comm )
{
   solver_comm = comm;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: setVerbosity( int v )
{
   verbosity = v;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: setTimerCPU( tnlTimerCPU* timer )
{
   cpu_timer = timer;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: setTimerRT( tnlTimerRT* timer )
{
   rt_timer = timer;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: printOut() const
{
   if( verbosity > 0 )
   {
      int cpu_time = 0;
      if( cpu_timer ) cpu_time = cpu_timer -> GetTime( 0, solver_comm );
      if( MPIGetRank() != 0 ) return;
      // TODO: add EST
      //cout << " EST: " << estimated;
      cout << " ITER:" << setw( 8 ) << getIterationsNumber()
           << " TAU:" << setprecision( 5 ) << setw( 12 ) << getTau()
           << " T:" << setprecision( 5 ) << setw( 12 ) << getTime()
           << " RES:" << setprecision( 5 ) << setw( 12 ) << getResidue();
      if( cpu_timer )
         cout << " CPU: " << setw( 8 ) << cpu_time;
      if( rt_timer )
         cout << " ELA: " << setw( 8 ) << rt_timer -> GetTime();
      double flops = ( double ) tnl_flops_counter. getFlops();
      if( flops )
      {
       cout << " GFLOPS:  " << setw( 8 ) << 1.0e-9 * flops / rt_timer -> GetTime();
      }
      cout << "   \r" << flush;
   }
}

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlExplicitSolver < Problem, Mesh, Real, Device, Index > :: setTestingMode( bool testingMode )
{
   this -> testingMode = testingMode;
}


#endif
