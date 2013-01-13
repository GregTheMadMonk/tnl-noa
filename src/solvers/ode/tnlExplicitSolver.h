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

template< class Problem, class Mesh >
class tnlExplicitSolver : public tnlObject
{
   public:
   
   typedef Problem :: ProblemType;
   typedef Mesh :: MeshType;
   typedef typename Problem :: Real RealType;
   typedef typename Problem :: Device DeviceType;
   typedef typename Problem :: Index IndexType;

   tnlExplicitSolver( const tnlString& name );

   void setTime( const RealType& t );

   const RealType& getTime() const;
   
   void setStopTime( const RealType& stopTime );

   RealType getStopTime() const;

   void setTau( const RealType& t );
   
   const RealType& getTau() const;

   const RealType& getResidue() const;

   void setMaxResidue( const RealType& maxResidue );

   RealType getMaxResidue() const;

   IndexType getIterationsNumber() const;

   void setMaxIterationsNumber( IndexType maxIterationsNumber );

   IndexType getMaxIterationsNumber() const;

   void setMPIComm( MPI_Comm comm );
  
   void setVerbosity( IndexType v );

   void setTimerCPU( tnlTimerCPU* timer );

   void setTimerRT( tnlTimerRT* timer );

   void printOut() const;
   
   virtual bool solve( Problem& scheme,
                       Mesh& u ) = 0;

   void setTestingMode( bool testingMode );

protected:
    
   //! Current time of the parabolic problem.
   RealType time;

   //! The method solve will stop when reaching the stopTime.
   RealType stopTime;

   //! Current time step.
   RealType tau;

   RealType residue;

   //! The method solve will stop when the residue drops below tolerance given by maxResidue.
   /****
    * This is useful for finding steady state solution.
    */
   RealType maxResidue;

   //! Number of iterations.
   IndexType iteration;

   //! The method solve will stop when reaching the maximal number of iterations.
   IndexType maxIterationsNumber;

   MPI_Comm solver_comm;

   IndexType verbosity;

   tnlTimerCPU* cpu_timer;
   
   tnlTimerRT* rt_timer;

   bool testingMode;
};

template< class Problem, class Mesh >
tnlExplicitSolver < Problem, Mesh > :: tnlExplicitSolver( const tnlString&  name )
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

template< class Problem, class Mesh >
void tnlExplicitSolver < Problem, Mesh > :: setTime( const RealType& t )
{
   time = t;
};

template< class Problem, class Mesh >
const RealType& tnlExplicitSolver < Problem, Mesh > :: getTime() const
{
   return time;
};

template< class Problem, class Mesh >
IndexType tnlExplicitSolver < Problem, Mesh > :: getIterationsNumber() const
{
   return iteration;
};

template< class Problem, class Mesh >
void tnlExplicitSolver < Problem, Mesh > :: setTau( const RealType& t )
{
   tau = t;
};

template< class Problem, class Mesh >
const RealType& tnlExplicitSolver < Problem, Mesh > :: getTau() const
{
   return tau;
};

template< class Problem, class Mesh >
const RealType& tnlExplicitSolver < Problem, Mesh > :: getResidue() const
{
   return residue;
};

template< class Problem, class Mesh >
IndexType tnlExplicitSolver < Problem, Mesh > :: getMaxIterationsNumber() const
{
    return maxIterationsNumber;
}

template< class Problem, class Mesh >
RealType tnlExplicitSolver < Problem, Mesh > :: getMaxResidue() const
{
    return maxResidue;
}

template< class Problem, class Mesh >
RealType tnlExplicitSolver < Problem, Mesh > :: getStopTime() const
{
    return stopTime;
}

template< class Problem, class Mesh >
void tnlExplicitSolver < Problem, Mesh > :: setMaxIterationsNumber( IndexType maxIterationsNumber )
{
    this -> maxIterationsNumber = maxIterationsNumber;
}

template< class Problem, class Mesh >
void tnlExplicitSolver < Problem, Mesh > :: setMaxResidue( const RealType& maxResidue )
{
    this -> maxResidue = maxResidue;
}

template< class Problem, class Mesh >
void tnlExplicitSolver < Problem, Mesh > :: setStopTime( const RealType& stopTime )
{
    this -> stopTime = stopTime;
}

template< class Problem, class Mesh >
void tnlExplicitSolver < Problem, Mesh > :: setMPIComm( MPI_Comm comm )
{
   solver_comm = comm;
};

template< class Problem, class Mesh >
void tnlExplicitSolver < Problem, Mesh > :: setVerbosity( IndexType v )
{
   verbosity = v;
};

template< class Problem, class Mesh >
void tnlExplicitSolver < Problem, Mesh > :: setTimerCPU( tnlTimerCPU* timer )
{
   cpu_timer = timer;
};

template< class Problem, class Mesh >
void tnlExplicitSolver < Problem, Mesh > :: setTimerRT( tnlTimerRT* timer )
{
   rt_timer = timer;
};

template< class Problem, class Mesh >
void tnlExplicitSolver < Problem, Mesh > :: printOut() const
{
   if( verbosity > 0 )
   {
      IndexType cpu_time = 0;
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

template< class Problem, class Mesh >
void tnlExplicitSolver < Problem, Mesh > :: setTestingMode( bool testingMode )
{
   this -> testingMode = testingMode;
}


#endif
