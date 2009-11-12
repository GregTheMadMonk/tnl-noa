/***************************************************************************
                          mExplicitSolver.h  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef mExplicitSolverH
#define mExplicitSolverH

#include <iomanip>
#include <mcore.h>

template< class GRID, class SCHEME, typename T = double > class mExplicitSolver : public tnlObject
{
   public:
   
   mExplicitSolver()
   :  iteration( 0 ), 
      time( 0 ),
      tau( 0 ),
      residue( 0 ),
      solver_comm( MPI_COMM_WORLD ),
      verbosity( 0 ),
      cpu_timer( &default_mcore_cpu_timer ),
      rt_timer( &default_mcore_rt_timer )
      {
      };


   void SetTime( const double& t )
   {
      time = t;
   };

   const double& GetTime() const
   {
      return time;
   };

   //void SetFinalTime( const double& t );

   long int GetIterationNumber() const
   {
      return iteration;
   };
   
   void SetTau( const T& t )
   {
      tau = t;
   };
   
   const T& GetTau() const
   {
      return tau;
   };

   const double& GetResidue() const
   {
      return residue;
   };

   void SetMPIComm( MPI_Comm comm )
   {
      solver_comm = comm;
   };
  
   void SetVerbosity( int v )
   {
      verbosity = v;
   };

   void SetTimerCPU( const mTimerCPU* timer )
   {
      cpu_timer = timer;
   };

   void SetTimerRT( const mTimerRT* timer )
   {
      rt_timer = timer;
   };
  
   void PrintOut()
   {
      if( verbosity > 0 )
      {
         int cpu_time = 0;
         if( cpu_timer ) cpu_time = cpu_timer -> GetTime( 0, solver_comm );
         if( MPIGetRank() != 0 ) return;
         // TODO: add EST
         //cout << " EST: " << estimated;
         cout << " ITER:" << setw( 8 ) << GetIterationNumber()
              << " TAU:" << setprecision( 5 ) << setw( 12 ) << GetTau()
              << " T:" << setprecision( 5 ) << setw( 12 ) << GetTime()
              << " RES:" << setprecision( 5 ) << setw( 12 ) << GetResidue();
         if( cpu_timer )
            cout << " CPU: " << setw( 8 ) << cpu_time;
         if( rt_timer )
            cout << " ELA: " << setw( 8 ) << rt_timer -> GetTime();
         cout << "   \r" << flush;
      }
   }

   virtual int GetRungeKuttaIndex() const { return 0; };
   
   virtual bool Solve( SCHEME& scheme,
                       GRID& u,
                       const double& stop_time,
                       const double& max_res,
                       const long int max_iter ) = 0;
     
   protected:
    
   long int iteration;

   double time;

   T tau;

   double residue;

   MPI_Comm solver_comm;

   int verbosity;

   const mTimerCPU* cpu_timer;
   
   const mTimerRT* rt_timer;
};

#endif
