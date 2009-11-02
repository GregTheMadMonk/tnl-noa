/***************************************************************************
                          mMatrixSolver.h  -  description
                             -------------------
    begin                : 2007/07/30
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

#ifndef mMatrixSolverH
#define mMatrixSolverH

#include <mTimerCPU.h>
#include <mTimerRT.h>
#include <mpi-supp.h>
#include "mBaseMatrix.h"
#include "mPreconditioner.h"

template< typename T > class mMatrixSolver 
{
   public:

   mMatrixSolver()
   : iteration( 0 ),
     residue( 0.0 ),
     solver_comm( MPI_COMM_WORLD ),
     verbosity( 0 ),
     cpu_timer( &default_mcore_cpu_timer ),
     rt_timer( &default_mcore_rt_timer )
   {
   };

   long int GetIterationNumber() const
   {
      return iteration;
   };

   const T& GetResidue() const
   {
      return residue;
   };

   void SetVerbosity( int verbose )
   {
      verbosity = verbose;
   };

   void SetTimerCPU( const mTimerCPU* timer )
   {
      cpu_timer = timer;
   };

   void SetTimerRT( const mTimerRT* timer )
   {
      rt_timer = timer;
   };
   
   virtual void PrintOut()
   {
      if( verbosity > 0 )
      {
         int cpu_time = 0;
         if( cpu_timer ) cpu_time = cpu_timer -> GetTime( 0, solver_comm );
         if( MPIGetRank() != 0 ) return;
         // TODO: add EST
         //cout << " EST: " << estimated;
         cout << " ITER:" << setw( 8 ) << GetIterationNumber()
              << " RES:" << setprecision( 5 ) << setw( 12 ) << GetResidue();
         if( cpu_timer )
            cout << " CPU: " << setw( 8 ) << cpu_time;
         if( rt_timer )
            cout << " ELA: " << setw( 8 ) << rt_timer -> GetTime();
         cout << "   \r" << flush;
      }
   };

   virtual bool Solve( const mBaseMatrix< T >& A,
                       const T* b,
                       T* x, 
                       const double& max_residue,
                       const long int max_iterations,
                       mPreconditioner< T >* precond = 0 ) = 0;

   virtual ~mMatrixSolver() {};

   protected:

   long int iteration;

   T residue;

   MPI_Comm solver_comm;
   
   int verbosity;

   const mTimerCPU* cpu_timer;
   
   const mTimerRT* rt_timer;

};

#endif
