/***************************************************************************
                          tnlMatrixSolver.h  -  description
                             -------------------
    begin                : 2007/07/30
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

#ifndef tnlMatrixSolverH
#define tnlMatrixSolverH

#include <core/tnlTimerCPU.h>
#include <core/tnlTimerRT.h>
#include <core/mpi-supp.h>
#include <core/tnlObject.h>
#include <matrix/tnlMatrix.h>
#include <solvers/tnlPreconditioner.h>

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlMatrixSolver : public tnlObject
{
   public:

   tnlMatrixSolver( const tnlString& name );

   Index getIterationNumber() const;

   const Real& getResidue() const;

   void setVerbosity( int verbose );

   void setTimerCPU( tnlTimerCPU* timer );

   void setTimerRT( tnlTimerRT* timer );
   
   virtual void printOut();

   virtual bool solve( const tnlMatrix< Real, Device, Index >& A,
                       const tnlVector< Real, Device, Index >& b,
                       tnlVector< Real, Device, Index >& x,
                       const Real& max_residue,
                       const Index max_iterations,
                       tnlPreconditioner< Real >* precond = 0 ) = 0;

   Real getResidue( const tnlMatrix< Real, Device, Index >& A,
                    const tnlVector< Real, Device, Index >& b,
                    const tnlVector< Real, Device, Index >& x,
                    const Real b_norm = 1.0 );

   virtual ~tnlMatrixSolver();

   protected:

   Index iteration;

   Real residue;

   MPI_Comm solver_comm;
   
   int verbosity;

   tnlTimerCPU* cpu_timer;
   
   tnlTimerRT* rt_timer;

};

template< typename Real, typename Device, typename Index >
tnlMatrixSolver< Real, Device, Index > :: tnlMatrixSolver( const tnlString& name )
: tnlObject( name ),
  iteration( 0 ),
  residue( 0.0 ),
  solver_comm( MPI_COMM_WORLD ),
  verbosity( 0 ),
  cpu_timer( &default_mcore_cpu_timer ),
  rt_timer( &default_mcore_rt_timer )
{
};

template< typename Real, typename Device, typename Index >
Index tnlMatrixSolver< Real, Device, Index > :: getIterationNumber() const
{
   return this -> iteration;
};

template< typename Real, typename Device, typename Index >
const Real& tnlMatrixSolver< Real, Device, Index > :: getResidue() const
{
   return this -> residue;
};

template< typename Real, typename Device, typename Index >
void tnlMatrixSolver< Real, Device, Index > :: setVerbosity( int verbose )
{
   this -> verbosity = verbose;
};

template< typename Real, typename Device, typename Index >
void tnlMatrixSolver< Real, Device, Index > :: setTimerCPU( tnlTimerCPU* timer )
{
   this -> cpu_timer = timer;
};

template< typename Real, typename Device, typename Index >
void tnlMatrixSolver< Real, Device, Index > :: setTimerRT( tnlTimerRT* timer )
{
   this -> rt_timer = timer;
};

template< typename Real, typename Device, typename Index >
void tnlMatrixSolver< Real, Device, Index > :: printOut()
{
   if( this -> verbosity > 0 )
   {
      int cpu_time = 0;
      if( this -> cpu_timer ) cpu_time = this -> cpu_timer -> GetTime( 0, this -> solver_comm );
      if( MPIGetRank() != 0 ) return;
      // TODO: add EST
      //cout << " EST: " << estimated;
      cout << " ITER:" << setw( 8 ) << getIterationNumber()
           << " RES:" << setprecision( 5 ) << setw( 12 ) << getResidue();
      if( this -> cpu_timer )
         cout << " CPU: " << setw( 8 ) << cpu_time;
      if( this -> rt_timer )
         cout << " ELA: " << setw( 8 ) << this -> rt_timer -> GetTime();
      cout << "   \r" << flush;
   }
};

template< typename Real, typename Device, typename Index >
Real tnlMatrixSolver< Real, Device, Index > :: getResidue( const tnlMatrix< Real, Device, Index >& A,
                                                           const tnlVector< Real, Device, Index >& b,
                                                           const tnlVector< Real, Device, Index >& x,
                                                           const Real b_norm )
{
   const Index size = A. getSize();
   Real res( ( Real ) 0.0 );
   for( Index i = 0; i < size; i ++ )
   {
      Real err = fabs( A. rowProduct( i, x ) - b[ i ] );
      res += err * err;
   }
   return sqrt( res ) / b_norm;
   //return res;// / ( Real ) size;
};


template< typename Real, typename Device, typename Index >
tnlMatrixSolver< Real, Device, Index > :: ~tnlMatrixSolver()
{
};


#endif
