/***************************************************************************
                          tnlMatrixSolver.h  -  description
                             -------------------
    begin                : 2007/07/30
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlMatrixSolverH
#define tnlMatrixSolverH

#include <TNL/TimerCPU.h>
#include <TNL/TimerRT.h>
#include <TNL/core/mpi-supp.h>
#include <TNL/Object.h>
#include <TNL/matrices/tnlMatrix.h>
#include <TNL/legacy/solvers/tnlPreconditioner.h>

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlMatrixSolver : public Object
{
   public:

   tnlMatrixSolver( const String& name );

   Index getIterationNumber() const;

   const Real& getResidue() const;

   void setVerbosity( int verbose );

   void setTimerCPU( TimerCPU* timer );

   void setTimerRT( TimerRT* timer );
 
   virtual void printOut();

   virtual bool solve( const tnlMatrix< Real, Device, Index >& A,
                       const Vector< Real, Device, Index >& b,
                       Vector< Real, Device, Index >& x,
                       const Real& max_residue,
                       const Index max_iterations,
                       tnlPreconditioner< Real >* precond = 0 ) = 0;

   Real getResidue( const tnlMatrix< Real, Device, Index >& A,
                    const Vector< Real, Device, Index >& b,
                    const Vector< Real, Device, Index >& x,
                    const Real b_norm = 1.0 );

   virtual ~tnlMatrixSolver();

   protected:

   Index iteration;

   Real residue;

   MPI_Comm solver_comm;
 
   int verbosity;

   TimerCPU* cpu_timer;
 
   TimerRT* rt_timer;

};

template< typename Real, typename Device, typename Index >
tnlMatrixSolver< Real, Device, Index > :: tnlMatrixSolver( const String& name )
: Object( name ),
  iteration( 0 ),
  residue( 0.0 ),
  solver_comm( MPI_COMM_WORLD ),
  verbosity( 0 )/*,
  cpu_timer( &defaultCPUTimer ),
  rt_timer( &default_mcore_rt_timer )*/
{
};

template< typename Real, typename Device, typename Index >
Index tnlMatrixSolver< Real, Device, Index > :: getIterationNumber() const
{
   return this->iteration;
};

template< typename Real, typename Device, typename Index >
const Real& tnlMatrixSolver< Real, Device, Index > :: getResidue() const
{
   return this->residue;
};

template< typename Real, typename Device, typename Index >
void tnlMatrixSolver< Real, Device, Index > :: setVerbosity( int verbose )
{
   this->verbosity = verbose;
};

template< typename Real, typename Device, typename Index >
void tnlMatrixSolver< Real, Device, Index > :: setTimerCPU( TimerCPU* timer )
{
   this->cpu_timer = timer;
};

template< typename Real, typename Device, typename Index >
void tnlMatrixSolver< Real, Device, Index > :: setTimerRT( TimerRT* timer )
{
   this->rt_timer = timer;
};

template< typename Real, typename Device, typename Index >
void tnlMatrixSolver< Real, Device, Index > :: printOut()
{
   if( this->verbosity > 0 )
   {
      int cpu_time = 0;
      if( this->cpu_timer ) cpu_time = this->cpu_timer -> getTime( 0, this->solver_comm );
      if( MPIGetRank() != 0 ) return;
      // TODO: add EST
      //cout << " EST: " << estimated;
     std::cout << " ITER:" << std::setw( 8 ) << getIterationNumber()
           << " RES:" << std::setprecision( 5 ) << std::setw( 12 ) << getResidue();
      if( this->cpu_timer )
        std::cout << " CPU: " << std::setw( 8 ) << cpu_time;
      if( this->rt_timer )
        std::cout << " ELA: " << std::setw( 8 ) << this->rt_timer -> getTime();
     std::cout << "   \r" << std::flush;
   }
};

template< typename Real, typename Device, typename Index >
Real tnlMatrixSolver< Real, Device, Index > :: getResidue( const tnlMatrix< Real, Device, Index >& A,
                                                           const Vector< Real, Device, Index >& b,
                                                           const Vector< Real, Device, Index >& x,
                                                           const Real b_norm )
{
   const Index size = A. getSize();
   Real res( ( Real ) 0.0 );
   for( Index i = 0; i < size; i ++ )
   {
      Real err = fabs( A. rowProduct( i, x ) - b[ i ] );
      res += err * err;
   }
   return ::sqrt( res ) / b_norm;
   //return res;// / ( Real ) size;
};


template< typename Real, typename Device, typename Index >
tnlMatrixSolver< Real, Device, Index > :: ~tnlMatrixSolver()
{
};


#endif
