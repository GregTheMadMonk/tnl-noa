/***************************************************************************
                          tnlEulerSolver.h  -  description
                             -------------------
    begin                : 2008/04/01
    copyright            : (C) 2008 by Tomas Oberhuber
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

#ifndef tnlEulerSolverH
#define tnlEulerSolverH

#include <math.h>
#include <solvers/tnlExplicitSolver.h>

template< class Problem, class Mesh, typename Real = double, typename Device = tnlHost, typename Index = int >
class tnlEulerSolver : public tnlExplicitSolver< Problem, Mesh, Real, Device, Index >
{
   public:

   tnlEulerSolver( const tnlString& name );

   tnlString getType() const;

   bool solve( Problem& scheme,
               Mesh& u );

   protected:
   void computeNewTimeLevel( Mesh& u,
                             Real tau,
                             Real& currentResidue );

   
   Mesh k1;
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
tnlEulerSolver< Problem, Mesh, Real, Device, Index > :: tnlEulerSolver( const tnlString& name )
: tnlExplicitSolver< Problem, Mesh, Real, Device, Index >( name ),
  k1( "tnlEulerSolver:k1" )
{
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
tnlString tnlEulerSolver< Problem, Mesh, Real, Device, Index > :: getType() const
{
   Mesh m( "m" );
   Problem p( "p" );
   return tnlString( "tnlEulerSolver< " ) +
          p. getType() +
          tnlString( ", " ) +
          m. getType() +
          tnlString( ", " ) +
          GetParameterType( Real ( 0  ) ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( ", " ) +
          tnlString( GetParameterType( Index ( 0 ) ) ) +
          tnlString( ", " ) +
          tnlString( " >" );
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
bool tnlEulerSolver< Problem, Mesh, Real, Device, Index > :: solve( Problem& scheme,
                                                                    Mesh& u )
{
   /****
    * First setup the supporting meshes k1...k5 and k_tmp.
    */
   if( ! k1. setLike( u ) )
   {
      cerr << "I do not have enough memory to allocate a supporting grid for the Euler explicit solver." << endl;
      return false;
   }
   k1. setValue( 0.0 );


   /****
    * Set necessary parameters
    */
   Real& time = this -> time;
   Real currentTau = this -> tau;
   Real& residue = this -> residue;
   Index& iteration = this -> iteration;
   if( time + currentTau > this -> getStopTime() ) currentTau = this -> getStopTime() - time;
   if( currentTau == 0.0 ) return true;
   iteration = 0;

   /****
    * Do a printout ...
    */
   if( this -> verbosity > 0 )
      this -> printOut();

   /****
    * Start the main loop
    */
   while( 1 )
   {
      /****
       * Compute the RHS
       */
      scheme. GetExplicitRHS( time, u, k1 );

      Real lastResidue = residue;
      computeNewTimeLevel( u, currentTau, residue );

      /****
       * When time is close to stopTime the new residue
       * may be inaccurate significantly.
       */
      if( currentTau + time == this -> stopTime ) residue = lastResidue;
      time += currentTau;
      iteration ++;

      /****
       * Compute the new time step.
       */
      if( time + currentTau > this -> getStopTime() )
         currentTau = this -> getStopTime() - time; //we don't want to keep such tau
      else this -> tau = currentTau;

      /****
       * Do printouts if verbosity is on
       */
      if( this -> verbosity > 1 )
          this ->  printOut();

      /****
       * Check stop conditions.
       */
      if( time >= this -> getStopTime() ||
          ( this -> getMaxResidue() != 0.0 && residue < this -> getMaxResidue() ) )
       {
         if( this -> verbosity > 0 )
            this -> printOut();
          return true;
       }
      if( iteration == this -> getMaxIterationsNumber() ) return false;
   }
};

template< class Problem, class Mesh, typename Real, typename Device, typename Index >
void tnlEulerSolver< Problem, Mesh, Real, Device, Index > :: computeNewTimeLevel( Mesh& u,
                                                                                  Real tau,
                                                                                  Real& currentResidue )
{
   Real localResidue = Real( 0.0 );
   Index size = k1. getSize();
   Real* _u = u. getData();
   Real* _k1 = k1. getData();

#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:localResidue) firstprivate( _u, _k1, tau )
#endif
   for( Index i = 0; i < size; i ++ )
   {
      const Real add = tau * _k1[ i ];
      _u[ i ] += add;
      localResidue += fabs( add );
   }
   localResidue /= tau * ( Real ) size;
   :: MPIAllreduce( localResidue, currentResidue, 1, MPI_SUM, this -> solver_comm );
}


#endif
