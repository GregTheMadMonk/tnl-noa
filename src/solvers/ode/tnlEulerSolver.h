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
#include <solvers/ode/tnlExplicitSolver.h>

template< class Problem, class Mesh >
class tnlEulerSolver : public tnlExplicitSolver< Problem, Mesh >
{
   public:

   typedef typename Problem  ProblemType;
   typedef typename Mesh  MeshType;
   typedef typename Problem :: Real RealType;
   typedef typename Problem :: Device DeviceType;
   typedef typename Problem :: Index IndexType;


   tnlEulerSolver( const tnlString& name );

   tnlString getType() const;

   bool solve( ProblemType& scheme,
               MeshType& u );

   protected:
   void computeNewTimeLevel( MeshType& u,
                             RealType tau,
                             RealType& currentResidue );

   
   MeshType k1;
};

template< class Problem, class Mesh >
tnlEulerSolver< Problem, Mesh > :: tnlEulerSolver( const tnlString& name )
: tnlExplicitSolver< Problem, Mesh >( name ),
  k1( "tnlEulerSolver:k1" )
{
};

template< class Problem, class Mesh >
tnlString tnlEulerSolver< Problem, Mesh > :: getType() const
{
   Mesh m( "m" );
   Problem p( "p" );
   return tnlString( "tnlEulerSolver< " ) +
          p. getType() +
          tnlString( ", " ) +
          m. getType() +
          tnlString( ", " ) +
          GetParameterType( RealType ( 0  ) ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( ", " ) +
          tnlString( GetParameterType( IndexType ( 0 ) ) ) +
          tnlString( ", " ) +
          tnlString( " >" );
};

template< class Problem, class Mesh >
bool tnlEulerSolver< Problem, Mesh > :: solve( Problem& scheme,
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
   RealType& time = this -> time;
   RealType currentTau = this -> tau;
   RealType& residue = this -> residue;
   IndexType& iteration = this -> iteration;
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
      scheme. GetExplicitRHS( time, currentTau, u, k1 );

      RealType lastResidue = residue;
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

template< class Problem, class Mesh >
void tnlEulerSolver< Problem, Mesh > :: computeNewTimeLevel( Mesh& u,
                                        RealType tau,
                                        RealType& currentResidue )
{
   RealType localResidue = RealType( 0.0 );
   IndexType size = k1. getSize();
   RealType* _u = u. getData();
   RealType* _k1 = k1. getData();

#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:localResidue) firstprivate( _u, _k1, tau )
#endif
   for( IndexType i = 0; i < size; i ++ )
   {
      const RealType add = tau * _k1[ i ];
      _u[ i ] += add;
      localResidue += fabs( add );
   }
   localResidue /= tau * ( RealType ) size;
   :: MPIAllreduce( localResidue, currentResidue, 1, MPI_SUM, this -> solver_comm );
}


#endif
