/***************************************************************************
                          tnlEulerSolver_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef tnlEulerSolver_implH
#define tnlEulerSolver_implH


template< typename Problem >
tnlEulerSolver< Problem > :: tnlEulerSolver()
: k1( "tnlEulerSolver:k1" ),
  cflCondition( 1.0 )
{
   this->setName( "EulerSolver" );
};

template< typename Problem >
tnlString tnlEulerSolver< Problem > :: getType() const
{
   return tnlString( "tnlEulerSolver< " ) +
          Problem :: getTypeStatic() +
          tnlString( " >" );
};

template< typename Problem >
void tnlEulerSolver< Problem > :: configSetup( tnlConfigDescription& config,
                                               const tnlString& prefix )
{
   tnlExplicitSolver< Problem >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "euler-cfl", "Coefficient C in the Courant–Friedrichs–Lewy condition.", 1.0 );
};

template< typename Problem >
bool tnlEulerSolver< Problem > :: init( const tnlParameterContainer& parameters,
                                        const tnlString& prefix )
{
   tnlExplicitSolver< Problem >::init( parameters, prefix );
   if( parameters.CheckParameter( prefix + "euler-cfl" ) )
      this->setCFLCondition( parameters.GetParameter< double >( prefix + "euler-cfl" ) );
}

template< typename Problem >
void tnlEulerSolver< Problem > :: setCFLCondition( const RealType& cfl )
{
   this -> cflCondition = cfl;
}

template< typename Problem >
const typename Problem :: RealType& tnlEulerSolver< Problem > :: getCFLCondition() const
{
   return this -> cflCondition;
}

template< typename Problem >
bool tnlEulerSolver< Problem > :: solve( DofVectorType& u )
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

   this -> refreshSolverMonitor();

   /****
    * Start the main loop
    */
   while( 1 )
   {
      /****
       * Compute the RHS
       */
      this -> problem -> GetExplicitRHS( time, currentTau, u, k1 );

      RealType lastResidue = residue;
      RealType maxResidue( 0.0 );
      if( this -> cflCondition != 0.0 )
      {
         maxResidue = k1. absMax();
         if( currentTau * maxResidue > this -> cflCondition )
         {
            currentTau *= 0.9;
            continue;
         }
      }
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

      this -> refreshSolverMonitor();

      /****
       * Check stop conditions.
       */
      if( time >= this -> getStopTime() ||
          ( this -> getMaxResidue() != 0.0 && residue < this -> getMaxResidue() ) )
       {
         this -> refreshSolverMonitor();
         return true;
       }
      if( iteration == this -> getMaxIterationsNumber() ) return false;

      if( this -> cflCondition != 0.0 )
         currentTau /= 0.95;
   }
};

template< typename Problem >
void tnlEulerSolver< Problem > :: computeNewTimeLevel( DofVectorType& u,
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
