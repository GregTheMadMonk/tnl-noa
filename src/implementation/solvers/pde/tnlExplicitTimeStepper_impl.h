/***************************************************************************
                          tnlExplicitTimeStepper_impl.h  -  description
                             -------------------
    begin                : Jan 15, 2013
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

#ifndef TNLEXPLICITTIMESTEPPER_IMPL_H_
#define TNLEXPLICITTIMESTEPPER_IMPL_H_

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
tnlExplicitTimeStepper< Problem, OdeSolver > :: tnlExplicitTimeStepper()
: odeSolver( 0 ),
  problem( 0 ),
  timeStep( 0 )
{
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
void
tnlExplicitTimeStepper< Problem, OdeSolver >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
}

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
bool
tnlExplicitTimeStepper< Problem, OdeSolver >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   return true;
}

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
bool
tnlExplicitTimeStepper< Problem, OdeSolver >::
init( const MeshType& mesh )
{
   return true;
}

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
void tnlExplicitTimeStepper< Problem, OdeSolver >::setSolver(
      typename tnlExplicitTimeStepper< Problem, OdeSolver >::OdeSolverType& odeSolver )
{
   this -> odeSolver = &odeSolver;
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
void tnlExplicitTimeStepper< Problem, OdeSolver > :: setProblem( ProblemType& problem )
{
   this -> problem = &problem;
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
Problem* tnlExplicitTimeStepper< Problem, OdeSolver > :: getProblem() const
{
    return this -> problem;
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
bool tnlExplicitTimeStepper< Problem, OdeSolver > :: setTimeStep( const RealType& timeStep )
{
   if( timeStep <= 0.0 )
   {
      cerr << "Tau for tnlExplicitTimeStepper must be positive. " << endl;
      return false;
   }
   this -> timeStep = timeStep;
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
bool tnlExplicitTimeStepper< Problem, OdeSolver >::solve( const RealType& time,
                                                          const RealType& stopTime,
                                                          const MeshType& mesh,
                                                          DofVectorType& dofVector )
{
   tnlAssert( this->odeSolver, );
   this->odeSolver->setTau( this -> timeStep );
   this->odeSolver->setProblem( * this );
   this->odeSolver->setTime( time );
   this->odeSolver->setStopTime( stopTime );
   this->mesh = &mesh;
   return this->odeSolver->solve( dofVector );
}

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
void tnlExplicitTimeStepper< Problem, OdeSolver >::getExplicitRHS( const RealType& time,
                                                                   const RealType& tau,
                                                                   DofVectorType& _u,
                                                                   DofVectorType& _fu )
{
   this->problem->getExplicitRHS( time, tau, *( this->mesh ), _u, _fu );
}

#endif /* TNLEXPLICITTIMESTEPPER_IMPL_H_ */
