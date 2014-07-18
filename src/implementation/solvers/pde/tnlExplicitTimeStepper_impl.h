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
  problem( 0 )
{
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
void tnlExplicitTimeStepper< Problem, OdeSolver >::configSetup( tnlConfigDescription& config,
                                                                const tnlString& prefix )
{
   config.addEntry< double >( "tau", "Time step for the time discretisation.", 1.0 );
}

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
bool tnlExplicitTimeStepper< Problem, OdeSolver >::init( const tnlParameterContainer& parameters,
                                                         const tnlString& prefix )
{
   this->setTau( parameters.GetParameter< double >( "tau") );
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
bool tnlExplicitTimeStepper< Problem, OdeSolver > :: setTau( const RealType& tau )
{
   if( tau <= 0.0 )
   {
      cerr << "Tau for tnlExplicitTimeStepper must be positive. " << endl;
      return false;
   }
   this -> tau = tau;
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
bool tnlExplicitTimeStepper< Problem, OdeSolver >::solve( const RealType& time,
                                                          const RealType& stopTime )
{
   this -> odeSolver -> setTau( this -> tau );
   this -> odeSolver -> setProblem( * this );
   DofVectorType& u = problem -> getDofVector();
   this -> odeSolver -> setTime( time );
   this -> odeSolver -> setStopTime( stopTime );
   return this -> odeSolver -> solve( u );
}

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
void tnlExplicitTimeStepper< Problem, OdeSolver >::GetExplicitRHS( const RealType& time,
                                                                   const RealType& tau,
                                                                   DofVectorType& _u,
                                                                   DofVectorType& _fu )
{
   return this->problem->GetExplicitRHS( time, tau, _u, _fu );
}

#endif /* TNLEXPLICITTIMESTEPPER_IMPL_H_ */
