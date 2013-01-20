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
void tnlExplicitTimeStepper< Problem, OdeSolver > :: setSolver( OdeSolver< Problem >& odeSolver )
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
bool tnlExplicitTimeStepper< Problem, OdeSolver > :: solve( const RealType& time,
                                                            const RealType& stopTime )
{
   this -> odeSolver -> setTau( this -> tau );
   this -> odeSolver -> setProblem( * this -> problem );
   DofVectorType& u = problem -> getDofVector();
   this -> odeSolver -> setTime( time );
   this -> odeSolver -> setStopTime( stopTime );
   return this -> odeSolver -> solve( u );
}

#endif /* TNLEXPLICITTIMESTEPPER_IMPL_H_ */
