/***************************************************************************
                          tnlSemiImplicitTimeStepper_impl.h  -  description
                             -------------------
    begin                : Oct 4, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLSEMIIMPLICITTIMESTEPPER_IMPL_H_
#define TNLSEMIIMPLICITTIMESTEPPER_IMPL_H_

template< typename Problem,
          typename MatrixSolver >
tnlSemiImplicitTimeStepper< Problem, MatrixSolver >::
tnlSemiImplicitTimeStepper()
: problem( 0 )
{
};

template< typename Problem,
          typename MatrixSolver >
void
tnlSemiImplicitTimeStepper< Problem, MatrixSolver >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   config.addEntry< double >( "tau", "Time step for the time discretisation.", 1.0 );
}

template< typename Problem,
          typename MatrixSolver >
bool
tnlSemiImplicitTimeStepper< Problem, MatrixSolver >::
init( const tnlParameterContainer& parameters,
      const tnlString& prefix )
{
   this->setTau( parameters.GetParameter< double >( "tau") );
   return true;
}

template< typename Problem,
          typename MatrixSolver >
void tnlSemiImplicitTimeStepper< Problem, MatrixSolver >::setProblem( ProblemType& problem )
{
   this -> problem = &problem;
};

template< typename Problem,
          typename MatrixSolver >
Problem* tnlSemiImplicitTimeStepper< Problem, MatrixSolver >::getProblem() const
{
    return this -> problem;
};

template< typename Problem,
          typename MatrixSolver >
bool tnlSemiImplicitTimeStepper< Problem, MatrixSolver >::setTau( const RealType& tau )
{
   if( tau <= 0.0 )
   {
      cerr << "Tau for tnlSemiImplicitTimeStepper must be positive. " << endl;
      return false;
   }
   this -> tau = tau;
};

template< typename Problem,
          typename MatrixSolver >
bool tnlSemiImplicitTimeStepper< Problem, MatrixSolver >::solve( const RealType& time,
                                                                 const RealType& stopTime,
                                                                 const MeshType& mesh,
                                                                 DofVectorType& dofVector )
{
   tnlAssert( this->problem != 0, );
   RealType t = time;
   while( t < stopTime )
   {
      Real currentTau = tnlMin( this->tau, stopTime - t );
      this->problem->assemblyLinearSystem( t, tau, mesh, dofVector, matrix, b );

   }
}

#endif /* TNLSEMIIMPLICITTIMESTEPPER_IMPL_H_ */
