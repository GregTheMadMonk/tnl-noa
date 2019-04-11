/***************************************************************************
                          transportEquationProblemEoc_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>
#include <TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <TNL/Solvers/PDE/BackwardTimeDiscretisation.h>
#include <TNL/Functions/Analytic/Paraboloid.h>
#include <TNL/Operators/Analytic/Heaviside.h>
#include <TNL/Operators/Analytic/Shift.h>

#include "transportEquationProblem.h"

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
String
transportEquationProblemEoc< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
getType()
{
   return String( "transportEquationProblemEoc< " ) + Mesh :: getType() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
String
transportEquationProblemEoc< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
getPrologHeader() const
{
   return String( "Transport Equation EOC" );
}


template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
bool
transportEquationProblemEoc< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( ! this->velocityField->setup( this->getMesh(), parameters, prefix + "velocity-field-" ) ||
       ! this->differentialOperatorPointer->setup( this->getMesh(), parameters, prefix ) ||
       ! this->boundaryConditionPointer->setup( this->getMesh(), parameters, prefix + "boundary-conditions-" ) )
      return false;
   
   /****
    * Render the exact solution
    */
   const String& initialCondition = parameters.getParameter< String >( "initial-condition" );
   const double& finalTime = parameters.getParameter< double >( "final-time" );
   const double& snapshotPeriod = parameters.getParameter< double >( "snapshot-period" );
   static const int Dimension = Mesh::getMeshDimension();
   typedef typename MeshPointer::ObjectType MeshType;
   typedef Functions::MeshFunction< MeshType > MeshFunction;
   Pointers::SharedPointer< MeshFunction > u( this->getMesh() );
   if( initialCondition == "heaviside-vector-norm" )
   {
      typedef Functions::Analytic::VectorNorm< Dimension, RealType > VectorNormType;
      typedef Operators::Analytic::Heaviside< Dimension, RealType > HeavisideType;
      typedef Functions::OperatorFunction< HeavisideType, VectorNormType > InitialConditionType;
      String velocityFieldType = parameters.getParameter< String >( "velocity-field" );
      if( velocityFieldType == "constant" )
      {      
         typedef Operators::Analytic::Shift< Dimension, RealType > ShiftOperatorType;
         typedef Functions::OperatorFunction< ShiftOperatorType, InitialConditionType > ExactSolutionType;
         Pointers::SharedPointer<  ExactSolutionType, Devices::Host > exactSolution;
         if( ! exactSolution->getFunction().setup( parameters, prefix + "vector-norm-" ) ||
             ! exactSolution->getOperator().setup( parameters, prefix + "heaviside-" ) )
            return false;
         Containers::StaticVector< Dimension, RealType > velocity;
         for( int i = 0; i < Dimension; i++ )
            velocity[ i ] = parameters.getParameter< double >( "velocity-field-" + convertToString( i ) + "-constant" );

         Functions::MeshFunctionEvaluator< MeshFunction, ExactSolutionType > evaluator;
         RealType time( 0.0 );
         int step( 0 );
         exactSolution->getOperator().setShift( 0.0 * velocity );
         evaluator.evaluate( u, exactSolution, time );
         FileName fileName;
         fileName.setFileNameBase( "exact-u-" );
         fileName.setExtension( "tnl" );
         fileName.setIndex( step );
         u->save( fileName.getFileName() );
         while( time < finalTime )
         {
            time += snapshotPeriod;
            if( time > finalTime )
               time = finalTime;
            exactSolution->getOperator().setShift( time * velocity );            
            std::cerr << time * velocity << std::endl;
            std::cerr << exactSolution->getOperator().getShift() << std::endl;
            evaluator.evaluate( u, exactSolution, time );
            fileName.setIndex( ++step );
            u->save( fileName.getFileName() );
         }
      }
      if( velocityFieldType == "rotation" )
      {
         // TODO: implement this using RotationXY operator
      }
   }
   
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator >
bool
transportEquationProblemEoc< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     DofVectorPointer& dofs )
{
   this->bindDofs( dofs );
   //const String& initialConditionFile = parameters.getParameter< String >( "initial-condition" );
   FileName fileName;
   fileName.setFileNameBase( "exact-u-" );
   fileName.setExtension( "tnl" );
   fileName.setIndex( 0 );   
   try
   {
      this->uPointer->boundLoad( fileName.getFileName() );
   }
   catch(...)
   {
      std::cerr << "I am not able to load the initial condition from the file " << fileName.getFileName() << "." << std::endl;
      return false;
   }
   return true;
}

} // namespace TNL
