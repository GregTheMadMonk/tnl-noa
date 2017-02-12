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
          typename DifferentialOperator >
String
transportEquationProblemEoc< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getTypeStatic()
{
   return String( "transportEquationProblemEoc< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
transportEquationProblemEoc< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   return String( "Transport Equation EOC" );
}


template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
transportEquationProblemEoc< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setup( const MeshPointer& meshPointer,
       const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( ! this->velocityField->setup( meshPointer, parameters, prefix + "velocity-field-" ) ||
       ! this->differentialOperatorPointer->setup( meshPointer, parameters, prefix ) ||
       ! this->boundaryConditionPointer->setup( meshPointer, parameters, prefix + "boundary-conditions-" ) )
      return false;
   
   /****
    * Render the exact solution
    */
   const String& initialCondition = parameters.getParameter< String >( "initial-condition" );
   const double& finalTime = parameters.getParameter< double >( "final-time" );
   const double& snapshotPeriod = parameters.getParameter< double >( "snapshot-period" );
   const int Dimensions = MeshPointer::ObjectType::getMeshDimensions();
   typedef typename MeshPointer::ObjectType MeshType;
   typedef Functions::MeshFunction< MeshType > MeshFunction;
   SharedPointer< MeshFunction > u( meshPointer );
   if( initialCondition == "heaviside-sphere" )
   {
      typedef Functions::Analytic::Paraboloid< Dimensions, RealType > ParaboloidType;
      typedef Operators::Analytic::Heaviside< Dimensions, RealType > HeavisideType;
      typedef Functions::OperatorFunction< HeavisideType, ParaboloidType > InitialConditionType;
      String velocityFieldType = parameters.getParameter< String >( "velocity-field" );
      if( velocityFieldType == "constant" )
      {      
         typedef Operators::Analytic::Shift< Dimensions, RealType > ShiftOperatorType;
         typedef Functions::OperatorFunction< ShiftOperatorType, InitialConditionType > ExactSolutionType;
         SharedPointer< ExactSolutionType, Devices::Host > exactSolution;
         if( ! exactSolution->getFunction().setup( parameters, prefix ) )
            return false;
         Containers::StaticVector< Dimensions, RealType > velocity;
         for( int i = 0; i < Dimensions; i++ )
            velocity[ i ] = parameters.getParameter< double >( "velocity-field-" + String( i ) + "-constant" );

         Functions::MeshFunctionEvaluator< MeshFunction, ExactSolutionType > evaluator;
         RealType time( 0.0 );
         int step( 0 );
         exactSolution->getOperator().setShift( 0.0 * velocity );
         evaluator.evaluate( u, exactSolution, time );
         FileName fileName;
         fileName.setFileNameBase( "exact-u-" );
         fileName.setExtension( "tnl" );
         fileName.setIndex( step );
         if( ! u->save( fileName.getFileName() ) )
            return false;
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
            if( ! u->save( fileName.getFileName() ) )
               return false;
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
          typename DifferentialOperator >
bool
transportEquationProblemEoc< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     const MeshPointer& meshPointer,
                     DofVectorPointer& dofs,
                     MeshDependentDataPointer& meshDependentData )
{
   this->bindDofs( meshPointer, dofs );
   //const String& initialConditionFile = parameters.getParameter< String >( "initial-condition" );
   FileName fileName;
   fileName.setFileNameBase( "exact-u-" );
   fileName.setExtension( "tnl" );
   fileName.setIndex( 0 );   
   if( ! this->uPointer->boundLoad( fileName.getFileName() ) )
   {
      std::cerr << "I am not able to load the initial condition from the file " << fileName.getFileName() << "." << std::endl;
      return false;
   }
   return true;
}

} // namespace TNL
