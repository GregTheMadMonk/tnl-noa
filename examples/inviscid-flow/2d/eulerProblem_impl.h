#pragma once

#include <TNL/core/mfilename.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Solvers/pde/tnlExplicitUpdater.h>
#include <TNL/Solvers/pde/tnlLinearSystemAssembler.h>
#include <TNL/Solvers/pde/tnlBackwardTimeDiscretisation.h>

#include "LaxFridrichsContinuity.h"
#include "LaxFridrichsEnergy.h"
#include "LaxFridrichsMomentumX.h"
#include "LaxFridrichsMomentumY.h"
#include "EulerPressureGetter.h"
#include "EulerVelXGetter.h"
#include "EulerVelYGetter.h"
#include "EulerVelGetter.h"

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getTypeStatic()
{
   return String( "eulerProblem< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   return String( "euler2D" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
writeProlog( Logger& logger, const Config::ParameterContainer& parameters ) const
{
   /****
    * Add data you want to have in the computation report (log) as follows:
    * logger.writeParameter< double >( "Parameter description", parameter );
    */
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setup( const Config::ParameterContainer& parameters )
{
   if( ! this->boundaryConditionPointer->setup( parameters, "boundary-conditions-" ) ||
       ! this->rightHandSidePointer->setup( parameters, "right-hand-side-" ) )
      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs( const MeshPointer& mesh ) const
{
   /****
    * Return number of  DOFs (degrees of freedom) i.e. number
    * of unknowns to be resolved by the main solver.
    */
   return 4*mesh->template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const MeshPointer& mesh,
          DofVectorPointer& dofVector )
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     const MeshPointer& mesh,
                     DofVectorPointer& dofs,
                     MeshDependentDataType& meshDependentData )
{
   typedef typename MeshType::Cell Cell;
   double gamma = parameters.getParameter< double >( "gamma" );
   double rhoL = parameters.getParameter< double >( "left-density" );
   double velLX = parameters.getParameter< double >( "left-velocityX" );
   double velLY = parameters.getParameter< double >( "left-velocityY" );
   double preL = parameters.getParameter< double >( "left-pressure" );
   double eL = ( preL / (gamma - 1) ) + 0.5 * rhoL * ::pow(velLX,2) + ::pow(velLY,2);
   double rhoR = parameters.getParameter< double >( "right-density" );
   double velRX = parameters.getParameter< double >( "right-velocityX" );
   double velRY = parameters.getParameter< double >( "right-velocityY" );
   double preR = parameters.getParameter< double >( "right-pressure" );
   double eR = ( preR / (gamma - 1) ) + 0.5 * rhoR * ::pow(velRX,2) + ::pow(velRY,2);
   double x0 = parameters.getParameter< double >( "riemann-border" );
   int size = mesh->template getEntitiesCount< Cell >();
   int size2 = size * size;
   this->rho.bind( *dofs, 0, size2 );
   this->rhoVelX.bind( *dofs, size2,size2);
   this->rhoVelY.bind( *dofs, 2*size2,size2);
   this->energy.bind( *dofs,3*size2,size2);
   this->data.setSize( 4*size2);
   this->pressure.bind(this->data,0,size2);
   this->velocity.bind(this->data,size2,size2);
   this->velocityX.bind(this->data,2*size2,size2);
   this->velocityY.bind(this->data,3*size2,size2);
   for(long int j = 0; j < size; j++)   
      for(long int i = 0; i < size; i++)
         if ((i < x0 * size)&&(j < x0 * size) )
            {
               this->rho[j*size+i] = rhoL;
               this->rhoVelX[j*size+i] = rhoL * velLX;
               this->rhoVelY[j*size+i] = rhoL * velLY;
               this->energy[j*size+i] = eL;
               this->velocity[j*size+i] = ::sqrt( ::pow(velLX,2) + ::pow(velLY,2) );
               this->velocityX[j*size+i] = velLX;
               this->velocityY[j*size+i] = velLY;
               this->pressure[j*size+i] = preL;
            }
         else
            {
               this->rho[j*size+i] = rhoR;
               this->rhoVelX[j*size+i] = rhoR * velRX;
               this->rhoVelY[j*size+i] = rhoR * velRY;
               this->energy[j*size+i] = eR;
               this->velocity[j*size+i] = ::sqrt( ::pow(velRX,2) + :: pow(velRY,2) );
               this->velocityX[j*size+i] = velRX;
               this->velocityY[j*size+i] = velRY;
               this->pressure[j*size+i] = preR;
            };
   this->gamma = gamma;
   return true; 
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
bool
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setupLinearSystem( const MeshPointer& mesh,
                   Matrix& matrix )
{
/*   const IndexType dofs = this->getDofs( mesh );
   typedef typename Matrix::CompressedRowsLengthsVector CompressedRowsLengthsVectorType;
   CompressedRowsLengthsVectorType rowLengths;
   if( ! rowLengths.setSize( dofs ) )
      return false;
   MatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, CompressedRowsLengthsVectorType > matrixSetter;
   matrixSetter.template getCompressedRowsLengths< typename Mesh::Cell >( mesh,
                                                                          differentialOperator,
                                                                          boundaryCondition,
                                                                          rowLengths );
   matrix.setDimensions( dofs, dofs );
   if( ! matrix.setCompressedRowsLengths( rowLengths ) )
      return false;*/
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshPointer& mesh,
              DofVectorPointer& dofs,
              MeshDependentDataType& meshDependentData )
{
  std::cout << std::endl << "Writing output at time " << time << " step " << step << "." << std::endl;
   this->bindDofs( mesh, dofs );
   String fileName;
   FileNameBaseNumberEnding( "rho-", step, 5, ".tnl", fileName );
   if( ! this->rho.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "rhoVelX-", step, 5, ".tnl", fileName );
   if( ! this->rhoVelX.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "rhoVelY-", step, 5, ".tnl", fileName );
   if( ! this->rhoVelY.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "energy-", step, 5, ".tnl", fileName );
   if( ! this->energy.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "velocityX-", step, 5, ".tnl", fileName );
   if( ! this->velocityX.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "velocityY-", step, 5, ".tnl", fileName );
   if( ! this->velocityY.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "velocity-", step, 5, ".tnl", fileName );
   if( ! this->velocity.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "pressure-", step, 5, ".tnl", fileName );
   if( ! this->pressure.save( fileName ) )
      return false;

   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getExplicitRHS( const RealType& time,
                const RealType& tau,
                const MeshPointer& mesh,
                DofVectorPointer& _u,
                DofVectorPointer& _fu,
                MeshDependentDataType& meshDependentData )
{
    typedef typename MeshType::Cell Cell;
    int count = mesh->template getEntitiesCount< Cell >()/4;
	//bind _u
    this->_uRho.bind( *_u,0,count);
    this->_uRhoVelocityX.bind( *_u,count,count);
    this->_uRhoVelocityY.bind( *_u,2 * count,count);
    this->_uEnergy.bind( *_u,3 * count,count);
		
	//bind _fu
    this->_fuRho.bind( *_u,0,count);
    this->_fuRhoVelocityX.bind( *_u,count,count);
    this->_fuRhoVelocityY.bind( *_u,2 * count,count);
    this->_fuEnergy.bind( *_u,3 * count,count);
   //bind MeshFunctionType
   MeshFunctionPointer velocity( mesh, this->velocity );
   MeshFunctionPointer velocityX( mesh, this->velocityX );
   MeshFunctionPointer velocityY( mesh, this->velocityY );
   MeshFunctionPointer pressure( mesh, this->pressure );
   MeshFunctionPointer uRho( mesh, _uRho ); 
   MeshFunctionPointer fuRho( mesh, _fuRho );
   MeshFunctionPointer uRhoVelocityX( mesh, _uRhoVelocityX ); 
   MeshFunctionPointer fuRhoVelocityX( mesh, _fuRhoVelocityX );
   MeshFunctionPointer uRhoVelocityY( mesh, _uRhoVelocityY ); 
   MeshFunctionPointer fuRhoVelocityY( mesh, _fuRhoVelocityY );
   MeshFunctionPointer uEnergy( mesh, _uEnergy ); 
   MeshFunctionPointer fuEnergy( mesh, _fuEnergy );
   //generate Operators
   tnlSharedPointer< Continuity > lF2DContinuity;
   tnlSharedPointer< MomentumX > lF2DMomentumX;
   tnlSharedPointer< MomentumY > lF2DMomentumY;
   tnlSharedPointer< Energy > lF2DEnergy;

   this->bindDofs( mesh, _u );
   //rho
   lF2DContinuity->setTau(tau);
   lF2DContinuity->setVelocityX( *velocityX );
   lF2DContinuity->setVelocityY( *velocityY );
   Solvers::tnlExplicitUpdater< Mesh, MeshFunctionType, Continuity, BoundaryCondition, RightHandSide > explicitUpdaterContinuity; 
   explicitUpdaterContinuity.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF2DContinuity,
                                                           this->boundaryConditionPointer,
                                                           this->rightHandSidePointer,
                                                           uRho,
                                                           fuRho );

   //rhoVelocityX
   lF2DMomentumX->setTau(tau);
   lF2DMomentumX->setVelocityX( *velocityX );
   lF2DMomentumX->setVelocityY( *velocityY );
   lF2DMomentumX->setPressure( *pressure );
   Solvers::tnlExplicitUpdater< Mesh, MeshFunctionType, MomentumX, BoundaryCondition, RightHandSide > explicitUpdaterMomentumX; 
   explicitUpdaterMomentumX.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF2DMomentumX,
                                                           this->boundaryConditionPointer,
                                                           this->rightHandSidePointer,
                                                           uRhoVelocityX,
                                                           fuRhoVelocityX );

   //rhoVelocityY
   lF2DMomentumY->setTau(tau);
   lF2DMomentumY->setVelocityX( *velocityX );
   lF2DMomentumY->setVelocityY( *velocityY );
   lF2DMomentumY->setPressure( *pressure );
   Solvers::tnlExplicitUpdater< Mesh, MeshFunctionType, MomentumY, BoundaryCondition, RightHandSide > explicitUpdaterMomentumY;
   explicitUpdaterMomentumY.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF2DMomentumY,
                                                           this->boundaryConditionPointer,
                                                           this->rightHandSidePointer,
                                                           uRhoVelocityY,
                                                           fuRhoVelocityY );
  
   //energy
   lF2DEnergy->setTau(tau);
   lF2DEnergy->setVelocityX( *velocityX ); 
   lF2DEnergy->setVelocityY( *velocityY ); 
   lF2DEnergy->setPressure( *pressure );
   Solvers::tnlExplicitUpdater< Mesh, MeshFunctionType, Energy, BoundaryCondition, RightHandSide > explicitUpdaterEnergy;
   explicitUpdaterEnergy.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF2DEnergy,
                                                           this->boundaryConditionPointer,
                                                           this->rightHandSidePointer,
                                                           uEnergy,
                                                           fuEnergy );

/*
   tnlBoundaryConditionsSetter< MeshFunctionType, BoundaryCondition > boundaryConditionsSetter; 
   boundaryConditionsSetter.template apply< typename Mesh::Cell >( 
      this->boundaryCondition, 
      time + tau, 
       u );*/
 }

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
void
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const MeshPointer& mesh,
                      DofVectorPointer& _u,
                      Matrix& matrix,
                      DofVectorPointer& b,
                      MeshDependentDataType& meshDependentData )
{
/*   tnlLinearSystemAssembler< Mesh,
                             MeshFunctionType,
                             DifferentialOperator,
                             BoundaryCondition,
                             RightHandSide,
                             tnlBackwardTimeDiscretisation,
                             Matrix,
                             DofVectorType > systemAssembler;

   MeshFunction< Mesh > u( mesh, _u );
   systemAssembler.template assembly< typename Mesh::Cell >( time,
                                                             tau,
                                                             mesh,
                                                             this->differentialOperator,
                                                             this->boundaryCondition,
                                                             this->rightHandSide,
                                                             u,
                                                             matrix,
                                                             b );*/
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
postIterate( const RealType& time,
             const RealType& tau,
             const MeshPointer& mesh,
             DofVectorPointer& dofs,
             MeshDependentDataType& meshDependentData )
{
    typedef typename MeshType::Cell Cell;
    int count = mesh->template getEntitiesCount< Cell >()/4;
	//bind _u
    this->_uRho.bind( *dofs, 0, count);
    this->_uRhoVelocityX.bind( *dofs, count, count);
    this->_uRhoVelocityY.bind( *dofs, 2 * count, count);
    this->_uEnergy.bind( *dofs, 3 * count, count);

   MeshFunctionType velocity( mesh, this->velocity );
   MeshFunctionType velocityX( mesh, this->velocityX );
   MeshFunctionType velocityY( mesh, this->velocityY );
   MeshFunctionType pressure( mesh, this->pressure );
   MeshFunctionType uRho( mesh, _uRho ); 
   MeshFunctionType uRhoVelocityX( mesh, _uRhoVelocityX ); 
   MeshFunctionType uRhoVelocityY( mesh, _uRhoVelocityY ); 
   MeshFunctionType uEnergy( mesh, _uEnergy ); 
   //Generating differential operators
   Velocity euler2DVelocity;
   VelocityX euler2DVelocityX;
   VelocityY euler2DVelocityY;
   Pressure euler2DPressure;

   //velocityX
   euler2DVelocityX.setRhoVelX(uRhoVelocityX);
   euler2DVelocityX.setRho(uRho);
//   OperatorFunction< VelocityX, MeshFunction, void, true > OFVelocityX;
//   velocityX = OFVelocityX;

   //velocityY
   euler2DVelocityY.setRhoVelY(uRhoVelocityY);
   euler2DVelocityY.setRho(uRho);
//   OperatorFunction< VelocityY, MeshFunction, void, time > OFVelocityY;
//   velocityY = OFVelocityY;

   //velocity
   euler2DVelocity.setVelX(velocityX);
   euler2DVelocity.setVelY(velocityY);
//   OperatorFunction< Velocity, MeshFunction, void, time > OFVelocity;
//   velocity = OFVelocity;

   //pressure
   euler2DPressure.setGamma(gamma);
   euler2DPressure.setVelocity(velocity);
   euler2DPressure.setEnergy(uEnergy);
   euler2DPressure.setRho(uRho);
//   OperatorFunction< euler2DPressure, MeshFunction, void, time > OFPressure;
//   pressure = OFPressure;

}

} // namespace TNL

