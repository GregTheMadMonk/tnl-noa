#ifndef eulerPROBLEM_IMPL_H_
#define eulerPROBLEM_IMPL_H_

#include <core/mfilename.h>
#include <matrices/tnlMatrixSetter.h>
#include <solvers/pde/tnlExplicitUpdater.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>
#include <solvers/pde/tnlBackwardTimeDiscretisation.h>

#include "LaxFridrichsContinuity.h"
#include "LaxFridrichsEnergy.h"
#include "LaxFridrichsMomentumX.h"
#include "LaxFridrichsMomentumY.h"
#include "EulerPressureGetter.h"
#include "EulerVelXGetter.h"
#include "EulerVelGetter.h"

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
tnlString
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getTypeStatic()
{
   return tnlString( "eulerProblem< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
tnlString
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   return tnlString( "euler2D" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
writeProlog( tnlLogger& logger, const tnlParameterContainer& parameters ) const
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
setup( const tnlParameterContainer& parameters )
{
   if( ! this->boundaryCondition.setup( parameters, "boundary-conditions-" ) ||
       ! this->rightHandSide.setup( parameters, "right-hand-side-" ) )
      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs( const MeshType& mesh ) const
{
   /****
    * Return number of  DOFs (degrees of freedom) i.e. number
    * of unknowns to be resolved by the main solver.
    */
   return 4*mesh.template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const MeshType& mesh,
          DofVectorType& dofVector )
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const tnlParameterContainer& parameters,
                     const MeshType& mesh,
                     DofVectorType& dofs,
                     MeshDependentDataType& meshDependentData )
{
   typedef typename MeshType::Cell Cell;
   gamma = parameters.getParameter< RealType >( "gamma" );
   RealType rhoL = parameters.getParameter< RealType >( "left-density" );
   RealType velLX = parameters.getParameter< RealType >( "left-velocityX" );
   RealType velLY = parameters.getParameter< RealType >( "left-velocityY" );
   RealType preL = parameters.getParameter< RealType >( "left-pressure" );
   RealType eL = ( preL / (gamma - 1) ) + 0.5 * rhoL * pow(velLX,2)+pow(velLY,2);
   RealType rhoR = parameters.getParameter< RealType >( "right-density" );
   RealType velRX = parameters.getParameter< RealType >( "right-velocityX" );
   RealType velRY = parameters.getParameter< RealType >( "right-velocityY" );
   RealType preR = parameters.getParameter< RealType >( "right-pressure" );
   RealType eR = ( preR / (gamma - 1) ) + 0.5 * rhoR * pow(velRX,2)+pow(velRY,2);
   RealType x0 = parameters.getParameter< RealType >( "riemann-border" );
   int size = mesh.template getEntitiesCount< Cell >();
   uRho.bind(mesh, dofs, 0);
   uRhoVelocityX.bind(mesh, dofs, size);
   uRhoVelocityY.bind(mesh, dofs, 2*size);
   uEnergy.bind(mesh, dofs, 3*size);
   tnlVector< RealType, DeviceType, IndexType > data;
   data.setSize(4*size);
   pressure.bind(mesh, data, 0);
   velocity.bind(mesh, data, size);
   velocityX.bind(mesh, data, 2*size);
   velocityY.bind(mesh, data, 3*size);
   for(IndexType j = 0; j < size; j++)   
      for(IndexType i = 0; i < size; i++)
         if ((i < x0 * size)&&(j < x0 * size) )
            {
               uRho[j*size+i] = rhoL;
               uRhoVelocityX[j*size+i] = rhoL * velLX;
               uRhoVelocityY[j*size+i] = rhoL * velLY;
               uEnergy[j*size+i] = eL;
               velocity[j*size+i] = sqrt(pow(velLX,2)+pow(velLY,2));
               velocityX[j*size+i] = velLX;
               velocityY[j*size+i] = velLY;
               pressure[j*size+i] = preL;
            }
         else
            {
               uRho[j*size+i] = rhoR;
               uRhoVelocityX[j*size+i] = rhoR * velRX;
               uRhoVelocityY[j*size+i] = rhoR * velRY;
               uEnergy[j*size+i] = eR;
               velocity[j*size+i] = sqrt(pow(velRX,2)+pow(velRY,2));
               velocityX[j*size+i] = velRX;
               velocityY[j*size+i] = velRY;
               pressure[j*size+i] = preR;
            };
   return true; 
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
bool
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setupLinearSystem( const MeshType& mesh,
                   Matrix& matrix )
{
/*   const IndexType dofs = this->getDofs( mesh );
   typedef typename Matrix::CompressedRowsLengthsVector CompressedRowsLengthsVectorType;
   CompressedRowsLengthsVectorType rowLengths;
   if( ! rowLengths.setSize( dofs ) )
      return false;
   tnlMatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, CompressedRowsLengthsVectorType > matrixSetter;
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
              const MeshType& mesh,
              DofVectorType& dofs,
              MeshDependentDataType& meshDependentData )
{
   cout << endl << "Writing output at time " << time << " step " << step << "." << endl;
   this->bindDofs( mesh, dofs );
   tnlString fileName;
   FileNameBaseNumberEnding( "rho-", step, 5, ".tnl", fileName );
   if( ! this->uRho.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "rhoVelX-", step, 5, ".tnl", fileName );
   if( ! this->uRhoVelocityX.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "rhoVelY-", step, 5, ".tnl", fileName );
   if( ! this->uRhoVelocityY.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "energy-", step, 5, ".tnl", fileName );
   if( ! this->uEnergy.save( fileName ) )
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
                const MeshType& mesh,
                DofVectorType& _u,
                DofVectorType& _fu,
                MeshDependentDataType& meshDependentData )
{
    typedef typename MeshType::Cell Cell;
    int count = mesh.template getEntitiesCount< Cell >();
   //bind MeshFunctionType fu
   fuRho.bind( mesh, _fu, 0 );
   fuRhoVelocityX.bind( mesh, _fu, count );
   fuRhoVelocityY.bind( mesh, _fu, 2*count );
   fuEnergy.bind( mesh, _fu, 3*count );
   //generate Operators
   Continuity lF2DContinuity;
   MomentumX lF2DMomentumX;
   MomentumY lF2DMomentumY;
   Energy lF2DEnergy;

   this->bindDofs( mesh, _u );
   //rho
   lF2DContinuity.setTau(tau);
   lF2DContinuity.setVelocityX(velocityX);
   lF2DContinuity.setVelocityY(velocityY);
   tnlExplicitUpdater< Mesh, MeshFunctionType, Continuity, BoundaryCondition, RightHandSide > explicitUpdaterContinuity; 
   explicitUpdaterContinuity.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF2DContinuity,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uRho,
                                                           fuRho );

   //rhoVelocityX
   lF2DMomentumX.setTau(tau);
   lF2DMomentumX.setVelocityX(velocityX);
   lF2DMomentumX.setVelocityY(velocityY);
   lF2DMomentumX.setPressure(pressure);
   tnlExplicitUpdater< Mesh, MeshFunctionType, MomentumX, BoundaryCondition, RightHandSide > explicitUpdaterMomentumX; 
   explicitUpdaterMomentumX.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF2DMomentumX,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uRhoVelocityX,
                                                           fuRhoVelocityX );

   //rhoVelocityY
   lF2DMomentumY.setTau(tau);
   lF2DMomentumY.setVelocityX(velocityX);
   lF2DMomentumY.setVelocityY(velocityY);
   lF2DMomentumY.setPressure(pressure);
   tnlExplicitUpdater< Mesh, MeshFunctionType, MomentumY, BoundaryCondition, RightHandSide > explicitUpdaterMomentumY;
   explicitUpdaterMomentumY.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF2DMomentumY,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uRhoVelocityY,
                                                           fuRhoVelocityY );
  
   //energy
   lF2DEnergy.setTau(tau);
   lF2DEnergy.setVelocityX(velocityX); 
   lF2DEnergy.setVelocityY(velocityY); 
   lF2DEnergy.setPressure(pressure);
   tnlExplicitUpdater< Mesh, MeshFunctionType, Energy, BoundaryCondition, RightHandSide > explicitUpdaterEnergy;
   explicitUpdaterEnergy.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF2DEnergy,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
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
                      const MeshType& mesh,
                      DofVectorType& _u,
                      Matrix& matrix,
                      DofVectorType& b,
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

   tnlMeshFunction< Mesh > u( mesh, _u );
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
             const MeshType& mesh,
             DofVectorType& dofs,
             MeshDependentDataType& meshDependentData )
{
   //velocityX
   this->velocityX.setMesh( mesh );
   VelocityX velocityXGetter( uRho, uRhoVelocityX );
   this->velocityX = velocityXGetter;
   //velocityY
   this->velocityY.setMesh( mesh );
   VelocityX velocityYGetter( uRho, uRhoVelocityY );
   this->velocityY = velocityYGetter;
   //velocity
   this->velocity.setMesh( mesh );
   Velocity velocityGetter( uRho, uRhoVelocityX, uRhoVelocityY );
   this->velocity = velocityGetter;
   //pressure
   this->pressure.setMesh( mesh );
   Pressure pressureGetter( uRho, uRhoVelocityX, uRhoVelocityY, uEnergy, gamma );
   this->pressure = pressureGetter;

}

#endif /* eulerPROBLEM_IMPL_H_ */
