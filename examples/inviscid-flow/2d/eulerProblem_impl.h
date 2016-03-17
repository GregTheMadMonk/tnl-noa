#ifndef eulerPROBLEM_IMPL_H_
#define eulerPROBLEM_IMPL_H_

#include <core/mfilename.h>
#include <matrices/tnlMatrixSetter.h>
#include <solvers/pde/tnlExplicitUpdater.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>
#include <solvers/pde/tnlBackwardTimeDiscretisation.h>

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
   double gamma = parameters.getParameter< double >( "gamma" );
   double rhoL = parameters.getParameter< double >( "left-density" );
   double velLX = parameters.getParameter< double >( "left-velocityX" );
   double velLY = parameters.getParameter< double >( "left-velocityY" );
   double preL = parameters.getParameter< double >( "left-pressure" );
   double eL = ( preL / (gamma - 1) ) + 0.5 * rhoL * pow(velLX,2)+pow(velLY,2);
   double rhoR = parameters.getParameter< double >( "right-density" );
   double velRX = parameters.getParameter< double >( "right-velocityX" );
   double velRY = parameters.getParameter< double >( "right-velocityY" );
   double preR = parameters.getParameter< double >( "right-pressure" );
   double eR = ( preR / (gamma - 1) ) + 0.5 * rhoR * pow(velRX,2)+pow(velRY,2);
   double x0 = parameters.getParameter< double >( "riemann-border" );
   int size = mesh.template getEntitiesCount< Cell >();
   int size2 = pow(size,2);
   this->rho.bind(dofs,0,size2);
   this->rhoVelX.bind(dofs,size2,size2);
   this->rhoVelY.bind(dofs,2*size2,size2);
   this->energy.bind(dofs,3*size2,size2);
   this->data.setSize(4*size2);
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
               this->velocity[j*size+i] = sqrt(pow(velLX,2)+pow(velLY,2));
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
               this->velocity[j*size+i] = sqrt(pow(velRX,2)+pow(velRY,2));
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
setupLinearSystem( const MeshType& mesh,
                   Matrix& matrix )
{
   const IndexType dofs = this->getDofs( mesh );
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
      return false;
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
                const MeshType& mesh,
                DofVectorType& _u,
                DofVectorType& _fu,
                MeshDependentDataType& meshDependentData )
{
    typedef typename MeshType::Cell Cell;
    int count = mesh.template getEntitiesCount< Cell >()/4;
	//bind _u
    this->_uRho.bind(_u,0,count);
    this->_uRhoVelocityX.bind(_u,count,count);
    this->_uRhoVelocityY.bind(_u,2 * count,count);
    this->_uEnergy.bind(_u,3 * count,count);
		
	//bind _fu
    this->_fuRho.bind(_u,0,count);
    this->_fuRhoVelocityX.bind(_u,count,count);
    this->_fuRhoVelocityY.bind(_u,2 * count,count);
    this->_fuEnergy.bind(_u,3 * count,count);

   MeshFunctionType velocity( mesh, this->velocity );
   MeshFunctionType velocityX( mesh, this->velocityX );
   MeshFunctionType velocityY( mesh, this->velocityY );
   MeshFunctionType pressure( mesh, this->pressure );
   MeshFunctionType uRho( mesh, _uRho ); 
   MeshFunctionType fuRho( mesh, _fuRho );
   MeshFunctionType uRhoVelocityX( mesh, _uRhoVelocityX ); 
   MeshFunctionType fuRhoVelocityX( mesh, _fuRhoVelocityX );
   MeshFunctionType uRhoVelocityY( mesh, _uRhoVelocityY ); 
   MeshFunctionType fuRhoVelocityY( mesh, _fuRhoVelocityY );
   MeshFunctionType uEnergy( mesh, _uEnergy ); 
   MeshFunctionType fuEnergy( mesh, _fuEnergy );
/*

   //velocityX
   EulerVelXGetter.setRhoVelX(uRhoVelocityX)
   EulerVelXGetter.setRho(uRho);
   tnlOperatorFunction< EulerVelXGetter, MeshFunction, void, time > OFVelocityX;
   velocityX = OFVelocityX;

   //velocityX
   EulerVelXGetter.setRhoVelY(uRhoVelocityY)
   EulerVelXGetter.setRho(uRho);
   tnlOperatorFunction< EulerVelYGetter, MeshFunction, void, time > OFVelocityY;
   velocityY = OFVelocityY;

   //velocityX
   EulerVelXGetter.setVelX(velocityX)
   EulerVelXGetter.setVelY(velocityY);
   tnlOperatorFunction< EulerVelGetter, MeshFunction, void, time > OFVelocity;
   velocity = OFVelocity;

   //pressure
   EulerPressureGetter.setGamma(gamma);
   EulerPressureGetter.setVelocity(velocity);
   EulerPressureGetter.setEnergy(uEnergy);
   EulerPressureGetter.setRho(uRho);
   tnlOperatorFunction< EulerPressureGetter, MeshFunction, void, time > OFPressure;
   pressure = OFPressure;

   //rho
   this->bindDofs( mesh, _u );
   tnlExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   LaxFridrichsContinuity.setTau(tau);
   LaxFridrichsContinuity.setVelocityX(velocityX);
   LaxFridrichsContinuity.setVelocityY(velocityY); 
   explicitUpdater.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           this->LaxFridrichsContinuity,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uRho,
                                                           fuRho );

   //rhoVelocityX
   tnlExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   LaxFridrichsMomentumX.setTau(tau);
   LaxFridrichsMomentumX.setVelocityX(velocityX);
   LaxFridrichsMomentumX.setVelocityY(velocityY);
   LaxFridrichsMomentumX.setPressure(pressure); 
   explicitUpdater.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           this->LaxFridrichsMomentumX,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uRhoVelocityX,
                                                           fuRhoVelocityX );

   //rhoVelocityY
   tnlExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   LaxFridrichsMomentumY.setTau(tau);
   LaxFridrichsMomentumY.setVelocityX(velocityX);
   LaxFridrichsMomentumY.setVelocityY(velocityY);
   LaxFridrichsMomentumY.setPressure(pressure); 
   explicitUpdater.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           this->LaxFridrichsMomentumY,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uRhoVelocityY,
                                                           fuRhoVelocityY );
  
   //energy
   tnlExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   LaxFridrichsEnergy.setTau(tau);
   LaxFridrichsEnergy.setVelocityX(velocityX); 
   LaxFridrichsEnergy.setVelocityY(velocityY); 
   LaxFridrichsEnergy.setPressure(pressure);
   explicitUpdater.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           this->LaxFridrichsEnergy,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uEnergy,
                                                           fuEnergy );


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
   tnlLinearSystemAssembler< Mesh,
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
                                                             b );
}

#endif /* eulerPROBLEM_IMPL_H_ */
