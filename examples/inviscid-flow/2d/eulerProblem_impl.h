#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>
#include <TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <TNL/Solvers/PDE/BackwardTimeDiscretisation.h>
#include "LaxFridrichsContinuity.h"
#include "LaxFridrichsEnergy.h"
#include "LaxFridrichsMomentumX.h"
#include "LaxFridrichsMomentumY.h"
#include "EulerPressureGetter.h"
#include "Euler2DVelXGetter.h"
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
setup( const MeshPointer& meshPointer,
       const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( ! this->boundaryConditionPointer->setup( meshPointer, parameters, prefix + "boundary-conditions-" ) ||
       ! this->rightHandSidePointer->setup( parameters, prefix + "right-hand-side-" ) )
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
                     MeshDependentDataPointer& meshDependentData )
{
   typedef typename MeshType::Cell Cell;
   gamma = parameters.getParameter< RealType >( "gamma" );
   RealType rhoLu = parameters.getParameter< RealType >( "NW-density" );
   RealType velLuX = parameters.getParameter< RealType >( "NW-velocityX" );
   RealType velLuY = parameters.getParameter< RealType >( "NW-velocityY" );
   RealType preLu = parameters.getParameter< RealType >( "NW-pressure" );
   RealType eLu = ( preLu / ( rhoLu * (gamma - 1) ) );
   //RealType eLu = ( preLu / (gamma - 1) ) + 0.5 * rhoLu * pow(velLuX,2)+pow(velLuY,2);
   RealType rhoLd = parameters.getParameter< RealType >( "SW-density" );
   RealType velLdX = parameters.getParameter< RealType >( "SW-velocityX" );
   RealType velLdY = parameters.getParameter< RealType >( "SW-velocityY" );
   RealType preLd = parameters.getParameter< RealType >( "SW-pressure" );
   RealType eLd = ( preLd / ( rhoLd * (gamma - 1) ) );
   //RealType eLd = ( preLd / (gamma - 1) ) + 0.5 * rhoLd * pow(velLdX,2)+pow(velLdY,2);
   RealType rhoRu = parameters.getParameter< RealType >( "NE-density" );
   RealType velRuX = parameters.getParameter< RealType >( "NE-velocityX" );
   RealType velRuY = parameters.getParameter< RealType >( "NE-velocityY" );
   RealType preRu = parameters.getParameter< RealType >( "NE-pressure" );
   RealType eRu = ( preRu / ( rhoRu * (gamma - 1) ) );
   //RealType eRu = ( preRu / (gamma - 1) ) + 0.5 * rhoRu * pow(velRuX,2)+pow(velRuY,2);
   RealType rhoRd = parameters.getParameter< RealType >( "SE-density" );
   RealType velRdX = parameters.getParameter< RealType >( "SE-velocityX" );
   RealType velRdY = parameters.getParameter< RealType >( "SE-velocityY" );
   RealType preRd = parameters.getParameter< RealType >( "SE-pressure" );
   RealType eRd = ( preRd / ( rhoRd * (gamma - 1) ) );
   //RealType eRd = ( preRd / (gamma - 1) ) + 0.5 * rhoRd * pow(velRdX,2)+pow(velRdY,2);
   RealType x0 = parameters.getParameter< RealType >( "riemann-border" );
   int size = mesh->template getEntitiesCount< Cell >();
   uRho->bind(mesh, dofs, 0);
   uRhoVelocityX->bind(mesh, dofs, size);
   uRhoVelocityY->bind(mesh, dofs, 2*size);
   uEnergy->bind(mesh, dofs, 3*size);
   Containers::Vector< RealType, DeviceType, IndexType > data;
   data.setSize(4*size);
   pressure->bind(mesh, data, 0);
   velocity->bind(mesh, data, size);
   velocityX->bind(mesh, data, 2*size);
   velocityY->bind(mesh, data, 3*size);
   for(IndexType j = 0; j < std::sqrt(size); j++)   
      for(IndexType i = 0; i < std::sqrt(size); i++)
         if ((i <= x0 * std::sqrt(size))&&(j <= x0 * std::sqrt(size)) )
            {
               (* uRho)[j*std::sqrt(size)+i] = rhoLd;
               (* uRhoVelocityX)[j*std::sqrt(size)+i] = rhoLd * velLdX;
               (* uRhoVelocityY)[j*std::sqrt(size)+i] = rhoLd * velLdY;
               (* uEnergy)[j*std::sqrt(size)+i] = eLd;
               (* velocity)[j*std::sqrt(size)+i] = std::sqrt(std::pow(velLdX,2)+std::pow(velLdY,2));
               (* velocityX)[j*std::sqrt(size)+i] = velLdX;
               (* velocityY)[j*std::sqrt(size)+i] = velLdY;
               (* pressure)[j*std::sqrt(size)+i] = preLd;
            }
         else
         if ((i <= x0 * std::sqrt(size))&&(j > x0 * std::sqrt(size)) )
            {
               (* uRho)[j*std::sqrt(size)+i] = rhoLu;
               (* uRhoVelocityX)[j*std::sqrt(size)+i] = rhoLu * velLuX;
               (* uRhoVelocityY)[j*std::sqrt(size)+i] = rhoLu * velLuY;
               (* uEnergy)[j*std::sqrt(size)+i] = eLu;
               (* velocity)[j*std::sqrt(size)+i] = std::sqrt(std::pow(velLuX,2)+std::pow(velLuY,2));
               (* velocityX)[j*std::sqrt(size)+i] = velLuX;
               (* velocityY)[j*std::sqrt(size)+i] = velLuY;
               (* pressure)[j*std::sqrt(size)+i] = preLu;
            }
         else
         if ((i > x0 * std::sqrt(size))&&(j > x0 * std::sqrt(size)) )
            {
               (* uRho)[j*std::sqrt(size)+i] = rhoRu;
               (* uRhoVelocityX)[j*std::sqrt(size)+i] = rhoRu * velRuX;
               (* uRhoVelocityY)[j*std::sqrt(size)+i] = rhoRu * velRuY;
               (* uEnergy)[j*std::sqrt(size)+i] = eRu;
               (* velocity)[j*std::sqrt(size)+i] = std::sqrt(std::pow(velRuX,2)+std::pow(velRuY,2));
               (* velocityX)[j*std::sqrt(size)+i] = velRuX;
               (* velocityY)[j*std::sqrt(size)+i] = velRuY;
               (* pressure)[j*std::sqrt(size)+i] = preRu;
            }
         else
            {
               (* uRho)[j*std::sqrt(size)+i] = rhoRd;
               (* uRhoVelocityX)[j*std::sqrt(size)+i] = rhoRd * velRdX;
               (* uRhoVelocityY)[j*std::sqrt(size)+i] = rhoRd * velRdY;
               (* uEnergy)[j*std::sqrt(size)+i] = eRd;
               (* velocity)[j*std::sqrt(size)+i] = std::sqrt(std::pow(velRdX,2)+std::pow(velRdY,2));
               (* velocityX)[j*std::sqrt(size)+i] = velRdX;
               (* velocityY)[j*std::sqrt(size)+i] = velRdY;
               (* pressure)[j*std::sqrt(size)+i] = preRd;
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
              MeshDependentDataPointer& meshDependentData )
{
  std::cout << std::endl << "Writing output at time " << time << " step " << step << "." << std::endl;
   FileName fileName;
   fileName.setExtension( "tnl" );
   fileName.setIndex( step );
   fileName.setFileNameBase( "rho-" );
   if( ! uRho->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "rhoVelX-" );
   if( ! uRhoVelocityX->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "rhoVelY-" );
   if( ! uRhoVelocityY->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "energy-" );
   if( ! uEnergy->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "velocityX-" );
   if( ! velocityX->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "velocityY-" );
   if( ! velocityY->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "velocity-" );
   if( ! velocity->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "pressue-" );
   if( ! pressure->save( fileName.getFileName() ) )
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
                MeshDependentDataPointer& meshDependentData )
{
    typedef typename MeshType::Cell Cell;
    int count = mesh->template getEntitiesCount< Cell >();
   //bind MeshFunctionType fu
   fuRho->bind( mesh, _fu, 0 );
   fuRhoVelocityX->bind( mesh, _fu, count );
   fuRhoVelocityY->bind( mesh, _fu, 2*count );
   fuEnergy->bind( mesh, _fu, 3*count );
   SharedPointer< Continuity > lF2DContinuity;
   SharedPointer< MomentumX > lF2DMomentumX;
   SharedPointer< MomentumY > lF2DMomentumY;
   SharedPointer< Energy > lF2DEnergy;

   this->bindDofs( mesh, _u );
   //rho
   lF2DContinuity->setTau(tau);
   lF2DContinuity->setVelocityX( *velocityX );
   lF2DContinuity->setVelocityY( *velocityY );
   Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, Continuity, BoundaryCondition, RightHandSide > explicitUpdaterContinuity; 
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
   Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, MomentumX, BoundaryCondition, RightHandSide > explicitUpdaterMomentumX; 
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
   Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, MomentumY, BoundaryCondition, RightHandSide > explicitUpdaterMomentumY;
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
   Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, Energy, BoundaryCondition, RightHandSide > explicitUpdaterEnergy;
   explicitUpdaterEnergy.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF2DEnergy,
                                                           this->boundaryConditionPointer,
                                                           this->rightHandSidePointer,
                                                           uEnergy,
                                                           fuEnergy );
/*
cout << "rho " << uRho.getData() << endl;
getchar();
cout << "rhoVelX " << uRhoVelocityX.getData() << endl;
getchar();
cout << "rhoVelY " << uRhoVelocityY.getData() << endl;
getchar();
cout << "Energy " << uEnergy.getData() << endl;
getchar();
cout << "velocity " << velocity.getData() << endl;
getchar();
cout << "velocityX " << velocityX.getData() << endl;
getchar();
cout << "velocityY " << velocityY.getData() << endl;
getchar();
cout << "pressure " << pressure.getData() << endl;
getchar();
*/


/*
   BoundaryConditionsSetter< MeshFunctionType, BoundaryCondition > boundaryConditionsSetter; 
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
                      MeshDependentDataPointer& meshDependentData )
{
/*   LinearSystemAssembler< Mesh,
                             MeshFunctionType,
                             DifferentialOperator,
                             BoundaryCondition,
                             RightHandSide,
                             BackwardTimeDiscretisation,
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
             MeshDependentDataPointer& meshDependentData )
{

   //velocityX
   this->velocityX->setMesh( mesh );
   VelocityX velocityXGetter( *uRho, *uRhoVelocityX );
   *this->velocityX = velocityXGetter;

   //velocityY
   this->velocityY->setMesh( mesh );
   VelocityX velocityYGetter( *uRho, *uRhoVelocityY );
   *this->velocityY = velocityYGetter;

   //velocity
   this->velocity->setMesh( mesh );
   Velocity velocityGetter( *uRho, *uRhoVelocityX, *uRhoVelocityY );
   *this->velocity = velocityGetter;

   //pressure
   this->pressure->setMesh( mesh );
   Pressure pressureGetter( *uRho, *uRhoVelocityX, *uRhoVelocityY, *uEnergy, gamma );
   *this->pressure = pressureGetter;

   return true;
}

} // namespace TNL

