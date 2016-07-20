#ifndef eulerPROBLEM_IMPL_H_
#define eulerPROBLEM_IMPL_H_

#include <TNL/core/mfilename.h>
#include <TNL/matrices/tnlMatrixSetter.h>
#include <TNL/solvers/pde/tnlExplicitUpdater.h>
#include <TNL/solvers/pde/tnlLinearSystemAssembler.h>
#include <TNL/solvers/pde/tnlBackwardTimeDiscretisation.h>
#include "LaxFridrichsContinuity.h"
#include "LaxFridrichsMomentum.h"
#include "LaxFridrichsEnergy.h"
#include "EulerVelGetter.h"
#include "EulerPressureGetter.h"

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
   return String( "euler" );
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
   return 3*mesh.template getEntitiesCount< typename MeshType::Cell >();
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
setInitialCondition( const Config::ParameterContainer& parameters,
                     const MeshType& mesh,
                     DofVectorType& dofs,
                     MeshDependentDataType& meshDependentData )
{
  std::cout << std::endl << "get conditions from CML";
   typedef typename MeshType::Cell Cell;
   this->gamma = parameters.getParameter< RealType >( "gamma" );
   RealType rhoL = parameters.getParameter< RealType >( "left-density" );
   RealType velL = parameters.getParameter< RealType >( "left-velocity" );
   RealType preL = parameters.getParameter< RealType >( "left-pressure" );
   RealType eL = ( preL / (gamma - 1) ) + 0.5 * rhoL * velL * velL;
   RealType rhoR = parameters.getParameter< RealType >( "right-density" );
   RealType velR = parameters.getParameter< RealType >( "right-velocity" );
   RealType preR = parameters.getParameter< RealType >( "right-pressure" );
   RealType eR = ( preR / (gamma - 1) ) + 0.5 * rhoR * velR * velR;
   RealType x0 = parameters.getParameter< RealType >( "riemann-border" );
   std::cout << std::endl << gamma << " " << rhoL << " " << velL << " " << preL << " " << eL << " " << rhoR << " " << velR << " " << preR << " " << eR << " " << x0 << " " << gamma << std::endl;
   int count = mesh.template getEntitiesCount< Cell >();
   std::cout << count << std::endl;
   uRho.bind(mesh, dofs, 0);
   uRhoVelocity.bind(mesh, dofs, count);
   uEnergy.bind(mesh, dofs, 2 * count);
   tnlVector < RealType, DeviceType, IndexType > data;
   data.setSize(2*count);
   velocity.bind( mesh, data, 0);
   pressure.bind( mesh, data, count );
  std::cout << std::endl << "set conditions from CML"<< std::endl;   
   for(IndexType i = 0; i < count; i++)
      if (i < x0 * count )
         {
            uRho[i] = rhoL;
            uRhoVelocity[i] = rhoL * velL;
            uEnergy[i] = eL;
            velocity[i] = velL;
            pressure[i] = preL;
         }
      else
         {
            uRho[i] = rhoR;
            uRhoVelocity[i] = rhoR * velR;
            uEnergy[i] = eR;
            velocity[i] = velR;
            pressure[i] = preR;
         };
  std::cout << "dofs = " << dofs << std::endl;
   getchar();
  
   
   /*
   const String& initialConditionFile = parameters.getParameter< String >( "initial-condition" );
   if( ! dofs.load( initialConditionFile ) )
   {
      std::cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << std::endl;
      return false;
   }
   */
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
  std::cout << std::endl << "Writing output at time " << time << " step " << step << "." << std::endl;
   this->bindDofs( mesh, dofs );
   String fileName;
   typedef typename MeshType::Cell Cell;
   int count = mesh.template getEntitiesCount< Cell >();
   std::ofstream vysledek;
/*  std::cout << "pressure:" << std::endl;
   for (IndexType i = 0; i<count; i++)std::cout << this->pressure[i] << " " << i ;
      vysledek.open("pressure" + to_string(step) + ".txt");
   for (IndexType i = 0; i<count; i++)
      vysledek << 0.01*i << " " << pressure[i] << std::endl;
   vysledek.close();
  std::cout << " " << std::endl;
  std::cout << "velocity:" << std::endl;
   for (IndexType i = 0; i<count; i++)std::cout << this->velocity[i] << " " ;
      vysledek.open("velocity" + to_string(step) + ".txt");
   for (IndexType i = 0; i<count; i++)
      vysledek << 0.01*i << " " << pressure[i] << std::endl;
   vysledek.close();
  std::cout << "energy:" << std::endl;
   for (IndexType i = 0; i<count; i++)std::cout << this->uEnergy[i] << " " ;
      vysledek.open("energy" + to_string(step) + ".txt");
   for (IndexType i = 0; i<count; i++)
      vysledek << 0.01*i << " " << uEnergy[i] << std::endl;
   vysledek.close();
  std::cout << " " << std::endl;
  std::cout << "density:" << std::endl;
   for (IndexType i = 0; i<count; i++)std::cout << this->uRho[i] << " " ;
      vysledek.open("density" + to_string(step) + ".txt");
   for (IndexType i = 0; i<count; i++)
      vysledek << 0.01*i << " " << uRho[i] << std::endl;
   vysledek.close();
*/   getchar();

   FileNameBaseNumberEnding( "rho-", step, 5, ".tnl", fileName );
   if( ! uRho.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "rhoVel-", step, 5, ".tnl", fileName );
   if( ! uRhoVelocity.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "energy-", step, 5, ".tnl", fileName );
   if( ! uEnergy.save( fileName ) )
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
   std::cout << "explicitRHS" << std::endl;
    typedef typename MeshType::Cell Cell;
    int count = mesh.template getEntitiesCount< Cell >();
	//bind _u
    this->uRho.bind(mesh, _u, 0);
    this->uRhoVelocity.bind(mesh, _u ,count);
    this->uEnergy.bind(mesh, _u, 2 * count);
		
	//bind _fu
    this->fuRho.bind(mesh, _u, 0);
    this->fuRhoVelocity.bind(mesh, _u, count);
    this->fuEnergy.bind(mesh, _u, 2 * count);

   //generating Differential operator object
   Continuity lF1DContinuity;
   Momentum lF1DMomentum;
   Energy lF1DEnergy;

   
   
  std::cout << "explicitRHSrho" << std::endl;   
   //rho
   this->bindDofs( mesh, _u );
   lF1DContinuity.setTau(tau);
   lF1DContinuity.setVelocity(velocity);
   /*tnlExplicitUpdater< Mesh, MeshFunctionType, Continuity, BoundaryCondition, RightHandSide > explicitUpdaterContinuity;
   explicitUpdaterContinuity.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF1DContinuity,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uRho,
                                                           fuRho );*/

  std::cout << "explicitRHSrhovel" << std::endl;
   //rhoVelocity
   lF1DMomentum.setTau(tau);
   lF1DMomentum.setVelocity(velocity);
   lF1DMomentum.setPressure(pressure);
   tnlExplicitUpdater< Mesh, MeshFunctionType, Momentum, BoundaryCondition, RightHandSide > explicitUpdaterMomentum;
   explicitUpdaterMomentum.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF1DMomentum,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uRhoVelocity,
                                                           fuRhoVelocity );
   
  std::cout << "explicitRHSenergy" << std::endl;
   //energy
   lF1DEnergy.setTau(tau);
   lF1DEnergy.setPressure(pressure);
   lF1DEnergy.setVelocity(velocity);
   tnlExplicitUpdater< Mesh, MeshFunctionType, Energy, BoundaryCondition, RightHandSide > explicitUpdaterEnergy;
   explicitUpdaterEnergy.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF1DEnergy,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uEnergy,
                                                           fuEnergy );  
 
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
   //velocity
   this->velocity.setMesh( mesh );
   Velocity velocityGetter( uRho, uRhoVelocity );
   this->velocity = velocityGetter;
   //pressure
   this->pressure.setMesh( mesh );
   Pressure pressureGetter( uRho, uRhoVelocity, uEnergy, gamma );
   this->pressure = pressureGetter;
}

} // namespace TNL

#endif /* eulerPROBLEM_IMPL_H_ */
