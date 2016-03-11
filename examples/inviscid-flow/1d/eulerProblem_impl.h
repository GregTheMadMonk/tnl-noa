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
   return tnlString( "euler" );
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
setInitialCondition( const tnlParameterContainer& parameters,
                     const MeshType& mesh,
                     DofVectorType& dofs,
                     MeshDependentDataType& meshDependentData )
{
   typedef typename MeshType::Cell Cell;
   double gamma = parameters.getParameter< double >( "gamma" );
   double rhoL = parameters.getParameter< double >( "left-density" );
   double velL = parameters.getParameter< double >( "left-velocity" );
   double preL = parameters.getParameter< double >( "left-pressure" );
   double eL = ( preL / (gamma - 1) ) + 0.5 * rhoL * velL * velL;
   double rhoR = parameters.getParameter< double >( "right-density" );
   double velR = parameters.getParameter< double >( "right-velocity" );
   double preR = parameters.getParameter< double >( "right-pressure" );
   double eR = ( preR / (gamma - 1) ) + 0.5 * rhoR * velR * velR;
   double x0 = parameters.getParameter< double >( "riemann-border" );
   cout << gamma << " " << rhoL << " " << velL << " " << preL << " " << eL << " " << rhoR << " " << velR << " " << preR << " " << eR << " " << x0 << " " << gamma << endl;
   int count = mesh.template getEntitiesCount< Cell >()/3;
   this->rho.bind(dofs,0,count);
   this->rhoVel.bind(dofs,count,count);
   this->energy.bind(dofs,2 * count,count);
   this->data.setSize(2*count);
   this->pressure.bind(this->data,0,count);
   this->velocity.bind(this->data,count,count);
   for(long int i = 0; i < count; i++)
      if (i < x0 * count )
         {
            this->rho[i] = rhoL;
            this->rhoVel[i] = rhoL * velL;
            this->energy[i] = eL;
            this->velocity[i] = velL;
            this->pressure[i] = preL;
         }
      else
         {
            this->rho[i] = rhoR;
            this->rhoVel[i] = rhoR * velR;
            this->energy[i] = eR;
            this->velocity[i] = velR;
            this->pressure[i] = preR;
         };
   this->gamma = gamma;
   cout << "dofs = " << dofs << endl;
   getchar();
  
   
   /*
   const tnlString& initialConditionFile = parameters.getParameter< tnlString >( "initial-condition" );
   if( ! dofs.load( initialConditionFile ) )
   {
      cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << endl;
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
   ofstream vysledek;
   cout << "pressure:" << endl;
   for (int i = 0; i<100; i++) cout << this->pressure[i] << " " ;
   vysledek.open("pressure" + to_string(step) + ".txt");
      for (int i = 0; i<101; i++)
      vysledek << 0.01*i << " " << pressure[i] << endl;
   vysledek.close();
   cout << " " << endl;
   cout << "velocity:" << endl;
   for (int i = 0; i<100; i++) cout << this->velocity[i] << " " ;
   vysledek.open("velocity" + to_string(step) + ".txt");
      for (int i = 0; i<101; i++)
      vysledek << 0.01*i << " " << pressure[i] << endl;
   vysledek.close();
   cout << "energy:" << endl;
   for (int i = 0; i<100; i++) cout << this->energy[i] << " " ;
   vysledek.open("energy" + to_string(step) + ".txt");
      for (int i = 0; i<101; i++)
      vysledek << 0.01*i << " " << energy[i] << endl;
   vysledek.close();
   cout << " " << endl;
   cout << "density:" << endl;
   for (int i = 0; i<100; i++) cout << this->rho[i] << " " ;
   vysledek.open("density" + to_string(step) + ".txt");
      for (int i = 0; i<101; i++)
      vysledek << 0.01*i << " " << rho[i] << endl;
   vysledek.close();
   getchar();

   FileNameBaseNumberEnding( "rho-", step, 5, ".tnl", fileName );
   if( ! rho.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "rhoVel-", step, 5, ".tnl", fileName );
   if( ! rhoVel.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "energy-", step, 5, ".tnl", fileName );
   if( ! energy.save( fileName ) )
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
   /* 
    W[1] [0				...	count-1	]
    W[2] [count	...	2*count-1	]
    W[3] [2*count	...	3*count-1	]
    V this->velocity[]
    p this->pressure[]
    */
    typedef typename MeshType::Cell Cell;
    int count = mesh.template getEntitiesCount< Cell >()/3;
	//bind _u
    this->_uRho.bind(_u,0,count);
    this->_uRhoVelocity.bind(_u,count,count);
    this->_uEnergy.bind(_u,2 * count,count);
		
	//bind _fu
    this->_fuRho.bind(_u,0,count);
    this->_fuRhoVelocity.bind(_u,count,count);
    this->_fuEnergy.bind(_u,2 * count,count);

   MeshFunctionType velocity( mesh, this->velocity );
   MeshFunctionType pressure( mesh, this->pressure );
	//rho
   this->bindDofs( mesh, _u );
   tnlExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   MeshFunctionType uRho( mesh, _uRho ); 
   MeshFunctionType fuRho( mesh, _fuRho );
   diffrrentialOperatorRho.setTau(tau);
   differentialOperatorRho.setVelocity(velocity) 
   explicitUpdater.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           this->differentialOperatorRho,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uRho,
                                                           fuRho );
   //rhoVelocity
   MeshFunctionType uRhoVelocity( mesh, _uRhoVelocity ); 
   MeshFunctionType fuRhoVelocity( mesh, _fuRhoVelocity );
   diffrrentialOperatorRhoVelocity.setTau(tau);
   differentialOperatorRhoVelocity.setVelocity(velocity)
   differentialOperatorRhoVelocity.setPressure(pressure) 
   explicitUpdater.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           this->differentialOperatorRhoVelocity,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uRhoVelocity,
                                                           fuRhoVelocity );
   
   //energy
   MeshFunctionType uEnergy( mesh, _uEnergy ); 
   MeshFunctionType fuEnergy( mesh, _fuEnergy );
   diffrrentialOperatorEnergy.setTau(tau);
   diffrrentialOperatorEnergy.setPressure(pressure);
   diffrrentialOperatorEnergy.setVelocity(velocity); 
   explicitUpdater.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           this->differentialOperatorEnergy,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           uEnergy,
                                                           fuEnergy );
   tnlBoundaryConditionsSetter< MeshFunctionType, BoundaryCondition > boundaryConditionsSetter; 
   boundaryConditionsSetter.template apply< typename Mesh::Cell >( 
      this->boundaryCondition, 
      time + tau, 
       u );
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
