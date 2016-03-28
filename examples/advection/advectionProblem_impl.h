#ifndef advectionPROBLEM_IMPL_H_
#define advectionPROBLEM_IMPL_H_

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
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getTypeStatic()
{
   return tnlString( "advectionProblem< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
tnlString
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   return tnlString( "advection" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
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
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
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
typename advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs( const MeshType& mesh ) const
{
   /****
    * Return number of  DOFs (degrees of freedom) i.e. number
    * of unknowns to be resolved by the main solver.
    */
   return mesh.template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const MeshType& mesh,
          DofVectorType& dofVector )
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const tnlParameterContainer& parameters,
                     const MeshType& mesh,
                     DofVectorType& dofs,
                     MeshDependentDataType& meshDependentData )
{
   typedef typename MeshType::Cell Cell;
   int count = mesh.template getEntitiesCount< Cell >();
   const RealType& size = 1;//mesh.template getSpaceStepsProducts< -1, 0 >();
   const tnlString& beginChoice = parameters.getParameter< tnlString >( "begin" );
   int dimensions = 2; //vyresit!!!!
   if (beginChoice == "sin_square")
      {
	   double constantFunction;
	   if (dimensions == 1)
	       {
		   dofs[0] = 0;
		   double expValue;
		   for (long int i = 1; i < count; i++)
		   {
			expValue = exp(-pow(size*i-2,2));
			if ((i>0.4*count) && (i<0.5*count)) constantFunction=1; else constantFunction=0;
			if (expValue>constantFunction) dofs[i] = expValue; else dofs[i] = constantFunction;
		   };
		   dofs[count] = 0;
		};
	    if (dimensions == 2)
	       {
		   double expValue;
		   for (long int i = 1; i < count; i++)
		   for (long int j = 1; j < count; j++)
		   {
			expValue = exp(-pow(size*i-2,2)-pow(size*j-2,2));
			if ((i>0.4*count) && (i<0.5*count)&&((j>0.4*count) && (j<0.5*count))) constantFunction=1; else constantFunction=0;
			if (expValue>constantFunction) dofs[(i-1) * count -1+ j] = expValue; else dofs[(i-1) * count -1+ j] = constantFunction;
		   };
		};
       }
   else if (beginChoice == "sin")
      {
	   if (dimensions == 1)
	      {
		   dofs[0] = 0;
		   for (long int i = 1; i < count; i++)
		   {
			dofs[i] = exp(-pow(size*i-2,2));
		   };
		   dofs[count] = 0;
		};
	    if (dimensions == 2)
	       {
		   for (long int i = 1; i < count; i++)
		   for (long int j = 1; j < count; j++)
		   {
			dofs[(i-1) * count -1+ j] = exp(-pow(size*i-2,2)-pow(size*j-2,2));
		   };
		};
     };
//setting velocity field

   
   /*const tnlString& initialConditionFile = parameters.getParameter< tnlString >( "initial-condition" );
   if( ! dofs.load( initialConditionFile ) )
   {
      cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << endl;
      return false;
   }
   return true;*/
   dofs.save( "dofs.tnl" );
   this->velocityType = parameters.getParameter< tnlString >( "move" );
   const double artificalViscosity = parameters.getParameter< double >( "artifical-viscosity" );
   differentialOperator.setViscosity(artificalViscosity);
   const double advectionSpeedX = parameters.getParameter< double >( "advection-speedX" );
   differentialOperator.setAdvectionSpeedX(advectionSpeedX);
   const double advectionSpeedY = parameters.getParameter< double >( "advection-speedY" );
   differentialOperator.setAdvectionSpeedY(advectionSpeedY);
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
bool
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
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
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshType& mesh,
              DofVectorType& dofs,
              MeshDependentDataType& meshDependentData )
{
   cout << endl << "Writing output at time " << time << " step " << step << "." << endl;
   this->bindDofs( mesh, dofs );
   tnlString fileName;
   FileNameBaseNumberEnding( "u-", step, 5, ".tnl", fileName );
   if( ! dofs.save( fileName ) )
      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getExplicitRHS( const RealType& time,
                const RealType& tau,
                const MeshType& mesh,
                DofVectorType& _u,
                DofVectorType& _fu,
                MeshDependentDataType& meshDependentData )
{
   /****
    * If you use an explicit solver like tnlEulerSolver or tnlMersonSolver, you
    * need to implement this method. Compute the right-hand side of
    *
    *   d/dt u(x) = fu( x, u )
    *
    * You may use supporting mesh dependent data if you need.
    */
   typedef typename MeshType::Cell Cell;
   int count = mesh.template getEntitiesCount< Cell >();
   const RealType& size = 1;//mesh.template getSpaceStepsProducts< -1, 0 >();
   if (this->velocityType == "rotation")
   {
      double radius;
      for (int i =1; i < count; i++)
         for (int j =1; j < count; j++)
            {
               radius = sqrt(pow(i-1-(count/2.0),2) + pow(j-1-(count/2.0),2));
            if (radius != 0.0)
               _fu[(i-1)*count+j-1] =(0.25*tau)*differentialOperator.artificalViscosity*			//smoothening part
               (_u[(i-1)*count-2+j]+_u[(i-1)*count+j]+
               _u[i*count+j-1]+_u[(i-2)*count+j-1]- 
               4.0*_u[(i-1)*count+j-1])
               -((1.0/(2.0*count))*differentialOperator.advectionSpeedX						//X addition
               *radius*(-1)*((j-1-(count/2.0))/radius)
	       *(_u[(i-1)*count+j]-_u[(i-1)*count+j-2])) 
	       -((1.0/(2.0*count))*differentialOperator.advectionSpeedY						//Y addition
               *radius*((i-1-(count/2.0))/radius)
	       *(_u[i*count+j-1]-_u[(i-2)*count+j-1]))
            ;}
  }
   else if (this->velocityType == "advection")
  { 
   this->bindDofs( mesh, _u );
   tnlExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   MeshFunctionType u( mesh, _u ); 
   MeshFunctionType fu( mesh, _fu ); 
   explicitUpdater.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           this->differentialOperator,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           u,
                                                           fu );
   tnlBoundaryConditionsSetter< MeshFunctionType, BoundaryCondition > boundaryConditionsSetter; 
   boundaryConditionsSetter.template apply< typename Mesh::Cell >( 
      this->boundaryCondition, 
      time + tau, 
       u ); 
 }
}
template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
void
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
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

#endif /* advectionPROBLEM_IMPL_H_ */