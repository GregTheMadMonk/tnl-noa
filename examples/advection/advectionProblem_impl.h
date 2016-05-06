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
   int dimensions = parameters.getParameter< int >( "dimension" );
   int count = mesh.template getEntitiesCount< Cell >();
   int inverseSquareCount = sqrt(count);
   const RealType& size = parameters.getParameter< RealType >( "realSize" ) / pow(count, 1.0/dimensions);
   const tnlString& beginChoice = parameters.getParameter< tnlString >( "begin" );
   tnlVector < RealType, DeviceType, IndexType > data1;
   data1.setSize(count);
   this->analyt.bind(mesh,data1);
   this->dimension = dimensions;
   this->choice = beginChoice;
   this->size = size;
   this->schemeSize =parameters.getParameter< RealType >( "realSize" );
   if (beginChoice == "square")
      {
	   if (dimensions == 1)
	       {
		   dofs[0] = 0;
		   for (IndexType i = 1; i < count-2; i++)
		   {
			if ((i>0.4*count) && (i<0.5*count)) dofs[i]=1; else dofs[i]=0;
		   };
		   dofs[count-1] = 0;
		}
	    else if (dimensions == 2)
	       {
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
                      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			if ((i>0.4*inverseSquareCount) && (i<0.5*inverseSquareCount) && (j>0.4*inverseSquareCount) && (j<0.5*inverseSquareCount))
                        dofs[i * inverseSquareCount + j]=1; else dofs[i * inverseSquareCount + j]=0;
		      };
		};
       }
   else if (beginChoice == "exp_square")
      {
	   RealType constantFunction;
	   if (dimensions == 1)
	       {
		   dofs[0] = 0;
		   RealType expValue;
		   for (IndexType i = 1; i < count-2; i++)
		   {
			expValue = exp(-pow(10/(size*count)*(size*i-0.2*size*count),2));
			if ((i>0.4*count) && (i<0.5*count)) constantFunction=1; else constantFunction=0;
			if (expValue>constantFunction) dofs[i] = expValue; else dofs[i] = constantFunction;
		   };
		   dofs[count-1] = 0;
		}
	    else if (dimensions == 2)
	       {
		   RealType expValue;
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
                      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			expValue = exp(-pow(10/(size*inverseSquareCount)*(size*i-0.2*size*inverseSquareCount),2)-pow(10/(size*inverseSquareCount)*(size*j-0.2*size*inverseSquareCount),2));
			if ((i>0.4*inverseSquareCount) && (i<0.5*inverseSquareCount) && (j>0.4*inverseSquareCount) && (j<0.5*inverseSquareCount))
                        constantFunction=1; else constantFunction=0;
			if (expValue>constantFunction) dofs[i * inverseSquareCount + j] = expValue; else dofs[i * inverseSquareCount + j]
                         = constantFunction;
		      };
		};
       }
   else if (beginChoice == "exp")
      {
	   if (dimensions == 1)
	      {
		   dofs[0] = 0;
		   for (IndexType i = 1; i < count-2; i++)
		   {
			dofs[i] = exp(-pow(10/(size*count)*(size*i-0.2*size*count),2));
		   };
		   dofs[count-1] = 0;
		}
	    else if (dimensions == 2)
	       {
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
		      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			   dofs[i * inverseSquareCount + j] = exp(-pow(10/(size*inverseSquareCount)*(size*i-0.2*size*inverseSquareCount),2)-pow(10/(size*inverseSquareCount)*(size*j-0.2*size*inverseSquareCount)*size,2));
		      };
		};
     }
   else if (beginChoice == "riemann")
      {
	   if (dimensions == 1)
	      {
		   dofs[0] = 0;
		   for (IndexType i = 1; i < count-2; i++)
		   {
			if (i < 0.5 * count ) dofs[i] = 1; else
                        dofs[i] = 0;
		   };
		   dofs[count-1] = 0;
		}
	    else if (dimensions == 2)
	       {
		   for (IndexType i = 1; i < inverseSquareCount-1; i++)
		      for (IndexType j = 1; j < inverseSquareCount-1; j++)
		      {
			   if (i < 0.5*inverseSquareCount && j < 0.5*inverseSquareCount) dofs[i * inverseSquareCount + j] = 1; else
                           dofs[i * inverseSquareCount + j] = 0;
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
   tnlString velocityType = parameters.getParameter< tnlString >( "move" );
   RealType artificalViscosity = parameters.getParameter< RealType >( "artifical-viscosity" );
   differentialOperator.setViscosity(artificalViscosity);
   tnlVector < RealType, DeviceType, IndexType > data;
   data.setSize(2*count);
   velocityX.bind(mesh, data, 0);
   velocityY.bind(mesh, data, count);
   RealType advectionSpeedX = parameters.getParameter< RealType >( "advection-speedX" );
   RealType advectionSpeedY = parameters.getParameter< RealType >( "advection-speedY" );
   this->speedX =advectionSpeedX;
   this->speedY =advectionSpeedY;
   if (velocityType == "advection")
   {
      for (IndexType i = 0; i<count; i++)
         {
            velocityX[i] = advectionSpeedX;
            velocityY[i] = advectionSpeedY;
         };
   }
   else if (velocityType == "rotation")
   {
      RealType radius;
      for(IndexType i = 0; i<count; i++)
         {
            radius = sqrt(pow(i/inverseSquareCount - (inverseSquareCount/2.0), 2) + pow(i%inverseSquareCount - (inverseSquareCount/2.0), 2))/inverseSquareCount;
            if ( radius == 0 ) {velocityX[i] = 0; velocityY[i] = 0; } else
            velocityX[i] = advectionSpeedX * radius * sin(atan(1) * 4 * (i/inverseSquareCount-inverseSquareCount/2.0) / inverseSquareCount);
            velocityY[i] = advectionSpeedX * radius * sin( (-1) * atan(1) * 4 * (i%inverseSquareCount-inverseSquareCount/2.0) / inverseSquareCount);
         };
   };
   differentialOperator.setAdvectionSpeedX(velocityX);      
   differentialOperator.setAdvectionSpeedY(velocityY);
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
   FileNameBaseNumberEnding( "a-", step, 5, ".tnl", fileName );
   if( ! this->analyt.save( fileName ) )
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
   step++;
   typedef typename MeshType::Cell Cell;
   double count = mesh.template getEntitiesCount< Cell >();
   double inverseSquareCount = sqrt(count);
   if (this->choice == "square")
      {
	   if (dimension == 1)
	       {
		   this->analyt[0];
		   for (IndexType i = 1; i < count-2; i++)
		   {
			if ((i - step * tau * (count/this->schemeSize) * this -> speedX>0.4*count) && (i - step * tau * (count/this->schemeSize) * this -> speedX<0.5*count)) analyt[i]=1; else analyt[i]=0;
		   };
		   this->analyt[count-1] = 0;
		}
	    else if (dimension == 2)
	       {
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
                      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			if ((i - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedX>0.4*inverseSquareCount) && (i - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedX<0.5*inverseSquareCount) 
                         && (j - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedY>0.4*inverseSquareCount) && (j - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedY<0.5*inverseSquareCount))
                        analyt[i]=1; else analyt[i]=0;
		      };
		};
       }
   else if (this->choice == "exp_square")
      {
	   RealType constantFunction;
	   if (dimension == 1)
	       {
		   this->analyt[0] = 0;
		   RealType expValue;
		   for (IndexType i = 1; i < count-2; i++)
		   {
			expValue = exp(-pow(this->size*i-0.2*size-step * tau * this->speedX,2));
			if ((i - step * tau * (count/this->schemeSize) * this -> speedX>0.4*count) && (i - step * tau * (count/this->schemeSize) * this -> speedX<0.5*count)) constantFunction=1; else constantFunction=0;
			if (expValue>constantFunction) this->analyt[i] = expValue; else this->analyt[i] = constantFunction;
		   };
		   this->analyt[count-1] = 0;
		}
	    else if (dimension == 2)
	       {
		   RealType expValue;
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
                      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			expValue = exp(-pow(this->size*i-0.2*size-step * tau * this->speedX,2)-pow(this->size*j-0.2*size-step * tau * this->speedY,2));
			if ((i - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedX>0.4*inverseSquareCount) && (i - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedX<0.5*inverseSquareCount) 
                         && (j - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedY>0.4*inverseSquareCount) && (j - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedY<0.5*inverseSquareCount))
                        constantFunction=1; else constantFunction=0;
			if (expValue>constantFunction) this->analyt[i * inverseSquareCount + j] = expValue; else this->analyt[i * inverseSquareCount + j]
                         = constantFunction;
		      };
		};
       }
   else if (this->choice == "exp")
      {
	   if (dimension == 1)
	      {
		   this->analyt[0] = 0;
		   for (IndexType i = 1; i < count-2; i++)
		   {
			this->analyt[i] = exp(-pow(this->size*i-0.2*size-step * tau * this->speedX,2));
		   };
		   this->analyt[count-1] = 0;
		}
	    else if (dimension == 2)
	       {
                   count = sqrt(count);
		   for (IndexType i = 1; i < inverseSquareCount-1; i++)
		      for (IndexType j = 1; j < inverseSquareCount-1; j++)
		      {
			   this->analyt[i * inverseSquareCount + j] = exp(-pow(this->size*i-0.2*size-step * tau * this->speedX,2)-pow(this->size*j-0.2*size-step * tau * this->speedY,2));
		      };
		};
     }
   else if (this->choice == "riemann")
      {
	   if (dimension == 1)
	      {
		   this->analyt[0] = 0;
		   for (IndexType i = 1; i < count-2; i++)
		   {
			if (i - step * tau * (count/this->schemeSize) * this -> speedX < 0.5 * count ) this->analyt[i] = 1; else
                        this->analyt[i] = 0;
		   };
		   this->analyt[count-1] = 0;
		}
	    else if (dimension == 2)
	       {
                   count = sqrt(count);
		   for (IndexType i = 1; i < inverseSquareCount-1; i++)
		      for (IndexType j = 1; j < inverseSquareCount-1; j++)
		      {
			   if (i - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedX < 0.5*inverseSquareCount && 
                               j - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedY < 0.5*inverseSquareCount) 
                              this->analyt[i * inverseSquareCount + j] = 1; 
                           else
                              this->analyt[i * inverseSquareCount + j] = 0;
		      };
		};
     };
   this->bindDofs( mesh, _u );
   tnlExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   MeshFunctionType u( mesh, _u ); 
   MeshFunctionType fu( mesh, _fu );
   differentialOperator.setTau(tau); 
   explicitUpdater.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           this->differentialOperator,
                                                           this->boundaryCondition,
                                                           this->rightHandSide,
                                                           u,
                                                           fu );
/*   tnlBoundaryConditionsSetter< MeshFunctionType, BoundaryCondition > boundaryConditionsSetter; 
   boundaryConditionsSetter.template apply< typename Mesh::Cell >( 
      this->boundaryCondition, 
      time + tau, 
       u ); */

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
   /*tnlLinearSystemAssembler< Mesh,
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

#endif /* advectionPROBLEM_IMPL_H_ */
