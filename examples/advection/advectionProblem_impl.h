#ifndef advectionPROBLEM_IMPL_H_
#define advectionPROBLEM_IMPL_H_

#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>
#include <TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <TNL/Solvers/PDE/BackwardTimeDiscretisation.h>

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getTypeStatic()
{
   return String( "advectionProblem< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   return String( "advection" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
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
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
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
typename advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs( const MeshPointer& mesh ) const
{
   /****
    * Return number of  DOFs (degrees of freedom) i.e. number
    * of unknowns to be resolved by the main solver.
    */
   return mesh->template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const MeshPointer& mesh,
          DofVectorPointer& dofVector )
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     const MeshPointer& mesh,
                     DofVectorPointer& dofs,
                     MeshDependentDataPointer& meshDependentData )
{
   typedef typename MeshType::Cell Cell;
   int dimensions = parameters.getParameter< int >( "dimension" );
   int count = mesh->template getEntitiesCount< Cell >();
   int inverseSquareCount = std::sqrt(count);
   int inverseCubeCount = std::pow(count, (1.0/3.0));
   const RealType& size = parameters.getParameter< RealType >( "realSize" ) / ::pow(count, 1.0/dimensions);
   const String& beginChoice = parameters.getParameter< String >( "begin" );
   Containers::Vector < RealType, DeviceType, IndexType > data1;
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
		   ( *dofs )[0] = 0;
		   for (IndexType i = 1; i < count-2; i++)
		   {
			if ((i>0.4*count) && (i<0.5*count)) ( *dofs )[i]=1; else ( *dofs )[i]=0;
		   };
		   ( *dofs )[count-1] = 0;
		}
	    else if (dimensions == 2)
	       {
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
                      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			if ((i>0.4*inverseSquareCount) && (i<0.5*inverseSquareCount) && (j>0.4*inverseSquareCount) && (j<0.5*inverseSquareCount))
                        ( *dofs )[i * inverseSquareCount + j]=1; else ( *dofs)[i * inverseSquareCount + j]=0;
		      };
		}
           else if (dimensions == 3)
	       {
		   for (IndexType i = 0; i < inverseCubeCount-1; i++)
                      for (IndexType j = 0; j < inverseCubeCount-1; j++)
                         for (IndexType k = 0; k < inverseCubeCount-1; k++)
		           {
			     if ((i>0.4*inverseCubeCount) && (i<0.5*inverseCubeCount) && (k>0.4*inverseCubeCount) && (k<0.5*inverseCubeCount) && (j>0.4*inverseCubeCount) && (j<0.5*inverseCubeCount))
                             ( *dofs )[i * std::pow(inverseCubeCount,3) + j * inverseCubeCount + k]=1; else ( *dofs)[i * std::pow(inverseCubeCount,3) + j * inverseCubeCount + k]=0;
		           };
		};
       }
   else if (beginChoice == "exp_square")
      {
	   RealType constantFunction;
	   if (dimensions == 1)
	       {
		   ( *dofs )[0] = 0;
		   RealType expValue;
		   for (IndexType i = 1; i < count-2; i++)
		   {
			expValue = std::exp(-std::pow(10/(size*count)*(size*i-0.2*size*count),2));
			if ((i>0.4*count) && (i<0.5*count)) constantFunction=1; else constantFunction=0;
			if (expValue>constantFunction) ( *dofs )[i] = expValue; else ( *dofs )[i] = constantFunction;
		   };
		   ( *dofs )[count-1] = 0;
		}
	    else if (dimensions == 2)
	       {
		   RealType expValue;
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
                      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			expValue = std::exp(-std::pow(10/(size*inverseSquareCount)*(size*i-0.2*size*inverseSquareCount),2)-std::pow(10/(size*inverseSquareCount)*(size*j-0.2*size*inverseSquareCount),2));
			if ((i>0.4*inverseSquareCount) && (i<0.5*inverseSquareCount) && (j>0.4*inverseSquareCount) && (j<0.5*inverseSquareCount))
                        constantFunction=1; else constantFunction=0;
			if (expValue>constantFunction) (* dofs )[i * inverseSquareCount + j] = expValue; else (* dofs )[i * inverseSquareCount + j]
                         = constantFunction;
		      };
		}
            else if (dimensions == 3)
	       {
		   RealType expValue;
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
                      for (IndexType j = 0; j < inverseSquareCount-1; j++)
                         for (IndexType k = 0; k < inverseSquareCount-1; k++)
		         {
			   expValue = std::exp(-std::pow(10/(size*inverseCubeCount)*(size*i-0.2*size*inverseCubeCount),2)-std::pow(10/(size*inverseCubeCount)*(size*j-0.2*size*inverseCubeCount),2)-std::pow(10/(size*inverseCubeCount)*(size*k-0.2*size*inverseCubeCount),2));
			   if ((i>0.4*inverseCubeCount) && (i<0.5*inverseCubeCount) && (j>0.4*inverseCubeCount) && (j<0.5*inverseCubeCount)&& (k>0.4*inverseCubeCount) && (k<0.5*inverseCubeCount))
                           constantFunction=1; else constantFunction=0;
			   if (expValue>constantFunction) (* dofs )[i * std::pow(inverseCubeCount,2) + j * inverseCubeCount + k] = expValue; else (* dofs )[i * std::pow(inverseCubeCount,2) + j * inverseCubeCount + k]
                            = constantFunction;
		      };
		};
       }
   else if (beginChoice == "exp")
      {
	   if (dimensions == 1)
	      {
		   ( *dofs )[0] = 0;
		   for (IndexType i = 1; i < count-2; i++)
		   {
			( *dofs )[i] = std::exp(-std::pow(10/(size*count)*(size*i-0.2*size*count),2));
		   };
		   ( *dofs )[count-1] = 0;
		}
	    else if (dimensions == 2)
	       {
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
		      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			   ( *dofs )[i * inverseSquareCount + j] = std::exp(-std::pow(10/(size*inverseSquareCount)*(size*i-0.2*size*inverseSquareCount),2)-std::pow(10/(size*inverseSquareCount)*(size*j-0.2*size*inverseSquareCount),2));
		      };
		}
             else if (dimensions == 3)
	       {
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
		      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		         for (IndexType k = 0; k < inverseSquareCount-1; k++)
		         {
			      ( *dofs )[i * std::pow(inverseCubeCount,2) + j * inverseCubeCount + k] = std::exp(-std::pow(10/(size*inverseCubeCount)*(size*i-0.2*size*inverseCubeCount),2)-std::pow(10/(size*inverseCubeCount)*(size*j-0.2*size*inverseCubeCount),2) - std::pow(10/(size*inverseCubeCount)*(size*k-0.2*size*inverseCubeCount),2));
		         };
		};
     }
   else if (beginChoice == "riemann")
      {
	   if (dimensions == 1)
	      {
		   ( *dofs )[0] = 0;
		   for (IndexType i = 1; i < count-2; i++)
		   {
			if (i < 0.5 * count ) ( *dofs )[i] = 1; else
                        ( *dofs )[i] = 0;
		   };
		   ( *dofs )[count-1] = 0;
		}
	    else if (dimensions == 2)
	       {
		   for (IndexType i = 1; i < inverseSquareCount-1; i++)
		      for (IndexType j = 1; j < inverseSquareCount-1; j++)
		      {
			   if (i < 0.5*inverseSquareCount && j < 0.5*inverseSquareCount) ( *dofs )[i * inverseSquareCount + j] = 1; else
                           ( *dofs )[i * inverseSquareCount + j] = 0;
		      };
		}
	    else if (dimensions == 3)
	       {
		   for (IndexType i = 1; i < inverseSquareCount-1; i++)
		      for (IndexType j = 1; j < inverseSquareCount-1; j++)
		         for (IndexType k = 1; k < inverseSquareCount-1; k++)
		         {
			      if (i < 0.5*inverseSquareCount && j < 0.5*inverseSquareCount && k < 0.5*inverseSquareCount) ( *dofs )[i * std::pow(inverseCubeCount,2) + j * inverseCubeCount + k] = 1; else
                              ( *dofs )[i * std::pow(inverseCubeCount,2) + j * inverseCubeCount + k] = 0;
		         };
		};
     };
   //setting velocity field   
   /*const tnlString& initialConditionFile = parameters.getParameter< tnlString >( "initial-condition" );
   if( ! dofs.load( initialConditionFile ) )
   {
      std::cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << std::endl;
      return false;
   }
   return true;*/
   ( *dofs ).save( "dofs.tnl" );
   String velocityType = parameters.getParameter< String >( "move" );
   RealType artificalViscosity = parameters.getParameter< RealType >( "artifical-viscosity" );
   differentialOperatorPointer->setViscosity(artificalViscosity);
   Containers::Vector < RealType, DeviceType, IndexType > data;
   data.setSize(3*count);
   ( *velocityX).bind(mesh, data, 0);
   ( *velocityY).bind(mesh, data, count);
   ( *velocityZ).bind(mesh, data, 2*count);
   RealType advectionSpeedX = parameters.getParameter< RealType >( "advection-speedX" );
   RealType advectionSpeedY = parameters.getParameter< RealType >( "advection-speedY" );
   RealType advectionSpeedZ = parameters.getParameter< RealType >( "advection-speedZ" );
   this->speedX =advectionSpeedX;
   this->speedY =advectionSpeedY;
   this->speedZ =advectionSpeedZ;
   if (velocityType == "advection")
   {
      for (IndexType i = 0; i<count; i++)
         {
            ( *velocityX) [i] = advectionSpeedX;
            ( *velocityY) [i] = advectionSpeedY;
            ( *velocityZ) [i] = advectionSpeedZ;
         };
   }
   else if (velocityType == "rotation")
   {
      RealType radius;
      if(dimensions == 2)
         for(IndexType i = 0; i<count; i++)
            {
               radius = ::sqrt(::pow(i/inverseSquareCount - (inverseSquareCount/2.0), 2) + std::pow(i%inverseSquareCount - (inverseSquareCount/2.0), 2))/inverseSquareCount;
               if ( radius == 0 ) {( *velocityX) [i] = 0; ( *velocityY) [i] = 0;} else {
               (* velocityX) [i] = advectionSpeedX * radius * std::sin(std::atan(1) * 4 * (i/inverseSquareCount-inverseSquareCount/2.0) / inverseSquareCount);
               (* velocityY) [i] = advectionSpeedX * radius * std::sin( (-1) * std::atan(1) * 4 * (i%inverseSquareCount-inverseSquareCount/2.0) / inverseSquareCount);
               }
            }
/*       else if(dimensions == 3)
          for(IndexType i = 0; i<count; i++)
             {
               radius = ::sqrt(::pow(i/inverseCubeCount - (inverseCubeCount/2.0), 2) + std::pow(i%inverseCubeCount - (inverseCubeCount/2.0), 2))/inverseCubeCount;
               if ( radius == 0 ) {( *velocityX) [i] = 0; ( *velocityY) [i] = 0; ( *velocityZ)[i] = 0; } else {
               (* velocityX) [i] = advectionSpeedX * radius * std::sin(std::atan(1) * 4 * (i+(std::pow(inverseCubeCount,2))/inverseCubeCount-inverseCubeCount/2.0) / inverseCubeCount);
               //(* velocityY) [i] = advectionSpeedX * radius * std::sin( (-1) * std::atan(1) * 4 * (i%(std::pow(inverseCubeCount,2))%inverseCubeCount-inverseCubeCount/2.0) / inverseCubeCount);
               //(* velocityZ) [i] = advectionspeedX * radius * std::sin( (-1) * std::atan(1) * 4 * (i/(std::pow(inverseCubeCount,2))-inverseCubeCount/2.0) / inverseCubeCount);
               }
             }*/
   };
   differentialOperatorPointer->setAdvectionSpeedX((* velocityX));      
   differentialOperatorPointer->setAdvectionSpeedY((* velocityY));
   differentialOperatorPointer->setAdvectionSpeedZ((* velocityZ));
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
bool
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setupLinearSystem( const MeshPointer& mesh,
                   Matrix& matrix )
{
   const IndexType dofs = this->getDofs( mesh );
   typedef typename Matrix::ObjectType::CompressedRowsLengthsVector CompressedRowsLengthsVectorType;
   SharedPointer< CompressedRowsLengthsVectorType > rowLengths;
   if( ! rowLengths->setSize( dofs ) )
      return false;
   Matrices::MatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, CompressedRowsLengthsVectorType > matrixSetter;
   matrixSetter.template getCompressedRowsLengths< typename Mesh::Cell >( mesh,
                                                                          differentialOperatorPointer,
                                                                          boundaryConditionPointer,
                                                                          rowLengths );
   matrix->setDimensions( dofs, dofs );
   if( ! matrix->setCompressedRowsLengths( *rowLengths ) )
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
              const MeshPointer& mesh,
              DofVectorPointer& dofs,
              MeshDependentDataPointer& meshDependentData )
{
   std::cout << std::endl << "Writing output at time " << time << " step " << step << "." << std::endl;
   this->bindDofs( mesh, dofs );
   FileName fileNameU;
   fileNameU.setFileNameBase( "u-" );
   fileNameU.setExtension( "tnl" );
   fileNameU.setIndex( step );
   if( ! dofs->save( fileNameU.getFileName() ) )
      return false;
   FileName fileNameAnalyt;
   fileNameAnalyt.setFileNameBase( "a-" );
   fileNameAnalyt.setExtension( "tnl" );
   fileNameAnalyt.setIndex( step );
   if( ! analyt.save( fileNameAnalyt.getFileName() ) )
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
                const MeshPointer& mesh,
                DofVectorPointer& _u,
                DofVectorPointer& _fu,
                MeshDependentDataPointer& meshDependentData )
{
   step++;
   typedef typename MeshType::Cell Cell;
   int count = mesh->template getEntitiesCount< typename MeshType::Cell >();
   int inverseSquareCount = std::sqrt(count);
   int inverseCubeCount = std::pow(count, (1.0/3.0));
   if (tau > 10e-9)
      {
   if (this->choice == "square")
      {
	   if (dimension == 1)
	       {
		   this->analyt[0] = 0;
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
                        analyt[i * inverseSquareCount + j]=1; else analyt[i * inverseSquareCount + j]=0;
		      };
		}
	    else if (dimension == 3)
	       {
		   for (IndexType i = 0; i < inverseCubeCount-1; i++)
                      for (IndexType j = 0; j < inverseCubeCount-1; j++)
                         for (IndexType k = 0; k < inverseCubeCount-1; k++)
		         {
			   if ((i - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedX>0.4*inverseCubeCount) && (i - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedX<0.5*inverseCubeCount) 
                            && (j - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedY>0.4*inverseCubeCount) && (j - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedY<0.5*inverseCubeCount)
                            && (k - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedZ>0.4*inverseCubeCount) && (k - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedZ<0.5*inverseCubeCount))
                           analyt[i * std::pow(inverseCubeCount,2) + j * inverseCubeCount + k]=1; else analyt[i * std::pow(inverseCubeCount,2) + j * inverseCubeCount + k]=0;
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
			expValue = std::exp(-std::pow(10/(size*count)*(size*i-0.2*size*count)-step * 10 * tau * this->speedX,2));
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
			expValue = std::exp(-std::pow(10/(size*inverseSquareCount)*(size*i-0.2*size*inverseSquareCount)-step * 10 * tau * this->speedX,2)-std::pow(10/(size*inverseSquareCount)*(size*j-0.2*size*inverseSquareCount)-step * 10 * tau * this->speedY,2));
			if ((i - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedX>0.4*inverseSquareCount) && (i - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedX<0.5*inverseSquareCount) 
                         && (j - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedY>0.4*inverseSquareCount) && (j - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedY<0.5*inverseSquareCount))
                        constantFunction=1; else constantFunction=0;
			if (expValue>constantFunction) this->analyt[i * inverseSquareCount + j] = expValue; else this->analyt[i * inverseSquareCount + j]
                         = constantFunction;
		      };
		}
	    else if (dimension == 3)
	       {
		   RealType expValue;
		   for (IndexType i = 0; i < inverseCubeCount-1; i++)
                      for (IndexType j = 0; j < inverseCubeCount-1; j++)
                         for (IndexType k = 0; k < inverseCubeCount-1; k++)
		         {
			   expValue = std::exp(-std::pow(10/(size*inverseCubeCount)*(size*k-0.2*size*inverseCubeCount)-step * 10 * tau * this->speedX,2)-std::pow(10/(size*inverseCubeCount)*(size*j-0.2*size*inverseCubeCount)-step * 10 * tau * this->speedY,2)-std::pow(10/(size*inverseCubeCount)*(size*i-0.2*size*inverseCubeCount)-step * 10 * tau * this->speedZ,2));
			   if ((i - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedX>0.4*inverseCubeCount) && (i - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedX<0.5*inverseCubeCount) 
                            && (j - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedY>0.4*inverseCubeCount) && (j - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedY<0.5*inverseCubeCount)
                            && (k - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedZ>0.4*inverseCubeCount) && (k - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedZ<0.5*inverseCubeCount))
                           constantFunction=1; else constantFunction=0;
			   if (expValue>constantFunction) this->analyt[i * std::pow(inverseCubeCount,2) + j * inverseCubeCount + k] = expValue; else this->analyt[i * std::pow(inverseCubeCount,2) + j * inverseCubeCount + k]
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
			this->analyt[i] = std::exp(-std::pow(10/(size*count)*(size*i-0.2*size*count)-step * 10 * tau * this->speedX,2));
		   };
		   this->analyt[count-1] = 0;
		}
	    else if (dimension == 2)
	       {
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
		      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			   this->analyt[i * inverseSquareCount + j] = std::exp(-std::pow(10/(size*inverseSquareCount)*(size*i-0.2*size*inverseSquareCount)-step * 10 * tau * this->speedX,2)-std::pow(10/(size*inverseSquareCount)*(size*j-0.2*size*inverseSquareCount)-step * 10 * tau * this->speedY,2));
		      };
		}
	    else if (dimension == 3)
	       {
		   for (IndexType i = 0; i < inverseCubeCount-1; i++)
		      for (IndexType j = 0; j < inverseCubeCount-1; j++)
		         for (IndexType k = 0; k < inverseCubeCount-1; k++)
		         {
			      this->analyt[i * std::pow(inverseCubeCount,2) + j * inverseCubeCount + k] = std::exp(-std::pow(10/(size*inverseCubeCount)*(size*i-0.2*size*inverseCubeCount)-step * 10 * tau * this->speedX,2)-std::pow(10/(size*inverseCubeCount)*(size*j-0.2*size*inverseCubeCount)-step * 10 * tau * this->speedY,2)-std::pow(10/(size*inverseCubeCount)*(size*k-0.2*size*inverseCubeCount)-step * 10 * tau * this->speedZ,2));
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
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
		      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			   if (i - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedX < 0.5*inverseSquareCount && 
                               j - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedY < 0.5*inverseSquareCount) 
                              this->analyt[i * inverseSquareCount + j] = 1; 
                           else
                              this->analyt[i * inverseSquareCount + j] = 0;
		      };
		}
	    else if (dimension == 3)
	       {
		   for (IndexType i = 0; i < inverseCubeCount-1; i++)
		      for (IndexType j = 0; j < inverseCubeCount-1; j++)
		         for (IndexType k = 0; k < inverseCubeCount-1; k++)
		         {
			   if (i - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedX < 0.5*inverseCubeCount && 
                               j - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedY < 0.5*inverseCubeCount &&
                               k - step * tau * (inverseCubeCount/this->schemeSize) * this -> speedZ < 0.5*inverseCubeCount) 
                              this->analyt[i * std::pow(inverseCubeCount,2) + j * inverseCubeCount + k] = 1; 
                           else
                              this->analyt[i * std::pow(inverseCubeCount,2) + j * inverseCubeCount + k] = 0;
		         };
		};
     };
     };
   /*
   cout << step << endl;
   cout << tau << endl;
   cout << this->speedX << endl;
   cout << step * 10 * tau * this->speedX<< endl;*/
   this->bindDofs( mesh, _u );
   Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   SharedPointer< MeshFunctionType > u( mesh, _u ); 
   SharedPointer< MeshFunctionType > fu( mesh, _fu );
   differentialOperatorPointer->setTau(tau); 
   explicitUpdater.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           this->differentialOperatorPointer,
                                                           this->boundaryConditionPointer,
                                                           this->rightHandSidePointer,
                                                           u,
                                                           fu );
/*   BoundaryConditionsSetter< MeshFunctionType, BoundaryCondition > boundaryConditionsSetter; 
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
                      const MeshPointer& mesh,
                      DofVectorPointer& _u,
                      Matrix& matrix,
                      DofVectorPointer& b,
                      MeshDependentDataPointer& meshDependentData )
{
   /*LinearSystemAssembler< Mesh,
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

} // namespace TNL

#endif /* advectionPROBLEM_IMPL_H_ */
