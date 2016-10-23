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
<<<<<<< HEAD
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
=======
   std::cout << "vaules adding";
   typedef typename MeshType::Cell Cell;
   int dimensions = parameters.getParameter< int >( "dimension" );
   int count = mesh->template getEntitiesCount< Cell >();
   const RealType& size = parameters.getParameter< double >( "realSize" ) / ::pow(count, 1.0/dimensions);
   const String& beginChoice = parameters.getParameter< String >( "begin" );
   std::cout << beginChoice << " " << dimensions << "   " << size << "   " << count << "   "<< 1/dimensions << std::endl;
   getchar();
   if (beginChoice == "sin_square")
>>>>>>> develop
      {
	   if (dimensions == 1)
	       {
<<<<<<< HEAD
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
=======
                  std::cout << "adding DOFS" << std::endl;
		   ( *dofs )[0] = 0;
		   double expValue;
		   for (IndexType i = 1; i < count-2; i++)
		   {
			expValue = std::exp(-std::pow(size*i-2,2));
			if( (i>0.4*count) && (i<0.5*count))
                            constantFunction=1;
                        else
                            constantFunction=0;
			if(expValue>constantFunction)
                           ( *dofs )[i] = expValue;
                        else ( *dofs )[i] = constantFunction;
>>>>>>> develop
		   };
		   ( *dofs )[count-1] = 0;
		}
	    else if (dimensions == 2)
	       {
<<<<<<< HEAD
		   RealType expValue;
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
                      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			expValue = exp(-pow(10/(size*inverseSquareCount)*(size*i-0.2*size*inverseSquareCount),2)-pow(10/(size*inverseSquareCount)*(size*j-0.2*size*inverseSquareCount),2));
			if ((i>0.4*inverseSquareCount) && (i<0.5*inverseSquareCount) && (j>0.4*inverseSquareCount) && (j<0.5*inverseSquareCount))
                        constantFunction=1; else constantFunction=0;
			if (expValue>constantFunction) dofs[i * inverseSquareCount + j] = expValue; else dofs[i * inverseSquareCount + j]
                         = constantFunction;
=======
                   count = std::sqrt(count);
		   double expValue;
		   for (IndexType i = 0; i < count-1; i++)
                      for (IndexType j = 0; j < count-1; j++)
		      {
			expValue = std::exp(-std::pow(size*i-2,2)-std::pow(size*j-2,2));
			if( (i>0.4*count) && (i<0.5*count) && (j>0.4*count) && (j<0.5*count) )
                           constantFunction=1;
                        else constantFunction=0;
			if( expValue>constantFunction)
                            ( *dofs )[i * count + j] = expValue;
                        else ( *dofs )[i * count + j] = constantFunction;
>>>>>>> develop
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
<<<<<<< HEAD
			dofs[i] = exp(-pow(10/(size*count)*(size*i-0.2*size*count),2));
=======
			( *dofs )[i] = std::exp(-std::pow(size*i-2,2));
>>>>>>> develop
		   };
		   ( *dofs )[count-1] = 0;
		}
	    else if (dimensions == 2)
	       {
<<<<<<< HEAD
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
		      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			   dofs[i * inverseSquareCount + j] = exp(-pow(10/(size*inverseSquareCount)*(size*i-0.2*size*inverseSquareCount),2)-pow(10/(size*inverseSquareCount)*(size*j-0.2*size*inverseSquareCount),2));
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
=======
                   count = ::sqrt(count);
		   for (IndexType i = 1; i < count-1; i++)
		      for (IndexType j = 1; j < count-1; j++)
		      {
			   ( *dofs )[i * count + j] = std::exp(-std::pow(size*i-2,2)-std::pow(size*j-2,2));
		      };
		};
     };
   //setting velocity field
   std::cout << *dofs << std::endl;
   getchar();
   /*const String& initialConditionFile = parameters.getParameter< String >( "initial-condition" );
>>>>>>> develop
   if( ! dofs.load( initialConditionFile ) )
   {
      std::cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << std::endl;
      return false;
   }
   return true;*/
<<<<<<< HEAD
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
=======
   dofs->save( "dofs.tnl" );
   this->velocityType = parameters.getParameter< String >( "move" );
   const double artificalViscosity = parameters.getParameter< double >( "artifical-viscosity" );
   differentialOperatorPointer->setViscosity(artificalViscosity);
   const double advectionSpeedX = parameters.getParameter< double >( "advection-speedX" );
   differentialOperatorPointer->setAdvectionSpeedX(advectionSpeedX);
   const double advectionSpeedY = parameters.getParameter< double >( "advection-speedY" );
   differentialOperatorPointer->setAdvectionSpeedY(advectionSpeedY);
   std::cout << "vaules added";
>>>>>>> develop
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
<<<<<<< HEAD
   tnlString fileName;
   MeshFunctionType dofsh;
   dofsh.bind(mesh,dofs);
   FileNameBaseNumberEnding( "u-", step, 5, ".tnl", fileName );
   if( ! dofsh.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "a-", step, 5, ".tnl", fileName );
   if( ! this->analyt.save( fileName ) )
=======
   FileName fileName;
   fileName.setFileNameBase( "u-" );
   fileName.setExtension( "tnl" );
   fileName.setIndex( step );
   if( ! dofs->save( fileName.getFileName() ) )
>>>>>>> develop
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
<<<<<<< HEAD
   step++;
   typedef typename MeshType::Cell Cell;
   double count = mesh.template getEntitiesCount< Cell >();
   double inverseSquareCount = sqrt(count);
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
			expValue = exp(-pow(10/(size*count)*(size*i-0.2*size*count)-step * 10 * tau * this->speedX,2));
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
			expValue = exp(-pow(10/(size*inverseSquareCount)*(size*i-0.2*size*inverseSquareCount)-step * 10 * tau * this->speedX,2)-pow(10/(size*inverseSquareCount)*(size*j-0.2*size*inverseSquareCount)-step * 10 * tau * this->speedY,2));
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
			this->analyt[i] = exp(-pow(10/(size*count)*(size*i-0.2*size*count)-step * 10 * tau * this->speedX,2));
		   };
		   this->analyt[count-1] = 0;
		}
	    else if (dimension == 2)
	       {
                   count = sqrt(count);
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
		      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			   this->analyt[i * inverseSquareCount + j] = exp(-pow(10/(size*inverseSquareCount)*(size*i-0.2*size*inverseSquareCount)-step * 10 * tau * this->speedX,2)-pow(10/(size*inverseSquareCount)*(size*j-0.2*size*inverseSquareCount)-step * 10 * tau * this->speedY,2));
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
		   for (IndexType i = 0; i < inverseSquareCount-1; i++)
		      for (IndexType j = 0; j < inverseSquareCount-1; j++)
		      {
			   if (i - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedX < 0.5*inverseSquareCount && 
                               j - step * tau * (inverseSquareCount/this->schemeSize) * this -> speedY < 0.5*inverseSquareCount) 
                              this->analyt[i * inverseSquareCount + j] = 1; 
                           else
                              this->analyt[i * inverseSquareCount + j] = 0;
		      };
		};
     };
     };
/*
   cout << step << endl;
   cout << tau << endl;
   cout << this->speedX << endl;
   cout << step * 10 * tau * this->speedX<< endl;*/
=======
   /****
    * If you use an explicit solver like Euler or Merson, you
    * need to implement this method. Compute the right-hand side of
    *
    *   d/dt u(x) = fu( x, u )
    *
    * You may use supporting mesh dependent data if you need.
    */
   typedef typename MeshType::Cell Cell;
   int count = ::sqrt(mesh->template getEntitiesCount< Cell >());
//   const RealType& size = parameters.getParameter< double >( "realSize" ) / ::pow(count, 0.5);
/*   if (this->velocityType == "rotation")
   {
      double radius;
      for (int i =1; i < count; i++)
         for (int j =1; j < count; j++)
            {
               radius = ::sqrt(pow(i-1-(count/2.0),2) + ::pow(j-1-(count/2.0),2));
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
*/  { 
>>>>>>> develop
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
