#ifndef advectionPROBLEM_IMPL_H_
#define advectionPROBLEM_IMPL_H_

#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>
#include <TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <TNL/Solvers/PDE/BackwardTimeDiscretisation.h>

#include "advectionProblem.h"

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
   if( ! this->velocityField->setup( meshPointer, parameters, prefix + "velocity-field-" ) ||
       ! this->differentialOperatorPointer->setup( meshPointer, parameters, prefix ) ||
       ! this->boundaryConditionPointer->setup( meshPointer, parameters, prefix + "boundary-conditions-" ) )
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
bindDofs( const MeshPointer& meshPointer,
          DofVectorPointer& dofVector )
{
   const IndexType dofs = meshPointer->template getEntitiesCount< typename MeshType::Cell >();
   this->uPointer->bind( meshPointer, dofVector );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
advectionProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     const MeshPointer& meshPointer,
                     DofVectorPointer& dofs,
                     MeshDependentDataPointer& meshDependentData )
{
   this->bindDofs( meshPointer, dofs );
   const String& initialConditionFile = parameters.getParameter< String >( "initial-condition" );
   if( ! this->uPointer->boundLoad( initialConditionFile ) )
   {
      std::cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << std::endl;
      return false;
   }
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
   /*const IndexType dofs = this->getDofs( mesh );
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
      return false;*/
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
   FileName fileName;
   fileName.setFileNameBase( "u-" );
   fileName.setExtension( "tnl" );
   fileName.setIndex( step );
   if( ! dofs->save( fileName.getFileName() ) )
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
   this->bindDofs( mesh, _u );
   Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   SharedPointer< MeshFunctionType > u( mesh, _u ); 
   SharedPointer< MeshFunctionType > fu( mesh, _fu );
   differentialOperatorPointer->setTau(tau); 
   differentialOperatorPointer->setVelocityField( this->velocityField );
   explicitUpdater.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           this->differentialOperatorPointer,
                                                           this->boundaryConditionPointer,
                                                           this->rightHandSidePointer,
                                                           u,
                                                           fu );
   /*BoundaryConditionsSetter< MeshFunctionType, BoundaryCondition > boundaryConditionsSetter; 
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
}

} // namespace TNL

#endif /* advectionPROBLEM_IMPL_H_ */
