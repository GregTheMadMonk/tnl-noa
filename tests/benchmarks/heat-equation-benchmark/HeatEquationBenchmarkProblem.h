#ifndef HeatEquationBenchmarkPROBLEM_H_
#define HeatEquationBenchmarkPROBLEM_H_

#include <TNL/problems/tnlPDEProblem.h>
#include <TNL/Functions/tnlMeshFunction.h>

using namespace TNL;

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
           typename DifferentialOperator >
class HeatEquationBenchmarkProblem:
   public tnlPDEProblem< Mesh,
                         typename DifferentialOperator::RealType,
                         typename Mesh::DeviceType,
                         typename DifferentialOperator::IndexType >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef tnlMeshFunction< Mesh > MeshFunctionType;
      typedef tnlPDEProblem< Mesh, RealType, DeviceType, IndexType > BaseType;

      using typename BaseType::MeshType;
      using typename BaseType::DofVectorType;
      using typename BaseType::MeshDependentDataType;

      static String getTypeStatic();

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;

      bool setup( const Config::ParameterContainer& parameters );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                const MeshType& mesh,
                                DofVectorType& dofs,
                                MeshDependentDataType& meshDependentData );

      template< typename Matrix >
      bool setupLinearSystem( const MeshType& mesh,
                              Matrix& matrix );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         const MeshType& mesh,
                         DofVectorType& dofs,
                         MeshDependentDataType& meshDependentData );

      IndexType getDofs( const MeshType& mesh ) const;

      void bindDofs( const MeshType& mesh,
                     DofVectorType& dofs );

      void getExplicitRHS( const RealType& time,
                           const RealType& tau,
                           const MeshType& mesh,
                           DofVectorType& _u,
                           DofVectorType& _fu,
                           MeshDependentDataType& meshDependentData );

      template< typename Matrix >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 const MeshType& mesh,
                                 DofVectorType& dofs,
                                 Matrix& matrix,
                                 DofVectorType& rightHandSide,
                                 MeshDependentDataType& meshDependentData );

   protected:

      DifferentialOperator differentialOperator;
      BoundaryCondition boundaryCondition;
      RightHandSide rightHandSide;
};

#include "HeatEquationBenchmarkProblem_impl.h"

#endif /* HeatEquationBenchmarkPROBLEM_H_ */
