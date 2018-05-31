#ifndef HeatEquationBenchmarkPROBLEM_H_
#define HeatEquationBenchmarkPROBLEM_H_

#include <TNL/Problems/PDEProblem.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>

using namespace TNL;
using namespace TNL::Problems;

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename CommType,
          typename DifferentialOperator >
class HeatEquationBenchmarkProblem:
   public PDEProblem< Mesh,
                         typename DifferentialOperator::RealType,
                         typename Mesh::DeviceType,
                         typename DifferentialOperator::IndexType >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      typedef SharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;
      typedef PDEProblem< Mesh, RealType, DeviceType, IndexType > BaseType;
      typedef SharedPointer< DifferentialOperator > DifferentialOperatorPointer;
      typedef SharedPointer< BoundaryCondition > BoundaryConditionPointer;
      typedef SharedPointer< RightHandSide, DeviceType > RightHandSidePointer;
      
      typedef CommType CommunicatorType;

      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorPointer;
      using typename BaseType::MeshDependentDataType;
      using typename BaseType::MeshDependentDataPointer;

      HeatEquationBenchmarkProblem();
      
      static String getTypeStatic();

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;

      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                const MeshPointer& meshPointer,
                                DofVectorPointer& dofsPointer,
                                MeshDependentDataPointer& meshDependentData );

      template< typename Matrix >
      bool setupLinearSystem( const MeshType& mesh,
                              Matrix& matrix );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         const MeshPointer& meshPointer,
                         DofVectorPointer& dofsPointer,
                         MeshDependentDataPointer& meshDependentData );

      IndexType getDofs( const MeshPointer& meshPointer ) const;

      void bindDofs( const MeshPointer& meshPointer,
                     DofVectorPointer& dofsPointer );

      void getExplicitUpdate( const RealType& time,
                           const RealType& tau,
                           const MeshPointer& meshPointer,
                           DofVectorPointer& _uPointer,
                           DofVectorPointer& _fuPointer,
                           MeshDependentDataPointer& meshDependentData );

      template< typename MatrixPointer >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 const MeshPointer& mesh,
                                 DofVectorPointer& dofs,
                                 MatrixPointer& matrix,
                                 DofVectorPointer& rightHandSide,
                                 MeshDependentDataPointer& meshDependentData );
      
      ~HeatEquationBenchmarkProblem();

   protected:

      DifferentialOperatorPointer differentialOperatorPointer;
      BoundaryConditionPointer boundaryConditionPointer;
      RightHandSidePointer rightHandSidePointer;
      
      MeshFunctionPointer fu, u;
      
      String cudaKernelType;
      
      MeshType* cudaMesh;
      BoundaryCondition* cudaBoundaryConditions;
      RightHandSide* cudaRightHandSide;
      DifferentialOperator* cudaDifferentialOperator;
      
      Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
      
};

#include "HeatEquationBenchmarkProblem_impl.h"

#endif /* HeatEquationBenchmarkPROBLEM_H_ */
