#ifndef HeatEquationBenchmarkPROBLEM_H_
#define HeatEquationBenchmarkPROBLEM_H_

#include <TNL/Problems/PDEProblem.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>
#include "Tuning/ExplicitUpdater.h"

using namespace TNL;
using namespace TNL::Problems;

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator,
          typename Communicator >
class HeatEquationBenchmarkProblem:
   public PDEProblem< Mesh,
                      Communicator,
                      typename DifferentialOperator::RealType,
                      typename Mesh::DeviceType,
                      typename DifferentialOperator::IndexType >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      typedef Pointers::SharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;
      typedef PDEProblem< Mesh, Communicator, RealType, DeviceType, IndexType > BaseType;
      typedef Pointers::SharedPointer< DifferentialOperator > DifferentialOperatorPointer;
      typedef Pointers::SharedPointer< BoundaryCondition > BoundaryConditionPointer;
      typedef Pointers::SharedPointer< RightHandSide, DeviceType > RightHandSidePointer;
      
      typedef Communicator CommunicatorType;

      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorPointer;

      HeatEquationBenchmarkProblem();
      
      static String getType();

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                DofVectorPointer& dofsPointer );

      template< typename Matrix >
      bool setupLinearSystem( Matrix& matrix );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         DofVectorPointer& dofsPointer );

      IndexType getDofs() const;

      void bindDofs( DofVectorPointer& dofsPointer );

      void getExplicitUpdate( const RealType& time,
                              const RealType& tau,
                              DofVectorPointer& _uPointer,
                              DofVectorPointer& _fuPointer );
      
      void applyBoundaryConditions( const RealType& time,
                                       DofVectorPointer& dofs );        

      template< typename MatrixPointer >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 DofVectorPointer& dofs,
                                 MatrixPointer& matrix,
                                 DofVectorPointer& rightHandSide );
      
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
      
      TNL::ExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > tuningExplicitUpdater;
      TNL::Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
      
};

#include "HeatEquationBenchmarkProblem_impl.h"

#endif /* HeatEquationBenchmarkPROBLEM_H_ */
