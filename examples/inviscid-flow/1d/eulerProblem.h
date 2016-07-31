#ifndef eulerPROBLEM_H_
#define eulerPROBLEM_H_

#include <TNL/problems/tnlPDEProblem.h>
#include <TNL/Functions/tnlMeshFunction.h>

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
           typename DifferentialOperator >
class eulerProblem:
   public tnlPDEProblem< Mesh,
                         typename DifferentialOperator::RealType,
                         typename Mesh::DeviceType,
                         typename DifferentialOperator::IndexType >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef Functions::tnlMeshFunction< Mesh > MeshFunctionType;
      typedef tnlSharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;
      typedef tnlSharedPointer< DifferentialOperator > DifferentialOperatorPointer;
      typedef tnlSharedPointer< BoundaryCondition > BoundaryConditionPointer;
      typedef tnlSharedPointer< RightHandSide, DeviceType > RightHandSidePointer;
      typedef tnlPDEProblem< Mesh, RealType, DeviceType, IndexType > BaseType;      
      
      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorType;
      using typename BaseType::DofVectorPointer;
      using typename BaseType::MeshDependentDataType;        

      typedef typename DifferentialOperator::Continuity Continuity;
      typedef typename DifferentialOperator::Momentum Momentum;
      typedef typename DifferentialOperator::Energy Energy;
      typedef typename DifferentialOperator::Velocity Velocity;
      typedef typename DifferentialOperator::Pressure Pressure;
      


      static String getTypeStatic();

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;

      bool setup( const Config::ParameterContainer& parameters );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                const MeshPointer& mesh,
                                DofVectorPointer& dofs,
                                MeshDependentDataType& meshDependentData );

      template< typename Matrix >
      bool setupLinearSystem( const MeshPointer& mesh,
                              Matrix& matrix );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         const MeshPointer& mesh,
                         DofVectorPointer& dofs,
                         MeshDependentDataType& meshDependentData );

      IndexType getDofs( const MeshPointer& mesh ) const;

      void bindDofs( const MeshPointer& mesh,
                     DofVectorPointer& dofs );

      void getExplicitRHS( const RealType& time,
                           const RealType& tau,
                           const MeshPointer& mesh,
                           DofVectorPointer& _u,
                           DofVectorPointer& _fu,
                           MeshDependentDataType& meshDependentData );

      template< typename Matrix >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 const MeshPointer& mesh,
                                 DofVectorPointer& dofs,
                                 Matrix& matrix,
                                 DofVectorPointer& rightHandSide,
                                 MeshDependentDataType& meshDependentData );
      
      bool postIterate( const RealType& time,
                        const RealType& tau,
                        const MeshPointer& mesh,
                        DofVectorPointer& dofs,
                        MeshDependentDataType& meshDependentData );

   protected:

      DifferentialOperatorPointer differentialOperatorPointer;
      BoundaryConditionPointer boundaryConditionsPointer;
      RightHandSidePointer rightHandSidePointer;
      
      MeshFunctionPointer uRho, uRhoVelocity, uEnergy;
      MeshFunctionPointer fuRho, fuRhoVelocity, fuEnergy;
      
      MeshFunctionPointer pressure, velocity, rho, rhoVel, energy;
      
      RealType gamma;


};

} // namepsace TNL

#include "eulerProblem_impl.h"

#endif /* eulerPROBLEM_H_ */
