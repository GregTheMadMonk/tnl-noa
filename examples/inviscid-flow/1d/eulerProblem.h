#ifndef eulerPROBLEM_H_
#define eulerPROBLEM_H_

#include <problems/tnlPDEProblem.h>
#include <functions/tnlMeshFunction.h>

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
           typename DifferentialOperator >
class eulerProblem:
   public tnlPDEProblem< Mesh,
                         TimeDependentProblem,
                         typename DifferentialOperator::RealType,
                         typename Mesh::DeviceType,
                         typename DifferentialOperator::IndexType >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef tnlMeshFunction< Mesh > MeshFunctionType;
      typedef tnlPDEProblem< Mesh, TimeDependentProblem, RealType, DeviceType, IndexType > BaseType;

      using typename BaseType::MeshType;
      using typename BaseType::DofVectorType;
      using typename BaseType::MeshDependentDataType;
      typedef tnlMeshFunction<Mesh,Mesh::getMeshDimensions(),RealType>MeshFunction;
      
      typedef typename DifferentialOperator::Continuity Continuity;
      typedef typename DifferentialOperator::Momentum Momentum;
      typedef typename DifferentialOperator::Energy Energy;
      typedef typename DifferentialOperator::Velocity Velocity;
      typedef typename DifferentialOperator::Pressure Pressure;
      


      static tnlString getTypeStatic();

      tnlString getPrologHeader() const;

      void writeProlog( tnlLogger& logger,
                        const tnlParameterContainer& parameters ) const;

      bool setup( const tnlParameterContainer& parameters );

      bool setInitialCondition( const tnlParameterContainer& parameters,
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
      
      bool postIterate( const RealType& time,
                        const RealType& tau,
                        const MeshType& mesh,
                        DofVectorType& dofs,
                        MeshDependentDataType& meshDependentData );

   protected:

      DifferentialOperator differentialOperator;
      BoundaryCondition boundaryCondition;
      RightHandSide rightHandSide;
      
      MeshFunctionType uRho, uRhoVelocity, uEnergy;
      MeshFunctionType fuRho, fuRhoVelocity, fuEnergy;
      
      MeshFunctionType pressure, velocity, rho, rhoVel, energy;
      
      RealType gamma;


};

#include "eulerProblem_impl.h"

#endif /* eulerPROBLEM_H_ */
