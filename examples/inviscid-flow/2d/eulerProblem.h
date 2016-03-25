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

      typedef typename DifferentialOperator::Continuity Continuity;
      typedef typename DifferentialOperator::MomentumX MomentumX;
      typedef typename DifferentialOperator::MomentumY MomentumY;
      typedef typename DifferentialOperator::Energy Energy;
      typedef typename DifferentialOperator::Velocity Velocity;
      typedef typename DifferentialOperator::VelocityX VelocityX;
      typedef typename DifferentialOperator::VelocityY VelocityY;
      typedef typename DifferentialOperator::Pressure Pressure;

    //definition
	tnlVector< RealType, DeviceType, IndexType > _uRho;
	tnlVector< RealType, DeviceType, IndexType > _uRhoVelocityX;
	tnlVector< RealType, DeviceType, IndexType > _uRhoVelocityY;
	tnlVector< RealType, DeviceType, IndexType > _uEnergy;

	tnlVector< RealType, DeviceType, IndexType > _fuRho;
	tnlVector< RealType, DeviceType, IndexType > _fuRhoVelocityX;
	tnlVector< RealType, DeviceType, IndexType > _fuRhoVelocityY;
	tnlVector< RealType, DeviceType, IndexType > _fuEnergy;

      tnlVector< RealType, DeviceType, IndexType > rho;
      tnlVector< RealType, DeviceType, IndexType > rhoVelX;
      tnlVector< RealType, DeviceType, IndexType > rhoVelY;
      tnlVector< RealType, DeviceType, IndexType > energy;
      tnlVector< RealType, DeviceType, IndexType > data;
      tnlVector< RealType, DeviceType, IndexType > pressure;
      tnlVector< RealType, DeviceType, IndexType > velocity;
      tnlVector< RealType, DeviceType, IndexType > velocityX;
      tnlVector< RealType, DeviceType, IndexType > velocityY;
      double gamma;

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
};

#include "eulerProblem_impl.h"

#endif /* eulerPROBLEM_H_ */
