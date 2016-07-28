#pragma once

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
	   Vectors::Vector< RealType, DeviceType, IndexType > _uRho;
	   Vectors::Vector< RealType, DeviceType, IndexType > _uRhoVelocityX;
	   Vectors::Vector< RealType, DeviceType, IndexType > _uRhoVelocityY;
	   Vectors::Vector< RealType, DeviceType, IndexType > _uEnergy;

	   Vectors::Vector< RealType, DeviceType, IndexType > _fuRho;
	   Vectors::Vector< RealType, DeviceType, IndexType > _fuRhoVelocityX;
	   Vectors::Vector< RealType, DeviceType, IndexType > _fuRhoVelocityY;
	   Vectors::Vector< RealType, DeviceType, IndexType > _fuEnergy;

      Vectors::Vector< RealType, DeviceType, IndexType > rho;
      Vectors::Vector< RealType, DeviceType, IndexType > rhoVelX;
      Vectors::Vector< RealType, DeviceType, IndexType > rhoVelY;
      Vectors::Vector< RealType, DeviceType, IndexType > energy;
      Vectors::Vector< RealType, DeviceType, IndexType > data;
      Vectors::Vector< RealType, DeviceType, IndexType > pressure;
      Vectors::Vector< RealType, DeviceType, IndexType > velocity;
      Vectors::Vector< RealType, DeviceType, IndexType > velocityX;
      Vectors::Vector< RealType, DeviceType, IndexType > velocityY;
      double gamma;

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

} // namespace TNL

#include "eulerProblem_impl.h"

