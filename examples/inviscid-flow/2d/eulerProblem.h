#pragma once

#include <TNL/Problems/PDEProblem.h>
#include <TNL/Functions/MeshFunction.h>

using namespace TNL::Problems;

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
           typename DifferentialOperator >
class eulerProblem:
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
      typedef SharedPointer< DifferentialOperator > DifferentialOperatorPointer;
      typedef SharedPointer< BoundaryCondition > BoundaryConditionPointer;
      typedef SharedPointer< RightHandSide, DeviceType > RightHandSidePointer;
      typedef PDEProblem< Mesh, RealType, DeviceType, IndexType > BaseType;      
      
      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorType;
      using typename BaseType::DofVectorPointer;
      using typename BaseType::MeshDependentDataType;       

      typedef typename DifferentialOperator::Continuity Continuity;
      typedef typename DifferentialOperator::MomentumX MomentumX;
      typedef typename DifferentialOperator::MomentumY MomentumY;
      typedef typename DifferentialOperator::Energy Energy;
      typedef typename DifferentialOperator::Velocity Velocity;
      typedef typename DifferentialOperator::VelocityX VelocityX;
      typedef typename DifferentialOperator::VelocityY VelocityY;
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
      BoundaryConditionPointer boundaryConditionPointer;
      RightHandSidePointer rightHandSidePointer;
      
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

      
};

} // namespace TNL

#include "eulerProblem_impl.h"

