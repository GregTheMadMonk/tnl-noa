/***************************************************************************
                          eulerProblem.h  -  description
                             -------------------
    begin                : Feb 13, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Problems/PDEProblem.h>
#include <TNL/Functions/MeshFunction.h>
#include "CompressibleConservativeVariables.h"


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
      typedef PDEProblem< Mesh, RealType, DeviceType, IndexType > BaseType;
      
      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorType;
      using typename BaseType::DofVectorPointer;
      using typename BaseType::MeshDependentDataType;
      using typename BaseType::MeshDependentDataPointer;

      static const int Dimensions = MeshType::getMeshDimensions();      

      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      typedef CompressibleConservativeVariables< MeshType > ConservativeVariablesType;
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VelocityFieldType;
      typedef SharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;
      typedef SharedPointer< ConservativeVariablesType > ConservativeVariablesPointer;
      typedef SharedPointer< VelocityFieldType > VelocityFieldPointer;
      typedef SharedPointer< DifferentialOperator > DifferentialOperatorPointer;
      typedef SharedPointer< BoundaryCondition > BoundaryConditionPointer;
      typedef SharedPointer< RightHandSide, DeviceType > RightHandSidePointer;
      
      

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

      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                const MeshPointer& mesh,
                                DofVectorPointer& dofs,
                                MeshDependentDataPointer& meshDependentData );

      template< typename Matrix >
      bool setupLinearSystem( const MeshPointer& mesh,
                              Matrix& matrix );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         const MeshPointer& mesh,
                         DofVectorPointer& dofs,
                         MeshDependentDataPointer& meshDependentData );

      IndexType getDofs( const MeshPointer& mesh ) const;

      void bindDofs( const MeshPointer& mesh,
                     DofVectorPointer& dofs );

      void getExplicitRHS( const RealType& time,
                           const RealType& tau,
                           const MeshPointer& mesh,
                           DofVectorPointer& _u,
                           DofVectorPointer& _fu,
                           MeshDependentDataPointer& meshDependentData );

      template< typename Matrix >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 const MeshPointer& mesh,
                                 DofVectorPointer& dofs,
                                 Matrix& matrix,
                                 DofVectorPointer& rightHandSide,
                                 MeshDependentDataPointer& meshDependentData );

      bool postIterate( const RealType& time,
                        const RealType& tau,
                        const MeshPointer& mesh,
                        DofVectorPointer& dofs,
                        MeshDependentDataPointer& meshDependentData );

   protected:

      DifferentialOperatorPointer differentialOperatorPointer;
      BoundaryConditionPointer boundaryConditionPointer;
      RightHandSidePointer rightHandSidePointer;
      
      ConservativeVariablesPointer conservativeVariables;
      
      VelocityFieldPointer velocity;
      MeshFunctionPointer pressure, energy;
      
      
      //definition
	   Containers::Vector< RealType, DeviceType, IndexType > _uRho;
	   Containers::Vector< RealType, DeviceType, IndexType > _uRhoVelocityX;
	   Containers::Vector< RealType, DeviceType, IndexType > _uRhoVelocityY;
	   Containers::Vector< RealType, DeviceType, IndexType > _uEnergy;

	   Containers::Vector< RealType, DeviceType, IndexType > _fuRho;
	   Containers::Vector< RealType, DeviceType, IndexType > _fuRhoVelocityX;
	   Containers::Vector< RealType, DeviceType, IndexType > _fuRhoVelocityY;
	   Containers::Vector< RealType, DeviceType, IndexType > _fuEnergy;

      Containers::Vector< RealType, DeviceType, IndexType > rho;
      Containers::Vector< RealType, DeviceType, IndexType > rhoVelX;
      Containers::Vector< RealType, DeviceType, IndexType > rhoVelY;
      Containers::Vector< RealType, DeviceType, IndexType > energy;
      Containers::Vector< RealType, DeviceType, IndexType > data;
      Containers::Vector< RealType, DeviceType, IndexType > pressure;
      Containers::Vector< RealType, DeviceType, IndexType > velocity;
      Containers::Vector< RealType, DeviceType, IndexType > velocityX;
      Containers::Vector< RealType, DeviceType, IndexType > velocityY;
      double gamma;
};

} // namespace TNL

#include "eulerProblem_impl.h"

