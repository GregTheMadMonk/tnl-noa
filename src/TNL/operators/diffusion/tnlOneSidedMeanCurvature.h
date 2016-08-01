/***************************************************************************
                          tnlOneSidedMeanCurvature.h  -  description
                             -------------------
    begin                : Feb 17, 2016
    copyright            : (C) 2016 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/operators/tnlOperator.h>
#include <TNL/operators/tnlFunctionInverseOperator.h>
#include <TNL/operators/geometric/tnlFDMGradientNorm.h>
#include <TNL/operators/tnlNeumannBoundaryConditions.h>
#include <TNL/operators/diffusion/tnlOneSidedNonlinearDiffusion.h>
#include <TNL/Functions/OperatorFunction.h>
#include <TNL/Functions/Analytic/ConstantFunction.h>
#include <TNL/operators/diffusion/tnlExactMeanCurvature.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType,
          bool EvaluateNonlinearityOnFly = false >
class tnlOneSidedMeanCurvature
   : public tnlOperator< Mesh, Functions::MeshInteriorDomain, Mesh::getMeshDimensions(), Mesh::getMeshDimensions(), Real, Index >
{
   public:
 
      typedef Mesh MeshType;
      typedef tnlSharedPointer< MeshType > MeshPointer;
      typedef Real RealType;
      typedef Index IndexType;
      typedef tnlFDMGradientNorm< MeshType, tnlForwardFiniteDifference, RealType, IndexType > GradientNorm;
      typedef tnlFunctionInverseOperator< GradientNorm > NonlinearityOperator;
      typedef Functions::MeshFunction< MeshType, MeshType::getMeshDimensions(), RealType > NonlinearityMeshFunction;
      typedef Functions::tnlConstantFunction< MeshType::getMeshDimensions(), RealType > NonlinearityBoundaryConditionsFunction;
      typedef tnlNeumannBoundaryConditions< MeshType, NonlinearityBoundaryConditionsFunction > NonlinearityBoundaryConditions;
      typedef Functions::OperatorFunction< NonlinearityOperator, NonlinearityMeshFunction, NonlinearityBoundaryConditions, EvaluateNonlinearityOnFly > Nonlinearity;
      typedef tnlOneSidedNonlinearDiffusion< Mesh, Nonlinearity, RealType, IndexType > NonlinearDiffusion;
      typedef tnlExactMeanCurvature< Mesh::getMeshDimensions(), RealType > ExactOperatorType;
      
      tnlOneSidedMeanCurvature( const MeshPointer& meshPointer )
      : nonlinearityOperator( gradientNorm ),
        nonlinearity( nonlinearityOperator, nonlinearityBoundaryConditions, meshPointer ),
        nonlinearDiffusion( nonlinearity ){}
 
      static String getType()
      {
         return String( "tnlOneSidedMeanCurvature< " ) +
            MeshType::getType() + ", " +
           TNL::getType< Real >() + ", " +
           TNL::getType< Index >() + " >";
      }
 
      void setRegularizationEpsilon( const RealType& eps )
      {
         this->gradientNorm.setEps( eps );
      }
 
      void setPreimageFunction( typename Nonlinearity::PreimageFunctionType& preimageFunction )
      {
         this->nonlinearity.setPreimageFunction( preimageFunction );
      }
 
      bool refresh( const RealType& time = 0.0 )
      {
         return this->nonlinearity.refresh( time );
      }
 
      bool deepRefresh( const RealType& time = 0.0 )
      {
         return this->nonlinearity.deepRefresh( time );
      }
 
      template< typename MeshFunction,
                typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         return this->nonlinearDiffusion( u, entity, time );
      }

      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const
      {
         return this->nonlinearDiffusion.getLinearSystemRowLength( mesh, index, entity );
      }

      template< typename MeshEntity,
                typename MeshFunction,
                typename Vector,
                typename Matrix >
      __cuda_callable__
      void setMatrixElements( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunction& u,
                               Vector& b,
                               Matrix& matrix ) const
      {
         this->nonlinearDiffusion.setMatrixElements( time, tau, mesh, index, entity, u, b, matrix );
      }
 
   protected:
 
      NonlinearityBoundaryConditions nonlinearityBoundaryConditions;
 
      GradientNorm gradientNorm;

      NonlinearityOperator nonlinearityOperator;
 
      Nonlinearity nonlinearity;
 
      NonlinearDiffusion nonlinearDiffusion;
};

} // namespace TNL
