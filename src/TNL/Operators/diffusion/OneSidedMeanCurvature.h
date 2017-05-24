/***************************************************************************
                          OneSidedMeanCurvature.h  -  description
                             -------------------
    begin                : Feb 17, 2016
    copyright            : (C) 2016 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */

#pragma once

#include <TNL/Operators/Operator.h>
#include <TNL/Operators/FunctionInverseOperator.h>
#include <TNL/Operators/geometric/FDMGradientNorm.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include <TNL/Operators/diffusion/OneSidedNonlinearDiffusion.h>
#include <TNL/Functions/OperatorFunction.h>
#include <TNL/Functions/Analytic/Constant.h>
#include <TNL/Operators/diffusion/ExactMeanCurvature.h>

namespace TNL {
namespace Operators {   

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType,
          bool EvaluateNonlinearityOnFly = false >
class OneSidedMeanCurvature
   : public Operator< Mesh, Functions::MeshInteriorDomain, Mesh::getMeshDimension(), Mesh::getMeshDimension(), Real, Index >
{
   public:
 
      typedef Mesh MeshType;
      typedef SharedPointer< MeshType > MeshPointer;
      typedef Real RealType;
      typedef Index IndexType;
      typedef FDMGradientNorm< MeshType, ForwardFiniteDifference, RealType, IndexType > GradientNorm;
      typedef FunctionInverseOperator< GradientNorm > NonlinearityOperator;
      typedef Functions::MeshFunction< MeshType, MeshType::getMeshDimension(), RealType > NonlinearityMeshFunction;
      typedef Functions::Analytic::Constant< MeshType::getMeshDimension(), RealType > NonlinearityBoundaryConditionsFunction;
      typedef NeumannBoundaryConditions< MeshType, NonlinearityBoundaryConditionsFunction > NonlinearityBoundaryConditions;
      typedef Functions::OperatorFunction< NonlinearityOperator, NonlinearityMeshFunction, NonlinearityBoundaryConditions, EvaluateNonlinearityOnFly > Nonlinearity;
      typedef OneSidedNonlinearDiffusion< Mesh, Nonlinearity, RealType, IndexType > NonlinearDiffusion;
      typedef ExactMeanCurvature< Mesh::getMeshDimension(), RealType > ExactOperatorType;
      
      OneSidedMeanCurvature( const MeshPointer& meshPointer )
      : nonlinearityOperator( gradientNorm ),
        nonlinearity( nonlinearityOperator, nonlinearityBoundaryConditions, meshPointer ),
        nonlinearDiffusion( nonlinearity ){}
 
      static String getType()
      {
         return String( "OneSidedMeanCurvature< " ) +
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

} // namespace Operators
} // namespace TNL
