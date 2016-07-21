/***************************************************************************
                          tnlHeatEquationProblem.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/operators/diffusion/tnlOneSidedMeanCurvature.h>
#include <TNL/problems/tnlPDEProblem.h>
#include <TNL/operators/operator-Q/tnlOneSideDiffOperatorQ.h>
#include <TNL/matrices/tnlCSRMatrix.h>
#include <TNL/functions/tnlMeshFunction.h>

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator =
            tnlOneSidedMeanCurvature< Mesh,
                                      typename Mesh::RealType,
                                      typename Mesh::IndexType,
                                      false > >
class tnlMeanCurvatureFlowProblem : public tnlPDEProblem< Mesh,
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
      typedef tnlCSRMatrix< RealType, DeviceType, IndexType> MatrixType;

      using typename BaseType::MeshType;
      using typename BaseType::DofVectorType;
      using typename BaseType::MeshDependentDataType;

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


      protected:

      tnlSharedVector< RealType, DeviceType, IndexType > solution;

      DifferentialOperator differentialOperator;

      BoundaryCondition boundaryCondition;
 
      RightHandSide rightHandSide;
};


} // namespace TNL

#include "tnlMeanCurvatureFlowProblem_impl.h"
