/***************************************************************************
                          tnlHeatEquationProblem.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLMEANCURVATUREFLOWPROBLEM_H_
#define TNLMEANCURVATUREFLOWPROBLEM_H_

#include <operators/diffusion/tnlNonlinearDiffusion.h>
#include <operators/diffusion/nonlinear-diffusion-operators/tnlOneSideDiffNonlinearOperator.h>
#include <problems/tnlPDEProblem.h>
#include <operators/operator-Q/tnlOneSideDiffOperatorQForGraph.h>
#include <matrices/tnlCSRMatrix.h>

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator = tnlNonlinearDiffusion< Mesh,
                                                          tnlOneSideDiffNonlinearOperator< Mesh, tnlOneSideDiffOperatorQForGraph<Mesh, typename Mesh::RealType,
                                                          typename Mesh::IndexType, 0>, typename Mesh::RealType, typename Mesh::IndexType >, 
                                                          typename Mesh::RealType, typename Mesh::IndexType > >
class tnlMeanCurvatureFlowProblem : public tnlPDEProblem< Mesh,
                                                     typename DifferentialOperator::RealType,
                                                     typename Mesh::DeviceType,
                                                     typename DifferentialOperator::IndexType >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef tnlPDEProblem< Mesh, RealType, DeviceType, IndexType > BaseType;
      typedef tnlCSRMatrix< RealType, DeviceType, IndexType> MatrixType;

      using typename BaseType::MeshType;
      using typename BaseType::DofVectorType;

      static tnlString getTypeStatic();

      tnlString getPrologHeader() const;

      void writeProlog( tnlLogger& logger,
                        const tnlParameterContainer& parameters ) const;

      bool setup( const tnlParameterContainer& parameters );

      bool setInitialCondition( const tnlParameterContainer& parameters,
                                const MeshType& mesh,
                                DofVectorType& dofs,
                                DofVectorType& auxDofs );

      template< typename Matrix >
      bool setupLinearSystem( const MeshType& mesh,
                              Matrix& matrix );
      
      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         const MeshType& mesh,
                         DofVectorType& dofs,
                         DofVectorType& auxDofs );

      IndexType getDofs( const MeshType& mesh ) const;

      void bindDofs( const MeshType& mesh,
                     DofVectorType& dofs );

      void getExplicitRHS( const RealType& time,
                           const RealType& tau,
                           const MeshType& mesh,
                           DofVectorType& _u,
                           DofVectorType& _fu );

      template< typename Matrix >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 const MeshType& mesh,
                                 DofVectorType& dofs,
                                 DofVectorType& auxDofs,
                                 Matrix& matrix,
                                 DofVectorType& rightHandSide );


      protected:

      tnlSharedVector< RealType, DeviceType, IndexType > solution;

      DifferentialOperator differentialOperator;

      BoundaryCondition boundaryCondition;
   
      RightHandSide rightHandSide;
};

#include "tnlMeanCurvatureFlowProblem_impl.h"

#endif /* TNLMEANCURVATUREFLOWPROBLEM_H_ */
