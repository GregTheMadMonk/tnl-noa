/***************************************************************************
                          tnlHeatEquationProblem.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber et al.
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

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */


#ifndef TNLHEATEQUATIONPROBLEM_H_
#define TNLHEATEQUATIONPROBLEM_H_

#include <problems/tnlPDEProblem.h>
#include <operators/diffusion/tnlLinearDiffusion.h>
#include <matrices/tnlEllpackMatrix.h>
#include <functions/tnlMeshFunction.h>

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator = tnlLinearDiffusion< Mesh,
                                                              typename BoundaryCondition::RealType > >
class tnlHeatEquationProblem : public tnlPDEProblem< Mesh,
                                                     typename DifferentialOperator::RealType,
                                                     typename Mesh::DeviceType,
                                                     typename DifferentialOperator::IndexType  >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef tnlMeshFunction< Mesh > MeshFunctionType;
      typedef tnlPDEProblem< Mesh, RealType, DeviceType, IndexType > BaseType;
      typedef tnlCSRMatrix< RealType, DeviceType, IndexType > MatrixType;

      using typename BaseType::MeshType;
      using typename BaseType::DofVectorType;
      using typename BaseType::MeshDependentDataType;

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
                     const DofVectorType& dofs );

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
                                 const DofVectorType& dofs,                                 
                                 Matrix& matrix,
                                 DofVectorType& rightHandSide,
				 MeshDependentDataType& meshDependentData );


      protected:
         
         MeshFunctionType u;
      
         DifferentialOperator differentialOperator;

         BoundaryCondition boundaryCondition;

         RightHandSide rightHandSide;
};

#include <problems/tnlHeatEquationProblem_impl.h>

#endif /* TNLHEATEQUATIONPROBLEM_H_ */
