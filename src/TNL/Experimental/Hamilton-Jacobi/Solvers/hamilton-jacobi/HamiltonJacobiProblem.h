/***************************************************************************
                          hamiltonJacobiProblem.h  -  description
                             -------------------
    begin                : Jul 8 , 2014
    copyright            : (C) 2014 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#pragma once

#include <problems/tnlPDEProblem.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlSolverMonitor.h>
#include <core/tnlLogger.h>
#include <core/vectors/tnlVector.h>
#include <solvers/pde/tnlExplicitUpdater.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>
#include <functions/tnlMeshFunction.h>

template< typename Mesh,
		    typename DifferentialOperator,
		    typename BoundaryCondition,
		    typename RightHandSide>
class HamiltonJacobiProblem : public tnlPDEProblem< Mesh,
                                                    TimeDependentProblem,
                                                    typename DifferentialOperator::RealType,
                                                    typename Mesh::DeviceType,
                                                    typename DifferentialOperator::IndexType  >
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

      static String getTypeStatic();

      String getPrologHeader() const;

      void writeProlog( tnlLogger& logger,
                        const Config::ParameterContainer& parameters ) const;

      bool setup( const Config::ParameterContainer& parameters );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                const MeshType& mesh,
                                DofVectorType& dofs,
                                MeshDependentDataType& meshDependentData );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         const MeshType& mesh,
                         DofVectorType& dofs,
                         MeshDependentDataType& meshDependentData );

      IndexType getDofs( const MeshType& mesh ) const;

      void bindDofs( const MeshType& mesh,
                     DofVectorType& dofs );

      void getExplicitUpdate( const RealType& time,
                           const RealType& tau,
                           const MeshType& mesh,
                           DofVectorType& _u,
                           DofVectorType& _fu,
                           MeshDependentDataType& meshDependentData );

   protected:

      MeshFunctionType solution;

      tnlExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide  > explicitUpdater;

      DifferentialOperator differentialOperator;

      BoundaryCondition boundaryCondition;

      RightHandSide rightHandSide;
};

#include "HamiltonJacobiProblem_impl.h"

