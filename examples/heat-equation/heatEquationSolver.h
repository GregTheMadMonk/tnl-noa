/***************************************************************************
                          simpleProblemSolver.h  -  description
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

#ifndef HEATEQUATIONSOLVER_H_
#define HEATEQUATIONSOLVER_H_

#include <matrices/tnlCSRMatrix.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlSolverMonitor.h>
#include <core/tnlLogger.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <solvers/pde/tnlExplicitUpdater.h>
#include "heatEquationSolver.h"
#include "tnlAnalyticSolution.h"


template< typename Mesh,
          typename Diffusion,
          typename BoundaryCondition,
          typename RightHandSide >
class heatEquationSolver
{
   public:

   typedef typename Diffusion::RealType RealType;
   typedef typename Mesh::DeviceType DeviceType;
   typedef typename Diffusion::IndexType IndexType;
   typedef Mesh MeshType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;

   static tnlString getTypeStatic();

   tnlString getPrologHeader() const;

   void writeProlog( tnlLogger& logger,
                     const tnlParameterContainer& parameters ) const;

   bool init( const tnlParameterContainer& parameters );

   bool setInitialCondition( const tnlParameterContainer& parameters,
                             const MeshType& mesh );

   bool makeSnapshot( const RealType& time,
                      const IndexType& step,
                      const MeshType& mesh );

   IndexType getDofs( const MeshType& mesh ) const;

   IndexType getAuxiliaryDofs( const MeshType& mesh ) const;

   void bindDofs( const MeshType& mesh,
                  DofVectorType& dofs );

   void bindAuxiliaryDofs( const MeshType& mesh,
                           DofVectorType& auxiliaryDofs );

   void GetExplicitRHS( const RealType& time,
                        const RealType& tau,
                        const MeshType& mesh,
                        DofVectorType& _u,
                        DofVectorType& _fu );

   tnlSolverMonitor< RealType, IndexType >* getSolverMonitor();
   
   protected:

   tnlSharedVector< RealType, DeviceType, IndexType > numericalSolution,
                                                      exactSolution,
                                                      analyticLaplace,
                                                      numericalLaplace;

   tnlExplicitUpdater< Mesh, DofVectorType, BoundaryCondition, Diffusion > explicitUpdater;

   //AnalyticSpaceFunction analyticSpaceFunction;
   //TimeFunction timeFunction;
   //AnalyticSolution< MeshType, RealType, IndexType > analyticSolution;

   BoundaryCondition boundaryCondition;

   Diffusion diffusion;

   RightHandSide RHS;
   //IndexType ifLaplaceCompare, ifSolutionCompare;
};

#include "heatEquationSolver_impl.h"

#endif /* HEATEQUATIONSOLVER_H_ */
