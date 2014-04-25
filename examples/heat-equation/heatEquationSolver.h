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
#include "heatEquationSolver.h"
#include "tnlAnalyticSolution.h"


template< typename Mesh, typename Diffusion, typename BoundaryCondition, typename RightHandSide, typename TimeFunction, typename AnalyticSpaceFunction>
class heatEquationSolver
{
   public:

   typedef typename Mesh :: RealType RealType;
   typedef typename Mesh :: DeviceType DeviceType;
   typedef typename Mesh :: IndexType IndexType;
   typedef Mesh MeshType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlCSRMatrix< RealType, DeviceType, IndexType > DiscreteSolverMatrixType;
   typedef tnlDummyPreconditioner< RealType, DeviceType, IndexType > DiscreteSolverPreconditioner;

   static tnlString getTypeStatic();

   tnlString getPrologHeader() const;

   void writeProlog( tnlLogger& logger,
                     const tnlParameterContainer& parameters ) const;

   bool init( const tnlParameterContainer& parameters );

   bool setInitialCondition( const tnlParameterContainer& parameters );

   bool makeSnapshot( const RealType& time, const IndexType& step );

   DofVectorType& getDofVector();

   void GetExplicitRHS( const RealType& time,
                        const RealType& tau,
                        DofVectorType& _u,
                        DofVectorType& _fu );

   tnlSolverMonitor< RealType, IndexType >* getSolverMonitor();
   
   protected:

   DofVectorType dofVector,dofVector2,analyticLaplace,numericalLaplace;
   tnlSharedVector< RealType, DeviceType, IndexType > u,v;
   MeshType mesh;
   AnalyticSpaceFunction analyticSpaceFunction;
   TimeFunction timeFunction;
   AnalyticSolution<MeshType> analyticSolution;
   BoundaryCondition boundaryCondition;
   Diffusion diffusion;
   RightHandSide RHS;
   IndexType ifLaplaceDiff;
};

#include "heatEquationSolver_impl.h"

#endif /* HEATEQUATIONSOLVER_H_ */
