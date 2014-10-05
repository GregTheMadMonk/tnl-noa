/***************************************************************************
                          navierStokesSolver.h  -  description
                             -------------------
    begin                : Jan 13, 2013
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

#ifndef NAVIERSTOKESSOLVER_H_
#define NAVIERSTOKESSOLVER_H_

#include <core/tnlLogger.h>
#include <core/tnlHost.h>
#include <core/vectors/tnlVector.h>
#include <config/tnlParameterContainer.h>
#include <matrices/tnlCSRMatrix.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlSolverMonitor.h>
#include <operators/euler/fvm/tnlLaxFridrichs.h>
#include <operators/gradient/tnlCentralFDMGradient.h>
#include <operators/diffusion/tnlLinearDiffusion.h>
#include <mesh/tnlLinearGridGeometry.h>
#include <solvers/cfd/navier-stokes/tnlNavierStokesSolver.h>

#include "navierStokesSolverMonitor.h"
#include "navierStokesBoundaryConditions.h"

template< typename Mesh,
          typename EulerScheme >
class navierStokesSolver
{
   public:

   typedef typename Mesh :: RealType RealType;
   typedef typename Mesh :: DeviceType DeviceType;
   typedef typename Mesh :: IndexType IndexType;
   typedef Mesh MeshType;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;

   typedef tnlCSRMatrix< RealType, DeviceType, IndexType > DiscreteSolverMatrixType;
   typedef tnlDummyPreconditioner< RealType, DeviceType, IndexType > DiscreteSolverPreconditioner;

   enum BoundaryConditionType { dirichlet, neumann, noSlip };

   enum ProblemType { riser, cavity };

   navierStokesSolver();

   static tnlString getTypeStatic();

   tnlString getPrologHeader() const;

   void writeProlog( tnlLogger& logger,
                     const tnlParameterContainer& parameters ) const;

   template< typename Geom >
   bool setMeshGeometry( Geom& geometry ) const;

   bool setMeshGeometry( tnlLinearGridGeometry< 2, RealType, DeviceType, IndexType >& geometry ) const;

   template< typename InitMesh >
   bool initMesh( InitMesh& mesh, const tnlParameterContainer& parameters ) const;

   template< typename Real, typename Device, typename Index, template< int, typename, typename, typename > class Geometry >
   bool initMesh( tnlGrid< 1, Real, Device, Index, Geometry >& mesh,
                  const tnlParameterContainer& parameters ) const;

   template< typename Real, typename Device, typename Index, template< int, typename, typename, typename > class Geometry >
   bool initMesh( tnlGrid< 2, Real, Device, Index, Geometry >& mesh,
                  const tnlParameterContainer& parameters ) const;

   template< typename Real, typename Device, typename Index, template< int, typename, typename, typename > class Geometry >
   bool initMesh( tnlGrid< 3, Real, Device, Index, Geometry >& mesh,
                  const tnlParameterContainer& parameters ) const;

   bool setup( const tnlParameterContainer& parameters );

   bool setInitialCondition( const tnlParameterContainer& parameters );

   DofVectorType& getDofVector();

   bool makeSnapshot( const RealType& t,
                      const IndexType step );

   bool solve();

   void GetExplicitRHS( const RealType& time,
                        const RealType& tau,
                        DofVectorType& _u,
                        DofVectorType& _fu );

   tnlSolverMonitor< RealType, IndexType >* getSolverMonitor();

   protected:

   //RealType regularize( const RealType& r ) const;

   ProblemType problem;

   MeshType mesh;

   DofVectorType dofVector, rhsDofVector;

   RealType p_0, gravity, T;

   EulerScheme eulerScheme;

   tnlNavierStokesSolver< EulerScheme,
                          tnlLinearDiffusion< MeshType >,
                          navierStokesBoundaryConditions< MeshType > > nsSolver;

   tnlLinearDiffusion< MeshType > u1Viscosity, u2Viscosity, eViscosity;
   tnlCentralFDMGradient< MeshType > pressureGradient;

   navierStokesBoundaryConditions< MeshType > boundaryConditions;

   navierStokesSolverMonitor< RealType, IndexType > solverMonitor;

   IndexType rhsIndex;
};

#include "navierStokesSolver_impl.h"

#endif /* NAVIERSTOKESSOLVER_H_ */
