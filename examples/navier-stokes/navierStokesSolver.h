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

#include <TNL/Logger.h>
#include <TNL/core/tnlHost.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/matrices/tnlCSRMatrix.h>
#include <TNL/solvers/preconditioners/tnlDummyPreconditioner.h>
#include <TNL/solvers/tnlSolverMonitor.h>
#include <TNL/operators/euler/fvm/tnlLaxFridrichs.h>
#include <TNL/operators/gradient/tnlCentralFDMGradient.h>
#include <TNL/operators/diffusion/tnlLinearDiffusion.h>
#include <TNL/mesh/tnlLinearGridGeometry.h>
#include <TNL/solvers/cfd/navier-stokes/tnlNavierStokesSolver.h>

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

   static String getTypeStatic();

   String getPrologHeader() const;

   void writeProlog( Logger& logger,
                     const Config::ParameterContainer& parameters ) const;

   template< typename Geom >
   bool setMeshGeometry( Geom& geometry ) const;

   bool setMeshGeometry( tnlLinearGridGeometry< 2, RealType, DeviceType, IndexType >& geometry ) const;

   template< typename InitMesh >
   bool initMesh( InitMesh& mesh, const Config::ParameterContainer& parameters ) const;

   template< typename Real, typename Device, typename Index, template< int, typename, typename, typename > class Geometry >
   bool initMesh( tnlGrid< 1, Real, Device, Index, Geometry >& mesh,
                  const Config::ParameterContainer& parameters ) const;

   template< typename Real, typename Device, typename Index, template< int, typename, typename, typename > class Geometry >
   bool initMesh( tnlGrid< 2, Real, Device, Index, Geometry >& mesh,
                  const Config::ParameterContainer& parameters ) const;

   template< typename Real, typename Device, typename Index, template< int, typename, typename, typename > class Geometry >
   bool initMesh( tnlGrid< 3, Real, Device, Index, Geometry >& mesh,
                  const Config::ParameterContainer& parameters ) const;

   bool setup( const Config::ParameterContainer& parameters );

   bool setInitialCondition( const Config::ParameterContainer& parameters );

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
