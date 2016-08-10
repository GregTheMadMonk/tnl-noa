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
#include <TNL/Devices/Host.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Matrices/CSR.h>
#include <TNL/Solvers/preconditioners/Dummy.h>
#include <TNL/Solvers/SolverMonitor.h>
#include <TNL/Operators/euler/fvm/LaxFridrichs.h>
#include <TNL/Operators/gradient/tnlCentralFDMGradient.h>
#include <TNL/Operators/diffusion/LinearDiffusion.h>
#include <TNL/Meshes/tnlLinearGridGeometry.h>
#include <TNL/Solvers/cfd/navier-stokes/NavierStokesSolver.h>

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
   typedef Vector< RealType, DeviceType, IndexType> DofVectorType;

   typedef CSR< RealType, DeviceType, IndexType > DiscreteSolverMatrixType;
   typedef Dummy< RealType, DeviceType, IndexType > DiscreteSolverPreconditioner;

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
   bool initMesh( Meshes::Grid< 1, Real, Device, Index, Geometry >& mesh,
                  const Config::ParameterContainer& parameters ) const;

   template< typename Real, typename Device, typename Index, template< int, typename, typename, typename > class Geometry >
   bool initMesh( Meshes::Grid< 2, Real, Device, Index, Geometry >& mesh,
                  const Config::ParameterContainer& parameters ) const;

   template< typename Real, typename Device, typename Index, template< int, typename, typename, typename > class Geometry >
   bool initMesh( Meshes::Grid< 3, Real, Device, Index, Geometry >& mesh,
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

   SolverMonitor< RealType, IndexType >* getSolverMonitor();

   protected:

   //RealType regularize( const RealType& r ) const;

   ProblemType problem;

   MeshType mesh;

   DofVectorType dofVector, rhsDofVector;

   RealType p_0, gravity, T;

   EulerScheme eulerScheme;

   NavierStokesSolver< EulerScheme,
                          LinearDiffusion< MeshType >,
                          navierStokesBoundaryConditions< MeshType > > nsSolver;

   LinearDiffusion< MeshType > u1Viscosity, u2Viscosity, eViscosity;
   tnlCentralFDMGradient< MeshType > pressureGradient;

   navierStokesBoundaryConditions< MeshType > boundaryConditions;

   navierStokesSolverMonitor< RealType, IndexType > solverMonitor;

   IndexType rhsIndex;
};

#include "navierStokesSolver_impl.h"

#endif /* NAVIERSTOKESSOLVER_H_ */
