/***************************************************************************
                          tnlNavierStokesSolverSolver.h  -  description
                             -------------------
    begin                : Oct 22, 2013
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

#ifndef tnlNavierStokesSolver_H_
#define tnlNavierStokesSolver_H_

#include <core/tnlString.h>
#include <core/vectors/tnlVector.h>

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
class tnlNavierStokesSolver
{
   public:

   typedef AdvectionScheme AdvectionSchemeType;
   typedef DiffusionScheme DiffusionSchemeType;
   typedef BoundaryConditions BoundaryConditionsType;
   typedef typename AdvectionScheme::MeshType MeshType;
   typedef typename AdvectionScheme::RealType RealType;
   typedef typename AdvectionScheme::DeviceType DeviceType;
   typedef typename AdvectionScheme::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType > VectorType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > DofVectorType;

   tnlNavierStokesSolver();

   static tnlString getTypeStatic();

   void setAdvectionScheme( AdvectionSchemeType& advection );

   void setDiffusionScheme( DiffusionSchemeType& u1Viscosity,
                            DiffusionSchemeType& u2Viscosity );

   void setBoundaryConditions( BoundaryConditionsType& boundaryConditions );

   void setMesh( MeshType& mesh );

   void setMu( const RealType& mu );

   const RealType& getMu() const;

   void setR( const RealType& R );

   const RealType& getR() const;

   void setT( const RealType& T );

   const RealType& getT() const;

   void setGravity( const RealType& gravity );

   const RealType& getGravity() const;

   VectorType& getRho();

   const VectorType& getRho() const;

   VectorType& getU1();

   const VectorType& getU1() const;

   VectorType& getU2();

   const VectorType& getU2() const;

   VectorType& getPressure();

   const VectorType& getPressure() const;

   VectorType& getEnergy();

   const VectorType& getEnergy() const;


   IndexType getDofs() const;

   void bindDofVector( RealType* );

   DofVectorType& getDofVector();

   template< typename Vector >
   void updatePhysicalQuantities( const Vector& rho,
                                  const Vector& rho_u1,
                                  const Vector& rho_u2,
                                  const Vector& e );

   template< typename SolverVectorType >
   void getExplicitRhs( const RealType& time,
                        const RealType& tau,
                        SolverVectorType& u,
                        SolverVectorType& fu );

   bool writePhysicalVariables( const RealType& t,
                                const IndexType step );

   bool writeConservativeVariables( const RealType& t,
                                   const IndexType step );


   protected:

   AdvectionSchemeType* advection;

   DiffusionSchemeType  *u1Viscosity, *u2Viscosity;

   BoundaryConditionsType* boundaryConditions;

   MeshType* mesh;

   VectorType rho, u1, u2, p, e;

   RealType mu, gravity, R, T;

   DofVectorType dofVector;

};

#include <implementation/solvers/cfd/navier-stokes/tnlNavierStokesSolver_impl.h>

#endif /* tnlNavierStokesSolver_H_ */
