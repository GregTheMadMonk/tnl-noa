/***************************************************************************
                          tnlNavierStokes.h  -  description
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

#ifndef TNLNAVIERSTOKES_H_
#define TNLNAVIERSTOKES_H_

#include <core/tnlString.h>
#include <core/vectors/tnlVector.h>

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
class tnlNavierStokes
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
   typedef VectorType DofVectorType;

   tnlNavierStokes();

   static tnlString getTypeStatic();

   void setAdvectionScheme( AdvectionSchemeType& advection );

   void setDiffusionScheme( DiffusionSchemeType& diffusion );

   void setBoundaryConditions( BoundaryConditionsType& boundaryConditions );

   void setMesh( MeshType& mesh );

   void setMu( const RealType& mu );

   const RealType& getMu() const;

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

   template< typename Vector >
   void updatePhysicalQuantities( const Vector& rho,
                                  const Vector& rho_u1,
                                  const Vector& rho_u2 );

   void getExplicitRhs( const RealType& time,
                        const RealType& tau,
                        DofVectorType& u,
                        DofVectorType& fu ) const;


   protected:

   AdvectionSchemeType* advection;

   DiffusionSchemeType* diffusion;

   BoundaryConditionsType* boundaryConditions;

   MeshType* mesh;

   VectorType rho, u1, u2, p;

   RealType mu, gravity;

};

#include <implementation/schemes/navier-stokes/tnlNavierStokes_impl.h>

#endif /* TNLNAVIERSTOKES_H_ */
