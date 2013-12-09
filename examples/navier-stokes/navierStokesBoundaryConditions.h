/***************************************************************************
                          navierStokesBoundaryConditions.h  -  description
                             -------------------
    begin                : Oct 24, 2013
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

#ifndef NAVIERSTOKESBOUNDARYCONDITIONS_H_
#define NAVIERSTOKESBOUNDARYCONDITIONS_H_

#include <config/tnlParameterContainer.h>

template< typename Mesh >
class navierStokesBoundaryConditions
{
   public:

   typedef Mesh MeshType;
   typedef typename Mesh::RealType RealType;
   typedef typename Mesh::DeviceType DeviceType;
   typedef typename Mesh::IndexType IndexType;

   navierStokesBoundaryConditions();

   bool init( const tnlParameterContainer& parameters );

   void setMesh( const MeshType& mesh );

   template< typename Vector >
   void apply( const RealType& time,
               const RealType& tau,
               Vector& rho,
               Vector& u1,
               Vector& u2,
               Vector& temperature );

   protected:

   const MeshType* mesh;

   RealType maxInflowVelocity, startUp, T, R, p0;
};

#include "navierStokesBoundaryConditions_impl.h"

#endif /* NAVIERSTOKESBOUNDARYCONDITIONS_H_ */
