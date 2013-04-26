/***************************************************************************
                          tnlLaxFridrichs.h  -  description
                             -------------------
    begin                : Mar 1, 2013
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

#ifndef TNLLAXFRIDRICHS_H_
#define TNLLAXFRIDRICHS_H_

template< typename MeshType >
class tnlLaxFridrichs
{
   public:

   typedef typename MeshType :: RealType RealType;
   typedef typename MeshType :: DeviceType DeviceType;
   typedef typename MeshType :: IndexType IndexType;

   tnlLaxFridrichs();

   template< typename Vector >
   void getExplicitRhs( const IndexType centralVolume,
                        const Vector& rho,
                        const Vector& rho_u1,
                        const Vector& rho_u2,
                        Vector& rho_t,
                        Vector& rho_u1_t,
                        Vector& rho_u2_t ) const;

   void setRegularization( const RealType& epsilon );

   void setViscosityCoefficient( const RealType& v );

   void bindMesh( const MeshType& mesh );

   protected:

   RealType regularize( const RealType& r ) const;

   RealType regularizeEps, viscosityCoefficient;

   const MeshType* mesh;
};

#include <implementation/schemes/euler/fvm/tnlLaxFridrichs_impl.h>

#endif
