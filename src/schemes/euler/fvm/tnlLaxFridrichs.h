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

   template< typename ConservativeVector,
             typename VelocityVector >
   void getExplicitRhs( const MeshType& mesh,
                        const IndexType centralVolume,
                        const ConservativeVector& rho,
                        const ConservativeVector& rho_u1,
                        const ConservativeVector& rho_u2,
                        ConservativeVector& rho_t,
                        ConservativeVector& rho_u1_t,
                        ConservativeVector& rho_u2_t,
                        const typename MeshType :: RealType viscosityCoeff = 1.0 ) const;

   void setRegularization( const RealType& epsilon );

   protected:

   RealType regularize( const RealType& r );

   RealType regularizeEps;
};

#include <implementation/schemes/euler/fvm/tnlLaxFridrichs_impl.h>

#endif
