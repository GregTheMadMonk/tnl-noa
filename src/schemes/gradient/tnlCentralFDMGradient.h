/***************************************************************************
                          tnlCentralFDMGradient.h  -  description
                             -------------------
    begin                : Apr 26, 2013
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

#ifndef TNLCENTRALFDMGRADIENT_H_
#define TNLCENTRALFDMGRADIENT_H_

#include <mesh/tnlGrid.h>
#include <core/tnlHost.h>

template< typename Mesh >
class tnlCentralFDMGradient
{
};

template< typename Real, typename Device, typename Index >
class tnlCentralFDMGradient< tnlGrid< 2, Real, Device, Index > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   tnlCentralFDMGradient();

   void bindMesh( const tnlGrid< 2, RealType, DeviceType, IndexType >& mesh );

   template< typename Vector >
   void setFunction( Vector& f ); // TODO: add const

   void getGradient( const Index& i,
                     RealType& f_x,
                     RealType& f_y ) const;
   protected:

   // TODO: change to ConstSharedVector
   tnlSharedVector< RealType, DeviceType, IndexType > f;

   const tnlGrid< 2, RealType, DeviceType, IndexType >* mesh;
};

#include <implementation/schemes/gradient/tnlCentralFDMGradient_impl.h>

#endif
